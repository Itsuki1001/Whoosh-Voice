"""
server.py — FastAPI WebSocket server.
Wires together STTSession, TTSSentenceStreamer, and the LLM graph.
"""

import asyncio
import json
import os
import queue as _q
import re
import sys
import threading
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from stt import STTSession
from tts import TTSSentenceStreamer, init_tts, split_sentences

load_dotenv()

SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
CARTESIA_API_KEY  = os.getenv("CARTESIA_API_KEY")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from graph.graph_voice import graph

# ── Config ────────────────────────────────────────────────────────────────────
MIN_SENTENCE_CHARS = 0
LLM_DEBOUNCE       = 0

# ── Tool filler responses ─────────────────────────────────────────────────────
TOOL_FILLERS = {
    "get_room_availability": "Sure, let me check room availability for you...",
    "get_distance_to_homestay":        "Let me look up the distance for you...",
    "rag_tool":    "Let me pull up that information for you...",
}
DEFAULT_FILLER = "Give me just a moment..."

# ── Initialise TTS client once ────────────────────────────────────────────────
init_tts(CARTESIA_API_KEY)

app = FastAPI()

with open(os.path.join(os.path.dirname(__file__), "voice_ui.html"), encoding="utf-8") as f:
    HTML = f.read()

@app.get("/")
async def index():
    return HTMLResponse(HTML)


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[•\*#`]', '', text)
    return re.sub(r'\n', ' ', text).strip()

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


# ── LLM sentence streamer (runs in executor thread) ───────────────────────────
def stream_graph_sentences(
    transcript: str,
    thread_id: str,
    sentence_q: _q.Queue,
    cancel_flag: threading.Event,
):
    config = {"configurable": {"thread_id": thread_id}}
    buffer = ""
    filler_sent = False

    try:
        for chunk, _ in graph.stream(
            {"messages": transcript}, config, stream_mode="messages"
        ):
            if cancel_flag.is_set():
                break
            if hasattr(chunk, "content"):
                if type(chunk).__name__ in ("AIMessageChunk", "AIMessage"):

                    # ── Filler ────────────────────────────────────────────
                    if not filler_sent and getattr(chunk, "tool_calls", None):
                        tool_name = chunk.tool_calls[0].get("name", "")
                        if tool_name:
                            filler = TOOL_FILLERS.get(tool_name, DEFAULT_FILLER)
                            sentence_q.put(filler)
                            filler_sent = True
                        continue

                    # ── Normal streaming ──────────────────────────────────
                    if chunk.content:
                        buffer += chunk.content
                        parts = _SENT_RE.split(buffer)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                s = clean(sentence)
                                if len(s) >= MIN_SENTENCE_CHARS:
                                    sentence_q.put(s)
                            buffer = parts[-1]

        # flush remainder
        if buffer.strip() and not cancel_flag.is_set():
            r = clean(buffer.strip())
            if r:
                sentence_q.put(r)
    finally:
        sentence_q.put(None)


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(browser_ws: WebSocket):
    await browser_ws.accept()

    thread_id    = f"web-{id(browser_ws)}"
    bot_speaking = asyncio.Event()
    cancel_event = asyncio.Event()
    audio_queue  = asyncio.Queue()
    current_task = [None]

    # ── helpers ───────────────────────────────────────────────────────────────

    async def send_json(data: dict):
        try:
            await browser_ws.send_text(json.dumps(data))
        except Exception:
            pass

    async def send_bytes_ws(data: bytes):
        try:
            await browser_ws.send_bytes(data)
        except Exception:
            pass

    # ── LLM + TTS pipeline ────────────────────────────────────────────────────

    async def process_transcript(transcript: str):
        sid  = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        cancel_event.clear()

        await send_json({"type": "transcript", "text": transcript})
        await send_json({"type": "thinking"})

        llm_cancel = threading.Event()
        sentence_q = _q.Queue()

        # ── CHANGE: signal audio_start FIRST so TTS WS opens in parallel ──────
        await send_json({"type": "audio_start", "session_id": sid})
        bot_speaking.set()

        # now kick off LLM — TTS WS is already warming up concurrently
        llm_future = loop.run_in_executor(
            None,
            stream_graph_sentences,
            transcript, thread_id, sentence_q, llm_cancel,
        )

        tts_streamer = TTSSentenceStreamer(on_audio_chunk=send_bytes_ws)

        try:
            _, full_parts = await asyncio.gather(
                llm_future,
                tts_streamer.stream(sentence_q, cancel_event),
            )
        except asyncio.CancelledError:
            llm_cancel.set()
            raise
        finally:
            llm_cancel.set()
            bot_speaking.clear()

        if not cancel_event.is_set():
            full_response = " ".join(full_parts)
            await send_json({"type": "response", "text": full_response})

        await send_json({"type": "audio_end", "session_id": sid})

    async def handle_transcript(transcript: str):
        if current_task[0] and not current_task[0].done():
            cancel_event.set()
            bot_speaking.clear()
            current_task[0].cancel()
            try:
                await current_task[0]
            except asyncio.CancelledError:
                pass
            await send_json({"type": "barge_in"})
        cancel_event.clear()
        current_task[0] = asyncio.create_task(process_transcript(transcript))

    # ── STT session ───────────────────────────────────────────────────────────

    stt = STTSession(
        api_key=SARVAM_API_KEY,
        on_transcript=handle_transcript,
        on_interim=lambda text: send_json({"type": "interim", "text": text}),
        on_barge_in=lambda: send_json({"type": "barge_in"}),
    )

    async def stt_loop():
        await stt.run(audio_queue, bot_speaking, cancel_event, current_task)

    # ── Browser message receiver ──────────────────────────────────────────────

    async def receive_from_browser():
        while True:
            msg = await browser_ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if msg.get("bytes"):
                await audio_queue.put(msg["bytes"])

    # ── Run everything ────────────────────────────────────────────────────────

    try:
        await asyncio.gather(receive_from_browser(), stt_loop())
    except WebSocketDisconnect:
        pass
    finally:
        if current_task[0] and not current_task[0].done():
            current_task[0].cancel()
        await audio_queue.put(None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)