"""
websocket.py — WebSocket endpoint, rate limiting, STT/TTS/LLM wiring.
"""

import asyncio
import json
import os
import queue as _q
import re
import threading
import uuid
import random
from collections import defaultdict
import time
import logging
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from voice.stt import STTSession
from voice.tts import TTSSentenceStreamer, init_tts
from graph.graph_voice import graph

load_dotenv()

SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")


router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
MIN_SENTENCE_CHARS = 0

# ── Pre-rendered limit audio ──────────────────────────────────────────────────
with open("assets//limit_message.pcm", "rb") as f:
    LIMIT_AUDIO = f.read()

# ── HTML ──────────────────────────────────────────────────────────────────────
with open("assets//voice_ui.html", encoding="utf-8") as f:
    HTML = f.read()

# ── Rate limiting stores ──────────────────────────────────────────────────────
ip_connections     = defaultdict(int)
session_requests   = defaultdict(int)
ip_hourly_requests = defaultdict(lambda: {"count": 0, "reset_at": time.time() + 3600})
ip_daily_requests  = defaultdict(lambda: {"count": 0, "reset_at": time.time() + 86400})

MAX_CONCURRENT_PER_IP    = 3
MAX_REQUESTS_PER_SESSION = 50
MAX_HOURLY_REQUESTS      = 20
MAX_DAILY_REQUESTS       = 100

# ── Rate limit helper ─────────────────────────────────────────────────────────
def is_ip_limit_reached(ip: str) -> bool:
    now = time.time()

    hourly = ip_hourly_requests[ip]
    if now > hourly["reset_at"]:
        hourly["count"] = 0
        hourly["reset_at"] = now + 3600
    if hourly["count"] >= MAX_HOURLY_REQUESTS:
        logging.warning(f"IP {ip} hit hourly limit")
        return True

    daily = ip_daily_requests[ip]
    if now > daily["reset_at"]:
        daily["count"] = 0
        daily["reset_at"] = now + 86400
    if daily["count"] >= MAX_DAILY_REQUESTS:
        logging.warning(f"IP {ip} hit daily limit")
        return True

    hourly["count"] += 1
    daily["count"] += 1
    return False

# ── Tool filler responses ─────────────────────────────────────────────────────
TOOL_FILLERS = {
    "get_room_availability": [
        "Sure, let me check room availability for you...",
        "Let me look that up for you...",
        "Give me a moment to check the rooms...",
        "Checking availability right now...",
        "Let me see what we have available...",
    ],
    "get_distance_to_homestay": [
        "Let me look up the distance for you...",
        "Let me calculate that for you...",
        "Give me a second to check the directions...",
        "Let me find that out for you...",
        "Checking the distance right now...",
    ],
    "rag_tool": [
        "Let me pull up that information for you...",
        "Give me a moment to look into that...",
        "Let me find the details for you...",
        "I'll look that up right away...",
        "Let me check our information on that...",
    ],
}

DEFAULT_FILLERS = [
    "Give me just a moment...",
    "Let me look into that...",
    "One moment please...",
    "Let me check that for you...",
    "Give me a second...",
]

# ── Initialise TTS ────────────────────────────────────────────────────────────
init_tts(CARTESIA_API_KEY)

# ── Text helpers ──────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[•\*#`]', '', text)
    return re.sub(r'\n', ' ', text).strip()

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

# ── Graph state repair ────────────────────────────────────────────────────────
def fix_broken_graph_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph.get_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            return
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tool_call in last.tool_calls:
                graph.update_state(config, {
                    "messages": [ToolMessage(
                        content="Request was interrupted by user.",
                        tool_call_id=tool_call["id"]
                    )]
                })
            logging.info(f"[GRAPH] Repaired state for thread {thread_id} after interruption.")
    except Exception as e:
        logging.error(f"[GRAPH] state fix failed: {e}")

# ── LLM sentence streamer ─────────────────────────────────────────────────────
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
                    if getattr(chunk, "tool_calls", None):
                        if not filler_sent:
                            tool_name = chunk.tool_calls[0].get("name", "")
                            fillers = TOOL_FILLERS.get(tool_name, DEFAULT_FILLERS)
                            sentence_q.put(random.choice(fillers))
                            filler_sent = True
                        continue_processing = False
                    else:
                        continue_processing = True

                    if continue_processing and chunk.content:
                        buffer += chunk.content
                        parts = _SENT_RE.split(buffer)
                        if len(parts) > 1:
                            for sentence in parts[:-1]:
                                s = clean(sentence)
                                if len(s) >= MIN_SENTENCE_CHARS:
                                    sentence_q.put(s)
                            buffer = parts[-1]

        if buffer.strip() and not cancel_flag.is_set():
            r = clean(buffer.strip())
            if r:
                sentence_q.put(r)
    finally:
        sentence_q.put(None)

# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/")
async def index():
    return HTMLResponse(HTML)

@router.websocket("/ws")
async def websocket_endpoint(browser_ws: WebSocket):
    await browser_ws.accept()

    client_ip = browser_ws.client.host

    if ip_connections[client_ip] >= MAX_CONCURRENT_PER_IP:
        await browser_ws.close(code=1008)
        return

    ip_connections[client_ip] += 1

    thread_id    = f"web-{id(browser_ws)}"
    bot_speaking = asyncio.Event()
    cancel_event = asyncio.Event()
    audio_queue  = asyncio.Queue(maxsize=50)
    current_task = [None]

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

    async def cancel_current():
        cancel_event.set()
        bot_speaking.clear()
        task = current_task[0]
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.get_running_loop().run_in_executor(
            None, fix_broken_graph_state, thread_id
        )

    async def handle_barge_in():
        await cancel_current()
        await send_json({"type": "barge_in"})
        logging.info("[BARGE-IN] User interrupted the bot.")

    async def _speak_limit(message_audio: bytes):
        sid = str(uuid.uuid4())
        bot_speaking.set()
        await send_json({"type": "audio_start", "session_id": sid})
        chunk_size = 4800
        for i in range(0, len(message_audio), chunk_size):
            await send_bytes_ws(message_audio[i:i + chunk_size])
        await send_json({"type": "audio_end", "session_id": sid})
        bot_speaking.clear()

    async def handle_transcript(transcript: str):
        if is_ip_limit_reached(client_ip):
            await _speak_limit(LIMIT_AUDIO)
            return

        session_requests[thread_id] += 1
        if session_requests[thread_id] > MAX_REQUESTS_PER_SESSION:
            await _speak_limit(LIMIT_AUDIO)
            await asyncio.sleep(3)
            await browser_ws.close(code=1008)
            return

        if current_task[0] and not current_task[0].done():
            await cancel_current()
            await send_json({"type": "barge_in"})
        cancel_event.clear()
        current_task[0] = asyncio.create_task(process_transcript(transcript))

    async def process_transcript(transcript: str):
        sid  = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        cancel_event.clear()

        await send_json({"type": "transcript", "text": transcript})
        await send_json({"type": "thinking"})

        llm_cancel = threading.Event()
        sentence_q = _q.Queue()

        await send_json({"type": "audio_start", "session_id": sid})
        bot_speaking.set()

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

    async def on_interim(text):
        await send_json({"type": "interim", "text": text})

    stt = STTSession(
        api_key=SARVAM_API_KEY,
        on_transcript=handle_transcript,
        on_interim=on_interim,
        on_barge_in=handle_barge_in,
    )

    async def stt_loop():
        await stt.run(audio_queue, bot_speaking)

    async def receive_from_browser():
        while True:
            msg = await browser_ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if msg.get("bytes"):
                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await audio_queue.put(msg["bytes"])

    try:
        await asyncio.gather(receive_from_browser(), stt_loop())
    except WebSocketDisconnect:
        pass
    finally:
        if current_task[0] and not current_task[0].done():
            current_task[0].cancel()
        await audio_queue.put(None)
        ip_connections[client_ip] -= 1
        session_requests.pop(thread_id, None)