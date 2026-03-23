import asyncio
import sys
import os
import re
import json
import uuid
import time
import threading
import queue as _q
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from graph.graph_voice import graph

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_INPUT_WORDS    = 1
BARGE_IN_RMS       = 0.08
SAMPLE_RATE        = 16000
CHUNK_DURATION     = 0.1
SILENCE_HOLD       = 0.0
HESITATION_WORDS   = {"um", "uh", "hmm"}
TTS_BATCH_CHARS    = 120
MIN_SENTENCE_CHARS = 8

VOICE_ID  = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = "eleven_multilingual_v2"
# ─────────────────────────────────────────────

eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)
app    = FastAPI()

with open(os.path.join(os.path.dirname(__file__), "voice_ui.html"), encoding="utf-8") as f:
    HTML = f.read()

@app.get("/")
async def index():
    return HTMLResponse(HTML)


# ── Timing helper ─────────────────────────────────────────────────────────────

def ms(t: float) -> str:
    """Format seconds as milliseconds string."""
    return f"{(t * 1000):.1f}ms"

def ts() -> float:
    return time.perf_counter()


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[•\*#`]', '', text)
    return re.sub(r'\n', ' ', text).strip()

def only_hesitation(text: str) -> bool:
    words = text.lower().split()
    return bool(words) and all(w in HESITATION_WORDS for w in words)

def is_acceptable(text: str) -> bool:
    words = text.strip().split()
    return len(words) >= MIN_INPUT_WORDS

# ── Audio helpers ─────────────────────────────────────────────────────────────

def check_rms(audio_bytes: bytes) -> float:
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

# ── Sentence splitter ─────────────────────────────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

# ── LLM sentence streamer (runs in executor thread) ───────────────────────────

def stream_graph_sentences(transcript: str, thread_id: str,
                            sentence_q: _q.Queue, cancel_flag: threading.Event,
                            t_llm_input: float):
    """
    Streams LLM tokens, splits on sentence boundaries, pushes complete
    sentences onto sentence_q. Pushes None as sentinel when done.
    t_llm_input is passed in so we can measure time-to-first-token.
    """
    config = {"configurable": {"thread_id": thread_id}}
    buffer = ""
    first_token = True
    try:
        for chunk, _ in graph.stream(
            {"messages": transcript}, config, stream_mode="messages"
        ):
            if cancel_flag.is_set():
                break
            if hasattr(chunk, "content") and chunk.content:
                if type(chunk).__name__ in ("AIMessageChunk", "AIMessage"):
                    if first_token:
                        t_first_token = ts()
                        print(f"  ⚡ [LLM] first token received        +{ms(t_first_token - t_llm_input)} after LLM input")
                        first_token = False
                    buffer += chunk.content
                    parts = _SENT_RE.split(buffer)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            s = clean(sentence)
                            if len(s) >= MIN_SENTENCE_CHARS:
                                sentence_q.put((s, ts()))   # tuple: (text, time_enqueued)
                        buffer = parts[-1]
        # flush remainder
        if buffer.strip() and not cancel_flag.is_set():
            r = clean(buffer.strip())
            if r:
                sentence_q.put((r, ts()))
    finally:
        sentence_q.put(None)   # sentinel


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(browser_ws: WebSocket):
    await browser_ws.accept()

    thread_id    = f"web-{id(browser_ws)}"
    bot_speaking = asyncio.Event()
    cancel_event = asyncio.Event()
    audio_queue  = asyncio.Queue()
    current_task = [None]
    loop         = asyncio.get_event_loop()

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

    # ── STT loop ─────────────────────────────────────────────────────────────

    async def stt_loop():
        utt = {
            "confirmed_chunks":  [],
            "current_segment":   "",
            "in_speech":         False,
            "segment_finalised": False,
        }
        finalize_task = [None]

        # Per-turn timing state (shared via list so closures can mutate)
        t_speech_start = [None]   # when SpeechStarted fired
        t_final        = [None]   # when is_final arrived

        def full_text() -> str:
            parts = utt["confirmed_chunks"][:]
            if utt["current_segment"]:
                parts.append(utt["current_segment"])
            return " ".join(p.strip() for p in parts if p.strip())

        def reset_utterance():
            utt["confirmed_chunks"]  = []
            utt["current_segment"]   = ""
            utt["in_speech"]         = False
            utt["segment_finalised"] = False

        async def finalize(t_final_received: float):
            try:
                await asyncio.sleep(SILENCE_HOLD)
            except asyncio.CancelledError:
                return

            if utt["current_segment"]:
                utt["confirmed_chunks"].append(utt["current_segment"])
                utt["current_segment"] = ""

            text = full_text()

            if not text:
                reset_utterance(); return

            if only_hesitation(text):
                print(f"  [TURN] SKIP — hesitation only: '{text}'")
                reset_utterance(); return

            if not is_acceptable(text):
                print(f"  [TURN] SKIP — too short: '{text}'")
                reset_utterance(); return

            t_llm_input = ts()
            print(f"\n{'─'*60}")
            if t_speech_start[0]:
                print(f"  🎙  [STT ] speech started                (t=0 reference)")
                print(f"  📝  [STT ] final transcript received     +{ms(t_final_received - t_speech_start[0])} after speech start")
                print(f"  🧠  [LLM ] input sent to LLM             +{ms(t_llm_input - t_speech_start[0])} after speech start  |  +{ms(t_llm_input - t_final_received)} after final")
            else:
                print(f"  📝  [STT ] final transcript received     t={t_final_received:.3f}")
                print(f"  🧠  [LLM ] input sent to LLM             +{ms(t_llm_input - t_final_received)} after final")
            print(f"  📨  [TURN] ACCEPTED: '{text}'")

            reset_utterance()
            await handle_transcript(text, t_llm_input, t_speech_start[0])

        def schedule_finalize():
            t_now = ts()
            if finalize_task[0] and not finalize_task[0].done():
                finalize_task[0].cancel()
            finalize_task[0] = asyncio.run_coroutine_threadsafe(
                finalize(t_now), loop
            )

        def cancel_finalize():
            if finalize_task[0] and not finalize_task[0].done():
                finalize_task[0].cancel()

        client     = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        stop_event = threading.Event()

        with client.listen.v1.connect(
            model            = "nova-2",
            language         = "en-IN",
            encoding         = "linear16",
            channels         = "1",
            sample_rate      = str(SAMPLE_RATE),
            punctuate        = "true",
            smart_format     = "true",
            interim_results  = "true",
            vad_events       = "true",
            endpointing      = "150",
        ) as connection:

            def on_open(_):
                print("[DEEPGRAM] Connected — streaming PCM directly")

            def on_message(message):
                msg_type = getattr(message, "type", None)

                if msg_type == "Results":
                    try:
                        sentence = message.channel.alternatives[0].transcript
                    except (AttributeError, IndexError):
                        return
                    if not sentence:
                        return

                    if message.is_final:
                        t_final[0] = ts()
                        utt["segment_finalised"] = True
                        utt["confirmed_chunks"].append(sentence)
                        utt["current_segment"] = ""
                        acc = full_text()
                        print(f"  📝  [STT ] is_final=True: '{acc}'")
                        asyncio.run_coroutine_threadsafe(
                            send_json({"type": "interim", "text": acc}), loop
                        )
                        schedule_finalize()

                    else:
                        if utt["segment_finalised"]:
                            return
                        utt["current_segment"] = sentence
                        acc = full_text()
                        print(f"  〰   [STT ] interim: '{acc}'")
                        asyncio.run_coroutine_threadsafe(
                            send_json({"type": "interim", "text": acc}), loop
                        )

                elif msg_type == "SpeechStarted":
                    t_speech_start[0]        = ts()
                    utt["in_speech"]         = True
                    utt["segment_finalised"] = False
                    print(f"\n  🎙  [VAD ] SpeechStarted")
                    cancel_finalize()

                elif msg_type == "UtteranceEnd":
                    utt["in_speech"] = False
                    print(f"  🔇  [VAD ] UtteranceEnd (fallback={'YES' if not utt['segment_finalised'] else 'no — final already handled'})")
                    if not utt["segment_finalised"]:
                        if utt["current_segment"]:
                            utt["confirmed_chunks"].append(utt["current_segment"])
                            utt["current_segment"] = ""
                        schedule_finalize()

            def on_error(error):
                print(f"[DEEPGRAM ERROR] {error}")

            def on_close(_):
                print("[DEEPGRAM] Connection closed")

            connection.on(EventType.OPEN,    on_open)
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.ERROR,   on_error)
            connection.on(EventType.CLOSE,   on_close)

            listener_thread = threading.Thread(
                target=connection.start_listening, daemon=True
            )
            listener_thread.start()

            async def send_audio_to_deepgram():
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    rms = check_rms(chunk)
                    if rms > BARGE_IN_RMS:
                        cancel_event.set()
                        bot_speaking.clear()
                        if current_task[0] and not current_task[0].done():
                            current_task[0].cancel()
                        await send_json({"type": "barge_in"})
                    if bot_speaking.is_set():
                        continue
                    connection.send_media(chunk)

            try:
                await send_audio_to_deepgram()
            finally:
                stop_event.set()
                try:
                    connection.send_close_stream()
                except Exception:
                    pass
                listener_thread.join(timeout=2)

    # ── LLM + TTS pipeline ────────────────────────────────────────────────────

    async def process_transcript(transcript: str, t_llm_input: float, t_speech_start: float | None):
        sid = str(uuid.uuid4())
        cancel_event.clear()

        await send_json({"type": "transcript", "text": transcript})
        await send_json({"type": "thinking"})

        llm_cancel  = threading.Event()
        sentence_q  = _q.Queue()
        full_parts  = []
        first_tts   = [True]   # flag: have we fired TTS yet this turn

        llm_future = loop.run_in_executor(
            None, stream_graph_sentences,
            transcript, thread_id, sentence_q, llm_cancel, t_llm_input
        )

        await send_json({"type": "audio_start", "session_id": sid})
        bot_speaking.set()

        async def stream_tts_sentences():
            batch     = []
            batch_len = 0

            async def flush_batch():
                nonlocal batch, batch_len
                if not batch:
                    return
                # batch is list of (text, t_enqueued) tuples
                texts       = [item[0] for item in batch]
                t_enqueued  = batch[0][1]   # time when first sentence was enqueued
                text        = " ".join(texts)
                batch       = []
                batch_len   = 0

                t_tts_input = ts()
                print(f"  🔊  [TTS ] input sent to ElevenLabs      +{ms(t_tts_input - t_llm_input)} after LLM input  |  +{ms(t_tts_input - t_enqueued)} after sentence ready")
                print(f"       text ({len(text)} chars): '{text[:80]}{'…' if len(text)>80 else ''}'")

                full_parts.append(text)
                try:
                    t_tts_start = [None]
                    first_chunk = [True]

                    audio_stream = await loop.run_in_executor(
                        None,
                        lambda t=text: eleven.text_to_speech.stream(
                            text=t,
                            voice_id=VOICE_ID,
                            model_id=TTS_MODEL,
                            output_format="pcm_24000",
                        )
                    )
                    for chunk in audio_stream:
                        if cancel_event.is_set():
                            return
                        if chunk:
                            if first_chunk[0]:
                                t_tts_start[0] = ts()
                                first_chunk[0] = False

                                print(f"  🔈  [TTS ] first audio chunk received   +{ms(t_tts_start[0] - t_tts_input)} after TTS input")

                                if first_tts[0]:
                                    first_tts[0] = False
                                    if t_speech_start:
                                        e2e = t_tts_start[0] - t_speech_start
                                        print(f"\n  {'★'*50}")
                                        print(f"  ★  END-TO-END LATENCY (speech end → first audio)")
                                        print(f"  ★  {ms(e2e)}  total")
                                        print(f"  ★  breakdown:")
                                        print(f"  ★    STT final latency    : +{ms(t_llm_input - t_speech_start)} (speech start → LLM input)")
                                        print(f"  ★    LLM time-to-sentence : +{ms(t_tts_input - t_llm_input)} (LLM input → TTS input)")
                                        print(f"  ★    TTS time-to-audio    : +{ms(t_tts_start[0] - t_tts_input)} (TTS input → first chunk)")
                                        print(f"  {'★'*50}\n")
                                    else:
                                        print(f"\n  ★  First audio chunk out  (no speech-start reference)")

                            await send_bytes_ws(chunk)
                            await asyncio.sleep(0)
                except Exception as e:
                    print(f"[TTS ERROR] {e}")

            while True:
                try:
                    item = sentence_q.get_nowait()
                except _q.Empty:
                    await asyncio.sleep(0.005)
                    continue

                if cancel_event.is_set():
                    while True:
                        try:
                            if sentence_q.get_nowait() is None:
                                break
                        except _q.Empty:
                            await asyncio.sleep(0.005)
                    break

                if item is None:
                    await flush_batch()
                    break

                sentence, t_enqueued = item
                batch.append((sentence, t_enqueued))
                batch_len += len(sentence)

                if batch_len >= TTS_BATCH_CHARS:
                    await flush_batch()

        try:
            await asyncio.gather(llm_future, stream_tts_sentences())
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

    async def handle_transcript(transcript: str, t_llm_input: float, t_speech_start: float | None):
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
        current_task[0] = asyncio.create_task(
            process_transcript(transcript, t_llm_input, t_speech_start)
        )

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