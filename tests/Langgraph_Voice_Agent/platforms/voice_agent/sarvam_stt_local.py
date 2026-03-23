"""
Local Voice Assistant — sentence-streaming with pyttsx3
========================================================
• Mic     — sounddevice (callback)
• STT     — Sarvam AI (saaras:v3)
• LLM     — graph.graph_voice (sentence-streaming)
• TTS     — pyttsx3 (local SAPI, no network)

Install:
    pip install sounddevice numpy sarvamai pyttsx3 python-dotenv
"""

import asyncio
import base64
import os
import re
import struct
import sys
import threading
import time
import queue as _q
from datetime import datetime

import numpy as np
import sounddevice as sd
import pyttsx3
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from graph.graph_voice import graph

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY", "")
SAMPLE_RATE        = 16000
CHUNK_DURATION     = 0.5
LANGUAGE           = "en-IN"
HIGH_VAD_SENS      = True
SILENCE_HOLD       = 0.6
MIN_INPUT_WORDS    = 2
LLM_DEBOUNCE       = 0.1
BARGE_IN_RMS       = 0.08
HESITATION_WORDS   = {"um", "uh", "hmm"}
MIC_DEVICE         = 2
TTS_RATE           = 165          # words per minute
MIN_SENTENCE_CHARS = 20           # don't synthesise tiny fragments alone

# ─── Colours ──────────────────────────────────────────────────────────────────
RESET="\033[0m"; CYAN="\033[96m"; GREEN="\033[92m"
YELLOW="\033[93m"; RED="\033[91m"; MAG="\033[95m"

def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(tag, msg, c=""):
    print(f"\n{c}[{ts()}] [{tag:<6}] {msg}{RESET if c else ''}", flush=True)

def ms(t0, t1):
    return (t1 - t0) * 1000

def bbar(seconds, scale=4):
    return "▓" * max(1, int(seconds * scale))

def print_timeline(T):
    T0 = T.get("start_speech")
    T1 = T.get("first_interim")
    T2 = T.get("end_speech")
    T3 = T.get("silence_expired")
    T4 = T.get("llm_start")
    T5 = T.get("llm_first_sentence")
    T6 = T.get("tts_first_audio")
    T7 = T.get("tts_end")

    rows = []
    if T0 and T1: rows.append(("T0→T1  Sarvam 1st word",        ms(T0, T1)))
    if T0 and T2: rows.append(("T0→T2  You speaking",            ms(T0, T2)))
    if T2 and T3: rows.append(("T2→T3  Silence hold",            ms(T2, T3)))
    if T3 and T4: rows.append(("T3→T4  Debounce",                ms(T3, T4)))
    if T4 and T5: rows.append(("T4→T5  LLM → 1st sentence",     ms(T4, T5)))
    if T5 and T6: rows.append(("T5→T6  pyttsx3 speak start",     ms(T5, T6)))
    if T6 and T7: rows.append(("T6→T7  TTS speaking duration",   ms(T6, T7)))
    if T2 and T6: rows.append(("T2→T6  ⚡ END_SPEECH→1st audio", ms(T2, T6)))
    if T0 and T7: rows.append(("T0→T7  ★ TOTAL wall time",       ms(T0, T7)))

    print(f"\n{MAG}{'─'*64}")
    print(f"  TURN LATENCY BREAKDOWN")
    print(f"{'─'*64}")
    for label, millis in rows:
        b = bbar(millis / 1000)
        star = "★" if "TOTAL" in label else ("⚡" if "END_SPEECH" in label else " ")
        print(f"  {star} {label:<32} {millis:>7.0f} ms  {b}")
    print(f"{'─'*64}{RESET}\n", flush=True)

# ─── Audio helpers ────────────────────────────────────────────────────────────

def add_wav_header(pcm: bytes, sr: int = SAMPLE_RATE) -> bytes:
    n = len(pcm)
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", n + 36, b"WAVE",
        b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16,
        b"data", n,
    ) + pcm

def calc_rms(pcm: bytes) -> float:
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(s ** 2))) if len(s) else 0.0

def clean_for_tts(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[•\*#`]", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()

def only_hesitation(text: str) -> bool:
    return bool(text) and all(w in HESITATION_WORDS for w in text.lower().split())

# ─── pyttsx3 — single dedicated thread owns the engine forever ───────────────
# One engine, one thread, lives for the whole process.
# Each sentence is a (text, done_event, cancel_flag) tuple on the queue.
# The worker checks cancel_flag between words via an onWord callback;
# if set it calls engine.stop() to cut playback short.

_tts_q = _q.Queue()

def _tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", TTS_RATE)

    while True:
        item = _tts_q.get()
        if item is None:        # shutdown signal
            break

        sentence, done_ev, cancel_flag = item

        # onWord fires during speech — lets us bail mid-sentence
        def on_word(name, location, length):
            if cancel_flag.is_set():
                engine.stop()

        token = engine.connect("started-word", on_word)
        try:
            engine.say(sentence)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            engine.disconnect(token)   # pass token, not topic string
            done_ev.set()

def speak_sentence_sync(sentence: str, cancel_flag: threading.Event) -> bool:
    """
    Speak one sentence synchronously.
    cancel_flag is a threading.Event — set it to interrupt.
    Returns True if completed normally, False if cancelled.
    """
    if cancel_flag.is_set():
        return False
    done_ev = threading.Event()
    _tts_q.put((sentence, done_ev, cancel_flag))
    done_ev.wait()              # blocks executor thread, not the event loop
    return not cancel_flag.is_set()

# ─── Sentence splitter ────────────────────────────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

# ─── LLM streaming with sentence extraction (runs in executor) ───────────────

def stream_graph_sentences(transcript: str, thread_id: str,
                            sentence_q: _q.Queue, cancel_flag: threading.Event):
    config = {"configurable": {"thread_id": thread_id}}
    buffer = ""
    try:
        for chunk, _ in graph.stream(
            {"messages": transcript}, config, stream_mode="messages"
        ):
            if cancel_flag.is_set():
                break
            if hasattr(chunk, "content") and chunk.content:
                if type(chunk).__name__ in ("AIMessageChunk", "AIMessage"):
                    buffer += chunk.content
                    parts = _SENT_RE.split(buffer)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            s = clean_for_tts(sentence)
                            if len(s) >= MIN_SENTENCE_CHARS:
                                sentence_q.put(s)
                        buffer = parts[-1]
        # flush remainder
        if buffer.strip() and not cancel_flag.is_set():
            r = clean_for_tts(buffer.strip())
            if r:
                sentence_q.put(r)
    finally:
        sentence_q.put(None)   # sentinel

# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    if not SARVAM_API_KEY:
        sys.exit("ERROR: SARVAM_API_KEY not set in .env")

    # Start pyttsx3 worker thread
    tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_thread.start()

    thread_id    = "local-session-001"
    ev_loop      = asyncio.get_event_loop()
    bot_speaking = asyncio.Event()
    cancel_event = asyncio.Event()
    current_task = [None]

    mic_queue = asyncio.Queue()
    chunk_sz  = int(SAMPLE_RATE * CHUNK_DURATION)

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC STATUS] {status}", flush=True)
        pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        ev_loop.call_soon_threadsafe(mic_queue.put_nowait, pcm)

    sarvam  = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    utt     = {"confirmed_chunks": [], "current_segment": "", "in_speech": False}
    end_tmr = None
    T       = {}

    def reset_T(): T.clear()

    def full_text():
        parts = utt["confirmed_chunks"][:]
        if utt["current_segment"]:
            parts.append(utt["current_segment"])
        return " ".join(p.strip() for p in parts if p.strip())

    def reset_utterance():
        utt["confirmed_chunks"] = []
        utt["current_segment"]  = ""
        utt["in_speech"]        = False

    # ── Core pipeline ─────────────────────────────────────────────────────────

    async def process_transcript(text: str):
        cancel_event.clear()
        log("YOU", text, GREEN)
        log("BOT", "thinking …", YELLOW)

        llm_cancel = threading.Event()   # cancels LLM executor thread
        tts_cancel = threading.Event()   # cancels pyttsx3 — fresh every turn
        sentence_q = _q.Queue()
        full_parts = []
        first_done = False

        T["llm_start"] = time.perf_counter()

        llm_future = ev_loop.run_in_executor(
            None, stream_graph_sentences, text, thread_id, sentence_q, llm_cancel
        )

        async def play_sentences():
            nonlocal first_done
            while True:
                try:
                    sentence = sentence_q.get_nowait()
                except _q.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if sentence is None:
                    break

                if cancel_event.is_set():
                    tts_cancel.set()
                    while sentence_q.get() is not None:
                        pass
                    break

                full_parts.append(sentence)
                log("BOT", f"[sentence] {sentence}", CYAN)

                if not first_done:
                    T["llm_first_sentence"] = time.perf_counter()
                    T["tts_first_audio"]    = time.perf_counter()
                    first_done = True

                # Pass threading.Event — fresh per turn, never reused
                completed = await ev_loop.run_in_executor(
                    None, speak_sentence_sync, sentence, tts_cancel
                )
                if not completed or cancel_event.is_set():
                    tts_cancel.set()
                    while sentence_q.get() is not None:
                        pass
                    break

        bot_speaking.set()
        try:
            await asyncio.gather(llm_future, play_sentences())
        except asyncio.CancelledError:
            llm_cancel.set()
            tts_cancel.set()
            raise
        finally:
            llm_cancel.set()
            bot_speaking.clear()

        if cancel_event.is_set():
            return

        T["tts_end"] = time.perf_counter()
        log("BOT", f"[full] {' '.join(full_parts)}", CYAN)
        print_timeline(T)
        reset_T()

    async def handle_transcript(text: str):
        if current_task[0] and not current_task[0].done():
            cancel_event.set()
            bot_speaking.clear()
            current_task[0].cancel()
            try:
                await current_task[0]
            except asyncio.CancelledError:
                pass
        cancel_event.clear()
        current_task[0] = asyncio.create_task(process_transcript(text))

    # ── Sarvam STT ────────────────────────────────────────────────────────────

    async with sarvam.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code=LANGUAGE,
        high_vad_sensitivity=HIGH_VAD_SENS,
        vad_signals=True,
    ) as stt_ws:

        with sd.InputStream(
            device=MIC_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_sz,
            callback=mic_callback,
        ):
            log("SYS", "🎙  Listening … (Ctrl-C to quit)", GREEN)

            async def sender():
                while True:
                    pcm   = await mic_queue.get()
                    level = calc_rms(pcm)

                    if level > BARGE_IN_RMS and bot_speaking.is_set():
                        cancel_event.set()
                        bot_speaking.clear()
                        if current_task[0] and not current_task[0].done():
                            current_task[0].cancel()
                        log("BARGE", "interrupting bot …", RED)

                    if bot_speaking.is_set():
                        continue

                    if level > 0.005:
                        print(f"\r  mic RMS={level:.4f} {'█'*int(level*200):<30}",
                              end="", flush=True)

                    wav = add_wav_header(pcm)
                    b64 = base64.b64encode(wav).decode()
                    await stt_ws.transcribe(
                        audio=b64, encoding="audio/wav", sample_rate=SAMPLE_RATE,
                    )

            async def receiver():
                nonlocal end_tmr

                async def finalize():
                    await asyncio.sleep(SILENCE_HOLD)
                    T["silence_expired"] = time.perf_counter()

                    if utt["current_segment"]:
                        utt["confirmed_chunks"].append(utt["current_segment"])
                        utt["current_segment"] = ""

                    text = full_text()

                    def skip(reason):
                        log("TURN", f"SKIP — {reason}", RED)
                        reset_utterance(); reset_T()

                    if not text:                            return skip("empty")
                    if len(text.split()) < MIN_INPUT_WORDS: return skip("too short")
                    if only_hesitation(text):               return skip("hesitation only")

                    log("TURN", f"ACCEPTED: '{text}'", GREEN)
                    await asyncio.sleep(LLM_DEBOUNCE)
                    reset_utterance()
                    await handle_transcript(text)

                async for msg in stt_ws:
                    mtype = getattr(msg, "type", None)
                    data  = getattr(msg, "data", None)

                    if mtype == "events" and data:
                        sig = getattr(data, "signal_type", "") or ""
                        if sig == "START_SPEECH":
                            utt["in_speech"] = True
                            if "start_speech" not in T:
                                T["start_speech"] = time.perf_counter()
                            log("VAD", "▶  START_SPEECH", CYAN)
                            if end_tmr and not end_tmr.done():
                                end_tmr.cancel()
                        elif sig == "END_SPEECH":
                            utt["in_speech"] = False
                            T["end_speech"] = time.perf_counter()
                            log("VAD", "■  END_SPEECH", CYAN)
                            if utt["current_segment"]:
                                utt["confirmed_chunks"].append(utt["current_segment"])
                                utt["current_segment"] = ""
                                log("VAD", f"   chunks: {utt['confirmed_chunks']}", CYAN)
                            if end_tmr and not end_tmr.done():
                                end_tmr.cancel()
                            end_tmr = asyncio.create_task(finalize())

                    elif mtype == "data" and data:
                        seg = (getattr(data, "transcript", "") or "").strip()
                        if seg:
                            if "first_interim" not in T:
                                T["first_interim"] = time.perf_counter()
                            utt["current_segment"] = seg
                            print(f"\r{YELLOW}[INTER] {full_text():<80}{RESET}",
                                  end="", flush=True)

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())
            try:
                done, pending = await asyncio.wait(
                    [send_task, recv_task], return_when=asyncio.FIRST_EXCEPTION,
                )
                for t in pending: t.cancel()
                for t in done:
                    exc = t.exception()
                    if exc: log("ERR", repr(exc), RED)
            except asyncio.CancelledError:
                pass

    _tts_q.put(None)
    tts_thread.join(timeout=2)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("SYS", "Stopped.", YELLOW)