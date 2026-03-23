"""
test_stt_local.py — Local test for STTSession.
Mimics exactly what the server does: sounddevice callback → audio_queue → STTSession.
Run: python test_stt_local.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# ── same imports as server ────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from stt import STTSession, SAMPLE_RATE, CHUNK_DURATION

try:
    import sounddevice as sd
except ImportError:
    print("[ERROR] pip install sounddevice")
    raise SystemExit(1)


async def main():
    print("🎤  Speak into your mic — Ctrl-C to stop\n")

    audio_queue  = asyncio.Queue()   # same as server's audio_queue
    bot_speaking = asyncio.Event()   # never set — we're not doing TTS here
    cancel_event = asyncio.Event()
    current_task = [None]

    loop = asyncio.get_running_loop()

    # ── callbacks (same as server's on_transcript / on_interim) ──────────────
    async def on_transcript(text: str):
        print(f"\n✅  TRANSCRIPT: {text}\n")

    async def on_interim(text: str):
        print(f"   …{text}", end="\r", flush=True)

    async def on_barge_in():
        print("[barge-in]")

    stt = STTSession(
        api_key=os.getenv("SARVAM_API_KEY"),
        on_transcript=on_transcript,
        on_interim=on_interim,
        on_barge_in=on_barge_in,
    )

    # ── mic → audio_queue (exactly like browser WebSocket → audio_queue) ──────
    chunk_size = int(SAMPLE_RATE * CHUNK_DURATION)

    def mic_callback(indata, frames, time, status):
        # browser sends raw PCM bytes; sounddevice gives us the same thing
        asyncio.run_coroutine_threadsafe(
            audio_queue.put(bytes(indata)), loop
        )

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=chunk_size,   # fires every CHUNK_DURATION seconds — real-time
        dtype="int16",
        channels=1,
        callback=mic_callback,
    )
    stream.start()

    try:
        await stt.run(audio_queue, bot_speaking, cancel_event, current_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stream.stop()
        stream.close()
        await audio_queue.put(None)
        print("\nStopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass