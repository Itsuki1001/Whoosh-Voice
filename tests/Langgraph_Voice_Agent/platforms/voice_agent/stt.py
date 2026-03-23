"""
stt.py — Ultra-low-latency Speech-to-Text via Sarvam AI streaming.

Key design decisions for minimum latency:
  - Raw PCM sent immediately (no buffering, no WAV re-wrapping per chunk)
  - RMS computed with fast integer math, no NumPy allocation per chunk
  - Barge-in check is non-blocking (fire-and-forget coroutine)
  - Finalization timer uses asyncio.call_later, not sleep()
  - Queue uses put_nowait to avoid scheduler yield on hot path
  - No hesitation / sentence-complete filtering (caller decides)

Usage:
    session = STTSession(api_key, on_transcript, on_interim, on_barge_in)
    await session.run(audio_queue, bot_speaking, cancel_event, current_task)

Test locally:
    python stt.py
"""

from __future__ import annotations

import asyncio
import base64
import struct
import array
import math
from sarvamai import AsyncSarvamAI

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK_DURATION = 0.05        # 50 ms — halved from 100 ms for lower first-byte latency
SILENCE_HOLD   = 0.3       # seconds after END_SPEECH before finalizing
BARGE_IN_RMS   = 0.08        # normalised RMS threshold


# ── Audio helpers ─────────────────────────────────────────────────────────────

# Single WAV header sent once at stream open; subsequent chunks are raw PCM.
def make_wav_header(sample_rate: int = SAMPLE_RATE) -> bytes:
    """Return a 44-byte WAV header with data_size=0 (streaming / unknown length)."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",   # size=MAX signals streaming
        b"fmt ", 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b"data", 0xFFFFFFFF,
    )


def rms_int16(pcm_bytes: bytes) -> float:
    """Fast RMS on raw int16 PCM without NumPy allocation."""
    if not pcm_bytes:
        return 0.0
    samples = array.array("h")          # signed short
    samples.frombytes(pcm_bytes)
    n = len(samples)
    if n == 0:
        return 0.0
    sq_sum = sum(s * s for s in samples)
    return math.sqrt(sq_sum / n) / 32768.0


# ── STT Session ───────────────────────────────────────────────────────────────

class STTSession:
    """
    Single STT streaming session.  One instance per WebSocket connection.

    Callbacks (all async):
        on_transcript(text)   — final, confirmed utterance
        on_interim(text)      — rolling partial (optional, may be None)
        on_barge_in()         — user spoke while bot was talking (optional)
    """

    __slots__ = ("_sarvam", "on_transcript", "on_interim", "on_barge_in")

    def __init__(
        self,
        api_key: str,
        on_transcript,           # async (text: str) -> None
        on_interim=None,         # async (text: str) -> None  |  None
        on_barge_in=None,        # async ()           -> None  |  None
    ):
        self._sarvam       = AsyncSarvamAI(api_subscription_key=api_key)
        self.on_transcript = on_transcript
        self.on_interim    = on_interim
        self.on_barge_in   = on_barge_in

    # ── public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        audio_queue: asyncio.Queue,   # bytes chunks | None sentinel
        bot_speaking: asyncio.Event,
        cancel_event: asyncio.Event,
        current_task: list,           # current_task[0] = active asyncio.Task | None
    ) -> None:
        loop = asyncio.get_running_loop()

        async with self._sarvam.speech_to_text_streaming.connect(
            model="saaras:v3",
            mode="transcribe",
            language_code="en-IN",
            high_vad_sensitivity=True,
            vad_signals=True,
        ) as ws:
            # Send the WAV header once to declare the stream format
            header_b64 = base64.b64encode(make_wav_header()).decode()
            await ws.transcribe(audio=header_b64, encoding="audio/wav",
                                sample_rate=SAMPLE_RATE)

            await asyncio.gather(
                self._send_audio(ws, audio_queue, bot_speaking,
                                 cancel_event, current_task, loop),
                self._recv_transcripts(ws, loop),
            )

    # ── internal: audio sender ────────────────────────────────────────────────

    async def _send_audio(self, ws, audio_queue, bot_speaking,
                          cancel_event, current_task, loop):
        while True:
            chunk: bytes | None = await audio_queue.get()
            if chunk is None:
                break

            # Barge-in: fire-and-forget, never blocks the audio path
            if rms_int16(chunk) > BARGE_IN_RMS:
                loop.create_task(
                    self._handle_barge_in(bot_speaking, cancel_event, current_task)
                )

            if bot_speaking.is_set():
                continue                # discard audio while bot speaks

            # Send raw PCM directly — no per-chunk WAV header
            await ws.transcribe(
                audio=base64.b64encode(chunk).decode(),
                encoding="audio/wav",   # stream continuation; header already sent
                sample_rate=SAMPLE_RATE,
            )

    async def _handle_barge_in(self, bot_speaking, cancel_event, current_task):
        cancel_event.set()
        bot_speaking.clear()
        task = current_task[0]
        if task and not task.done():
            task.cancel()
        if self.on_barge_in:
            await self.on_barge_in()

    # ── internal: transcript receiver ─────────────────────────────────────────

    async def _recv_transcripts(self, ws, loop):
        end_speech_received = False
        confirmed: list[str] = []
        current:   str       = ""
        fin_handle           = None   # asyncio.TimerHandle

        def full_text() -> str:
            parts = [*confirmed, current] if current else confirmed
            return " ".join(p.strip() for p in parts if p.strip())

        def schedule_finalize():
            nonlocal fin_handle
            if fin_handle:
                fin_handle.cancel()
            fin_handle = loop.call_later(SILENCE_HOLD, _trigger_finalize)

        def _trigger_finalize():
            loop.create_task(_finalize())

        async def _finalize():
            print("[STT] finalize fired")
            nonlocal confirmed, current, fin_handle
            text = full_text()
            confirmed.clear()
            current   = ""
            fin_handle = None
            if not text:
                return
            print(f"[STT] ✔ '{text}'")
            await self.on_transcript(text)

        async for msg in ws:
            msg_type = getattr(msg, "type", None)
            data     = getattr(msg, "data", None)
            if not data:
                continue

            if msg_type == "events":
                sig = getattr(data, "signal_type", "") or ""
                if sig == "START_SPEECH":
                    if fin_handle:
                        fin_handle.cancel()
                        fin_handle = None
                        if current:                        # ← save previous sentence
                            confirmed.append(current)
                            current = ""
                    print("[STT] ▶")

                elif sig == "END_SPEECH":
                    end_speech_received = True   # ← just set a flag, nothing else
                    print("[STT] ■")

            elif msg_type == "data":
                t = (getattr(data, "transcript", "") or "").strip()
                if t:
                    current = t
                    if end_speech_received:
                        end_speech_received = False
                        if fin_handle:
                            fin_handle.cancel()
                        fin_handle = loop.call_later(SILENCE_HOLD, _trigger_finalize)
                        print(f"[STT] timer started for {SILENCE_HOLD}s")
                    else:
                        if self.on_interim:
                            loop.create_task(self.on_interim(full_text()))


# ── Local test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import sounddevice as sd

    load_dotenv()

    async def _test():
        print("🎤  Listening… Ctrl-C to stop\n")

        audio_queue  = asyncio.Queue()
        bot_speaking = asyncio.Event()
        cancel_event = asyncio.Event()
        current_task = [None]
        loop         = asyncio.get_running_loop()

        async def on_transcript(text: str):
            print(f"\n✅  {text}\n")

        stt = STTSession(
            api_key=os.getenv("SARVAM_API_KEY"),
            on_transcript=on_transcript,
        )

        chunk_frames = int(SAMPLE_RATE * CHUNK_DURATION)

        def mic_cb(indata, frames, time, status):
            try:
                audio_queue.put_nowait(bytes(indata))   # put_nowait: no scheduler yield
            except asyncio.QueueFull:
                pass  # drop under backpressure

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=chunk_frames,
            dtype="int16",
            channels=1,
            callback=mic_cb,
            device=2
        )
        stream.start()
        try:
            await stt.run(audio_queue, bot_speaking, cancel_event, current_task)
        finally:
            stream.stop()
            stream.close()
            audio_queue.put_nowait(None)

    try:
        asyncio.run(_test())
    except KeyboardInterrupt:
        print("\nStopped.")