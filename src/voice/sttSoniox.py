"""
stt.py — Speech-to-Text via Soniox WebSocket API (async, no SDK).

Usage:
    session = STTSession(api_key, on_transcript, on_interim, on_barge_in)
    await session.run(audio_queue, bot_speaking)

Test locally:
    python stt.py
"""

from __future__ import annotations

import asyncio
import array
import json
import math
import logging

import websockets

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
CHUNK_DURATION     = 0.05       # 50 ms chunks
SILENCE_HOLD       = 0.3        # seconds after endpoint before finalizing
BARGE_IN_RMS       = 0.12       # normalised RMS threshold for barge-in
KEEPALIVE_INTERVAL = 5.0        # keepalive ping every N seconds during silence

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


# ── Soniox session config ─────────────────────────────────────────────────────
def _build_start_message(api_key: str) -> str:
    config = {
        "api_key":                        api_key,
        "model":                          "stt-rt-v4",
        "audio_format":                   "pcm_s16le",
        "sample_rate":                    SAMPLE_RATE,
        "num_channels":                   1,
        "enable_endpoint_detection":      True,
        "enable_language_identification": True,
    }
    return json.dumps(config)


# ── Audio helpers ─────────────────────────────────────────────────────────────
def rms_int16(pcm_bytes: bytes) -> float:
    """Fast RMS on raw int16 PCM — used for barge-in detection."""
    if not pcm_bytes:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm_bytes)
    n = len(samples)
    if n == 0:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / n) / 32768.0


# ── STT Session ───────────────────────────────────────────────────────────────
class STTSession:
    """
    Single async STT session backed by Soniox WebSocket API.
    One instance per user / call leg.

    Callbacks (all async):
        on_transcript(text)  — final confirmed utterance → sent to LLM
        on_interim(text)     — rolling partial shown in UI (optional)
        on_barge_in()        — user spoke while bot was talking (optional)
    """

    __slots__ = ("_api_key", "on_transcript", "on_interim", "on_barge_in")

    def __init__(
        self,
        api_key: str,
        on_transcript,
        on_interim=None,
        on_barge_in=None,
    ):
        self._api_key      = api_key
        self.on_transcript = on_transcript
        self.on_interim    = on_interim
        self.on_barge_in   = on_barge_in

    # ── public entry point ────────────────────────────────────────────────────
    async def run(
        self,
        audio_queue: asyncio.Queue,
        bot_speaking: asyncio.Event,
    ) -> None:
        logging.info("[STT] Connecting to Soniox...")

        async with websockets.connect(
            SONIOX_WS_URL,
            max_size=10 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            await ws.send(_build_start_message(self._api_key))
            logging.info("[STT] Session started.")

            await asyncio.gather(
                self._send_audio(ws, audio_queue, bot_speaking),
                self._recv_transcripts(ws),
            )

    # ── internal: audio sender ────────────────────────────────────────────────
    async def _send_audio(
        self,
        ws,
        audio_queue: asyncio.Queue,
        bot_speaking: asyncio.Event,
    ) -> None:
        while True:
            try:
                chunk: bytes | None = await asyncio.wait_for(
                    audio_queue.get(), timeout=KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                await ws.send(json.dumps({"type": "keepalive"}))
                continue

            if chunk is None:
                await ws.send(b"")
                break

            if self.on_barge_in and rms_int16(chunk) > BARGE_IN_RMS:
                asyncio.ensure_future(self.on_barge_in())

            if bot_speaking.is_set():
                # Send silence to keep Soniox connection alive
                silence = bytes(len(chunk))   # zero-filled same size as chunk
                await ws.send(silence)
                continue

            await ws.send(chunk)
    # ── internal: transcript receiver ─────────────────────────────────────────
    async def _recv_transcripts(self, ws) -> None:
        """
        Soniox token flow:
          - is_final=False  → interim, updates rolling partial
          - is_final=True   → confirmed word, accumulate into confirmed[]
          - text="<end>"    → endpoint marker token (stt-rt-v3 style) — filter out
          - finished=True   → endpoint signal (stt-rt-v4 style)
          Both trigger the SILENCE_HOLD finalize timer.
          Session NEVER breaks on endpoint — stays alive for all utterances.
        """
        loop = asyncio.get_running_loop()

        confirmed: list[str] = []
        current:   str       = ""
        fin_handle           = None
        in_speech            = False

        def full_text() -> str:
            parts = [*confirmed, current] if current else list(confirmed)
            return " ".join(p.strip() for p in parts if p.strip())

        def _trigger_finalize():
            loop.create_task(_finalize())

        async def _finalize():
            nonlocal confirmed, current, fin_handle, in_speech
            if not confirmed and not current:
                return
            text = full_text()
            confirmed.clear()
            current    = ""
            fin_handle = None
            in_speech  = False
            if not text:
                return
            logging.info(f"[STT] final: {text}")
            await self.on_transcript(text)      # ← sent to LLM

        def schedule_finalize():
            nonlocal fin_handle
            if fin_handle:
                fin_handle.cancel()
            fin_handle = loop.call_later(SILENCE_HOLD, _trigger_finalize)

        async for raw in ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logging.warning(f"[STT] Non-JSON frame: {raw!r}")
                continue

            if event.get("error_code"):
                logging.error(f"[STT] Error {event['error_code']}: {event.get('error_message')}")
                continue

            # ── Process tokens ────────────────────────────────────────────────
            new_finals    = []
            new_nonfinals = []
            endpoint_hit  = False

            for tok in event.get("tokens", []):
                t = tok.get("text", "")
                if not t:
                    continue

                # <end> token = Soniox endpoint marker (stt-rt-v3).
                # String comparison is nanoseconds — zero latency impact.
                # Filter it out of transcript text, use as endpoint trigger.
                if t == "<end>":
                    endpoint_hit = True
                    continue

                if tok.get("is_final"):
                    new_finals.append(t)
                else:
                    new_nonfinals.append(t)

            # Accumulate confirmed final words
            if new_finals:
                chunk = "".join(new_finals).strip()
                if chunk:
                    if not in_speech:
                        in_speech = True
                        if fin_handle:
                            fin_handle.cancel()
                            fin_handle = None
                        logging.info("[STT] START_SPEECH")
                    confirmed.append(chunk)
                    current = ""

            # Update rolling interim for UI
            if new_nonfinals:
                current = "".join(new_nonfinals).strip()
                if self.on_interim and in_speech:
                    asyncio.ensure_future(self.on_interim(full_text()))

            # ── Endpoint detection ────────────────────────────────────────────
            # <end> token (v3) OR finished field (v4) — both handled.
            # Start finalize timer. Never break — session stays alive.
            if endpoint_hit or event.get("finished"):
                if in_speech or confirmed or current:
                    logging.info("[STT] END_SPEECH — starting finalize timer")
                    schedule_finalize()
                # else: silent endpoint, keep listening


# ── Local test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import sounddevice as sd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    load_dotenv()

    async def _test():
        logging.info("Listening... Ctrl-C to stop")

        audio_queue  = asyncio.Queue(maxsize=20)
        bot_speaking = asyncio.Event()

        async def on_transcript(text: str):
            print(f"\n FINAL: {text}\n")

        async def on_interim(text: str):
            print(f"\r  interim: {text}   ", end="", flush=True)

        stt = STTSession(
            api_key=os.getenv("SONIOX_API_KEY", ""),
            on_transcript=on_transcript,
            on_interim=on_interim,
        )

        chunk_frames = int(SAMPLE_RATE * CHUNK_DURATION)

        def mic_cb(indata, frames, time, status):
            if audio_queue.full():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                audio_queue.put_nowait(bytes(indata))
            except asyncio.QueueFull:
                pass

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=chunk_frames,
            dtype="int16",
            channels=1,
            callback=mic_cb,
        )
        stream.start()
        try:
            await stt.run(audio_queue, bot_speaking)
        finally:
            stream.stop()
            stream.close()
            await audio_queue.put(None)

    try:
        asyncio.run(_test())
    except KeyboardInterrupt:
        logging.info("Stopped.")