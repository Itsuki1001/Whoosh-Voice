"""
tts.py — Cartesia WebSocket streaming with continuations.
Single persistent WS connection per turn, sentences pushed as they arrive.
"""

import asyncio
import queue as _q
import re

from cartesia import Cartesia

# ── Config ────────────────────────────────────────────────────────────────────
VOICE_ID           = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
TTS_MODEL          = "sonic-3"
OUTPUT_FORMAT      = {
    "container":   "raw",
    "encoding":    "pcm_s16le",
    "sample_rate": 24000,
}
MIN_SENTENCE_CHARS = 1

_cartesia: Cartesia | None = None


def init_tts(api_key: str):
    global _cartesia
    _cartesia = Cartesia(api_key=api_key)


_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(buffer: str) -> tuple[list[str], str]:
    parts = _SENT_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    complete = [s for s in parts[:-1] if len(s.strip()) >= MIN_SENTENCE_CHARS]
    return complete, parts[-1]


# ── WebSocket TTS worker (runs in executor — Cartesia WS is sync) ─────────────

def _ws_tts_worker(
    sentence_q: _q.Queue,
    audio_q:    _q.Queue,
    cancel_flag: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
    collected_sentences: list, 
):
    """
    Blocking worker — runs in a thread via run_in_executor.
    Opens ONE WebSocket, pushes sentences as they arrive, puts audio
    chunks onto audio_q for the async side to forward.
    Puts None on audio_q as sentinel when done.
    """
    with _cartesia.tts.websocket_connect() as connection:
        ctx = connection.context(
            model_id=TTS_MODEL,
            voice={"mode": "id", "id": VOICE_ID},
            output_format=OUTPUT_FORMAT,
        )

        # ── sender: push sentences from sentence_q into WS context ───────────
        def send_sentences():
            while True:
                try:
                    sentence = sentence_q.get(timeout=0.05)
                except _q.Empty:
                    if cancel_flag.is_set():
                        break
                    continue
                if sentence is None:
                    break
                if cancel_flag.is_set():
                    break
                collected_sentences.append(sentence) 
                print(f"[TTS] pushing: {sentence[:60]}…")
                ctx.push(sentence)
            ctx.no_more_inputs()

        import threading
        sender = threading.Thread(target=send_sentences, daemon=True)
        sender.start()

        # ── receiver: forward audio chunks onto audio_q ───────────────────────
        for response in ctx.receive():
            if cancel_flag.is_set():
                break
            if response.type == "chunk" and response.audio:
                audio_q.put(response.audio)

        sender.join()

    audio_q.put(None)  # sentinel


# ── High-level sentence streamer ──────────────────────────────────────────────

class TTSSentenceStreamer:

    def __init__(self, on_audio_chunk):
        self.on_audio_chunk = on_audio_chunk

    async def stream(
        self,
        sentence_q: _q.Queue,
        cancel_event: asyncio.Event,
    ) -> list[str]:
        assert _cartesia is not None, "Call init_tts(api_key) before using TTS."

        loop      = asyncio.get_event_loop()
        audio_q   = _q.Queue()
        full_parts: list[str] = []

        # track sentences for return value
        original_put = sentence_q.put

        # ── start WS worker in executor ───────────────────────────────────────
        worker_future = loop.run_in_executor(
            None,
            _ws_tts_worker,
            sentence_q, audio_q, cancel_event, loop,full_parts
        )

        # ── drain audio_q and forward chunks ─────────────────────────────────
        while True:
            try:
                chunk = audio_q.get_nowait()
            except _q.Empty:
                await asyncio.sleep(0.01)
                continue

            if chunk is None:
                break

            await self.on_audio_chunk(chunk)
            await asyncio.sleep(0)

        await worker_future
        return full_parts