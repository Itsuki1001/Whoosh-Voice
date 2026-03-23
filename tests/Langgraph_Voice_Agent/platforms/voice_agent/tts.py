"""
tts.py — Text-to-Speech module using ElevenLabs sentence-streaming.
"""

import asyncio
import queue as _q
import re
import threading

from elevenlabs.client import ElevenLabs

# ── Config ────────────────────────────────────────────────────────────────────
VOICE_ID           = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL          = "eleven_multilingual_v2"
MIN_SENTENCE_CHARS = 20

# ── ElevenLabs client (module-level singleton) ────────────────────────────────
_eleven: ElevenLabs | None = None


def init_tts(api_key: str):
    """Call once at startup with your ElevenLabs API key."""
    global _eleven
    _eleven = ElevenLabs(api_key=api_key)


# ── Sentence splitter ─────────────────────────────────────────────────────────
_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Returns (complete_sentences, remainder).
    Sentences shorter than MIN_SENTENCE_CHARS are kept in the remainder.
    """
    parts = _SENT_RE.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    complete = [s for s in parts[:-1] if len(s.strip()) >= MIN_SENTENCE_CHARS]
    return complete, parts[-1]


# ── Core TTS streaming function ───────────────────────────────────────────────

def synthesise_sentence(sentence: str):
    """
    Blocking generator — yields raw PCM chunks from ElevenLabs.
    Run this inside loop.run_in_executor().
    """
    assert _eleven is not None, "Call init_tts(api_key) before using TTS."
    return _eleven.text_to_speech.stream(
        text=sentence,
        voice_id=VOICE_ID,
        model_id=TTS_MODEL,
        output_format="pcm_24000",
    )


# ── High-level sentence-streaming TTS ────────────────────────────────────────

class TTSSentenceStreamer:
    """
    Pulls sentences from a queue (produced by LLM streaming) and synthesises
    each one with ElevenLabs, forwarding raw PCM chunks via on_audio_chunk.

    Usage:
        streamer = TTSSentenceStreamer(on_audio_chunk=my_async_fn)
        await streamer.stream(sentence_q, cancel_event)
    """

    def __init__(self, on_audio_chunk):
        """
        on_audio_chunk: async callable(chunk: bytes) — called for every PCM chunk.
        """
        self.on_audio_chunk = on_audio_chunk

    async def stream(
        self,
        sentence_q: _q.Queue,
        cancel_event: asyncio.Event,
    ) -> list[str]:
        """
        Drains sentence_q until a None sentinel is received or cancel_event fires.
        Returns the list of sentences that were fully synthesised.
        """
        loop       = asyncio.get_event_loop()
        full_parts = []

        while True:
            # Non-blocking poll so we yield to the event loop between checks.
            try:
                sentence = sentence_q.get_nowait()
            except _q.Empty:
                await asyncio.sleep(0.01)
                continue

            if sentence is None:
                break   # LLM sentinel — we're done

            if cancel_event.is_set():
                # Drain queue so the LLM thread can exit cleanly.
                while sentence_q.get() is not None:
                    pass
                break

            full_parts.append(sentence)
            print(f"[TTS] synthesising: {sentence[:60]}…")

            try:
                audio_stream = await loop.run_in_executor(
                    None,
                    lambda s=sentence: synthesise_sentence(s),
                )
                for chunk in audio_stream:
                    if cancel_event.is_set():
                        break
                    if chunk:
                        await self.on_audio_chunk(chunk)
                        await asyncio.sleep(0)   # yield to event loop
            except Exception as exc:
                print(f"[TTS ERROR] {exc}")

        return full_parts