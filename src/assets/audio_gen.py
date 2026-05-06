# generate_limit_audio.py
import asyncio
import queue as _q
import os
from dotenv import load_dotenv
from voice.tts import init_tts, TTSSentenceStreamer

load_dotenv()

async def main():
    init_tts(os.getenv("CARTESIA_API_KEY"))
    
    chunks = []

    async def collect(chunk: bytes):
        chunks.append(chunk)

    sentence_q = _q.Queue()
    sentence_q.put("You have reached your usage limit. Please try again later.")
    sentence_q.put(None)

    streamer = TTSSentenceStreamer(on_audio_chunk=collect)
    await streamer.stream(sentence_q, asyncio.Event())

    audio_bytes = b"".join(chunks)
    with open("limit_message.pcm", "wb") as f:
        f.write(audio_bytes)
    print(f"Saved {len(audio_bytes)} bytes")

asyncio.run(main())