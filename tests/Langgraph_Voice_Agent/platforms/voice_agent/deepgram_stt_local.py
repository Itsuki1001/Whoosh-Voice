"""
Deepgram Real-Time STT — Microphone Streaming (sounddevice)
SDK : deepgram-sdk == 6.0.x
Mic : sounddevice

Install deps:
    pip install "deepgram-sdk>=6.0.0" python-dotenv sounddevice numpy

.env must contain:
    DEEPGRAM_API_KEY=your_key_here
"""

import os
import queue
import threading
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from deepgram import DeepgramClient
from deepgram.core.events import EventType

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in .env")

SAMPLE_RATE = 16000
CHANNELS    = 1
BLOCK_SIZE  = 2000


def main():
    client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    with client.listen.v1.connect(
        model            = "nova-2",
        language         = "en-US",
        encoding         = "linear16",
        channels         = str(CHANNELS),
        sample_rate      = str(SAMPLE_RATE),
        punctuate        = "true",
        smart_format     = "true",
        interim_results  = "true",
        utterance_end_ms = "1000",
        vad_events       = "true",
        endpointing      = "300",
    ) as connection:

        stop_event  = threading.Event()
        audio_queue = queue.Queue()

        # ── Event handlers ───────────────────────────────────────────────────
        def on_open(_):
            print("✅ Connected to Deepgram — speak now (Ctrl+C to stop)\n")

        def on_message(message):
            msg_type = getattr(message, "type", None)
            if msg_type == "Results":
                try:
                    sentence = message.channel.alternatives[0].transcript
                except (AttributeError, IndexError):
                    return
                if not sentence:
                    return
                tag = "✔ FINAL   " if message.is_final else "… interim "
                print(f"{tag}: {sentence}")
            elif msg_type == "SpeechStarted":
                print("🎙️  Speech detected...")
            elif msg_type == "UtteranceEnd":
                print("🔇 Utterance ended\n")
            elif msg_type == "Metadata":
                print(f"ℹ️  request_id={getattr(message, 'request_id', '?')}")

        def on_error(error):
            print(f"❌ Error: {error}")

        def on_close(_):
            print("🔌 Deepgram connection closed.")

        connection.on(EventType.OPEN,    on_open)
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR,   on_error)
        connection.on(EventType.CLOSE,   on_close)

        # ── sounddevice mic callback ─────────────────────────────────────────
        def sd_callback(indata, frames, time, status):
            if status:
                print(f"⚠️  sounddevice: {status}")
            pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            audio_queue.put(pcm)

        # ── sender thread drains queue → Deepgram ────────────────────────────
        def sender_loop():
            while not stop_event.is_set():
                try:
                    pcm = audio_queue.get(timeout=0.5)
                    connection.send_media(pcm)
                except queue.Empty:
                    continue
                except Exception as e:
                    if not stop_event.is_set():
                        print(f"⚠️  send error: {e}")
                    break

        # ── Start mic FIRST so audio is queued immediately ───────────────────
        stream = sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = CHANNELS,
            blocksize  = BLOCK_SIZE,
            dtype      = "float32",
            callback   = sd_callback,
        )
        stream.start()
        print("🎤 Microphone open. Recording...\n")

        sender_thread = threading.Thread(target=sender_loop, daemon=True)
        sender_thread.start()

        # ── THEN start the WebSocket receive loop ────────────────────────────
        # start_listening() blocks in its receive loop on this thread,
        # so run it in a background thread and wait on stop_event instead.
        listener_thread = threading.Thread(
            target=connection.start_listening, daemon=True
        )
        listener_thread.start()

        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            stop_event.set()
            stream.stop()
            stream.close()
            try:
                connection.send_close_stream()
            except Exception:
                pass
            sender_thread.join(timeout=2)
            listener_thread.join(timeout=2)

    print("Done.")


if __name__ == "__main__":
    main()