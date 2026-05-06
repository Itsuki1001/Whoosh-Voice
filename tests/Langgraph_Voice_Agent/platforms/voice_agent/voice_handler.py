import asyncio
import sys
import sounddevice as sd
import voice.ws_routes as ws_routes
import json
import numpy as np
import time
import threading
import os
import re
import queue
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
print(ELEVENLABS_API_KEY)

DEEPGRAM_API_KEY   = os.getenv("DEEPGRAM_API_KEY")
print(DEEPGRAM_API_KEY)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from graph.graph_voice import graph



SAMPLE_RATE        = 16000
VOICE_THREAD_ID    = "voice-session-1"
MIN_INPUT_WORDS    = 3
FLUSH_CHARS        = {'.', '!', '?', ','}
MIN_CHUNK_WORDS    = 5
LLM_DEBOUNCE_SEC   = 0.4
BARGE_IN_RMS       = 0.04  # raise = harder to interrupt, lower = more sensitive

VOICE_ID  = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = "eleven_multilingual_v2"

eleven = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ─── State ────────────────────────────────────────────────────────
bot_speaking       = threading.Event()
barge_in_event     = threading.Event()
tts_lock           = threading.Lock()
llm_debounce_timer = None

# ─── Audio player ─────────────────────────────────────────────────
audio_queue = queue.Queue()

def clear_audio_queue():
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: pass

def audio_player():
    with sd.OutputStream(samplerate=24000, channels=1, dtype="int16") as stream:
        while True:
            try:
                chunk = audio_queue.get(timeout=0.2)
                if chunk is None:
                    return
                for i in range(0, len(chunk), 2048):
                    if barge_in_event.is_set():
                        clear_audio_queue()
                        print("\n[BARGE-IN] Stopped.")
                        return
                    stream.write(chunk[i:i+2048])
            except queue.Empty:
                return

# ─── Latency ──────────────────────────────────────────────────────
class Latency:
    def __init__(self):
        self.t = {}

    def mark(self, event: str):
        self.t[event] = time.perf_counter()

    def report(self):
        t = self.t
        print("\n── Latency ─────────────────────────")
        if "llm_first_token" in t and "speech_final" in t:
            print(f"  STT → LLM first token : {int((t['llm_first_token'] - t['speech_final'])*1000)}ms")
        if "tts_start" in t and "llm_done" in t:
            print(f"  LLM done → TTS fetch  : {int((t['tts_start'] - t['llm_done'])*1000)}ms")
        if "audio_play" in t and "speech_final" in t:
            print(f"  STT → audio playing   : {int((t['audio_play'] - t['speech_final'])*1000)}ms (total)")
        print("────────────────────────────────────\n")

# ─── TTS ──────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = re.sub(r'[•\*#]', '', text)
    return re.sub(r'\n', ' ', text).strip()

def speak_phrases(phrases: list[str], latency: Latency):
    full_text = " ".join(clean(p) for p in phrases)
    if not full_text:
        return

    with tts_lock:
        latency.mark("tts_start")
        try:
            audio_gen = eleven.text_to_speech.convert(
                text=full_text,
                voice_id=VOICE_ID,
                model_id=TTS_MODEL,
                output_format="pcm_24000",
            )
            audio_np = np.frombuffer(b"".join(audio_gen), dtype=np.int16)
        except Exception as e:
            print(f"\n[TTS ERROR] {e}")
            return

        if barge_in_event.is_set():
            print("\n[TTS] Skipped — barge-in during fetch")
            return

        barge_in_event.clear()
        bot_speaking.set()
        latency.mark("audio_play")

        try:
            t = threading.Thread(target=audio_player, daemon=True)
            t.start()
            audio_queue.put(audio_np)
            audio_queue.put(None)
            t.join()
        finally:
            bot_speaking.clear()
            latency.report()

# ─── LangGraph ────────────────────────────────────────────────────
def process_with_graph(transcript: str):
    if bot_speaking.is_set():
        barge_in_event.set()
        with tts_lock:
            pass
    barge_in_event.clear()

    latency = Latency()
    latency.mark("speech_final")

    print(f"\n[YOU] {transcript}")
    print("[BOT] ", end="", flush=True)

    config = {"configurable": {"thread_id": VOICE_THREAD_ID}}
    buf = ""
    phrases = []
    first_token = True

    for chunk, metadata in graph.stream(
        {"messages": transcript},
        config,
        stream_mode="messages",
    ):
        if hasattr(chunk, "content") and chunk.content:
            if type(chunk).__name__ in ("AIMessageChunk", "AIMessage"):
                token = chunk.content
                if first_token:
                    latency.mark("llm_first_token")
                    first_token = False
                print(token, end="", flush=True)
                buf += token
                last_char = buf.rstrip()[-1] if buf.rstrip() else ""
                if last_char in FLUSH_CHARS and len(buf.split()) >= MIN_CHUNK_WORDS:
                    phrases.append(buf)
                    buf = ""

    if buf.strip():
        phrases.append(buf)

    latency.mark("llm_done")
    print()

    if phrases:
        threading.Thread(target=speak_phrases, args=(phrases, latency), daemon=True).start()

# ─── Main voice loop ──────────────────────────────────────────────
async def voice_loop():
    uri = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-2&language=en-US"
        "&encoding=linear16"
        f"&sample_rate={SAMPLE_RATE}"
        "&interim_results=true"
        "&vad_events=true"
        "&utterance_end_ms=1200"
    )
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    print("🎙️  Voice agent ready. Speak now...\n")

    async with ws_routes.connect(uri, additional_headers=headers) as ws:

        async def send_audio():
            loop = asyncio.get_event_loop()
            def callback(indata, frames, time_info, status):
                # RMS-based barge-in — only triggers if voice is loud enough
                if bot_speaking.is_set():
                    rms = float(np.sqrt(np.mean(indata ** 2)))
                    print(f"RMS: {rms:.4f}", end="\r")
                    if rms > BARGE_IN_RMS:
                        barge_in_event.set()
                audio = (indata * 32767).astype(np.int16).tobytes()
                asyncio.run_coroutine_threadsafe(ws.send(audio), loop)

            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", callback=callback,device=2):
                await asyncio.sleep(9999)

        async def receive_and_process():
            global llm_debounce_timer

            async for msg in ws:
                result = json.loads(msg)

                if result.get("type") == "Results":
                    transcript   = result["channel"]["alternatives"][0]["transcript"]
                    is_final     = result.get("is_final", False)
                    speech_final = result.get("speech_final", False)

                    if transcript and is_final:
                        print(f"[{time.strftime('%H:%M:%S')}] ✅ {transcript}", end="")

                    if speech_final and transcript and len(transcript.split()) >= MIN_INPUT_WORDS:
                        if llm_debounce_timer:
                            llm_debounce_timer.cancel()
                        llm_debounce_timer = threading.Timer(
                            LLM_DEBOUNCE_SEC,
                            lambda t=transcript: threading.Thread(
                                target=process_with_graph, args=(t,), daemon=True
                            ).start()
                        )
                        llm_debounce_timer.start()
                    elif speech_final and transcript:
                        print(f"\n[IGNORED] '{transcript}'")

        await asyncio.gather(send_audio(), receive_and_process())

if __name__ == "__main__":
    asyncio.run(voice_loop())