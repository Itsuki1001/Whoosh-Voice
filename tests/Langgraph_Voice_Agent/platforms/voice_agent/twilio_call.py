import asyncio
import sys
import os
import re
import json
import uuid
import base64
import struct
import threading
import queue as _q
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from sarvamai import AsyncSarvamAI
from elevenlabs.client import ElevenLabs
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE       = os.getenv("TWILIO_PHONE_NUMBER")
YOUR_NUMBER        = os.getenv("YOUR_INDIAN_NUMBER")
NGROK_URL          = os.getenv("NGROK_URL")

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from graph.graph_voice import graph

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_INPUT_WORDS    = 1
BARGE_IN_RMS       = 0.002
SAMPLE_RATE        = 16000
SILENCE_HOLD       = 0.4
HESITATION_WORDS   = {"um", "uh", "hmm", "like"}
MIN_SENTENCE_CHARS = 20

VOICE_ID  = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL = "eleven_multilingual_v2"

DEBUG_AUDIO_DIR = "debug_audio"
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
# ─────────────────────────────────────────────

eleven        = ElevenLabs(api_key=ELEVENLABS_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app           = FastAPI()

call_in_progress = [False]

# ── Outbound call ─────────────────────────────────────────────────────────────

@app.get("/call")
async def make_call():
    if call_in_progress[0]:
        return {"status": "already_calling"}
    call_in_progress[0] = True
    call = twilio_client.calls.create(
        to=YOUR_NUMBER,
        from_=TWILIO_PHONE,
        url=f"{NGROK_URL}/incoming-call",
    )
    print(f"[CALL] SID: {call.sid}")
    return {"status": "calling", "call_sid": call.sid}

# ── TwiML ─────────────────────────────────────────────────────────────────────

@app.post("/incoming-call")
async def incoming_call(request: Request):
    host  = NGROK_URL.replace("https://", "")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{host}/ws"/>
  </Connect>
</Response>"""
    print("[TWIML] Serving stream URL")
    return Response(content=twiml, media_type="application/xml")

# ── Audio conversion ──────────────────────────────────────────────────────────

MULAW_BIAS = 33
MULAW_CLIP = 32635

def ulaw2lin(mulaw_bytes: bytes) -> np.ndarray:
    mulaw = np.frombuffer(mulaw_bytes, dtype=np.uint8).astype(np.int16)
    mulaw = ~mulaw
    sign  = mulaw & 0x80
    exp   = (mulaw >> 4) & 0x07
    mant  = mulaw & 0x0F
    samp  = ((mant << 1) + 33) << exp
    samp  = np.where(sign != 0, -samp, samp)
    return samp.astype(np.int16)

def lin2ulaw(samples: np.ndarray) -> bytes:
    samples = samples.astype(np.int16)
    sign    = np.where(samples < 0, 0x80, 0x00).astype(np.uint8)
    samples = np.clip(np.abs(samples.astype(np.int32)), 0, MULAW_CLIP)
    samples = samples + MULAW_BIAS
    exp     = (np.floor(np.log2(samples + 1)) - 5).clip(0, 7).astype(np.int16)
    mant    = (samples >> (exp + 1)) & 0x0F
    mulaw   = ~(sign | (exp.astype(np.uint8) << 4) | mant.astype(np.uint8))
    return mulaw.tobytes()

def resample(pcm_bytes: bytes, from_rate: int, to_rate: int) -> bytes:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0: return b""
    new_count = int(len(samples) * to_rate / from_rate)
    if new_count == 0: return b""
    old_ix = np.linspace(0, len(samples) - 1, len(samples))
    new_ix = np.linspace(0, len(samples) - 1, new_count)
    return np.interp(new_ix, old_ix, samples).astype(np.int16).tobytes()

def mulaw_to_pcm16k(mulaw_bytes: bytes) -> bytes:
    pcm_8k = ulaw2lin(mulaw_bytes).tobytes()
    return resample(pcm_8k, 8000, 16000)

def pcm24k_to_mulaw8k(pcm_24k: bytes) -> bytes:
    pcm_8k  = resample(pcm_24k, 24000, 8000)
    samples = np.frombuffer(pcm_8k, dtype=np.int16)
    return lin2ulaw(samples)

def add_wav_header(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", data_size + 36, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate,
        sample_rate * 2, 2, 16,
        b"data", data_size,
    )
    return header + pcm_bytes

def check_rms(audio_bytes: bytes) -> float:
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

def save_wav(filename: str, pcm_bytes: bytes, sample_rate: int = 16000):
    path = os.path.join(DEBUG_AUDIO_DIR, filename)
    with open(path, "wb") as f:
        f.write(add_wav_header(pcm_bytes, sample_rate=sample_rate))
    print(f"[WAV] Saved {path} ({len(pcm_bytes)/(sample_rate*2):.1f}s)")

# ── Text helpers ──────────────────────────────────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?])\s+')

def clean(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[•\*#`]', '', text)
    return re.sub(r'\n', ' ', text).strip()

def only_hesitation(text: str) -> bool:
    return bool(text) and all(w in HESITATION_WORDS for w in text.lower().split())

def is_sentence_complete(text: str) -> bool:
    text = text.strip()
    if not text: return False
    if text[-1] in ".!?": return True
    if len(text.split()) >= 6: return True
    return False

# ── LLM sentence streamer ─────────────────────────────────────────────────────

def stream_graph_sentences(transcript: str, thread_id: str,
                            sentence_q: _q.Queue, cancel_flag: threading.Event):
    config = {"configurable": {"thread_id": thread_id}}
    buffer = ""
    try:
        for chunk, _ in graph.stream(
            {"messages": transcript}, config, stream_mode="messages"
        ):
            if cancel_flag.is_set(): break
            if hasattr(chunk, "content") and chunk.content:
                if type(chunk).__name__ in ("AIMessageChunk", "AIMessage"):
                    buffer += chunk.content
                    parts = _SENT_RE.split(buffer)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            s = clean(sentence)
                            if len(s) >= MIN_SENTENCE_CHARS:
                                sentence_q.put(s)
                        buffer = parts[-1]
        if buffer.strip() and not cancel_flag.is_set():
            r = clean(buffer.strip())
            if r: sentence_q.put(r)
    except Exception as e:
        print(f"[LLM ERROR] {e}")
    finally:
        sentence_q.put(None)

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] ✓ Connected")

    thread_id    = f"twilio-{uuid.uuid4()}"
    bot_speaking = asyncio.Event()
    cancel_event = asyncio.Event()
    audio_queue  = asyncio.Queue()
    current_task = [None]
    stream_sid   = [None]
    media_count  = [0]

    pcm_collect     = []
    snapshots_saved = [0]

    sarvam = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

    # ── Twilio helpers ────────────────────────────────────────────────────────

    async def send_audio_to_twilio(mulaw_bytes: bytes):
        if not stream_sid[0]: return
        try:
            await ws.send_text(json.dumps({
                "event":     "media",
                "streamSid": stream_sid[0],
                "media":     {"payload": base64.b64encode(mulaw_bytes).decode()}
            }))
        except Exception as e:
            print(f"[OUT ERROR] {e}")

    async def send_clear_to_twilio():
        """Flush Twilio's audio buffer — stops playback immediately."""
        if not stream_sid[0]: return
        try:
            await ws.send_text(json.dumps({
                "event":     "clear",
                "streamSid": stream_sid[0]
            }))
            print("[TWILIO] ✓ Buffer cleared")
        except Exception as e:
            print(f"[CLEAR ERROR] {e}")

    # ── STT ───────────────────────────────────────────────────────────────────

    async def stt_loop():
        bytes_per_chunk = int(SAMPLE_RATE * 0.5) * 2   # 500ms = 16000 bytes
        buffer = b""

        utt = {"confirmed_chunks": [], "current_segment": "", "in_speech": False}
        end_tmr = None

        def full_text():
            parts = utt["confirmed_chunks"][:]
            if utt["current_segment"]: parts.append(utt["current_segment"])
            return " ".join(p.strip() for p in parts if p.strip())

        def reset_utterance():
            utt["confirmed_chunks"] = []
            utt["current_segment"]  = ""
            utt["in_speech"]        = False

        print("[STT] Connecting to Sarvam...")
        try:
            async with sarvam.speech_to_text_streaming.connect(
                model="saaras:v3",
                mode="transcribe",
                language_code="en-IN",
                high_vad_sensitivity=True,
                vad_signals=True,
            ) as stt_ws:
                print("[STT] ✓ Connected")

                async def send_audio_to_sarvam():
                    nonlocal buffer
                    chunks_sent        = 0
                    barge_in_triggered = False

                    while True:
                        chunk = await audio_queue.get()
                        if chunk is None:
                            if buffer:
                                print(f"[STT] Flushing {len(buffer)} bytes")
                                wav = add_wav_header(buffer)
                                await stt_ws.transcribe(
                                    audio=base64.b64encode(wav).decode(),
                                    encoding="audio/wav",
                                    sample_rate=SAMPLE_RATE,
                                )
                            break

                        rms = check_rms(chunk)

                        # Barge-in — only trigger once per bot turn
                        if bot_speaking.is_set() and not barge_in_triggered \
                                and rms > BARGE_IN_RMS:
                            print(f"[BARGE-IN] ✓ rms={rms:.4f} — interrupting bot")
                            barge_in_triggered = True
                            cancel_event.set()
                            bot_speaking.clear()
                            if current_task[0] and not current_task[0].done():
                                current_task[0].cancel()
                            await send_clear_to_twilio()   # flush Twilio buffer

                        # Reset barge-in flag when bot finishes speaking
                        if not bot_speaking.is_set():
                            barge_in_triggered = False

                        # Don't send audio to Sarvam while bot is speaking
                        if bot_speaking.is_set():
                            continue

                        # Debug snapshots
                        pcm_collect.append(chunk)
                        total = sum(len(c) for c in pcm_collect)
                        if total >= SAMPLE_RATE * 2 * 3 and snapshots_saved[0] < 3:
                            snapshots_saved[0] += 1
                            data = b"".join(pcm_collect)
                            pcm_collect.clear()
                            save_wav(f"snap_{snapshots_saved[0]}.wav", data)

                        buffer += chunk
                        if len(buffer) >= bytes_per_chunk:
                            chunks_sent += 1
                            wav = add_wav_header(buffer)
                            await stt_ws.transcribe(
                                audio=base64.b64encode(wav).decode(),
                                encoding="audio/wav",
                                sample_rate=SAMPLE_RATE,
                            )
                            buffer = b""
                            print(f"[STT] Sent chunk #{chunks_sent} (500ms)")

                async def receive_from_sarvam():
                    nonlocal end_tmr
                    msg_count = 0

                    async def finalize():
                        await asyncio.sleep(SILENCE_HOLD)
                        if utt["current_segment"]:
                            utt["confirmed_chunks"].append(utt["current_segment"])
                            utt["current_segment"] = ""
                        text = full_text()
                        print(f"[FINALIZE] '{text}'")
                        if not text:
                            reset_utterance(); return
                        if len(text.split()) < MIN_INPUT_WORDS:
                            print(f"[SKIP] too short: '{text}'")
                            reset_utterance(); return
                        if only_hesitation(text):
                            print(f"[SKIP] hesitation: '{text}'")
                            reset_utterance(); return
                        if not is_sentence_complete(text):
                            print(f"[SKIP] incomplete: '{text}'")
                            reset_utterance(); return
                        print(f"[TURN] ✓ '{text}'")
                        reset_utterance()
                        await handle_transcript(text)

                    async for message in stt_ws:
                        msg_count += 1
                        msg_type = getattr(message, "type", None)
                        data     = getattr(message, "data", None)
                        print(f"[SARVAM #{msg_count}] type={msg_type} "
                              f"data={str(data)[:150]}")

                        if msg_type == "events" and data:
                            signal_type = getattr(data, "signal_type", "") or ""

                            if signal_type == "START_SPEECH":
                                utt["in_speech"] = True
                                print("[VAD] ▶ START_SPEECH")
                                if end_tmr and not end_tmr.done():
                                    end_tmr.cancel()

                                # VAD-based barge-in as backup
                                if bot_speaking.is_set():
                                    print("[BARGE-IN] ✓ VAD detected — interrupting bot")
                                    cancel_event.set()
                                    bot_speaking.clear()
                                    if current_task[0] and not current_task[0].done():
                                        current_task[0].cancel()
                                    await send_clear_to_twilio()

                            elif signal_type == "END_SPEECH":
                                utt["in_speech"] = False
                                print("[VAD] ■ END_SPEECH")
                                if utt["current_segment"]:
                                    utt["confirmed_chunks"].append(utt["current_segment"])
                                    utt["current_segment"] = ""
                                if end_tmr and not end_tmr.done():
                                    end_tmr.cancel()
                                end_tmr = asyncio.create_task(finalize())

                        elif msg_type == "data" and data:
                            text = (getattr(data, "transcript", "") or "").strip()
                            print(f"[TRANSCRIPT] '{text}'")
                            if text:
                                utt["current_segment"] = text
                                print(f"[INTER] '{full_text()}'")

                await asyncio.gather(send_audio_to_sarvam(), receive_from_sarvam())

        except Exception as e:
            print(f"[STT ERROR] {e}")
            import traceback; traceback.print_exc()

    # ── LLM + TTS ─────────────────────────────────────────────────────────────

    async def process_transcript(transcript: str):
        loop = asyncio.get_event_loop()
        cancel_event.clear()

        llm_cancel = threading.Event()
        sentence_q = _q.Queue()
        full_parts = []

        llm_future = loop.run_in_executor(
            None, stream_graph_sentences,
            transcript, thread_id, sentence_q, llm_cancel
        )

        bot_speaking.set()
        print(f"[BOT] Responding to: '{transcript}'")

        async def stream_tts_sentences():
            while True:
                try:
                    sentence = sentence_q.get_nowait()
                except _q.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if sentence is None:
                    print("[TTS] All sentences done")
                    break

                # Check cancel before starting each new sentence
                if cancel_event.is_set():
                    print("[TTS] Cancelled before sentence — draining")
                    while sentence_q.get() is not None: pass
                    break

                full_parts.append(sentence)
                print(f"[TTS] '{sentence[:60]}…'")

                try:
                    audio_stream = await loop.run_in_executor(
                        None,
                        lambda s=sentence: eleven.text_to_speech.stream(
                            text=s,
                            voice_id=VOICE_ID,
                            model_id=TTS_MODEL,
                            output_format="pcm_24000",
                        )
                    )
                    tts_pcm = b""
                    for chunk in audio_stream:
                        # Check cancel on every audio chunk
                        if cancel_event.is_set():
                            print("[TTS] ✓ Cancelled mid-sentence")
                            return   # stop immediately
                        if chunk:
                            tts_pcm += chunk
                            mulaw = pcm24k_to_mulaw8k(chunk)
                            await send_audio_to_twilio(mulaw)
                            await asyncio.sleep(0)

                    if tts_pcm:
                        save_wav(f"tts_{len(full_parts)}.wav", tts_pcm, 24000)

                except Exception as e:
                    print(f"[TTS ERROR] {e}")
                    import traceback; traceback.print_exc()

        try:
            await asyncio.gather(llm_future, stream_tts_sentences())
        except asyncio.CancelledError:
            llm_cancel.set()
            raise
        finally:
            llm_cancel.set()
            bot_speaking.clear()
            print("[BOT] Done speaking")

    async def handle_transcript(transcript: str):
        if current_task[0] and not current_task[0].done():
            cancel_event.set()
            bot_speaking.clear()
            current_task[0].cancel()
            try: await current_task[0]
            except asyncio.CancelledError: pass
        cancel_event.clear()
        current_task[0] = asyncio.create_task(process_transcript(transcript))

    # ── Receive from Twilio ───────────────────────────────────────────────────

    async def receive_from_twilio():
        async for message in ws.iter_text():
            data  = json.loads(message)
            event = data.get("event")

            if event != "media":
                print(f"[TWILIO] {event}")

            if event == "connected":
                print("[TWILIO] ✓ Protocol connected")

            elif event == "start":
                stream_sid[0] = data["start"]["streamSid"]
                tracks = data["start"].get("tracks", [])
                fmt    = data["start"].get("mediaFormat", {})
                print(f"[TWILIO] Stream started — tracks={tracks} format={fmt}")

            elif event == "media":
                media_count[0] += 1
                mulaw_bytes = base64.b64decode(data["media"]["payload"])
                pcm_16k     = mulaw_to_pcm16k(mulaw_bytes)
                await audio_queue.put(pcm_16k)
                if media_count[0] % 100 == 0:
                    print(f"[TWILIO] {media_count[0]} chunks received")

            elif event == "stop":
                print(f"[TWILIO] Stopped ({media_count[0]} total chunks)")
                call_in_progress[0] = False
                if pcm_collect:
                    save_wav("final.wav", b"".join(pcm_collect))
                await audio_queue.put(None)
                break

    # ── Run ───────────────────────────────────────────────────────────────────

    try:
        await asyncio.gather(receive_from_twilio(), stt_loop())
    except WebSocketDisconnect:
        print("[WS] Disconnected")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()
    finally:
        call_in_progress[0] = False
        if current_task[0] and not current_task[0].done():
            current_task[0].cancel()
        await audio_queue.put(None)
        print("[MAIN] Done")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")