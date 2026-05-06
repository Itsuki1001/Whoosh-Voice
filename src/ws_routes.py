"""
websocket.py — WebSocket endpoint, rate limiting, STT/TTS/LLM wiring.
Supports runtime STT switching (Soniox / Sarvam) via query param ?stt=soniox|sarvam
"""

import asyncio
import json
import os
import queue as _q
import re
import threading
import uuid
import random
from collections import defaultdict
import time
import logging
from langchain_core.messages import ToolMessage, AIMessage
from dotenv import load_dotenv
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
from concurrent.futures import ThreadPoolExecutor
from voice.sttSoniox import STTSession as SonioxSTT
from voice.sttSarvam import STTSession as SarvamSTT
from voice.tts import TTSSentenceStreamer, init_tts
from graph.graph_sales_voice import graph as sales_graph
from graph.graph_voice import graph as resort_graph
from graph.graph_customer_support import graph as support_graph

load_dotenv()

SONIOX_API_KEY   = os.getenv("SONIOX_API_KEY")
SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

router = APIRouter()

# ── Config ────────────────────────────────────────────────────────────────────
MIN_SENTENCE_CHARS = 0

# ── Pre-rendered audio files ──────────────────────────────────────────────────
with open("assets//limit_message.pcm", "rb") as f:
    LIMIT_AUDIO = f.read()

with open("assets//greeting_resort.pcm", "rb") as f:
    RESORT_GREETING_AUDIO = f.read()

with open("assets//greeting_sales.pcm", "rb") as f:
    SALES_GREETING_AUDIO = f.read()

with open("assets//greeting_support.pcm", "rb") as f:
    SUPPORT_GREETING_AUDIO = f.read()

# ── HTML ──────────────────────────────────────────────────────────────────────
with open("assets//voice_ui.html", encoding="utf-8") as f:
    HTML = f.read()

# ── Rate limiting ─────────────────────────────────────────────────────────────
ip_connections     = defaultdict(int)
session_requests   = defaultdict(int)
ip_hourly_requests = defaultdict(lambda: {"count": 0, "reset_at": time.time() + 3600})
ip_daily_requests  = defaultdict(lambda: {"count": 0, "reset_at": time.time() + 86400})

MAX_CONCURRENT_PER_IP    = 3
MAX_REQUESTS_PER_SESSION = 100
MAX_HOURLY_REQUESTS      = 100
MAX_DAILY_REQUESTS       = 100

RESORT_VOICE_ID  = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
SALES_VOICE_ID   = "228fca29-3a0a-435c-8728-5cb483251068"
SUPPORT_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"





# ── Rate limit helper ─────────────────────────────────────────────────────────
def is_ip_limit_reached(ip: str) -> bool:
    now = time.time()
    hourly = ip_hourly_requests[ip]
    if now > hourly["reset_at"]:
        hourly["count"] = 0
        hourly["reset_at"] = now + 3600
    if hourly["count"] >= MAX_HOURLY_REQUESTS:
        return True
    daily = ip_daily_requests[ip]
    if now > daily["reset_at"]:
        daily["count"] = 0
        daily["reset_at"] = now + 86400
    if daily["count"] >= MAX_DAILY_REQUESTS:
        return True
    hourly["count"] += 1
    daily["count"] += 1
    return False


# ── Language detection ────────────────────────────────────────────────────────
_LANG_PATTERNS = [
    (re.compile(r'[\u0D00-\u0D7F]'), "ml"),
    (re.compile(r'[\u0900-\u097F]'), "hi"),
    (re.compile(r'[\u0B80-\u0BFF]'), "ta"),
    (re.compile(r'[\u0600-\u06FF]'), "ar"),
]

def detect_language(text: str) -> str:
    for pattern, lang in _LANG_PATTERNS:
        if pattern.search(text):
            return lang
    return "en"


# ── RESORT AGENT FILLERS ──────────────────────────────────────────────────────
RESORT_FILLERS: dict[str, dict[str, list[str]]] = {
    "check_availability_and_prices": {
        "en": ["Sure, let me check availability for those dates...", "Give me a moment to check the rooms...", "Let me see what we have open..."],
        "ml": ["ഒരു നിമിഷം, ഞാൻ മുറികളുടെ ലഭ്യത നോക്കുന്നു...", "ഒന്ന് നോക്കട്ടെ...", "ആ തീയതികൾക്ക് ലഭ്യത ഉണ്ടോ എന്ന് നോക്കാം..."],
        "hi": ["एक पल रुकिए, मैं कमरों की उपलब्धता देख रहा हूँ...", "जरा रुकिए, मैं देखता हूँ..."],
        "ta": ["ஒரு நிமிடம், அறைகள் கிடைக்குமா என்று பார்க்கிறேன்..."],
    },
    "find_next_available_dates": {
        "en": ["Let me find the earliest available dates for you...", "Give me a moment to check when we're free..."],
        "ml": ["ഏറ്റവും അടുത്ത ലഭ്യമായ തീയതികൾ നോക്കുന്നു...", "ഒരു നിമിഷം, ഞാൻ നോക്കുന്നു..."],
        "hi": ["अगली उपलब्ध तारीखें देख रहा हूँ..."],
        "ta": ["அடுத்த கிடைக்கும் தேதிகளை பார்க்கிறேன்..."],
    },
    "hold_room_and_generate_payment": {
        "en": ["Let me reserve that room for you right away...", "Just a moment while I secure your booking..."],
        "ml": ["ഒരു നിമിഷം, ഞാൻ മുറി ബുക്ക് ചെയ്യുന്നു...", "ഉടൻ ബുക്ക് ചെയ്യുന്നു..."],
        "hi": ["अभी कमरा बुक कर रहा हूँ..."],
        "ta": ["உங்கள் அறையை இப்போது பதிவு செய்கிறேன்..."],
    },
    "get_room_details": {
        "en": ["Let me pull up the room details for you...", "Give me a second to check that room..."],
        "ml": ["ഒരു നിമിഷം, ആ മുറിയുടെ വിവരങ്ങൾ നോക്കുന്നു..."],
        "hi": ["उस कमरे की जानकारी देख रहा हूँ..."],
        "ta": ["அந்த அறையின் விவரங்களை பார்க்கிறேன்..."],
    },
    "get_distance_to_homestay": {
        "en": ["Let me calculate the distance for you...", "Give me a second to check that route..."],
        "ml": ["ദൂരം നോക്കുന്നു, ഒരു നിമിഷം...", "വഴി പരിശോധിക്കുന്നു..."],
        "hi": ["दूरी देख रहा हूँ, एक पल..."],
        "ta": ["தூரத்தை சரிபார்க்கிறேன்..."],
    },
    "rag_tool": {
        "en": ["Let me look that up for you...", "Give me a moment to find that information...", "I'll check that right away..."],
        "ml": ["ഒരു നിമിഷം, ഞാൻ നോക്കുന്നു...", "അത് ഉടൻ നോക്കുന്നു..."],
        "hi": ["एक पल में देखता हूँ..."],
        "ta": ["ஒரு நிமிடம், நான் பார்க்கிறேன்..."],
    },
}

RESORT_DEFAULT_FILLERS: dict[str, list[str]] = {
    "en": ["Give me just a moment...", "One moment please...", "Let me check that..."],
    "ml": ["ഒരു നിമിഷം...", "ഉടൻ നോക്കുന്നു..."],
    "hi": ["एक पल रुकिए...", "अभी देखता हूँ..."],
    "ta": ["ஒரு நிமிடம்...", "சற்று நிறுத்துங்கள்..."],
}

# ── SALES AGENT FILLERS ───────────────────────────────────────────────────────
SALES_FILLERS: dict[str, dict[str, list[str]]] = {
    "search_products": {
        "en": ["Let me search our catalog for you...", "Give me a moment to find those products...", "Searching our inventory now..."],
        "ml": ["ഞങ്ങളുടെ കാറ്റലോഗ് തിരയുന്നു...", "ഒരു നിമിഷം, ഉൽപ്പന്നങ്ങൾ കണ്ടെത്തുന്നു..."],
        "hi": ["कैटलॉग में खोज रहा हूँ..."],
        "ta": ["எங்கள் பட்டியலில் தேடுகிறேன்..."],
    },
    "get_product_details": {
        "en": ["Let me get the details for that product...", "Pulling up the specifications now..."],
        "ml": ["ആ ഉൽപ്പന്നത്തിന്റെ വിശദാംശങ്ങൾ എടുക്കുന്നു..."],
        "hi": ["उत्पाद की जानकारी देख रहा हूँ..."],
        "ta": ["தயாரிப்பு விவரங்களைப் பார்க்கிறேன்..."],
    },
    "check_stock": {
        "en": ["Checking stock levels for you...", "Let me verify availability..."],
        "ml": ["സ്റ്റോക്ക് ലഭ്യത പരിശോധിക്കുന്നു..."],
        "hi": ["स्टॉक की जाँच कर रहा हूँ..."],
        "ta": ["இருப்பு நிலையை சரிபார்க்கிறேன்..."],
    },
    "create_order": {
        "en": ["Processing your order now...", "Let me set that up for you..."],
        "ml": ["നിങ്ങളുടെ ഓർഡർ പ്രോസസ്സ് ചെയ്യുന്നു..."],
        "hi": ["आपका ऑर्डर तैयार कर रहा हूँ..."],
        "ta": ["உங்கள் ஆர்டரை செயல்படுத்துகிறேன்..."],
    },
    "calculate_discount": {
        "en": ["Let me calculate the best price for you...", "Checking available discounts..."],
        "ml": ["ഏറ്റവും മികച്ച വില കണക്കാക്കുന്നു..."],
        "hi": ["छूट की गणना कर रहा हूँ..."],
        "ta": ["தள்ளுபடியை கணக்கிடுகிறேன்..."],
    },
}

SALES_DEFAULT_FILLERS: dict[str, list[str]] = {
    "en": ["One moment...", "Let me check...", "Just a second..."],
    "ml": ["ഒരു നിമിഷം...", "നോക്കട്ടെ..."],
    "hi": ["एक पल...", "देखता हूँ..."],
    "ta": ["ஒரு நிமிடம்...", "பார்க்கிறேன்..."],
}

# ── SUPPORT AGENT FILLERS ─────────────────────────────────────────────────────
SUPPORT_FILLERS: dict[str, dict[str, list[str]]] = {
    "get_order_details":          {"en": ["Let me check your order details...", "Give me a moment to pull up your order..."]},
    "check_refund_eligibility":   {"en": ["Let me check if you're eligible for a refund...", "I'll quickly verify that for you..."]},
    "initiate_return_pickup":     {"en": ["Let me arrange a pickup for you...", "I'll schedule the return pickup now..."]},
    "initiate_refund":            {"en": ["Processing your refund now...", "Let me initiate that refund for you..."]},
    "product_support_rag":        {"en": ["Let me check that for you...", "Give me a moment to find a solution..."]},
    "create_support_ticket":      {"en": ["Let me raise a support ticket for this...", "I'll log this issue for you..."]},
    "escalate_to_human":          {"en": ["Let me connect you to a specialist...", "I'll transfer you to our support team..."]},
}

SUPPORT_DEFAULT_FILLERS = {"en": ["One moment please...", "Let me check that..."]}


# ── Filler getters ────────────────────────────────────────────────────────────
def get_resort_filler(tool_name: str, lang: str) -> str:
    tool_map = RESORT_FILLERS.get(tool_name, {})
    options = tool_map.get(lang) or tool_map.get("en") or RESORT_DEFAULT_FILLERS.get(lang) or RESORT_DEFAULT_FILLERS["en"]
    return random.choice(options)

def get_sales_filler(tool_name: str, lang: str) -> str:
    tool_map = SALES_FILLERS.get(tool_name, {})
    options = tool_map.get(lang) or tool_map.get("en") or SALES_DEFAULT_FILLERS.get(lang) or SALES_DEFAULT_FILLERS["en"]
    return random.choice(options)

def get_support_filler(tool_name: str, lang: str) -> str:
    tool_map = SUPPORT_FILLERS.get(tool_name, {})
    options = tool_map.get(lang) or tool_map.get("en") or SUPPORT_DEFAULT_FILLERS.get(lang) or SUPPORT_DEFAULT_FILLERS["en"]
    return random.choice(options)


# ── Initialise TTS ────────────────────────────────────────────────────────────
init_tts(CARTESIA_API_KEY)

_llm_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="llm")




# ── Text helpers ──────────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[•\*#`]', '', text)
    return re.sub(r'\n', ' ', text).strip()

_SENT_RE = re.compile(r'(?<=[.!?,;])\s+')


# ── Graph state repair ────────────────────────────────────────────────────────
def fix_broken_graph_state(graph_instance, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = graph_instance.get_state(config)
        messages = state.values.get("messages", [])
        if not messages:
            return
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tool_call in last.tool_calls:
                graph_instance.update_state(config, {
                    "messages": [ToolMessage(content="Request was interrupted by user.", tool_call_id=tool_call["id"])]
                })
            logging.info(f"[GRAPH] Repaired state for thread {thread_id}")
    except Exception as e:
        logging.error(f"[GRAPH] state fix failed: {e}")


# ── LLM sentence streamer ─────────────────────────────────────────────────────
def stream_graph_sentences(graph_instance, transcript, thread_id, sentence_q, cancel_flag, lang, filler_getter_fn):
    config = {"configurable": {"thread_id": thread_id}}
    buffer = ""
    filler_sent = False

    try:
        for chunk, _ in graph_instance.stream({"messages": transcript}, config, stream_mode="messages"):
            if cancel_flag.is_set():
                break
            if hasattr(chunk, "content"):
                if type(chunk).__name__ in ("AIMessageChunk", "AIMessage"):
                    if getattr(chunk, "tool_calls", None):
                        if not filler_sent:
                            tool_name = chunk.tool_calls[0].get("name", "")
                            sentence_q.put(filler_getter_fn(tool_name, lang))
                            filler_sent = True
                        continue_processing = False
                    else:
                        continue_processing = True

                    if continue_processing and chunk.content:
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
            if r:
                sentence_q.put(r)
    finally:
        sentence_q.put(None)


# ── STT factory ───────────────────────────────────────────────────────────────
def make_stt(stt_choice: str, on_transcript, on_interim, on_barge_in):
    """Return the right STTSession based on user's choice."""
    if stt_choice == "sarvam":
        logging.info("[STT] Using Sarvam AI")
        return SarvamSTT(
            api_key=SARVAM_API_KEY,
            on_transcript=on_transcript,
            on_interim=on_interim,
            on_barge_in=on_barge_in,
        )
    else:
        logging.info("[STT] Using Soniox (default)")
        return SonioxSTT(
            api_key=SONIOX_API_KEY,
            on_transcript=on_transcript,
            on_interim=on_interim,
            on_barge_in=on_barge_in,
        )


# ── Generic WebSocket handler ─────────────────────────────────────────────────
async def websocket_handler(
    browser_ws: WebSocket,
    graph_instance,
    thread_prefix: str,
    greeting_audio: bytes,
    greeting_message: str,
    filler_getter_fn,
    voice_id: str,
    stt_choice: str = "soniox",   # ← new param
):
    await browser_ws.accept()

    client_ip = browser_ws.client.host

    if ip_connections[client_ip] >= MAX_CONCURRENT_PER_IP:
        await browser_ws.close(code=1008)
        return

    ip_connections[client_ip] += 1

    thread_id    = f"{thread_prefix}-{id(browser_ws)}"
    bot_speaking = asyncio.Event()
    cancel_event = asyncio.Event()
    audio_queue  = asyncio.Queue(maxsize=50)
    current_task = [None]

    if greeting_message:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            graph_instance.update_state(config, {"messages": [AIMessage(content=greeting_message)]})
            logging.info(f"[{thread_prefix.upper()}] Initialized with AI greeting")
        except Exception as e:
            logging.error(f"Failed to initialize: {e}")

    async def send_json(data: dict):
        try:
            await browser_ws.send_text(json.dumps(data))
        except Exception:
            pass

    async def send_bytes_ws(data: bytes):
        try:
            await browser_ws.send_bytes(data)
        except Exception:
            pass

    async def cancel_current():
        cancel_event.set()
        bot_speaking.clear()
        task = current_task[0]
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.get_running_loop().run_in_executor(
            None, fix_broken_graph_state, graph_instance, thread_id
        )

    async def handle_barge_in():
        await cancel_current()
        await send_json({"type": "barge_in"})
        logging.info("[BARGE-IN] User interrupted the bot.")

    async def custom_speech(message_audio: bytes):
        sid = str(uuid.uuid4())
        bot_speaking.set()
        await send_json({"type": "audio_start", "session_id": sid})
        chunk_size = 4800
        for i in range(0, len(message_audio), chunk_size):
            await send_bytes_ws(message_audio[i:i + chunk_size])
        await send_json({"type": "audio_end", "session_id": sid})
        bot_speaking.clear()

    async def handle_transcript(transcript: str):
        if is_ip_limit_reached(client_ip):
            await custom_speech(LIMIT_AUDIO)
            return

        session_requests[thread_id] += 1
        if session_requests[thread_id] > MAX_REQUESTS_PER_SESSION:
            await custom_speech(LIMIT_AUDIO)
            await asyncio.sleep(3)
            await browser_ws.close(code=1008)
            return

        if current_task[0] and not current_task[0].done():
            await cancel_current()
            await send_json({"type": "barge_in"})

        cancel_event.clear()
        current_task[0] = asyncio.create_task(process_transcript(transcript))

    async def process_transcript(transcript: str):
        sid  = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        cancel_event.clear()

        lang = detect_language(transcript)

        await send_json({"type": "transcript", "text": transcript})
        await send_json({"type": "thinking"})

        llm_cancel = threading.Event()
        sentence_q = _q.Queue()

        await send_json({"type": "audio_start", "session_id": sid})
        bot_speaking.set()

        llm_future = loop.run_in_executor(
            _llm_executor, stream_graph_sentences,
            graph_instance, transcript, thread_id,
            sentence_q, llm_cancel, lang, filler_getter_fn,
        )

        tts_streamer = TTSSentenceStreamer(on_audio_chunk=send_bytes_ws, voice_id=voice_id)

        try:
            _, full_parts = await asyncio.gather(
                llm_future,
                tts_streamer.stream(sentence_q, cancel_event),
            )
        except asyncio.CancelledError:
            llm_cancel.set()
            raise
        finally:
            llm_cancel.set()
            bot_speaking.clear()

        if not cancel_event.is_set():
            full_response = " ".join(full_parts)
            await send_json({"type": "response", "text": full_response})
            await send_json({"type": "audio_end", "session_id": sid})

    async def on_interim(text):
        await send_json({"type": "interim", "text": text})

    # ── Build STT based on user's choice ──────────────────────────────────────
    stt = make_stt(stt_choice, handle_transcript, on_interim, handle_barge_in)

    # Send greeting after STT initialized
    await custom_speech(greeting_audio)

    async def stt_loop():
        await stt.run(audio_queue, bot_speaking)

    async def receive_from_browser():
        while True:
            msg = await browser_ws.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            if msg.get("bytes"):
                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await audio_queue.put(msg["bytes"])

    try:
        await asyncio.gather(receive_from_browser(), stt_loop())
    except WebSocketDisconnect:
        pass
    finally:
        if current_task[0] and not current_task[0].done():
            current_task[0].cancel()
        await audio_queue.put(None)
        ip_connections[client_ip] -= 1
        session_requests.pop(thread_id, None)


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/")
async def index():
    return HTMLResponse(HTML)


@router.websocket("/ws/resort")
async def websocket_resort_endpoint(browser_ws: WebSocket, stt: str = Query(default="soniox")):
    await websocket_handler(
        browser_ws=browser_ws,
        graph_instance=resort_graph,
        thread_prefix="resort",
        greeting_audio=RESORT_GREETING_AUDIO,
        filler_getter_fn=get_resort_filler,
        greeting_message="Hi, this is Acsa from Paradise Resort Cherai. How can I help you?",
        voice_id=RESORT_VOICE_ID,
        stt_choice=stt,
    )


@router.websocket("/ws/sales")
async def websocket_sales_endpoint(browser_ws: WebSocket, stt: str = Query(default="soniox")):
    await websocket_handler(
        browser_ws=browser_ws,
        graph_instance=sales_graph,
        thread_prefix="sales",
        greeting_audio=SALES_GREETING_AUDIO,
        filler_getter_fn=get_sales_filler,
        greeting_message="Hey, I'm Alex. I help businesses handle customer enquiries instantly and convert more leads. Just curious — how are you currently managing your incoming messages or calls?",
        voice_id=SALES_VOICE_ID,
        stt_choice=stt,
    )


@router.websocket("/ws/support")
async def websocket_support_endpoint(browser_ws: WebSocket, stt: str = Query(default="soniox")):
    await websocket_handler(
        browser_ws=browser_ws,
        graph_instance=support_graph,
        thread_prefix="support",
        greeting_audio=SUPPORT_GREETING_AUDIO,
        filler_getter_fn=get_support_filler,
        greeting_message="Hey, this is Riya from support. What can I help you with today?",
        voice_id=SUPPORT_VOICE_ID,
        stt_choice=stt,
    )