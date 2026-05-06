import os
import asyncio
import logging
import threading
from collections import OrderedDict
from queue import Queue, Full
import hmac
import hashlib
import aiosqlite
from dotenv import load_dotenv
from fastapi import APIRouter, Request, HTTPException,Header
from fastapi.responses import JSONResponse, PlainTextResponse
from .handler import process_whatsapp_message
from .state import seen, mark  # kept for backward compat if needed


# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
APP_SECRET = os.getenv("APP_SECRET")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

router = APIRouter()


# -------------------------------------------------------------------
# In-Memory Deduplication (replaces SQLite dedup in hot path)
# -------------------------------------------------------------------

_DEDUP_MAX = 10_000
_seen_cache: OrderedDict = OrderedDict()
_seen_lock = threading.Lock()


def _seen_fast(message_id: str) -> bool:
    """Returns True if already seen (duplicate). Thread-safe, O(1)."""
    with _seen_lock:
        if message_id in _seen_cache:
            return True
        _seen_cache[message_id] = True
        if len(_seen_cache) > _DEDUP_MAX:
            _seen_cache.popitem(last=False)  # evict oldest
        return False


# -------------------------------------------------------------------
# Background Queue & Workers
# -------------------------------------------------------------------

MESSAGE_QUEUE_SIZE = 1_000
WORKER_COUNT = 8  # increased — safe for I/O-bound work

message_queue: Queue = Queue(maxsize=MESSAGE_QUEUE_SIZE)


def background_worker():
    while True:
        sender, text = message_queue.get()
        try:
            print("hi")
            process_whatsapp_message(sender, text)
            
        except Exception:
            logging.exception("Background worker failed")
        finally:
            message_queue.task_done()


def start_workers():
    for _ in range(WORKER_COUNT):
        t = threading.Thread(target=background_worker, daemon=True)
        t.start()
    logging.info("Started %d background workers", WORKER_COUNT)


# -------------------------------------------------------------------
# Utility: WhatsApp Payload Parsing
# -------------------------------------------------------------------

def extract_whatsapp_message(payload: dict):
    """
    Returns (sender, text, message_id) or None.
    Handles both text and non-text message types gracefully.
    """
    try:
        entry = payload["entry"][0]
        value = entry["changes"][0]["value"]

        messages = value.get("messages")
        if not messages:
            return None

        msg = messages[0]
        sender = msg.get("from")
        message_id = msg.get("id")

        if not sender or not message_id:
            logging.warning("Missing sender or message_id in payload")
            return None

        if msg.get("type") == "text":
            text = msg["text"]["body"]
        else:
            text = str(msg)  # fallback: serialize non-text messages

        return sender, text, message_id

    except (KeyError, IndexError, TypeError) as e:
        logging.debug("Could not parse WhatsApp payload: %s", e)
        return None


# -------------------------------------------------------------------
# Webhook Verification
# -------------------------------------------------------------------

@router.get("/webhook")
async def verify_webhook(request: Request):
    hub_mode = request.query_params.get("hub.mode")
    hub_verify_token = request.query_params.get("hub.verify_token")
    hub_challenge = request.query_params.get("hub.challenge")

    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Verification failed")


# -------------------------------------------------------------------
# Webhook Receiver  (hot path — kept as lean as possible)
# -------------------------------------------------------------------

# @router.post("/webhook")
# async def webhook_receiver(request: Request):
#     payload = await request.json()

#     extracted = extract_whatsapp_message(payload)
#     if not extracted:
#         return {"status": "ignored"}

#     sender, text, message_id = extracted

#     # Fast in-memory dedup — no disk I/O on the hot path
#     if _seen_fast(message_id):
#         logging.info("Duplicate WhatsApp message ignored: %s", message_id)
#         return {"status": "ok"}

#     try:
#         message_queue.put_nowait((sender, text))
#     except Full:
#         logging.warning("Message queue full — dropping message from %s", sender)
#         return {"status": "busy"}

#     return {"status": "ok"}

@router.post("/webhook")
async def webhook_receiver(
    request: Request, x_hub_signature_256: str = Header(None)
):
    # 1️⃣ Read the raw body (bytes) — required for HMAC verification
    raw_body = await request.body()

    # 2️⃣ Compute HMAC using your APP_SECRET
    expected_signature = "sha256=" + hmac.new(
        APP_SECRET.encode(), raw_body,  hashlib.sha256
    ).hexdigest()

    # 3️⃣ Compare the signature securely
    if not hmac.compare_digest(expected_signature, x_hub_signature_256 or ""):
        logging.warning("Invalid signature attempt from %s", request.client.host)
        raise HTTPException(status_code=403, detail="Invalid signature")

    # 4️⃣ Parse the JSON body after verifying signature
    payload = await request.json()

    # 5️⃣ Extract WhatsApp message
    extracted = extract_whatsapp_message(payload)
    if not extracted:
        return {"status": "ignored"}

    sender, text, message_id = extracted

    # 6️⃣ Deduplicate messages in memory
    if _seen_fast(message_id):
        logging.info("Duplicate WhatsApp message ignored: %s", message_id)
        return {"status": "ok"}

    # 7️⃣ Push to background queue
    try:
        message_queue.put_nowait((sender, text))
    except Full:
        logging.warning("Message queue full — dropping message from %s", sender)
        return {"status": "busy"}

    # 8️⃣ Success
    return {"status": "ok"}


# -------------------------------------------------------------------
# Booking Sync Endpoint  (async SQLite — no event-loop blocking)
# -------------------------------------------------------------------

DB_PATH = "Databases/bookings.db"


@router.post("/webhook/sync-booking")
async def sync_booking(request: Request):
    data = await request.json()

    if not data or "phone" not in data:
        raise HTTPException(status_code=400, detail="Invalid payload")

    phone = str(data.get("phone", "")).strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone cannot be empty")

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                INSERT INTO bookings (name, checkin, checkout, phone, status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(phone) DO UPDATE SET
                    name     = excluded.name,
                    checkin  = excluded.checkin,
                    checkout = excluded.checkout,
                    status   = excluded.status
                """,
                (
                    data.get("name"),
                    data.get("checkin"),
                    data.get("checkout"),
                    phone,
                    data.get("status") or "active",
                ),
            )
            await db.commit()

        return {"ok": True}

    except Exception:
        logging.exception("Booking sync failed for phone=%s", phone)
        raise HTTPException(status_code=500, detail="Internal error")