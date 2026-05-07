import hashlib
import hmac
import logging
import os
import threading
import time
from queue import Full, Queue

import aiosqlite
from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .handler import process_whatsapp_message

try:
    import redis
except ImportError:
    redis = None

load_dotenv()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
APP_SECRET = os.getenv("APP_SECRET")
REDIS_URL = os.getenv("REDIS_URL")

DEDUP_TTL_SECONDS = int(os.getenv("WHATSAPP_DEDUP_TTL_SECONDS", str(24 * 60 * 60)))
MESSAGE_QUEUE_SIZE = int(os.getenv("WHATSAPP_QUEUE_SIZE", "1000"))
WORKER_COUNT = int(os.getenv("WHATSAPP_WORKER_COUNT", "8"))
BOOKINGS_DB_PATH = "Databases/bookings.db"

log = logging.getLogger(__name__)
router = APIRouter()
message_queue: Queue = Queue(maxsize=MESSAGE_QUEUE_SIZE)

_redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True) if redis and REDIS_URL else None
_seen_cache: dict[str, float] = {}
_seen_lock = threading.Lock()


def mark_message_if_new(message_id: str) -> bool:
    """
    Return True only for the first delivery of a WhatsApp message id.
    Redis is atomic across processes; the in-memory fallback is for local dev.
    """
    if _redis_client:
        return bool(
            _redis_client.set(
                f"wa:dedupe:{message_id}",
                "1",
                nx=True,
                ex=DEDUP_TTL_SECONDS,
            )
        )

    now = time.time()
    with _seen_lock:
        expired = [key for key, expires_at in _seen_cache.items() if expires_at <= now]
        for key in expired:
            _seen_cache.pop(key, None)

        if message_id in _seen_cache:
            return False

        _seen_cache[message_id] = now + DEDUP_TTL_SECONDS
        return True


def forget_message(message_id: str) -> None:
    """Allow Meta to retry later if the message could not be queued."""
    if _redis_client:
        _redis_client.delete(f"wa:dedupe:{message_id}")
        return

    with _seen_lock:
        _seen_cache.pop(message_id, None)


def verify_signature(raw_body: bytes, signature: str | None) -> None:
    if not APP_SECRET:
        log.error("APP_SECRET is not configured")
        raise HTTPException(status_code=500, detail="Webhook signing is not configured")

    expected = "sha256=" + hmac.new(
        APP_SECRET.encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature or ""):
        log.warning("Invalid WhatsApp webhook signature")
        raise HTTPException(status_code=403, detail="Invalid signature")


def extract_whatsapp_message(payload: dict):
    """
    Return (sender, text, message_id) for inbound user messages.
    Status callbacks and unsupported webhook shapes return None.
    """
    try:
        value = payload["entry"][0]["changes"][0]["value"]
        messages = value.get("messages")
        if not messages:
            return None

        msg = messages[0]
        sender = msg.get("from")
        message_id = msg.get("id")
        if not sender or not message_id:
            log.warning("Missing sender or message_id in WhatsApp payload")
            return None

        if msg.get("type") == "text":
            text = msg["text"]["body"]
        else:
            text = str(msg)

        return sender, text, message_id

    except (KeyError, IndexError, TypeError) as exc:
        log.debug("Could not parse WhatsApp payload: %s", exc)
        return None


def background_worker() -> None:
    while True:
        sender, text = message_queue.get()
        try:
            process_whatsapp_message(sender, text)
        except Exception:
            log.exception("WhatsApp background worker failed")
        finally:
            message_queue.task_done()


def start_workers() -> None:
    for _ in range(WORKER_COUNT):
        thread = threading.Thread(target=background_worker, daemon=True)
        thread.start()
    log.info("Started %d WhatsApp background workers", WORKER_COUNT)


@router.get("/webhook")
async def verify_webhook(request: Request):
    hub_mode = request.query_params.get("hub.mode")
    hub_verify_token = request.query_params.get("hub.verify_token")
    hub_challenge = request.query_params.get("hub.challenge")

    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook")
async def webhook_receiver(
    request: Request,
    x_hub_signature_256: str = Header(None),
):
    raw_body = await request.body()
    verify_signature(raw_body, x_hub_signature_256)

    extracted = extract_whatsapp_message(await request.json())
    if not extracted:
        return {"status": "ignored"}

    sender, text, message_id = extracted

    if not mark_message_if_new(message_id):
        log.info("Duplicate WhatsApp message ignored: %s", message_id)
        return {"status": "ok"}

    try:
        message_queue.put_nowait((sender, text))
    except Full:
        forget_message(message_id)
        log.warning("WhatsApp queue full; asking Meta to retry message from %s", sender)
        raise HTTPException(status_code=503, detail="Message queue is full")

    return {"status": "ok"}


@router.post("/webhook/sync-booking")
async def sync_booking(request: Request):
    data = await request.json()
    phone = str((data or {}).get("phone", "")).strip()

    if not data or not phone:
        raise HTTPException(status_code=400, detail="Invalid payload")

    try:
        async with aiosqlite.connect(BOOKINGS_DB_PATH) as db:
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
        log.exception("Booking sync failed for phone=%s", phone)
        raise HTTPException(status_code=500, detail="Internal error")
