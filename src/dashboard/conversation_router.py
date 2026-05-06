"""
conversations_router.py
-----------------------
Add to your FastAPI app:

    from conversation_router import router as conversations_router
    app.include_router(conversations_router)

GET /conversations          → fast list from conversation_meta (no decryption)
GET /conversations/{id}     → full messages decrypted from checkpoints
"""

import sqlite3
import base64
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from dotenv import load_dotenv
load_dotenv()
from dashboard.feature_flags import set_flag, _flags
router = APIRouter(prefix="/conversations", tags=["conversations"])

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "Databases" / "petesinn.sqlite"


def get_serde() -> EncryptedSerializer:
    key_b64 = os.getenv("ENCRYPTION_KEY")
    if not key_b64:
        raise RuntimeError("ENCRYPTION_KEY not set in environment")
    key = base64.b64decode(key_b64)
    return EncryptedSerializer.from_pycryptodome_aes(key=key)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Helpers ───────────────────────────────────────────────────────────────────

VOICE_PREFIXES = ("resort", "sales", "support")


def source_from_thread_id(thread_id: str) -> str:
    return "voice" if thread_id.split("-", 1)[0] in VOICE_PREFIXES else "whatsapp"


def thread_id_to_meta(thread_id: str) -> dict:
    parts  = thread_id.split("-", 1)
    prefix = parts[0]
    suffix = parts[1] if len(parts) == 2 else thread_id
    if prefix in VOICE_PREFIXES:
        return {"display_name": suffix}
    name = f"+{thread_id}" if thread_id.isdigit() else thread_id
    return {"display_name": name}


def status_from_count(count: int) -> str:
    if count >= 8: return "hot"
    if count >= 4: return "warm"
    return "cold"

def initials_from_name(name: str) -> str:
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    return name[:2].upper()


AVATAR_COLORS = [
    "#EDE9FE", "#DBEAFE", "#D1FAE5", "#FEE2E2",
    "#FEF3C7", "#FCE7F3", "#E0F2FE", "#F0FDF4",
]

def color_for_index(i: int) -> str:
    return AVATAR_COLORS[i % len(AVATAR_COLORS)]


def list_threads(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT   thread_id
        FROM     checkpoints
        GROUP BY thread_id
        ORDER BY MAX(checkpoint_id) DESC
        """
    ).fetchall()
    return [r["thread_id"] for r in rows]


def extract_messages_for_thread(
    conn: sqlite3.Connection,
    serde: EncryptedSerializer,
    thread_id: str,
) -> list[dict]:
    row = conn.execute(
        """
        SELECT type, checkpoint FROM checkpoints
        WHERE  thread_id = ?
        ORDER  BY checkpoint_id DESC
        LIMIT  1
        """,
        (thread_id,),
    ).fetchone()

    if not row:
        return []

    try:
        obj = serde.loads_typed((row[0], row[1]))
    except Exception:
        return []

    if not isinstance(obj, dict) or "channel_values" not in obj:
        return []

    result = []
    for m in obj["channel_values"].get("messages", []):
        role    = type(m).__name__
        content = getattr(m, "content", "")

        if "Tool" in role:
            continue
        if "AI" in role:
            if getattr(m, "tool_calls", []):
                continue
            if not content or (isinstance(content, str) and not content.strip()):
                continue

        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            ).strip()

        if not content.strip():
            continue

        result.append({
            "id":   getattr(m, "id", f"{thread_id}-{len(result)}"),
            "from": "guest" if "Human" in role else "agent",
            "text": content,
            "time": "",
        })

    return result


# ── Pydantic models ───────────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    """Returned by GET /conversations — no messages, reads from meta table."""
    id:       str
    name:     str
    initials: str
    source:   Literal["whatsapp", "voice"]
    status:   Literal["hot", "warm", "cold"]
    preview:  str
    time:     str
    color:    str


class Conversation(ConversationSummary):
    """Returned by GET /conversations/{id} — full messages from checkpoints."""
    messages: list[dict]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ConversationSummary])
def get_conversations():
    """
    Fast list — reads only from conversation_meta.
    No checkpoint decryption, no message extraction.
    """
    conn = get_conn()

    rows = conn.execute(
        "SELECT thread_id, name, source, last_message, last_time, message_count FROM conversation_meta ORDER BY updated_at DESC"
    ).fetchall()

    conn.close()

    return [
        ConversationSummary(
            id       = row["thread_id"],
            name     = row["name"] or row["thread_id"],
            initials = initials_from_name(row["name"] or row["thread_id"]),
            source   = row["source"],
            status = status_from_count(row["message_count"] or 0),          # lightweight default; enrich if needed
            preview  = (row["last_message"] or "")[:80],
            time     = row["last_time"] or "",
            color    = color_for_index(i),
        )
        for i, row in enumerate(rows)
    ]


@router.get("/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    """
    Full detail — header info from meta, messages decrypted from checkpoints.
    """
    conn  = get_conn()
    serde = get_serde()

    meta_row = conn.execute(
        "SELECT * FROM conversation_meta WHERE thread_id = ?",
        (conversation_id,),
    ).fetchone()

    if not meta_row:
        # Thread exists in checkpoints but not yet in meta (edge case)
        if conversation_id not in list_threads(conn):
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        name     = thread_id_to_meta(conversation_id)["display_name"]
        source   = source_from_thread_id(conversation_id)
        time_str = ""
        color_i  = 0
    else:
        name     = meta_row["name"] or conversation_id
        source   = meta_row["source"]
        time_str = meta_row["last_time"] or ""
        # Derive color index from position in ordered meta table
        color_i  = conn.execute(
            "SELECT COUNT(*) FROM conversation_meta WHERE updated_at > ?",
            (meta_row["updated_at"],),
        ).fetchone()[0]

    msgs   = extract_messages_for_thread(conn, serde, conversation_id)
    status = status_from_count(len(msgs))
    conn.close()

    return Conversation(
        id       = conversation_id,
        name     = name,
        initials = initials_from_name(name),
        source   = source,
        status   = status,
        preview  = msgs[-1]["text"][:80] if msgs else "",
        time     = time_str,
        color    = color_for_index(color_i),
        messages = msgs,
    )


# ── Debug routes ──────────────────────────────────────────────────────────────

@router.get("/debug/meta-stats")
def debug_meta_stats():
    conn   = get_conn()
    total  = conn.execute("SELECT COUNT(*) FROM conversation_meta").fetchone()[0]
    latest = conn.execute(
        "SELECT thread_id, last_time, source FROM conversation_meta ORDER BY updated_at DESC LIMIT 5"
    ).fetchall()
    conn.close()
    return {"meta_row_count": total, "latest_5": [dict(r) for r in latest]}


@router.get("/debug")
def debug_conversations():
    conn         = get_conn()
    tables       = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    thread_count = conn.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints").fetchone()[0]
    sample       = conn.execute("SELECT thread_id FROM checkpoints LIMIT 1").fetchone()
    all_threads  = list_threads(conn)
    conn.close()
    return {
        "db_path":          str(DB_PATH),
        "tables":           [t[0] for t in tables],
        "thread_count":     thread_count,
        "voice_count":      sum(1 for t in all_threads if source_from_thread_id(t) == "voice"),
        "whatsapp_count":   sum(1 for t in all_threads if source_from_thread_id(t) == "whatsapp"),
        "sample_thread_id": sample[0] if sample else None,
    }

@router.get("/flags")
def get_flags():
    return _flags

@router.post("/flags/{key}")
def toggle_flag(key: str, enabled: bool):
    conn = get_conn()
    set_flag(conn, key, enabled)   # ← add conn
    conn.close()
    return {"key": key, "enabled": enabled}