"""
conversations_router.py
-----------------------
Add to your FastAPI app:

    from conversations_router import router as conversations_router
    app.include_router(conversations_router)

Also add to your .env:
    ENCRYPTION_KEY=<your existing key>

Exposes:
    GET /conversations          -> list of all conversations
    GET /conversations/{id}     -> single conversation with full messages
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

router = APIRouter(prefix="/conversations", tags=["conversations"])

# ── same paths as your existing setup_memory() ──────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
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

# ── helpers ──────────────────────────────────────────────────────────────────

def decrypt_value(serde: EncryptedSerializer, type_: str, raw: bytes):
    """Decrypt + deserialize a single writes.value blob."""
    try:
        return serde.loads_typed((type_, raw))
    except Exception:
        return None


def thread_id_to_meta(thread_id: str) -> dict:
    """
    Your thread_ids look like: resort-2758183895440
    The suffix after the first dash is treated as the phone number.
    """
    parts = thread_id.split("-", 1)
    phone = parts[1] if len(parts) == 2 else thread_id
    name  = f"+{phone}" if phone.isdigit() else phone
    return {"phone": phone, "display_name": name}


def status_from_messages(messages: list) -> str:
    """Simple heuristic — override with your own logic."""
    count = len(messages)
    if count >= 8:
        return "hot"
    if count >= 4:
        return "warm"
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


# ── data extraction ───────────────────────────────────────────────────────────
def extract_messages_for_thread(
    conn: sqlite3.Connection,
    serde: EncryptedSerializer,
    thread_id: str,
) -> list[dict]:
    # Get only the LAST checkpoint — it has the full conversation
    row = conn.execute(
        """SELECT type, checkpoint FROM checkpoints 
           WHERE thread_id = ? 
           ORDER BY checkpoint_id DESC 
           LIMIT 1""",
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

    raw_messages = obj["channel_values"].get("messages", [])
    result = []

    for m in raw_messages:
        role = type(m).__name__
        content = getattr(m, "content", "")

        # Skip tool messages
        if "Tool" in role:
            continue

        # Skip AI messages that are tool calls (empty content or has tool_calls)
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
            "id": getattr(m, "id", f"{thread_id}-{len(result)}"),
            "from": "guest" if "Human" in role else "agent",
            "text": content,
            "time": "",
        })

    return result

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


# ── Pydantic models ───────────────────────────────────────────────────────────

class Conversation(BaseModel):
    id:       str
    name:     str
    initials: str
    source:   Literal["whatsapp", "voice"]
    status:   Literal["hot", "warm", "cold"]
    preview:  str
    time:     str
    color:    str
    messages: list[dict]


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[Conversation])
def get_conversations():
    conn  = get_conn()
    serde = get_serde()
    threads = list_threads(conn)
    result  = []

    for i, thread_id in enumerate(threads):
        msgs    = extract_messages_for_thread(conn, serde, thread_id)
        meta    = thread_id_to_meta(thread_id)
        preview = msgs[-1]["text"][:80] if msgs else ""
        status  = status_from_messages(msgs)
        name    = meta["display_name"]

        # If phone exists in handoffs, use that
        row = conn.execute(
            "SELECT phone FROM handoffs WHERE id = ?", (thread_id,)
        ).fetchone()
        if row:
            name = row["phone"]

        result.append(Conversation(
            id       = thread_id,
            name     = name,
            initials = initials_from_name(name),
            source   = "whatsapp",
            status   = status,
            preview  = preview,
            time     = "",
            color    = color_for_index(i),
            messages = msgs,
        ))

    conn.close()
    return result


@router.get("/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str):
    conn  = get_conn()
    serde = get_serde()
    threads = list_threads(conn)

    if conversation_id not in threads:
        raise HTTPException(status_code=404, detail="Conversation not found")

    i       = threads.index(conversation_id)
    msgs    = extract_messages_for_thread(conn, serde, conversation_id)
    meta    = thread_id_to_meta(conversation_id)
    preview = msgs[-1]["text"][:80] if msgs else ""
    status  = status_from_messages(msgs)
    name    = meta["display_name"]

    row = conn.execute(
        "SELECT phone FROM handoffs WHERE id = ?", (conversation_id,)
    ).fetchone()
    if row:
        name = row["phone"]

    conn.close()
    return Conversation(
        id       = conversation_id,
        name     = name,
        initials = initials_from_name(name),
        source   = "whatsapp",
        status   = status,
        preview  = preview,
        time     = "",
        color    = color_for_index(i),
        messages = msgs,
    )

@router.get("/debug")
def debug_conversations():
    conn = get_conn()
    
    # Check if DB file exists
    db_exists = DB_PATH.exists()
    
    # Check tables
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    
    # Check raw thread count
    thread_count = conn.execute(
        "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
    ).fetchone()[0]
    
    # Sample a raw thread_id
    sample = conn.execute(
        "SELECT thread_id FROM checkpoints LIMIT 1"
    ).fetchone()
    
    conn.close()
    return {
        "db_exists": db_exists,
        "db_path": str(DB_PATH),
        "tables": [t[0] for t in tables],
        "thread_count": thread_count,
        "sample_thread_id": sample[0] if sample else None,
    }
@router.get("/debug/schema")
def debug_schema():
    conn = get_conn()
    
    thread_id = conn.execute(
        "SELECT thread_id FROM checkpoints LIMIT 1"
    ).fetchone()[0]
    
    channels = conn.execute(
        "SELECT DISTINCT channel FROM writes WHERE thread_id = ?",
        (thread_id,)
    ).fetchall()
    
    cols = conn.execute("PRAGMA table_info(checkpoints)").fetchall()
    
    conn.close()
    return {
        "thread_id": thread_id,
        "write_channels": [r[0] for r in channels],
        "checkpoint_columns": [r[0] for r in cols],
    }
@router.get("/debug/meta")
def debug_meta():
    conn  = get_conn()
    serde = get_serde()

    thread_id = conn.execute(
        "SELECT thread_id FROM checkpoints LIMIT 1"
    ).fetchone()[0]

    rows = conn.execute(
        "SELECT checkpoint_id, type, checkpoint, metadata FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id",
        (thread_id,)
    ).fetchall()

    result = []
    for row in rows:
        try:
            obj = serde.loads_typed((row[1], row[2]))
            msgs = []
            if isinstance(obj, dict) and "channel_values" in obj:
                cv = obj["channel_values"]
                if "messages" in cv:
                    for m in cv["messages"]:
                        msgs.append({
                            "type": type(m).__name__,
                            "content": str(getattr(m, "content", m))[:80]
                        })
            result.append({
                "checkpoint_id": row[0],
                "metadata": str(row[3])[:200],
                "messages": msgs
            })
        except Exception as e:
            result.append({"checkpoint_id": row[0], "error": str(e)})

    conn.close()
    return result