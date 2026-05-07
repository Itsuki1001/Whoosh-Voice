import logging
import os
import re
import sqlite3
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_openai import ChatOpenAI

from graph.graph_whatsapp import graph
from .client import send_text

log = logging.getLogger(__name__)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "Databases" / "petesinn.sqlite"
DEFAULT_HUMAN_PHONE = os.getenv("HUMAN_HANDOFF_PHONE", "918606842144")
_faq_classifier = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("FAQ_CLASSIFIER_MODEL", "gpt-4.1-mini"),
    timeout=15,
)


def _connect():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def save_handoff(handoff_id: str, phone: str, message: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO handoffs (id, phone, message) VALUES (?, ?, ?)",
            (handoff_id, phone, message),
        )


def get_handoff(handoff_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT phone, message FROM handoffs WHERE id = ?",
            (handoff_id,),
        ).fetchone()

    if not row:
        return None

    return {"user_phone": row[0], "user_message": row[1]}


def delete_handoff(handoff_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM handoffs WHERE id = ?", (handoff_id,))


def update_faq(user_question: str, staff_reply: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "INSERT INTO faq (question, answer) VALUES (?, ?)",
            (user_question, staff_reply),
        )


def is_general_query(user_message: str) -> bool:
    prompt = f"""
Classify this resort guest question for an FAQ system.

GENERAL means it applies to most guests: facilities, policies, location, nearby places.
USER-SPECIFIC means it depends on one guest, booking, refund, complaint, or situation.

Answer only true or false.

Question: {user_message}
"""
    response = _faq_classifier.invoke(prompt)
    return response.content.strip().lower() == "true"


def store_handoff_reply(thread_id: str, user_message: str, staff_text: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    current_state = graph.get_state(config)
    messages = current_state.values.get("messages", [])

    updates = [
        HumanMessage(content=f"Staff clarification for '{user_message}': {staff_text}")
    ]
    if messages:
        updates.insert(0, RemoveMessage(id=messages[-1].id))

    graph.update_state(config, {"messages": updates})


def generate_handoff_id() -> str:
    return uuid.uuid4().hex[:8].upper()


def trigger_handoff(user_phone: str, user_message: str) -> None:
    handoff_id = generate_handoff_id()
    save_handoff(handoff_id, user_phone, user_message)

    send_text(
        DEFAULT_HUMAN_PHONE,
        (
            "BOT NEEDS HELP\n\n"
            f"Handoff ID: {handoff_id}\n"
            f"User: {user_phone}\n\n"
            f"Question:\n\"{user_message}\"\n\n"
            f"Reply as:\nANSWER {handoff_id}: "
        ),
    )

    send_text(user_phone, "Let me check this with the staff and get back to you.")


def handle_human_reply(from_phone: str, message: str) -> bool:
    if from_phone != DEFAULT_HUMAN_PHONE:
        return False

    match = re.search(r"ANSWER\s+([A-F0-9]{8}):\s*(.+)", message, re.DOTALL)
    if not match:
        return False

    handoff_id, answer = match.groups()
    handoff = get_handoff(handoff_id)

    if not handoff:
        send_text(from_phone, "Invalid or expired handoff ID.")
        return True

    answer = answer.strip()
    send_text(
        handoff["user_phone"],
        (
            "Here is an update from our staff:\n\n"
            f"Your question:\n{handoff['user_message']}\n\n"
            f"Answer:\n{answer}"
        ),
    )
    send_text(from_phone, "Reply sent.")

    try:
        store_handoff_reply(
            thread_id=handoff["user_phone"],
            user_message=handoff["user_message"],
            staff_text=answer,
        )
    except Exception:
        log.exception("Failed to store handoff reply in graph state")

    try:
        if is_general_query(handoff["user_message"]):
            update_faq(handoff["user_message"], answer)
    except Exception:
        log.exception("Failed to update FAQ from handoff reply")

    delete_handoff(handoff_id)
    return True
