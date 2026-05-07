import logging

from dashboard.conversation_meta import upsert_meta
from dashboard.conversation_router import get_conn
from dashboard.feature_flags import is_enabled
from graph.graph_whatsapp import graph

from .client import send_text
from .handoff import handle_human_reply, trigger_handoff
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

log = logging.getLogger(__name__)
HANDOFF_TRIGGER = "let me check with the staff"


def process_whatsapp_message(sender_id: str, user_message: str) -> str | None:
    if not is_enabled("whatsapp_agent"):
        graph.update_state(
            {"configurable": {"thread_id": sender_id}},
            {"messages": [HumanMessage(content=user_message)]},
        )
        return None

    log.info("[INCOMING] sender=%s message=%r", sender_id, user_message[:120])
    _update_meta(sender_id, last_message=user_message, last_sender="guest")

    if handle_human_reply(sender_id, user_message):
        log.info("[HANDLER] Human reply handled. Skipping AI. sender=%s", sender_id)
        return None

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            {"configurable": {"thread_id": sender_id}},
        )
    except Exception:
        log.exception("[ERROR] graph.invoke failed. sender=%s", sender_id)
        return None

    reply = _extract_reply(result, sender_id)
    if not reply:
        return None

    log.info("[BOT] sender=%s reply=%r", sender_id, reply[:120])
    _update_meta(sender_id, last_message=reply, last_sender="agent")

    try:
        if HANDOFF_TRIGGER in reply.lower().strip():
            trigger_handoff(sender_id, user_message)
        else:
            send_text(sender_id, reply)
    except Exception:
        log.exception("[ERROR] WhatsApp send failed. sender=%s", sender_id)

    return reply


def _update_meta(thread_id: str, last_message: str, last_sender: str) -> None:
    try:
        conn = get_conn()
        name = f"+{thread_id}" if thread_id.isdigit() else thread_id
        upsert_meta(
            conn,
            thread_id=thread_id,
            name=name,
            source="whatsapp",
            last_message=last_message,
            last_sender=last_sender,
        )
        conn.close()
    except Exception:
        log.exception("[META] Failed to update meta for thread_id=%s", thread_id)


def _extract_reply(result: dict, sender_id: str) -> str | None:
    if not isinstance(result, dict):
        log.warning("[WARN] Unexpected result type: %s. sender=%s", type(result), sender_id)
        return None

    messages = result.get("messages", [])
    if not messages:
        log.warning("[WARN] No messages in graph result. sender=%s", sender_id)
        return None

    last = messages[-1]
    content = getattr(last, "content", None) or (
        last.get("content") if isinstance(last, dict) else None
    )
    if not content or not content.strip():
        log.warning("[WARN] Empty AI reply. sender=%s", sender_id)
        return None

    return content.strip()
