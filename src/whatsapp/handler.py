"""
whatsapp/handler.py
"""

import logging
import concurrent.futures

from graph.graph_whatsapp import graph
from .sender import send_whatsapp_message
from .handoff import handle_human_reply
from dashboard.conversation_router import get_conn
from dashboard.conversation_meta import upsert_meta
from dashboard.feature_flags import is_enabled
log = logging.getLogger(__name__)

_GRAPH_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="graph-worker",
)
GRAPH_TIMEOUT_SECONDS = 30


def process_whatsapp_message(sender_id: str, user_message: str) -> str | None:
    if not is_enabled("whatsapp_agent"):
        return None
    else:
        log.info("[INCOMING] sender=%s message=%r", sender_id, user_message[:120])

        # Update meta immediately so the dashboard shows the guest's message
        _update_meta(sender_id, last_message=user_message, last_sender="guest")

        if handle_human_reply(sender_id, user_message):
            log.info("[HANDLER] Human reply handled. Skipping AI. sender=%s", sender_id)
            return None

        config = {"configurable": {"thread_id": sender_id}}
        try:
            future = _GRAPH_EXECUTOR.submit(graph.invoke, {"messages": user_message}, config)
            result = future.result(timeout=GRAPH_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            log.error("[TIMEOUT] graph.invoke exceeded %ds. sender=%s", GRAPH_TIMEOUT_SECONDS, sender_id)
            return None
        except Exception:
            log.exception("[ERROR] graph.invoke failed. sender=%s", sender_id)
            return None

        reply = _extract_reply(result, sender_id)
        if not reply:
            return None

        log.info("[BOT] sender=%s reply=%r", sender_id, reply[:120])

        # Update meta with the agent's reply
        _update_meta(sender_id, last_message=reply, last_sender="agent")

        try:
            send_whatsapp_message(sender_id, reply, user_message)
        except Exception:
            log.exception("[ERROR] send_whatsapp_message failed. sender=%s", sender_id)

        return reply


def _update_meta(thread_id: str, last_message: str, last_sender: str) -> None:
    """Best-effort meta update — never crashes the main message flow."""
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
    last    = messages[-1]
    content = getattr(last, "content", None) or (
        last.get("content") if isinstance(last, dict) else None
    )
    if not content or not content.strip():
        log.warning("[WARN] Empty AI reply. sender=%s", sender_id)
        return None
    return content.strip()