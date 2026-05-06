"""
conversation_meta.py
--------------------
Manages the `conversation_meta` table in petesinn.sqlite.
"""

import sqlite3
import time
import logging
from datetime import datetime, timezone

from dashboard.conversation_router import (
    extract_messages_for_thread,
    list_threads,
    source_from_thread_id,
    thread_id_to_meta,
    get_serde,
)

log = logging.getLogger(__name__)

# Expected columns in the current schema (excluding thread_id)
_REQUIRED_COLUMNS = {"name", "source", "last_message", "last_sender", "last_time", "updated_at"}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_meta (
    thread_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'whatsapp',
    last_message  TEXT NOT NULL DEFAULT '',
    last_sender   TEXT NOT NULL DEFAULT 'guest',
    last_time     TEXT NOT NULL DEFAULT '',
    updated_at    REAL NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0 
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_meta_updated ON conversation_meta(updated_at DESC)
"""


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(conversation_meta)").fetchall()
    return {row[1] for row in rows}


def _table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_meta'"
    ).fetchone()
    return row is not None


def init_meta_table(conn: sqlite3.Connection) -> None:
    """
    Create conversation_meta with the correct schema.
    If an old/incompatible table exists (missing or extra columns),
    drop it and recreate — bootstrap will repopulate it from checkpoints.
    """
    if _table_exists(conn):
        existing = _existing_columns(conn)
        if not _REQUIRED_COLUMNS.issubset(existing):
            log.warning(
                "[META] Stale schema detected (columns=%s). Dropping and recreating table.",
                existing,
            )
            conn.execute("DROP TABLE conversation_meta")
            conn.commit()

    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_INDEX)
    conn.commit()
    log.info("[META] conversation_meta table ready")


def bootstrap_meta(conn: sqlite3.Connection) -> None:
    """
    On startup: insert meta rows for threads missing from the meta table.
    Already-present rows are left untouched.
    """
    serde   = get_serde()
    threads = list_threads(conn)

    if not threads:
        log.info("[META] No threads found — nothing to bootstrap")
        return

    existing = {
        r[0] for r in conn.execute("SELECT thread_id FROM conversation_meta").fetchall()
    }
    to_bootstrap = [t for t in threads if t not in existing]
    log.info("[META] Bootstrapping %d new threads (of %d total)", len(to_bootstrap), len(threads))

    inserted = 0
    for thread_id in to_bootstrap:
        try:
            msgs = extract_messages_for_thread(conn, serde, thread_id)
            if not msgs:
                continue

            last   = msgs[-1]
            meta   = thread_id_to_meta(thread_id)
            source = source_from_thread_id(thread_id)
            name   = meta["display_name"]

            if source == "whatsapp":
                row = conn.execute(
                    "SELECT phone FROM handoffs WHERE id = ?", (thread_id,)
                ).fetchone()
                if row:
                    name = row["phone"]

            upsert_meta(
                conn,
                thread_id=thread_id,
                name=name,
                source=source,
                last_message=last["text"],
                last_sender=last["from"],
                message_count=len(msgs),   # ADD THIS
                commit=False,
            )
            inserted += 1
        except Exception:
            log.exception("[META] Bootstrap failed for thread_id=%s", thread_id)

    conn.commit()
    log.info("[META] Bootstrap complete — inserted %d rows", inserted)


def upsert_meta(
    conn, thread_id, name, source,
    last_message, last_sender,
    message_count: int | None = None,   # ADD THIS
    commit=True,
):
    now_ts  = time.time()
    now_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(timespec="seconds")

    conn.execute(
        """
        INSERT INTO conversation_meta
            (thread_id, name, source, last_message, last_sender, last_time, updated_at, message_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, 0))
        ON CONFLICT(thread_id) DO UPDATE SET
            name          = excluded.name,
            last_message  = excluded.last_message,
            last_sender   = excluded.last_sender,
            last_time     = excluded.last_time,
            updated_at    = excluded.updated_at,
            message_count = CASE 
                WHEN ? IS NULL THEN message_count + 1   -- increment on new message
                ELSE ?                                   -- set exact count on bootstrap
            END
        """,
        (thread_id, name, source, last_message[:500], last_sender, now_iso, now_ts,
         message_count, message_count, message_count),
    )

    if commit:
        conn.commit()


def get_all_meta(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT thread_id, name, source, last_message, last_sender, last_time
        FROM   conversation_meta
        ORDER  BY updated_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]

