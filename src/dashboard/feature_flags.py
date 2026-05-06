import time
import logging
import sqlite3

log = logging.getLogger(__name__)

_flags: dict[str, bool] = {}


def init_flags(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feature_flags (
            key        TEXT PRIMARY KEY,
            enabled    INTEGER NOT NULL DEFAULT 1,
            updated_at REAL NOT NULL DEFAULT 0
        )
    """)
    conn.executemany(
        "INSERT OR IGNORE INTO feature_flags (key, enabled, updated_at) VALUES (?, 1, ?)",
        [("whatsapp_agent", time.time()), ("voice_agent", time.time())]
    )
    conn.commit()
    log.info("[FLAGS] feature_flags table ready")


def load_flags(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT key, enabled FROM feature_flags").fetchall()
    for row in rows:
        _flags[row["key"]] = bool(row["enabled"])
    log.info("[FLAGS] Loaded: %s", _flags)


def is_enabled(key: str) -> bool:
    return _flags.get(key, True)


def set_flag(conn: sqlite3.Connection, key: str, enabled: bool) -> None:
    _flags[key] = enabled
    conn.execute(
        "INSERT INTO feature_flags (key, enabled, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET enabled = excluded.enabled, updated_at = excluded.updated_at",
        (key, int(enabled), time.time())
    )
    conn.commit()
    log.info("[FLAGS] %s set to %s", key, enabled)