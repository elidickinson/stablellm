"""SQLite request logging + the one-line-per-request console summary.

log_request() is the single terminal-owner for a proxied request: it writes the
DB row (when REQUEST_LOG_DB is set) and emits the human-readable summary line.
INFO for normal outcomes, WARNING for interrupted streams, ERROR for 5xx.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from config import REQUEST_LOG_DB

DB_PATH = Path(REQUEST_LOG_DB) if REQUEST_LOG_DB else None

log = logging.getLogger("stablellm")

# Columns added to pre-existing DBs (CREATE TABLE IF NOT EXISTS won't migrate).
_NEW_COLUMNS = {
    "req_id": "TEXT",
    "status": "TEXT",
    "reason": "TEXT",
    "ttfb_ms": "REAL",
    "elapsed_ms": "REAL",
    "tokens": "INTEGER",
}


def init():
    if DB_PATH is None:
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            api_key_id TEXT,
            model_requested TEXT,
            provider_served TEXT,
            model_served TEXT,
            ttft_ms REAL,
            tokens_per_sec REAL
        )
    """)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
    for col, ddl in _NEW_COLUMNS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE requests ADD COLUMN {col} {ddl}")
    conn.commit()
    conn.close()


@dataclass
class RequestMetrics:
    req_id: str = ""
    keyname: str = ""
    model_requested: str = ""
    provider_served: str = ""
    model_served: str = ""
    via: str = ""
    mode: str = ""
    stream: bool | None = None
    status: str = ""
    reason: str = ""
    ttfb_ms: float | None = None
    ttft_ms: float | None = None
    elapsed_ms: float | None = None
    tokens: int | None = None
    tokens_per_sec: float | None = None


def _summary_line(m: RequestMetrics) -> str:
    parts = [f"req={m.req_id or '-'}", m.status or "?"]
    parts.append(f"model={m.model_requested}")
    if m.model_served and m.model_served != m.model_requested:
        parts.append(f"served={m.model_served}")
    if m.provider_served:
        parts.append(f"provider={m.provider_served}")
    if m.via:
        parts.append(f"via={m.via}")
    if m.mode:
        parts.append(f"mode={m.mode}")
    if m.stream is not None:
        parts.append(f"stream={'yes' if m.stream else 'no'}")
    if m.stream:
        if m.ttfb_ms is not None:
            parts.append(f"ttfb={m.ttfb_ms:.0f}ms")
        if m.ttft_ms is not None:
            parts.append(f"ttft={m.ttft_ms:.0f}ms")
    elif m.elapsed_ms is not None:
        parts.append(f"elapsed={m.elapsed_ms:.0f}ms")
    if m.tokens is not None:
        parts.append(f"tokens={m.tokens}")
    if m.tokens_per_sec is not None:
        parts.append(f"tok/s={m.tokens_per_sec:.0f}")
    parts.append(f"keyname={m.keyname or '-'}")
    if m.reason:
        parts.append(m.reason)
    return " ".join(parts)


def log_request(m: RequestMetrics):
    """Record the request in SQLite (if enabled) and emit the console summary line."""
    if m.status.startswith("5"):
        level = logging.ERROR
    elif m.status == "interrupted":
        level = logging.WARNING
    else:
        level = logging.INFO
    log.log(level, "%s", _summary_line(m))

    if DB_PATH is None:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO requests (timestamp, api_key_id, req_id, status, reason, model_requested, "
        "provider_served, model_served, ttfb_ms, ttft_ms, elapsed_ms, tokens, tokens_per_sec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            time.time(),
            m.keyname or None,
            m.req_id or None,
            m.status or None,
            m.reason or None,
            m.model_requested,
            m.provider_served,
            m.model_served,
            m.ttfb_ms,
            m.ttft_ms,
            m.elapsed_ms,
            m.tokens,
            m.tokens_per_sec,
        ),
    )
    conn.commit()
    conn.close()
