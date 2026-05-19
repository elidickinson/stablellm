import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from config import REQUEST_LOG_DB

DB_PATH = Path(REQUEST_LOG_DB) if REQUEST_LOG_DB else None


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
    conn.commit()
    conn.close()


@dataclass
class RequestMetrics:
    model_requested: str = ""
    provider_served: str = ""
    model_served: str = ""
    ttft_ms: float | None = None
    tokens_per_sec: float | None = None


def log_request(m: RequestMetrics):
    if DB_PATH is None:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO requests (timestamp, api_key_id, model_requested, provider_served, model_served, ttft_ms, tokens_per_sec) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (time.time(), None, m.model_requested, m.provider_served, m.model_served, m.ttft_ms, m.tokens_per_sec),
    )
    conn.commit()
    conn.close()
