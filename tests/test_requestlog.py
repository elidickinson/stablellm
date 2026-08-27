"""Tests for the SQLite request log."""
import sqlite3
import sys

import pytest


@pytest.fixture
def fresh_requestlog(monkeypatch, tmp_path):
    """Return the requestlog module with DB_PATH pointed at a temp file."""
    db_path = tmp_path / "subdir" / "requests.db"
    sys.modules.pop("requestlog", None)
    import requestlog
    monkeypatch.setattr(requestlog, "DB_PATH", db_path)
    return requestlog, db_path


def test_init_creates_table_and_directory(fresh_requestlog):
    requestlog, db_path = fresh_requestlog
    assert not db_path.parent.exists()
    requestlog.init()
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    conn.close()
    assert "requests" in tables


def test_log_request_round_trip(fresh_requestlog):
    requestlog, db_path = fresh_requestlog
    requestlog.init()

    requestlog.log_request(requestlog.RequestMetrics(
        req_id="a1b2c3d4e5f6a7b8",
        keyname="alice",
        model_requested="kimi-k2p6",
        provider_served="fireworks",
        model_served="accounts/fireworks/routers/kimi-k2p6-turbo",
        status="200",
        ttft_ms=123.4,
        tokens=250,
        tokens_per_sec=98.7,
    ))

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT api_key_id, req_id, status, model_requested, provider_served, model_served, ttft_ms, tokens, tokens_per_sec FROM requests"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0] == (
        "alice",
        "a1b2c3d4e5f6a7b8",
        "200",
        "kimi-k2p6",
        "fireworks",
        "accounts/fireworks/routers/kimi-k2p6-turbo",
        123.4,
        250,
        98.7,
    )


def test_init_migrates_old_schema(fresh_requestlog):
    """A DB created before req_id/status/... columns existed gains them on init()."""
    requestlog, db_path = fresh_requestlog
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE requests (
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

    requestlog.init()

    conn = sqlite3.connect(db_path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(requests)")}
    conn.close()
    assert {"req_id", "status", "reason", "ttfb_ms", "elapsed_ms", "tokens"} <= cols


def test_summary_line(fresh_requestlog, caplog):
    """log_request emits the one-line-per-request summary at the right level."""
    requestlog, _ = fresh_requestlog
    requestlog.init()

    with caplog.at_level("INFO", logger="stablellm"):
        requestlog.log_request(requestlog.RequestMetrics(
            req_id="a1b2c3d4e5f6a7b8",
            keyname="alice",
            model_requested="llama-3.3",
            model_served="llama-3.3-70b",
            provider_served="cerebras",
            mode="race",
            stream=True,
            status="200",
            ttfb_ms=181.0,
            ttft_ms=220.0,
            tokens=245,
            tokens_per_sec=78.0,
        ))
        requestlog.log_request(requestlog.RequestMetrics(
            req_id="f" * 16,
            model_requested="llama-3.3",
            status="502",
            reason="all endpoints failed (last: HTTP 502)",
        ))

    success, failure = caplog.records
    assert success.levelname == "INFO"
    assert "req=a1b2c3d4e5f6a7b8 200 model=llama-3.3 served=llama-3.3-70b provider=cerebras mode=race stream=yes" in success.getMessage()
    assert "ttfb=181ms ttft=220ms tokens=245 tok/s=78 keyname=alice" in success.getMessage()
    assert failure.levelname == "ERROR"
    assert failure.getMessage().startswith(f"req={'f' * 16} 502 model=llama-3.3")
    assert "all endpoints failed (last: HTTP 502)" in failure.getMessage()


def test_no_op_when_db_path_unset(monkeypatch, tmp_path):
    """When REQUEST_LOG_DB isn't set, init() and log_request() skip the DB write
    (the console summary line still logs)."""
    sys.modules.pop("requestlog", None)
    import requestlog
    monkeypatch.setattr(requestlog, "DB_PATH", None)

    requestlog.init()  # must not raise
    requestlog.log_request(requestlog.RequestMetrics(model_requested="x"))  # must not raise


def test_log_request_handles_none_metrics(fresh_requestlog):
    """ttft_ms / tokens_per_sec may be None (non-streaming with no usage)."""
    requestlog, db_path = fresh_requestlog
    requestlog.init()
    requestlog.log_request(requestlog.RequestMetrics(model_requested="m"))

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT ttft_ms, tokens_per_sec FROM requests").fetchone()
    conn.close()
    assert row == (None, None)
