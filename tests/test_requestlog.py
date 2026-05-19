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
        model_requested="kimi-k2p6",
        provider_served="https://api.fireworks.ai/inference/v1",
        model_served="accounts/fireworks/routers/kimi-k2p6-turbo",
        ttft_ms=123.4,
        tokens_per_sec=98.7,
    ))

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT model_requested, provider_served, model_served, ttft_ms, tokens_per_sec FROM requests"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0] == (
        "kimi-k2p6",
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/routers/kimi-k2p6-turbo",
        123.4,
        98.7,
    )


def test_no_op_when_db_path_unset(monkeypatch, tmp_path):
    """When REQUEST_LOG_DB isn't set, init() and log_request() are silent no-ops."""
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
