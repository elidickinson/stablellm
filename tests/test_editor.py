import sqlite3
import sys
import time

import httpx
import pytest
import yaml as _yaml
from conftest import fresh_config

MINIMAL_CONFIG = {
    "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
    "groups": {"default": {"endpoints": [{"provider": "a"}]}},
}


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    """Build an app with the editor enabled (or not) at the given password."""
    def build(password: str | None, content=None):
        if password is None:
            monkeypatch.delenv("CONFIG_EDITOR_PASSWORD", raising=False)
        else:
            monkeypatch.setenv("CONFIG_EDITOR_PASSWORD", password)

        content = content or MINIMAL_CONFIG
        cfg = fresh_config(monkeypatch, tmp_path, content)
        sys.modules.pop("main", None)
        import main
        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        main._build_provider_groups()
        # Skip the brute-force-mitigation delay in tests
        main.EDITOR_AUTH_DELAY_SECS = 0
        return main.app, cfg

    return build


async def _get(app, path, headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.get(path, headers=headers or {})


async def _post(app, path, body, headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, content=body, headers=headers or {})


@pytest.mark.asyncio
async def test_editor_returns_404_when_password_not_set(app_factory):
    app, _ = app_factory(password=None)
    for path in ("/config/editor", "/config/api/content", "/dashboard", "/dashboard/api/state"):
        resp = await _get(app, path)
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_editor_unauthorized_without_password_header(app_factory):
    app, _ = app_factory(password="secret")
    for path in ("/config/api/content", "/dashboard/api/state", "/dashboard/api/history"):
        resp = await _get(app, path)
        assert resp.status_code == 401


# --- dashboard: state ---

MULTI_PROVIDER_CONFIG = {
    "providers": {
        "a": {"base_url": "https://a.test", "api_key": "k"},
        "b": {"base_url": "https://b.test", "api_key": "k"},
    },
    "groups": {
        "one": {"endpoints": [{"provider": "a"}, {"provider": "b"}]},
        "two": {"endpoints": [{"provider": "a"}]},
    },
}


@pytest.mark.asyncio
async def test_dashboard_state_merges_shared_endpoints(app_factory):
    app, _ = app_factory(password="secret", content=MULTI_PROVIDER_CONFIG)
    m = sys.modules["main"]
    m._stats["requests"][0] = 5
    m._stats["successes"][0] = 4
    m._stats["failures"][0] = 1
    m._last_failure[0] = "HTTP 503: upstream died"

    resp = await _get(app, "/dashboard/api/state", {"X-Config-Password": "secret"})
    assert resp.status_code == 200
    rows = {r["provider"]: r for r in resp.json()["rows"]}
    # Provider a appears in two groups but is one merged row.
    assert set(rows["a"]["groups"]) == {"one", "two"}
    assert rows["a"]["requests"] == 5
    assert rows["a"]["failures"] == 1
    assert rows["a"]["last_error"] == "HTTP 503: upstream died"
    assert rows["a"]["state"] == "up"


@pytest.mark.asyncio
async def test_dashboard_manual_down_up_and_state(app_factory):
    app, _ = app_factory(password="secret", content=MULTI_PROVIDER_CONFIG)
    hdr = {"X-Config-Password": "secret"}

    resp = await _post(app, "/dashboard/api/down/a", None, hdr)
    assert resp.status_code == 200
    assert resp.json() == {"provider": "a", "down": True}

    state = (await _get(app, "/dashboard/api/state", hdr)).json()
    assert state["manual_down"] == ["a"]
    row = next(r for r in state["rows"] if r["provider"] == "a")
    assert row["state"] == "down"

    # Unknown provider names are rejected, not silently ignored.
    resp = await _post(app, "/dashboard/api/down/nope", None, hdr)
    assert resp.status_code == 404

    resp = await _post(app, "/dashboard/api/up/a", None, hdr)
    assert resp.json() == {"provider": "a", "down": False}
    state = (await _get(app, "/dashboard/api/state", hdr)).json()
    assert state["manual_down"] == []


# --- dashboard: history ---

@pytest.mark.asyncio
async def test_dashboard_history_reads_request_log(app_factory, monkeypatch, tmp_path):
    app, _ = app_factory(password="secret")
    m = sys.modules["main"]
    db = tmp_path / "req.db"
    monkeypatch.setattr(m.requestlog, "DB_PATH", db)
    m.requestlog.init()

    m.requestlog.log_request(m.requestlog.RequestMetrics(
        req_id="r1", keyname="esd", model_requested="kimi", provider_served="syn",
        model_served="K", mode="seq", status="200", ttft_ms=1000.0, tokens_per_sec=50.0))
    now = time.time()
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO requests (timestamp, status, provider_served, model_served, ttft_ms) "
        "VALUES (?, '200', 'syn', 'K', 3000.0)", (now - 7200,))  # outside 15m window
    conn.execute(
        "INSERT INTO requests (timestamp, status, provider_served, model_served) "
        "VALUES (?, '502', 'syn', 'K')", (now - 10,))  # recent, no ttft
    conn.execute(
        "INSERT INTO requests (timestamp, status, provider_served) "
        "VALUES (?, '502', '')", (now - 10,))  # exhaustion row: no provider
    conn.commit()
    conn.close()

    resp = await _get(app, "/dashboard/api/history", {"X-Config-Password": "secret"})
    assert resp.status_code == 200
    data = resp.json()
    r1 = next(r for r in data["requests"] if r["req_id"] == "r1")
    assert r1["mode"] == "seq"

    summary = data["summary"]
    assert set(summary) == {"syn|K"}  # empty-provider exhaustion rows excluded
    s = summary["syn|K"]
    assert s["reqs"]["15m"] == 2  # recent 200 + 502; 1h-old row excluded
    assert s["reqs"]["24h"] == 3
    assert s["ttft_ms"]["15m"] == 1000.0  # 200-only avg: 502's NULL ttft excluded
    assert s["ttft_ms"]["24h"] == 2000.0  # (1000 + 3000) / 2


@pytest.mark.asyncio
async def test_editor_auth_applies_delay_on_failure_only(app_factory):
    import time
    app, _ = app_factory(password="secret")
    import main
    main.EDITOR_AUTH_DELAY_SECS = 0.1

    t0 = time.monotonic()
    bad = await _get(app, "/config/api/content", headers={"X-Config-Password": "wrong"})
    bad_elapsed = time.monotonic() - t0

    t0 = time.monotonic()
    good = await _get(app, "/config/api/content", headers={"X-Config-Password": "secret"})
    good_elapsed = time.monotonic() - t0

    assert bad.status_code == 401 and good.status_code == 200
    assert bad_elapsed >= 0.1
    # The delay only guards failures; the dashboard polls successes at 1s.
    assert good_elapsed < 0.1


@pytest.mark.asyncio
async def test_editor_loads_current_yaml(app_factory):
    app, _ = app_factory(password="secret")
    resp = await _get(app, "/config/api/content", headers={"X-Config-Password": "secret"})
    assert resp.status_code == 200
    assert "providers:" in resp.text
    assert "a:" in resp.text


@pytest.mark.asyncio
async def test_save_rejects_invalid_yaml(app_factory):
    app, _ = app_factory(password="secret")
    resp = await _post(app, "/config/api/save", b"providers: [unclosed", headers={"X-Config-Password": "secret"})
    assert resp.status_code == 400
    assert "YAML error" in resp.text


@pytest.mark.asyncio
async def test_save_rejects_invalid_config(app_factory):
    app, _ = app_factory(password="secret")
    bad = _yaml.safe_dump({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"default": {"endpoints": [{"provider": "nope"}]}},
    })
    resp = await _post(app, "/config/api/save", bad.encode(), headers={"X-Config-Password": "secret"})
    assert resp.status_code == 400
    assert "validation failed" in resp.text


@pytest.mark.asyncio
async def test_save_writes_disk_and_hot_reloads(app_factory, tmp_path):
    app, cfg = app_factory(password="secret")
    # Verify initial state — default group, one endpoint
    assert "default" in cfg.GROUPS
    orig_count = len(cfg.ENDPOINTS)

    new_yaml = _yaml.safe_dump({
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
        "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b", "model": "m2"}]}},
    })
    resp = await _post(app, "/config/api/save", new_yaml.encode(), headers={"X-Config-Password": "secret"})
    assert resp.status_code == 200

    # File on disk was updated
    on_disk = (tmp_path / "config.yaml").read_text()
    assert "b:" in on_disk

    # In-memory state reloaded — new endpoint visible
    import config
    assert len(config.ENDPOINTS) == orig_count + 1
    assert config.ENDPOINTS[-1].model == "m2"
    assert config.ENDPOINTS[-1].base_url == "https://b.test"
