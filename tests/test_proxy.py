"""Integration tests for the proxy runtime: failover, cooloff, race, streaming, auth."""
import asyncio
import json
import sys
import time

import httpx
import pytest
from conftest import fresh_config


def _ok_response(extra: dict | None = None) -> httpx.Response:
    body = {
        "id": "x",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"completion_tokens": 5},
    }
    if extra:
        body.update(extra)
    return httpx.Response(200, json=body)


@pytest.fixture
def proxy_app(monkeypatch, tmp_path):
    """Build a fresh proxy app. The caller supplies (config_dict, route_handler).

    The handler is called for each upstream request: handler(request) -> httpx.Response
    (or raises httpx.* errors to simulate transport failures).
    """
    def build(content, route_handler):
        fresh_config(monkeypatch, tmp_path, content)
        sys.modules.pop("main", None)
        import main

        calls: list[tuple[str, dict | None, str]] = []

        async def wrapped(request: httpx.Request) -> httpx.Response:
            try:
                body = json.loads(request.content.decode()) if request.content else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                body = None
            base = f"{request.url.scheme}://{request.url.host}"
            calls.append((base, body, str(request.url.path)))
            result = route_handler(request)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
        main._build_provider_groups()
        main._reset_runtime_state()
        return main.app, calls, main

    return build


async def _post(app, body, headers=None, path="/v1/chat/completions"):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(path, json=body, headers=headers or {})


async def _get(app, path, headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.get(path, headers=headers or {})


# --- failover on non-200 responses ---

@pytest.mark.asyncio
async def test_5xx_marks_endpoint_down_and_tries_next(proxy_app):
    def handler(req):
        if req.url.host == "a.test":
            return httpx.Response(503, json={"error": "down"})
        return _ok_response()

    app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://a.test", "https://b.test"]
    assert main._stats["failures"][0] == 1
    assert main._stats["successes"][1] == 1


@pytest.mark.asyncio
async def test_4xx_also_triggers_failover(proxy_app):
    """Any non-200 — including 400/401 — should fail over (per commit efe9e08)."""
    def handler(req):
        if req.url.host == "a.test":
            return httpx.Response(401, json={"error": "unauthorized"})
        return _ok_response()

    app, calls, _ = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_upstream_error_log_includes_group_and_reason(proxy_app, capsys):
    def handler(req):
        if req.url.host == "a.test":
            return httpx.Response(400, json={"error": {"message": "bad model"}})
        return _ok_response()

    app, _, _ = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k", "model": "upstream-a"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "default", "messages": [{"role": "user", "content": "x"}]})
    assert resp.status_code == 200

    logged = capsys.readouterr().err
    assert "req=" in logged
    assert "group='default'" in logged
    assert "messages=1" in logged
    assert "marked down" in logged
    assert "bad model" in logged


@pytest.mark.asyncio
async def test_all_endpoints_failing_returns_502(proxy_app):
    def handler(req):
        return httpx.Response(500, json={"error": "boom"})

    app, _calls, _ = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 502
    assert "exhausted" in resp.json()["error"]


@pytest.mark.asyncio
async def test_all_endpoints_cooling_off_returns_502(proxy_app):
    """With nothing left to try, exhaustion must still produce a 502, not a 500."""
    def handler(req):
        return httpx.Response(503, json={"error": "down"})

    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        handler,
    )
    first = await _post(app, {"model": "default", "messages": []})
    assert first.status_code == 502
    # Endpoint is cooling off now; the next request skips it and must still 502.
    second = await _post(app, {"model": "default", "messages": []})
    assert second.status_code == 502
    assert "cooling off" in second.json()["error"]


@pytest.mark.asyncio
async def test_race_group_without_raceable_providers_keeps_race_mode(proxy_app, capsys):
    """A race group with one available provider uses its preferred order
    without claiming that the group's routing mode changed to sequential."""
    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"mode": "race", "endpoints": [{"provider": "a"}]}},
        },
        lambda req: _ok_response(),
    )
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert resp.headers["x-stablellm-mode"] == "race"

    logged = capsys.readouterr().err
    assert "race failed" not in logged
    assert "mode=seq" not in logged


@pytest.mark.asyncio
async def test_connection_error_marks_down_and_tries_next(proxy_app):
    def handler(req):
        if req.url.host == "a.test":
            raise httpx.ConnectError("refused", request=req)
        return _ok_response()

    app, _, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert main._stats["failures"][0] == 1


# --- cooloff ---

@pytest.mark.asyncio
async def test_cooled_off_endpoint_is_skipped_on_next_request(proxy_app):
    """After a failure, the failed endpoint stays cooling off and is skipped."""
    state = {"calls": 0}

    def handler(req):
        state["calls"] += 1
        if req.url.host == "a.test":
            return httpx.Response(503)
        return _ok_response()

    app, calls, _ = proxy_app(
        {
            "settings": {"cooloff_seconds": 30},
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    # First request: a fails, b succeeds
    await _post(app, {"model": "default", "messages": []})
    assert len(calls) == 2

    # Second request: a is cooled off, only b is tried
    calls.clear()
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://b.test"]


# --- streaming ---

@pytest.mark.asyncio
async def test_streaming_response_passes_through_chunks(proxy_app):
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}],"usage":{"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(req):
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        handler,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, \
            c.stream("POST", "/v1/chat/completions", json={"model": "default", "stream": True, "messages": []}) as resp:
        assert resp.status_code == 200
        chunks = b"".join([chunk async for chunk in resp.aiter_bytes()])
    assert b"hello" in chunks
    assert b"[DONE]" in chunks


@pytest.mark.asyncio
async def test_midstream_upstream_failure_emits_single_interrupted_row(proxy_app, monkeypatch):
    """Upstream dying after headers yields exactly one terminal row: interrupted, with reason."""
    async def broken_stream():
        yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
        raise httpx.RemoteProtocolError("peer closed connection mid-stream")

    def handler(req):
        return httpx.Response(200, content=broken_stream(), headers={"content-type": "text/event-stream"})

    app, _calls, main = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        handler,
    )
    rows: list[object] = []
    monkeypatch.setattr(main.requestlog, "log_request", rows.append)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        with pytest.raises(httpx.RemoteProtocolError):
            async with c.stream("POST", "/v1/chat/completions", json={"model": "default", "stream": True, "messages": []}) as resp:
                async for _chunk in resp.aiter_bytes():
                    pass

    assert len(rows) == 1
    metrics = rows[0]
    assert metrics.status == "interrupted"
    assert "peer closed connection mid-stream" in metrics.reason


@pytest.mark.asyncio
async def test_streaming_failover_emits_single_terminal_row(proxy_app, monkeypatch):
    """A rejected first attempt must not log; only the successful attempt gets a row."""
    sse_body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    def handler(req):
        if req.url.host == "a.test":
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    app, _calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    rows: list[object] = []
    monkeypatch.setattr(main.requestlog, "log_request", rows.append)

    resp = await _post(app, {"model": "default", "messages": [], "stream": True})
    assert resp.status_code == 200
    assert [m.status for m in rows] == ["200"]


# --- /stats ---

@pytest.mark.asyncio
async def test_stats_reflects_request_counts(proxy_app):
    def handler(req):
        return _ok_response()

    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        handler,
    )
    await _post(app, {"model": "default", "messages": []})
    await _post(app, {"model": "default", "messages": []})

    resp = await _get(app, "/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["endpoints"][0]["requests"] == 2
    assert data["endpoints"][0]["successes"] == 2
    assert data["endpoints"][0]["failures"] == 0


# --- middleware: body size + auth + invalid path ---

@pytest.mark.asyncio
async def test_oversized_body_is_rejected(proxy_app, monkeypatch):
    monkeypatch.setenv("MAX_BODY_BYTES", "100")
    # Reload config so the new env var takes effect
    import importlib

    import config
    importlib.reload(config)

    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    big_body = {"model": "default", "messages": [{"role": "user", "content": "x" * 500}]}
    resp = await _post(app, big_body)
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_any_configured_api_key_grants_access(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "alice:secret-one, plain-two")

    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    # No auth header → 401
    no_auth = await _post(app, {"model": "default", "messages": []})
    assert no_auth.status_code == 401

    # Wrong key → 401
    wrong = await _post(app, {"model": "default", "messages": []}, headers={"Authorization": "Bearer wrong"})
    assert wrong.status_code == 401

    for key in ("secret-one", "plain-two"):
        ok = await _post(app, {"model": "default", "messages": []}, headers={"Authorization": f"Bearer {key}"})
        assert ok.status_code == 200


@pytest.mark.asyncio
async def test_authenticated_client_name_reaches_request_log(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "alice:secret-one,plain-two")

    app, _calls, main = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    logged: list[str] = []
    monkeypatch.setattr(main.requestlog, "log_request", lambda m: logged.append(m.keyname))

    await _post(app, {"model": "default", "messages": []}, headers={"Authorization": "Bearer secret-one"})
    await _post(app, {"model": "default", "messages": []}, headers={"Authorization": "Bearer plain-two"})

    # Unnamed keys get a stable id derived from their hash.
    assert logged[0] == "alice"
    assert logged[1].startswith("key-")


@pytest.mark.asyncio
async def test_path_traversal_rejected(proxy_app):
    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    resp = await _post(app, {"model": "default", "messages": []}, path="/v1/..%2Fsecret")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_unknown_model_returns_404(proxy_app):
    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"only-this-model": {"endpoints": [{"provider": "a", "model": "m"}]}},
        },
        lambda r: _ok_response(),
    )
    resp = await _post(app, {"model": "something-else", "messages": []})
    assert resp.status_code == 404


# --- race mode (via :race suffix and group default) ---

async def _race_winner_handler(req):
    """a is slow, b is fast — b wins the race."""
    if req.url.host == "a.test":
        await asyncio.sleep(0.05)
        return _ok_response({"who": "a"})
    return _ok_response({"who": "b"})


_TWO_PROVIDER_RACE_CFG = {
    "providers": {
        "a": {"base_url": "https://a.test", "api_key": "k"},
        "b": {"base_url": "https://b.test", "api_key": "k"},
    },
    "groups": {"fast": {"endpoints": [
        {"provider": "a", "model": "ma"},
        {"provider": "b", "model": "mb"},
    ]}},
}


@pytest.mark.asyncio
async def test_race_suffix_routes_to_winner(proxy_app):
    app, calls, _ = proxy_app(_TWO_PROVIDER_RACE_CFG, _race_winner_handler)
    resp = await _post(app, {"model": "fast:race", "messages": []})
    assert resp.status_code == 200
    assert resp.json()["who"] == "b"
    assert {c[0] for c in calls} == {"https://a.test", "https://b.test"}


@pytest.mark.asyncio
async def test_fastest_is_an_alias_for_race(proxy_app):
    app, calls, _ = proxy_app(_TWO_PROVIDER_RACE_CFG, _race_winner_handler)
    resp = await _post(app, {"model": "fast:fastest", "messages": []})
    assert resp.status_code == 200
    assert resp.json()["who"] == "b"
    assert {c[0] for c in calls} == {"https://a.test", "https://b.test"}


@pytest.mark.asyncio
async def test_streaming_race_publishes_winner_for_next_request(proxy_app, monkeypatch):
    """A completed streaming race must update the order used between races."""
    delays = {"a.test": 0.04, "b.test": 0.0, "c.test": 0.02}
    sse_body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    async def handler(req):
        await asyncio.sleep(delays[req.url.host])
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    cfg = {
        "providers": {
            name: {"base_url": f"https://{name}.test", "api_key": "k"}
            for name in "abc"
        },
        "groups": {"fast": {"mode": "race", "endpoints": [
            {"provider": name, "model": f"m{name}"} for name in "abc"
        ]}},
    }
    app, calls, main = proxy_app(cfg, handler)
    body = {"model": "fast", "stream": True, "messages": []}
    finish_calls = []
    real_finish_race = main._finish_race

    def finish_race(*args, **kwargs):
        finish_calls.append((args, kwargs))
        return real_finish_race(*args, **kwargs)

    monkeypatch.setattr(main, "_finish_race", finish_race)

    response = await _post(app, body)
    assert response.status_code == 200
    await asyncio.gather(*tuple(main._background_tasks), return_exceptions=False)
    assert len(finish_calls) == 1
    assert finish_calls[0][1]["accounted"] == 3
    assert finish_calls[0][1]["total"] == 3
    assert [main._pk_label("fast", key) for key in main._group_preferred_providers["fast"]] == ["b", "c", "a"]

    # The next request is between race intervals, so it uses the preferred
    # order without starting another race and retains race mode.
    calls.clear()
    response = await _post(app, body)
    assert response.status_code == 200
    assert response.headers["x-stablellm-mode"] == "race"
    assert [call[0] for call in calls] == ["https://b.test"]


@pytest.mark.asyncio
async def test_race_bounds_never_ending_loser(proxy_app, monkeypatch):
    """A loser that keeps streaming is timed out and still finalizes the race."""
    class EndlessStream(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            while True:
                await asyncio.sleep(0.001)
                yield b": keepalive\\n\\n"

        async def aclose(self):
            self.closed = True

    loser_stream = EndlessStream()

    async def handler(req):
        if req.url.host == "b.test":
            await asyncio.sleep(0.01)
            return httpx.Response(200, stream=loser_stream)
        return _ok_response()

    cfg = {
        "settings": {"race_settle_timeout_secs": 0.2},
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
        "groups": {"fast": {"mode": "race", "endpoints": [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ]}},
    }
    app, calls, main = proxy_app(cfg, handler)
    monkeypatch.setattr(main, "_RACE_DRAIN_MIN_GRACE_SECS", 0.02)

    finish_calls = []
    real_finish_race = main._finish_race

    def finish_race(*args, **kwargs):
        finish_calls.append((args, kwargs))
        return real_finish_race(*args, **kwargs)

    monkeypatch.setattr(main, "_finish_race", finish_race)

    response = await _post(app, {"model": "fast", "messages": []})
    assert response.status_code == 200
    await asyncio.gather(*tuple(main._background_tasks), return_exceptions=False)

    assert [call[0] for call in calls] == ["https://a.test", "https://b.test"]
    assert len(finish_calls) == 1
    assert finish_calls[0][1]["accounted"] == 2
    assert finish_calls[0][1]["total"] == 2
    assert [main._pk_label("fast", key) for key in main._group_preferred_providers["fast"]] == ["a", "b"]
    assert loser_stream.closed
    assert not main._cooloff_until
    assert not main._inflight


@pytest.mark.asyncio
async def test_race_bounds_loser_waiting_for_headers(proxy_app, monkeypatch):
    """A loser that never returns headers is timed out and accounted."""
    async def handler(req):
        if req.url.host == "b.test":
            await asyncio.sleep(10)
        return _ok_response()

    cfg = {
        "settings": {"race_settle_timeout_secs": 0.02},
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k", "max_concurrency": 1},
            "b": {"base_url": "https://b.test", "api_key": "k", "max_concurrency": 1},
        },
        "groups": {"fast": {"mode": "race", "endpoints": [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ]}},
    }
    app, calls, main = proxy_app(cfg, handler)

    finish_calls = []
    real_finish_race = main._finish_race

    def finish_race(*args, **kwargs):
        finish_calls.append((args, kwargs))
        return real_finish_race(*args, **kwargs)

    monkeypatch.setattr(main, "_finish_race", finish_race)

    response = await _post(app, {"model": "fast", "messages": []})
    assert response.status_code == 200
    await asyncio.gather(*tuple(main._background_tasks), return_exceptions=False)

    assert {call[0] for call in calls} == {"https://a.test", "https://b.test"}
    assert len(finish_calls) == 1
    assert finish_calls[0][1]["accounted"] == 2
    assert finish_calls[0][1]["total"] == 2
    assert all(value == 0 for value in main._inflight.values())
    assert not main._cooloff_until


@pytest.mark.asyncio
async def test_group_mode_race_triggers_race_without_suffix(proxy_app):
    """When the group itself declares mode: race, requests race by default."""
    cfg = {
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
        "groups": {"fast": {"mode": "race", "endpoints": [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ]}},
    }
    app, calls, _ = proxy_app(cfg, _race_winner_handler)
    resp = await _post(app, {"model": "fast", "messages": []})
    assert resp.status_code == 200
    assert {c[0] for c in calls} == {"https://a.test", "https://b.test"}


@pytest.mark.asyncio
async def test_seq_suffix_overrides_group_race_default(proxy_app):
    """A group with mode: race can be forced sequential via :seq suffix."""
    cfg = {
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
        "groups": {"fast": {"mode": "race", "endpoints": [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ]}},
    }
    app, calls, _ = proxy_app(cfg, lambda r: _ok_response())
    resp = await _post(app, {"model": "fast:seq", "messages": []})
    assert resp.status_code == 200
    # Sequential → only the first endpoint is hit
    assert len(calls) == 1
    assert calls[0][0] == "https://a.test"


@pytest.mark.asyncio
async def test_normal_is_an_alias_for_seq(proxy_app):
    cfg = {
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
        "groups": {"fast": {"mode": "race", "endpoints": [
            {"provider": "a", "model": "ma"},
            {"provider": "b", "model": "mb"},
        ]}},
    }
    app, calls, _ = proxy_app(cfg, lambda r: _ok_response())
    resp = await _post(app, {"model": "fast:normal", "messages": []})
    assert resp.status_code == 200
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_double_suffix_returns_400(proxy_app):
    app, _calls, _ = proxy_app(_TWO_PROVIDER_RACE_CFG, lambda r: _ok_response())
    resp = await _post(app, {"model": "fast:race:seq", "messages": []})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_race_falls_back_to_sequential_when_race_fails(proxy_app):
    """When all race candidates fail, the proxy retries them sequentially (cooloff=0 here so they stay eligible)."""
    def handler(req):
        return httpx.Response(503)

    app, calls, _ = proxy_app(
        {
            "settings": {"cooloff_seconds": 0},
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"fast": {"endpoints": [
                {"provider": "a", "model": "ma"},
                {"provider": "b", "model": "mb"},
            ]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "fast:race", "messages": []})
    assert resp.status_code == 502
    # Race tried both, sequential then tried both again (cooloff=0 ⇒ not skipped)
    assert len(calls) >= 4


@pytest.mark.asyncio
async def test_race_with_single_candidate_skips_race(proxy_app):
    """With only one (model, base_url) key there's no one to race against — route directly."""
    app, calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"only": {"endpoints": [{"provider": "a", "model": "ma"}]}},
        },
        lambda r: _ok_response(),
    )
    resp = await _post(app, {"model": "only:race", "messages": []})
    assert resp.status_code == 200
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_race_buffered_drain_closes_response_on_cancellation(proxy_app):
    """A cancelled race winner still closes its response before propagating."""
    class FakeResponse:
        status_code = 200

        def __init__(self, cancelled=False):
            self.cancelled = cancelled
            self.closed = False

        async def aiter_bytes(self):
            yield b"partial"
            if self.cancelled:
                raise asyncio.CancelledError

        async def aclose(self):
            self.closed = True

    winner = FakeResponse(cancelled=True)
    loser = FakeResponse()

    class FakeClient:
        def build_request(self, _method, url, **_kwargs):
            return url

        async def send(self, url, **_kwargs):
            if url.startswith("https://a.test/"):
                return winner
            await asyncio.sleep(0.01)
            return loser

    _app, _calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"fast": {"mode": "race", "endpoints": [
                {"provider": "a", "model": "ma"},
                {"provider": "b", "model": "mb"},
            ]}},
        },
        lambda _req: _ok_response(),
    )
    main.http_client = FakeClient()

    with pytest.raises(asyncio.CancelledError):
        await main._race_request(
            "chat/completions", {"model": "fast", "messages": []}, False,
            "fast", "", "req-test", "test", "",
        )

    assert winner.closed
    await asyncio.gather(*tuple(main._background_tasks), return_exceptions=True)


# --- response metadata headers ---


@pytest.mark.asyncio
async def test_meta_headers_on_buffered_response(proxy_app):
    app, _, _ = proxy_app(
        {
            "providers": {"openai": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"mygroup": {"endpoints": [{"provider": "openai", "model": "gpt-4"}]}},
        },
        lambda r: _ok_response(),
    )
    resp = await _post(app, {"model": "mygroup", "messages": []})
    assert resp.status_code == 200
    assert resp.headers["x-stablellm-provider"] == "openai"
    assert resp.headers["x-stablellm-model"] == "gpt-4"
    assert resp.headers["x-stablellm-mode"] == "seq"
    assert resp.headers["x-stablellm-group"] == "mygroup"


@pytest.mark.asyncio
async def test_meta_headers_on_streaming_response(proxy_app):
    sse_body = (
        b'data: {"choices":[{"delta":{"content":"hi"}}],"usage":{"completion_tokens":1}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(req):
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    app, _, _ = proxy_app(
        {
            "providers": {"cerebras": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"mygroup": {"endpoints": [{"provider": "cerebras", "model": "fast-m"}]}},
        },
        handler,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, \
            c.stream("POST", "/v1/chat/completions", json={"model": "mygroup", "stream": True, "messages": []}) as resp:
        assert resp.headers["x-stablellm-provider"] == "cerebras"
        assert resp.headers["x-stablellm-model"] == "fast-m"
        assert resp.headers["x-stablellm-mode"] == "seq"
        _ = [chunk async for chunk in resp.aiter_bytes()]


@pytest.mark.asyncio
async def test_meta_headers_on_race_response(proxy_app):
    app, _, _ = proxy_app(_TWO_PROVIDER_RACE_CFG, _race_winner_handler)
    resp = await _post(app, {"model": "fast:race", "messages": []})
    assert resp.status_code == 200
    assert resp.headers["x-stablellm-provider"] == "b"
    assert resp.headers["x-stablellm-model"] == "mb"
    assert resp.headers["x-stablellm-mode"] == "race"
    assert resp.headers["x-stablellm-group"] == "fast"


@pytest.mark.asyncio
async def test_meta_headers_after_failover(proxy_app):
    def handler(req):
        if req.url.host == "a.test":
            return httpx.Response(503)
        return _ok_response()

    app, _, _ = proxy_app(
        {
            "providers": {
                "bad": {"base_url": "https://a.test", "api_key": "k"},
                "good": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"g": {"endpoints": [{"provider": "bad"}, {"provider": "good"}]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "g", "messages": []})
    assert resp.status_code == 200
    assert resp.headers["x-stablellm-provider"] == "good"


# --- X-StableLLM-Via (OpenRouter sub-provider) ---


@pytest.mark.asyncio
async def test_via_header_from_openrouter_buffered(proxy_app):
    """OpenRouter upstreams tag the body with top-level `provider`; surface it."""
    app, _, _ = proxy_app(
        {
            "providers": {"or": {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"}},
            "groups": {"g": {"endpoints": [{"provider": "or", "model": "openai/gpt-4o-mini"}]}},
        },
        lambda r: _ok_response({"provider": "OpenAI", "model": "openai/gpt-4o-mini"}),
    )
    resp = await _post(app, {"model": "g", "messages": []})
    assert resp.status_code == 200
    assert resp.headers["x-stablellm-via"] == "OpenAI"
    # Upstream body passes through untouched (provider field preserved)
    assert resp.json()["provider"] == "OpenAI"


@pytest.mark.asyncio
async def test_via_header_from_openrouter_streaming(proxy_app):
    sse_body = (
        b': OPENROUTER PROCESSING\n\n'
        b'data: {"choices":[{"delta":{"content":"hi"}}],"provider":"Cerebras","model":"m"}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(req):
        return httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"})

    app, _, _ = proxy_app(
        {
            "providers": {"or": {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"}},
            "groups": {"g": {"endpoints": [{"provider": "or", "model": "m"}]}},
        },
        handler,
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, \
            c.stream("POST", "/v1/chat/completions", json={"model": "g", "stream": True, "messages": []}) as resp:
        assert resp.headers["x-stablellm-via"] == "Cerebras"
        body = b"".join([chunk async for chunk in resp.aiter_bytes()])
    # First-chunk priming must not corrupt the stream
    assert b"hi" in body
    assert b"[DONE]" in body


@pytest.mark.asyncio
async def test_no_via_header_for_non_openrouter(proxy_app):
    """Non-OpenRouter upstreams don't carry a sub-provider; no via header."""
    app, _, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"g": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    resp = await _post(app, {"model": "g", "messages": []})
    assert resp.status_code == 200
    assert "x-stablellm-via" not in resp.headers


@pytest.mark.asyncio
async def test_via_header_from_openrouter_race_streaming(proxy_app):
    """Race winner through OpenRouter: priming must read past the keepalive
    comment to find the provider, and the winner's stream stays intact."""
    def handler(req):
        sse = (
            b': OPENROUTER PROCESSING\n\n'
            b'data: {"choices":[{"delta":{"content":"x"}}],"provider":"Azure","model":"m"}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    cfg = {
        "providers": {
            "or1": {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"},
            "or2": {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"},
        },
        "groups": {"fast": {"endpoints": [
            {"provider": "or1", "model": "ma"},
            {"provider": "or2", "model": "mb"},
        ]}},
    }
    app, _, _ = proxy_app(cfg, handler)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c, \
            c.stream("POST", "/v1/chat/completions", json={"model": "fast:race", "stream": True, "messages": []}) as resp:
        assert resp.status_code == 200
        assert resp.headers["x-stablellm-via"] == "Azure"
        body = b"".join([chunk async for chunk in resp.aiter_bytes()])
    assert b"x" in body and b"[DONE]" in body


# --- reasoning strip ---

@pytest.mark.asyncio
async def test_reasoning_stripped_from_messages_by_default(proxy_app):
    def handler(req):
        return _ok_response()

    app, calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a", "model": "m"}]}},
        },
        handler,
    )
    await _post(app, {
        "model": "default",
        "messages": [{"role": "assistant", "content": "x", "reasoning": "secret", "thinking": "also"}],
    })
    sent_msg = calls[0][1]["messages"][0]
    assert "reasoning" not in sent_msg
    assert "thinking" not in sent_msg


@pytest.mark.asyncio
async def test_missing_model_returns_400(proxy_app):
    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    resp = await _post(app, {"messages": []})
    assert resp.status_code == 400
    assert "'model'" in resp.json()["error"]


@pytest.mark.asyncio
async def test_empty_body_returns_400(proxy_app):
    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post("/v1/chat/completions")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_invalid_json_body_returns_400(proxy_app):
    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.post("/v1/chat/completions", content=b"{not json")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_models_endpoint_requires_auth_when_api_key_set(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-proxy-key")
    import importlib

    import config
    importlib.reload(config)

    app, _calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {
                "default": {"endpoints": [{"provider": "a"}]},
                "fast": {"endpoints": [{"provider": "a", "model": "m"}]},
            },
        },
        lambda r: _ok_response(),
    )
    no_auth = await _get(app, "/v1/models")
    assert no_auth.status_code == 401

    ok = await _get(app, "/v1/models", headers={"Authorization": "Bearer secret-proxy-key"})
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_models_endpoint_includes_meta(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-proxy-key")
    app, _, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {
                "m1": {
                    "endpoints": [{"provider": "a"}],
                    "meta": {
                        "name": "M1",
                        "context": 100000,
                        "max_output": 8000,
                        "modalities": ["text", "image"],
                        "input_cost": 1.0,
                        "output_cost": 3.0,
                        "cache_read_cost": 0.1,
                        "cache_write_cost": 3.0,
                        "supports_reasoning": True,
                        "reasoning_efforts": ["low", "high"],
                    },
                },
            },
        },
        lambda r: _ok_response(),
    )
    resp = await _get(app, "/v1/models", headers={"Authorization": "Bearer secret-proxy-key"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    m = data[0]
    assert m["id"] == "m1"
    assert m["object"] == "model"  # OpenAI keys stay even in OpenRouter shape
    assert m["context_length"] == 100000
    assert m["architecture"] == {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "modality": "text+image->text",
    }
    assert m["pricing"]["prompt"] == "0.000001"
    assert m["pricing"]["completion"] == "0.000003"
    assert m["pricing"]["input_cache_read"] == "0.0000001"
    assert m["pricing"]["input_cache_write"] == "0.000003"
    assert m["top_provider"] == {
        "is_moderated": False, "context_length": 100000, "max_completion_tokens": 8000,
    }
    # No default_enabled given -> key omitted; default_effort falls back to first effort
    assert m["reasoning"] == {
        "mandatory": False, "supported_efforts": ["low", "high"], "default_effort": "low",
    }


@pytest.mark.asyncio
async def test_models_endpoint_partial_meta_has_no_null(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-proxy-key")
    app, _, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"only": {"endpoints": [{"provider": "a"}], "meta": {"name": "Only"}}},
        },
        lambda r: _ok_response(),
    )
    resp = await _get(app, "/v1/models", headers={"Authorization": "Bearer secret-proxy-key"})
    assert resp.status_code == 200
    assert resp.json()["data"][0] == {
        "id": "only", "object": "model", "created": 0, "owned_by": "stablellm",
        "name": "Only",
        "architecture": {
            "input_modalities": ["text"], "output_modalities": ["text"], "modality": "text->text",
        },
    }


@pytest.mark.asyncio
async def test_models_endpoint_mandatory_reasoning(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-proxy-key")
    app, _, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"r": {"endpoints": [{"provider": "a"}], "meta": {
                "supports_reasoning": True, "reasoning_mandatory": True,
            }}},
        },
        lambda r: _ok_response(),
    )
    # Mandatory reasoning without efforts mirrors real OpenRouter entries
    resp = await _get(app, "/v1/models", headers={"Authorization": "Bearer secret-proxy-key"})
    assert resp.json()["data"][0]["reasoning"] == {"mandatory": True, "default_effort": "high"}


@pytest.mark.asyncio
async def test_models_endpoint_without_meta_is_minimal(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-proxy-key")
    app, _, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"plain": {"endpoints": [{"provider": "a"}]}},
        },
        lambda r: _ok_response(),
    )
    resp = await _get(app, "/v1/models", headers={"Authorization": "Bearer secret-proxy-key"})
    m = resp.json()["data"][0]
    assert m == {"id": "plain", "object": "model", "created": 0, "owned_by": "stablellm"}


@pytest.mark.asyncio
async def test_reasoning_param_passthrough(proxy_app):
    sent = {}

    def handler(req):
        sent.update(json.loads(req.content))
        return _ok_response()

    app, _, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        handler,
    )
    await _post(app, {"model": "default", "messages": [], "reasoning": {"effort": "low"}})
    assert sent["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_reasoning_kept_with_flag(proxy_app):
    def handler(req):
        return _ok_response()

    app, calls, _ = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a", "model": "m", "flags": ["keep_reasoning"]}]}},
        },
        handler,
    )
    await _post(app, {
        "model": "default",
        "messages": [{"role": "assistant", "content": "x", "reasoning": "kept"}],
    })
    assert calls[0][1]["messages"][0]["reasoning"] == "kept"


# --- concurrency caps ---


@pytest.mark.asyncio
async def test_capped_endpoint_is_skipped_until_slot_frees(proxy_app):
    def handler(req):
        return _ok_response()

    _app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k", "max_concurrency": 1},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a", "model": "m"}, {"provider": "b", "model": "m"}]}},
        },
        handler,
    )
    body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}
    assert main._try_acquire_slot(main.config.ENDPOINTS[0], "m")

    resp = await _post(_app, body)
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://b.test"]
    assert main._inflight[("https://a.test", "m")] == 1

    main._slot_releaser(main.config.ENDPOINTS[0], "m")()
    calls.clear()
    # a different session (different first message -> no pin) starts at a again
    resp = await _post(_app, {"model": "default", "messages": [{"role": "user", "content": "second session"}]})
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://a.test"]
    assert main._inflight[("https://a.test", "m")] == 0


@pytest.mark.asyncio
async def test_streaming_request_holds_slot_until_stream_ends(proxy_app):
    async def two_chunk_stream():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def handler(req):
        if req.url.host == "a.test":
            return httpx.Response(200, content=two_chunk_stream())
        return _ok_response()

    _app, _calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k", "max_concurrency": 1},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a", "model": "m"}, {"provider": "b", "model": "m"}]}},
        },
        handler,
    )
    ep = main.config.ENDPOINTS[0]
    assert main._try_acquire_slot(ep, "m")
    release = main._slot_releaser(ep, "m")
    metrics = main.requestlog.RequestMetrics(
        req_id="t", keyname="", model_requested="m", model_served="m", provider_served="a", mode="seq", stream=True,
    )

    result, reason = await main._proxy_stream(ep, "/v1/chat/completions", {}, b"{}", metrics, "ctx", on_done=release)
    assert result is not None and reason is None
    # Slot is held after the response is returned, not just until headers arrive.
    assert main._inflight[("https://a.test", "m")] == 1

    chunks = [chunk async for chunk in result.body_iterator]
    assert b"[DONE]" in chunks[-1]
    # Released once the stream has been fully consumed.
    assert main._inflight[("https://a.test", "m")] == 0


@pytest.mark.asyncio
async def test_capped_provider_excluded_from_race(proxy_app):
    def handler(req):
        return _ok_response()

    app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k", "max_concurrency": 1},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"mode": "race", "endpoints": [{"provider": "a", "model": "m"}, {"provider": "b", "model": "m"}]}},
        },
        handler,
    )
    assert main._try_acquire_slot(main.config.ENDPOINTS[0], "m")

    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    # a is capped -> only b is raceable -> race skipped, sequential hits b
    assert [c[0] for c in calls] == ["https://b.test"]


# --- session pinning ---


@pytest.mark.asyncio
async def test_session_pin_keeps_session_on_bounced_endpoint(proxy_app):
    def handler(req):
        return _ok_response()

    app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}

    # a cools off; the session bounces to b and pins there
    main._cooloff_until[0] = time.monotonic() + 60
    resp = await _post(app, body)
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://b.test"]

    # a is back, but the session stays on the endpoint that served it
    del main._cooloff_until[0]
    calls.clear()
    resp = await _post(app, body)
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://b.test"]

    # a different session still starts at the configured head
    calls.clear()
    resp = await _post(app, {"model": "default", "messages": [{"role": "user", "content": "other session"}]})
    assert [c[0] for c in calls] == ["https://a.test"]

    # expiring the pin returns the first session to configured order
    import config

    skey = main._session_key(body)
    idx, home, ts = main._session_pins[("default", skey)]
    main._session_pins[("default", skey)] = (idx, home, ts - config.SETTINGS.session_pin_ttl_secs - 1)
    calls.clear()
    resp = await _post(app, body)
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://a.test"]


# --- ttfb deadline ---


@pytest.mark.asyncio
async def test_ttfb_deadline_fails_over_to_next_endpoint(proxy_app):
    async def slow(req):
        await asyncio.sleep(0.5)
        return _ok_response()

    def handler(req):
        if req.url.host == "a.test":
            return slow(req)
        return _ok_response()

    app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k", "ttfb_deadline_secs": 0.05},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://a.test", "https://b.test"]
    assert main._cooloff_until[0] > time.monotonic()  # a marked down


@pytest.mark.asyncio
async def test_stream_slot_released_when_client_never_iterates(proxy_app):
    """A response dropped before iteration (early client disconnect) must still
    release the slot: _proxy_stream starts the generator, so its finally runs."""
    async def two_chunk_stream():
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def handler(req):
        return httpx.Response(200, content=two_chunk_stream())

    _app, _calls, main = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k", "max_concurrency": 1}},
            "groups": {"default": {"endpoints": [{"provider": "a", "model": "m"}]}},
        },
        handler,
    )
    ep = main.config.ENDPOINTS[0]
    assert main._try_acquire_slot(ep, "m")
    release = main._slot_releaser(ep, "m")
    metrics = main.requestlog.RequestMetrics(
        req_id="t", keyname="", model_requested="m", model_served="m", provider_served="a", mode="seq", stream=True,
    )

    result, _reason = await main._proxy_stream(ep, "/v1/chat/completions", {}, b"{}", metrics, "ctx", on_done=release)
    assert result is not None
    await result.body_iterator.aclose()  # starlette drops it without iterating
    assert main._inflight[("https://a.test", "m")] == 0


# --- pin metadata header ---


@pytest.mark.asyncio
async def test_pin_header_states(proxy_app):
    def handler(req):
        return _ok_response()

    app, _calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}

    resp = await _post(app, body)
    assert resp.headers["x-stablellm-pin"] == "new; home=a"
    resp = await _post(app, body)
    assert resp.headers["x-stablellm-pin"] == "hit; home=a"
    assert main._pin_promotions == 1  # counts served-by-home, not promotions

    # home cools off -> bounce to b, then the new home sticks
    main._cooloff_until[0] = time.monotonic() + 60
    resp = await _post(app, body)
    assert resp.headers["x-stablellm-pin"] == "bounce; home=a"
    assert main._pin_promotions == 1  # a bounce is not a hit
    del main._cooloff_until[0]
    resp = await _post(app, body)
    assert resp.headers["x-stablellm-pin"] == "hit; home=b"
    assert main._pin_promotions == 2


@pytest.mark.asyncio
async def test_race_pins_winner_and_skips_pinned_sessions(proxy_app):
    def handler(req):
        return _ok_response()

    app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"mode": "race", "endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}

    resp = await _post(app, body)
    assert resp.headers["x-stablellm-pin"] == "new; home=a"
    assert len(calls) == 2  # raced both providers

    # Cadence is ripe again, but the session is pinned: no race, stays home.
    calls.clear()
    main._group_race_request_count["default"] = 9999
    resp = await _post(app, body)
    assert resp.headers["x-stablellm-pin"] == "hit; home=a"
    assert [c[0] for c in calls] == ["https://a.test"]  # no fan-out

    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.headers["x-stablellm-pin"] == "none"  # no session derivable


@pytest.mark.asyncio
async def test_race_cancel_during_winner_read_releases_slot(proxy_app):
    """Cancellation arriving during winner body read (or the close that
    follows it) must still release the race winner's concurrency slot."""
    class FakeResponse:
        status_code = 200

        def __init__(self, cancelled=False):
            self.cancelled = cancelled
            self.closed = False

        async def aiter_bytes(self):
            yield b"partial"
            if self.cancelled:
                raise asyncio.CancelledError

        async def aclose(self):
            self.closed = True

    winner = FakeResponse(cancelled=True)
    loser = FakeResponse()

    class FakeClient:
        def build_request(self, _method, url, **_kwargs):
            return url

        async def send(self, url, **_kwargs):
            if url.startswith("https://a.test/"):
                return winner
            await asyncio.sleep(0.01)
            return loser

    _app, _calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k", "max_concurrency": 1},
                "b": {"base_url": "https://b.test", "api_key": "k", "max_concurrency": 1},
            },
            "groups": {"fast": {"mode": "race", "endpoints": [
                {"provider": "a", "model": "ma"},
                {"provider": "b", "model": "mb"},
            ]}},
        },
        lambda _req: _ok_response(),
    )
    main.http_client = FakeClient()

    with pytest.raises(asyncio.CancelledError):
        await main._race_request(
            "chat/completions", {"model": "fast", "messages": []}, False,
            "fast", "", "req-test", "test", "",
        )
    await asyncio.gather(*tuple(main._background_tasks), return_exceptions=True)

    assert winner.closed
    assert main._inflight[("https://a.test", "ma")] == 0
    assert main._inflight[("https://b.test", "mb")] == 0


# --- qwen review regressions ---


@pytest.mark.asyncio
async def test_stale_pin_index_is_dropped_not_crashed(proxy_app):
    """A request in flight during a reload can write a pin to an index that no
    longer exists; later requests must route normally, not IndexError."""
    def handler(req):
        return _ok_response()

    app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}
    skey = main._session_key(body)
    main._session_pins[("default", skey)] = (999, "a", time.monotonic())  # out of range

    resp = await _post(app, body)
    assert resp.status_code == 200
    assert [c[0] for c in calls] == ["https://a.test"]
    assert main._session_pins[("default", skey)][0] == 0  # re-pinned to the real head


@pytest.mark.asyncio
async def test_race_cancel_after_racer_completes_releases_slot(proxy_app, monkeypatch):
    """A cancel landing in the instant a racer completed (asyncio.wait raised
    without delivering it) must still release the completed racer's slot."""
    class FakeResponse:
        status_code = 200

        def __init__(self):
            self.closed = False

        async def aiter_bytes(self):
            yield b"partial"

        async def aclose(self):
            self.closed = True

    responses = []

    class FakeClient:
        def build_request(self, _method, url, **_kwargs):
            return url

        async def send(self, url, **_kwargs):
            r = FakeResponse()
            responses.append((url.split("/")[2], r))
            await asyncio.sleep(0)
            return r

    _app, _calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k", "max_concurrency": 1},
                "b": {"base_url": "https://b.test", "api_key": "k", "max_concurrency": 1},
            },
            "groups": {"fast": {"mode": "race", "endpoints": [
                {"provider": "a", "model": "ma"},
                {"provider": "b", "model": "mb"},
            ]}},
        },
        lambda _req: _ok_response(),
    )
    main.http_client = FakeClient()

    real_wait = asyncio.wait

    async def flaky_wait(fs, **kwargs):
        await real_wait(fs, return_when=asyncio.FIRST_COMPLETED)
        raise asyncio.CancelledError  # cancel swallowed the completed racer

    monkeypatch.setattr(main.asyncio, "wait", flaky_wait)

    with pytest.raises(asyncio.CancelledError):
        await main._race_request(
            "chat/completions", {"model": "fast", "messages": []}, False,
            "fast", "", "req-test", "test", "",
        )
    await asyncio.gather(*tuple(main._background_tasks), return_exceptions=True)

    assert all(r.closed for _host, r in responses)
    assert main._inflight[("https://a.test", "ma")] == 0
    assert main._inflight[("https://b.test", "mb")] == 0


@pytest.mark.asyncio
async def test_reload_clears_race_generation(proxy_app):
    """A pre-reload race's late drain must not pass _maybe_finalize's guard
    after the reload (the gen counter is cleared). Regression: monotonic ids
    alone don't invalidate; the clear is what does."""
    _app, _calls, main = proxy_app(
        {
            "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}]}},
        },
        lambda _req: _ok_response(),
    )
    main._group_race_generation["default"] = 7
    main._reset_runtime_state()
    assert "default" not in main._group_race_generation


@pytest.mark.asyncio
async def test_stale_pin_with_mismatched_home_label_is_dropped(proxy_app):
    """A pin whose index now names a different provider (in range, wrong
    home) is dropped so the header stays truthful, not served as a stale hit."""
    def handler(req):
        return _ok_response()

    app, calls, main = proxy_app(
        {
            "providers": {
                "a": {"base_url": "https://a.test", "api_key": "k"},
                "b": {"base_url": "https://b.test", "api_key": "k"},
            },
            "groups": {"default": {"endpoints": [{"provider": "a"}, {"provider": "b"}]}},
        },
        handler,
    )
    body = {"model": "default", "messages": [{"role": "user", "content": "hi"}]}
    skey = main._session_key(body)
    # Pin points at index 1 but remembers home "a"; index 1 is actually "b".
    main._session_pins[("default", skey)] = (1, "a", time.monotonic())

    resp = await _post(app, body)
    assert resp.status_code == 200
    # The stale pin was dropped, so the request used configured order (a), not
    # the mislabeled home (b).
    assert [c[0] for c in calls] == ["https://a.test"]
    assert resp.headers["x-stablellm-pin"] == "new; home=a"
