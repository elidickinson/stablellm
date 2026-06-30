"""Integration tests for the proxy runtime: failover, cooloff, race, streaming, auth."""
import asyncio
import json
import sys

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
async def test_upstream_error_log_includes_request_context_and_body(proxy_app, capsys):
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
    assert "path=/v1/chat/completions" in logged
    assert "requested_model='default'" in logged
    assert "upstream_model='upstream-a'" in logged
    assert "messages=1" in logged
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
async def test_connection_error_marks_down_and_tries_next(proxy_app):
    def handler(req):
        if req.url.host == "a.test":
            raise httpx.ConnectError("refused", request=req)
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

    app, calls, main = proxy_app(
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
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={"model": "default", "stream": True, "messages": []}) as resp:
            assert resp.status_code == 200
            chunks = b"".join([chunk async for chunk in resp.aiter_bytes()])
    assert b"hello" in chunks
    assert b"[DONE]" in chunks


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
async def test_api_key_required_when_configured(proxy_app, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-proxy-key")
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
    # No auth header → 401
    no_auth = await _post(app, {"model": "default", "messages": []})
    assert no_auth.status_code == 401

    # Wrong key → 401
    wrong = await _post(app, {"model": "default", "messages": []}, headers={"Authorization": "Bearer wrong"})
    assert wrong.status_code == 401

    # Correct key → 200
    ok = await _post(app, {"model": "default", "messages": []}, headers={"Authorization": "Bearer secret-proxy-key"})
    assert ok.status_code == 200


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
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions", json={"model": "mygroup", "stream": True, "messages": []}) as resp:
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
