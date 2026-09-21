import asyncio
import json
import sys

import httpx
import pytest
from conftest import fresh_config

from performance_routing import PerformanceRouting, fast_tags


def _endpoint(tag, latency, throughput):
    return {
        "tag": tag,
        "latency_last_30m": {"p50": latency},
        "throughput_last_30m": {"p50": throughput},
    }


def test_fast_tags_use_projected_completion_time():
    tags = fast_tags([
        _endpoint("fast", 500, 100),
        _endpoint("near", 100, 90),
        _endpoint("slow", 100, 70),
        _endpoint("unknown", None, None),
    ], PerformanceRouting(target_tokens=10_000, tolerance=0.15))
    assert tags == ["fast", "near"]


@pytest.fixture
def performance_app(monkeypatch, tmp_path):
    def build(handler):
        fresh_config(monkeypatch, tmp_path, {
            "providers": {"openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"}},
            "groups": {
                "default": {
                    "endpoints": [{
                        "provider": "openrouter",
                        "model": "author/model",
                        "performance_routing": {},
                        "routing": {
                            "sort": "throughput",
                            "order": ["old"],
                            "preferred_max_latency": {"p50": 1},
                            "max_price": {"prompt": 1},
                            "quantizations": ["fp8"],
                        },
                    }],
                },
            },
        })
        sys.modules.pop("main", None)
        import main

        calls = []

        async def wrapped(request):
            body = json.loads(request.content) if request.content else None
            calls.append((request.method, request.url.path, body))
            response = handler(request, body)
            if asyncio.iscoroutine(response):
                response = await response
            return response

        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
        main._build_provider_groups()
        main._reset_runtime_state()
        return main.app, calls

    return build


async def _post(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
        })


@pytest.mark.asyncio
async def test_performance_routing_generates_only_and_caches_catalog(performance_app):
    rows = [_endpoint("fast", 400, 100), _endpoint("near", 100, 90), _endpoint("slow", 100, 60)]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={
            "provider": "fast",
            "choices": [],
            "usage": {"completion_tokens": 1},
        })

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    assert (await _post(app)).status_code == 200

    assert [method for method, _, _ in calls] == ["GET", "POST", "POST"]
    provider = calls[1][2]["provider"]
    assert provider["only"] == ["fast", "near"]
    assert provider["allow_fallbacks"] is True
    assert provider["max_price"] == {"prompt": 1}
    assert provider["quantizations"] == ["fp8"]
    assert "sort" not in provider
    assert "order" not in provider
    assert "preferred_max_latency" not in provider


@pytest.mark.asyncio
async def test_pinned_via_stays_in_generated_only(performance_app):
    rows = [_endpoint("fast", 400, 100), _endpoint("slow", 100, 60)]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={
            "provider": "cached-provider",
            "choices": [],
            "usage": {"completion_tokens": 1},
        })

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    assert (await _post(app)).status_code == 200
    assert calls[2][2]["provider"]["only"] == ["fast", "cached-provider"]


@pytest.mark.asyncio
async def test_no_compatible_generated_only_retries_raw_routing(performance_app):
    rows = [_endpoint("fast", 400, 100)]
    post_calls = 0

    def handler(request, _body):
        nonlocal post_calls
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        post_calls += 1
        if post_calls == 1:
            return httpx.Response(400, json={"error": {"message": "No providers available"}})
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    assert calls[1][2]["provider"]["only"] == ["fast"]
    assert calls[1][2]["provider"]["allow_fallbacks"] is True
    assert "only" not in calls[2][2]["provider"]
    assert calls[2][2]["provider"]["sort"] == "throughput"
