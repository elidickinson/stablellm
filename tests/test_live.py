"""End-to-end tests that hit a real upstream through the full proxy.

Opt-in: set STABLELLM_LIVE_TESTS=1 plus LIVE_TEST_BASE_URL / LIVE_TEST_API_KEY /
LIVE_TEST_MODEL in .env. Skipped by default to keep the standard suite offline
and free.

Example .env additions:

    STABLELLM_LIVE_TESTS=1
    LIVE_TEST_BASE_URL=https://api.cerebras.ai/v1
    LIVE_TEST_API_KEY=${CEREBRAS_API_KEY}
    LIVE_TEST_MODEL=zai-glm-4.7
"""
import os
import sys

import httpx
import pytest
import pytest_asyncio

from conftest import fresh_config


pytestmark = pytest.mark.live


if not os.getenv("STABLELLM_LIVE_TESTS"):
    pytest.skip("STABLELLM_LIVE_TESTS not set — live tests disabled", allow_module_level=True)


# Read live-test config from real env (not via dotenv neutralization)
import dotenv as _dotenv
_dotenv.load_dotenv()
LIVE_BASE_URL = os.getenv("LIVE_TEST_BASE_URL")
LIVE_API_KEY = os.getenv("LIVE_TEST_API_KEY")
LIVE_MODEL = os.getenv("LIVE_TEST_MODEL")

if not (LIVE_BASE_URL and LIVE_API_KEY and LIVE_MODEL):
    pytest.skip(
        "LIVE_TEST_BASE_URL / LIVE_TEST_API_KEY / LIVE_TEST_MODEL must all be set",
        allow_module_level=True,
    )


@pytest_asyncio.fixture
async def live_app(monkeypatch, tmp_path):
    """Proxy app wired to the real upstream from env, with a tiny race threshold."""
    # Re-enable dotenv so api_key interpolation works against the real env
    fresh_config(monkeypatch, tmp_path, {
        "settings": {"cooloff_seconds": 5, "race_interval_requests": 1},
        "providers": {"live": {"base_url": LIVE_BASE_URL, "api_key": LIVE_API_KEY}},
        "groups": {"live": [{"provider": "live", "model": LIVE_MODEL}]},
    })
    sys.modules.pop("main", None)
    import main

    main.http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=4.0))
    main._build_provider_groups()
    main._reset_runtime_state()
    try:
        yield main.app
    finally:
        await main.http_client.aclose()


TINY_PROMPT = {"messages": [{"role": "user", "content": "Reply with the single word: pong"}], "max_tokens": 16}


async def _post(app, body):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        return await c.post("/v1/chat/completions", json=body, timeout=30)


@pytest.mark.asyncio
async def test_live_models_lists_configured_group(live_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=live_app), base_url="http://t") as c:
        resp = await c.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "live" in ids


@pytest.mark.asyncio
async def test_live_chat_completion_non_streaming(live_app):
    resp = await _post(live_app, {**TINY_PROMPT, "model": "live"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Don't assert on response shape beyond "we got a choices list with a message" —
    # reasoning models like GLM may put output under reasoning/content/tool_calls.
    assert data["choices"][0]["message"]


@pytest.mark.asyncio
async def test_live_chat_completion_streaming(live_app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=live_app), base_url="http://t") as c:
        async with c.stream("POST", "/v1/chat/completions",
                             json={**TINY_PROMPT, "model": "live", "stream": True},
                             timeout=30) as resp:
            assert resp.status_code == 200
            body = b"".join([chunk async for chunk in resp.aiter_bytes()])
    assert b"data:" in body
    assert b"[DONE]" in body
