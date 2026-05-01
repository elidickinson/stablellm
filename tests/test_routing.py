import json
import sys

import httpx
import pytest

from conftest import fresh_config


@pytest.fixture
def app_with_endpoints(monkeypatch):
    """Build a fresh app with given env, plus a recorder of upstream requests.

    Returns (asgi_app, calls) where calls is a list of (base_url, body_dict).
    """
    def build(env):
        fresh_config(monkeypatch, env)
        sys.modules.pop("main", None)
        import main

        calls: list[tuple[str, dict]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            base = f"{request.url.scheme}://{request.url.host}"
            body = json.loads(request.content.decode()) if request.content else {}
            calls.append((base, body))
            return httpx.Response(
                200,
                json={"id": "x", "choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )

        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        # Skip lifespan; manually rebuild groups since lifespan() normally does it
        main._build_provider_groups()
        return main.app, calls

    return build


async def _post(app, body):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=body)
    return resp


@pytest.mark.asyncio
async def test_group_routes_only_to_listed_endpoints_in_order(app_with_endpoints):
    app, calls = app_with_endpoints({
        "ENDPOINT_A": "https://a.test|k|model-a",
        "ENDPOINT_B": "https://b.test|k|model-b",
        "ENDPOINT_C": "https://c.test|k|model-c",
        "GROUP_CHEAP": "c,a",
    })
    resp = await _post(app, {"model": "cheap", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    # Only the first endpoint in the group is hit when it succeeds
    assert len(calls) == 1
    assert calls[0][0] == "https://c.test"
    assert calls[0][1]["model"] == "model-c"


@pytest.mark.asyncio
async def test_unknown_model_falls_back_to_default_group(app_with_endpoints):
    app, calls = app_with_endpoints({
        "ENDPOINT_A": "https://a.test|k|",
        "ENDPOINT_B": "https://b.test|k|",
    })
    resp = await _post(app, {"model": "anything", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    # Empty endpoint model + default group → client model passes through
    assert calls[0][1]["model"] == "anything"


@pytest.mark.asyncio
async def test_empty_endpoint_model_in_named_group_uses_client_model(app_with_endpoints):
    app, calls = app_with_endpoints({
        "ENDPOINT_A": "https://a.test|k|",
        "ENDPOINT_B": "https://b.test|k|gpt-4o-mini",
        "GROUP_GPT_4O": "a,b",
    })
    resp = await _post(app, {"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    # Empty endpoint model + named group → client's original model passes through (preserves dashes)
    assert calls[0][1]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_group_lookup_normalizes_dashes_dots_underscores(app_with_endpoints):
    app, calls = app_with_endpoints({
        "ENDPOINT_A": "https://a.test|k|model-a",
        "GROUP_GPT_4_1": "a",
    })
    # Env-var name "GPT_4_1" should match request model "gpt-4.1"
    resp = await _post(app, {"model": "gpt-4.1", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    assert calls[0][1]["model"] == "model-a"


@pytest.mark.asyncio
async def test_fastest_suffix_is_stripped_before_group_lookup(app_with_endpoints):
    app, calls = app_with_endpoints({
        "ENDPOINT_A": "https://a.test|k|model-a",
        "ENDPOINT_B": "https://b.test|k|model-b",
        "GROUP_CHEAP": "a",  # single endpoint -> race short-circuits, falls through to sequential
    })
    resp = await _post(app, {"model": "cheap:fastest", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    # The literal "cheap" (or "cheap:fastest") must never reach upstream
    assert calls[0][1]["model"] == "model-a"
