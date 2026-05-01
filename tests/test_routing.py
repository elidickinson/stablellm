import json
import sys

import httpx
import pytest

from conftest import fresh_config


@pytest.fixture
def app_with_endpoints(monkeypatch, tmp_path):
    """Build a fresh app with given YAML config, plus a recorder of upstream requests.

    Returns (asgi_app, calls) where calls is a list of (base_url, body_dict).
    """
    def build(content):
        fresh_config(monkeypatch, tmp_path, content)
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
        "endpoints": {
            "a": {"base_url": "https://a.test", "api_key": "k", "model": "model-a"},
            "b": {"base_url": "https://b.test", "api_key": "k", "model": "model-b"},
            "c": {"base_url": "https://c.test", "api_key": "k", "model": "model-c"},
        },
        "groups": {"cheap": ["c", "a"]},
    })
    resp = await _post(app, {"model": "cheap", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0][0] == "https://c.test"
    assert calls[0][1]["model"] == "model-c"


@pytest.mark.asyncio
async def test_unknown_model_falls_back_to_default_group(app_with_endpoints):
    app, calls = app_with_endpoints({
        "endpoints": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
    })
    resp = await _post(app, {"model": "anything", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    assert calls[0][1]["model"] == "anything"


@pytest.mark.asyncio
async def test_empty_endpoint_model_in_named_group_uses_client_model(app_with_endpoints):
    app, calls = app_with_endpoints({
        "endpoints": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k", "model": "gpt-4o-mini"},
        },
        "groups": {"gpt_4o": ["a", "b"]},
    })
    resp = await _post(app, {"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    # Empty endpoint model + named group → client's original model passes through (preserves dashes)
    assert calls[0][1]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_fastest_suffix_is_stripped_before_group_lookup(app_with_endpoints):
    app, calls = app_with_endpoints({
        "endpoints": {
            "a": {"base_url": "https://a.test", "api_key": "k", "model": "model-a"},
            "b": {"base_url": "https://b.test", "api_key": "k", "model": "model-b"},
        },
        "groups": {"cheap": ["a"]},  # single endpoint -> race short-circuits
    })
    resp = await _post(app, {"model": "cheap:fastest", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    assert calls[0][1]["model"] == "model-a"


@pytest.mark.asyncio
async def test_group_lookup_normalizes_dashes_dots_underscores(app_with_endpoints):
    app, calls = app_with_endpoints({
        "endpoints": {"a": {"base_url": "https://a.test", "api_key": "k", "model": "model-a"}},
        "groups": {"gpt_4_1": ["a"]},
    })
    resp = await _post(app, {"model": "gpt-4.1", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    assert calls[0][1]["model"] == "model-a"


@pytest.mark.asyncio
async def test_reload_picks_up_new_endpoints(app_with_endpoints, tmp_path, monkeypatch):
    app, calls = app_with_endpoints({
        "endpoints": {"a": {"base_url": "https://a.test", "api_key": "k", "model": "model-a"}},
    })
    resp = await _post(app, {"model": "anything", "messages": []})
    assert resp.status_code == 200
    assert calls[-1][0] == "https://a.test"

    # Rewrite the same config file with a new endpoint and reload
    import yaml as _yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_yaml.safe_dump({
        "endpoints": {
            "a": {"base_url": "https://a.test", "api_key": "k", "model": "model-a"},
            "b": {"base_url": "https://b.test", "api_key": "k", "model": "model-b"},
        },
        "groups": {"default": ["b", "a"]},
    }))

    import config
    import main
    config.reload()
    main._build_provider_groups()

    resp = await _post(app, {"model": "anything", "messages": []})
    assert resp.status_code == 200
    # New default order puts b first
    assert calls[-1][0] == "https://b.test"
