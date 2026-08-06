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
            import json
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
async def test_models_lists_all_groups(app_with_endpoints):
    app, _calls = app_with_endpoints({
        "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        "groups": {
            "default": {"endpoints": [{"provider": "a"}]},
            "cheap": {"endpoints": [{"provider": "a", "model": "m-cheap"}]},
            "fast": {"endpoints": [{"provider": "a", "model": "m-fast"}]},
        },
    })
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert {m["id"] for m in data["data"]} == {"default", "cheap", "fast"}


@pytest.mark.asyncio
async def test_named_group_routes_in_order(app_with_endpoints):
    app, calls = app_with_endpoints({
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
            "c": {"base_url": "https://c.test", "api_key": "k"},
        },
        "groups": {
            "default": {"endpoints": [{"provider": "a"}]},
            "cheap": {"endpoints": [{"provider": "c", "model": "model-c"}, {"provider": "a", "model": "model-a"}]},
        },
    })
    resp = await _post(app, {"model": "cheap", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert len(calls) == 1
    assert calls[0][0] == "https://c.test"
    assert calls[0][1]["model"] == "model-c"


@pytest.mark.asyncio
async def test_unknown_model_returns_404(app_with_endpoints):
    app, calls = app_with_endpoints({
        "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        "groups": {"default": {"endpoints": [{"provider": "a"}]}},
    })
    resp = await _post(app, {"model": "anything", "messages": []})
    assert resp.status_code == 404
    assert calls == []


@pytest.mark.asyncio
async def test_no_model_on_group_entry_passes_through_client_model(app_with_endpoints):
    """When a group entry omits 'model', the client's requested model passes through."""
    app, calls = app_with_endpoints({
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
        "groups": {
            "default": {"endpoints": [{"provider": "a"}, {"provider": "b", "model": "gpt-4o-mini"}]},
            "gpt-4o": {"endpoints": [{"provider": "a"}, {"provider": "b", "model": "gpt-4o-mini"}]},
        },
    })
    resp = await _post(app, {"model": "gpt-4o", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    # No model on entry → named group → client's model passes through
    assert calls[0][1]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_mode_suffix_is_stripped_before_group_lookup(app_with_endpoints):
    app, calls = app_with_endpoints({
        "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        "groups": {
            "default": {"endpoints": [{"provider": "a"}]},
            "cheap": {"endpoints": [{"provider": "a", "model": "model-a"}]},
        },
    })
    resp = await _post(app, {"model": "cheap:race", "messages": []})
    assert resp.status_code == 200
    assert calls[0][0] == "https://a.test"
    assert calls[0][1]["model"] == "model-a"


@pytest.mark.asyncio
async def test_group_name_matching_is_case_insensitive(app_with_endpoints):
    """A group declared as 'GPT-4o' matches whether the client sends 'gpt-4o' or 'GPT-4O'."""
    app, calls = app_with_endpoints({
        "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        "groups": {
            "default": {"endpoints": [{"provider": "a", "model": "fallback"}]},
            "GPT-4o": {"endpoints": [{"provider": "a", "model": "model-a"}]},
        },
    })
    for client_model in ("gpt-4o", "GPT-4O", "Gpt-4O"):
        resp = await _post(app, {"model": client_model, "messages": []})
        assert resp.status_code == 200
        assert calls[-1][1]["model"] == "model-a"


@pytest.mark.asyncio
async def test_group_name_separators_are_not_normalized(app_with_endpoints):
    """Only case is normalized — 'gpt-4.1' and 'gpt_4_1' remain distinct."""
    app, _calls = app_with_endpoints({
        "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        "groups": {"gpt-4.1": {"endpoints": [{"provider": "a", "model": "model-a"}]}},
    })
    resp = await _post(app, {"model": "gpt_4_1", "messages": []})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_reload_picks_up_new_providers(app_with_endpoints, tmp_path, monkeypatch):
    import yaml as _yaml
    app, calls = app_with_endpoints({
        "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        "groups": {"default": {"endpoints": [{"provider": "a", "model": "model-a"}]}},
    })
    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert calls[-1][0] == "https://a.test"

    # Rewrite config file with a new provider and reload
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_yaml.safe_dump({
        "providers": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "b": {"base_url": "https://b.test", "api_key": "k"},
        },
        "groups": {"default": {"endpoints": [{"provider": "b", "model": "model-b"}, {"provider": "a", "model": "model-a"}]}},
    }))

    import config
    import main
    config.reload()
    main._build_provider_groups()

    resp = await _post(app, {"model": "default", "messages": []})
    assert resp.status_code == 200
    assert calls[-1][0] == "https://b.test"
    assert calls[-1][1]["model"] == "model-b"
