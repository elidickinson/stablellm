import sys

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
    for path in ("/config/editor", "/config/api/content"):
        resp = await _get(app, path)
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_editor_unauthorized_without_password_header(app_factory):
    app, _ = app_factory(password="secret")
    resp = await _get(app, "/config/api/content")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_editor_auth_applies_constant_delay(app_factory):
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
    assert good_elapsed >= 0.1
    assert abs(bad_elapsed - good_elapsed) < 0.05


@pytest.mark.asyncio
async def test_editor_loads_current_yaml(app_factory):
    app, cfg = app_factory(password="secret")
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
