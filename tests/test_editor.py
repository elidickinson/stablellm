import sys

import httpx
import pytest
import yaml as _yaml

from conftest import fresh_config


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    """Build an app with the editor enabled (or not) at the given password."""
    def build(password: str | None, content=None):
        if password is None:
            monkeypatch.delenv("CONFIG_EDITOR_PASSWORD", raising=False)
        else:
            monkeypatch.setenv("CONFIG_EDITOR_PASSWORD", password)

        content = content or {
            "endpoints": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        }
        cfg = fresh_config(monkeypatch, tmp_path, content)
        sys.modules.pop("main", None)
        import main
        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
        main._build_provider_groups()
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
async def test_editor_loads_current_yaml(app_factory):
    app, cfg = app_factory(password="secret")
    resp = await _get(app, "/config/api/content", headers={"X-Config-Password": "secret"})
    assert resp.status_code == 200
    assert "endpoints:" in resp.text
    assert "a:" in resp.text


@pytest.mark.asyncio
async def test_save_rejects_invalid_yaml(app_factory):
    app, _ = app_factory(password="secret")
    resp = await _post(app, "/config/api/save", b"endpoints: [unclosed", headers={"X-Config-Password": "secret"})
    assert resp.status_code == 400
    assert "YAML error" in resp.text


@pytest.mark.asyncio
async def test_save_rejects_invalid_config(app_factory):
    app, _ = app_factory(password="secret")
    bad = _yaml.safe_dump({
        "endpoints": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"x": ["doesnotexist"]},
    })
    resp = await _post(app, "/config/api/save", bad.encode(), headers={"X-Config-Password": "secret"})
    assert resp.status_code == 400
    assert "validation failed" in resp.text


@pytest.mark.asyncio
async def test_save_writes_disk_and_hot_reloads(app_factory, tmp_path):
    app, cfg = app_factory(password="secret")
    new_yaml = _yaml.safe_dump({
        "endpoints": {
            "a": {"base_url": "https://a.test", "api_key": "k"},
            "newep": {"base_url": "https://new.test", "api_key": "k", "model": "new-model"},
        },
    })
    resp = await _post(app, "/config/api/save", new_yaml.encode(), headers={"X-Config-Password": "secret"})
    assert resp.status_code == 200
    # File on disk was updated
    on_disk = (tmp_path / "config.yaml").read_text()
    assert "newep" in on_disk
    # In-memory state reloaded — new endpoint visible
    import config
    assert "newep" in config.ENDPOINT_NAMES
    assert config.ENDPOINTS[config.ENDPOINT_NAMES["newep"]].model == "new-model"
