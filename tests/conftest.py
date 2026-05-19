import importlib
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    """Provide a placeholder CONFIG_FILE so `import config` succeeds at test-module collection.
    Individual tests overwrite this via the make_config fixture."""
    placeholder = Path(tempfile.gettempdir()) / "stablellm_test_placeholder.yaml"
    placeholder.write_text(yaml.safe_dump({
        "providers": {"placeholder": {"base_url": "https://placeholder", "api_key": "k"}},
        "groups": {"default": {"endpoints": [{"provider": "placeholder"}]}},
    }))
    os.environ["CONFIG_FILE"] = str(placeholder)


def _write_yaml(tmp_path: Path, content: str | dict) -> Path:
    path = tmp_path / "config.yaml"
    if isinstance(content, dict):
        path.write_text(yaml.safe_dump(content))
    else:
        path.write_text(textwrap.dedent(content))
    return path


def fresh_config(monkeypatch, tmp_path: Path, content: str | dict):
    """Reload config.py against a fresh YAML file. Returns the config module."""
    path = _write_yaml(tmp_path, content)
    monkeypatch.setenv("CONFIG_FILE", str(path))
    # Clear any leaked env vars that could affect interpolation behavior
    for k in list(os.environ):
        if k.startswith("ENDPOINT_") or k.startswith("GROUP_"):
            monkeypatch.delenv(k, raising=False)
    # Neutralize dotenv so the real .env doesn't leak in
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("config", "main"):
        sys.modules.pop(mod, None)
    import config
    cfg = importlib.reload(config)
    cfg.reload()
    return cfg


@pytest.fixture
def make_config(monkeypatch, tmp_path):
    return lambda content: fresh_config(monkeypatch, tmp_path, content)
