import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fresh_config(monkeypatch, env: dict):
    """Reload config.py with only the given ENDPOINT_/GROUP_ vars set. Returns the config module."""
    for k in list(os.environ):
        if k.startswith(("ENDPOINT_", "GROUP_")):
            monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # Neutralize dotenv so a real .env file doesn't leak ENDPOINT_/GROUP_ vars into the test
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("config", "main"):
        sys.modules.pop(mod, None)
    import config
    return importlib.reload(config)


@pytest.fixture
def make_config(monkeypatch):
    return lambda env: fresh_config(monkeypatch, env)
