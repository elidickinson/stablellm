import os
import re
import sys
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str
    model: str = ""  # empty: defaults to client's requested model
    keep_reasoning: bool = False


@dataclass(frozen=True)
class Settings:
    """Reloadable runtime tunables loaded from config.yaml."""
    cooloff_seconds: float = 30.0
    race_interval_secs: int = 6 * 3600
    race_interval_requests: int = 25
    log_level: str = "INFO"


class ConfigError(Exception):
    pass


# --- Bind-time settings (env-only, require restart) ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "4"))
API_KEY = os.getenv("API_KEY", "")
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
CONFIG_EDITOR_PASSWORD = os.getenv("CONFIG_EDITOR_PASSWORD", "")

# --- Reloadable state (sourced from CONFIG_FILE) ---
ENDPOINTS: list[Endpoint] = []
ENDPOINT_NAMES: dict[str, int] = {}
GROUPS: dict[str, list[int]] = {}
SETTINGS: Settings = Settings()


def normalize_group_name(name: str) -> str:
    """Group lookup is case-insensitive and treats - . _ as equivalent.

    Lets YAML group keys (which usually use underscores) match request model
    IDs that commonly contain dashes or dots (e.g. "gpt-4.1" → "gpt_4_1").
    """
    return name.lower().replace("-", "_").replace(".", "_")


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _env_substitute(value: str) -> str:
    def _replace(m: re.Match) -> str:
        var = m.group(1) or m.group(2)
        result = os.environ.get(var)
        if result is None:
            raise ConfigError(f"environment variable '{var}' is not set")
        return result
    return _ENV_VAR_RE.sub(_replace, value)


def _parse_endpoints(raw: dict) -> tuple[list[Endpoint], dict[str, int]]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("'endpoints' must be a non-empty mapping")

    endpoints: list[Endpoint] = []
    name_to_idx: dict[str, int] = {}
    for name, ep in raw.items():
        if not isinstance(ep, dict):
            raise ConfigError(f"endpoint '{name}' must be a mapping")
        if "base_url" not in ep or "api_key" not in ep:
            raise ConfigError(f"endpoint '{name}' requires 'base_url' and 'api_key'")

        name_lower = str(name).lower()
        if name_lower in name_to_idx:
            raise ConfigError(f"duplicate endpoint name '{name}'")

        flags = ep.get("flags", [])
        if isinstance(flags, str):
            flags = [f.strip() for f in flags.split(",")]

        name_to_idx[name_lower] = len(endpoints)
        endpoints.append(Endpoint(
            base_url=str(ep["base_url"]).rstrip("/"),
            api_key=_env_substitute(str(ep["api_key"])),
            model=str(ep.get("model", "")),
            keep_reasoning="keep_reasoning" in flags,
        ))
    return endpoints, name_to_idx


def _parse_groups(raw: dict | None, name_to_idx: dict[str, int], n_endpoints: int) -> dict[str, list[int]]:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("'groups' must be a mapping")

    groups: dict[str, list[int]] = {}
    for group_name, members in raw.items():
        if not isinstance(members, list) or not members:
            raise ConfigError(f"group '{group_name}' must be a non-empty list")
        indices = []
        for member in members:
            ep_name = str(member).strip().lower()
            if ep_name not in name_to_idx:
                raise ConfigError(
                    f"group '{group_name}' references endpoint '{member}' which does not exist. "
                    f"Available: {', '.join(sorted(name_to_idx))}"
                )
            indices.append(name_to_idx[ep_name])
        groups[normalize_group_name(str(group_name))] = indices

    if "default" not in groups:
        groups["default"] = list(range(n_endpoints))
    return groups


def _parse_settings(raw: dict | None) -> Settings:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("'settings' must be a mapping")
    defaults = Settings()
    known = {f.name for f in defaults.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown settings keys: {', '.join(sorted(unknown))}")
    return Settings(
        cooloff_seconds=float(raw.get("cooloff_seconds", defaults.cooloff_seconds)),
        race_interval_secs=int(raw.get("race_interval_secs", defaults.race_interval_secs)),
        race_interval_requests=int(raw.get("race_interval_requests", defaults.race_interval_requests)),
        log_level=str(raw.get("log_level", defaults.log_level)).upper(),
    )


def parse_config(raw: object) -> tuple[list[Endpoint], dict[str, int], dict[str, list[int]], Settings]:
    """Validate and parse a yaml dict. Raises ConfigError."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    if "endpoints" not in raw:
        raise ConfigError("config must define 'endpoints'")
    endpoints, name_to_idx = _parse_endpoints(raw["endpoints"])
    groups = _parse_groups(raw.get("groups"), name_to_idx, len(endpoints))
    settings = _parse_settings(raw.get("settings"))
    return endpoints, name_to_idx, groups, settings


def load_from_file(path: str | None = None) -> tuple[list[Endpoint], dict[str, int], dict[str, list[int]], Settings]:
    path = path or CONFIG_FILE
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file '{path}' not found")
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}")
    return parse_config(raw)


def reload() -> None:
    """Reload config from disk into module state. Raises ConfigError on bad input."""
    global ENDPOINTS, ENDPOINT_NAMES, GROUPS, SETTINGS
    ENDPOINTS, ENDPOINT_NAMES, GROUPS, SETTINGS = load_from_file()


# Initial load — fatal on failure
try:
    reload()
except ConfigError as exc:
    print(f"FATAL: {exc}", file=sys.stderr)
    sys.exit(1)
