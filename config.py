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
    model_override: str  # empty string means pass through client's model
    keep_reasoning: bool = False  # if True, preserve reasoning fields in messages


class ConfigError(Exception):
    pass


# Server settings (env-only, not reloadable)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4000"))
API_KEY = os.getenv("API_KEY", "")
COOLOFF_SECONDS = float(os.getenv("COOLOFF_SECONDS", "30"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "4"))
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
CONFIG_EDITOR_PASSWORD = os.getenv("CONFIG_EDITOR_PASSWORD", "")

# Reloadable state
ENDPOINTS: list[Endpoint] = []
ENDPOINT_NAMES: dict[str, int] = {}
GROUPS: dict[str, list[int]] = {}


def _env_substitute(value: str) -> str:
    """Replace ${VAR} or $VAR references with environment variable values."""
    def _replace(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        result = os.environ.get(var_name)
        if result is None:
            raise ConfigError(f"environment variable '{var_name}' is not set")
        return result
    return re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, value)


def parse_config(raw: object) -> tuple[list[Endpoint], dict[str, int], dict[str, list[int]]]:
    """Validate and parse a yaml dict into endpoints + groups. Raises ConfigError."""
    if not isinstance(raw, dict) or "endpoints" not in raw:
        raise ConfigError("config must have an 'endpoints' mapping")

    raw_endpoints = raw["endpoints"]
    if not isinstance(raw_endpoints, dict) or not raw_endpoints:
        raise ConfigError("'endpoints' must be a non-empty mapping")

    endpoints: list[Endpoint] = []
    name_to_idx: dict[str, int] = {}
    for name, ep_conf in raw_endpoints.items():
        if not isinstance(ep_conf, dict):
            raise ConfigError(f"endpoint '{name}' must be a mapping")
        if "base_url" not in ep_conf or "api_key" not in ep_conf:
            raise ConfigError(f"endpoint '{name}' requires 'base_url' and 'api_key'")

        name_lower = str(name).lower()
        if name_lower in name_to_idx:
            raise ConfigError(f"duplicate endpoint name '{name}'")

        base_url = str(ep_conf["base_url"]).rstrip("/")
        api_key = _env_substitute(str(ep_conf["api_key"]))
        model_override = str(ep_conf.get("model", ""))
        flags = ep_conf.get("flags", [])
        if isinstance(flags, str):
            flags = [f.strip() for f in flags.split(",")]

        name_to_idx[name_lower] = len(endpoints)
        endpoints.append(Endpoint(
            base_url=base_url,
            api_key=api_key,
            model_override=model_override,
            keep_reasoning="keep_reasoning" in flags,
        ))

    groups: dict[str, list[int]] = {}
    raw_groups = raw.get("groups", {}) or {}
    if not isinstance(raw_groups, dict):
        raise ConfigError("'groups' must be a mapping")

    for group_name, members in raw_groups.items():
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
        groups[str(group_name).lower()] = indices

    if "default" not in groups:
        groups["default"] = list(range(len(endpoints)))

    return endpoints, name_to_idx, groups


def load_from_file(path: str | None = None) -> tuple[list[Endpoint], dict[str, int], dict[str, list[int]]]:
    """Read and parse the config file. Raises ConfigError."""
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
    """Reload config from disk into module state. Raises ConfigError."""
    global ENDPOINTS, ENDPOINT_NAMES, GROUPS
    ENDPOINTS, ENDPOINT_NAMES, GROUPS = load_from_file()


# Initial load at import time — fatal on failure
try:
    reload()
except ConfigError as exc:
    print(f"FATAL: {exc}", file=sys.stderr)
    sys.exit(1)
