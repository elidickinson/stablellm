import logging
import os
import re
import sys
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("stablellm.config")


@dataclass(frozen=True)
class Provider:
    """A raw upstream provider: base_url + api_key + optional default model."""
    base_url: str
    api_key: str
    model: str = ""  # default model when group entry omits it


@dataclass(frozen=True)
class Endpoint:
    """A concrete (provider + model) tuple. Indexed for routing."""
    base_url: str
    api_key: str
    model: str = ""  # empty → pass through client's model
    keep_reasoning: bool = False


@dataclass(frozen=True)
class Settings:
    cooloff_seconds: float = 30.0
    race_interval_secs: int = 6 * 3600
    race_interval_requests: int = 25
    log_level: str = "INFO"


class ConfigError(Exception):
    pass


# --- Bind-time settings ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "4"))
API_KEY = os.getenv("API_KEY", "")
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
CONFIG_EDITOR_PASSWORD = os.getenv("CONFIG_EDITOR_PASSWORD", "")
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(50 * 1024 * 1024)))

# --- Reloadable state (updated atomically by apply_config) ---
ENDPOINTS: list[Endpoint] = []
GROUPS: dict[str, list[int]] = {}
SETTINGS: Settings = Settings()

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _env_substitute(value: str) -> str:
    def _replace(m: re.Match) -> str:
        var = m.group(1) or m.group(2)
        result = os.environ.get(var)
        if result is None:
            log.warning("environment variable '%s' is not set", var)
            return m.group(0)
        return result

    return _ENV_VAR_RE.sub(_replace, value)


def _parse_providers(raw: object) -> dict[str, Provider]:
    """Parse top-level 'providers' mapping. Returns {name_lower: Provider}."""
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("'providers' must be a non-empty mapping")

    providers: dict[str, Provider] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"provider '{name}' must be a mapping")
        if "base_url" not in entry or "api_key" not in entry:
            raise ConfigError(f"provider '{name}' requires 'base_url' and 'api_key'")

        name_lower = str(name).lower()
        if name_lower in providers:
            raise ConfigError(f"duplicate provider name '{name}'")

        providers[name_lower] = Provider(
            base_url=str(entry["base_url"]).rstrip("/"),
            api_key=_env_substitute(str(entry["api_key"])),
            model=str(entry.get("model", "")),
        )
    return providers


def _parse_groups(raw: object, providers: dict[str, Provider]) -> tuple[dict[str, list[int]], list[Endpoint]]:
    """Parse 'groups' mapping. Returns (groups, endpoints).

    Each group name is used as-is (no normalization) and matched directly
    against the request model. Endpoints are built flat and indexed.
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("'groups' must be a mapping")

    groups: dict[str, list[int]] = {}
    endpoints: list[Endpoint] = []

    for group_name, members in raw.items():
        if not isinstance(members, list) or not members:
            raise ConfigError(f"group '{group_name}' must be a non-empty list")

        indices: list[int] = []
        for i, entry in enumerate(members):
            if not isinstance(entry, dict):
                raise ConfigError(f"entry {i} in group '{group_name}' must be a mapping")

            prov_name = entry.get("provider", "")
            if not prov_name:
                raise ConfigError(f"entry {i} in group '{group_name}' is missing 'provider'")

            prov_lower = str(prov_name).lower()
            if prov_lower not in providers:
                raise ConfigError(
                    f"group '{group_name}' references unknown provider '{prov_name}'. "
                    f"Available: {', '.join(sorted(providers))}"
                )

            prov = providers[prov_lower]
            flags = entry.get("flags", [])
            if not isinstance(flags, list):
                raise ConfigError(f"entry {i} in group '{group_name}': 'flags' must be a list")

            endpoints.append(Endpoint(
                base_url=prov.base_url,
                api_key=prov.api_key,
                model=str(entry.get("model", prov.model)),
                keep_reasoning="keep_reasoning" in flags,
            ))
            indices.append(len(endpoints) - 1)

        groups[str(group_name)] = indices

    if not groups:
        raise ConfigError("at least one group is required")

    return groups, endpoints


def _parse_settings(raw: object) -> Settings:
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


def parse_config(raw: object) -> tuple[list[Endpoint], dict[str, list[int]], Settings]:
    """Validate and parse a yaml dict. Returns (endpoints, groups, settings).
    Raises ConfigError on invalid input. Does NOT mutate module state."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")

    providers = _parse_providers(raw.get("providers", {}))
    groups, endpoints = _parse_groups(raw.get("groups"), providers)
    settings = _parse_settings(raw.get("settings"))
    return endpoints, groups, settings


def apply_config(endpoints: list[Endpoint], groups: dict[str, list[int]], settings: Settings) -> None:
    """Atomically swap in new config state."""
    global ENDPOINTS, GROUPS, SETTINGS
    ENDPOINTS = endpoints
    GROUPS = groups
    SETTINGS = settings


def load_from_file(path: str | None = None) -> None:
    """Load config from disk and apply it. Raises ConfigError."""
    path = path or CONFIG_FILE
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file '{path}' not found")
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}")
    endpoints, groups, settings = parse_config(raw)
    apply_config(endpoints, groups, settings)


def reload() -> None:
    """Reload config from disk. Raises ConfigError on bad input."""
    load_from_file()


def load_or_exit() -> None:
    """Initial load with hard exit on failure."""
    try:
        reload()
    except ConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
