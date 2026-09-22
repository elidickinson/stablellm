import hashlib
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv

from performance_routing import (
    _QUANT_FLOOR_AUTO,
    _QUANT_TIER_LABELS,
    _QUANT_TIERS,
    PerformanceRouting,
)

load_dotenv()

log = logging.getLogger("stablellm.config")


@dataclass(frozen=True)
class Provider:
    """A raw upstream provider: base_url + api_key + optional default model."""
    base_url: str
    api_key: str
    model: str = ""  # default model when group entry omits it
    max_concurrency: int = 0  # per-(model) in-flight cap; 0 = unlimited
    ttfb_deadline_secs: float = 0.0  # fail over if response headers don't arrive; 0 = disabled
    routing: dict | None = None  # passthrough sent as OpenRouter's `provider` body object


@dataclass(frozen=True)
class Endpoint:
    """A concrete (provider + model) tuple. Indexed for routing."""
    base_url: str
    api_key: str
    model: str = ""  # empty → pass through client's model
    keep_reasoning: bool = False
    provider: str = ""
    max_concurrency: int = 0
    ttfb_deadline_secs: float = 0.0
    routing: dict | None = None
    performance_routing: PerformanceRouting | None = None
    reasoning_effort: str = ""  # sent when the request omits one; "" = leave client's choice
    reasoning_force: bool = False  # override the client's reasoning params with reasoning_effort


@dataclass(frozen=True)
class ModelMeta:
    """Descriptive per-group metadata, surfaced on /v1/models.

    All fields are optional. Costs are dollars per million tokens in config and
    serialized as OpenRouter's per-token strings.
    """
    name: str = ""
    description: str = ""
    context: int | None = None
    max_output: int | None = None
    modalities: tuple[str, ...] = ()
    input_cost: float | None = None
    output_cost: float | None = None
    cache_read_cost: float | None = None
    cache_write_cost: float | None = None
    supports_reasoning: bool = False
    reasoning_mandatory: bool = False
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_default: str = ""
    reasoning_default_enabled: bool | None = None


MODE_SEQ = "seq"
MODE_RACE = "race"
VALID_MODES = frozenset({MODE_SEQ, MODE_RACE})


@dataclass(frozen=True)
class Group:
    """An ordered set of endpoint indices and the routing mode used to dispatch them."""
    endpoints: list[int]
    mode: str = MODE_SEQ
    meta: ModelMeta | None = None


@dataclass(frozen=True)
class Settings:
    cooloff_seconds: float = 30.0
    race_interval_secs: int = 6 * 3600
    race_interval_requests: int = 25
    race_settle_timeout_secs: float = 120.0
    session_pin_ttl_secs: float = 15 * 60
    log_level: str = "INFO"


class ConfigError(Exception):
    pass


def _parse_api_keys(raw: str) -> dict[str, str]:
    """Parse API_KEY ('name:secret,secret,...') into {sha256(secret): client name}.

    Names are optional and only used for logging; unnamed keys get a stable id
    derived from their own hash so they stay distinct in the request log.
    """
    keys: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, secret = entry.partition(":")
        if not sep:
            name, secret = "", name
        if not secret:
            raise ConfigError(f"API_KEY entry '{entry}' has a name but no key")
        digest = hashlib.sha256(secret.encode()).hexdigest()
        keys[digest] = name or f"key-{digest[:6]}"
    return keys


# --- Bind-time settings ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "4"))
API_KEYS = _parse_api_keys(os.getenv("API_KEY", ""))
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
CONFIG_EDITOR_PASSWORD = os.getenv("CONFIG_EDITOR_PASSWORD", "")
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(50 * 1024 * 1024)))
REQUEST_LOG_DB = os.getenv("REQUEST_LOG_DB", "")

# --- Reloadable state (updated atomically by apply_config) ---
ENDPOINTS: list[Endpoint] = []
GROUPS: dict[str, Group] = {}
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


def _default_openrouter() -> Provider:
    return Provider(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )


def _require_openrouter(value: object, key: str, provider_name: str, where: str) -> None:
    """Reject an OpenRouter-only setting configured on another provider."""
    if value is not None and provider_name != "openrouter":
        raise ConfigError(f"{where}: '{key}' is only valid on the 'openrouter' provider")


def _parse_provider(name: str, entry: object, defaults: Provider | None = None) -> Provider:
    if not isinstance(entry, dict):
        raise ConfigError(f"provider '{name}' must be a mapping")

    if defaults is None:
        if "base_url" not in entry or "api_key" not in entry:
            raise ConfigError(f"provider '{name}' requires 'base_url' and 'api_key'")
        base_url = str(entry["base_url"]).rstrip("/")
        api_key = _env_substitute(str(entry["api_key"]))
        model_default = ""
        max_concurrency_default = 0
        ttfb_deadline_default = 0.0
    else:
        base_url = str(entry.get("base_url", defaults.base_url)).rstrip("/")
        api_key = (
            _env_substitute(str(entry["api_key"]))
            if "api_key" in entry else defaults.api_key
        )
        model_default = defaults.model
        max_concurrency_default = defaults.max_concurrency
        ttfb_deadline_default = defaults.ttfb_deadline_secs

    routing = _opt_mapping(entry.get("routing"), "routing")
    _routing_constraints(routing, f"provider '{name}'")
    _require_openrouter(routing, "routing", name, f"provider '{name}'")
    return Provider(
        base_url=base_url,
        api_key=api_key,
        model=str(entry.get("model", model_default)),
        max_concurrency=_opt_count(entry.get("max_concurrency"), "max_concurrency", max_concurrency_default),
        ttfb_deadline_secs=_opt_secs(entry.get("ttfb_deadline_secs"), "ttfb_deadline_secs", ttfb_deadline_default),
        routing=routing,
    )


def _parse_providers(raw: object) -> dict[str, Provider]:
    """Parse top-level 'providers' mapping. Returns {name_lower: Provider}."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("'providers' must be a mapping")

    providers: dict[str, Provider] = {"openrouter": _default_openrouter()}
    declared: set[str] = set()
    for name, entry in raw.items():
        name_lower = str(name).lower()
        if name_lower in declared:
            raise ConfigError(f"duplicate provider name '{name}'")
        declared.add(name_lower)
        defaults = providers["openrouter"] if name_lower == "openrouter" else None
        providers[name_lower] = _parse_provider(name_lower, entry, defaults)
    return providers


def _opt_count(value: object, key: str, default: int) -> int:
    """Optional non-negative int (0 = unlimited); absent → default."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"'{key}' must be a non-negative integer (0 = unlimited)")
    return value


def _opt_mapping(value: object, key: str) -> dict | None:
    """Optional passthrough mapping (forwarded verbatim upstream); absent → None."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping")
    # yaml can produce types json.dumps rejects (dates, sets); catch at parse
    # time instead of failing every request to the endpoint.
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' must be JSON-serializable: {exc}") from exc
    return value


def _opt_secs(value: object, key: str, default: float) -> float:
    """Optional non-negative number of seconds (0 = disabled); absent → default."""
    if value is None:
        return default
    # isfinite: yaml .nan/.inf pass the < 0 check and would silently break
    # every request to the endpoint (asyncio.wait_for(nan) fires instantly).
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigError(f"'{key}' must be a non-negative number of seconds (0 = disabled)")
    return float(value)


def _opt_performance_routing(value: object) -> PerformanceRouting | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError("'performance_routing' must be a mapping")
    unknown = set(value) - {
        "target_tokens", "speed_tolerance", "cache_ttl_seconds",
        "price_cap_tolerance", "quantization_floor", "include_unknown_quantization",
    }
    if unknown:
        raise ConfigError(f"unknown performance_routing keys: {', '.join(sorted(map(str, unknown)))}")
    defaults = PerformanceRouting()
    target_tokens = _opt_count(value.get("target_tokens"), "performance_routing.target_tokens", defaults.target_tokens)
    if target_tokens == 0:
        raise ConfigError("'performance_routing.target_tokens' must be greater than 0")
    # _opt_secs maps an explicit null to the default, so blank (meaning OFF)
    # is distinguished by key presence, not a value test.
    speed_tolerance = (
        _opt_secs(value["speed_tolerance"], "performance_routing.speed_tolerance", defaults.speed_tolerance)
        if "speed_tolerance" in value else defaults.speed_tolerance
    )
    price_cap_tolerance = defaults.price_cap_tolerance
    if "price_cap_tolerance" in value:
        raw = value["price_cap_tolerance"]
        # An explicit null is off; an absent key takes the default.
        if raw is None:
            price_cap_tolerance = None
        else:
            # 0 is meaningful (cap at the median); only an explicit null disables.
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
                raise ConfigError("'performance_routing.price_cap_tolerance' must be a non-negative number (null = off)")
            price_cap_tolerance = float(raw)
    floor = _QUANT_FLOOR_AUTO
    if "quantization_floor" in value:
        raw_floor = value["quantization_floor"]
        # An explicit null is off; auto (the default) derives the floor.
        if raw_floor is None:
            floor = None
        elif raw_floor != _QUANT_FLOOR_AUTO:
            if isinstance(raw_floor, bool) or not isinstance(raw_floor, int) or raw_floor not in _QUANT_TIERS:
                raise ConfigError(f"'performance_routing.quantization_floor' must be one of: {_QUANT_TIER_LABELS}")
            floor = raw_floor
    include_unknown = True
    if "include_unknown_quantization" in value:
        raw_unknown = value["include_unknown_quantization"]
        if not isinstance(raw_unknown, bool):
            raise ConfigError("'performance_routing.include_unknown_quantization' must be a boolean")
        include_unknown = raw_unknown
    return PerformanceRouting(
        target_tokens,
        speed_tolerance,
        _opt_secs(value.get("cache_ttl_seconds"), "performance_routing.cache_ttl_seconds", defaults.cache_ttl_seconds),
        price_cap_tolerance=price_cap_tolerance,
        quantization_floor=floor,
        include_unknown_quantization=include_unknown,
    )


def _routing_constraints(routing: dict | None, where: str) -> None:
    """Validate the keys performance routing filters on. A malformed value is not
    read as absent: the endpoint would either skip every request or quietly route
    as if the author had written nothing."""
    if routing is None:
        return
    quantizations = routing.get("quantizations")
    if quantizations is not None and (
        not isinstance(quantizations, list) or not quantizations or not all(isinstance(q, str) for q in quantizations)
    ):
        raise ConfigError(f"{where}: 'quantizations' must be a non-empty list of quantization names")
    max_price = routing.get("max_price")
    if max_price is None:
        return
    if not isinstance(max_price, dict) or not max_price:
        raise ConfigError(f"{where}: 'max_price' must be a mapping of price fields to numbers")
    for field, amount in max_price.items():
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount < 0:
            raise ConfigError(f"{where}: 'max_price.{field}' must be a non-negative number")


def _meta_str(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"meta '{key}' must be a string")
    return value


def _meta_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"meta '{key}' must be a boolean")
    return value


def _meta_pos_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"meta '{key}' must be a positive integer")
    return value


def _meta_nonneg_num(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigError(f"meta '{key}' must be a non-negative number")
    return float(value)


def _meta_str_list(value: object, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ConfigError(f"meta '{key}' must be a list of strings")
    return tuple(value)


_META_FIELDS = {
    "name": _meta_str,
    "description": _meta_str,
    "context": _meta_pos_int,
    "max_output": _meta_pos_int,
    "modalities": _meta_str_list,
    "input_cost": _meta_nonneg_num,
    "output_cost": _meta_nonneg_num,
    "cache_read_cost": _meta_nonneg_num,
    "cache_write_cost": _meta_nonneg_num,
    "supports_reasoning": _meta_bool,
    "reasoning_mandatory": _meta_bool,
    "reasoning_efforts": _meta_str_list,
    "reasoning_default": _meta_str,
    "reasoning_default_enabled": _meta_bool,
}


def _parse_meta(raw: object) -> ModelMeta | None:
    """Parse an optional group 'meta' mapping. Returns None when absent/empty."""
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise ConfigError("'meta' must be a mapping")

    unknown = set(raw) - _META_FIELDS.keys()
    if unknown:
        raise ConfigError(f"unknown meta keys: {', '.join(sorted(map(str, unknown)))}")

    fields = {key: _META_FIELDS[key](value, key) for key, value in raw.items()}
    reasoning_keys = {"reasoning_mandatory", "reasoning_efforts", "reasoning_default", "reasoning_default_enabled"}
    if reasoning_keys & fields.keys() and not fields.get("supports_reasoning"):
        raise ConfigError("reasoning fields require 'supports_reasoning: true'")
    default = fields.get("reasoning_default", "")
    efforts = fields.get("reasoning_efforts", ())
    if default and default not in efforts:
        raise ConfigError("'reasoning_default' must be one of 'reasoning_efforts'")
    if fields.get("reasoning_default_enabled") is False and not efforts:
        raise ConfigError("'reasoning_default_enabled: false' requires 'reasoning_efforts'")
    return ModelMeta(**fields)


def _parse_groups(raw: object, providers: dict[str, Provider]) -> tuple[dict[str, Group], list[Endpoint]]:
    """Parse 'groups' mapping. Returns (groups, endpoints).

    Each group is a mapping with required 'endpoints' (a non-empty list) and
    optional 'mode' (defaults to MODE_SEQ). Group names match request models exactly.
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError("'groups' must be a mapping")

    groups: dict[str, Group] = {}
    endpoints: list[Endpoint] = []

    for group_name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"group '{group_name}' must be a mapping with 'endpoints' (and optional 'mode')")

        members = spec.get("endpoints")
        if not isinstance(members, list) or not members:
            raise ConfigError(f"group '{group_name}': 'endpoints' must be a non-empty list")

        mode = str(spec.get("mode", MODE_SEQ))
        if mode not in VALID_MODES:
            raise ConfigError(
                f"group '{group_name}': unknown mode '{mode}'. Valid: {', '.join(sorted(VALID_MODES))}"
            )

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

            reasoning_effort = entry.get("reasoning_effort", "")
            if not isinstance(reasoning_effort, str):
                raise ConfigError(f"entry {i} in group '{group_name}': 'reasoning_effort' must be a string")
            reasoning_force = entry.get("reasoning_force", False)
            if not isinstance(reasoning_force, bool):
                raise ConfigError(f"entry {i} in group '{group_name}': 'reasoning_force' must be a boolean")
            if reasoning_force and not reasoning_effort:
                raise ConfigError(f"entry {i} in group '{group_name}': 'reasoning_force' requires 'reasoning_effort'")

            routing = _opt_mapping(entry.get("routing"), "routing")
            performance_routing = _opt_performance_routing(entry.get("performance_routing"))
            where = f"group '{group_name}' entry {i}"
            _routing_constraints(routing, where)
            _require_openrouter(routing, "routing", prov_lower, where)
            _require_openrouter(performance_routing, "performance_routing", prov_lower, where)
            if routing is None:
                routing = prov.routing

            endpoints.append(Endpoint(
                base_url=prov.base_url,
                api_key=prov.api_key,
                model=str(entry.get("model", prov.model)),
                keep_reasoning="keep_reasoning" in flags,
                provider=prov_lower,
                max_concurrency=_opt_count(entry.get("max_concurrency"), "max_concurrency", prov.max_concurrency),
                ttfb_deadline_secs=_opt_secs(entry.get("ttfb_deadline_secs"), "ttfb_deadline_secs", prov.ttfb_deadline_secs),
                routing=routing,
                performance_routing=performance_routing,
                reasoning_effort=reasoning_effort,
                reasoning_force=reasoning_force,
            ))
            indices.append(len(endpoints) - 1)

        meta = _parse_meta(spec.get("meta"))

        name_lower = str(group_name).lower()
        if name_lower in groups:
            raise ConfigError(f"duplicate group name (case-insensitive): '{group_name}'")
        groups[name_lower] = Group(endpoints=indices, mode=mode, meta=meta)

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
    cooloff = float(raw.get("cooloff_seconds", defaults.cooloff_seconds))
    if not math.isfinite(cooloff) or cooloff < 0:
        raise ConfigError("'cooloff_seconds' must be a non-negative number of seconds")
    race_settle_timeout = _opt_secs(
        raw.get("race_settle_timeout_secs"),
        "race_settle_timeout_secs",
        defaults.race_settle_timeout_secs,
    )
    if race_settle_timeout == 0:
        raise ConfigError("'race_settle_timeout_secs' must be greater than 0")
    pin_ttl = float(raw.get("session_pin_ttl_secs", defaults.session_pin_ttl_secs))
    if not math.isfinite(pin_ttl) or pin_ttl < 0:
        raise ConfigError("'session_pin_ttl_secs' must be a non-negative number of seconds")
    return Settings(
        cooloff_seconds=cooloff,
        race_interval_secs=int(raw.get("race_interval_secs", defaults.race_interval_secs)),
        race_interval_requests=int(raw.get("race_interval_requests", defaults.race_interval_requests)),
        race_settle_timeout_secs=race_settle_timeout,
        session_pin_ttl_secs=pin_ttl,
        log_level=str(raw.get("log_level", defaults.log_level)).upper(),
    )


def parse_config(raw: object) -> tuple[list[Endpoint], dict[str, Group], Settings]:
    """Validate and parse a yaml dict. Returns (endpoints, groups, settings).
    Raises ConfigError on invalid input. Does NOT mutate module state."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")

    providers = _parse_providers(raw.get("providers", {}))
    groups, endpoints = _parse_groups(raw.get("groups"), providers)
    settings = _parse_settings(raw.get("settings"))
    if any(ep.provider == "openrouter" and not ep.api_key for ep in endpoints):
        log.warning("provider 'openrouter' has no api_key: set OPENROUTER_API_KEY or declare it in 'providers'")
    return endpoints, groups, settings


def apply_config(endpoints: list[Endpoint], groups: dict[str, Group], settings: Settings) -> None:
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
