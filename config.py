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


def _env_substitute(value: str) -> str:
    """Replace ${VAR} or $VAR references with environment variable values."""
    def _replace(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        result = os.environ.get(var_name)
        if result is None:
            print(f"FATAL: environment variable '{var_name}' is not set", file=sys.stderr)
            sys.exit(1)
        return result
    return re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, value)


def _load_config() -> tuple[list[Endpoint], dict[str, int], dict[str, list[int]]]:
    config_path = os.getenv("CONFIG_FILE", "config.yaml")
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: config file '{config_path}' not found. Set CONFIG_FILE env var or create config.yaml.",
              file=sys.stderr)
        sys.exit(1)

    if not isinstance(raw, dict) or "endpoints" not in raw:
        print(f"FATAL: config file must have an 'endpoints' mapping", file=sys.stderr)
        sys.exit(1)

    # Parse endpoints
    endpoints: list[Endpoint] = []
    name_to_idx: dict[str, int] = {}
    raw_endpoints = raw["endpoints"]
    if not isinstance(raw_endpoints, dict) or not raw_endpoints:
        print("FATAL: 'endpoints' must be a non-empty mapping", file=sys.stderr)
        sys.exit(1)

    for name, ep_conf in raw_endpoints.items():
        name_lower = str(name).lower()
        if not isinstance(ep_conf, dict):
            print(f"FATAL: endpoint '{name}' must be a mapping", file=sys.stderr)
            sys.exit(1)
        if "base_url" not in ep_conf or "api_key" not in ep_conf:
            print(f"FATAL: endpoint '{name}' requires 'base_url' and 'api_key'", file=sys.stderr)
            sys.exit(1)

        base_url = str(ep_conf["base_url"]).rstrip("/")
        api_key = _env_substitute(str(ep_conf["api_key"]))
        model_override = str(ep_conf.get("model", ""))
        flags = ep_conf.get("flags", [])
        if isinstance(flags, str):
            flags = [f.strip() for f in flags.split(",")]

        idx = len(endpoints)
        name_to_idx[name_lower] = idx
        endpoints.append(Endpoint(
            base_url=base_url,
            api_key=api_key,
            model_override=model_override,
            keep_reasoning="keep_reasoning" in flags,
        ))

    # Parse groups
    groups: dict[str, list[int]] = {}
    raw_groups = raw.get("groups", {})
    if raw_groups:
        if not isinstance(raw_groups, dict):
            print("FATAL: 'groups' must be a mapping", file=sys.stderr)
            sys.exit(1)

        for group_name, members in raw_groups.items():
            group_lower = str(group_name).lower()
            if not isinstance(members, list) or not members:
                print(f"FATAL: group '{group_name}' must be a non-empty list", file=sys.stderr)
                sys.exit(1)

            indices = []
            for member in members:
                ep_name = str(member).strip().lower()
                if ep_name not in name_to_idx:
                    print(f"FATAL: group '{group_name}' references endpoint '{member}' which does not exist. "
                          f"Available: {', '.join(sorted(name_to_idx))}",
                          file=sys.stderr)
                    sys.exit(1)
                ep_idx = name_to_idx[ep_name]
                if group_lower != "default" and not endpoints[ep_idx].model_override:
                    print(f"WARNING: group '{group_name}' includes endpoint '{member}' which has no model set",
                          file=sys.stderr)
                indices.append(ep_idx)
            groups[group_lower] = indices

    # Implicit default group if not defined
    if "default" not in groups:
        groups["default"] = list(range(len(endpoints)))

    return endpoints, name_to_idx, groups


ENDPOINTS, ENDPOINT_NAMES, GROUPS = _load_config()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4000"))
API_KEY = os.getenv("API_KEY", "")
COOLOFF_SECONDS = float(os.getenv("COOLOFF_SECONDS", "30"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "4"))
