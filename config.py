import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str
    model_override: str  # empty string means pass through client's model
    keep_reasoning: bool = False  # if True, preserve reasoning fields in messages


def _parse_endpoints() -> tuple[list[Endpoint], dict[str, int]]:
    endpoints = []
    name_to_idx: dict[str, int] = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith("ENDPOINT_"):
            continue
        parts = value.split("|")
        if len(parts) < 2:
            print(f"WARNING: {key} malformed, expected 'base_url|api_key[|model_override[|flags]]'", file=sys.stderr)
            continue
        base_url = parts[0].rstrip("/")
        api_key = parts[1]
        model_override = parts[2] if len(parts) > 2 else ""
        flags = {f.strip() for f in parts[3].split(",")} if len(parts) > 3 else set()
        name = key.removeprefix("ENDPOINT_").lower()
        idx = len(endpoints)
        name_to_idx[name] = idx
        endpoints.append(Endpoint(
            base_url=base_url, api_key=api_key, model_override=model_override,
            keep_reasoning="keep_reasoning" in flags,
        ))
    return endpoints, name_to_idx


ENDPOINTS, ENDPOINT_NAMES = _parse_endpoints()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "4000"))
API_KEY = os.getenv("API_KEY", "")
COOLOFF_SECONDS = float(os.getenv("COOLOFF_SECONDS", "30"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT", "4"))

if not ENDPOINTS:
    print("FATAL: No ENDPOINT_* variables found. See .env.example.", file=sys.stderr)
    sys.exit(1)


def _parse_groups() -> dict[str, list[int]]:
    """Parse GROUP_<name>=<comma-separated endpoint names> from env.

    Example: GROUP_CHEAP=cerebras1,openai maps model name "cheap" to
    ENDPOINT_CEREBRAS1 and ENDPOINT_OPENAI (in that order).
    """
    groups: dict[str, list[int]] = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith("GROUP_"):
            continue
        name = key.removeprefix("GROUP_").lower()
        indices = []
        for part in value.split(","):
            ep_name = part.strip().lower()
            if not ep_name:
                continue
            if ep_name not in ENDPOINT_NAMES:
                print(f"FATAL: {key} references endpoint '{part.strip()}' which does not exist. "
                      f"Available endpoints: {', '.join(n.upper() for n in sorted(ENDPOINT_NAMES))}",
                      file=sys.stderr)
                sys.exit(1)
            ep_idx = ENDPOINT_NAMES[ep_name]
            if not ENDPOINTS[ep_idx].model_override:
                print(f"WARNING: {key} includes ENDPOINT_{ep_name.upper()} which has no model_override set",
                      file=sys.stderr)
            indices.append(ep_idx)
        if indices:
            groups[name] = indices
        else:
            print(f"WARNING: {key} resolved to no valid endpoints", file=sys.stderr)
    return groups


GROUPS = _parse_groups()
