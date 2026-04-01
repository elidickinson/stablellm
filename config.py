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


def _parse_endpoints() -> list[Endpoint]:
    endpoints = []
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
        endpoints.append(Endpoint(
            base_url=base_url, api_key=api_key, model_override=model_override,
            keep_reasoning="keep_reasoning" in flags,
        ))
    return endpoints


ENDPOINTS = _parse_endpoints()
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
    """Parse GROUP_<name>=<comma-separated endpoint numbers> from env.

    Example: GROUP_CHEAP=1,3,5 maps model name "cheap" to ENDPOINT_1, ENDPOINT_3, ENDPOINT_5.
    Endpoint numbers correspond to ENDPOINT_N suffixes (1-based), stored as 0-based indices.
    """
    # Build mapping from ENDPOINT_N suffix -> 0-based index
    endpoint_num_to_idx: dict[int, int] = {}
    idx = 0
    for key in sorted(os.environ.keys()):
        if key.startswith("ENDPOINT_"):
            try:
                num = int(key.removeprefix("ENDPOINT_"))
            except ValueError:
                continue
            endpoint_num_to_idx[num] = idx
            idx += 1

    groups: dict[str, list[int]] = {}
    for key, value in sorted(os.environ.items()):
        if not key.startswith("GROUP_"):
            continue
        name = key.removeprefix("GROUP_").lower()
        indices = []
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                num = int(part)
            except ValueError:
                print(f"WARNING: {key} contains non-integer '{part}'", file=sys.stderr)
                continue
            if num not in endpoint_num_to_idx:
                print(f"WARNING: {key} references ENDPOINT_{num} which does not exist", file=sys.stderr)
                continue
            ep_idx = endpoint_num_to_idx[num]
            if not ENDPOINTS[ep_idx].model_override:
                print(f"WARNING: {key} includes ENDPOINT_{num} which has no model_override set", file=sys.stderr)
            indices.append(ep_idx)
        if indices:
            groups[name] = indices
        else:
            print(f"WARNING: {key} resolved to no valid endpoints", file=sys.stderr)
    return groups


GROUPS = _parse_groups()
