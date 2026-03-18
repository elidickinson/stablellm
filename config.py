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


def _parse_endpoints() -> list[Endpoint]:
    endpoints = []
    for key, value in sorted(os.environ.items()):
        if not key.startswith("ENDPOINT_"):
            continue
        parts = value.split("|")
        if len(parts) < 2:
            print(f"WARNING: {key} malformed, expected 'base_url|api_key[|model_override]'", file=sys.stderr)
            continue
        base_url = parts[0].rstrip("/")
        api_key = parts[1]
        model_override = parts[2] if len(parts) > 2 else ""
        endpoints.append(Endpoint(base_url=base_url, api_key=api_key, model_override=model_override))
    return endpoints


ENDPOINTS = _parse_endpoints()
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))
API_KEY = os.getenv("API_KEY", "")
COOLOFF_SECONDS = float(os.getenv("COOLOFF_SECONDS", "30"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))

if not ENDPOINTS:
    print("FATAL: No ENDPOINT_* variables found. See .env.example.", file=sys.stderr)
    sys.exit(1)
