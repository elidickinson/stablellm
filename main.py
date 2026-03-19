import hashlib
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import API_KEY, COOLOFF_SECONDS, CONNECT_TIMEOUT, ENDPOINTS, HOST, PORT, REQUEST_TIMEOUT, Endpoint

logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
log = logging.getLogger("stablellm")

# endpoint index -> timestamp when it becomes available again
_cooloff_until: dict[int, float] = {}

# endpoint index -> stats
_stats = {
    "requests": defaultdict(int),
    "successes": defaultdict(int),
    "failures": defaultdict(int),
}

http_client: httpx.AsyncClient

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT))
    log.info("stablellm started with %d endpoint(s)", len(ENDPOINTS))
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


def _is_available(idx: int) -> bool:
    return time.monotonic() >= _cooloff_until.get(idx, 0)


def _mark_down(idx: int, reason: str):
    _cooloff_until[idx] = time.monotonic() + COOLOFF_SECONDS
    _stats["failures"][idx] += 1
    ep = ENDPOINTS[idx]
    log.warning("endpoint %s marked down for %ss: %s", ep.base_url, COOLOFF_SECONDS, reason)


def _check_auth(authorization: str | None) -> JSONResponse | None:
    if not API_KEY:
        return None
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {API_KEY}"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


async def _proxy_stream(ep: Endpoint, path: str, headers: dict, body: bytes):
    """Stream response from upstream. Returns (StreamingResponse, None) or (None, status_code)."""
    url = f"{ep.base_url}/{path}"
    req = http_client.build_request("POST", url, headers=headers, content=body)
    resp = await http_client.send(req, stream=True)

    if resp.status_code in RETRYABLE_STATUSES:
        await resp.aclose()
        return None, resp.status_code

    async def generate():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    # Forward upstream headers, excluding hop-by-hop and encoding headers
    excluded = {"transfer-encoding", "connection", "keep-alive", "content-encoding", "content-length"}
    forward_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}

    return StreamingResponse(
        generate(),
        status_code=resp.status_code,
        headers=forward_headers,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    ), None


async def _proxy_buffered(ep: Endpoint, method: str, path: str, headers: dict, body: bytes):
    """Non-streaming: send request, return full response or (None, status_code)."""
    url = f"{ep.base_url}/{path}"
    t0 = time.monotonic()
    resp = await http_client.request(method, url, headers=headers, content=body)
    elapsed = time.monotonic() - t0

    if resp.status_code in RETRYABLE_STATUSES:
        return None, resp.status_code

    try:
        data = resp.json()
    except Exception:
        return JSONResponse({"error": "upstream returned invalid JSON"}, status_code=502)

    log.info("%s TTFB %.0fms", ep.base_url, elapsed * 1000)

    return JSONResponse(content=data, status_code=resp.status_code), None


def _build_upstream_headers(ep: Endpoint) -> dict:
    return {
        "Authorization": f"Bearer {ep.api_key}",
        "Content-Type": "application/json",
    }


# Only pass these known-supported params to Cerebras
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "stream",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "stop",
    "seed",
    "logprobs",
    "top_logprobs",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "prediction",
    "user",
    "reasoning_effort",
    "clear_thinking",
    "disable_reasoning",
}


def _rewrite_model(body: dict, ep: Endpoint) -> dict:
    if ep.model_override:
        body = {**body, "model": ep.model_override}
    return body


def _strip_unsupported(body: dict, ep: Endpoint) -> dict:
    """Keep only supported params and apply model override."""
    base = {k: v for k, v in body.items() if k in SUPPORTED_PARAMS}
    return _rewrite_model(base, ep)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    result = {"endpoints": []}
    for idx, ep in enumerate(ENDPOINTS):
        result["endpoints"].append({
            "index": idx,
            "base_url": ep.base_url,
            "model_override": ep.model_override or "(none)",
            "requests": _stats["requests"].get(idx, 0),
            "successes": _stats["successes"].get(idx, 0),
            "failures": _stats["failures"].get(idx, 0),
        })
    return result


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(request: Request, path: str, authorization: str | None = Header(None)):
    auth_err = _check_auth(authorization)
    if auth_err:
        return auth_err

    raw_body = await request.body()
    is_streaming = False

    # Parse body for POST to detect streaming and rewrite model
    if request.method == "POST" and raw_body:
        try:
            body_dict = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        is_streaming = body_dict.get("stream", False)
    else:
        body_dict = None

    last_failure = None
    for idx, ep in enumerate(ENDPOINTS):
        if not _is_available(idx):
            log.info("skipping %s (cooling off)", ep.base_url)
            continue

        _stats["requests"][idx] += 1

        if body_dict is not None:
            stripped = _strip_unsupported(body_dict, ep)
            send_body = json.dumps(stripped).encode()
            log.debug("-> %s body (keys): %s", ep.base_url, list(stripped.keys()))
        else:
            send_body = raw_body

        headers = _build_upstream_headers(ep)

        try:
            if is_streaming:
                result, status = await _proxy_stream(ep, path, headers, send_body)
            else:
                result, status = await _proxy_buffered(ep, request.method, path, headers, send_body)

            if result is not None:
                _stats["successes"][idx] += 1
                return result

            _mark_down(idx, f"HTTP {status}")
            last_failure = f"HTTP {status}"

        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            _mark_down(idx, str(exc))
            last_failure = type(exc).__name__

    log.error("all endpoints exhausted for /v1/%s (last: %s)", path, last_failure)
    return JSONResponse(
        {"error": f"all endpoints exhausted (last: {last_failure})"},
        status_code=502,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT)
