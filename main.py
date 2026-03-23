import asyncio
import hashlib
import hmac
import json
import logging
import os
import posixpath
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
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
_EXCLUDED_HEADERS = {"transfer-encoding", "connection", "keep-alive", "content-encoding", "content-length"}

# Fastest mode: race providers to find the fastest
RACE_INTERVAL_SECS = 6 * 3600
RACE_INTERVAL_REQUESTS = 15
_provider_groups: dict[tuple[str, str], list[int]] = {}  # (model, base_url) -> endpoint indices
_preferred_providers: list[tuple[str, str]] = []
_race_request_count = 0
_last_race_time = 0.0
_background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks


def _build_provider_groups():
    global _provider_groups, _preferred_providers
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, ep in enumerate(ENDPOINTS):
        key = (ep.model_override or "default", ep.base_url)
        groups.setdefault(key, []).append(idx)
    _provider_groups = groups
    _preferred_providers = list(groups.keys())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT))
    _build_provider_groups()
    log.info("stablellm started with %d endpoint(s), %d provider group(s)", len(ENDPOINTS), len(_provider_groups))
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    try:
        if resp.status_code in RETRYABLE_STATUSES:
            await resp.aclose()
            return None, resp.status_code

        async def generate():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()

        return _streaming_response(resp, generate()), None
    except BaseException:
        await resp.aclose()
        raise


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


def _streaming_response(resp: httpx.Response, generator) -> StreamingResponse:
    forward_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _EXCLUDED_HEADERS}
    return StreamingResponse(
        generator,
        status_code=resp.status_code,
        headers=forward_headers,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


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


def _should_race() -> bool:
    if _last_race_time == 0.0:
        return True
    if _race_request_count >= RACE_INTERVAL_REQUESTS:
        return True
    if time.monotonic() - _last_race_time >= RACE_INTERVAL_SECS:
        return True
    return False


def _finish_race(race_times: dict[tuple[str, str], float]):
    global _preferred_providers
    sorted_keys = sorted(race_times, key=race_times.get)
    new_order = list(sorted_keys)
    for k in _provider_groups:
        if k not in race_times:
            new_order.append(k)
    _preferred_providers = new_order
    log.info("race complete: %s", [(k[1], f"{v:.1f}s") for k, v in sorted(race_times.items(), key=lambda x: x[1])])


async def _race_request(path: str, body_dict: dict, is_streaming: bool):
    """Race one endpoint per provider group with real request. Returns response or None."""
    global _race_request_count, _last_race_time

    # One available endpoint per provider group
    candidates: list[tuple[tuple[str, str], int]] = []
    for key, indices in _provider_groups.items():
        for idx in indices:
            if _is_available(idx):
                candidates.append((key, idx))
                break

    if len(candidates) <= 1:
        return None

    log.info("race: starting with %d providers: %s", len(candidates), [pk[1] for pk, _ in candidates])

    race_times: dict[tuple[str, str], float] = {}
    race_start = time.monotonic()
    race_state = {"failures": 0}

    def _maybe_finalize():
        if len(race_times) + race_state["failures"] >= len(candidates):
            _finish_race(race_times)

    # All racers use streaming internally so we can detect first non-error response
    async def _send(pk: tuple, idx: int):
        ep = ENDPOINTS[idx]
        headers = _build_upstream_headers(ep)
        stripped = _strip_unsupported(body_dict, ep)
        send_body = json.dumps(stripped).encode()
        url = f"{ep.base_url}/{path}"
        req = http_client.build_request("POST", url, headers=headers, content=send_body)
        resp = await http_client.send(req, stream=True)
        if resp.status_code in RETRYABLE_STATUSES:
            await resp.aclose()
            raise Exception(f"HTTP {resp.status_code}")
        return pk, idx, resp

    tasks = {asyncio.create_task(_send(pk, idx)): (pk, idx) for pk, idx in candidates}

    # Find first non-error response
    winner = None
    losers_to_drain: list[tuple[tuple[str, str], httpx.Response]] = []
    pending = set(tasks.keys())

    while pending and winner is None:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            pk, idx = tasks[task]
            try:
                rpk, ridx, resp = task.result()
                if winner is None:
                    winner = (rpk, ridx, resp)
                else:
                    losers_to_drain.append((rpk, resp))
            except Exception as exc:
                race_state["failures"] += 1
                _mark_down(idx, str(exc))

    if winner is None:
        for t in pending:
            t.cancel()
        return None

    win_pk, win_idx, win_resp = winner
    log.info("race: first response from %s (%.0fms)", win_pk[1], (time.monotonic() - race_start) * 1000)
    _stats["requests"][win_idx] += 1
    _stats["successes"][win_idx] += 1
    _race_request_count = 0
    _last_race_time = time.monotonic()

    # Background: drain losers to measure total time
    async def _drain(pk, resp, idx=None):
        try:
            async for _ in resp.aiter_bytes():
                pass
            await resp.aclose()
            race_times[pk] = time.monotonic() - race_start
        except Exception as exc:
            race_state["failures"] += 1
            if idx is not None:
                _mark_down(idx, str(exc))
        _maybe_finalize()

    async def _await_and_drain(task, pk, idx):
        try:
            _, _, resp = await task
        except Exception as exc:
            race_state["failures"] += 1
            _mark_down(idx, str(exc))
            _maybe_finalize()
            return
        await _drain(pk, resp, idx)

    def _bg(coro):
        t = asyncio.create_task(coro)
        _background_tasks.add(t)
        t.add_done_callback(_background_tasks.discard)

    for pk, resp in losers_to_drain:
        _bg(_drain(pk, resp))
    for task in pending:
        pk, idx = tasks[task]
        _bg(_await_and_drain(task, pk, idx))

    # Stream winner to client
    if is_streaming:
        async def generate():
            try:
                async for chunk in win_resp.aiter_bytes():
                    yield chunk
            finally:
                await win_resp.aclose()
                race_times[win_pk] = time.monotonic() - race_start
                _maybe_finalize()

        return _streaming_response(win_resp, generate())
    else:
        # Non-streaming: accumulate winner's response
        chunks = []
        try:
            async for chunk in win_resp.aiter_bytes():
                chunks.append(chunk)
        finally:
            await win_resp.aclose()
            race_times[win_pk] = time.monotonic() - race_start
            _maybe_finalize()
        try:
            data = json.loads(b"".join(chunks))
        except Exception:
            return JSONResponse({"error": "upstream returned invalid JSON"}, status_code=502)
        return JSONResponse(content=data, status_code=win_resp.status_code)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    result = {"endpoints": []}
    for idx, ep in enumerate(ENDPOINTS):
        result["endpoints"].append({
            "index": idx,
            "model_override": ep.model_override or "(none)",
            "requests": _stats["requests"].get(idx, 0),
            "successes": _stats["successes"].get(idx, 0),
            "failures": _stats["failures"].get(idx, 0),
        })
    result["fastest_mode"] = {
        "preferred_providers": [{"model": m, "base_url": u} for m, u in _preferred_providers],
        "requests_since_last_race": _race_request_count,
    }
    return result


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(request: Request, path: str, authorization: str | None = Header(None)):
    auth_err = _check_auth(authorization)
    if auth_err:
        return auth_err

    decoded = unquote(path)
    normalized = posixpath.normpath(decoded)
    if normalized.startswith("..") or "/../" in f"/{decoded}/" or decoded.startswith("/"):
        return JSONResponse({"error": "invalid path"}, status_code=400)

    raw_body = await request.body()
    is_streaming = False
    fastest_mode = False

    # Parse body for POST to detect streaming and rewrite model
    if request.method == "POST" and raw_body:
        try:
            body_dict = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        is_streaming = body_dict.get("stream", False)

        # Detect :fastest mode from model name
        model = body_dict.get("model", "")
        if model.endswith(":fastest"):
            fastest_mode = True
            body_dict = {**body_dict, "model": model.removesuffix(":fastest")}
    else:
        body_dict = None

    # Fastest mode: race or use preferred provider order
    if fastest_mode:
        global _race_request_count
        _race_request_count += 1

        if _should_race():
            result = await _race_request(path, body_dict, is_streaming)
            if result is not None:
                return result
            log.warning("race failed, falling back to sequential")

        # Use preferred provider order
        endpoint_order = []
        for pk in _preferred_providers:
            endpoint_order.extend(_provider_groups[pk])
    else:
        endpoint_order = list(range(len(ENDPOINTS)))

    last_failure = None
    for idx in endpoint_order:
        ep = ENDPOINTS[idx]
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
