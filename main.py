import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import API_KEY, COOLOFF_SECONDS, ENDPOINTS, HOST, PORT, REQUEST_TIMEOUT, Endpoint

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stablellm")

# endpoint base_url -> timestamp when it becomes available again
_cooloff_until: dict[str, float] = {}

http_client: httpx.AsyncClient

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0))
    log.info("stablellm started with %d endpoint(s)", len(ENDPOINTS))
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


def _is_available(ep: Endpoint) -> bool:
    return time.monotonic() >= _cooloff_until.get(ep.base_url, 0)


def _mark_down(ep: Endpoint, reason: str):
    _cooloff_until[ep.base_url] = time.monotonic() + COOLOFF_SECONDS
    log.warning("endpoint %s marked down for %ss: %s", ep.base_url, COOLOFF_SECONDS, reason)


def _check_auth(authorization: str | None) -> JSONResponse | None:
    if not API_KEY:
        return None
    if not authorization or authorization != f"Bearer {API_KEY}":
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


async def _proxy_stream(ep: Endpoint, path: str, headers: dict, body: bytes):
    """Stream response from upstream. Returns (StreamingResponse, None) or (None, status_code)."""
    url = f"{ep.base_url}/{path}"
    t0 = time.monotonic()
    req = http_client.build_request("POST", url, headers=headers, content=body)
    resp = await http_client.send(req, stream=True)

    if resp.status_code in RETRYABLE_STATUSES:
        await resp.aclose()
        return None, resp.status_code

    async def generate():
        first_chunk = True
        token_count = 0
        t_first = t0
        try:
            async for chunk in resp.aiter_bytes():
                if first_chunk:
                    t_first = time.monotonic()
                    log.info("%s TTFB %.0fms (stream)", ep.base_url, (t_first - t0) * 1000)
                    first_chunk = False
                # Count SSE data lines as approximate token count
                token_count += chunk.count(b"\ndata: ")
                yield chunk
        finally:
            elapsed = time.monotonic() - t_first
            if elapsed > 0 and token_count > 0:
                log.info("%s %d chunks in %.1fs (%.0f tok/s)", ep.base_url, token_count, elapsed, token_count / elapsed)
            await resp.aclose()

    return StreamingResponse(
        generate(),
        status_code=resp.status_code,
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

    data = resp.json()
    ttfb_ms = elapsed * 1000
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens and elapsed > 0:
        log.info("%s TTFB %.0fms, %d tokens in %.1fs (%.0f tok/s)", ep.base_url, ttfb_ms, completion_tokens, elapsed, completion_tokens / elapsed)
    else:
        log.info("%s TTFB %.0fms", ep.base_url, ttfb_ms)

    return JSONResponse(content=data, status_code=resp.status_code), None


def _build_upstream_headers(ep: Endpoint) -> dict:
    return {
        "Authorization": f"Bearer {ep.api_key}",
        "Content-Type": "application/json",
    }


def _rewrite_model(body: dict, ep: Endpoint) -> dict:
    if ep.model_override:
        body = {**body, "model": ep.model_override}
    return body


@app.get("/health")
async def health():
    return {"status": "ok"}


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
    for ep in ENDPOINTS:
        if not _is_available(ep):
            log.info("skipping %s (cooling off)", ep.base_url)
            continue

        if body_dict is not None:
            send_body = json.dumps(_rewrite_model(body_dict, ep)).encode()
        else:
            send_body = raw_body

        headers = _build_upstream_headers(ep)

        try:
            if is_streaming:
                result, status = await _proxy_stream(ep, path, headers, send_body)
            else:
                result, status = await _proxy_buffered(ep, request.method, path, headers, send_body)

            if result is not None:
                return result

            _mark_down(ep, f"HTTP {status}")
            last_failure = f"HTTP {status}"

        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            _mark_down(ep, str(exc))
            last_failure = type(exc).__name__

    log.error("all endpoints exhausted for /v1/%s (last: %s)", path, last_failure)
    return JSONResponse(
        {"error": f"all endpoints exhausted (last: {last_failure})"},
        status_code=502,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT)
