import asyncio
import hmac
import json
import logging
import posixpath
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from urllib.parse import unquote, urlparse

import httpx
import yaml
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

import config
import requestlog
from config import API_KEY, CONNECT_TIMEOUT, HOST, PORT, REQUEST_TIMEOUT, Endpoint

config.load_or_exit()

class _DokployFormatter(logging.Formatter):
    """Embed a keyword Dokploy's content-based log classifier recognises so
    WARN/ERROR lines tag correctly instead of falling into 'success'.
    Ref: https://deepwiki.com/Dokploy/dokploy/11-monitoring-and-logging
    """
    _PREFIX_BY_LEVEL = {
        logging.WARNING: " [warning]",
        logging.ERROR: " [failed]",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.dokploy = self._PREFIX_BY_LEVEL.get(record.levelno, "")
        return super().format(record)


_LOG_HANDLER = logging.StreamHandler()
_LOG_HANDLER.setFormatter(_DokployFormatter("%(asctime)s %(levelname)s%(dokploy)s %(name)s: %(message)s"))
logging.basicConfig(
    level=getattr(logging, config.SETTINGS.log_level, logging.INFO),
    handlers=[_LOG_HANDLER],
    force=True,
)
log = logging.getLogger("stablellm")

# Reduce noise from external libraries — these spew at DEBUG (httpcore is the
# lower layer under httpx and is especially chatty).
for noisy in ("uvicorn.access", "httpx", "httpcore", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# endpoint index -> timestamp when it becomes available again
_cooloff_until: dict[int, float] = {}

# Track last used endpoint for change detection
_last_endpoint_idx: int | None = None

# endpoint index -> stats
_stats = {
    "requests": defaultdict(int),
    "successes": defaultdict(int),
    "failures": defaultdict(int),
}

http_client: httpx.AsyncClient

_EXCLUDED_HEADERS = {"transfer-encoding", "connection", "keep-alive", "content-encoding", "content-length"}

_background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks

# Per-group state (every request resolves to exactly one group by exact model match)
_group_provider_groups: dict[str, dict[tuple[str, str], list[int]]] = {}
_group_preferred_providers: dict[str, list[tuple[str, str]]] = {}
_group_race_request_count: dict[str, int] = defaultdict(int)
_group_last_race_time: dict[str, float] = defaultdict(float)
_group_race_generation: dict[str, int] = defaultdict(int)


def _build_provider_groups():
    """(Re)build the per-group provider partitioning from current config.GROUPS."""
    global _group_provider_groups, _group_preferred_providers
    _group_provider_groups = {}
    _group_preferred_providers = {}
    for group_name, group in config.GROUPS.items():
        g: dict[tuple[str, str], list[int]] = {}
        for idx in group.endpoints:
            ep = config.ENDPOINTS[idx]
            key = (ep.model, ep.base_url)
            g.setdefault(key, []).append(idx)
        _group_provider_groups[group_name] = g
        _group_preferred_providers[group_name] = list(g.keys())


# Canonical names are :race / :seq. :fastest and :normal are aliases.
_SUFFIX_MODES: dict[str, str] = {
    f":{config.MODE_RACE}": config.MODE_RACE,
    ":fastest": config.MODE_RACE,
    f":{config.MODE_SEQ}": config.MODE_SEQ,
    ":normal": config.MODE_SEQ,
}


def _parse_model_suffix(model: str) -> tuple[str, str | None]:
    """Strip a single mode suffix (case-insensitive). Returns (stripped, override_mode).

    Recognized suffixes: :race / :fastest (race), :seq / :normal (seq).
    Raises ValueError if more than one suffix is present. The returned stem
    preserves the original case of the model so it can be passed upstream as-is.
    """
    model_lc = model.lower()
    for suffix, mode in _SUFFIX_MODES.items():
        if model_lc.endswith(suffix):
            stem = model[:-len(suffix)]
            if any(stem.lower().endswith(s) for s in _SUFFIX_MODES):
                raise ValueError(f"multiple mode suffixes on model '{model}'")
            return stem, mode
    return model, None


def _effective_model(ep: Endpoint, client_model: str) -> str:
    return ep.model or client_model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global http_client
    requestlog.init()
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT))
    _build_provider_groups()
    log.info("stablellm started with %d endpoint(s), groups: %s", len(config.ENDPOINTS), list(config.GROUPS))
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-StableLLM-Provider",
        "X-StableLLM-Model",
        "X-StableLLM-Mode",
        "X-StableLLM-Group",
        "X-StableLLM-Via",
    ],
)


@app.middleware("http")
async def _limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            n = int(cl)
        except ValueError:
            return JSONResponse({"error": "invalid content-length"}, status_code=400)
        if n > config.MAX_BODY_BYTES:
            return JSONResponse({"error": "request body too large"}, status_code=413)
    return await call_next(request)


def _is_available(idx: int) -> bool:
    return time.monotonic() >= _cooloff_until.get(idx, 0)


def _mark_down(idx: int, reason: str, request_context: str = ""):
    cooloff = config.SETTINGS.cooloff_seconds
    _cooloff_until[idx] = time.monotonic() + cooloff
    _stats["failures"][idx] += 1
    ep = config.ENDPOINTS[idx]
    context = f" ({request_context})" if request_context else ""
    log.warning("endpoint %s marked down for %ss%s: %s", ep.base_url, cooloff, context, reason)


class UpstreamError(Exception):
    pass


def _exception_detail(exc: BaseException) -> str:
    msg = str(exc)
    if isinstance(exc, UpstreamError) and msg:
        return msg
    if msg:
        return f"{type(exc).__name__}: {msg}"
    return type(exc).__name__


def _body_snippet(content: bytes, limit: int = 1000) -> str:
    if not content:
        return ""
    try:
        obj = json.loads(content)
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        text = content.decode(errors="replace").strip()
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


async def _http_error_reason(resp: httpx.Response) -> str:
    content = await resp.aread()
    detail = _body_snippet(content)
    reason = f"HTTP {resp.status_code}"
    if resp.reason_phrase:
        reason = f"{reason} {resp.reason_phrase}"
    if detail:
        return f"{reason}: {detail}"
    return reason


def _request_context(path: str, body: dict, group: str = "", ep: Endpoint | None = None) -> str:
    parts = [f"path=/v1/{path}"]
    if group:
        parts.append(f"group={group!r}")
    client_model = body.get("model", "")
    if client_model:
        parts.append(f"requested_model={client_model!r}")
    if ep is not None:
        parts.append(f"upstream_model={_effective_model(ep, client_model)!r}")
    parts.append(f"stream={bool(body.get('stream', False))}")
    messages = body.get("messages")
    if isinstance(messages, list):
        parts.append(f"messages={len(messages)}")
    keys = sorted(k for k in body if k != "messages")
    parts.append(f"keys={keys}")
    return " ".join(parts)


def _check_auth(authorization: str | None) -> JSONResponse | None:
    if not API_KEY:
        return None
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {API_KEY}"):
        log.warning("auth failed: %s", "missing header" if not authorization else "invalid token")
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


def _extract_usage_from_sse(buffer: bytearray, chunk: bytes) -> int | None:
    buffer.extend(chunk)
    while b"\n\n" in buffer:
        event, _, rest = buffer.partition(b"\n\n")
        buffer[:] = rest
        for line in event.split(b"\n"):
            if line.startswith(b"data:"):
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                    usage = obj.get("usage")
                    if usage:
                        ct = usage.get("completion_tokens")
                        if ct is not None:
                            return ct
                except json.JSONDecodeError:
                    pass
    return None


async def _proxy_stream(ep: Endpoint, path: str, headers: dict, body: bytes, metrics, request_context: str):
    """Stream response from upstream. Returns (StreamingResponse, None) or (None, reason)."""
    url = f"{ep.base_url}/{path}"
    t0 = time.monotonic()
    req = http_client.build_request("POST", url, headers=headers, content=body)
    resp = await http_client.send(req, stream=True)
    ttfb = time.monotonic() - t0

    try:
        if resp.status_code != 200:
            reason = await _http_error_reason(resp)
            await resp.aclose()
            log.warning("upstream rejected streaming request to %s (%s): %s", ep.base_url, request_context, reason)
            return None, reason

        # Prime upstream chunks until we see the serving sub-provider (OpenRouter
        # tags every data chunk with top-level `provider`, but may first send
        # `: OPENROUTER PROCESSING` keepalive comments) so we can set the
        # X-StableLLM-Via header before response headers are committed.
        want_via = _openrouter_via(ep)
        byte_iter = resp.aiter_bytes()
        primed: list[bytes] = []
        via = None
        if want_via:
            try:
                while len(primed) < 8:
                    chunk = await byte_iter.__anext__()
                    if metrics.ttft_ms is None:
                        metrics.ttft_ms = (time.monotonic() - t0) * 1000
                        log.info("%s TTFT %.0fms (TTFB %.0fms)", ep.base_url, metrics.ttft_ms, ttfb * 1000)
                    primed.append(chunk)
                    via = _provider_from_sse(chunk)
                    if via:
                        break
            except StopAsyncIteration:
                pass

        async def generate():
            t_first = None
            completion_tokens = None
            sse_buf = bytearray()
            try:
                for chunk in primed:
                    ct = _extract_usage_from_sse(sse_buf, chunk)
                    if ct is not None:
                        completion_tokens = ct
                    if t_first is None:
                        t_first = time.monotonic()
                    yield chunk
                async for chunk in byte_iter:
                    if t_first is None:
                        t_first = time.monotonic()
                        metrics.ttft_ms = (t_first - t0) * 1000
                        log.info("%s TTFT %.0fms (TTFB %.0fms)", ep.base_url, metrics.ttft_ms, ttfb * 1000)
                    ct = _extract_usage_from_sse(sse_buf, chunk)
                    if ct is not None:
                        completion_tokens = ct
                    yield chunk
                if metrics.ttft_ms is not None and completion_tokens is not None:
                    metrics.tokens_per_sec = completion_tokens / (time.monotonic() - t_first)
            finally:
                await resp.aclose()
                await asyncio.to_thread(requestlog.log_request, metrics)

        result = _streaming_response(resp, generate())
        if via:
            result.headers["X-StableLLM-Via"] = via
        return result, None
    except BaseException:
        await resp.aclose()
        raise


async def _proxy_buffered(ep: Endpoint, path: str, headers: dict, body: bytes, metrics, request_context: str):
    """Non-streaming: send request, return full response or (None, reason)."""
    url = f"{ep.base_url}/{path}"
    t0 = time.monotonic()
    resp = await http_client.post(url, headers=headers, content=body)
    elapsed = time.monotonic() - t0

    if resp.status_code != 200:
        reason = await _http_error_reason(resp)
        log.warning("upstream rejected buffered request to %s (%s): %s", ep.base_url, request_context, reason)
        return None, reason

    try:
        data = resp.json()
    except Exception as exc:
        reason = f"invalid JSON: {_exception_detail(exc)}"
        detail = _body_snippet(resp.content)
        if detail:
            reason = f"{reason}: {detail}"
        log.warning("upstream returned invalid JSON from %s (%s): %s", ep.base_url, request_context, reason)
        return None, reason

    metrics.ttft_ms = elapsed * 1000
    usage = data.get("usage")
    if usage:
        ct = usage.get("completion_tokens")
        if ct is not None:
            metrics.tokens_per_sec = ct / elapsed

    log.info("%s TTFB %.0fms", ep.base_url, elapsed * 1000)

    via = _openrouter_served_provider(data) if _openrouter_via(ep) else None
    result = JSONResponse(content=data, status_code=resp.status_code)
    if via:
        result.headers["X-StableLLM-Via"] = via

    await asyncio.to_thread(requestlog.log_request, metrics)
    return result, None


def _build_upstream_headers(ep: Endpoint) -> dict:
    return {
        "Authorization": f"Bearer {ep.api_key}",
        "Content-Type": "application/json",
    }


def _openrouter_via(ep: Endpoint) -> bool:
    """True if the endpoint routes through OpenRouter, which tags every
    response/chunk with the serving sub-provider in a top-level ``provider`` field."""
    host = urlparse(ep.base_url).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _openrouter_served_provider(data: object) -> str | None:
    """Serving sub-provider from an OpenRouter response body / stream chunk
    (top-level ``provider`` field), or None if absent."""
    if not isinstance(data, dict):
        return None
    p = data.get("provider")
    if not isinstance(p, str) or not p:
        return None
    # Header-unsafe control chars would break response encoding; strip to latin-1 safe.
    return p if p.isascii() and "\r" not in p and "\n" not in p else None


def _provider_from_sse(bytes_data: bytes) -> str | None:
    """Scan an SSE payload for the first ``data: {...}`` chunk carrying a
    top-level ``provider`` field (OpenRouter). Skips comment/keepalive lines."""
    for line in bytes_data.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        via = _openrouter_served_provider(obj)
        if via:
            return via
    return None


def _streaming_response(resp: httpx.Response, generator) -> StreamingResponse:
    forward_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _EXCLUDED_HEADERS}
    return StreamingResponse(
        generator,
        status_code=resp.status_code,
        headers=forward_headers,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


def _set_meta_headers(response, *, provider: str, model: str, mode: str, group: str, via: str | None = None):
    response.headers["X-StableLLM-Provider"] = provider
    response.headers["X-StableLLM-Model"] = model
    response.headers["X-StableLLM-Mode"] = mode
    response.headers["X-StableLLM-Group"] = group
    if via:
        response.headers["X-StableLLM-Via"] = via


# Only pass these known-supported params to upstream
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "functions",
    "function_call",
    "response_format",
    "prediction",
    "service_tier",
    "store",
    "metadata",
    "audio",
    "modalities",
    "web_search_options",
    "user",
    "reasoning_effort",
    "thinking",
    "clear_thinking",
    "disable_reasoning",
}

_REASONING_KEYS = frozenset({"reasoning", "reasoning_content", "thinking"})


def _rewrite_model(body: dict, ep: Endpoint) -> dict:
    return {**body, "model": _effective_model(ep, body.get("model", ""))}


def _strip_message_reasoning(messages: list) -> list:
    return [{k: v for k, v in msg.items() if k not in _REASONING_KEYS} for msg in messages]


def _strip_unsupported(body: dict, ep: Endpoint) -> dict:
    """Keep only supported params, strip reasoning from messages, apply model name."""
    base = {k: v for k, v in body.items() if k in SUPPORTED_PARAMS}
    if not ep.keep_reasoning and "messages" in base:
        base["messages"] = _strip_message_reasoning(base["messages"])
    return _rewrite_model(base, ep)


def _should_race(group: str) -> bool:
    last_time = _group_last_race_time[group]
    count = _group_race_request_count[group]
    if last_time == 0.0:
        return True
    if count >= config.SETTINGS.race_interval_requests:
        return True
    if time.monotonic() - last_time >= config.SETTINGS.race_interval_secs:
        return True
    return False


def _finish_race(race_times: dict[tuple[str, str], float], group: str):
    sorted_keys = sorted(race_times, key=race_times.get)
    new_order = list(sorted_keys)
    for k in _group_provider_groups[group]:
        if k not in race_times:
            new_order.append(k)
    _group_preferred_providers[group] = new_order
    log.info(
        "race complete (group=%s): %s",
        group,
        [(f"{k[1]} model={k[0]}", f"{v:.1f}s") for k, v in sorted(race_times.items(), key=lambda x: x[1])],
    )


async def _race_request(path: str, body_dict: dict, is_streaming: bool, group: str):
    """Race one endpoint per provider group with real request. Returns response or None."""
    pg = _group_provider_groups[group]

    candidates: list[tuple[tuple[str, str], int]] = []
    for key, indices in pg.items():
        for idx in indices:
            if _is_available(idx):
                candidates.append((key, idx))
                break

    if len(candidates) <= 1:
        return None

    # Reset race-cadence state on attempt so a failed race doesn't cause every
    # following request to retry the race against still-cooling-off endpoints.
    _group_race_request_count[group] = 0
    _group_last_race_time[group] = time.monotonic()
    _group_race_generation[group] += 1
    gen = _group_race_generation[group]

    race_context = _request_context(path, body_dict, group)
    log.info(
        "race: starting with %d providers (%s): %s",
        len(candidates),
        race_context,
        [f"{pk[1]} model={_effective_model(config.ENDPOINTS[idx], body_dict.get('model', ''))}" for pk, idx in candidates],
    )

    race_times: dict[tuple[str, str], float] = {}
    race_start = time.monotonic()
    race_state = {"failures": 0}

    def _maybe_finalize():
        # Skip if a newer race for this group has already started; otherwise late
        # background drains from this race could clobber the newer race's order.
        if _group_race_generation[group] != gen:
            return
        if len(race_times) + race_state["failures"] >= len(candidates):
            _finish_race(race_times, group)

    async def _send(pk: tuple, idx: int):
        ep = config.ENDPOINTS[idx]
        headers = _build_upstream_headers(ep)
        stripped = _strip_unsupported(body_dict, ep)
        send_body = json.dumps(stripped).encode()
        url = f"{ep.base_url}/{path}"
        req = http_client.build_request("POST", url, headers=headers, content=send_body)
        resp = await http_client.send(req, stream=True)
        if resp.status_code != 200:
            reason = await _http_error_reason(resp)
            await resp.aclose()
            raise UpstreamError(reason)
        return pk, idx, resp

    tasks = {asyncio.create_task(_send(pk, idx)): (pk, idx) for pk, idx in candidates}

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
                _mark_down(idx, _exception_detail(exc), _request_context(path, body_dict, group, config.ENDPOINTS[idx]))

    if winner is None:
        for t in pending:
            t.cancel()
        return None

    win_pk, win_idx, win_resp = winner
    ep = config.ENDPOINTS[win_idx]
    model_name = _effective_model(ep, body_dict.get("model", ""))
    log.info("race: first response from %s (model: %s) %.0fms", win_pk[1], model_name, (time.monotonic() - race_start) * 1000)
    _stats["requests"][win_idx] += 1

    global _last_endpoint_idx
    if _last_endpoint_idx != win_idx:
        log.info("using endpoint: %s (model: %s)", ep.base_url, model_name)
        _last_endpoint_idx = win_idx

    race_metrics = requestlog.RequestMetrics(
        model_requested=body_dict.get("model", "") if body_dict else "",
        provider_served=win_pk[1],
        model_served=model_name,
    )

    async def _drain(pk, resp, idx=None):
        try:
            async for _ in resp.aiter_bytes():
                pass
            await resp.aclose()
            race_times[pk] = time.monotonic() - race_start
        except Exception as exc:
            race_state["failures"] += 1
            if idx is not None:
                _mark_down(idx, _exception_detail(exc), _request_context(path, body_dict, group, config.ENDPOINTS[idx]))
        _maybe_finalize()

    async def _await_and_drain(task, pk, idx):
        try:
            _, _, resp = await task
        except Exception as exc:
            race_state["failures"] += 1
            _mark_down(idx, _exception_detail(exc), _request_context(path, body_dict, group, config.ENDPOINTS[idx]))
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

    if is_streaming:
        t0_race = race_start
        want_via = _openrouter_via(ep)
        byte_iter = win_resp.aiter_bytes()
        primed: list[bytes] = []
        via = None
        if want_via:
            try:
                while len(primed) < 8:
                    chunk = await byte_iter.__anext__()
                    if race_metrics.ttft_ms is None:
                        race_metrics.ttft_ms = (time.monotonic() - t0_race) * 1000
                    primed.append(chunk)
                    via = _provider_from_sse(chunk)
                    if via:
                        break
            except StopAsyncIteration:
                pass
            except BaseException as exc:
                # Winner stalled mid-priming (transport error or client cancel).
                # Close it and fail the race so the caller falls back to
                # sequential rather than 500-ing the client / leaking the conn.
                race_state["failures"] += 1
                await win_resp.aclose()
                _mark_down(win_idx, _exception_detail(exc), race_context)
                _maybe_finalize()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return None

        async def generate():
            t_first = None
            completion_tokens = None
            sse_buf = bytearray()
            try:
                for chunk in primed:
                    ct = _extract_usage_from_sse(sse_buf, chunk)
                    if ct is not None:
                        completion_tokens = ct
                    if t_first is None:
                        t_first = time.monotonic()
                    yield chunk
                async for chunk in byte_iter:
                    if t_first is None:
                        t_first = time.monotonic()
                        race_metrics.ttft_ms = (t_first - t0_race) * 1000
                    ct = _extract_usage_from_sse(sse_buf, chunk)
                    if ct is not None:
                        completion_tokens = ct
                    yield chunk
                if race_metrics.ttft_ms is not None and completion_tokens is not None:
                    race_metrics.tokens_per_sec = completion_tokens / (time.monotonic() - t_first)
            finally:
                await win_resp.aclose()
                race_times[win_pk] = time.monotonic() - race_start
                _maybe_finalize()
                await asyncio.to_thread(requestlog.log_request, race_metrics)

        result = _streaming_response(win_resp, generate())
        _stats["successes"][win_idx] += 1
        _set_meta_headers(result, provider=ep.provider, model=model_name, mode=config.MODE_RACE, group=group, via=via)
        return result
    else:
        chunks = []
        try:
            async for chunk in win_resp.aiter_bytes():
                chunks.append(chunk)
        finally:
            await win_resp.aclose()
            race_times[win_pk] = time.monotonic() - race_start
            _maybe_finalize()
        response_body = b"".join(chunks)
        try:
            data = json.loads(response_body)
        except Exception as exc:
            reason = f"invalid JSON from race winner: {_exception_detail(exc)}"
            detail = _body_snippet(response_body)
            if detail:
                reason = f"{reason}: {detail}"
            log.error("upstream returned invalid JSON from race winner %s (%s): %s", ep.base_url, race_context, reason)
            return JSONResponse({"error": "upstream returned invalid JSON"}, status_code=502)
        elapsed = time.monotonic() - race_start
        race_metrics.ttft_ms = elapsed * 1000
        usage = data.get("usage")
        if usage:
            ct = usage.get("completion_tokens")
            if ct is not None:
                race_metrics.tokens_per_sec = ct / elapsed
        await asyncio.to_thread(requestlog.log_request, race_metrics)
        result = JSONResponse(content=data, status_code=win_resp.status_code)
        via = _openrouter_served_provider(data) if _openrouter_via(ep) else None
        _stats["successes"][win_idx] += 1
        _set_meta_headers(result, provider=ep.provider, model=model_name, mode=config.MODE_RACE, group=group, via=via)
        return result


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(None)):
    auth_err = _check_auth(authorization)
    if auth_err:
        return auth_err
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "stablellm"}
            for name in config.GROUPS
        ],
    }


@app.get("/stats")
async def stats():
    result = {"endpoints": []}
    for idx, ep in enumerate(config.ENDPOINTS):
        result["endpoints"].append({
            "index": idx,
            "model": ep.model or "(none)",
            "requests": _stats["requests"].get(idx, 0),
            "successes": _stats["successes"].get(idx, 0),
            "failures": _stats["failures"].get(idx, 0),
        })
    result["groups"] = {}
    for name, group in config.GROUPS.items():
        result["groups"][name] = {
            "endpoints": group.endpoints,
            "mode": group.mode,
            "preferred_providers": [{"model": m, "base_url": u} for m, u in _group_preferred_providers.get(name, [])],
            "requests_since_last_race": _group_race_request_count.get(name, 0),
        }
    return result


CONFIG_EDITOR_HTML = """<!DOCTYPE html>
<html>
<head>
<title>stablellm config</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/theme/material-darker.min.css">
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #1e1e1e; color: #ccc; margin: 0; padding: 20px; }
  h1 { font-size: 1.1em; margin: 0 0 12px; font-weight: 500; }
  .bar { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
  input[type=password] { background: #2d2d2d; color: #ccc; border: 1px solid #444; padding: 6px 10px; font-family: monospace; border-radius: 3px; }
  input[type=password]:focus { outline: none; border-color: #0e639c; }
  button { background: #0e639c; color: white; border: none; padding: 6px 14px; cursor: pointer; font-size: 13px; border-radius: 3px; }
  button:hover:not(:disabled) { background: #1177bb; }
  button:disabled { background: #555; cursor: not-allowed; }
  .CodeMirror { height: 72vh; border: 1px solid #444; font-size: 13px; border-radius: 3px; }
  #status { margin-top: 10px; padding: 8px 12px; border-radius: 3px; min-height: 1.2em; font-family: monospace; font-size: 12px; white-space: pre-wrap; }
  #status.ok { background: #1e4620; color: #a7d9a8; }
  #status.err { background: #5a1d1d; color: #f5a5a5; }
  #status.info { background: #2d2d2d; color: #8ab4f8; }
  .spacer { flex: 1; }
  .hint { color: #666; font-size: 12px; }
</style>
</head>
<body>
<h1>stablellm config editor</h1>
<div class="bar">
  <input type="password" id="pw" placeholder="password" autofocus>
  <button id="load">Load</button>
  <button id="save" disabled>Save &amp; Reload</button>
  <div class="spacer"></div>
  <span class="hint">Ctrl/Cmd+S to save</span>
</div>
<textarea id="editor"></textarea>
<div id="status" class="info">Enter password and click Load</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/codemirror.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/mode/yaml/yaml.min.js"></script>
<script>
const cm = CodeMirror.fromTextArea(document.getElementById('editor'), {
  mode: 'yaml',
  theme: 'material-darker',
  lineNumbers: true,
  indentUnit: 2,
  tabSize: 2,
  lineWrapping: false,
  extraKeys: { 'Ctrl-S': saveConfig, 'Cmd-S': saveConfig },
});

const pw = document.getElementById('pw');
const loadBtn = document.getElementById('load');
const saveBtn = document.getElementById('save');
const status = document.getElementById('status');

function setStatus(msg, kind) {
  status.textContent = msg;
  status.className = kind;
}

async function loadConfig() {
  if (!pw.value) { setStatus('enter password', 'err'); return; }
  setStatus('loading...', 'info');
  try {
    const r = await fetch('api/content', { headers: { 'X-Config-Password': pw.value } });
    const text = await r.text();
    if (!r.ok) { setStatus(text || ('HTTP ' + r.status), 'err'); return; }
    cm.setValue(text);
    saveBtn.disabled = false;
    setStatus('loaded', 'ok');
  } catch (e) { setStatus(String(e), 'err'); }
}

async function saveConfig() {
  if (saveBtn.disabled) return;
  setStatus('saving...', 'info');
  try {
    const r = await fetch('api/save', {
      method: 'POST',
      headers: { 'X-Config-Password': pw.value, 'Content-Type': 'text/plain' },
      body: cm.getValue(),
    });
    const text = await r.text();
    setStatus(text, r.ok ? 'ok' : 'err');
  } catch (e) { setStatus(String(e), 'err'); }
}

loadBtn.addEventListener('click', loadConfig);
saveBtn.addEventListener('click', saveConfig);
pw.addEventListener('keydown', e => { if (e.key === 'Enter') loadConfig(); });
</script>
</body>
</html>
"""


def _reset_runtime_state():
    """Clear stats/cooloff/race state. Endpoint indices may have shifted after reload."""
    _cooloff_until.clear()
    _stats["requests"].clear()
    _stats["successes"].clear()
    _stats["failures"].clear()
    _group_race_request_count.clear()
    _group_last_race_time.clear()
    global _last_endpoint_idx
    _last_endpoint_idx = None


EDITOR_AUTH_DELAY_SECS = 0.5


async def _editor_auth(password: str | None) -> PlainTextResponse | None:
    if not config.CONFIG_EDITOR_PASSWORD:
        return PlainTextResponse("not found", status_code=404)
    # Constant delay applied to both success and failure to slow brute-force
    # and avoid leaking timing info about which side of the compare differed.
    ok = bool(password) and hmac.compare_digest(password, config.CONFIG_EDITOR_PASSWORD)
    await asyncio.sleep(EDITOR_AUTH_DELAY_SECS)
    if not ok:
        return PlainTextResponse("unauthorized", status_code=401)
    return None


@app.get("/config/editor")
async def config_editor_page():
    if not config.CONFIG_EDITOR_PASSWORD:
        return PlainTextResponse("not found", status_code=404)
    return HTMLResponse(CONFIG_EDITOR_HTML)


@app.get("/config/api/content")
async def config_get_content(x_config_password: str | None = Header(None)):
    err = await _editor_auth(x_config_password)
    if err:
        return err
    try:
        with open(config.CONFIG_FILE) as f:
            return PlainTextResponse(f.read())
    except OSError as exc:
        return PlainTextResponse(f"read failed: {exc}", status_code=500)


@app.post("/config/api/save")
async def config_save(request: Request, x_config_password: str | None = Header(None)):
    err = await _editor_auth(x_config_password)
    if err:
        return err

    try:
        new_content = (await request.body()).decode("utf-8")
    except UnicodeDecodeError as exc:
        return PlainTextResponse(f"invalid UTF-8: {exc}", status_code=400)

    # Validate before touching disk
    try:
        raw = yaml.safe_load(new_content)
    except yaml.YAMLError as exc:
        return PlainTextResponse(f"YAML error: {exc}", status_code=400)
    try:
        config.parse_config(raw)
    except config.ConfigError as exc:
        return PlainTextResponse(f"validation failed: {exc}", status_code=400)

    try:
        with open(config.CONFIG_FILE, "w") as f:
            f.write(new_content)
    except OSError as exc:
        return PlainTextResponse(f"write failed: {exc}", status_code=500)

    config.reload()
    _build_provider_groups()
    _reset_runtime_state()
    logging.getLogger().setLevel(getattr(logging, config.SETTINGS.log_level, logging.INFO))

    group_names = list(config.GROUPS)
    log.info("config reloaded via editor: %d endpoint(s), groups: %s", len(config.ENDPOINTS), group_names)
    return PlainTextResponse(
        f"saved and reloaded: {len(config.ENDPOINTS)} endpoint(s), groups: {group_names}"
    )


@app.post("/v1/{path:path}")
async def proxy(request: Request, path: str, authorization: str | None = Header(None)):
    auth_err = _check_auth(authorization)
    if auth_err:
        return auth_err

    decoded = unquote(path)
    normalized = posixpath.normpath(decoded)
    if normalized.startswith("..") or "/../" in f"/{decoded}/" or decoded.startswith("/"):
        return JSONResponse({"error": "invalid path"}, status_code=400)

    raw_body = await request.body()
    if not raw_body:
        return JSONResponse({"error": "request body is required"}, status_code=400)
    try:
        body_dict = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    is_streaming = body_dict.get("stream", False)
    model = body_dict.get("model", "")
    if not model:
        return JSONResponse({"error": "'model' is required"}, status_code=400)

    try:
        model, mode_override = _parse_model_suffix(model)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if mode_override is not None:
        body_dict = {**body_dict, "model": model}

    group_name = model.lower()
    if group_name not in config.GROUPS:
        log.warning("unknown model for /v1/%s: requested_model=%r available=%s", path, model, sorted(config.GROUPS))
        return JSONResponse({"error": f"unknown model: '{model}'"}, status_code=404)

    mode = mode_override or config.GROUPS[group_name].mode

    log.info(
        "-> /v1/%s model=%r group=%s mode=%s stream=%s",
        path, model, group_name, mode, is_streaming,
    )

    if mode == config.MODE_RACE:
        _group_race_request_count[group_name] += 1

        if _should_race(group_name):
            result = await _race_request(path, body_dict, is_streaming, group_name)
            if result is not None:
                return result
            log.warning("race failed (%s), falling back to sequential", _request_context(path, body_dict, group_name))

        pref = _group_preferred_providers[group_name]
        pg = _group_provider_groups[group_name]
        endpoint_order = []
        for pk in pref:
            endpoint_order.extend(pg[pk])
    else:
        endpoint_order = config.GROUPS[group_name].endpoints

    last_failure = None
    for idx in endpoint_order:
        ep = config.ENDPOINTS[idx]
        if not _is_available(idx):
            log.info("skipping %s (cooling off)", ep.base_url)
            continue

        _stats["requests"][idx] += 1

        stripped = _strip_unsupported(body_dict, ep)
        send_body = json.dumps(stripped).encode()
        log.debug("-> %s body (keys): %s", ep.base_url, list(stripped.keys()))

        headers = _build_upstream_headers(ep)
        log.debug(
            "-> %s headers: %s",
            ep.base_url,
            {k: v for k, v in headers.items() if k.lower() != "authorization"},
        )

        client_model = body_dict.get("model", "")
        model_name = _effective_model(ep, client_model)
        request_context = _request_context(path, body_dict, group_name, ep)
        metrics = requestlog.RequestMetrics(
            model_requested=client_model,
            provider_served=ep.base_url,
            model_served=model_name,
        )

        try:
            if is_streaming:
                result, reason = await _proxy_stream(ep, path, headers, send_body, metrics, request_context)
            else:
                result, reason = await _proxy_buffered(ep, path, headers, send_body, metrics, request_context)

            if result is not None:
                _stats["successes"][idx] += 1
                global _last_endpoint_idx
                if _last_endpoint_idx != idx:
                    log.info("using endpoint: %s (model: %s)", ep.base_url, model_name)
                    _last_endpoint_idx = idx
                _set_meta_headers(result, provider=ep.provider, model=model_name, mode=mode, group=group_name)
                return result

            _mark_down(idx, reason, request_context)
            last_failure = reason

        except httpx.TransportError as exc:
            reason = _exception_detail(exc)
            _mark_down(idx, reason, request_context)
            last_failure = reason

    log.error("all endpoints failed (exhausted) for %s (last: %s)", _request_context(path, body_dict, group_name), last_failure)
    return JSONResponse(
        {"error": f"all endpoints exhausted (last: {last_failure})"},
        status_code=502,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT)
