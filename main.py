import asyncio
import hashlib
import hmac
import json
import logging
import posixpath
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlparse

import httpx
import yaml
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

import config
import requestlog
from config import (
    API_KEYS,
    CONNECT_TIMEOUT,
    HOST,
    PORT,
    REQUEST_TIMEOUT,
    Endpoint,
    ModelMeta,
)

config.load_or_exit()

class _DokployFormatter(logging.Formatter):
    """Embed a keyword Dokploy's content-based log classifier recognises so
    WARN/ERROR lines tag correctly instead of falling into 'success'.
    Ref: https://deepwiki.com/Dokploy/dokploy/11-monitoring-and-logging
    """
    _PREFIX_BY_LEVEL: ClassVar[dict[int, str]] = {
        logging.WARNING: " [warning]",
        logging.ERROR: " [failed]",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.dokploy = self._PREFIX_BY_LEVEL.get(record.levelno, "")
        return super().format(record)


_LOG_HANDLER = logging.StreamHandler()
_LOG_HANDLER.setFormatter(_DokployFormatter("%(asctime)s %(levelname)-7s%(dokploy)s %(name)s: %(message)s"))
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

# (base_url, effective model) -> in-flight request count, for provider
# concurrency caps. Keyed by model because providers like synthetic cap per
# model, and across groups so shared provider+model pairs share one budget.
# NOT cleared on config reload: keys survive index shifts, and in-flight
# requests from the old config must release against the same counters.
_inflight: dict[tuple[str, str], int] = defaultdict(int)

# (group, session key) -> (endpoint index, last-used monotonic timestamp).
# Keeps a session on the endpoint that last served it so upstream prompt
# caches stay warm; the cap-skip may bounce a session once, and the pin then
# follows it to the new endpoint instead of re-contesting the old one.
_session_pins: dict[tuple[str, str], tuple[int, float]] = {}
_PIN_TTL_SECS = 600.0
_PIN_TABLE_MAX = 500
_HASH_CAP = 65536  # serialized bytes of a message fed to the session-key hash
_HASH_STR_CAP = 8192  # per-string cap, applied BEFORE serialization so huge strings don't stall the loop


def _hash_shrink(obj, depth=0):
    """Bound serialization cost: truncate strings and lists before dumping."""
    if depth > 8:
        return "..."
    if isinstance(obj, str):
        return obj[:_HASH_STR_CAP]
    if isinstance(obj, dict):
        return {k: _hash_shrink(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_hash_shrink(v, depth + 1) for v in obj[:64]]
    return obj

# Times the seq loop routed a request to its pinned endpoint; zero would mean
# pinning never engages (e.g. clients with per-turn-volatile system prompts).
_pin_promotions = 0

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
        "X-StableLLM-Pin",
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


def _at_cap(ep: Endpoint, model_name: str) -> bool:
    return bool(ep.max_concurrency) and _inflight[(ep.base_url, model_name)] >= ep.max_concurrency


def _try_acquire_slot(ep: Endpoint, model_name: str) -> bool:
    """Non-blocking acquire of the endpoint's per-model concurrency slot."""
    if not ep.max_concurrency:
        return True
    if _at_cap(ep, model_name):
        return False
    _inflight[(ep.base_url, model_name)] += 1
    return True


def _slot_releaser(ep: Endpoint, model_name: str):
    """Idempotent release closure; safe to call from multiple exit paths.
    Slots are held until the response is fully consumed (stream included)."""
    released = False

    def release():
        nonlocal released
        if released:
            return
        released = True
        if ep.max_concurrency:
            _inflight[(ep.base_url, model_name)] -= 1

    return release


def _session_key(body: dict) -> str:
    """Stable per-conversation key for session pinning.

    OpenAI-compatible requests carry no conversation id, so combine what is
    stable within a session but distinct across a client's concurrent
    sessions: the 'user' field, the first message (usually the system prompt),
    and the first user turn (the opening instruction). Everything is hashed;
    large values are truncated before serialization so hashing never stalls
    the event loop."""
    parts = []
    user = body.get("user")
    if isinstance(user, str) and user:
        parts.append(user)
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        picks = [messages[0]]
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                picks.append(msg)
                break
        for msg in picks:
            try:
                serialized = json.dumps(_hash_shrink(msg), sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                return ""
            parts.append(serialized[:_HASH_CAP])
    if not parts:
        return ""
    # errors=replace: malformed client JSON can carry lone surrogates, which
    # strict utf-8 encoding would reject; replacement is deterministic so the
    # key stays stable.
    return "m:" + hashlib.sha256("\n".join(parts).encode(errors="replace")).hexdigest()[:16]


def _pinned_endpoint(group: str, skey: str) -> int | None:
    if not skey:
        return None
    entry = _session_pins.get((group, skey))
    if entry is None:
        return None
    idx, ts = entry
    if time.monotonic() - ts > _PIN_TTL_SECS:
        del _session_pins[(group, skey)]
        return None
    return idx


def _set_session_pin(group: str, skey: str, idx: int):
    if not skey:
        return
    if len(_session_pins) >= _PIN_TABLE_MAX and (group, skey) not in _session_pins:
        now = time.monotonic()
        for k in [k for k, (_, ts) in _session_pins.items() if now - ts > _PIN_TTL_SECS]:
            del _session_pins[k]
        if len(_session_pins) >= _PIN_TABLE_MAX:
            del _session_pins[min(_session_pins, key=lambda k: _session_pins[k][1])]
    _session_pins[(group, skey)] = (idx, time.monotonic())


def _mark_down(idx: int, reason: str, request_context: str = ""):
    cooloff = config.SETTINGS.cooloff_seconds
    _cooloff_until[idx] = time.monotonic() + cooloff
    _stats["failures"][idx] += 1
    ep = config.ENDPOINTS[idx]
    context = f" ({request_context})" if request_context else ""
    log.warning("endpoint %s marked down for %ss%s: %s", ep.base_url, cooloff, context, _clip(reason))


class UpstreamError(Exception):
    pass


class CapReached(UpstreamError):
    """A race racer lost the (provider, model) slot between candidate
    pre-check and send. Not an endpoint failure: the endpoint is healthy, so
    it must not be marked down."""


async def _send_upstream(req: httpx.Request, ep: Endpoint, *, stream: bool) -> httpx.Response:
    """Send the request, honoring the endpoint's ttfb_deadline_secs.

    A request queued behind a provider's concurrency limit is indistinguishable
    from a slow one: the provider withholds response headers until a slot frees,
    with no 429 and no keepalives. The header deadline is the only externally
    visible tripwire for that state."""
    if not ep.ttfb_deadline_secs:
        return await http_client.send(req, stream=stream)
    try:
        return await asyncio.wait_for(http_client.send(req, stream=stream), ep.ttfb_deadline_secs)
    except TimeoutError:
        raise UpstreamError(f"no response headers within {ep.ttfb_deadline_secs:g}s") from None


def _exception_detail(exc: BaseException) -> str:
    msg = str(exc)
    if isinstance(exc, UpstreamError) and msg:
        return msg
    if msg:
        return f"{type(exc).__name__}: {msg}"
    return type(exc).__name__


def _clip(text: str, limit: int = 200) -> str:
    """Collapse whitespace (log lines must stay single-line) and cap length."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


async def _close_quietly(resp: httpx.Response):
    """Release an upstream connection without masking the in-flight outcome."""
    try:
        await resp.aclose()
    except Exception:  # noqa: BLE001, S110 - cleanup must not mask the in-flight outcome
        pass


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


def _request_context(body: dict, group: str = "", req_id: str = "") -> str:
    """Slim per-request context for warnings: correlation id + routing shape."""
    parts = [f"req={req_id or '-'}"]
    if group:
        parts.append(f"group={group!r}")
    parts.append(f"stream={bool(body.get('stream', False))}")
    messages = body.get("messages")
    if isinstance(messages, list):
        parts.append(f"messages={len(messages)}")
    return " ".join(parts)


def _authenticate(authorization: str | None, req_id: str = "") -> tuple[str, JSONResponse | None]:
    """Returns (key name, error response). Name is empty when auth is disabled."""
    if not API_KEYS:
        return "", None
    token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    keyname = API_KEYS.get(hashlib.sha256(token.encode()).hexdigest()) if token else None
    if keyname is None:
        log.warning("req=%s auth failed: %s", req_id or "-", "missing header" if not authorization else "invalid token")
        return "", JSONResponse({"error": "unauthorized"}, status_code=401)
    return keyname, None


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
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    ct = usage.get("completion_tokens")
                    if ct is not None:
                        return ct
    return None


async def _proxy_stream(ep: Endpoint, path: str, headers: dict, body: bytes, metrics, request_context: str, on_done=None):
    """Stream response from upstream. Returns (StreamingResponse, None) or (None, reason).

    on_done (optional) fires once the stream has been fully consumed; every
    earlier exit leaves cleanup to the caller."""
    url = f"{ep.base_url}/{path}"
    t0 = time.monotonic()
    req = http_client.build_request("POST", url, headers=headers, content=body)
    try:
        resp = await _send_upstream(req, ep, stream=True)
    except UpstreamError as exc:
        reason = _exception_detail(exc)
        log.warning("upstream rejected streaming request to %s (%s): %s", ep.base_url, request_context, reason)
        return None, reason
    ttfb = time.monotonic() - t0
    metrics.ttfb_ms = ttfb * 1000

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
        # Always prime at least one chunk (more for OpenRouter via-detection):
        # an async generator that is never started does not run its finally —
        # not on aclose, not on GC — so starting generate() below before the
        # response is returned is what guarantees the slot release fires even
        # if the client disconnects before Starlette iterates the stream.
        try:
            while len(primed) < (8 if want_via else 1):
                chunk = await byte_iter.__anext__()
                if metrics.ttft_ms is None:
                    metrics.ttft_ms = (time.monotonic() - t0) * 1000
                    log.debug("req=%s %s TTFT %.0fms (TTFB %.0fms)", metrics.req_id, ep.provider, metrics.ttft_ms, ttfb * 1000)
                primed.append(chunk)
                if want_via:
                    via = _provider_from_sse(chunk)
                    if via:
                        break
        except StopAsyncIteration:
            pass
        metrics.via = via or ""

        async def generate():
            t_first = None
            completion_tokens = None
            sse_buf = bytearray()
            outcome = "200"
            try:
                yield b""  # placeholder, consumed by the priming step below
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
                        log.debug("req=%s %s TTFT %.0fms (TTFB %.0fms)", metrics.req_id, ep.provider, metrics.ttft_ms, ttfb * 1000)
                    ct = _extract_usage_from_sse(sse_buf, chunk)
                    if ct is not None:
                        completion_tokens = ct
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                # Client hung up mid-stream (or response was closed early).
                outcome = "aborted"
                raise
            except BaseException as exc:
                # Upstream died after headers were sent; client saw a truncated stream.
                outcome = "interrupted"
                metrics.reason = _clip(_exception_detail(exc))
                raise
            finally:
                if metrics.ttft_ms is not None and completion_tokens is not None and t_first is not None:
                    duration = time.monotonic() - t_first
                    if duration > 0:
                        metrics.tokens_per_sec = completion_tokens / duration
                metrics.tokens = completion_tokens
                metrics.status = outcome
                if on_done:
                    on_done()
                await asyncio.to_thread(requestlog.log_request, metrics)
                await _close_quietly(resp)

        # Start the generator so its finally (and the slot release) is armed
        # before anything can drop the response un-iterated.
        gen = generate()
        await gen.__anext__()
        result = _streaming_response(resp, gen)
        if via:
            result.headers["X-StableLLM-Via"] = via
        return result, None
    except BaseException:
        await _close_quietly(resp)
        raise


async def _proxy_buffered(ep: Endpoint, path: str, headers: dict, body: bytes, metrics, request_context: str):
    """Non-streaming: send request, return full response or (None, reason)."""
    url = f"{ep.base_url}/{path}"
    t0 = time.monotonic()
    req = http_client.build_request("POST", url, headers=headers, content=body)
    try:
        resp = await _send_upstream(req, ep, stream=True)
    except UpstreamError as exc:
        reason = _exception_detail(exc)
        log.warning("upstream rejected buffered request to %s (%s): %s", ep.base_url, request_context, reason)
        return None, reason
    metrics.ttfb_ms = (time.monotonic() - t0) * 1000
    try:
        if resp.status_code != 200:
            reason = await _http_error_reason(resp)
            log.warning("upstream rejected buffered request to %s (%s): %s", ep.base_url, request_context, reason)
            return None, reason
        content = await resp.aread()
    finally:
        await _close_quietly(resp)
    elapsed = time.monotonic() - t0

    try:
        data = json.loads(content)
    except Exception as exc:  # noqa: BLE001 - upstream garbage must degrade, not crash
        reason = f"invalid JSON: {_exception_detail(exc)}"
        detail = _body_snippet(content)
        if detail:
            reason = f"{reason}: {detail}"
        log.warning("upstream returned invalid JSON from %s (%s): %s", ep.base_url, request_context, reason)
        return None, reason

    metrics.elapsed_ms = elapsed * 1000
    usage = data.get("usage")
    tokens = None
    if usage:
        ct = usage.get("completion_tokens")
        if ct is not None and elapsed > 0:
            tokens = ct
            metrics.tokens_per_sec = ct / elapsed

    log.debug("req=%s %s TTFB %.0fms", metrics.req_id, ep.provider, elapsed * 1000)

    via = _openrouter_served_provider(data) if _openrouter_via(ep) else None
    metrics.via = via or ""
    metrics.tokens = tokens
    metrics.status = "200"
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


def _set_meta_headers(response, *, provider: str, model: str, mode: str, group: str, via: str | None = None, pin: str = ""):
    response.headers["X-StableLLM-Provider"] = provider
    response.headers["X-StableLLM-Model"] = model
    response.headers["X-StableLLM-Mode"] = mode
    response.headers["X-StableLLM-Group"] = group
    if via:
        response.headers["X-StableLLM-Via"] = via
    if pin:
        response.headers["X-StableLLM-Pin"] = pin


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
    "reasoning",
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


def _should_race(group: str) -> tuple[bool, str]:
    """(whether to race now, human-readable trigger reason)."""
    last_time = _group_last_race_time[group]
    count = _group_race_request_count[group]
    if last_time == 0.0:
        return True, "first request"
    if count >= config.SETTINGS.race_interval_requests:
        return True, f"{count} requests since last race (>= {config.SETTINGS.race_interval_requests})"
    if time.monotonic() - last_time >= config.SETTINGS.race_interval_secs:
        return True, f"{config.SETTINGS.race_interval_secs}s since last race"
    return False, ""


def _endpoint_label(ep: Endpoint) -> str:
    return ep.provider or ep.base_url


def _pk_label(group: str, pk: tuple[str, str]) -> str:
    """Display name for a (model, base_url) provider-group key."""
    indices = _group_provider_groups.get(group, {}).get(pk, [])
    if indices and indices[0] < len(config.ENDPOINTS):
        return _endpoint_label(config.ENDPOINTS[indices[0]])
    return pk[1]


def _finish_race(race_times: dict[tuple[str, str], float], group: str):
    sorted_keys = sorted(race_times, key=race_times.get)
    new_order = list(sorted_keys)
    for k in _group_provider_groups[group]:
        if k not in race_times:
            new_order.append(k)
    _group_preferred_providers[group] = new_order
    log.info(
        "race complete group=%s order=%s",
        group,
        " ".join(f"{_pk_label(group, k)}({v:.1f}s)" for k, v in sorted(race_times.items(), key=lambda x: x[1])),
    )


async def _race_request(path: str, body_dict: dict, is_streaming: bool, group: str, keyname: str, req_id: str, trigger: str):
    """Race one endpoint per provider group with real request. Returns response or None."""
    pg = _group_provider_groups[group]

    candidates: list[tuple[tuple[str, str], int]] = []
    for key, indices in pg.items():
        for idx in indices:
            ep = config.ENDPOINTS[idx]
            model_name = _effective_model(ep, body_dict.get("model", ""))
            if _is_available(idx) and not _at_cap(ep, model_name):
                candidates.append((key, idx))
                break

    if len(candidates) <= 1:
        log.debug("req=%s race: skipped (need 2+ available candidates, have %d)", req_id, len(candidates))
        return None, False

    # Reset race-cadence state on attempt so a failed race doesn't cause every
    # following request to retry the race against still-cooling-off endpoints.
    _group_race_request_count[group] = 0
    _group_last_race_time[group] = time.monotonic()
    _group_race_generation[group] += 1
    gen = _group_race_generation[group]

    race_context = _request_context(body_dict, group, req_id)
    log.debug(
        "req=%s race: trigger=%s candidates=%s gen=%d",
        req_id,
        trigger,
        [f"{_pk_label(group, pk)} model={_effective_model(config.ENDPOINTS[idx], body_dict.get('model', ''))}" for pk, idx in candidates],
        gen,
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
        model_name = _effective_model(ep, body_dict.get("model", ""))
        if not _try_acquire_slot(ep, model_name):
            # Candidates were pre-checked; this can only happen if another
            # route to the same (base_url, model) grabbed the slot meanwhile.
            raise CapReached(f"concurrency cap {ep.max_concurrency} reached")
        release = _slot_releaser(ep, model_name)
        try:
            headers = _build_upstream_headers(ep)
            stripped = _strip_unsupported(body_dict, ep)
            send_body = json.dumps(stripped).encode()
            url = f"{ep.base_url}/{path}"
            req = http_client.build_request("POST", url, headers=headers, content=send_body)
            resp = await _send_upstream(req, ep, stream=True)
        except BaseException:
            release()
            raise
        if resp.status_code != 200:
            reason = await _http_error_reason(resp)
            await _close_quietly(resp)
            release()
            raise UpstreamError(reason)
        return pk, idx, resp, release

    tasks = {asyncio.create_task(_send(pk, idx)): (pk, idx) for pk, idx in candidates}

    winner = None
    losers_to_drain: list[tuple] = []
    pending = set(tasks.keys())

    try:
        while pending and winner is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pk, idx = tasks[task]
                try:
                    rpk, ridx, resp, release = task.result()
                    if winner is None:
                        winner = (rpk, ridx, resp, release)
                    else:
                        losers_to_drain.append((rpk, ridx, resp, release))
                except CapReached as exc:
                    # Healthy endpoint, merely full: skip the mark-down.
                    race_state["failures"] += 1
                    log.debug("req=%s race: %s", req_id, exc)
                except Exception as exc:  # noqa: BLE001 - a failed racer must not abort the request
                    race_state["failures"] += 1
                    _mark_down(idx, _exception_detail(exc), _request_context(body_dict, group, req_id))
    except BaseException:
        # Request cancelled mid-race (client hangup): still-acquiring racers must
        # release their concurrency slots, not leak them.
        for t in pending:
            t.cancel()
        raise

    if winner is None:
        for t in pending:
            t.cancel()
        return None, True

    win_pk, win_idx, win_resp, win_release = winner
    ep = config.ENDPOINTS[win_idx]
    model_name = _effective_model(ep, body_dict.get("model", ""))
    log.debug("req=%s race: winner %s (model=%s) %.0fms", req_id, _endpoint_label(ep), model_name, (time.monotonic() - race_start) * 1000)
    _stats["requests"][win_idx] += 1

    race_metrics = requestlog.RequestMetrics(
        req_id=req_id,
        keyname=keyname,
        model_requested=body_dict.get("model", "") if body_dict else "",
        provider_served=_endpoint_label(ep),
        model_served=model_name,
        mode=config.MODE_RACE,
        stream=is_streaming,
        ttfb_ms=(time.monotonic() - race_start) * 1000,
    )

    async def _drain(pk, resp, idx, release):
        try:
            async for _ in resp.aiter_bytes():
                pass
            race_times[pk] = time.monotonic() - race_start
            log.debug("req=%s race: drain %s done %.1fs", req_id, _pk_label(group, pk), race_times[pk])
        except Exception as exc:  # noqa: BLE001 - drain failures are incidental
            race_state["failures"] += 1
            _mark_down(idx, _exception_detail(exc), _request_context(body_dict, group, req_id))
        finally:
            # Sync accounting before the close await: a cancellation arriving
            # during _close_quietly must not skip the slot release.
            release()
            _maybe_finalize()
            await _close_quietly(resp)

    async def _await_and_drain(task, pk, idx):
        try:
            _, _, resp, release = await task
        except CapReached as exc:
            race_state["failures"] += 1
            log.debug("req=%s race: %s", req_id, exc)
            _maybe_finalize()
            return
        except Exception as exc:  # noqa: BLE001 - a failed racer must not abort the request
            race_state["failures"] += 1
            _mark_down(idx, _exception_detail(exc), _request_context(body_dict, group, req_id))
            _maybe_finalize()
            return
        await _drain(pk, resp, idx, release)

    def _bg(coro):
        t = asyncio.create_task(coro)
        _background_tasks.add(t)
        t.add_done_callback(_background_tasks.discard)

    for pk, idx, resp, release in losers_to_drain:
        _bg(_drain(pk, resp, idx, release))
    for task in pending:
        pk, idx = tasks[task]
        _bg(_await_and_drain(task, pk, idx))

    if is_streaming:
        t0_race = race_start
        want_via = _openrouter_via(ep)
        byte_iter = win_resp.aiter_bytes()
        primed: list[bytes] = []
        via = None
        # Prime unconditionally: same never-started-generator slot leak as the
        # seq path if the response is dropped before iteration begins.
        try:
            while len(primed) < (8 if want_via else 1):
                chunk = await byte_iter.__anext__()
                if race_metrics.ttft_ms is None:
                    race_metrics.ttft_ms = (time.monotonic() - t0_race) * 1000
                primed.append(chunk)
                if want_via:
                    via = _provider_from_sse(chunk)
                    if via:
                        break
        except StopAsyncIteration:
            pass
        except (asyncio.CancelledError, GeneratorExit):
            # Request is dying (shutdown or client cancel): not the endpoint's
            # fault, so no mark-down. This aborted row is the terminal line;
            # the cancel propagates and no sequential fallback follows.
            race_metrics.status = "aborted"
            win_release()
            await asyncio.to_thread(requestlog.log_request, race_metrics)
            await _close_quietly(win_resp)
            raise
        except BaseException as exc:  # noqa: BLE001 - any priming failure must fail the race, not the request
            # Winner stalled mid-priming (transport error or similar). Close it
            # and fail the race so the caller falls back to sequential rather
            # than 500-ing the client / leaking the conn. No terminal row here:
            # the sequential outcome owns it.
            race_state["failures"] += 1
            _mark_down(win_idx, _exception_detail(exc), race_context)
            win_release()
            await _close_quietly(win_resp)
            _maybe_finalize()
            return None, True
        race_metrics.via = via or ""

        async def generate():
            t_first = None
            completion_tokens = None
            sse_buf = bytearray()
            outcome = "200"
            try:
                yield b""  # placeholder, consumed by the priming step below
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
            except (asyncio.CancelledError, GeneratorExit):
                outcome = "aborted"
                raise
            except BaseException as exc:
                outcome = "interrupted"
                race_metrics.reason = _clip(_exception_detail(exc))
                raise
            finally:
                if race_metrics.ttft_ms is not None and completion_tokens is not None and t_first is not None:
                    duration = time.monotonic() - t_first
                    if duration > 0:
                        race_metrics.tokens_per_sec = completion_tokens / duration
                race_metrics.tokens = completion_tokens
                race_metrics.status = outcome
                # Sync accounting first: a cancellation arriving during the awaits
                # below must not skip race finalization.
                race_times[win_pk] = time.monotonic() - race_start
                _maybe_finalize()
                win_release()
                await asyncio.to_thread(requestlog.log_request, race_metrics)
                await _close_quietly(win_resp)

        # Start the generator so its finally (and win_release) is armed before
        # anything can drop the response un-iterated.
        gen = generate()
        await gen.__anext__()
        result = _streaming_response(win_resp, gen)
        _stats["successes"][win_idx] += 1
        _set_meta_headers(result, provider=ep.provider, model=model_name, mode=config.MODE_RACE, group=group, via=via)
        return result, True
    else:
        chunks = []
        try:
            async for chunk in win_resp.aiter_bytes():
                chunks.append(chunk)
        except httpx.TransportError as exc:
            # Winner died mid-body; fail the race so the caller falls back to
            # sequential. No terminal row: the sequential outcome owns it.
            race_state["failures"] += 1
            _mark_down(win_idx, _exception_detail(exc), race_context)
            return None, True
        finally:
            # Sync accounting before the close await: a cancellation arriving
            # during _close_quietly must not skip the slot release.
            win_release()
            race_times[win_pk] = time.monotonic() - race_start
            _maybe_finalize()
            await _close_quietly(win_resp)
        elapsed = time.monotonic() - race_start
        race_metrics.elapsed_ms = elapsed * 1000
        response_body = b"".join(chunks)
        try:
            data = json.loads(response_body)
        except Exception as exc:  # noqa: BLE001 - upstream garbage must degrade, not crash
            reason = f"invalid JSON from race winner: {_exception_detail(exc)}"
            detail = _body_snippet(response_body)
            if detail:
                reason = f"{reason}: {detail}"
            race_metrics.status = "502"
            race_metrics.reason = _clip(reason)
            await asyncio.to_thread(requestlog.log_request, race_metrics)
            return JSONResponse({"error": "upstream returned invalid JSON"}, status_code=502), True
        usage = data.get("usage")
        tokens = None
        if usage:
            ct = usage.get("completion_tokens")
            if ct is not None and elapsed > 0:
                tokens = ct
                race_metrics.tokens_per_sec = ct / elapsed
        race_metrics.tokens = tokens
        via = _openrouter_served_provider(data) if _openrouter_via(ep) else None
        race_metrics.via = via or ""
        race_metrics.status = "200"
        await asyncio.to_thread(requestlog.log_request, race_metrics)
        result = JSONResponse(content=data, status_code=win_resp.status_code)
        if via:
            result.headers["X-StableLLM-Via"] = via
        _stats["successes"][win_idx] += 1
        _set_meta_headers(result, provider=ep.provider, model=model_name, mode=config.MODE_RACE, group=group, via=via)
        return result, True


@app.get("/health")
async def health():
    return {"status": "ok"}


def _usd_per_token(cost_per_million: float) -> str:
    """Convert $/million-tokens config value to OpenRouter's $/token string.
    Decimal-based so sub-$0.0001/M and high-precision costs aren't truncated."""
    return format(Decimal(str(cost_per_million)).scaleb(-6).normalize(), "f")


def _serialize_meta(meta: ModelMeta) -> dict:
    """Render a ModelMeta into OpenRouter-compatible /v1/models fields.
    Only present keys are emitted — unset values are omitted, not null."""
    entry: dict = {}
    if meta.name:
        entry["name"] = meta.name
    if meta.description:
        entry["description"] = meta.description
    if meta.context is not None:
        entry["context_length"] = meta.context

    architecture: dict = {"input_modalities": ["text"], "output_modalities": ["text"]}
    if meta.modalities:
        architecture["input_modalities"] = list(meta.modalities)
    architecture["modality"] = "+".join(architecture["input_modalities"]) + "->text"
    entry["architecture"] = architecture

    pricing: dict = {}
    if meta.input_cost is not None:
        pricing["prompt"] = _usd_per_token(meta.input_cost)
    if meta.output_cost is not None:
        pricing["completion"] = _usd_per_token(meta.output_cost)
    if meta.cache_read_cost is not None:
        pricing["input_cache_read"] = _usd_per_token(meta.cache_read_cost)
    if meta.cache_write_cost is not None:
        pricing["input_cache_write"] = _usd_per_token(meta.cache_write_cost)
    if pricing:
        entry["pricing"] = pricing

    # OpenRouter's is_moderated refers to its own moderation layer, not the
    # upstream model's — a proxy always reports false.
    if meta.context is not None or meta.max_output is not None:
        top_provider: dict = {"is_moderated": False}
        if meta.context is not None:
            top_provider["context_length"] = meta.context
        if meta.max_output is not None:
            top_provider["max_completion_tokens"] = meta.max_output
        entry["top_provider"] = top_provider

    if meta.supports_reasoning:
        reasoning: dict = {"mandatory": meta.reasoning_mandatory}
        if meta.reasoning_default_enabled is not None:
            reasoning["default_enabled"] = meta.reasoning_default_enabled
        if meta.reasoning_efforts:
            reasoning["supported_efforts"] = list(meta.reasoning_efforts)
        # OpenRouter emits default_effort with default_enabled: true when no
        # default is configured; kept mandatory for OpenRouter compatibility.
        reasoning["default_effort"] = meta.reasoning_default or (meta.reasoning_efforts[0] if meta.reasoning_efforts else "high")
        entry["reasoning"] = reasoning

    return entry


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(None)):
    _, auth_err = _authenticate(authorization)
    if auth_err:
        return auth_err
    return {
        "object": "list",
        "data": [
            # OpenAI keys always present (strict SDKs require them); OpenRouter
            # keys are additive when meta is configured.
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "stablellm",
                **(_serialize_meta(group.meta) if group.meta else {}),
            }
            for name, group in config.GROUPS.items()
        ],
    }


@app.get("/stats")
async def stats(authorization: str | None = Header(None)):
    _, auth_err = _authenticate(authorization)
    if auth_err:
        return auth_err
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
    # No filter: a zero or negative row is exactly the accounting bug signal we want visible.
    result["inflight"] = {f"{base_url} model={model!r}": n for (base_url, model), n in sorted(_inflight.items())}
    result["session_pins"] = len(_session_pins)
    result["session_pin_hits"] = _pin_promotions
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
    _group_race_generation.clear()  # invalidate in-flight races from the old config
    _session_pins.clear()  # pin values are endpoint indices; stale after reload
    global _pin_promotions
    _pin_promotions = 0


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
        return PlainTextResponse(await asyncio.to_thread(Path(config.CONFIG_FILE).read_text))
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
        await asyncio.to_thread(Path(config.CONFIG_FILE).write_text, new_content)
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
    req_id = secrets.token_hex(8)
    keyname, auth_err = _authenticate(authorization, req_id)
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
        log.warning("req=%s unknown model: requested_model=%r available=%s", req_id, model, sorted(config.GROUPS))
        return JSONResponse({"error": f"unknown model: '{model}'"}, status_code=404)

    mode = mode_override or config.GROUPS[group_name].mode

    log.debug(
        "req=%s -> model=%r group=%s mode=%s stream=%s keyname=%s",
        req_id, model, group_name, mode, is_streaming, keyname or "-",
    )

    dispatch_mode = mode
    if mode == config.MODE_RACE:
        _group_race_request_count[group_name] += 1

        should_race, trigger = _should_race(group_name)
        if should_race:
            result, raced = await _race_request(path, body_dict, is_streaming, group_name, keyname, req_id, trigger)
            if result is not None:
                result.headers["X-StableLLM-Pin"] = "none"  # races ignore pins by design
                return result
            if raced:
                dispatch_mode = "race-fallback"
                log.warning("race failed (%s), falling back to sequential", _request_context(body_dict, group_name, req_id))
            else:
                dispatch_mode = "seq"
        else:
            dispatch_mode = "seq"

        pref = _group_preferred_providers[group_name]
        pg = _group_provider_groups[group_name]
        endpoint_order = []
        for pk in pref:
            endpoint_order.extend(pg[pk])
    else:
        endpoint_order = config.GROUPS[group_name].endpoints

    last_failure = None
    session_key = _session_key(body_dict)
    order = list(endpoint_order)
    pinned = _pinned_endpoint(group_name, session_key)
    had_pin = pinned is not None
    pin_home = _endpoint_label(config.ENDPOINTS[pinned]) if had_pin else ""
    if pinned in order:
        # Prefer the endpoint that served this session's previous request so
        # upstream prompt caches stay warm.
        global _pin_promotions
        _pin_promotions += 1
        order.remove(pinned)
        order.insert(0, pinned)
        log.debug("req=%s pinned session -> %s", req_id, _endpoint_label(config.ENDPOINTS[pinned]))
    total = len(order)
    for attempt, idx in enumerate(order, 1):
        ep = config.ENDPOINTS[idx]
        if not _is_available(idx):
            log.debug("req=%s skipping %s (cooling off, %.0fs left)", req_id, ep.provider or ep.base_url, _cooloff_until[idx] - time.monotonic())
            continue

        client_model = body_dict.get("model", "")
        model_name = _effective_model(ep, client_model)

        if not _try_acquire_slot(ep, model_name):
            log.debug("req=%s skipping %s (model %r at concurrency cap %d)", req_id, ep.provider or ep.base_url, model_name, ep.max_concurrency)
            continue

        _stats["requests"][idx] += 1

        stripped = _strip_unsupported(body_dict, ep)
        send_body = json.dumps(stripped).encode()

        log.debug(
            "req=%s attempt %d/%d -> %s model=%s body_keys=%s bytes=%d",
            req_id, attempt, total, ep.provider or ep.base_url, model_name, list(stripped.keys()), len(send_body),
        )

        headers = _build_upstream_headers(ep)
        request_context = _request_context(body_dict, group_name, req_id)
        metrics = requestlog.RequestMetrics(
            req_id=req_id,
            keyname=keyname,
            model_requested=client_model,
            provider_served=ep.provider or ep.base_url,
            model_served=model_name,
            mode=dispatch_mode,
            stream=is_streaming,
        )

        release = _slot_releaser(ep, model_name)
        owns_release = True
        try:
            if is_streaming:
                result, reason = await _proxy_stream(ep, path, headers, send_body, metrics, request_context, on_done=release)
                if result is not None:
                    # Ownership moved to the response generator: the slot is
                    # held until the client has consumed the whole stream.
                    owns_release = False
            else:
                result, reason = await _proxy_buffered(ep, path, headers, send_body, metrics, request_context)

            if result is not None:
                _stats["successes"][idx] += 1
                _set_session_pin(group_name, session_key, idx)
                if had_pin:
                    pin = f"{'hit' if idx == pinned else 'bounce'}; home={pin_home}"
                elif session_key:
                    pin = f"new; home={_endpoint_label(ep)}"
                else:
                    pin = "none"
                _set_meta_headers(result, provider=ep.provider, model=model_name, mode=mode, group=group_name, pin=pin)
                return result

            _mark_down(idx, reason, request_context)
            last_failure = reason

        except httpx.TransportError as exc:
            reason = _exception_detail(exc)
            _mark_down(idx, reason, request_context)
            last_failure = reason
        finally:
            if owns_release:
                release()

    if last_failure is None:
        # Every endpoint was skipped (cooling off) -- nothing was actually tried.
        last_failure = "no endpoints available (all cooling off)"
    exhaustion = requestlog.RequestMetrics(
        req_id=req_id,
        keyname=keyname,
        model_requested=model,
        mode=dispatch_mode,
        stream=is_streaming,
        status="502",
        reason=f"all endpoints failed (last: {_clip(last_failure)})",
    )
    await asyncio.to_thread(requestlog.log_request, exhaustion)
    return JSONResponse(
        {"error": f"all endpoints exhausted (last: {last_failure})"},
        status_code=502,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT)
