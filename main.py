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
from itertools import count
from pathlib import Path
from typing import ClassVar, Final
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
_session_pins: dict[tuple[str, str], tuple[int, str, float]] = {}
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

# Times a request was served by its session's pinned endpoint. Zero means
# pinning never engages (per-turn-volatile system prompts) OR the pinned home
# is persistently unavailable (constant bounces).
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
_group_race_generation: dict[str, int] = {}  # group -> id of its latest race
_race_ids = count(1)  # process-wide monotonic: never reused, even across reloads


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


# Providers manually disabled from the dashboard. Keyed by provider name (not
# endpoint index) so it survives config reloads; pruned to configured names on
# reload. Affects routing of new requests only; in-flight requests finish.
_manual_down: set[str] = set()
# Last failure reason per endpoint index, surfaced on the dashboard.
_last_failure: dict[int, str] = {}


def _is_available(idx: int) -> bool:
    return time.monotonic() >= _cooloff_until.get(idx, 0) and config.ENDPOINTS[idx].provider not in _manual_down


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
        parts.append(user[:_HASH_STR_CAP])
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


def _pinned_endpoint(group: str, skey: str) -> tuple[int, str] | None:
    if not skey:
        return None
    entry = _session_pins.get((group, skey))
    if entry is None:
        return None
    idx, home, ts = entry
    if time.monotonic() - ts > config.SETTINGS.session_pin_ttl_secs:
        del _session_pins[(group, skey)]
        return None
    if idx >= len(config.ENDPOINTS) or _endpoint_label(config.ENDPOINTS[idx]) != home:
        # Stale after a reload: the index is gone, or it now names a different
        # provider than the one that actually served this session.
        del _session_pins[(group, skey)]
        return None
    return idx, home


def _set_session_pin(group: str, skey: str, idx: int):
    if not skey:
        return
    if len(_session_pins) >= _PIN_TABLE_MAX and (group, skey) not in _session_pins:
        now = time.monotonic()
        ttl = config.SETTINGS.session_pin_ttl_secs
        for k in [k for k, (_, _, ts) in _session_pins.items() if now - ts > ttl]:
            del _session_pins[k]
        if len(_session_pins) >= _PIN_TABLE_MAX:
            del _session_pins[min(_session_pins, key=lambda k: _session_pins[k][2])]
    _session_pins[(group, skey)] = (idx, _endpoint_label(config.ENDPOINTS[idx]), time.monotonic())


def _mark_down(idx: int, reason: str, request_context: str = ""):
    _last_failure[idx] = _clip(reason)
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
    if model := body.get("model"):
        parts.append(f"model={model!r}")
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
        # OpenRouter tags every chunk with the serving sub-provider; prime up
        # to 8 chunks to find it. (The placeholder yield + __anext__ below arms
        # the generator's finally without any priming for other providers.)
        if want_via:
            try:
                while len(primed) < 8:
                    chunk = await byte_iter.__anext__()
                    if metrics.ttft_ms is None:
                        metrics.ttft_ms = (time.monotonic() - t0) * 1000
                        log.debug("req=%s %s TTFT %.0fms (TTFB %.0fms)", metrics.req_id, ep.provider, metrics.ttft_ms, ttfb * 1000)
                    primed.append(chunk)
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
        response_generator = generate()
        await response_generator.__anext__()
        result = _streaming_response(resp, response_generator)
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
    if ep.routing is not None:
        base["provider"] = {**ep.routing}  # OpenRouter provider-selection params
    return _rewrite_model(base, ep)


def _should_race(group: str, is_pinned: bool) -> tuple[bool, str]:
    """(whether to race now, human-readable trigger reason).

    A pinned session is never raced: it has a warm upstream prompt cache on its
    home endpoint and a race would move it off. Fresh sessions have nothing to
    lose, so a ripe cadence waits for one."""
    if is_pinned:
        return False, ""
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


_RACE_DRAIN_GRACE_RATIO = 0.5
_RACE_DRAIN_MIN_GRACE_SECS = 1.0


def _finish_race(
    race_times: dict[tuple[str, str], float],
    group: str,
    *,
    req_id: str = "-",
    generation: int | None = None,
    accounted: int | None = None,
    total: int | None = None,
):
    sorted_keys = sorted(race_times, key=lambda key: race_times[key])
    new_order = list(sorted_keys)
    for k in _group_provider_groups[group]:
        if k not in race_times:
            new_order.append(k)
    _group_preferred_providers[group] = new_order
    accounting = f"{accounted}/{total}" if accounted is not None and total is not None else "-"
    log.info(
        "req=%s race complete group=%s generation=%s accounted=%s order=%s",
        req_id,
        group,
        generation if generation is not None else "-",
        accounting,
        " ".join(
            f"{_pk_label(group, k)} model={k[0] or '-'}({v:.1f}s)"
            for k, v in sorted(race_times.items(), key=lambda item: item[1])
        ),
    )


async def _race_request(path: str, body_dict: dict, is_streaming: bool, group: str, keyname: str, req_id: str, trigger: str, session_key: str):
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
    # Keep the race generation separate from the response generator below.
    # The finalization closure captures this value while background drains run.
    race_generation: Final[int] = next(_race_ids)
    _group_race_generation[group] = race_generation

    race_context = _request_context(body_dict, group, req_id)
    log.debug(
        "req=%s race: trigger=%s candidates=%s generation=%d",
        req_id,
        trigger,
        [f"{_pk_label(group, pk)} model={_effective_model(config.ENDPOINTS[idx], body_dict.get('model', ''))}" for pk, idx in candidates],
        race_generation,
    )

    race_times: dict[tuple[str, str], float] = {}
    race_start = time.monotonic()
    race_loop = asyncio.get_running_loop()
    race_deadline = race_loop.time() + config.SETTINGS.race_settle_timeout_secs
    race_fastest_secs: float | None = None
    race_timeouts: set[asyncio.Timeout] = set()
    race_state = {"failures": 0}
    race_finished = False

    def _new_race_timeout() -> asyncio.Timeout:
        timeout = asyncio.timeout_at(race_deadline)
        race_timeouts.add(timeout)
        return timeout

    def _note_first_finish(elapsed: float):
        nonlocal race_deadline, race_fastest_secs
        if race_fastest_secs is not None:
            return
        race_fastest_secs = elapsed
        grace = max(elapsed * _RACE_DRAIN_GRACE_RATIO, _RACE_DRAIN_MIN_GRACE_SECS)
        race_deadline = min(race_deadline, race_loop.time() + grace)
        for timeout in tuple(race_timeouts):
            try:
                timeout.reschedule(race_deadline)
            except RuntimeError:
                # The timeout is already expiring; its owner will account it.
                race_timeouts.discard(timeout)
        log.debug(
            "req=%s race: fastest complete %.1fs; loser deadline in %.1fs",
            req_id,
            elapsed,
            max(0.0, race_deadline - race_loop.time()),
        )

    def _record_race_timeout(pk: tuple[str, str]):
        race_state["failures"] += 1
        log.debug(
            "req=%s race: candidate %s model=%s exceeded drain budget at %.1fs accounted=%d/%d",
            req_id,
            _pk_label(group, pk),
            pk[0] or "-",
            time.monotonic() - race_start,
            len(race_times) + race_state["failures"],
            len(candidates),
        )

    def _maybe_finalize():
        nonlocal race_finished
        # Skip if a newer race for this group has already started; otherwise late
        # background drains from this race could clobber the newer race's order.
        # Race ids are monotonic process-wide, so a drain from a pre-reload race
        # can never match a post-reload race's id.
        current_generation = _group_race_generation.get(group)
        if current_generation != race_generation:
            log.debug(
                "req=%s race: ignoring stale finalization generation=%d current=%s",
                req_id,
                race_generation,
                current_generation if current_generation is not None else "-",
            )
            return
        accounted = len(race_times) + race_state["failures"]
        if race_finished or accounted < len(candidates):
            return
        race_finished = True
        _finish_race(
            race_times,
            group,
            req_id=req_id,
            generation=race_generation,
            accounted=accounted,
            total=len(candidates),
        )

    async def _abandon(resp, release):
        """Consume an orphaned racer response: release its slot, drop the conn."""
        release()
        await _close_quietly(resp)

    def _bg(coro):
        t = asyncio.create_task(coro)
        _background_tasks.add(t)
        t.add_done_callback(_background_tasks.discard)

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
                    log.debug(
                        "req=%s race: %s accounted=%d/%d",
                        req_id,
                        exc,
                        len(race_times) + race_state["failures"],
                        len(candidates),
                    )
                except Exception as exc:  # noqa: BLE001 - a failed racer must not abort the request
                    race_state["failures"] += 1
                    _mark_down(idx, _exception_detail(exc), _request_context(body_dict, group, req_id))
                    log.debug(
                        "req=%s race: candidate %s model=%s failed accounted=%d/%d: %s",
                        req_id,
                        _pk_label(group, pk),
                        pk[0] or "-",
                        len(race_times) + race_state["failures"],
                        len(candidates),
                        _exception_detail(exc),
                    )
    except BaseException:
        # Request cancelled mid-race (client hangup). Cancel still-acquiring
        # racers, and harvest any that completed in the same instant asyncio.wait
        # raised -- their release would otherwise never run. (A completed task's
        # result that was never delivered is invisible to the done/pending sets.)
        for t in pending:
            t.cancel()
        for t, (pk, idx) in tasks.items():
            if t.done() and not t.cancelled():
                try:
                    _, _, resp, release = t.result()
                except Exception as exc:  # noqa: BLE001 - _send released internally
                    log.debug("req=%s race harvest: %s", req_id, _exception_detail(exc))
                    continue
                _bg(_abandon(resp, release))
        raise

    if winner is None:
        for t in pending:
            t.cancel()
        return None, True

    win_pk, win_idx, win_resp, win_release = winner
    ep = config.ENDPOINTS[win_idx]
    model_name = _effective_model(ep, body_dict.get("model", ""))
    race_pin = f"new; home={_endpoint_label(ep)}" if session_key else "none"
    log.debug(
        "req=%s race: winner %s (model=%s) ttfb=%.0fms; draining %d other candidate(s)",
        req_id,
        _endpoint_label(ep),
        model_name,
        (time.monotonic() - race_start) * 1000,
        len(candidates) - 1,
    )
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

    async def _drain(pk, resp, idx, release, timeout: asyncio.Timeout | None = None):
        owns_timeout = timeout is None
        if timeout is None:
            timeout = _new_race_timeout()

        async def _consume():
            async for _ in resp.aiter_bytes():
                pass
            race_times[pk] = time.monotonic() - race_start
            _note_first_finish(race_times[pk])
            log.debug(
                "req=%s race: drain %s model=%s done %.1fs accounted=%d/%d",
                req_id,
                _pk_label(group, pk),
                pk[0] or "-",
                race_times[pk],
                len(race_times) + race_state["failures"],
                len(candidates),
            )

        try:
            if owns_timeout:
                async with timeout:
                    await _consume()
            else:
                await _consume()
        except TimeoutError:
            _record_race_timeout(pk)
        except Exception as exc:  # noqa: BLE001 - drain failures are incidental
            race_state["failures"] += 1
            _mark_down(idx, _exception_detail(exc), _request_context(body_dict, group, req_id))
            log.debug(
                "req=%s race: drain %s model=%s failed accounted=%d/%d: %s",
                req_id,
                _pk_label(group, pk),
                pk[0] or "-",
                len(race_times) + race_state["failures"],
                len(candidates),
                _exception_detail(exc),
            )
        finally:
            if owns_timeout:
                race_timeouts.discard(timeout)
            # Sync accounting before the close await: a cancellation arriving
            # during _close_quietly must not skip the slot release.
            release()
            _maybe_finalize()
            await _close_quietly(resp)

    async def _await_and_drain(task, pk, idx):
        timeout = _new_race_timeout()
        response_claimed = False
        try:
            async with timeout:
                try:
                    _, _, resp, release = await task
                    response_claimed = True
                except CapReached as exc:
                    race_state["failures"] += 1
                    log.debug(
                        "req=%s race: %s accounted=%d/%d",
                        req_id,
                        exc,
                        len(race_times) + race_state["failures"],
                        len(candidates),
                    )
                    _maybe_finalize()
                    return
                except Exception as exc:  # noqa: BLE001 - a failed racer must not abort the request
                    race_state["failures"] += 1
                    _mark_down(idx, _exception_detail(exc), _request_context(body_dict, group, req_id))
                    log.debug(
                        "req=%s race: candidate %s model=%s failed accounted=%d/%d: %s",
                        req_id,
                        _pk_label(group, pk),
                        pk[0] or "-",
                        len(race_times) + race_state["failures"],
                        len(candidates),
                        _exception_detail(exc),
                    )
                    _maybe_finalize()
                    return
                await _drain(pk, resp, idx, release, timeout)
        except TimeoutError:
            if not response_claimed:
                if not task.done():
                    task.cancel()
                try:
                    _, _, resp, release = await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001 - retrieve a timed-out task's exception during cleanup
                    log.debug("req=%s race: timed-out candidate cleanup: %s", req_id, _exception_detail(exc))
                else:
                    await _abandon(resp, release)
            _record_race_timeout(pk)
            _maybe_finalize()
        finally:
            race_timeouts.discard(timeout)

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
                if outcome == "200":
                    _note_first_finish(race_times[win_pk])
                _maybe_finalize()
                win_release()
                await asyncio.to_thread(requestlog.log_request, race_metrics)
                await _close_quietly(win_resp)

        # Start the generator so its finally (and win_release) is armed before
        # anything can drop the response un-iterated.
        response_generator = generate()
        await response_generator.__anext__()
        result = _streaming_response(win_resp, response_generator)
        _stats["successes"][win_idx] += 1
        _set_session_pin(group, session_key, win_idx)
        _set_meta_headers(result, provider=ep.provider, model=model_name, mode=config.MODE_RACE, group=group, via=via, pin=race_pin)
        return result, True
    else:
        chunks = []
        winner_body_completed = False
        try:
            async for chunk in win_resp.aiter_bytes():
                chunks.append(chunk)
            winner_body_completed = True
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
            if winner_body_completed:
                _note_first_finish(race_times[win_pk])
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
        buffered_result = JSONResponse(content=data, status_code=win_resp.status_code)
        if via:
            buffered_result.headers["X-StableLLM-Via"] = via
        _stats["successes"][win_idx] += 1
        _set_session_pin(group, session_key, win_idx)
        _set_meta_headers(buffered_result, provider=ep.provider, model=model_name, mode=config.MODE_RACE, group=group, via=via, pin=race_pin)
        return buffered_result, True


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
                "default_mode": group.mode,
                **(_serialize_meta(group.meta) if group.meta else {}),
            }
            for name, group in config.GROUPS.items()
        ],
    }


def _merged_state_rows() -> list[dict]:
    """Endpoint entries merged into one row per (base_url, model).

    Counters and cooloffs are per endpoint entry (summed / worst-of here);
    inflight and max_concurrency are already per (base_url, model)."""
    entry_groups: dict[int, str] = {}
    for gname, g in config.GROUPS.items():
        for i in g.endpoints:
            entry_groups.setdefault(i, gname)
    rows: dict[tuple[str, str], dict] = {}
    now = time.monotonic()
    for idx, ep in enumerate(config.ENDPOINTS):
        key = (ep.base_url, ep.model)
        row = rows.get(key)
        if row is None:
            if ep.model:
                inflight = _inflight.get(key, 0)
            else:
                # Passthrough endpoints forward the client's model, so their
                # inflight keys are (base_url, <client model>); sum the base_url.
                inflight = sum(n for (base, _), n in _inflight.items() if base == ep.base_url)
            row = {
                "provider": ep.provider,
                "base_url": ep.base_url,
                "model": ep.model or "(passthrough)",
                "groups": [],
                "state": "up",
                "cooloff_secs_left": 0.0,
                "last_error": "",
                "inflight": inflight,
                "max_concurrency": 0,
                "requests": 0,
                "successes": 0,
                "failures": 0,
            }
            rows[key] = row
        row["groups"].append(entry_groups.get(idx, "?"))
        row["max_concurrency"] = max(row["max_concurrency"], ep.max_concurrency)
        row["requests"] += _stats["requests"].get(idx, 0)
        row["successes"] += _stats["successes"].get(idx, 0)
        row["failures"] += _stats["failures"].get(idx, 0)
        row["last_error"] = row["last_error"] or _last_failure.get(idx, "")
        if ep.provider in _manual_down:
            row["state"] = "down"
        elif now < _cooloff_until.get(idx, 0) and row["state"] != "down":
            row["state"] = "cooling"
            row["cooloff_secs_left"] = max(row["cooloff_secs_left"], _cooloff_until[idx] - now)
    return sorted(rows.values(), key=lambda r: (r["provider"], r["model"]))


@app.get("/dashboard/api/state")
async def dashboard_state(x_config_password: str | None = Header(None)):
    err = await _editor_auth(x_config_password)
    if err:
        return err
    groups = {}
    for name, group in config.GROUPS.items():
        groups[name] = {
            "mode": group.mode,
            "endpoints": [f"{config.ENDPOINTS[i].provider}/{config.ENDPOINTS[i].model or '(passthrough)'}" for i in group.endpoints],
            "preferred_providers": [{"model": m, "base_url": u} for m, u in _group_preferred_providers.get(name, [])],
            "requests_since_last_race": _group_race_request_count.get(name, 0),
        }
    return {
        "rows": _merged_state_rows(),
        "groups": groups,
        "manual_down": sorted(_manual_down),
        "session_pins": len(_session_pins),
        "session_pin_hits": _pin_promotions,
    }


@app.get("/dashboard/api/history")
async def dashboard_history(x_config_password: str | None = Header(None)):
    err = await _editor_auth(x_config_password)
    if err:
        return err
    recent = await asyncio.to_thread(requestlog.recent_requests, 50)
    summary = await asyncio.to_thread(requestlog.window_summary)
    return {"requests": recent, "summary": summary}


async def _dashboard_set_down(provider: str, down: bool, x_config_password: str | None):
    err = await _editor_auth(x_config_password)
    if err:
        return err
    name = provider.lower()  # provider names are lowercased at parse time
    if name not in {ep.provider for ep in config.ENDPOINTS}:
        return PlainTextResponse(f"unknown provider {name!r}", status_code=404)
    if down:
        _manual_down.add(name)
    else:
        _manual_down.discard(name)
    return {"provider": name, "down": name in _manual_down}


@app.post("/dashboard/api/down/{provider}")
async def dashboard_mark_down(provider: str, x_config_password: str | None = Header(None)):
    return await _dashboard_set_down(provider, True, x_config_password)


@app.post("/dashboard/api/up/{provider}")
async def dashboard_mark_up(provider: str, x_config_password: str | None = Header(None)):
    return await _dashboard_set_down(provider, False, x_config_password)


@app.get("/dashboard")
async def dashboard_page():
    if not config.CONFIG_EDITOR_PASSWORD:
        return PlainTextResponse("not found", status_code=404)
    return HTMLResponse(DASHBOARD_HTML)


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


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<title>stablellm dashboard</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #1e1e1e; color: #ccc; margin: 0; padding: 20px; }
  h1 { font-size: 1.1em; margin: 0 0 12px; font-weight: 500; }
  h2 { font-size: 0.9em; margin: 18px 0 6px; font-weight: 500; color: #999; }
  .bar { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
  input[type=password] { background: #2d2d2d; color: #ccc; border: 1px solid #444; padding: 6px 10px; font-family: monospace; border-radius: 3px; }
  input[type=password]:focus { outline: none; border-color: #0e639c; }
  button { background: #0e639c; color: white; border: none; padding: 4px 12px; cursor: pointer; font-size: 12px; border-radius: 3px; }
  button:hover { background: #1177bb; }
  table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
  th { text-align: left; color: #777; font-weight: 500; padding: 3px 10px 3px 0; border-bottom: 1px solid #333; }
  td { padding: 4px 10px 4px 0; border-bottom: 1px solid #2a2a2a; vertical-align: top; }
  td.num, th.num { text-align: right; font-family: monospace; }
  .mono { font-family: monospace; }
  .pill { font-weight: 600; }
  .up { color: #7cc87c; } .cooling { color: #e5c07b; } .down { color: #e06c75; }
  .hot { color: #e06c75; font-weight: 600; }
  .ok { color: #7cc87c; } .warn { color: #e5c07b; } .err { color: #e06c75; }
  .dim { color: #666; }
  #status { margin-top: 10px; padding: 8px 12px; border-radius: 3px; min-height: 1.2em; font-family: monospace; font-size: 12px; }
  #status.err { background: #5a1d1d; color: #f5a5a5; }
  .hint { color: #666; font-size: 12px; }
  details { margin-top: 14px; font-size: 12px; color: #888; }
  summary { cursor: pointer; }
</style>
</head>
<body>
<h1>stablellm dashboard</h1>
<div class="bar">
  <input type="password" id="pw" placeholder="password" autofocus>
  <button id="load">Load</button>
  <span id="updated" class="dim"></span>
  <span class="hint">manual down survives config reload, not restarts</span>
</div>
<div id="content" style="display:none">
  <h2>providers</h2>
  <table id="providers"></table>
  <h2>endpoints (provider / model)</h2>
  <table id="endpoints"></table>
  <h2>recent requests</h2>
  <table id="reqs"></table>
  <details><summary>groups / race state</summary><div id="groups"></div></details>
</div>
<div id="status" class="dim">Enter password and click Load</div>
<script>
const $ = id => document.getElementById(id);
let pw = '', stateData = null, historyData = null, stateTimer = null, histTimer = null, paintTimer = null;

function setStatus(text, cls) { const s = $('status'); s.textContent = text; s.className = cls || 'dim'; }

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { 'X-Config-Password': pw, ...(opts.headers || {}) } });
  if (r.status === 401) { setStatus('unauthorized: wrong password', 'err'); throw new Error('unauthorized'); }
  if (!r.ok) throw new Error(path + ' -> HTTP ' + r.status);
  return r.json();
}

function el(tag, text, cls) { const n = document.createElement(tag); if (text !== null && text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; }

function pill(state, secsLeft) {
  if (state === 'down') return el('span', '\u25cf DOWN', 'pill down');
  if (state === 'cooling') return el('span', '\u25d0 ' + Math.max(0, secsLeft).toFixed(0) + 's', 'pill cooling');
  return el('span', '\u25cf UP', 'pill up');
}

function fmtMs(v) { return v == null ? '\u2013' : (v / 1000).toFixed(1) + 's'; }
function fmtTps(v) { return v == null ? '\u2013' : v.toFixed(1) + '/s'; }

function renderProviders() {
  if (!stateData) return;
  const t = $('providers'); t.textContent = '';
  const byProv = {};
  for (const r of stateData.rows) {
    const p = byProv[r.provider] || (byProv[r.provider] = { state: 'up', secs: 0, err: '' });
    if (r.state === 'down') p.state = 'down';
    else if (r.state === 'cooling' && p.state !== 'down') { p.state = 'cooling'; p.secs = Math.max(p.secs, r.secsLeft); }
    p.err = p.err || r.last_error;
  }
  const head = el('tr'); for (const h of ['provider', 'state', 'last error', '']) head.appendChild(el('th', h));
  t.appendChild(head);
  for (const [prov, p] of Object.entries(byProv).sort()) {
    const tr = el('tr');
    tr.appendChild(el('td', prov, 'mono'));
    tr.appendChild(el('td')).appendChild(pill(p.state, p.secs));
    tr.appendChild(el('td', p.err, 'err'));
    const act = el('td');
    const btn = el('button', p.state === 'down' ? 'bring up' : 'mark down');
    btn.addEventListener('click', async () => {
      try { await api('/dashboard/api/' + (p.state === 'down' ? 'up/' : 'down/') + prov, { method: 'POST' }); await refreshState(); }
      catch (e) { setStatus(String(e), 'err'); }
    });
    act.appendChild(btn); tr.appendChild(act);
    t.appendChild(tr);
  }
}

function renderEndpoints() {
  if (!stateData || !historyData) return;
  const t = $('endpoints'); t.textContent = '';
  const head = el('tr');
  for (const [h, cls] of [['provider / model'], ['groups'], ['inflight', 'num'], ['r/s/f', 'num'], ['15m \u00b7 1h \u00b7 24h (reqs \u00b7 ttft \u00b7 tok/s)']])
    head.appendChild(Object.assign(el('th', h), cls ? { className: cls } : {}));
  t.appendChild(head);
  for (const r of stateData.rows) {
    const tr = el('tr');
    tr.appendChild(el('td', r.provider + '  ' + r.model, 'mono'));
    tr.appendChild(el('td', [...new Set(r.groups)].join(', ')));
    const capped = r.max_concurrency > 0 && r.inflight >= r.max_concurrency;
    tr.appendChild(el('td', r.inflight + '/' + (r.max_concurrency > 0 ? r.max_concurrency : '\u2013'), 'num mono' + (capped ? ' hot' : '')));
    tr.appendChild(el('td', r.requests + '/' + r.successes + '/' + r.failures, 'num mono'));
    const s = historyData.summary[r.provider + '|' + r.model];
    const cell = el('td', null, 'mono');
    if (!s) { cell.appendChild(el('span', '\u2013', 'dim')); }
    else for (const label of ['15m', '1h', '24h']) {
      cell.appendChild(el('div', label + '  ' + (s.reqs[label] || 0) + ' \u00b7 ' + fmtMs(s.ttft_ms[label]) + ' \u00b7 ' + fmtTps(s.tok_s[label])));
    }
    tr.appendChild(cell);
    t.appendChild(tr);
  }
}

function renderReqs() {
  if (!historyData) return;
  const t = $('reqs'); t.textContent = '';
  const head = el('tr');
  for (const [h, cls] of [['time'], ['client'], ['group'], ['served by'], ['mode'], ['status'], ['ttft', 'num'], ['tok/s', 'num'], ['reason']])
    head.appendChild(Object.assign(el('th', h), cls ? { className: cls } : {}));
  t.appendChild(head);
  for (const r of historyData.requests) {
    const tr = el('tr');
    const cls = r.status === '200' ? 'ok' : (r.status === 'aborted' ? 'warn' : 'err');
    tr.appendChild(el('td', new Date(r.ts * 1000).toLocaleTimeString(), 'mono'));
    tr.appendChild(el('td', r.client || '\u2013'));
    tr.appendChild(el('td', r.group, 'mono'));
    tr.appendChild(el('td', (r.provider || '\u2013') + ' ' + (r.model || ''), 'mono'));
    tr.appendChild(el('td', r.mode || ''));
    tr.appendChild(el('td', r.status, cls));
    tr.appendChild(el('td', r.ttft_ms == null ? '\u2013' : (r.ttft_ms / 1000).toFixed(1) + 's', 'num mono'));
    tr.appendChild(el('td', r.tokens_per_sec == null ? '\u2013' : r.tokens_per_sec.toFixed(1), 'num mono'));
    tr.appendChild(el('td', r.reason || '', r.reason ? 'err' : ''));
    t.appendChild(tr);
  }
}

function renderGroups() {
  if (!stateData) return;
  const d = $('groups'); d.textContent = '';
  for (const [name, g] of Object.entries(stateData.groups)) {
    const line = name + ' [' + g.mode + ']  ' + g.endpoints.join(' > ')
      + '   preferred: ' + (g.preferred_providers.map(p => p.model + '@' + p.base_url.replace('https://', '')).join(', ') || '(none)')
      + '   reqs since race: ' + g.requests_since_last_race;
    d.appendChild(el('div', line, 'mono'));
  }
  d.appendChild(el('div', 'session pins: ' + stateData.session_pins + ' \u00b7 pin hits: ' + stateData.session_pin_hits));
}

// Endpoints store wall-clock deadlines so the countdown ticks between polls.
function captureDeadlines() {
  for (const r of stateData.rows) r.secsLeft = r.cooloff_secs_left;
  stateData.fetchedAt = Date.now();
}

function paint() {
  if (!stateData) return;
  const drift = (Date.now() - stateData.fetchedAt) / 1000;
  for (const r of stateData.rows) if (r.state === 'cooling') r.secsLeft = Math.max(0, r.cooloff_secs_left - drift);
  renderProviders(); renderEndpoints(); renderReqs(); renderGroups();
  $('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
}

async function refreshState() { stateData = await api('/dashboard/api/state'); captureDeadlines(); }
async function refreshHistory() { historyData = await api('/dashboard/api/history'); }

async function start() {
  pw = $('pw').value;
  try {
    await Promise.all([refreshState(), refreshHistory()]);
  } catch (e) { if (e.message !== 'unauthorized') setStatus(String(e), 'err'); return; }
  $('content').style.display = '';
  setStatus('');
  clearInterval(stateTimer); clearInterval(histTimer); clearInterval(paintTimer);
  paint();
  stateTimer = setInterval(() => refreshState().catch(e => setStatus(String(e), 'err')), 1000);
  histTimer = setInterval(() => refreshHistory().catch(e => setStatus(String(e), 'err')), 3000);
  paintTimer = setInterval(paint, 500);
}

$('load').addEventListener('click', start);
$('pw').addEventListener('keydown', e => { if (e.key === 'Enter') start(); });
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
    # Clearing race generations invalidates late drains from pre-reload races
    # (the guard is equality); ids stay monotonic so post-reload races can't
    # collide with pre-reload ones either. Both halves are required.
    _group_race_generation.clear()
    _session_pins.clear()  # pin values are endpoint indices; stale after reload
    global _pin_promotions
    _pin_promotions = 0
    _last_failure.clear()  # keyed by endpoint index; stale after reload
    # Manual downs are name-keyed and survive reloads, but a provider that no
    # longer exists must not stay disabled invisibly.
    _manual_down.intersection_update(ep.provider for ep in config.ENDPOINTS)


EDITOR_AUTH_DELAY_SECS = 0.5


async def _editor_auth(password: str | None) -> PlainTextResponse | None:
    if not config.CONFIG_EDITOR_PASSWORD:
        return PlainTextResponse("not found", status_code=404)
    # Constant-time compare; the delay runs only on failure to slow brute-force
    # without taxing every successful call (the dashboard polls at 1s).
    ok = bool(password) and hmac.compare_digest(password, config.CONFIG_EDITOR_PASSWORD)
    if not ok:
        await asyncio.sleep(EDITOR_AUTH_DELAY_SECS)
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
        "req=%s -> model=%r group=%s configured_mode=%s stream=%s keyname=%s",
        req_id, model, group_name, mode, is_streaming, keyname or "-",
    )

    session_key = _session_key(body_dict)
    pinned = _pinned_endpoint(group_name, session_key)

    if mode == config.MODE_RACE:
        _group_race_request_count[group_name] += 1

        should_race, trigger = _should_race(group_name, pinned is not None)
        if should_race:
            result, raced = await _race_request(path, body_dict, is_streaming, group_name, keyname, req_id, trigger, session_key)
            if result is not None:
                return result
            if raced:
                log.warning(
                    "req=%s race failed group=%s model=%r stream=%s; falling back to preferred order",
                    req_id,
                    group_name,
                    model,
                    is_streaming,
                )
            else:
                log.debug("req=%s race: skipped; using preferred order", req_id)
        else:
            age = time.monotonic() - _group_last_race_time[group_name]
            log.debug(
                "req=%s race: deferred; using preferred order pinned=%s requests_since_last=%d/%d age=%.0fs/%ss",
                req_id,
                pinned[1] if pinned else "-",
                _group_race_request_count[group_name],
                config.SETTINGS.race_interval_requests,
                age,
                config.SETTINGS.race_interval_secs,
            )

        pref = _group_preferred_providers[group_name]
        pg = _group_provider_groups[group_name]
        endpoint_order = []
        for pk in pref:
            endpoint_order.extend(pg[pk])
    else:
        endpoint_order = config.GROUPS[group_name].endpoints

    last_failure = None
    order = list(endpoint_order)
    had_pin = pinned is not None
    pinned_idx, pin_home = pinned if pinned else (None, "")
    global _pin_promotions
    if pinned_idx in order:
        # Prefer the endpoint that served this session's previous request so
        # upstream prompt caches stay warm.
        order.remove(pinned_idx)
        order.insert(0, pinned_idx)
        log.debug("req=%s pinned session -> %s", req_id, pin_home)
    total = len(order)
    for attempt, idx in enumerate(order, 1):
        ep = config.ENDPOINTS[idx]
        if not _is_available(idx):
            if ep.provider in _manual_down:
                why = "manually disabled via dashboard"
            else:
                why = f"cooling off, {_cooloff_until.get(idx, 0) - time.monotonic():.0f}s left"
            log.debug("req=%s skipping %s (%s)", req_id, ep.provider or ep.base_url, why)
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
            mode=mode,
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
                    pin = f"{'hit' if idx == pinned_idx else 'bounce'}; home={pin_home}"
                    if idx == pinned_idx:
                        _pin_promotions += 1
                elif session_key:
                    pin = f"new; home={_endpoint_label(ep)}"
                else:
                    pin = "none"
                _set_meta_headers(result, provider=ep.provider, model=model_name, mode=mode, group=group_name, pin=pin)
                return result

            assert reason is not None
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
        last_failure = "no endpoints available (all cooling off, manually disabled, or at concurrency cap)"
    exhaustion = requestlog.RequestMetrics(
        req_id=req_id,
        keyname=keyname,
        model_requested=model,
        mode=mode,
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
