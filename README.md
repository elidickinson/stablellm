# stablellm

OpenAI-compatible proxy that fans requests across multiple upstream providers with failover and optional latency racing.

## Configure

Two files. Bind-time settings live in `.env` (changing them requires a restart). Everything else lives in `config.yaml` and is reloadable via `GET /config/editor`.

### `.env` — bind-time settings

| Var | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `4000` | Server port |
| `REQUEST_TIMEOUT` | `120` | Outbound HTTP request timeout (seconds) |
| `CONNECT_TIMEOUT` | `4` | Outbound TCP connect timeout (seconds) |
| `API_KEY` | *(none)* | If set, clients must send `Authorization: Bearer <key>`. Comma-separated for multiple keys (see below) |
| `CONFIG_FILE` | `config.yaml` | Path to the YAML config |
| `MAX_BODY_BYTES` | `52428800` (50MB) | Max inbound request body size |
| `CONFIG_EDITOR_PASSWORD` | *(none)* | If set, enables the web config editor at `/config/editor` |
| `REQUEST_LOG_DB` | *(none)* | If set, SQLite request-logging is enabled at the given path |

API keys for upstream providers are set as individual vars here and referenced from YAML via `${VAR}` interpolation (e.g. `OPENAI_API_KEY`).

**Client keys.** `API_KEY` accepts a comma-separated list, so each client can hold its own revocable key. Prefix an entry with `name:` to label it; the name is recorded on every request the key makes (`keyname=` in the request summary log, `api_key_id` in the request log DB). Unlabelled keys get a stable `key-<hash>` id instead.

```
API_KEY=alice:sk-alice-secret,ci-bot:sk-ci-secret,sk-unlabelled
```

Leaving `API_KEY` unset disables client auth entirely.

> **Note:** secrets must not contain a colon. The `name:` prefix is split on the first `:` in each entry, so a secret like `sk-x:y` would be parsed as name `sk-x`, secret `y`.

### `config.yaml` — reloadable settings

```yaml
settings:
  cooloff_seconds: 30          # how long a failing endpoint is skipped
  race_interval_secs: 21600    # 6h — time between races (per group)
  race_interval_requests: 25   # request count between races (per group)
  race_settle_timeout_secs: 120  # hard cap for loser/header accounting from race start; must be > 0
  session_pin_ttl_secs: 900    # 15m — how long an idle session stays pinned to its endpoint
  log_level: INFO

providers:
  cerebras:
    base_url: https://api.cerebras.ai/v1
    api_key: ${CEREBRAS_API_KEY}

  openai:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}

groups:
  default:
    endpoints:
      - provider: openai
      - provider: cerebras
        model: zai-glm-4.7
        flags: [keep_reasoning]

  glm-4.7:
    mode: race
    endpoints:
      - provider: cerebras
        model: zai-glm-4.7
        flags: [keep_reasoning]
      - provider: openai
```

**`providers`** — a registry of upstream API endpoints. Each has a `base_url` and `api_key`. Optionally set `model` here as a provider-wide default — used when a group entry omits `model`. Three optional per-provider settings, all inheritable to (and overridable on) individual group entries:

- `max_concurrency` — maximum in-flight requests per model (0/unset = unlimited). When an endpoint is at its cap, routing skips it immediately instead of queueing behind providers like synthetic.new, which silently hold queued requests until a slot frees. Counted per `(provider, model)` across all groups. The slot is held until the response is fully consumed, including the whole lifetime of a stream.
- `ttfb_deadline_secs` — fail over if response headers don't arrive within this many seconds (0/unset = disabled). Queued requests are indistinguishable from slow ones — providers withhold headers while a request waits for a slot, with no error and no keepalives — so this is the only externally visible tripwire for queueing the proxy can't see (e.g. another client sharing the same API key).
- `routing` — passthrough mapping injected as the request's `provider` object, for OpenRouter's [provider-selection](https://openrouter.ai/docs/guides/routing/provider-selection) params (`sort`, `order`, `ignore`, `quantizations`, `max_price`, ...). Only meaningful for OpenRouter endpoints; other upstreams ignore the extra field. Injected after client params are stripped, so clients cannot override it.

**`groups`** — maps a request model name to a routing mode and an ordered list of upstream entries. Each group has:

- `mode` — `seq` (try in order, default) or `race` (send to all, first response wins). Can be overridden per-request with `:race`/`:seq` suffix on the model name.
- `endpoints` — ordered list of entries, each with:
  - `provider` — name from the providers section (required)
  - `model` — model name to send upstream. If omitted, falls back to the provider's `model` (if set), otherwise the client's requested model passes through unchanged.
  - `flags` — per-endpoint flags: `keep_reasoning` preserves `reasoning`/`reasoning_content`/`thinking` fields in messages (otherwise stripped).
  - `max_concurrency` / `ttfb_deadline_secs` / `routing` — per-endpoint overrides of the provider-level settings above. Concurrency is counted per `(provider, model)` across all groups: entries sharing a provider+model share one counter, so give them the same (smallest) cap.
- `meta` — optional descriptive metadata published on `/v1/models` in OpenRouter's response shape. When set, the entry uses OpenRouter keys (`context_length`, `architecture`, `pricing`, `top_provider`, `reasoning`) instead of the minimal OpenAI shape. All fields optional. Fields:
  - `name` / `description`
  - `context` / `max_output` — window and max completion tokens (tokens)
  - `modalities` — input modalities (e.g. `[text, image]`); output is reported as `text`
  - `input_cost` / `output_cost` / `cache_read_cost` / `cache_write_cost` — dollars per million tokens
  - `supports_reasoning` — whether reasoning is available (emits a `reasoning` block)
  - `reasoning_mandatory` — whether reasoning can't be disabled by the client
  - `reasoning_efforts` — supported effort levels
  - `reasoning_default` — default effort (must be one of `reasoning_efforts`; defaults to the first entry, or `"high"` when no efforts are declared)
  - `reasoning_default_enabled` — whether reasoning is on by default (omitted unless set)

`GET /v1/models` also publishes each group's `default_mode` (`seq` or `race`) so clients can expose routing variants without reading the server configuration.

**Group names match the client's `model` field case-insensitively.** `glm-4.7` matches `model: GLM-4.7`. Separators are not normalized, so `gpt-4.1` and `gpt_4_1` are distinct.

## Run

```
uv run uvicorn main:app --host $HOST --port $PORT
```

## Pi provider extension

[`pi-extension/`](pi-extension/) discovers this server's model groups, registers them as the `stablellm` provider, adds `:race` variants where useful, and displays the upstream route used for the last response. It has no built-in server address; configure one through `/login stablellm` or `STABLELLM_BASE_URL`. See [the extension README](pi-extension/README.md) for installation and authentication.

POST to `/v1/chat/completions` (or any path) like the OpenAI API. Every request must include a `model` field whose value matches a configured group name; otherwise the proxy returns 404.

## Routing

The request's `model` field selects the group. Within that group, the routing mode determines how endpoints are dispatched:

- **`seq`** (default) — try endpoints in order. A failing endpoint cools off for `cooloff_seconds` before being retried. If all endpoints fail, the request returns a 502.
- **`race`** — send the request to one endpoint per provider concurrently; the first response wins for the current request. The remaining responses are drained in the background, and once all candidate outcomes are accounted for their completion times update the preferred provider order. A re-race triggers when either `race_interval_requests` requests have passed or `race_interval_secs` seconds have elapsed since the last race (defaults: **25 requests** or **6 hours**) *and* the current request belongs to an unpinned session (see below). Between races, requests stay in `race` mode and use the current preferred order with normal failover. If a race fails, the proxy also falls back to that preferred order. Before any complete response anchors the budget, and for candidates still waiting for headers, `race_settle_timeout_secs` from race start is the hard cap. After the first complete response, an unfinished loser gets up to 50% of that response's elapsed time, with a 1-second minimum, subject to the same cap; it is moved to the end of the preferred order without being marked down.

The client can override a group's mode with a `:race` or `:seq` suffix on the model name (e.g. `glm-4.7:race`).

### Session pinning

Order-based routing is sticky per conversation: the session key (a hash of the client's `user` field, the first message, and the first user turn -- truncated so huge openers can't stall the proxy) is pinned to the endpoint that served its previous request, and that endpoint is tried first for `session_pin_ttl_secs` (default 15 minutes) after each use. This applies to sequential groups and to race groups. When something forces a failover (failure, cooloff, concurrency cap), the session bounces once and then re-pins to the new endpoint instead of re-contesting the old one — keeping upstream prompt caches warm.

A race would move a session off its home endpoint and throw away that warm cache, so races only run for unpinned sessions: a ripe race cadence waits for a new session (or one whose pin has expired) rather than firing at whoever asks next. A fresh session has no cache to lose, and the race winner becomes its pin. Clients whose system prompt changes every turn yield no stable session key, so they are always unpinned and race on cadence alone.

Pins are visible on `/dashboard` (`session_pins`) and are cleared on config reload.

## Response metadata

Every proxied response includes headers telling the client which upstream actually served the request:

| Header | Example | Description |
|---|---|---|
| `X-StableLLM-Provider` | `cerebras` | Provider name from config |
| `X-StableLLM-Model` | `zai-glm-4.7` | Model sent upstream (may differ from requested) |
| `X-StableLLM-Mode` | `race` | Routing policy selected (`seq` or `race`). A race-mode request between race attempts uses the preferred order without launching a new race. |
| `X-StableLLM-Group` | `glm-4.7` | Group the request resolved to |
| `X-StableLLM-Via` | `OpenAI` | Sub-provider that served the request — only present when routing through OpenRouter (see below) |
| `X-StableLLM-Pin` | `hit; home=synthetic` | Session pin state: `hit` (served by the session's pinned endpoint), `bounce` (pinned endpoint unavailable — served and re-pinned elsewhere; this turn is a cache miss), `new` (first request of a session, including a race winner), `none` (no derivable session) |

Headers are present on both streaming and non-streaming responses. They are exposed via CORS so browser clients can read them.

**OpenRouter sub-provider (`X-StableLLM-Via`).** OpenRouter itself fans a request out to one of several underlying providers (e.g. `OpenAI`, `Azure`, `Cerebras`). It tags every response body / stream chunk with that choice in a top-level `provider` field. When an endpoint's `base_url` points at `openrouter.ai`, stablellm reads that field and surfaces it as `X-StableLLM-Via`, so `X-StableLLM-Provider: openrouter` + `X-StableLLM-Via: Cerebras` means the request went through OpenRouter and was served by Cerebras. Non-OpenRouter endpoints don't set this header. (On streaming responses, stablellm peeks at the leading chunks to extract it before headers are committed.)

**Unknown request parameters** (not in the supported set) are silently stripped per-endpoint before forwarding. This lets providers with different capabilities share the same request body.

## Dashboard

`/dashboard` is a web UI gated by `CONFIG_EDITOR_PASSWORD` (the same password as the config editor). It shows one merged row per provider+model across groups -- state (up / cooling / manually down), in-flight vs cap, request/success/failure counters, and 15m/1h/24h request counts plus avg TTFT and tok/s computed from the request log -- along with a feed of recent requests and per-group race order. Each provider has a **mark down / bring up** button: a manual down pulls every group entry for that provider out of routing (new requests only; in-flight requests finish). Manual downs survive config reloads but not restarts.

Backing JSON, same auth: `GET /dashboard/api/state` and `GET /dashboard/api/history`. The request-log-backed views are empty unless `REQUEST_LOG_DB` is set.

## Deploy on Dokploy

The Docker image expects `config.yaml` to be mounted at `/app/config.yaml`. In Dokploy this file must be persisted outside the container so it survives redeployments.

1. **Create the application** — point Dokploy at this repo and let it build from the Dockerfile.

2. **Environment variables** — add `.env` vars (API keys, `CONFIG_EDITOR_PASSWORD`, etc.) in the Dokploy Environment tab.

3. **Persist `config.yaml`** — go to Advanced → Volumes/Mounts → **File Mount**:
   - **Content**: paste the contents of `config.example.yaml` (or your own config)
   - **File Path**: `config.yaml`
   - **Mount Path**: `/app/config.yaml`

   Dokploy stores file mounts in a host-side `files/` directory that persists across deploys. Since it's a bind mount, changes made via the web config editor (`/config/editor`) also persist.

4. **Redeploy** after adding the mount.

## Web config editor

Navigate to `/config/editor` and enter the password set in `CONFIG_EDITOR_PASSWORD`. The editor validates YAML and runs the full config parser before writing — invalid input is rejected without touching disk. On successful save, the new config is hot-reloaded in place; stats and cooloff state are reset since endpoint indices may have shifted.
