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
  race_settle_timeout_secs: 120  # hard cap for race completion/drain accounting from race start; must be > 0
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

**`providers`** — an optional registry of upstream API endpoints. `openrouter` is always predefined with base URL `https://openrouter.ai/api/v1` and the `OPENROUTER_API_KEY` environment variable; declaring it overrides supplied fields, so an `api_key`-only entry inherits the default URL. Other provider entries require `base_url` and `api_key`. Optionally set `model` here as a provider-wide default — used when a group entry omits `model`. Three optional per-provider settings, all inheritable to (and overridable on) individual group entries:

- `max_concurrency` — maximum in-flight requests per model (0/unset = unlimited). When an endpoint is at its cap, routing skips it immediately instead of queueing behind providers like synthetic.new, which silently hold queued requests until a slot frees. Counted per `(provider, model)` across all groups. The slot is held until the response is fully consumed, including the whole lifetime of a stream.
- `ttfb_deadline_secs` — fail over if response headers don't arrive within this many seconds (0/unset = disabled). Queued requests are indistinguishable from slow ones — providers withhold headers while a request waits for a slot, with no error and no keepalives — so this is the only externally visible tripwire for queueing the proxy can't see (e.g. another client sharing the same API key).
- `routing` — passthrough mapping injected as the request's `provider` object, for OpenRouter's [provider-selection](https://openrouter.ai/docs/guides/routing/provider-selection) params (`sort`, `order`, `ignore`, `quantizations`, `max_price`, ...). Valid only when the provider name is `openrouter`; config parsing rejects it for every other provider. Injected after client params are stripped, so clients cannot override it.

**`groups`** — maps a request model name to a routing mode and an ordered list of upstream entries. Each group has:

- `mode` — `seq` (try in order, default) or `race` (send to all, first fully-completed response wins). Can be overridden per-request with `:race`/`:seq` suffix on the model name.
- `endpoints` — ordered list of entries, each with:
  - `provider` — name from the providers section, or the predefined `openrouter` provider (required)
  - `model` — model name to send upstream. If omitted, falls back to the provider's `model` (if set), otherwise the client's requested model passes through unchanged.
  - `flags` — per-endpoint flags: `keep_reasoning` preserves `reasoning`/`reasoning_content`/`thinking` fields in messages (otherwise stripped).
  - `reasoning_effort` — effort level (e.g. `high`) injected as the top-level `reasoning_effort` param when the client sends no reasoning params of its own. Independent of group `meta` reasoning fields, which only advertise capabilities on `/v1/models`.
  - `reasoning_force` — with `reasoning_effort`, also override client-sent `reasoning`/`reasoning_effort`.
  - `max_concurrency` / `ttfb_deadline_secs` / `routing` — per-endpoint overrides of the provider-level settings above. Concurrency is counted per `(provider, model)` across all groups: entries sharing a provider+model share one counter, so give them the same (smallest) cap.
  - `performance_routing` — OpenRouter only, and endpoint-level only (not inheritable from `providers`). Filters the model's live OpenRouter endpoint catalog down to an eligible population, then ranks that — not the other way around. Two derived constraints, both on by default: a price cap (`price_cap_tolerance`, default `0.15`: accept only rows whose `pricing.prompt` AND `pricing.completion` each sit within the tolerance above that field's median, sent as OpenRouter's `provider.max_price` in dollars per million tokens) and a quantization floor (`quantization_floor`, default `auto`: the mode of the observed bit widths, ties to the lower tier, sent as `provider.quantizations` in short form; `include_unknown_quantization`, default `true`, decides whether unknown-width rows pass). `quantization_floor` has three distinct states: `auto` derives the floor from the catalog, an explicit tier (`4`/`8`/`16`/`32`) states a requirement instead of trusting the catalog, and the off spellings (`none`/`None`/`null`/`~`/blank) disable the rule entirely — `none` is never treated as `auto`. The off spellings also disable the price rule (`0` is a real tolerance, not off). Within the eligible rows the block still scores projected time to generate `target_tokens` (default `10000`) tokens from p50 latency and throughput, and allows those within `speed_tolerance` (default `0.15`) of the fastest as `provider.only`. A static `routing.max_price`/`quantizations` is honored verbatim and governs the filtering too — including when the matching derived rule is turned off; `routing.only` narrows the ranked result, and a static `only` naming no surviving tag is dropped (the constraints are hard, the speed optimization is not) rather than sent to be 404'd; `order`/`sort` are stripped; every other key (including `allow_fallbacks` and the `preferred_*` preferences) passes through. Rows with no measurements are eligible but can never rank, so `only` is simply omitted when nothing measurable survives. An endpoint whose constraints exclude every catalog row is skipped for that request (`reason=empty-derivation`) and the group falls to the next endpoint. The catalog is cached per model for `cache_ttl_seconds` (default `900`); if OpenRouter rejects the derived selection, the request retries once without it. Example: `performance_routing: {}` takes all defaults.
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

[`pi-extension/`](pi-extension/) discovers this server's model groups, registers them as the `stablellm` provider, adds `:race` variants where useful, and displays the upstream route used for the last response. It also follows the race redirect handshake transparently and shows `StableLLM is racing providers...` while a race is pending, so the OpenAI client never has to understand the `307`. It has no built-in server address; configure one through `/login stablellm` or `STABLELLM_BASE_URL`. See [the extension README](pi-extension/README.md) for installation and authentication.

POST to `/v1/chat/completions` (or any path) like the OpenAI API. Every request must include a `model` field whose value matches a configured group name; otherwise the proxy returns 404.

## Routing

The request's `model` field selects the group. Within that group, the routing mode determines how endpoints are dispatched:

- **`seq`** (default) — try endpoints in order. A failing endpoint cools off for `cooloff_seconds` before being retried. If all endpoints fail, the request returns a 502.
- **`race`** — send the request to one endpoint per provider group concurrently and wait for each candidate's **entire response body** to complete. The first candidate to finish successfully wins; the other responses are drained in the background, and once all candidate outcomes are accounted for their completion times update the preferred provider order. HTTP errors, stream errors, cancellation, and malformed JSON or SSE bodies can never win. The winner's complete response is delivered to the client afterward in one burst, so a race trades time-to-first-token for a reliable measure of which provider finishes fastest. A re-race triggers when either `race_interval_requests` requests have passed or `race_interval_secs` seconds have elapsed since the last race (defaults: **25 requests** or **6 hours**) *and* the current request belongs to an unpinned session (see below). Between races, requests stay in `race` mode and use the current preferred order with normal failover. If no candidate completes successfully, the proxy falls back to that preferred order. Before any successful completion, `race_settle_timeout_secs` from race start is the hard cap for all candidates. After one, unfinished racers get up to 50% of the winner's elapsed time, with a 1-second minimum, subject to the same cap; a loser that overruns is moved to the end of the preferred order without being marked down.

A race is confirmed by a redirect handshake: an eligible race request returns `307 Temporary Redirect` to the same path with a `stablellm_race_redirect=1` query parameter and `X-StableLLM-Race: pending`, before any upstream request is launched. Clients that follow it (the Pi extension does) get the race; clients that don't simply see the 307. The logical request is counted once, on the initial request; the marked follow-up re-checks pin, cadence, availability, and caps and only races if still eligible.

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
| `X-StableLLM-Race` | `pending` | Present only on the `307` race handshake, confirming an eligible race request was redirected to its marked follow-up |
| `X-StableLLM-Race-Candidates` | `3` | Present only on the `307` race handshake: number of provider-group candidates that would race |

Headers are present on both streaming and non-streaming responses. They are exposed via CORS so browser clients can read them.

**OpenRouter sub-provider (`X-StableLLM-Via`).** OpenRouter itself fans a request out to one of several underlying providers (e.g. `OpenAI`, `Azure`, `Cerebras`). It tags every response body / stream chunk with that choice in a top-level `provider` field. When an endpoint uses the provider named `openrouter`, stablellm reads that field and surfaces it as `X-StableLLM-Via`, even when its configured URL is a gateway or cache in front of OpenRouter. Non-OpenRouter endpoints don't set this header. (On race streaming responses, the buffered winner SSE is scanned before delivery.)

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
