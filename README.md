# stablellm

OpenAI-compatible proxy that fans requests across multiple upstream endpoints with failover and optional latency racing.

## Configure

Two files. Bind-time settings live in `.env` (changing them requires a restart). Everything else lives in `config.yaml` and is reloadable via `GET /config/editor`.

### `.env` — bind-time settings

| Var | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `4000` | Server port |
| `REQUEST_TIMEOUT` | `120` | Outbound HTTP request timeout (seconds) |
| `CONNECT_TIMEOUT` | `4` | Outbound TCP connect timeout (seconds) |
| `API_KEY` | *(none)* | If set, clients must send `Authorization: Bearer <API_KEY>` |
| `CONFIG_FILE` | `config.yaml` | Path to the YAML config |
| `MAX_BODY_BYTES` | `52428800` (50MB) | Max inbound request body size |
| `CONFIG_EDITOR_PASSWORD` | *(none)* | If set, enables the web config editor at `/config/editor` |

API keys for upstream providers are set as individual vars here and referenced from YAML via `${VAR}` interpolation (e.g. `OPENAI_API_KEY`).

### `config.yaml` — reloadable settings

```yaml
settings:
  cooloff_seconds: 30           # how long a failing endpoint is skipped
  race_interval_secs: 21600     # 6h — time between races (per group)
  race_interval_requests: 25    # request count between races (per group)
  log_level: INFO

endpoints:
  openai:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
  cerebras:
    base_url: https://api.cerebras.ai/v1
    api_key: ${CEREBRAS_API_KEY}
    model: llama-3.3-70b        # overrides the client-requested model
    flags: [keep_reasoning]

groups:
  default: [openai]
  cheap: [cerebras]
```

**Per-endpoint `model`** is optional. If empty, the client's requested model passes through unchanged.

**Per-endpoint `flags`:**
- `keep_reasoning` — preserve `reasoning`/`reasoning_content`/`thinking` fields in messages (otherwise stripped).

**Unknown request parameters** (not in the supported set) are silently stripped per-endpoint before forwarding. This lets endpoints with different capabilities share the same request body.

## Run

```
uv run uvicorn main:app --host $HOST --port $PORT
```

POST to `/v1/chat/completions` (or any path) like the OpenAI API.

## Routing

Every request resolves to a group:

1. If the request `model` matches a group name, that group's endpoint list is used.
2. Otherwise the `default` group is used (all endpoints in declaration order, unless overridden).

Within a group, endpoints are tried in order. A failing endpoint cools off for `cooloff_seconds` before being retried. If all endpoints fail, the request returns a 502.

Group lookup is case-insensitive and treats `-`, `.`, `_` as equivalent — YAML key `gpt_4_1` matches request model `gpt-4.1`.

### Fastest mode

Append `:fastest` to the model (e.g. `cheap:fastest`) to race one endpoint per provider. On the first request, each provider gets a concurrent request; the fastest response wins and becomes the preferred provider for subsequent requests. A re-race triggers when either `race_interval_requests` requests have passed or `race_interval_secs` seconds have elapsed since the last race.

Defaults: every **25 requests** or **6 hours**, whichever comes first.

### Group naming

Group names share a namespace with model names. If you name a group `gpt-4o`, every request with `model: gpt-4o` will be routed there instead of passing through to an upstream. The `:fastest` suffix can still be appended (`gpt-4o:fastest`) to enable racing within that group.

## Inspect

`GET /stats` — per-endpoint request/success/failure counts and per-group preferred provider order (which provider won the last race).

## Web config editor

Navigate to `/config/editor` and enter the password set in `CONFIG_EDITOR_PASSWORD`. The editor validates YAML and runs the full config parser before writing — invalid input is rejected without touching disk. On successful save, the new config is hot-reloaded in place; stats and cooloff state are reset since endpoint indices may have shifted.
