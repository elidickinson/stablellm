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
| `API_KEY` | *(none)* | If set, clients must send `Authorization: Bearer <API_KEY>` |
| `CONFIG_FILE` | `config.yaml` | Path to the YAML config |
| `MAX_BODY_BYTES` | `52428800` (50MB) | Max inbound request body size |
| `CONFIG_EDITOR_PASSWORD` | *(none)* | If set, enables the web config editor at `/config/editor` |

API keys for upstream providers are set as individual vars here and referenced from YAML via `${VAR}` interpolation (e.g. `OPENAI_API_KEY`).

### `config.yaml` — reloadable settings

```yaml
settings:
  cooloff_seconds: 30          # how long a failing endpoint is skipped
  race_interval_secs: 21600    # 6h — time between races (per group)
  race_interval_requests: 25   # request count between races (per group)
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
    - provider: openai
    - provider: cerebras
      model: zai-glm-4.7
      flags: [keep_reasoning]

  glm-4.7:
    - provider: cerebras
      model: zai-glm-4.7
      flags: [keep_reasoning]
    - provider: openai
```

**`providers`** — a registry of upstream API endpoints. Each has a `base_url` and `api_key`. Optionally set `model` here as a provider-wide default — used when a group entry omits `model`.

**`groups`** — maps a request model name to an ordered list of upstream entries. Each entry references a provider and can supply:

- `provider` — name from the providers section (required)
- `model` — model name to send upstream. If omitted, falls back to the provider's `model` (if set), otherwise the client's requested model passes through unchanged.
- `flags` — per-endpoint flags: `keep_reasoning` preserves `reasoning`/`reasoning_content`/`thinking` fields in messages (otherwise stripped).

**Group names are plain strings** — `glm-4.7` matches exactly `model: glm-4.7` from the client. No normalization or case folding.

**`default` group** is the fallback when no other group matches the request model. It is optional — if omitted, requests with an unknown model will receive a 400 error.

## Run

```
uv run uvicorn main:app --host $HOST --port $PORT
```

POST to `/v1/chat/completions` (or any path) like the OpenAI API.

## Routing

Every request resolves to a group:

1. If the request `model` matches a group name, that group's provider list is used.
2. Otherwise the `default` group is used.

Within a group, providers are tried in order. A failing endpoint cools off for `cooloff_seconds` before being retried. If all endpoints fail, the request returns a 502.

**Unknown request parameters** (not in the supported set) are silently stripped per-endpoint before forwarding. This lets providers with different capabilities share the same request body.

### Fastest mode

Append `:fastest` to the model (e.g. `glm-4.7:fastest`) to race one endpoint per provider. On the first request, each provider gets a concurrent request; the fastest response wins and becomes the preferred provider for subsequent requests. A re-race triggers when either `race_interval_requests` requests have passed or `race_interval_secs` seconds have elapsed since the last race.

Defaults: every **25 requests** or **6 hours**, whichever comes first.

## Inspect

`GET /stats` — per-endpoint request/success/failure counts and per-group preferred provider order (which provider won the last race).

## Web config editor

Navigate to `/config/editor` and enter the password set in `CONFIG_EDITOR_PASSWORD`. The editor validates YAML and runs the full config parser before writing — invalid input is rejected without touching disk. On successful save, the new config is hot-reloaded in place; stats and cooloff state are reset since endpoint indices may have shifted.
