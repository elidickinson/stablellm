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
| `REQUEST_LOG_DB` | *(none)* | If set, SQLite request-logging is enabled at the given path |

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

**`providers`** — a registry of upstream API endpoints. Each has a `base_url` and `api_key`. Optionally set `model` here as a provider-wide default — used when a group entry omits `model`.

**`groups`** — maps a request model name to a routing mode and an ordered list of upstream entries. Each group has:

- `mode` — `seq` (try in order, default) or `race` (send to all, first response wins). Can be overridden per-request with `:race`/`:seq` suffix on the model name.
- `endpoints` — ordered list of entries, each with:
  - `provider` — name from the providers section (required)
  - `model` — model name to send upstream. If omitted, falls back to the provider's `model` (if set), otherwise the client's requested model passes through unchanged.
  - `flags` — per-endpoint flags: `keep_reasoning` preserves `reasoning`/`reasoning_content`/`thinking` fields in messages (otherwise stripped).

**Group names match the client's `model` field case-insensitively.** `glm-4.7` matches `model: GLM-4.7`. Separators are not normalized, so `gpt-4.1` and `gpt_4_1` are distinct.

## Run

```
uv run uvicorn main:app --host $HOST --port $PORT
```

POST to `/v1/chat/completions` (or any path) like the OpenAI API. Every request must include a `model` field whose value matches a configured group name; otherwise the proxy returns 404.

## Routing

The request's `model` field selects the group. Within that group, the routing mode determines how endpoints are dispatched:

- **`seq`** (default) — try endpoints in order. A failing endpoint cools off for `cooloff_seconds` before being retried. If all endpoints fail, the request returns a 502.
- **`race`** — send the request to one endpoint per provider concurrently; the fastest response wins and becomes the preferred provider for subsequent requests. A re-race triggers when either `race_interval_requests` requests have passed or `race_interval_secs` seconds have elapsed since the last race (defaults: **25 requests** or **6 hours**). If the race fails, the proxy falls back to sequential.

The client can override a group's mode with a `:race` or `:seq` suffix on the model name (e.g. `glm-4.7:race`).

**Unknown request parameters** (not in the supported set) are silently stripped per-endpoint before forwarding. This lets providers with different capabilities share the same request body.

## Inspect

`GET /stats` — per-endpoint request/success/failure counts and per-group preferred provider order (which provider won the last race).

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
