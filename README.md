# stablellm

OpenAI-compatible proxy that fans requests across multiple upstream endpoints with failover and optional latency racing.

## Configure

Two files. Bind-time settings live in `.env` (changing them requires a restart). Everything else lives in `config.yaml` and is reloadable.

**`.env`** — copy from `.env.example`:

- `HOST`, `PORT`, `REQUEST_TIMEOUT`, `CONNECT_TIMEOUT` — server bind + outbound HTTP client.
- `API_KEY` — optional. If set, clients must send `Authorization: Bearer <API_KEY>`.
- `CONFIG_FILE` — path to the YAML config (default `config.yaml`).
- API key vars referenced from YAML via `${VAR}` interpolation (e.g. `OPENAI_API_KEY`).

**`config.yaml`** — copy from `config.example.yaml`:

```yaml
settings:
  cooloff_seconds: 30
  race_interval_secs: 21600
  race_interval_requests: 25
  log_level: INFO

endpoints:
  openai:
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
  cerebras:
    base_url: https://api.cerebras.ai/v1
    api_key: ${CEREBRAS_API_KEY}
    model: llama-3.3-70b
    flags: [keep_reasoning]

groups:
  default: [openai]
  cheap: [cerebras]
```

Per-endpoint `model` is optional. If empty, requests via a named group send the client's requested model; requests via the `default` group also pass through.

## Run

```
uv run uvicorn main:app --host $HOST --port $PORT
```

POST to `/v1/chat/completions` (or any path) like the OpenAI API.

## Routing

Every request resolves to a group:

1. If the request `model` matches a group, that group is used.
2. Otherwise the `default` group is used (all endpoints in declaration order, unless overridden).

Within a group, endpoints are tried in order. A failing endpoint cools off for `cooloff_seconds` before being retried.

Group lookup is case-insensitive and treats `-`, `.`, `_` as equivalent — so YAML key `gpt_4_1` matches request model `gpt-4.1`.

### Fastest mode

Append `:fastest` to the model (e.g. `cheap:fastest`) to race one endpoint per provider on the first request, then reuse the winner for `race_interval_requests` requests or `race_interval_secs` seconds.

### Group naming

Group names share a namespace with model names. Don't name a group after a real model ID unless you want every request for that model permanently routed there — there's no opt-out.

## Inspect

`GET /stats` — per-endpoint request/success/failure counts and per-group preferred provider order.

## Web config editor

Set `CONFIG_EDITOR_PASSWORD` in `.env` to enable a YAML editor at `/config/editor`. If unset, all `/config/*` routes 404.

The editor validates YAML and runs the full config parser before writing — invalid input is rejected without touching disk. On successful save, the new config is hot-reloaded in place. Stats and cooloff state are reset (endpoint indices may have shifted). Bind-time settings (`HOST`, `PORT`, timeouts) are not editable here — they live in `.env` and require a restart.
