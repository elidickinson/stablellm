# stablellm

OpenAI-compatible proxy that fans requests across multiple upstream endpoints with failover and optional latency racing.

## Configure

Copy `.env.example` to `.env` and set:

- `ENDPOINT_<NAME>=<base_url>|<api_key>|<model>|<flags>` — one per upstream. `model` and `flags` are optional. If `model` is empty, requests via a named group send the group name as the model; requests via the default group pass the client's model through. Flags: `keep_reasoning`.
- `GROUP_<NAME>=<ep1>,<ep2>,...` — optional. Maps a virtual model name to an ordered subset of endpoints.
- `GROUP_DEFAULT` — optional. Endpoints (and order) used when the request model doesn't match any group. Defaults to all endpoints in declaration order.

Other knobs: `HOST`, `PORT`, `API_KEY` (proxy auth), `COOLOFF_SECONDS`, `REQUEST_TIMEOUT`, `CONNECT_TIMEOUT`, `RACE_INTERVAL_SECS`, `RACE_INTERVAL_REQUESTS`, `LOG_LEVEL`.

## Run

```
uv run uvicorn main:app --host $HOST --port $PORT
```

POST to `/v1/chat/completions` (or any path) exactly like the OpenAI API.

## Routing

Every request resolves to a group:

1. If `model` (lowercased) matches a `GROUP_<NAME>`, that group is used.
2. Otherwise the `default` group is used.

Within a group, endpoints are tried in order. A failing endpoint cools off for `COOLOFF_SECONDS` before it's retried.

### Fastest mode

Append `:fastest` to the model (e.g. `model: "cheap:fastest"`) to race one endpoint per provider group on the first request, then reuse the winner for `RACE_INTERVAL_REQUESTS` requests or `RACE_INTERVAL_SECS` seconds.

### Group naming

Group names share a namespace with model names. A request's `model` either matches a group or passes through to `default`. **Do not name a group after a real model ID** — you'll permanently reroute that model with no opt-out. Use virtual names like `cheap`, `fast`, `smart`.

## Inspect

`GET /stats` — per-endpoint request/success/failure counts and per-group preferred provider order.
