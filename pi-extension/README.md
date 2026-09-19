# pi-stablellm

Pi provider extension for a StableLLM server. It discovers groups from `/v1/models`, maps their published capabilities into Pi model definitions, adds `:race` variants for sequential groups, follows the server's race redirect handshake transparently, and shows the upstream provider/model that served the last response.

## Configure

Set the StableLLM server URL before starting Pi. The extension normalizes it to the OpenAI-compatible `/v1` endpoint and deliberately has no built-in server address.

```bash
export STABLELLM_BASE_URL=https://stablellm.example.com/v1
```

Authenticate either through Pi or the environment:

```text
/login stablellm
```

```bash
export STABLELLM_API_KEY=...
```

Pi stores credentials entered through `/login` in its normal `auth.json`. The stored server URL and key each take precedence over `STABLELLM_BASE_URL` / `STABLELLM_API_KEY`; run `/login stablellm` again to point at a different server. The key is optional — a server with no API keys configured ignores it.

Scripted setup can skip the login prompts and write the same credential into `~/.pi/agent/auth.json` directly, with the URL in the credential's `env` object. Pi rewrites that file as a whole, so edit it while Pi is not running. Omit `key` for a server with no API keys.

```json
{
  "stablellm": {
    "type": "api_key",
    "key": "...",
    "env": { "STABLELLM_BASE_URL": "https://stablellm.example.com/v1" }
  }
}
```

## Install

Add the package directory to Pi's `settings.json`:

```json
{
  "packages": ["/path/to/stablellm/pi-extension"]
}
```

Restart Pi, or use `/reload` after changing the extension. Remove any hand-written `stablellm` provider entry from `~/.pi/agent/models.json`; Pi applies `models.json` overrides above extension providers, so an old static model list would override discovery.

## Model variants

A normal sequential group is exposed twice: `group` and `group:race`. A group whose server default is already `race` is exposed only as `group`; the extension does not add a `group:seq` alias. If Pi's `enabledModels` setting is in use, include the desired `:race` IDs or a `stablellm/*` pattern there as well.

When optional model metadata is missing, the extension carries conservative built-in metadata for known StableLLM model groups. Unknown groups use Pi's standard custom-model defaults: text input, a 128K context window, 16K maximum output, no reasoning, and zero cost.

## Race handshake

StableLLM confirms an eligible race by replying to the POST with `307 Temporary Redirect` to the same path plus a `stablellm_race_redirect=1` marker and `X-StableLLM-Race: pending`. The OpenAI SDK would treat that as an API error, so the extension wraps `stream` and `streamSimple` with a custom fetch: it makes the request with `redirect: "manual"`, validates that the 307 is same-origin and carries both the marker and race header, replays the original POST once to the marked Location with its method, body, headers, and abort signal intact, and returns only the final response to the SDK. A repeated pending redirect is rejected rather than followed again, and auth headers never leave the origin.

While a race is pending the extension shows `StableLLM is racing providers...` as Pi's working message, and clears it as soon as provider headers arrive or the turn ends, errors, aborts, switches models, or the session shuts down. A racing turn can therefore sit on that message a little longer than usual: the server buffers every candidate through its full response before delivering the winner.
