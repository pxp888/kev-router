# local router (Jev-style model routing on Kev)

Same routing idea as [thinkfarm router](https://thinkfarm.eu/#/router), but fully local.

Route requests automatically across ANY compatible inference providers using Typesafe Jev. Point your local tools to a single endpoint and let the router pick the best model for every turn.  

The routing descriptions are fully customizable, and ready for testing.


```
client -> POST /v1/chat/completions (model: "auto")
       -> Kev /v1/systemone choice question over your model descriptions
       -> probabilities x per-model weight, argmax
       -> forward to that provider, stream the answer back
```

## Run

```bash
pip install -r requirements.txt
python3 router.py            # http://0.0.0.0:8010  (PORT env to change)
```

Open `http://127.0.0.1:8010/` for the config webui (`ui.html` + `ui.css`) — the AutobotTab settings
form without the account/quota parts: Kev URL + key, decision prompt, providers
(name, api type, base URL, API key) and their models (model ID, weight, enabled,
routing description), plus the default fallback model. Save writes `router.json` and
routing picks the change up on the next request, no restart. Provider keys are write-only:
`GET /config` returns `"••••••••••••last4"`, never the key.

`ui.css` is the deployed look: the `:root` tokens from llamapage `src/index.css`
(token subset) followed by `src/components/AutobotTab.css` copied verbatim — same
class names, so the panels, accordions, switches and alerts render as they do on
thinkfarm. Light theme only (the site's dark theme is not replicated).

## Point clients at it

Point clients at it like any OpenAI-compatible endpoint — `http://127.0.0.1:8010/v1`
with any API key, or `model: "auto"`:

```bash
curl -s localhost:8010/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"write a python csv mean function with tests"}]}'
```

Log line per request shows the pick and the scores:

```
[router] picked mock / coder-32b scores={'coder-32b': 1.2579, 'deep-reasoner': 0.1003, 'small-fast': 0.061} confidence=0.758 kev_ms=309
```

## Config — `router.json`

Same schema the AutobotTab settings POST sends (`default_model`, `decision_prompt`,
`providers[].models[].{model_id,description,weight,enabled}`). Reloaded on change, so
edit and re-request without restarting.

- `kev_url` / `kev_api_key` — default `http://192.168.0.189:8008` / `hellothere` (env `KEV_URL`, `KEV_API_KEY`, `KEV_MODEL` override; `ROUTER_CONFIG` for another file).
- `description` — this is what Kev ranks. It decides routing; keep it about *task type*.
- `weight` — multiplier on Kev's probability. 1.0 neutral, >1 favour, <1 discourage.
- `default_model` — used when Kev is unreachable or returns garbage.
- `decision_prompt` — reword the question to steer routing (cost, latency, code quality).
- `api_type` — which protocol the upstream speaks: `openai-completions` (default),
  `openai-legacy-completions`, `anthropic-messages`, `google-gemini`. Requests arrive as
  OpenAI chat and get converted both ways (tools, images, thinking, SSE). Auth header per
  type: `Authorization: Bearer`, `x-api-key` + `anthropic-version`, `x-goog-api-key`.
  `api_type` is also inferred from `base_url` (`api.anthropic.com` → anthropic,
  `generativelanguage.googleapis.com` → gemini), so a Google/Anthropic provider works even
  with the field left at the default. Anthropic gets `max_tokens: 4096` when you omit it.
  ```json
  {"name": "anthropic", "base_url": "https://api.anthropic.com", "api_type": "anthropic-messages",
   "api_key": "sk-ant-...", "models": [{"model_id": "claude-3-5-sonnet-latest",
     "description": "hard reasoning, architecture, long code edits", "weight": 1.2, "enabled": true}]}
  ```
- Duplicate `model_id` across providers: second one gets option key `provider:model`.

## Checks

```bash
python3 selfcheck.py     # state building, weight flipping, fallback, URL/header resolution,
                         # config validation/key masking, anthropic + gemini adapters and SSE translation
```

```bash
curl -s localhost:8010/config | python3 -m json.tool   # what the webui edits
curl -s localhost:8010/health
```

Config POST accepts the same payload the deployed AutobotTab sends: `api_key: null` means
"keep the stored key", `clear_api_key: true` removes it. Base URLs may be `http://`
(local/LAN upstreams) instead of the deployed https-only rule.

## What is not here

- Client-side Anthropic format: the router only accepts OpenAI-shaped requests. There is no
  `/v1/messages` route for Claude-Code-style clients (the deployed router has one), and a
  legacy `/v1/completions` body (`prompt: "..."`) is forwarded as-is, so it is not adapted
  for an Anthropic/Gemini provider — use `/v1/chat/completions` with those providers.
- Accounts, API-key auth on the router, request quotas, token stats in a database, and
  the `auto.thinkfarm.eu` host rewrite. `GET /v1/models` advertises only `auto`.
- The webui has no "Router activation" toggle (no quotas locally) and no Routing
  statistics table (nothing records per-model token counts).
