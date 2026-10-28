# local router (Jev-style model routing on Kev)

Same routing idea as `2llamashare/server/autobot.py`, without Thinkfarm accounts,
quotas, or the Postgres stats: one FastAPI app that asks Kev which of your models fits
the request, then forwards the request to that provider.

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
- Duplicate `model_id` across providers: second one gets option key `provider:model`.

## Checks

```bash
python3 selfcheck.py     # state building, weight flipping, fallback, URL resolution, config validation/key masking
```

```bash
curl -s localhost:8010/config | python3 -m json.tool   # what the webui edits
curl -s localhost:8010/health
```

Config POST accepts the same payload the deployed AutobotTab sends: `api_key: null` means
"keep the stored key", `clear_api_key: true` removes it. Base URLs may be `http://`
(local/LAN upstreams) instead of the deployed https-only rule.

## What is not here

- Upstream adapters for Anthropic (`/v1/messages`) and Google Gemini. Only
  `openai-completions` / `openai-legacy-completions` upstreams work (Ollama, llama.cpp,
  vLLM, LM Studio, OpenAI). If you need them, the self-contained adapters in
  `autobot.py` (`openai_to_anthropic_request`, `anthropic_to_openai_response`,
  `_AnthropicStreamTranslator`, and the Gemini equivalents) copy over as-is.
- Accounts, API-key auth on the router, request quotas, token stats in a database, and
  the `auto.thinkfarm.eu` host rewrite. `GET /v1/models` advertises only `auto`.
- The webui has no "Router activation" toggle (no quotas locally) and no Routing
  statistics table (nothing records per-model token counts).
