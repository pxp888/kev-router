"""
Local Jev/Autobot-style router.

A single-file FastAPI app that does what 2llamashare/server/autobot.py does, minus
the Thinkfarm accounts/quotas: it asks a local Kev (System One) endpoint to pick the
best model for the request, then forwards the OpenAI-compatible request to that
provider and streams the answer back.

Mount points (same shape as the deployed router, without the /auto prefix):
  POST /v1/chat/completions   OpenAI chat, stream or not
  POST /v1/completions        legacy completions
  GET  /v1/models             advertises one virtual model: "auto"
  GET  /health                shows whether Kev answers
"""

import copy
import json
import logging
import os
import random
import re
import time
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

logger = logging.getLogger("local-router")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

KEV_URL = os.environ.get("KEV_URL", "http://192.168.0.189:8008").rstrip("/")
KEV_API_KEY = os.environ.get("KEV_API_KEY", "hellothere").strip()
KEV_MODEL = os.environ.get("KEV_MODEL", "jev-latest")
STATE_CHARS = int(os.environ.get("STATE_CHARS", "3000"))
KEV_TIMEOUT = float(os.environ.get("KEV_TIMEOUT_SECS", "15"))
UPSTREAM_TIMEOUT = httpx.Timeout(float(os.environ.get("UPSTREAM_TIMEOUT_SECS", "300")), connect=15.0)
CONFIG_PATH = os.environ.get("ROUTER_CONFIG", os.path.join(os.path.dirname(__file__), "router.json"))
UI_PATH = os.path.join(os.path.dirname(__file__), "ui.html")
UI_CSS_PATH = os.path.join(os.path.dirname(__file__), "ui.css")
DEFAULT_DECISION_PROMPT = "Which model is best suited for this request?"

app = FastAPI(title="local router")


# ---------- Config (mirrors the AutobotTab settings payload) ----------

_cfg_cache: tuple[float, dict] = (0.0, {})


def load_config() -> dict:
    """Read router.json, re-reading only when the file changed (edit + retry, no restart)."""
    global _cfg_cache
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        return _cfg_cache[1]
    if mtime != _cfg_cache[0]:
        with open(CONFIG_PATH) as f:
            _cfg_cache = (mtime, json.load(f))
        logger.info(f"[router] loaded {CONFIG_PATH}")
    return _cfg_cache[1]


def candidates(cfg: dict) -> list[tuple[str, str, dict, dict]]:
    """[(option_key, provider_name, model, provider), ...] for every enabled model."""
    out: list[tuple[str, str, dict, dict]] = []
    seen: set[str] = set()
    for idx, prov in enumerate(cfg.get("providers") or []):
        if not isinstance(prov, dict):
            continue
        pname = (prov.get("name") or f"provider_{idx + 1}").strip()
        base_url = (prov.get("base_url") or prov.get("baseUrl") or "").strip()
        if not base_url:
            continue
        norm_prov = {
            "name": pname,
            "base_url": base_url,
            "api_type": (prov.get("api_type") or prov.get("apiType") or "openai-completions").strip(),
            "api_key": (prov.get("api_key") or prov.get("apiKey") or "").strip(),
        }
        for m in prov.get("models") or []:
            if not isinstance(m, dict) or m.get("enabled") is False:
                continue
            mid = (m.get("model_id") or m.get("id") or "").strip()
            if not mid:
                continue
            key = f"{pname}:{mid}" if mid in seen else mid
            seen.add(mid)
            out.append((key, pname, {"id": mid, "description": (m.get("description") or "").strip(),
                                     "weight": m.get("weight", 1.0)}, norm_prov))
    return out


def model_weight(model: dict) -> float:
    try:
        w = float(model.get("weight", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return w if w == w else 1.0


# ---------- Routing state ----------

def content_text(c) -> str:
    """Flatten message content (str or list of parts), replacing images with a marker."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if (p.get("type") in ("image_url", "input_image", "image")
                        or p.get("image_url") or p.get("inlineData") or p.get("inline_data")):
                    parts.append("[Attached Image]")
                else:
                    parts.append(str(p.get("text", "")))
            elif p is not None:
                parts.append(str(p))
        return " ".join(parts)
    return str(c)


def build_state(body: dict) -> str:
    """Last ~3k chars of conversation, system messages dropped, images as markers."""
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        parts = []
        for m in messages:
            if not isinstance(m, dict) or m.get("role") == "system":
                continue
            text = content_text(m.get("content"))
            if m.get("images"):
                text = f"{text}\n[Attached Image]".strip()
            parts.append(f"{m.get('role', 'user')}:\n{text}")
        return ("\n\n".join(parts))[-STATE_CHARS:]
    if body.get("prompt") is not None:
        return content_text(body["prompt"])[-STATE_CHARS:]
    return ""


# ---------- Ask Kev ----------

def kev_base(cfg: dict) -> str:
    """Kev base URL: config override if present, else KEV_URL env."""
    u = (cfg.get("kev_url") or cfg.get("kevUrl") or KEV_URL).rstrip("/")
    return u[: -len("/v1/systemone")] if u.endswith("/v1/systemone") else u


async def kev_question(client: httpx.AsyncClient, cfg: dict, state: str, criteria: dict, instructions: str) -> dict:
    key = (cfg.get("kev_api_key") or cfg.get("kevApiKey") or KEV_API_KEY).strip()
    r = await client.post(
        f"{kev_base(cfg)}/v1/systemone",
        headers={"authorization": f"Bearer {key}"} if key else {},
        json={
            "model": KEV_MODEL,
            "state": state,
            "questions": {"model": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        },
    )
    r.raise_for_status()
    return r.json()["answers"]["model"]


async def decide(cfg: dict, state: str) -> tuple[tuple, bool]:
    """Pick (candidate, was_fallback): Kev probabilities x per-model weight, argmax wins."""
    cands = candidates(cfg)
    if not cands:
        raise ValueError("No enabled models configured in router.json")

    by_key = {c[0]: c for c in cands}
    default_id = (cfg.get("default_model") or cfg.get("defaultModel") or "").strip()
    fallback_key = next((k for k, _, m, _ in cands if m["id"] == default_id), cands[0][0])

    items = [(k, m["description"] or "(no description)") for k, _, m, _ in cands]
    if len(items) > 1:
        random.shuffle(items)  # Kev is order-sensitive; shuffle so option position does not bias it

    prompt = (cfg.get("decision_prompt") or cfg.get("decisionPrompt") or "").strip() or DEFAULT_DECISION_PROMPT
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=KEV_TIMEOUT) as client:
            answer = await kev_question(client, cfg, state, dict(items), prompt)
        probs = answer.get("probabilities") or {}
        weighted = {k: round(float(probs[k]) * model_weight(by_key[k][2]), 4) for k in probs if k in by_key}
        if not weighted:
            logger.warning(f"[router] malformed probabilities {probs}; fallback")
            return by_key[fallback_key], True
        cand = by_key[max(weighted, key=weighted.get)]
        logger.info(
            f"[router] picked {cand[1]} / {cand[2]['id']} scores={weighted} "
            f"confidence={answer.get('confidence')} kev_ms={int((time.perf_counter() - started) * 1000)}"
        )
        return cand, False
    except Exception as e:
        cand = by_key[fallback_key]
        logger.warning(f"[router] Kev failed ({e.__class__.__name__}: {e}); fallback to {cand[1]} / {cand[2]['id']}")
        return cand, True


# ---------- Forwarding ----------

def resolve_upstream_url(prov: dict, endpoint_suffix: str | None, model_id: str) -> str:
    """Build the upstream URL from base_url, whether or not it already ends in /v1."""
    target = "completions" if prov["api_type"] == "openai-legacy-completions" or endpoint_suffix == "completions" \
        else (endpoint_suffix or "chat/completions").lstrip("/")
    if target.startswith("v1/"):
        target = target[3:]

    base = prov["base_url"].rstrip("/")
    path = urlsplit(base).path.rstrip("/")
    if not path:
        return f"{base}/v1/{target}"
    if base.endswith(f"/{target}"):
        return base
    return f"{base}/{target}"


def upstream_headers(prov: dict) -> dict:
    h = {"content-type": "application/json"}
    if prov.get("api_key"):
        h["authorization"] = f"Bearer {prov['api_key']}"
    return h


async def forward(body: dict, cand: tuple, endpoint_suffix: str | None):
    _, pname, model, prov = cand
    body = {**body, "model": model["id"]}
    url = resolve_upstream_url(prov, endpoint_suffix, model["id"])
    headers = upstream_headers(prov)

    if not body.get("stream"):
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                r = await client.post(url, json=body, headers=headers)
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {
                "message": f"upstream '{pname}' request failed: {e}", "type": "upstream_error", "code": 502}})
        try:
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except ValueError:
            return PlainTextResponse(r.text, status_code=r.status_code,
                                     media_type=r.headers.get("content-type", "text/plain"))

    async def stream():
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                async with client.stream("POST", url, json=body, headers=headers) as r:
                    if r.status_code >= 300:
                        err = {"error": {"message": (await r.aread()).decode(errors="replace"),
                                         "type": "upstream_error", "code": r.status_code}}
                        yield f"data: {json.dumps(err)}\n\n".encode()
                        return
                    async for chunk in r.aiter_bytes():
                        yield chunk
        except Exception as e:
            err = {"error": {"message": f"stream error from '{pname}': {e}", "type": "upstream_error"}}
            yield f"data: {json.dumps(err)}\n\n".encode()

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------- Config webui (GET/POST /config + static ui.html) ----------

def mask_key(key: str) -> str:
    return ("\u2022" * 12 + key[-4:]) if len(key) >= 4 else "\u2022" * 12


def mask_secrets(cfg: dict) -> dict:
    """Config for the browser: real provider/Kev keys never leave the server."""
    out = copy.deepcopy(cfg)
    for p in out.get("providers") or []:
        key = (p.get("api_key") or "").strip()
        p["api_key"] = ""
        p["api_key_masked"] = mask_key(key) if key else ""
    kev_key = (out.get("kev_api_key") or "").strip()
    out["kev_api_key"] = ""
    out["kev_api_key_masked"] = mask_key(kev_key) if kev_key else ""
    return out


def validate_base_url(raw: str, name: str) -> str | None:
    """Like the deployed validator, but http:// is allowed for LAN/localhost upstreams."""
    url = (raw or "").strip()
    if not url:
        return f'Base URL for "{name}" is required.'
    if len(url) > 512:
        return f'Base URL for "{name}" is too long (max 512 characters).'
    if re.search(r"[\s\\]", url):
        return f'Base URL for "{name}" contains invalid characters.'
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        return f'Base URL for "{name}" must start with http:// or https://'
    if parsed.username or parsed.password:
        return f'Base URL for "{name}" must not include embedded credentials.'
    if parsed.query or parsed.fragment:
        return f'Base URL for "{name}" must not include query strings (?) or fragments (#).'
    if not parsed.hostname:
        return f'Base URL for "{name}" must include a valid hostname.'
    return None


def validate_config(cfg: dict) -> str | None:
    """Mirror of AutobotTab validateConfiguration(); None means valid."""
    providers = cfg.get("providers") or []
    if not providers:
        return "Please configure at least one provider."
    for p in providers:
        name = (p.get("name") or "").strip()
        if not name:
            return "Every provider must have a name."
        err = validate_base_url(p.get("base_url"), name)
        if err:
            return err
        for m in p.get("models") or []:
            mid = (m.get("model_id") or m.get("id") or "").strip()
            if not mid:
                return f'Every model in "{name}" must have a Model ID.'
            if not (m.get("description") or "").strip():
                return f'Model "{mid}" needs a description so Kev can route requests to it.'
    if not (cfg.get("default_model") or "").strip():
        return "Please select a default fallback model in case Kev is unreachable."
    return None


def merge_config(inc: dict, old: dict) -> dict:
    """
    Normalize the webui payload into stored config. api_key: null keeps the stored key
    (the browser never sees it), "" with clear_api_key removes it, text replaces it.
    """
    by_id = {str(p["id"]): (p.get("api_key") or "").strip() for p in old.get("providers") or [] if p.get("id") is not None}
    # stored configs may predate the webui and have no per-provider id: index is the fallback
    by_index = {str(i): (p.get("api_key") or "").strip() for i, p in enumerate(old.get("providers") or [])}
    providers = []
    for i, p in enumerate(inc.get("providers") or []):
        if not isinstance(p, dict):
            continue
        name = (p.get("name") or "").strip()
        if p.get("clear_api_key"):
            key = ""
        elif p.get("api_key"):
            key = str(p["api_key"]).strip()
        else:
            key = by_id.get(str(p["id"]), by_index.get(str(i), "")) if p.get("id") is not None else by_index[str(i)]
        models = []
        for m in p.get("models") or []:
            if not isinstance(m, dict):
                continue
            try:
                weight = float(m.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            models.append({
                "id": m.get("id"),
                "model_id": (m.get("model_id") or m.get("id") or "").strip(),
                "description": (m.get("description") or "").strip(),
                "weight": weight,
                "enabled": bool(m.get("enabled", True)),
            })
        providers.append({
            "id": p.get("id"),
            "name": name,
            "base_url": (p.get("base_url") or "").strip(),
            "api_type": (p.get("api_type") or "openai-completions").strip(),
            "api_key": key,
            "models": models,
        })

    out = copy.deepcopy(old)
    out.pop("stats", None)
    out["providers"] = providers
    out["default_model"] = (inc.get("default_model") or "").strip()
    out["decision_prompt"] = (inc.get("decision_prompt") or "").strip() or DEFAULT_DECISION_PROMPT
    out["kev_url"] = (inc.get("kev_url") or "").strip() or old.get("kev_url") or KEV_URL
    out["kev_api_key"] = (inc.get("kev_api_key") or "").strip() or (
        "" if inc.get("clear_kev_api_key") else (old.get("kev_api_key") or KEV_API_KEY))
    return out


@app.get("/")
async def webui():
    return FileResponse(UI_PATH, media_type="text/html")


@app.get("/ui.css")
async def webui_css():
    return FileResponse(UI_CSS_PATH, media_type="text/css")


@app.get("/config")
async def get_config():
    return mask_secrets(load_config())


@app.post("/config")
async def put_config(request: Request):
    try:
        inc = await request.json()
        if not isinstance(inc, dict):
            raise ValueError("body must be a JSON object")
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"invalid JSON: {e}"})

    cfg = merge_config(inc, load_config())
    if err := validate_config(cfg):
        return JSONResponse(status_code=400, content={"detail": err})

    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    os.replace(tmp, CONFIG_PATH)
    global _cfg_cache
    _cfg_cache = (0.0, cfg)  # force the next request to re-read
    logger.info(f"[router] saved {CONFIG_PATH}: {len(cfg['providers'])} providers, "
                f"{sum(len(p['models']) for p in cfg['providers'])} models")
    return {"ok": True, "config": mask_secrets(cfg)}


# ---------- Routes ----------

async def handle(request: Request, endpoint_suffix: str | None):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": {
            "message": f"invalid JSON request body: {e}", "type": "invalid_request_error", "code": 400}})

    try:
        cand, _ = await decide(load_config(), build_state(body))
    except ValueError as e:
        return JSONResponse(status_code=503, content={"error": {"message": str(e), "type": "configuration_error", "code": 503}})

    return await forward(body, cand, endpoint_suffix)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    return await handle(request, "chat/completions")


@app.post("/v1/completions")
@app.post("/completions")
async def completions(request: Request):
    return await handle(request, "completions")


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": "auto", "name": "auto", "object": "model", "created": 0, "owned_by": "local-router",
        "input_modalities": ["text", "image"], "output_modalities": ["text"],
        "capabilities": {"vision": True, "tools": True, "function_calling": True},
    }]}


@app.get("/health")
async def health():
    cfg = load_config()
    kev_up = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{kev_base(cfg)}/v1/models",
                                 headers={"authorization": f"Bearer {cfg.get('kev_api_key') or KEV_API_KEY}"})
            kev_up = r.status_code == 200
    except Exception as e:
        kev_up = str(e)
    return {"ok": True, "service": "local-router", "kev_url": KEV_URL, "kev_up": kev_up,
            "models": [f"{c[1]} / {c[2]['id']}" for c in candidates(cfg)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8010")))
