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
import uuid
from typing import Any
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

# ---------- Protocol adapters (verbatim from 2llamashare/server/autobot.py) ----------
def flatten_content(c: Any) -> str:
    """Message content may be a string or a list of typed parts."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            "" if p is None else (p.get("text", "") if isinstance(p, dict) else str(p))
            for p in c
        )
    return str(c)


def _parse_data_or_http_url(url_str: str, default_media_type: str = "image/png") -> dict[str, str] | None:
    """Parse a `data:...;base64,...` URI or `http(s)://` URL into a normalized image dict."""
    if not isinstance(url_str, str):
        return None
    s = url_str.strip()
    if not s:
        return None
    if s.startswith("data:") and ";base64," in s:
        header, b64_data = s.split(";base64,", 1)
        media_type = header[5:].split(";", 1)[0].strip() or default_media_type
        b64_clean = "".join(b64_data.split())
        if not b64_clean:
            return None
        return {
            "kind": "base64",
            "media_type": media_type,
            "data": b64_clean,
            "url": f"data:{media_type};base64,{b64_clean}",
        }
    if s.startswith(("http://", "https://")):
        return {
            "kind": "url",
            "media_type": default_media_type,
            "url": s,
        }
    return None


def _parse_image_part(part: Any) -> dict[str, str] | None:
    """
    Extract normalized image metadata from an OpenAI (`image_url` / `input_image`),
    Anthropic (`image`), or Gemini (`inlineData` / `inline_data`) content part.
    Returns `{'kind': 'base64'|'url', 'media_type': str, 'data': str, 'url': str}` or None.
    """
    if not isinstance(part, dict):
        return None
    ptype = part.get("type")

    if ptype in ("image_url", "input_image"):
        url_obj = part.get("image_url") if "image_url" in part else part.get("url")
        if isinstance(url_obj, dict):
            url_str = url_obj.get("url") or ""
        elif isinstance(url_obj, str):
            url_str = url_obj
        else:
            url_str = ""
        parsed = _parse_data_or_http_url(url_str)
        if parsed:
            return parsed
        b64 = part.get("b64_json") or part.get("data")
        if isinstance(b64, str) and b64.strip():
            media_type = (part.get("media_type") or part.get("mime_type") or "image/png").strip()
            b64_clean = "".join(b64.split())
            return {
                "kind": "base64",
                "media_type": media_type,
                "data": b64_clean,
                "url": f"data:{media_type};base64,{b64_clean}",
            }
        return None

    if ptype == "image":
        source = part.get("source")
        if isinstance(source, dict):
            stype = source.get("type")
            if stype == "base64":
                b64 = source.get("data") or ""
                if isinstance(b64, str) and b64.strip():
                    media_type = (source.get("media_type") or "image/png").strip()
                    b64_clean = "".join(b64.split())
                    return {
                        "kind": "base64",
                        "media_type": media_type,
                        "data": b64_clean,
                        "url": f"data:{media_type};base64,{b64_clean}",
                    }
            elif stype == "url":
                url_str = source.get("url") or ""
                return _parse_data_or_http_url(url_str)
        url_field = part.get("url") or part.get("image_url")
        if isinstance(url_field, str):
            return _parse_data_or_http_url(url_field)
        return None

    inline = part.get("inlineData") or part.get("inline_data")
    if isinstance(inline, dict):
        b64 = inline.get("data") or ""
        if isinstance(b64, str) and b64.strip():
            media_type = (inline.get("mimeType") or inline.get("mime_type") or "image/png").strip()
            b64_clean = "".join(b64.split())
            return {
                "kind": "base64",
                "media_type": media_type,
                "data": b64_clean,
                "url": f"data:{media_type};base64,{b64_clean}",
            }

    return None


def _extract_message_extra_images(msg: dict) -> list[dict[str, str]]:
    """Extract normalized images from a message's top-level `images` list (Ollama / OpenAI extension)."""
    raw_images = msg.get("images")
    if not isinstance(raw_images, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw_images:
        if isinstance(item, str):
            s = item.strip()
            if not s:
                continue
            parsed = _parse_data_or_http_url(s)
            if parsed:
                out.append(parsed)
            else:
                b64_clean = "".join(s.split())
                out.append(
                    {
                        "kind": "base64",
                        "media_type": "image/png",
                        "data": b64_clean,
                        "url": f"data:image/png;base64,{b64_clean}",
                    }
                )
        elif isinstance(item, dict):
            parsed = _parse_image_part(item)
            if parsed:
                out.append(parsed)
    return out


_gemini_thought_signatures: dict[str, str] = {}


def _store_gemini_thought_signature(tc_id: str, fn_name: str, args_obj: dict, sig: str) -> None:
    if not sig:
        return
    if len(_gemini_thought_signatures) > 1000:
        # Evict oldest half when cache exceeds 1000 entries
        for k in list(_gemini_thought_signatures.keys())[:500]:
            _gemini_thought_signatures.pop(k, None)
    if tc_id:
        _gemini_thought_signatures[tc_id] = sig
    try:
        canon_key = f"{fn_name}:{json.dumps(args_obj, sort_keys=True)}"
        _gemini_thought_signatures[canon_key] = sig
    except Exception:
        pass


def _lookup_gemini_thought_signature(tc_id: str, fn_name: str, args_obj: dict) -> str | None:
    if tc_id and tc_id in _gemini_thought_signatures:
        return _gemini_thought_signatures[tc_id]
    try:
        canon_key = f"{fn_name}:{json.dumps(args_obj, sort_keys=True)}"
        return _gemini_thought_signatures.get(canon_key)
    except Exception:
        return None


def _clean_gemini_schema(schema: Any) -> dict[str, Any]:
    """
    Sanitize a JSON Schema object for Google Gemini's strict OpenAPI / Protobuf `Schema` validator.
    Strips unsupported keys like `additionalProperties`, `$schema`, `strict`, `default`, etc.,
    and converts union types like `["string", "null"]` into `type: "string", nullable: True`.
    """
    if not isinstance(schema, dict):
        return {"type": "string"}

    out: dict[str, Any] = {}
    nullable = bool(schema.get("nullable"))

    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        non_null = [t for t in raw_type if t != "null"]
        if "null" in raw_type:
            nullable = True
        out["type"] = str(non_null[0]) if non_null else "string"
    elif isinstance(raw_type, str):
        if raw_type == "null":
            out["type"] = "string"
            nullable = True
        else:
            out["type"] = raw_type

    if "const" in schema and "enum" not in schema:
        out["enum"] = [str(schema["const"])]
        if "type" not in out:
            out["type"] = "string"

    for k in ("format", "description", "maxItems", "minItems"):
        if k in schema and schema[k] is not None:
            out[k] = schema[k]

    if "enum" in schema and isinstance(schema["enum"], list):
        out["enum"] = [str(v) for v in schema["enum"] if v is not None]

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        non_null_branches = [
            b for b in schema["anyOf"] if not (isinstance(b, dict) and b.get("type") == "null")
        ]
        if len(non_null_branches) < len(schema["anyOf"]):
            nullable = True
        if len(non_null_branches) == 1:
            merged = _clean_gemini_schema(non_null_branches[0])
            for k, v in out.items():
                if k != "type":
                    merged[k] = v
            if nullable:
                merged["nullable"] = True
            return merged
        elif non_null_branches:
            out["anyOf"] = [_clean_gemini_schema(b) for b in non_null_branches]

    if "properties" in schema and isinstance(schema["properties"], dict):
        props = {}
        for prop_name, prop_schema in schema["properties"].items():
            props[prop_name] = _clean_gemini_schema(prop_schema)
        out["properties"] = props
        if "type" not in out:
            out["type"] = "object"
        if "required" in schema and isinstance(schema["required"], list):
            valid_req = [r for r in schema["required"] if r in props]
            if valid_req:
                out["required"] = valid_req

    if "items" in schema and isinstance(schema["items"], dict):
        out["items"] = _clean_gemini_schema(schema["items"])
        if "type" not in out:
            out["type"] = "array"

    if nullable:
        out["nullable"] = True

    if "type" not in out and "anyOf" not in out:
        out["type"] = "object" if "properties" in out else "string"

    return out


# OpenAI-shaped request body <-> Anthropic /v1/messages and Google generateContent.
# Handles tools, images, system messages and SSE streaming in both directions.

def _parse_json_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        s = raw_args.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except Exception:
            return {"raw": raw_args}
    return {}


def openai_to_anthropic_request(body: dict, model_id: str) -> dict[str, Any]:
    """Translate an OpenAI `/v1/chat/completions` request payload into Anthropic `/v1/messages` format."""
    system_parts: list[str] = []
    if body.get("system"):
        sys_txt = flatten_content(body["system"]).strip()
        if sys_txt:
            system_parts.append(sys_txt)

    anth_messages: list[dict[str, Any]] = []
    raw_messages = body.get("messages") or []

    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")

        if role in ("system", "developer"):
            sys_txt = flatten_content(content).strip()
            if sys_txt:
                system_parts.append(sys_txt)
            continue

        if role == "tool":
            tool_use_id = str(m.get("tool_call_id") or m.get("id") or "call_0")
            res_text = flatten_content(content)
            block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": res_text,
            }
            if anth_messages and anth_messages[-1]["role"] == "user":
                # Anthropic requires tool_result blocks to precede text blocks in user messages
                existing = anth_messages[-1]["content"]
                tool_results = [b for b in existing if isinstance(b, dict) and b.get("type") == "tool_result"]
                others = [b for b in existing if not (isinstance(b, dict) and b.get("type") == "tool_result")]
                anth_messages[-1]["content"] = tool_results + [block] + others
            else:
                anth_messages.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str) and part:
                        blocks.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "text" and part.get("text"):
                            blocks.append({"type": "text", "text": part["text"]})
                        elif ptype == "tool_use":
                            blocks.append(part)

            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                for idx, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(tc.get("id") or f"toolu_{idx}_{uuid.uuid4().hex[:8]}"),
                            "name": str(fn.get("name") or "tool"),
                            "input": _parse_json_args(fn.get("arguments")),
                        }
                    )

            if blocks:
                if anth_messages and anth_messages[-1]["role"] == "assistant":
                    anth_messages[-1]["content"].extend(blocks)
                else:
                    anth_messages.append({"role": "assistant", "content": blocks})
            continue

        # Default: user role
        user_blocks: list[dict[str, Any]] = []
        if isinstance(content, str):
            if content:
                user_blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str) and part:
                    user_blocks.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    ptype = part.get("type")
                    img = _parse_image_part(part)
                    if img is not None:
                        if img["kind"] == "base64":
                            user_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": img["media_type"],
                                        "data": img["data"],
                                    },
                                }
                            )
                        elif img["kind"] == "url":
                            user_blocks.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": img["url"]},
                                }
                            )
                    elif ptype == "text" and part.get("text"):
                        user_blocks.append({"type": "text", "text": part["text"]})
                    elif ptype in ("image", "tool_result"):
                        user_blocks.append(part)

        for img in _extract_message_extra_images(m):
            if img["kind"] == "base64":
                user_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img["media_type"],
                            "data": img["data"],
                        },
                    }
                )
            elif img["kind"] == "url":
                user_blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": img["url"]},
                    }
                )

        if user_blocks:
            if anth_messages and anth_messages[-1]["role"] == "user":
                anth_messages[-1]["content"].extend(user_blocks)
            else:
                anth_messages.append({"role": "user", "content": user_blocks})

    if not anth_messages:
        prompt_txt = flatten_content(body.get("prompt") or "") or "."
        anth_messages.append({"role": "user", "content": [{"type": "text", "text": prompt_txt}]})
    elif anth_messages[0]["role"] != "user":
        anth_messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "."}]})

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or 4096
    try:
        max_tokens = max(1, int(max_tokens))
    except Exception:
        max_tokens = 4096

    anth_body: dict[str, Any] = {
        "model": model_id,
        "messages": anth_messages,
        "max_tokens": max_tokens,
        "stream": bool(body.get("stream")),
    }

    if system_parts:
        anth_body["system"] = "\n\n".join(system_parts)

    if body.get("temperature") is not None:
        try:
            anth_body["temperature"] = float(body["temperature"])
        except Exception:
            pass

    if body.get("top_p") is not None and "temperature" not in anth_body:
        try:
            anth_body["top_p"] = float(body["top_p"])
        except Exception:
            pass

    stop = body.get("stop") or body.get("stop_sequences")
    if stop:
        if isinstance(stop, str):
            anth_body["stop_sequences"] = [stop]
        elif isinstance(stop, list):
            anth_body["stop_sequences"] = [str(s) for s in stop if s]

    raw_tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    if isinstance(raw_tools, list) and raw_tools and tool_choice != "none":
        anth_tools: list[dict[str, Any]] = []
        for t in raw_tools:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                name = fn.get("name")
                if not name:
                    continue
                schema = fn.get("parameters") or {"type": "object", "properties": {}}
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                if "type" not in schema:
                    schema = {"type": "object", **schema}
                anth_tools.append(
                    {
                        "name": name,
                        "description": fn.get("description") or "",
                        "input_schema": schema,
                    }
                )
            elif t.get("name"):
                anth_tools.append(
                    {
                        "name": t["name"],
                        "description": t.get("description") or "",
                        "input_schema": t.get("input_schema")
                        or t.get("parameters")
                        or {"type": "object", "properties": {}},
                    }
                )
        if anth_tools:
            anth_body["tools"] = anth_tools
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    anth_body["tool_choice"] = {"type": "auto"}
                elif tool_choice in ("required", "any"):
                    anth_body["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
                    fn_name = tool_choice["function"].get("name")
                    if fn_name:
                        anth_body["tool_choice"] = {"type": "tool", "name": fn_name}
                elif tool_choice.get("type") in ("auto", "any", "tool"):
                    anth_body["tool_choice"] = tool_choice

    return anth_body


def anthropic_to_openai_response(data: dict, model_id: str) -> dict[str, Any]:
    """Translate an Anthropic `/v1/messages` JSON response into an OpenAI `chat.completion` object."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for idx, block in enumerate(data.get("content") or []):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif btype == "thinking" and block.get("thinking"):
            reasoning_parts.append(block["thinking"])
        elif btype == "tool_use":
            inp = block.get("input")
            args_str = json.dumps(inp if isinstance(inp, dict) else {})
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{idx}_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "tool",
                        "arguments": args_str,
                    },
                }
            )

    stop_reason = data.get("stop_reason")
    if stop_reason == "tool_use" or tool_calls:
        finish_reason = "tool_calls"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts or not tool_calls else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = data.get("usage") or {}
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)

    return {
        "id": data.get("id") or f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model") or model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


class _AnthropicStreamTranslator:
    """Stateful line-by-line translator from Anthropic SSE to OpenAI `chat.completion.chunk` SSE."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.chat_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        self.created = int(time.time())
        self.tool_call_index = -1
        self.has_tool_calls = False
        self.finish_reason = "stop"
        self.input_tokens = 0
        self.output_tokens = 0
        self.output_chars = 0
        self.sent_role = False

    def _make_chunk(
        self,
        delta: dict[str, Any],
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "id": self.chat_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    def feed_line(self, line: str) -> list[bytes]:
        s = line.strip()
        if not s.startswith("data:"):
            return []
        payload_str = s[5:].strip()
        if not payload_str or payload_str == "[DONE]":
            return []
        try:
            obj = json.loads(payload_str)
        except Exception:
            return []
        if not isinstance(obj, dict):
            return []

        out: list[bytes] = []
        etype = obj.get("type")

        if etype == "message_start":
            msg = obj.get("message") or {}
            if msg.get("id"):
                self.chat_id = msg["id"]
            usage = msg.get("usage") or {}
            self.input_tokens = int(usage.get("input_tokens") or 0)
            if not self.sent_role:
                self.sent_role = True
                out.append(self._make_chunk({"role": "assistant", "content": ""}))

        elif etype == "content_block_start":
            cb = obj.get("content_block") or {}
            if cb.get("type") == "tool_use":
                self.tool_call_index += 1
                self.has_tool_calls = True
                tc_id = cb.get("id") or f"call_{self.tool_call_index}_{uuid.uuid4().hex[:8]}"
                fn_name = cb.get("name") or "tool"
                self.output_chars += len(fn_name)
                delta: dict[str, Any] = {
                    "tool_calls": [
                        {
                            "index": self.tool_call_index,
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": ""},
                        }
                    ]
                }
                if not self.sent_role:
                    self.sent_role = True
                    delta["role"] = "assistant"
                out.append(self._make_chunk(delta))

        elif etype == "content_block_delta":
            delta_obj = obj.get("delta") or {}
            dtype = delta_obj.get("type")
            if dtype == "text_delta":
                text = delta_obj.get("text") or ""
                if text:
                    self.output_chars += len(text)
                    d: dict[str, Any] = {"content": text}
                    if not self.sent_role:
                        self.sent_role = True
                        d["role"] = "assistant"
                    out.append(self._make_chunk(d))
            elif dtype == "thinking_delta":
                thinking = delta_obj.get("thinking") or ""
                if thinking:
                    self.output_chars += len(thinking)
                    d = {"reasoning_content": thinking}
                    if not self.sent_role:
                        self.sent_role = True
                        d["role"] = "assistant"
                    out.append(self._make_chunk(d))
            elif dtype == "input_json_delta":
                partial_json = delta_obj.get("partial_json") or ""
                if partial_json:
                    self.output_chars += len(partial_json)
                    out.append(
                        self._make_chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": max(0, self.tool_call_index),
                                        "function": {"arguments": partial_json},
                                    }
                                ]
                            }
                        )
                    )

        elif etype == "message_delta":
            delta_obj = obj.get("delta") or {}
            stop_reason = delta_obj.get("stop_reason")
            if stop_reason == "tool_use" or self.has_tool_calls:
                self.finish_reason = "tool_calls"
            elif stop_reason == "max_tokens":
                self.finish_reason = "length"
            else:
                self.finish_reason = "stop"
            usage = obj.get("usage") or {}
            if "output_tokens" in usage:
                self.output_tokens = int(usage.get("output_tokens") or 0)

        elif etype == "error":
            err = obj.get("error") or {"message": "Anthropic stream error"}
            out.append(f"data: {json.dumps({'error': err})}\n\n".encode("utf-8"))

        return out

    def finalize(self) -> list[bytes]:
        usage = {
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens or ((self.output_chars + 3) // 4),
            "total_tokens": self.input_tokens + (self.output_tokens or ((self.output_chars + 3) // 4)),
        }
        return [
            self._make_chunk({}, finish_reason=self.finish_reason, usage=usage),
            b"data: [DONE]\n\n",
        ]


def openai_to_gemini_request(body: dict) -> dict[str, Any]:
    """Translate an OpenAI `/v1/chat/completions` request payload into Google Gemini `generateContent` format."""
    system_parts: list[str] = []
    if body.get("system"):
        sys_txt = flatten_content(body["system"]).strip()
        if sys_txt:
            system_parts.append(sys_txt)

    contents: list[dict[str, Any]] = []
    tool_id_to_name: dict[str, str] = {}
    raw_messages = body.get("messages") or []

    for m in raw_messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")

        if role in ("system", "developer"):
            sys_txt = flatten_content(content).strip()
            if sys_txt:
                system_parts.append(sys_txt)
            continue

        if role == "tool":
            tc_id = str(m.get("tool_call_id") or m.get("id") or "")
            fn_name = str(m.get("name") or tool_id_to_name.get(tc_id) or "tool")
            raw_res = flatten_content(content)
            try:
                parsed_res = json.loads(raw_res)
                if not isinstance(parsed_res, dict):
                    parsed_res = {"result": raw_res}
            except Exception:
                parsed_res = {"result": raw_res}

            part = {
                "functionResponse": {
                    "name": fn_name,
                    "response": parsed_res,
                }
            }
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            continue

        if role in ("assistant", "model"):
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                if content:
                    parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and item:
                        parts.append({"text": item})
                    elif isinstance(item, dict):
                        itype = item.get("type")
                        if itype == "text" and item.get("text"):
                            parts.append({"text": item["text"]})
                        elif itype == "tool_use":
                            tc_id = str(item.get("id") or "")
                            fn_name = str(item.get("name") or "tool")
                            if tc_id:
                                tool_id_to_name[tc_id] = fn_name
                            args_obj = _parse_json_args(item.get("input"))
                            fc_part: dict[str, Any] = {
                                "functionCall": {"name": fn_name, "args": args_obj}
                            }
                            sig = _lookup_gemini_thought_signature(tc_id, fn_name, args_obj)
                            if sig:
                                fc_part["thoughtSignature"] = sig
                            parts.append(fc_part)

            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = str(tc.get("id") or "")
                    fn = tc.get("function") or {}
                    fn_name = str(fn.get("name") or "tool")
                    if tc_id:
                        tool_id_to_name[tc_id] = fn_name
                    args_obj = _parse_json_args(fn.get("arguments"))
                    fc_part = {"functionCall": {"name": fn_name, "args": args_obj}}
                    sig = (
                        tc.get("thought_signature")
                        or (tc.get("extra_content") or {}).get("google", {}).get("thought_signature")
                        or _lookup_gemini_thought_signature(tc_id, fn_name, args_obj)
                    )
                    if sig:
                        fc_part["thoughtSignature"] = sig
                    parts.append(fc_part)

            if parts:
                if contents and contents[-1]["role"] == "model":
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": "model", "parts": parts})
            continue

        # Default: user role
        user_parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            if content:
                user_parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str) and item:
                    user_parts.append({"text": item})
                elif isinstance(item, dict):
                    itype = item.get("type")
                    img = _parse_image_part(item)
                    if img is not None:
                        if img["kind"] == "base64":
                            user_parts.append(
                                {
                                    "inlineData": {
                                        "mimeType": img["media_type"],
                                        "data": img["data"],
                                    }
                                }
                            )
                        elif img["kind"] == "url":
                            user_parts.append(
                                {
                                    "fileData": {
                                        "mimeType": img["media_type"],
                                        "fileUri": img["url"],
                                    }
                                }
                            )
                    elif itype == "text" and item.get("text"):
                        user_parts.append({"text": item["text"]})
                    elif itype == "tool_result":
                        tc_id = str(item.get("tool_use_id") or "")
                        fn_name = str(item.get("name") or tool_id_to_name.get(tc_id) or "tool")
                        res_txt = flatten_content(item.get("content"))
                        user_parts.append(
                            {
                                "functionResponse": {
                                    "name": fn_name,
                                    "response": {"result": res_txt},
                                }
                            }
                        )

        for img in _extract_message_extra_images(m):
            if img["kind"] == "base64":
                user_parts.append(
                    {
                        "inlineData": {
                            "mimeType": img["media_type"],
                            "data": img["data"],
                        }
                    }
                )
            elif img["kind"] == "url":
                user_parts.append(
                    {
                        "fileData": {
                            "mimeType": img["media_type"],
                            "fileUri": img["url"],
                        }
                    }
                )

        if user_parts:
            if contents and contents[-1]["role"] == "user":
                contents[-1]["parts"].extend(user_parts)
            else:
                contents.append({"role": "user", "parts": user_parts})

    if not contents:
        prompt_txt = flatten_content(body.get("prompt") or "") or "."
        contents.append({"role": "user", "parts": [{"text": prompt_txt}]})

    gemini_body: dict[str, Any] = {"contents": contents}

    if system_parts:
        gemini_body["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}]
        }

    gen_cfg: dict[str, Any] = {}
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_tokens is not None:
        try:
            gen_cfg["maxOutputTokens"] = max(1, int(max_tokens))
        except Exception:
            pass

    if body.get("temperature") is not None:
        try:
            gen_cfg["temperature"] = float(body["temperature"])
        except Exception:
            pass

    if body.get("top_p") is not None:
        try:
            gen_cfg["topP"] = float(body["top_p"])
        except Exception:
            pass

    stop = body.get("stop") or body.get("stop_sequences")
    if stop:
        if isinstance(stop, str):
            gen_cfg["stopSequences"] = [stop]
        elif isinstance(stop, list):
            gen_cfg["stopSequences"] = [str(s) for s in stop if s][:5]

    if gen_cfg:
        gemini_body["generationConfig"] = gen_cfg

    raw_tools = body.get("tools")
    tool_choice = body.get("tool_choice")
    if isinstance(raw_tools, list) and raw_tools and tool_choice != "none":
        decls: list[dict[str, Any]] = []
        for t in raw_tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if t.get("type") == "function" else t
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            decl: dict[str, Any] = {
                "name": fn["name"],
                "description": fn.get("description") or "",
            }
            raw_params = fn.get("parameters") or fn.get("input_schema")
            if isinstance(raw_params, dict) and raw_params:
                cleaned_params = _clean_gemini_schema(raw_params)
                if cleaned_params.get("properties") or cleaned_params.get("anyOf"):
                    decl["parameters"] = cleaned_params
            decls.append(decl)

        if decls:
            gemini_body["tools"] = [{"functionDeclarations": decls}]
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    gemini_body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
                elif tool_choice in ("required", "any"):
                    gemini_body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
            elif isinstance(tool_choice, dict):
                if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
                    fn_name = tool_choice["function"].get("name")
                    if fn_name:
                        gemini_body["toolConfig"] = {
                            "functionCallingConfig": {
                                "mode": "ANY",
                                "allowedFunctionNames": [fn_name],
                            }
                        }

    return gemini_body


def gemini_to_openai_response(data: dict, model_id: str) -> dict[str, Any]:
    """Translate a Google Gemini `GenerateContentResponse` into an OpenAI `chat.completion` object."""
    candidates = data.get("candidates") or []
    cand = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    parts = (cand.get("content") or {}).get("parts") or []

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True and part.get("text"):
            reasoning_parts.append(part["text"])
        elif part.get("text"):
            text_parts.append(part["text"])

        if isinstance(part.get("functionCall"), dict):
            fc = part["functionCall"]
            fn_name = fc.get("name") or "tool"
            args_obj = fc.get("args") if isinstance(fc.get("args"), dict) else {}
            tc_id = fc.get("id") or f"call_{idx}_{uuid.uuid4().hex[:10]}"
            if part.get("thoughtSignature"):
                _store_gemini_thought_signature(tc_id, fn_name, args_obj, part["thoughtSignature"])
            tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(args_obj),
                    },
                }
            )

    finish_raw = cand.get("finishReason") or "STOP"
    if tool_calls:
        finish_reason = "tool_calls"
    elif finish_raw == "MAX_TOKENS":
        finish_reason = "length"
    elif finish_raw in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
        finish_reason = "content_filter"
    else:
        finish_reason = "stop"

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts or not tool_calls else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_meta = data.get("usageMetadata") or {}
    in_tok = int(usage_meta.get("promptTokenCount") or 0)
    out_tok = int(usage_meta.get("candidatesTokenCount") or 0)
    tot_tok = int(usage_meta.get("totalTokenCount") or (in_tok + out_tok))

    return {
        "id": data.get("responseId") or f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": tot_tok,
        },
    }


class _GeminiStreamTranslator:
    """Stateful line-by-line translator from Gemini `?alt=sse` stream to OpenAI `chat.completion.chunk` SSE."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.chat_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        self.created = int(time.time())
        self.tool_call_index = 0
        self.has_tool_calls = False
        self.finish_reason = "stop"
        self.latest_usage: dict[str, int] | None = None
        self.output_chars = 0
        self.sent_role = False

    def _make_chunk(
        self,
        delta: dict[str, Any],
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "id": self.chat_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n".encode("utf-8")

    def feed_line(self, line: str) -> list[bytes]:
        s = line.strip()
        if not s.startswith("data:"):
            return []
        payload_str = s[5:].strip()
        if not payload_str or payload_str == "[DONE]":
            return []
        try:
            obj = json.loads(payload_str)
        except Exception:
            return []
        if not isinstance(obj, dict):
            return []

        if obj.get("responseId"):
            self.chat_id = obj["responseId"]

        usage_meta = obj.get("usageMetadata")
        if isinstance(usage_meta, dict):
            in_tok = int(usage_meta.get("promptTokenCount") or 0)
            out_tok = int(usage_meta.get("candidatesTokenCount") or 0)
            self.latest_usage = {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": int(usage_meta.get("totalTokenCount") or (in_tok + out_tok)),
            }

        out: list[bytes] = []
        candidates = obj.get("candidates") or []
        if not candidates or not isinstance(candidates[0], dict):
            return out

        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("thought") is True and part.get("text"):
                txt = part["text"]
                self.output_chars += len(txt)
                d: dict[str, Any] = {"reasoning_content": txt}
                if not self.sent_role:
                    self.sent_role = True
                    d["role"] = "assistant"
                out.append(self._make_chunk(d))
            elif part.get("text"):
                txt = part["text"]
                self.output_chars += len(txt)
                d = {"content": txt}
                if not self.sent_role:
                    self.sent_role = True
                    d["role"] = "assistant"
                out.append(self._make_chunk(d))

            if isinstance(part.get("functionCall"), dict):
                fc = part["functionCall"]
                fn_name = fc.get("name") or "tool"
                args_obj = fc.get("args") if isinstance(fc.get("args"), dict) else {}
                args_str = json.dumps(args_obj)
                tc_id = fc.get("id") or f"call_{self.tool_call_index}_{uuid.uuid4().hex[:10]}"
                if part.get("thoughtSignature"):
                    _store_gemini_thought_signature(tc_id, fn_name, args_obj, part["thoughtSignature"])

                self.output_chars += len(fn_name) + len(args_str)
                self.has_tool_calls = True
                d = {
                    "tool_calls": [
                        {
                            "index": self.tool_call_index,
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": args_str,
                            },
                        }
                    ]
                }
                if not self.sent_role:
                    self.sent_role = True
                    d["role"] = "assistant"
                out.append(self._make_chunk(d))
                self.tool_call_index += 1

        finish_raw = cand.get("finishReason")
        if finish_raw:
            if self.has_tool_calls:
                self.finish_reason = "tool_calls"
            elif finish_raw == "MAX_TOKENS":
                self.finish_reason = "length"
            elif finish_raw in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
                self.finish_reason = "content_filter"
            else:
                self.finish_reason = "stop"

        return out

    def finalize(self) -> list[bytes]:
        if self.has_tool_calls:
            self.finish_reason = "tool_calls"
        return [
            self._make_chunk({}, finish_reason=self.finish_reason, usage=self.latest_usage),
            b"data: [DONE]\n\n",
        ]


def resolve_effective_api_type(prov: dict) -> str:
    """Which upstream protocol to speak: `openai-completions`, `openai-legacy-completions`,
    `anthropic-messages` or `google-gemini`. base_url also auto-detects Anthropic/Google."""
    api_type = (prov.get("api_type") or "openai-completions").strip().lower()
    host = (urlsplit((prov.get("base_url") or "").strip()).hostname or "").lower()
    if api_type in ("google-gemini", "gemini", "google") or "generativelanguage.googleapis.com" in host:
        return "google-gemini"
    if api_type in ("anthropic-messages", "anthropic") or "api.anthropic.com" in host:
        return "anthropic-messages"
    if api_type == "openai-legacy-completions":
        return "openai-legacy-completions"
    return "openai-completions"


def resolve_anthropic_url(prov: dict) -> str:
    """Native Anthropic /v1/messages URL, from a base_url with or without /v1."""
    base = (prov.get("base_url") or "https://api.anthropic.com").rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1/messages", "/messages", "/v1"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            break
    return f"{base}/v1/messages"


def resolve_gemini_url(prov: dict, model_id: str, is_stream: bool) -> str:
    """Native Gemini generateContent / streamGenerateContent URL. Works whether base_url is
    https://generativelanguage.googleapis.com, .../v1beta, or .../v1beta/openai/chat/completions."""
    raw_base = (prov.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
    parsed = urlsplit(raw_base)
    scheme_netloc = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else raw_base
    path = parsed.path.rstrip("/")
    for suffix in ("/openai/chat/completions", "/openai/completions", "/openai",
                   "/chat/completions", "/completions", "/models"):
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
    if not path:
        path = "/v1beta"
    clean_model = model_id.strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model[len("models/"):]
    action = "streamGenerateContent?alt=sse" if is_stream else "generateContent"
    return f"{scheme_netloc}{path}/models/{clean_model}:{action}"


def resolve_upstream_url(prov: dict, endpoint_suffix: str | None, model_id: str = "", is_stream: bool = False) -> str:
    """Build the upstream URL from base_url, whether or not it already ends in /v1."""
    eff_type = resolve_effective_api_type(prov)
    if eff_type == "google-gemini":
        return resolve_gemini_url(prov, model_id=model_id, is_stream=is_stream)
    if eff_type == "anthropic-messages":
        return resolve_anthropic_url(prov)

    target = "completions" if eff_type == "openai-legacy-completions" or endpoint_suffix == "completions" \
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


def resolve_upstream_headers(prov: dict, effective_api_type: str | None = None) -> dict:
    """Provider auth header: Bearer for OpenAI-ish, x-api-key for Anthropic, x-goog-api-key for Google."""
    headers = {"content-type": "application/json"}
    api_key = prov.get("api_key")
    eff_type = effective_api_type or resolve_effective_api_type(prov)
    if api_key:
        if eff_type == "google-gemini":
            headers["x-goog-api-key"] = api_key
        elif eff_type == "anthropic-messages":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["authorization"] = f"Bearer {api_key}"
    return headers


async def forward(body: dict, cand: tuple, endpoint_suffix: str | None):
    _, pname, model, prov = cand
    model_id = model["id"]
    eff_type = resolve_effective_api_type(prov)
    url = resolve_upstream_url(prov, endpoint_suffix, model_id, is_stream=bool(body.get("stream")))
    headers = resolve_upstream_headers(prov, effective_api_type=eff_type)

    if eff_type == "google-gemini":
        fwd_body = openai_to_gemini_request(body)
    elif eff_type == "anthropic-messages":
        fwd_body = openai_to_anthropic_request(body, model_id=model_id)
    else:
        fwd_body = {**body, "model": model_id}

    if not body.get("stream"):
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                r = await client.post(url, json=fwd_body, headers=headers)
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {
                "message": f"upstream '{pname}' request failed: {e}", "type": "upstream_error", "code": 502}})
        try:
            data = r.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            if eff_type == "google-gemini":
                data = gemini_to_openai_response(data, model_id=model_id)
            elif eff_type == "anthropic-messages":
                data = anthropic_to_openai_response(data, model_id=model_id)
        if r.status_code >= 400 and isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]  # Google APIs wrap errors in a list
        if data is not None:
            return JSONResponse(content=data, status_code=r.status_code)
        return PlainTextResponse(r.text, status_code=r.status_code,
                                 media_type=r.headers.get("content-type", "text/plain"))

    translator = _GeminiStreamTranslator(model_id=model_id) if eff_type == "google-gemini" \
        else _AnthropicStreamTranslator(model_id=model_id) if eff_type == "anthropic-messages" else None

    async def stream():
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                async with client.stream("POST", url, json=fwd_body, headers=headers) as r:
                    if r.status_code >= 300:
                        err = {"error": {"message": (await r.aread()).decode(errors="replace"),
                                         "type": "upstream_error", "code": r.status_code}}
                        yield f"data: {json.dumps(err)}\n\n".encode()
                        return
                    if translator is None:
                        async for chunk in r.aiter_bytes():
                            yield chunk
                        return
                    buffer = ""
                    async for chunk in r.aiter_bytes():
                        buffer += chunk.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            for out in translator.feed_line(line):
                                yield out
                    if buffer.strip():
                        for out in translator.feed_line(buffer):
                            yield out
                    for out in translator.finalize():
                        yield out
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
