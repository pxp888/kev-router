"""Self-check for router.py logic (no network, no upstream needed): python3 selfcheck.py"""
import asyncio
import copy
import json

import router

TEST_CFG = {
    "default_model": "cheap",
    "providers": [
        {"name": "p1", "base_url": "http://h1:11434/v1", "models": [
            {"model_id": "cheap", "description": "fast summaries", "weight": 1.0},
            {"model_id": "dup", "description": "shared id", "weight": 1.0}]},
        {"name": "p2", "base_url": "http://h2:8080", "models": [
            {"model_id": "dup", "description": "shared id again", "weight": 2.0},
            {"model_id": "off", "description": "disabled", "weight": 9.0, "enabled": False}]},
    ],
}


def check_state():
    s = router.build_state({"messages": [
        {"role": "system", "content": "BOILERPLATE " * 500},
        {"role": "user", "content": "summarize this"},
        {"role": "assistant", "content": [{"type": "text", "text": "sure"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]},
    ]})
    assert "BOILERPLATE" not in s, "system message leaked into routing state"
    assert "user:\nsummarize this" in s and "assistant:\n" in s
    assert "[Attached Image]" in s, "image not marked for vision routing"
    assert len(s) <= router.STATE_CHARS
    assert router.build_state({"prompt": "legacy prompt"}) == "legacy prompt"


def check_candidates():
    keys = [c[0] for c in router.candidates(TEST_CFG)]
    assert keys == ["cheap", "dup", "p2:dup"], keys  # collision gets provider prefix
    assert "off" not in keys


def check_url_resolution():
    p = lambda base, api="openai-completions": {"base_url": base, "api_type": api}
    assert router.resolve_upstream_url(p("http://h:11434/v1"), "chat/completions", "m") == "http://h:11434/v1/chat/completions"
    assert router.resolve_upstream_url(p("http://h:11434"), "chat/completions", "m") == "http://h:11434/v1/chat/completions"
    assert router.resolve_upstream_url(p("http://h:11434/v1/chat/completions"), "chat/completions", "m") == "http://h:11434/v1/chat/completions"
    assert router.resolve_upstream_url(p("http://h:11434/v1"), "completions", "m") == "http://h:11434/v1/completions"
    assert router.resolve_upstream_url(p("http://h:8080/v1", "openai-legacy-completions"), "chat/completions", "m") == "http://h:8080/v1/completions"


async def check_decision():
    real = router.kev_question

    async def stub(client, cfg, state, criteria, instructions):
        # "dup" has the lower raw probability but weight 2.0 -> it must win
        return {"probabilities": {k: (0.50 if k == "cheap" else 0.45 if k == "p2:dup" else 0.05) for k in criteria},
                "confidence": 0.4}

    router.kev_question = stub
    cand, fell_back = await router.decide(TEST_CFG, "user: fix this stack trace")
    assert not fell_back and cand[2]["id"] == "dup" and cand[1] == "p2", cand

    async def broken(client, cfg, state, criteria, instructions):
        raise RuntimeError("kev down")

    router.kev_question = broken
    cand, fell_back = await router.decide(TEST_CFG, "user: hello")
    assert fell_back and cand[2]["id"] == "cheap", cand

    router.kev_question = stub
    empty = copy.deepcopy(TEST_CFG)
    empty["providers"][0]["models"][0]["enabled"] = False
    empty["providers"][1]["models"][0]["enabled"] = False
    try:
        await router.decide({"providers": []}, "")
        raise AssertionError("expected ValueError with no models")
    except ValueError:
        pass
    router.kev_question = real


def check_config():
    inc = {"default_model": "x", "decision_prompt": "  ", "providers": [
        {"id": "p1", "name": "p1", "base_url": "http://h1:11434/v1", "api_type": "openai-completions",
         "api_key": None, "models": [{"id": "m1", "model_id": "x", "description": "d", "weight": "1.5", "enabled": True}]},
        {"id": "pnew", "name": "pnew", "base_url": "http://h2", "api_key": "sk-new",
         "models": [{"id": "m2", "model_id": "y", "description": "d2", "weight": 1}]}]}
    old = {"providers": [{"id": "p1", "name": "p1", "base_url": "http://h1", "api_key": "sk-old", "models": []}],
           "kev_api_key": "kevkey"}

    out = router.merge_config(inc, old)
    assert out["providers"][0]["api_key"] == "sk-old", "null api_key must keep the stored key"
    assert out["providers"][1]["api_key"] == "sk-new"
    assert out["providers"][0]["models"][0]["weight"] == 1.5
    assert out["decision_prompt"] == router.DEFAULT_DECISION_PROMPT
    assert out["kev_api_key"] == "kevkey"
    assert router.validate_config(out) is None

    cleared = router.merge_config({**inc, "providers": [{**inc["providers"][0], "clear_api_key": True}]}, old)
    assert cleared["providers"][0]["api_key"] == ""

    # a stored config with no per-provider id must still keep its key when the webui
    # sends its own generated provider id
    legacy = {"providers": [{"name": "p1", "base_url": "http://h1", "api_key": "sk-legacy", "models": []}]}
    kept = router.merge_config({**inc, "providers": [{**inc["providers"][0], "id": "prov_12345"}]}, legacy)
    assert kept["providers"][0]["api_key"] == "sk-legacy", kept["providers"][0]

    def cfg_with(**over):
        base = {"providers": [{"name": "a", "base_url": "http://h", "models": [{"model_id": "m", "description": "d"}]}],
                "default_model": "m"}
        base["providers"][0].update(over.pop("prov", {}))
        base["providers"][0]["models"][0].update(over.pop("model", {}))
        base.update(over)
        return base

    assert router.validate_config(cfg_with(providers=[]))
    assert router.validate_config(cfg_with(default_model=""))
    assert router.validate_config(cfg_with(model={"description": "  "}))
    assert router.validate_config(cfg_with(prov={"name": ""}))
    assert router.validate_config(cfg_with(prov={"base_url": "h11434"}))          # no scheme
    assert router.validate_config(cfg_with(prov={"base_url": "ftp://h"}))         # wrong scheme
    assert router.validate_config(cfg_with(prov={"base_url": "http://u:p@h"}))    # embedded creds
    assert router.validate_config(cfg_with(prov={"base_url": "http://h/x?a=1"}))  # query string
    assert router.validate_config(cfg_with(prov={"base_url": "http://h t"}))      # space

    masked = router.mask_secrets(out)
    assert masked["providers"][0]["api_key"] == "" and masked["providers"][0]["api_key_masked"].endswith("old")
    assert masked["kev_api_key"] == "" and masked["kev_api_key_masked"].endswith("vkey")


def check_adapters():
    prov = lambda base, api=None: {"base_url": base, "api_type": api or "openai-completions", "api_key": "k"}

    # protocol pick, including auto-detection from base_url
    assert router.resolve_effective_api_type(prov("http://h/v1")) == "openai-completions"
    assert router.resolve_effective_api_type(prov("https://api.anthropic.com")) == "anthropic-messages"
    assert router.resolve_effective_api_type(prov("https://generativelanguage.googleapis.com/v1beta")) == "google-gemini"

    assert router.resolve_upstream_url(prov("https://api.anthropic.com/v1", "anthropic-messages"), "chat/completions", "claude") \
        == "https://api.anthropic.com/v1/messages"
    assert router.resolve_upstream_url(prov("https://api.anthropic.com", "anthropic-messages"), None, "claude") \
        == "https://api.anthropic.com/v1/messages"
    assert router.resolve_upstream_url(prov("https://gen.example/v1beta/openai/chat/completions", "google-gemini"),
                                       None, "models/gemini-2.5-flash", True) \
        == "https://gen.example/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    assert router.resolve_upstream_url(prov("https://gen.example", "google-gemini"), None, "gemini") \
        == "https://gen.example/v1beta/models/gemini:generateContent"
    assert router.resolve_upstream_headers(prov("x", "anthropic-messages"))["x-api-key"] == "k"
    assert router.resolve_upstream_headers(prov("x", "google-gemini"))["x-goog-api-key"] == "k"
    assert router.resolve_upstream_headers(prov("x")).get("authorization") == "Bearer k"

    openai_body = {
        "model": "ignored", "max_tokens": 321, "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "look at this"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "{\"hit\": 7}"},
            {"role": "user", "content": "answer now"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "description": "d",
                                                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
        "tool_choice": "required",
    }

    anth = router.openai_to_anthropic_request(openai_body, "claude-x")
    assert anth["model"] == "claude-x" and anth["max_tokens"] == 321 and anth["system"] == "be terse"
    assert anth["messages"][0]["role"] == "user"
    assert anth["messages"][0]["content"][1] == {"type": "image",
                                                  "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}
    assert anth["messages"][1]["content"][0]["type"] == "tool_use" \
        and anth["messages"][1]["content"][0]["input"] == {"q": "x"}
    assert anth["messages"][2]["content"][0]["type"] == "tool_result" \
        and anth["messages"][2]["content"][0]["tool_use_id"] == "call_1"
    assert anth["tools"][0]["input_schema"]["properties"]["q"]["type"] == "string"
    assert anth["tool_choice"] == {"type": "any"}
    minimal = router.openai_to_anthropic_request({"messages": [{"role": "user", "content": "hi"}]}, "m")
    assert isinstance(minimal["max_tokens"], int) and minimal["max_tokens"] > 0, "anthropic needs an explicit max_tokens"

    anth_reply = {"id": "msg_1", "model": "claude-x", "stop_reason": "tool_use",
                  "usage": {"input_tokens": 11, "output_tokens": 5},
                  "content": [{"type": "thinking", "thinking": "hmm"},
                              {"type": "text", "text": "answer"},
                              {"type": "tool_use", "id": "call_9", "name": "lookup", "input": {"q": "y"}}]}
    oai = router.anthropic_to_openai_response(anth_reply, "claude-x")
    assert oai["choices"][0]["message"]["content"] == "answer"
    assert oai["choices"][0]["message"]["reasoning_content"] == "hmm"
    assert oai["choices"][0]["message"]["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"q": "y"}'}
    assert oai["choices"][0]["finish_reason"] == "tool_calls" and oai["usage"]["total_tokens"] == 16

    gem = router.openai_to_gemini_request(openai_body)
    assert gem["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert gem["contents"][0]["role"] == "user"
    assert any("inlineData" in p for p in gem["contents"][0]["parts"]), "image dropped in gemini request"
    assert gem["contents"][1]["role"] == "model" and gem["contents"][1]["parts"][0]["functionCall"]["args"] == {"q": "x"}
    assert gem["contents"][2]["parts"][0]["functionResponse"]["response"] == {"hit": 7}
    assert gem["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["q"]["type"] == "string"

    gem_reply = {"responseId": "r1", "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4},
                 "candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                       {"text": "final"},
                                                       {"functionCall": {"name": "lookup", "args": {"q": "z"}}}]},
                                 "finishReason": "MAX_TOKENS"}]}
    oai = router.gemini_to_openai_response(gem_reply, "gemini-x")
    assert oai["choices"][0]["message"]["content"] == "final"
    assert oai["choices"][0]["message"]["reasoning_content"] == "thinking"
    assert oai["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert oai["choices"][0]["finish_reason"] == "tool_calls" and oai["usage"]["total_tokens"] == 12
    assert router.gemini_to_openai_response(
        {"candidates": [{"content": {"parts": [{"text": "cut off"}]}, "finishReason": "MAX_TOKENS"}]}, "g"
    )["choices"][0]["finish_reason"] == "length"

    def drain(translator, sse):
        out = []
        for line in sse.splitlines():
            out += translator.feed_line(line)
        out += translator.finalize()
        return [json.loads(c.decode().removeprefix("data: ").strip()) for c in out if b"[DONE]" not in c]

    anth_stream = """event: message_start
data: {"type":"message_start","message":{"id":"msg_7","usage":{"input_tokens":10}}}
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}
event: content_block_stop
data: {"type":"content_block_stop","index":0}
event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":3}}
event: message_stop
data: {"type":"message_stop"}
"""
    chunks = drain(router._AnthropicStreamTranslator("claude-x"), anth_stream)
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "Hello world"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length" and chunks[-1]["usage"]["completion_tokens"] == 3

    gem_stream = """data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}
data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"lookup","args":{"q":"w"}}}]},"finishReason":"STOP"}]}
"""
    chunks = drain(router._GeminiStreamTranslator("gemini-x"), gem_stream)
    assert "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks) == "Hi"
    assert any("tool_calls" in c["choices"][0]["delta"] for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


if __name__ == "__main__":
    check_state()
    check_candidates()
    check_url_resolution()
    check_adapters()
    check_config()
    asyncio.run(check_decision())
    print("selfcheck OK")
