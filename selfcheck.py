"""Self-check for router.py logic (no network, no upstream needed): python3 selfcheck.py"""
import asyncio
import copy

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


if __name__ == "__main__":
    check_state()
    check_candidates()
    check_url_resolution()
    check_config()
    asyncio.run(check_decision())
    print("selfcheck OK")
