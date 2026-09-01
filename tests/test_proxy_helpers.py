"""Pure-function tests for proxy helpers in main.py.

These don't need a running app — just the functions, exercised directly.
"""
import sys

import pytest
from conftest import fresh_config


@pytest.fixture
def main_module(monkeypatch, tmp_path):
    """A freshly-loaded main module with a minimal config."""
    fresh_config(monkeypatch, tmp_path, {
        "providers": {"a": {"base_url": "https://a.test", "api_key": "k"}},
        "groups": {"default": {"endpoints": [{"provider": "a"}]}},
    })
    sys.modules.pop("main", None)
    import main
    return main


# --- _extract_usage_from_sse ---

def test_sse_extracts_completion_tokens_from_finished_event(main_module):
    buf = bytearray()
    chunk = b'data: {"usage": {"completion_tokens": 42}}\n\n'
    assert main_module._extract_usage_from_sse(buf, chunk) == 42


def test_sse_returns_none_until_event_terminator_seen(main_module):
    buf = bytearray()
    # Partial event — no \n\n terminator yet
    assert main_module._extract_usage_from_sse(buf, b'data: {"usage": {"completion_tokens": 7}}') is None
    # Now flush the terminator
    assert main_module._extract_usage_from_sse(buf, b"\n\n") == 7


def test_sse_skips_done_sentinel(main_module):
    buf = bytearray()
    assert main_module._extract_usage_from_sse(buf, b"data: [DONE]\n\n") is None


def test_sse_ignores_events_without_usage(main_module):
    buf = bytearray()
    chunk = b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
    assert main_module._extract_usage_from_sse(buf, chunk) is None


def test_sse_tolerates_malformed_json(main_module):
    buf = bytearray()
    assert main_module._extract_usage_from_sse(buf, b"data: {not json\n\n") is None


@pytest.mark.asyncio
async def test_close_quietly_does_not_swallow_cancellation(main_module):
    import asyncio

    class Response:
        async def aclose(self):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await main_module._close_quietly(Response())


# --- _parse_model_suffix ---

def test_parse_model_suffix_no_suffix(main_module):
    assert main_module._parse_model_suffix("gpt-4o") == ("gpt-4o", None)


def test_parse_model_suffix_race(main_module):
    assert main_module._parse_model_suffix("cheap:race") == ("cheap", "race")


def test_parse_model_suffix_seq(main_module):
    assert main_module._parse_model_suffix("cheap:seq") == ("cheap", "seq")


def test_parse_model_suffix_fastest_aliases_race(main_module):
    assert main_module._parse_model_suffix("cheap:fastest") == ("cheap", "race")


def test_parse_model_suffix_normal_aliases_seq(main_module):
    assert main_module._parse_model_suffix("cheap:normal") == ("cheap", "seq")


def test_parse_model_suffix_double_suffix_raises(main_module):
    import pytest
    with pytest.raises(ValueError):
        main_module._parse_model_suffix("cheap:race:seq")
    with pytest.raises(ValueError):
        main_module._parse_model_suffix("cheap:fastest:normal")


def test_parse_model_suffix_is_case_insensitive(main_module):
    """Match suffixes regardless of case; stem keeps its original case."""
    assert main_module._parse_model_suffix("GPT-4o:RACE") == ("GPT-4o", "race")
    assert main_module._parse_model_suffix("GPT-4o:Fastest") == ("GPT-4o", "race")
    assert main_module._parse_model_suffix("GPT-4o:Seq") == ("GPT-4o", "seq")


def test_parse_model_suffix_colon_in_name_not_a_suffix(main_module):
    """A model name like 'hf:foo/bar' must not be treated as a suffixed name."""
    assert main_module._parse_model_suffix("hf:foo/bar") == ("hf:foo/bar", None)


# --- _effective_model ---

def test_effective_model_endpoint_model_wins(main_module):
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="gpt-4o-mini")
    assert main_module._effective_model(ep, "client-model") == "gpt-4o-mini"


def test_effective_model_falls_back_to_client_model(main_module):
    """No endpoint model → client's requested model passes through."""
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="")
    assert main_module._effective_model(ep, "gpt-4o") == "gpt-4o"


# --- _strip_unsupported ---

def test_strip_unsupported_drops_unknown_keys(main_module):
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="m")
    out = main_module._strip_unsupported(
        {"model": "ignored", "messages": [], "bogus_param": 1, "another": "x"},
        ep,
    )
    assert "bogus_param" not in out
    assert "another" not in out
    assert out["model"] == "m"


def test_strip_unsupported_removes_reasoning_from_messages_by_default(main_module):
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="m", keep_reasoning=False)
    out = main_module._strip_unsupported(
        {"messages": [
            {"role": "assistant", "content": "hi", "reasoning": "...", "thinking": "..."},
            {"role": "user", "content": "yo"},
        ]},
        ep,
    )
    msgs = out["messages"]
    assert "reasoning" not in msgs[0] and "thinking" not in msgs[0]
    assert msgs[0]["content"] == "hi"
    assert msgs[1] == {"role": "user", "content": "yo"}


def test_strip_unsupported_forwards_top_level_reasoning_param(main_module):
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="m", keep_reasoning=False)
    out = main_module._strip_unsupported(
        {"messages": [{"role": "user", "content": "hi", "reasoning": "drop"}],
         "reasoning": {"effort": "low"}},
        ep,
    )
    assert out["reasoning"] == {"effort": "low"}
    assert "reasoning" not in out["messages"][0]


def test_strip_unsupported_injects_routing_as_provider(main_module):
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="m", routing={"sort": "throughput"})
    out = main_module._strip_unsupported({"messages": []}, ep)
    assert out["provider"] == {"sort": "throughput"}


def test_strip_unsupported_no_provider_without_routing(main_module):
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="m")
    out = main_module._strip_unsupported({"messages": []}, ep)
    assert "provider" not in out


def test_usd_per_token_precision(main_module):
    assert main_module._usd_per_token(1.0) == "0.000001"
    assert main_module._usd_per_token(0.15) == "0.00000015"
    assert main_module._usd_per_token(2.50) == "0.0000025"
    # Sub-$0.0001/M must not truncate to "0", high precision must survive
    assert main_module._usd_per_token(0.00005) == "0.00000000005"
    assert main_module._usd_per_token(12.345678) == "0.000012345678"
    assert main_module._usd_per_token(0.0) == "0"


def test_serialize_meta_reasoning_variants(main_module):
    from config import ModelMeta
    # efforts + non-default default + default_enabled: all config-driven
    meta = ModelMeta(supports_reasoning=True, reasoning_efforts=("low", "medium", "high"),
                     reasoning_default="medium", reasoning_default_enabled=False)
    r = main_module._serialize_meta(meta)["reasoning"]
    assert r == {"mandatory": False, "default_enabled": False,
                 "supported_efforts": ["low", "medium", "high"], "default_effort": "medium"}


def test_strip_unsupported_keeps_reasoning_when_flag_set(main_module):
    from config import Endpoint
    ep = Endpoint(base_url="x", api_key="k", model="m", keep_reasoning=True)
    out = main_module._strip_unsupported(
        {"messages": [{"role": "assistant", "content": "hi", "reasoning": "kept"}]},
        ep,
    )
    assert out["messages"][0]["reasoning"] == "kept"


# --- _should_race ---

def test_should_race_true_on_first_call(main_module):
    main_module._group_last_race_time.clear()
    main_module._group_race_request_count.clear()
    should, trigger = main_module._should_race("default", False)
    assert should is True
    assert trigger == "first request"


def test_should_race_false_when_within_cadence(main_module):
    import time
    main_module._group_last_race_time["default"] = time.monotonic()
    main_module._group_race_request_count["default"] = 0
    should, _ = main_module._should_race("default", False)
    assert should is False


def test_should_race_true_after_request_threshold(main_module):
    import time
    main_module._group_last_race_time["default"] = time.monotonic()
    main_module._group_race_request_count["default"] = 9999
    should, _ = main_module._should_race("default", False)
    assert should is True


# --- _finish_race ---

def test_finish_race_orders_by_completion_time(main_module):
    """Fastest endpoints come first in preferred_providers; non-finishers go last."""
    main_module._group_provider_groups["default"] = {
        ("ma", "https://a.test"): [0],
        ("mb", "https://b.test"): [1],
        ("mc", "https://c.test"): [2],
    }
    times = {
        ("mb", "https://b.test"): 0.05,
        ("ma", "https://a.test"): 0.20,
        # c didn't finish
    }
    main_module._finish_race(times, "default")
    order = main_module._group_preferred_providers["default"]
    assert order[0] == ("mb", "https://b.test")
    assert order[1] == ("ma", "https://a.test")
    assert order[2] == ("mc", "https://c.test")  # unmeasured falls to the back


def test_should_race_true_after_time_threshold(main_module):
    import time

    import config
    main_module._group_last_race_time["default"] = time.monotonic() - (config.SETTINGS.race_interval_secs + 1)
    main_module._group_race_request_count["default"] = 0
    should, _ = main_module._should_race("default", False)
    assert should is True


# --- _session_key ---


def test_session_key_stable_within_session(main_module):
    body = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "opening turn"}]}
    key = main_module._session_key(body)
    assert key.startswith("m:")
    assert key == main_module._session_key(dict(reversed(list(body.items()))))


def test_session_key_distinguishes_sessions_sharing_a_system_prompt(main_module):
    """Concurrent sessions of one client share the system prompt; the opening
    user turn must be what tells them apart."""
    a = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "session A"}]}
    b = {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "session B"}]}
    assert main_module._session_key(a) != main_module._session_key(b)


def test_session_key_combines_user_with_messages(main_module):
    """'user' is an end-user id, not a conversation id: same user, two
    conversations -> two keys; two users, same transcript -> two keys."""
    msgs = [{"role": "user", "content": "hi"}]
    assert main_module._session_key({"user": "alice", "messages": msgs}) != main_module._session_key({"user": "alice", "messages": [{"role": "user", "content": "other"}]})
    assert main_module._session_key({"user": "alice", "messages": msgs}) != main_module._session_key({"user": "bob", "messages": msgs})


def test_session_key_empty_without_usable_input(main_module):
    assert main_module._session_key({}) == ""
    assert main_module._session_key({"messages": []}) == ""
    assert main_module._session_key({"user": "alice"}) != ""  # user alone still pins


# --- session key robustness ---


def test_session_key_survives_lone_surrogates(main_module):
    """Malformed client JSON can carry unpaired surrogates; strict utf-8
    encoding would crash every request for that session."""
    body = {"messages": [{"role": "user", "content": "bad \ud800 escape"}]}
    k1 = main_module._session_key(body)
    assert k1.startswith("m:")
    assert k1 == main_module._session_key(body)
    assert main_module._session_key({"user": "\udfff", "messages": []}) != ""


def test_session_key_truncates_huge_strings_before_serializing(main_module):
    """Content differences beyond the hash cap must not change the key
    (that's what keeps serialization cost bounded)."""
    base = "x" * 100_000
    a = {"messages": [{"role": "user", "content": base + "-A"}]}
    b = {"messages": [{"role": "user", "content": base + "-B"}]}
    assert main_module._session_key(a) == main_module._session_key(b)
