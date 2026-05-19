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
    assert main_module._should_race("default") is True


def test_should_race_false_when_within_cadence(main_module):
    import time
    main_module._group_last_race_time["default"] = time.monotonic()
    main_module._group_race_request_count["default"] = 0
    assert main_module._should_race("default") is False


def test_should_race_true_after_request_threshold(main_module):
    import time
    main_module._group_last_race_time["default"] = time.monotonic()
    main_module._group_race_request_count["default"] = 9999
    assert main_module._should_race("default") is True


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
    assert main_module._should_race("default") is True
