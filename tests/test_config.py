import pytest

from config import ConfigError, parse_config


def make_minimal(providers: dict | None = None, groups: dict | None = None) -> dict:
    """Build a minimal valid config dict."""
    return {
        "providers": providers or {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": groups or {"default": [{"provider": "a"}]},
    }


def test_provider_parsed(make_config):
    cfg = make_config(make_minimal())
    assert cfg.ENDPOINTS[0].base_url == "https://a"
    assert cfg.ENDPOINTS[0].api_key == "k"
    assert cfg.ENDPOINTS[0].model == ""  # no model → passthrough


def test_provider_with_model(make_config):
    cfg = make_config({
        "providers": {"foo": {"base_url": "https://a.example", "api_key": "k"}},
        "groups": {"default": [{"provider": "foo", "model": "gpt-4o"}]},
    })
    assert cfg.ENDPOINTS[0].model == "gpt-4o"


def test_group_routes_endpoints_in_order(make_config):
    cfg = make_config({
        "providers": {
            "a": {"base_url": "https://a", "api_key": "k"},
            "b": {"base_url": "https://b", "api_key": "k"},
            "c": {"base_url": "https://c", "api_key": "k"},
        },
        "groups": {
            "default": [{"provider": "c"}, {"provider": "a"}],
        },
    })
    assert cfg.GROUPS["default"] == [0, 1]
    assert cfg.ENDPOINTS[0].base_url == "https://c"
    assert cfg.ENDPOINTS[1].base_url == "https://a"
    assert len(cfg.ENDPOINTS) == 2


def test_group_name_is_plain_string_no_normalization(make_config):
    """Group names match request models directly — no case/separator munging."""
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {
            "default": [{"provider": "a"}],
            "gpt-4.1": [{"provider": "a", "model": "gpt-4.1"}],
        },
    })
    assert "gpt-4.1" in cfg.GROUPS
    assert "gpt_4_1" not in cfg.GROUPS
    assert "GPT-4.1" not in cfg.GROUPS


def test_unknown_provider_in_group_is_config_error():
    with pytest.raises(ConfigError, match="unknown provider"):
        parse_config(make_minimal(groups={"default": [{"provider": "nope"}]}))


def test_missing_provider_key_is_config_error():
    with pytest.raises(ConfigError, match="missing 'provider'"):
        parse_config(make_minimal(groups={"default": [{"model": "x"}]}))


def test_missing_default_group_ok_without_implicit_creation(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"other": [{"provider": "a", "model": "m1"}]},
    })
    assert "default" not in cfg.GROUPS
    assert "other" in cfg.GROUPS


def test_env_var_interpolation(make_config, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-value")
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "${MY_KEY}"}},
        "groups": {"default": [{"provider": "a"}]},
    })
    assert cfg.ENDPOINTS[0].api_key == "secret-value"


def test_missing_env_var_warns_and_preserves_placeholder(monkeypatch, caplog):
    """Missing env vars warn (not fatal) and the literal ${VAR} stays in place."""
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    with caplog.at_level("WARNING", logger="stablellm.config"):
        endpoints, _, _ = parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "${DEFINITELY_UNSET_VAR}"}},
            "groups": {"default": [{"provider": "a"}]},
        })
    assert endpoints[0].api_key == "${DEFINITELY_UNSET_VAR}"
    assert any("DEFINITELY_UNSET_VAR" in r.message for r in caplog.records)


def test_settings_loaded_from_yaml(make_config):
    cfg = make_config({
        "settings": {"cooloff_seconds": 5, "race_interval_requests": 100, "log_level": "debug"},
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"default": [{"provider": "a"}]},
    })
    assert cfg.SETTINGS.cooloff_seconds == 5
    assert cfg.SETTINGS.race_interval_requests == 100
    assert cfg.SETTINGS.log_level == "DEBUG"


def test_unknown_settings_key_is_config_error():
    with pytest.raises(ConfigError):
        parse_config({
            "settings": {"bogus": 1},
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": [{"provider": "a"}]},
        })


def test_flags_on_group_entry(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"default": [{"provider": "a", "model": "m", "flags": ["keep_reasoning"]}]},
    })
    assert cfg.ENDPOINTS[0].keep_reasoning is True


def test_provider_default_model_used_when_group_entry_omits_model(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k", "model": "provider-default"}},
        "groups": {"default": [{"provider": "a"}]},
    })
    assert cfg.ENDPOINTS[0].model == "provider-default"


def test_group_entry_model_overrides_provider_default(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k", "model": "provider-default"}},
        "groups": {"default": [{"provider": "a", "model": "group-override"}]},
    })
    assert cfg.ENDPOINTS[0].model == "group-override"


def test_same_provider_in_multiple_groups_creates_separate_endpoints(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {
            "default": [{"provider": "a", "model": "m1"}],
            "other": [{"provider": "a", "model": "m2"}],
        },
    })
    assert len(cfg.ENDPOINTS) == 2
    assert cfg.ENDPOINTS[0].model == "m1"
    assert cfg.ENDPOINTS[1].model == "m2"
