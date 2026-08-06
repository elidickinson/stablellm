import pytest

from config import MODE_RACE, MODE_SEQ, ConfigError, parse_config


def _group(endpoints, mode=None) -> dict:
    spec = {"endpoints": endpoints}
    if mode is not None:
        spec["mode"] = mode
    return spec


def make_minimal(providers: dict | None = None, groups: dict | None = None) -> dict:
    """Build a minimal valid config dict."""
    return {
        "providers": providers or {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": groups or {"default": _group([{"provider": "a"}])},
    }


def test_provider_parsed(make_config):
    cfg = make_config(make_minimal())
    assert cfg.ENDPOINTS[0].base_url == "https://a"
    assert cfg.ENDPOINTS[0].api_key == "k"
    assert cfg.ENDPOINTS[0].model == ""  # no model → passthrough


def test_provider_with_model(make_config):
    cfg = make_config({
        "providers": {"foo": {"base_url": "https://a.example", "api_key": "k"}},
        "groups": {"default": _group([{"provider": "foo", "model": "gpt-4o"}])},
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
            "default": _group([{"provider": "c"}, {"provider": "a"}]),
        },
    })
    assert cfg.GROUPS["default"].endpoints == [0, 1]
    assert cfg.GROUPS["default"].mode == MODE_SEQ
    assert cfg.ENDPOINTS[0].base_url == "https://c"
    assert cfg.ENDPOINTS[1].base_url == "https://a"
    assert len(cfg.ENDPOINTS) == 2


def test_group_names_stored_lowercased(make_config):
    """Group names are normalized to lowercase at parse time."""
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {
            "default": _group([{"provider": "a"}]),
            "GPT-4.1": _group([{"provider": "a", "model": "gpt-4.1"}]),
        },
    })
    assert "gpt-4.1" in cfg.GROUPS
    assert "GPT-4.1" not in cfg.GROUPS  # stored lowercased


def test_duplicate_group_names_case_insensitive_is_error():
    with pytest.raises(ConfigError, match="duplicate group name"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {
                "cheap": _group([{"provider": "a"}]),
                "Cheap": _group([{"provider": "a"}]),
            },
        })


def test_unknown_provider_in_group_is_config_error():
    with pytest.raises(ConfigError, match="unknown provider"):
        parse_config(make_minimal(groups={"default": _group([{"provider": "nope"}])}))


def test_missing_provider_key_is_config_error():
    with pytest.raises(ConfigError, match="missing 'provider'"):
        parse_config(make_minimal(groups={"default": _group([{"model": "x"}])}))


def test_group_name_default_has_no_special_meaning(make_config):
    """'default' is just a name; not auto-created and not required."""
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"other": _group([{"provider": "a", "model": "m1"}])},
    })
    assert "default" not in cfg.GROUPS
    assert "other" in cfg.GROUPS


def test_env_var_interpolation(make_config, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-value")
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "${MY_KEY}"}},
        "groups": {"default": _group([{"provider": "a"}])},
    })
    assert cfg.ENDPOINTS[0].api_key == "secret-value"


def test_missing_env_var_warns_and_preserves_placeholder(monkeypatch, caplog):
    """Missing env vars warn (not fatal) and the literal ${VAR} stays in place."""
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    with caplog.at_level("WARNING", logger="stablellm.config"):
        endpoints, _, _ = parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "${DEFINITELY_UNSET_VAR}"}},
            "groups": {"default": _group([{"provider": "a"}])},
        })
    assert endpoints[0].api_key == "${DEFINITELY_UNSET_VAR}"
    assert any("DEFINITELY_UNSET_VAR" in r.message for r in caplog.records)


def test_settings_loaded_from_yaml(make_config):
    cfg = make_config({
        "settings": {"cooloff_seconds": 5, "race_interval_requests": 100, "log_level": "debug"},
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"default": _group([{"provider": "a"}])},
    })
    assert cfg.SETTINGS.cooloff_seconds == 5
    assert cfg.SETTINGS.race_interval_requests == 100
    assert cfg.SETTINGS.log_level == "DEBUG"


def test_unknown_settings_key_is_config_error():
    with pytest.raises(ConfigError):
        parse_config({
            "settings": {"bogus": 1},
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": _group([{"provider": "a"}])},
        })


def test_flags_on_group_entry(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"default": _group([{"provider": "a", "model": "m", "flags": ["keep_reasoning"]}])},
    })
    assert cfg.ENDPOINTS[0].keep_reasoning is True


def test_provider_default_model_used_when_group_entry_omits_model(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k", "model": "provider-default"}},
        "groups": {"default": _group([{"provider": "a"}])},
    })
    assert cfg.ENDPOINTS[0].model == "provider-default"


def test_group_entry_model_overrides_provider_default(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k", "model": "provider-default"}},
        "groups": {"default": _group([{"provider": "a", "model": "group-override"}])},
    })
    assert cfg.ENDPOINTS[0].model == "group-override"


def test_same_provider_in_multiple_groups_creates_separate_endpoints(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {
            "default": _group([{"provider": "a", "model": "m1"}]),
            "other": _group([{"provider": "a", "model": "m2"}]),
        },
    })
    assert len(cfg.ENDPOINTS) == 2
    assert cfg.ENDPOINTS[0].model == "m1"
    assert cfg.ENDPOINTS[1].model == "m2"


# --- Mode field ---

def test_group_mode_defaults_to_seq(make_config):
    cfg = make_config(make_minimal())
    assert cfg.GROUPS["default"].mode == MODE_SEQ


def test_group_mode_race_parsed(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"default": _group([{"provider": "a"}], mode="race")},
    })
    assert cfg.GROUPS["default"].mode == MODE_RACE


def test_unknown_mode_is_config_error():
    with pytest.raises(ConfigError, match="unknown mode"):
        parse_config(make_minimal(groups={"default": _group([{"provider": "a"}], mode="turbo")}))


def test_group_must_be_a_mapping_not_a_list():
    """The dict shape is mandatory — bare list form is rejected."""
    with pytest.raises(ConfigError, match="must be a mapping"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": [{"provider": "a"}]},
        })


def test_group_missing_endpoints_is_config_error():
    with pytest.raises(ConfigError, match="'endpoints' must be a non-empty list"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"mode": "race"}},
        })


def test_empty_endpoints_list_is_config_error():
    with pytest.raises(ConfigError, match="'endpoints' must be a non-empty list"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": _group([])},
        })


# --- Group meta ---

def test_group_meta_parsed(make_config):
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {
            "default": {
                "endpoints": [{"provider": "a"}],
                "meta": {
                    "name": "GLM 5.2",
                    "description": "flagship",
                    "context": 200000,
                    "max_output": 128000,
                    "modalities": ["text", "image"],
                    "input_cost": 0.6,
                    "output_cost": 2.2,
                    "cache_read_cost": 0.06,
                    "cache_write_cost": 2.2,
                    "supports_reasoning": True,
                    "reasoning_efforts": ["low", "high"],
                },
            },
        },
    })
    m = cfg.GROUPS["default"].meta
    assert m is not None
    assert m.name == "GLM 5.2"
    assert m.context == 200000
    assert m.input_cost == 0.6
    assert m.modalities == ("text", "image")
    assert m.supports_reasoning is True
    assert m.reasoning_efforts == ("low", "high")


def test_group_meta_default_is_none(make_config):
    cfg = make_config(make_minimal())
    assert cfg.GROUPS["default"].meta is None


def test_unknown_meta_key_is_config_error():
    with pytest.raises(ConfigError, match="unknown meta keys"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {"bogus": 1}}},
        })


def test_meta_context_must_be_positive_int_is_config_error():
    with pytest.raises(ConfigError, match="context"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {"context": -5}}},
        })


def test_meta_cost_must_be_nonneg_is_config_error():
    with pytest.raises(ConfigError, match="input_cost"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {"input_cost": -1}}},
        })


def test_meta_supports_reasoning_must_be_bool_is_config_error():
    with pytest.raises(ConfigError, match="supports_reasoning"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {"supports_reasoning": "false"}}},
        })


def test_meta_name_must_be_string_is_config_error():
    with pytest.raises(ConfigError, match="name"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {"name": 123}}},
        })


def test_meta_reasoning_efforts_require_reasoning_is_config_error():
    with pytest.raises(ConfigError, match="supports_reasoning"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {"reasoning_efforts": ["low"]}}},
        })


def test_meta_reasoning_fields_require_supports_reasoning_is_config_error():
    for extra in ({"reasoning_mandatory": True}, {"reasoning_default": "high"},
                  {"reasoning_default_enabled": True}):
        with pytest.raises(ConfigError, match="supports_reasoning"):
            parse_config({
                "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
                "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": extra}},
            })


def test_meta_mandatory_without_efforts_is_valid(make_config):
    # Mandatory reasoning with no effort list exists in OpenRouter's catalog
    cfg = make_config({
        "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
        "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {
            "supports_reasoning": True, "reasoning_mandatory": True,
        }}},
    })
    m = cfg.GROUPS["default"].meta
    assert m.reasoning_mandatory is True
    assert m.reasoning_efforts == ()


def test_meta_reasoning_default_must_be_in_efforts_is_config_error():
    with pytest.raises(ConfigError, match="reasoning_default"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {
                "supports_reasoning": True, "reasoning_efforts": ["low", "high"],
                "reasoning_default": "medium",
            }}},
        })


def test_meta_default_enabled_false_requires_efforts_is_config_error():
    with pytest.raises(ConfigError, match="reasoning_default_enabled"):
        parse_config({
            "providers": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"default": {"endpoints": [{"provider": "a"}], "meta": {
                "supports_reasoning": True, "reasoning_default_enabled": False,
            }}},
        })
