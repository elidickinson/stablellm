import pytest

from config import ConfigError, parse_config


def test_endpoints_parsed_with_model_field(make_config):
    cfg = make_config({
        "endpoints": {
            "foo": {"base_url": "https://a.example", "api_key": "k", "model": "gpt-4o"},
        },
    })
    assert cfg.ENDPOINT_NAMES == {"foo": 0}
    assert cfg.ENDPOINTS[0].base_url == "https://a.example"
    assert cfg.ENDPOINTS[0].model == "gpt-4o"


def test_group_resolves_endpoint_names_in_order(make_config):
    cfg = make_config({
        "endpoints": {
            "a": {"base_url": "https://a", "api_key": "k"},
            "b": {"base_url": "https://b", "api_key": "k"},
            "c": {"base_url": "https://c", "api_key": "k"},
        },
        "groups": {"cheap": ["c", "a"]},
    })
    assert cfg.GROUPS["cheap"] == [cfg.ENDPOINT_NAMES["c"], cfg.ENDPOINT_NAMES["a"]]


def test_unknown_endpoint_in_group_is_config_error():
    with pytest.raises(ConfigError):
        parse_config({
            "endpoints": {"a": {"base_url": "https://a", "api_key": "k"}},
            "groups": {"x": ["a", "nope"]},
        })


def test_implicit_default_group_includes_all_endpoints_in_order(make_config):
    cfg = make_config({
        "endpoints": {
            "a": {"base_url": "https://a", "api_key": "k"},
            "b": {"base_url": "https://b", "api_key": "k"},
        },
    })
    assert cfg.GROUPS["default"] == [0, 1]


def test_explicit_default_group_overrides_order(make_config):
    cfg = make_config({
        "endpoints": {
            "a": {"base_url": "https://a", "api_key": "k"},
            "b": {"base_url": "https://b", "api_key": "k"},
        },
        "groups": {"default": ["b", "a"]},
    })
    assert cfg.GROUPS["default"] == [cfg.ENDPOINT_NAMES["b"], cfg.ENDPOINT_NAMES["a"]]


def test_env_var_interpolation(make_config, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret-value")
    cfg = make_config({
        "endpoints": {"a": {"base_url": "https://a", "api_key": "${MY_KEY}"}},
    })
    assert cfg.ENDPOINTS[0].api_key == "secret-value"


def test_missing_env_var_is_config_error(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    with pytest.raises(ConfigError):
        parse_config({
            "endpoints": {"a": {"base_url": "https://a", "api_key": "${DEFINITELY_UNSET_VAR}"}},
        })


def test_settings_loaded_from_yaml(make_config):
    cfg = make_config({
        "settings": {"cooloff_seconds": 5, "race_interval_requests": 100, "log_level": "debug"},
        "endpoints": {"a": {"base_url": "https://a", "api_key": "k"}},
    })
    assert cfg.SETTINGS.cooloff_seconds == 5
    assert cfg.SETTINGS.race_interval_requests == 100
    assert cfg.SETTINGS.log_level == "DEBUG"


def test_unknown_settings_key_is_config_error():
    with pytest.raises(ConfigError):
        parse_config({
            "settings": {"bogus": 1},
            "endpoints": {"a": {"base_url": "https://a", "api_key": "k"}},
        })
