import pytest


def test_endpoint_name_from_env_key(make_config):
    cfg = make_config({"ENDPOINT_FOO": "https://a.example|key1|gpt-4o"})
    assert cfg.ENDPOINT_NAMES == {"foo": 0}
    assert cfg.ENDPOINTS[0].base_url == "https://a.example"
    assert cfg.ENDPOINTS[0].model == "gpt-4o"


def test_group_resolves_endpoint_names_in_order(make_config):
    cfg = make_config({
        "ENDPOINT_A": "https://a|k|",
        "ENDPOINT_B": "https://b|k|",
        "ENDPOINT_C": "https://c|k|",
        "GROUP_CHEAP": "c,a",
    })
    assert cfg.GROUPS["cheap"] == [cfg.ENDPOINT_NAMES["c"], cfg.ENDPOINT_NAMES["a"]]


def test_unknown_endpoint_in_group_is_fatal(make_config):
    with pytest.raises(SystemExit):
        make_config({
            "ENDPOINT_A": "https://a|k|",
            "GROUP_X": "a,nope",
        })


def test_implicit_default_group_includes_all_endpoints_in_order(make_config):
    cfg = make_config({
        "ENDPOINT_A": "https://a|k|",
        "ENDPOINT_B": "https://b|k|",
    })
    assert cfg.GROUPS["default"] == [0, 1]


def test_explicit_default_group_overrides_order(make_config):
    cfg = make_config({
        "ENDPOINT_A": "https://a|k|",
        "ENDPOINT_B": "https://b|k|",
        "GROUP_DEFAULT": "b,a",
    })
    assert cfg.GROUPS["default"] == [cfg.ENDPOINT_NAMES["b"], cfg.ENDPOINT_NAMES["a"]]
