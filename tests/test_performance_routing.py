import asyncio
import json
import sys

import httpx
import pytest
from conftest import fresh_config

from performance_routing import (
    PerformanceRouting,
    derive_constraints,
    fast_tags,
    price_cap,
    quantization_floor,
)


def _endpoint(tag, latency, throughput, **extra):
    return {
        "tag": tag,
        "latency_last_30m": {"p50": latency},
        "throughput_last_30m": {"p50": throughput},
        **extra,
    }


def _row(tag, prompt, completion, quantization):
    return {
        "tag": tag,
        "pricing": {"prompt": prompt, "completion": completion},
        "quantization": quantization,
    }


def test_fast_tags_use_projected_completion_time():
    tags = fast_tags([
        _endpoint("fast", 500, 100),
        _endpoint("near", 100, 90),
        _endpoint("slow", 100, 70),
        _endpoint("unknown", None, None),
    ], PerformanceRouting(target_tokens=10_000, speed_tolerance=0.15))
    assert tags == ["fast", "near"]


def test_price_cap_converts_per_token_to_per_million():
    assert price_cap(5e-06, 0.15) == 5.75


def test_price_medians_exclude_nonpositive_rows():
    rows = [
        _row("a", "1e-06", "3e-06", "fp8"),
        _row("free", "0", "5e-06", "fp8"),  # :free variant drags nothing: it earns no vote
        _row("b", "3e-06", "3e-06", "fp8"),
    ]
    provider: dict = {}
    derive_constraints(rows, PerformanceRouting(), provider)
    assert provider["max_price"] == {"prompt": 2.3, "completion": 3.45}


def test_free_rows_pass_the_price_cap():
    rows = [
        _row("free", "0", "0", "fp8"),
        _row("a", "1e-06", "3e-06", "fp8"),
        _row("b", "3e-06", "3e-06", "fp8"),
    ]
    eligible, _floor = derive_constraints(rows, PerformanceRouting(), {})
    # free passes the cap; b exceeds the median-derived prompt cap
    assert [row["tag"] for row in eligible] == ["free", "a"]


def test_quantization_floor_is_median_width_rounded_up():
    fp4 = _row("a", "1e-06", "1e-06", "fp4")
    fp8 = _row("b", "1e-06", "1e-06", "fp8")
    fp16 = _row("c", "1e-06", "1e-06", "bf16")
    assert quantization_floor([fp4, fp8]) == 8  # median 6 rounds up to the next tier
    assert quantization_floor([fp4, fp4, fp8]) == 4  # median 4 is already a tier
    assert quantization_floor([fp4, fp8, fp16]) == 8  # median 8
    assert quantization_floor([fp8, fp16]) == 16  # median 12 rounds up
    assert quantization_floor([fp4, fp4, fp4, fp8, fp8, fp16]) == 8  # mode is 4, median 6
    assert quantization_floor([{"quantization": "unknown"}]) is None


def test_derived_floor_excluding_nothing_is_not_emitted():
    rows = [_row("a", "1e-06", "1e-06", "fp8"), _row("b", "1e-06", "1e-06", "fp8")]
    provider: dict = {}
    eligible, _floor = derive_constraints(rows, PerformanceRouting(), provider)
    assert provider.get("quantizations") is None
    assert {row["quantization"] for row in eligible} == {"fp8"}


def test_quantization_floor_off_disables_derivation():
    # The derived 8-bit floor would drop the fp4 row; off keeps it.
    rows = [
        _row("a", "1e-06", "1e-06", "fp8"),
        _row("b", "1e-06", "1e-06", "fp8"),
        _row("c", "1e-06", "1e-06", "fp4"),
    ]
    provider: dict = {}
    eligible, floor = derive_constraints(rows, PerformanceRouting(quantization_floor=None), provider)
    assert provider.get("quantizations") is None
    assert floor is None
    assert [row["tag"] for row in eligible] == ["a", "b", "c"]


def test_static_quantizations_report_their_own_floor():
    # The log line's quant= field is the effective floor bit width: the minimum
    # of the author's own list, not a None that prints as "Nonebit".
    rows = [_row("a", "1e-06", "1e-06", "fp8"), _row("b", "1e-06", "1e-06", "bf16")]
    provider = {"quantizations": ["fp8", "bf16"]}
    _eligible, floor = derive_constraints(rows, PerformanceRouting(), provider)
    assert provider["quantizations"] == ["fp8", "bf16"]
    assert floor == 8


def test_static_quantizations_short_form_admits_long_form_rows():
    # `fp4` is the selector's short form for the whole family, so a row
    # reporting `mxfp4` has to survive it. int4 is a different family.
    rows = [
        _row("long", "1e-06", "1e-06", "mxfp4"),
        _row("int", "1e-06", "1e-06", "int4"),
        _row("other", "1e-06", "1e-06", "fp8"),
    ]
    eligible, floor = derive_constraints(rows, PerformanceRouting(), {"quantizations": ["fp4"]})
    assert [row["tag"] for row in eligible] == ["long"]
    assert floor == 4


def test_static_quantizations_long_form_is_not_a_short_form():
    # The reverse does not hold: the selector reads `mxfp4` as itself, so a
    # plain-fp4 row must not satisfy it (verified live against OpenRouter).
    rows = [_row("short", "1e-06", "1e-06", "fp4"), _row("exact", "1e-06", "1e-06", "mxfp4")]
    eligible, _floor = derive_constraints(rows, PerformanceRouting(), {"quantizations": ["mxfp4"]})
    assert [row["tag"] for row in eligible] == ["exact"]


def test_static_max_price_ignores_fields_the_catalog_cannot_express():
    # A static cap keeps keys for fields rows do not price per token; they are
    # sent to OpenRouter but cannot take part in local filtering.
    rows = [_row("a", "2e-06", "4e-06", "fp8")]
    provider = {"max_price": {"prompt": 3.0, "completion": 5.0, "image": 0.03}}
    eligible, _floor = derive_constraints(rows, PerformanceRouting(price_cap_tolerance=None), provider)
    assert provider["max_price"] == {"prompt": 3.0, "completion": 5.0, "image": 0.03}
    assert [row["tag"] for row in eligible] == ["a"]


def test_partial_static_max_price_filters_on_the_fields_it_names():
    # A cap naming only one price field is a real config, not a crash.
    rows = [_row("cheap", "1e-06", "9e-06", "fp8"), _row("dear", "9e-06", "1e-06", "fp8")]
    for cap in ({"prompt": 2.0}, {"completion": 2.0}, {"image": 0.03}):
        eligible, _floor = derive_constraints(
            rows, PerformanceRouting(price_cap_tolerance=None), {"max_price": cap},
        )
        expected = ["cheap", "dear"] if "image" in cap else (["cheap"] if "prompt" in cap else ["dear"])
        assert [row["tag"] for row in eligible] == expected, cap


def test_static_max_price_still_excludes_rows_over_the_cap():
    rows = [_row("cheap", "1e-06", "1e-06", "fp8"), _row("dear", "9e-06", "1e-06", "fp8")]
    eligible, _floor = derive_constraints(
        rows, PerformanceRouting(price_cap_tolerance=None), {"max_price": {"prompt": 2.0}},
    )
    assert [row["tag"] for row in eligible] == ["cheap"]


def test_ranking_runs_inside_the_constraints():
    # The 4-bit endpoint is fastest overall; the floor must remove it before
    # ranking so it cannot anchor the speed window.
    rows = [
        {**_row("four", "1e-06", "1e-06", "fp4"), "latency_last_30m": {"p50": 100}, "throughput_last_30m": {"p50": 100}},
        {**_row("eight", "1e-06", "1e-06", "fp8"), "latency_last_30m": {"p50": 100}, "throughput_last_30m": {"p50": 100}},
        {**_row("out", "1e-06", "1e-06", "fp8"), "latency_last_30m": {"p50": 100}, "throughput_last_30m": {"p50": 50}},
    ]
    provider: dict = {}
    eligible, _floor = derive_constraints(rows, PerformanceRouting(), provider)
    assert provider["quantizations"] == ["int8", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown"]
    assert fast_tags(eligible, PerformanceRouting()) == ["eight"]


def test_static_constraints_govern_filtering_and_are_emitted_verbatim():
    rows = [_row("cheap", "1e-06", "2e-06", "fp8"), _row("dear", "5e-06", "9e-06", "fp8")]
    provider = {"max_price": {"prompt": 2}, "quantizations": ["fp8", "unknown"]}
    eligible, _floor = derive_constraints(rows, PerformanceRouting(), provider)
    assert provider["max_price"] == {"prompt": 2}
    assert provider["quantizations"] == ["fp8", "unknown"]
    assert [row["tag"] for row in eligible] == ["cheap"]


def test_static_constraints_govern_filtering_with_derived_rules_off():
    # The derived rule being off must not stop a static cap from filtering:
    # emitting it while sending the rows it forbids gets the request 404'd.
    rows = [_row("cheap", "1e-06", "2e-06", "fp8"), _row("dear", "9e-05", "9e-05", "fp8")]
    provider = {"max_price": {"prompt": 2.0, "completion": 2.0}}
    eligible, _floor = derive_constraints(rows, PerformanceRouting(price_cap_tolerance=None), provider)
    assert provider["max_price"] == {"prompt": 2.0, "completion": 2.0}
    assert [row["tag"] for row in eligible] == ["cheap"]


def test_derivation_is_none_when_every_row_is_excluded():
    # Opposite price shapes: the cheap-prompt row is not the cheap-completion row.
    rows = [_row("a", "1e-06", "9e-06", "fp8"), _row("b", "5e-06", "1e-06", "fp8")]
    assert derive_constraints(rows, PerformanceRouting(), {}) is None


def test_unmeasured_eligible_rows_rank_to_no_tags():
    provider: dict = {}
    eligible, _floor = derive_constraints([_row("u", "1e-06", "2e-06", "fp8")], PerformanceRouting(), provider)
    assert provider.get("quantizations") is None
    assert eligible
    assert fast_tags(eligible, PerformanceRouting()) == []


def _test_config(endpoints):
    return {
        "providers": {"openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key": "k"}},
        "groups": {"default": {"endpoints": endpoints}},
    }


_SINGLE_ENDPOINT_CONFIG = _test_config([{
    "provider": "openrouter",
    "model": "author/model",
    "performance_routing": {},
    "routing": {
        "sort": "throughput",
        "order": ["old"],
        "preferred_max_latency": {"p50": 1},
    },
}])


@pytest.fixture
def performance_app(monkeypatch, tmp_path):
    def build(handler, config=None):
        fresh_config(monkeypatch, tmp_path, config or _SINGLE_ENDPOINT_CONFIG)
        sys.modules.pop("main", None)
        import main

        calls = []

        async def wrapped(request):
            body = json.loads(request.content) if request.content else None
            calls.append((request.method, request.url.path, body))
            response = handler(request, body)
            if asyncio.iscoroutine(response):
                response = await response
            return response

        main.http_client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
        main._build_provider_groups()
        main._reset_runtime_state()
        return main.app, calls

    return build


async def _post(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": "hello"}],
        })


@pytest.mark.asyncio
async def test_static_only_disjoint_from_ranked_tags_is_dropped(performance_app):
    # The constraint is hard, the speed optimization is not: a static `only`
    # naming no rankable tag sends no `only` at all rather than a 404.
    rows = [_priced("fast", 400, 100), _priced("slow", 100, 60)]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler, _test_config([{
        "provider": "openrouter",
        "model": "author/model",
        "performance_routing": {},
        "routing": {"only": ["nomatch"]},
    }]))
    assert (await _post(app)).status_code == 200
    provider = calls[1][2]["provider"]
    assert "only" not in provider
    # The constraints reached OpenRouter without the dropped allowlist.
    assert provider["max_price"] == {"prompt": 2.3, "completion": 4.6}
    assert provider["allow_fallbacks"] is True


@pytest.mark.asyncio
async def test_quantization_floor_spellings_route_differently(performance_app):
    # off / auto / explicit tier are three different routing outcomes on the
    # same catalog: the derived 8-bit floor is what drops the 4-bit endpoint.
    rows_side_by_side = [_priced("four", 100, 100, "fp4"), _priced("eight1", 100, 100, "fp8"), _priced("eight2", 100, 100, "fp8")]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows_side_by_side}})
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 1}})

    async def _only_for(floor):
        app, calls = performance_app(handler, _test_config([{
            "provider": "openrouter",
            "model": "author/model",
            "performance_routing": {"quantization_floor": floor},
        }]))
        assert (await _post(app)).status_code == 200
        return calls[1][2]["provider"]

    off = await _only_for("none")
    assert off["only"] == ["four", "eight1", "eight2"]
    assert "quantizations" not in off
    # auto derives the mode from the catalog (the fp8 mode over one fp4 row).
    auto = await _only_for("auto")
    assert auto["only"] == ["eight1", "eight2"]
    assert auto["quantizations"] == ["int8", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown"]
    # An explicit tier states the same requirement by name.
    tier = await _only_for(8)
    assert tier["only"] == ["eight1", "eight2"]
    assert tier["quantizations"] == ["int8", "fp8", "mxfp8", "fp16", "bf16", "fp32", "unknown"]


def _priced(tag, latency, throughput, quantization="fp8"):
    return {
        **_endpoint(tag, latency, throughput),
        "pricing": {"prompt": "2e-06", "completion": "4e-06"},
        "quantization": quantization,
    }


def _catalog(rows):
    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 1}})

    return handler


@pytest.mark.asyncio
async def test_static_max_price_filters_when_derivation_is_off(performance_app):
    # The derived rule being off must not stop a static cap from filtering:
    # emitting it while sending the rows it forbids gets the request 404'd.
    rows = [
        {**_priced("dear", 100, 100), "pricing": {"prompt": "1e-4", "completion": "1e-4"}},
        {**_priced("cheap", 100, 100), "pricing": {"prompt": "2e-6", "completion": "2e-6"}},
    ]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={"provider": "cheap", "choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler, _test_config([{
        "provider": "openrouter",
        "model": "author/model",
        "performance_routing": {"price_cap_tolerance": "none"},
        "routing": {"max_price": {"prompt": 2.0, "completion": 2.0}},
    }]))
    assert (await _post(app)).status_code == 200
    provider = calls[1][2]["provider"]
    assert provider["only"] == ["cheap"]
    assert provider["max_price"] == {"prompt": 2.0, "completion": 2.0}


@pytest.mark.asyncio
async def test_static_quantizations_log_reports_effective_floor(performance_app, capsys):
    # The static list states its own floor; the log line must report that
    # tier, never a bogus None-bit width.
    rows = [_priced("eight", 100, 100, "fp8"), _priced("wide", 100, 100, "bf16")]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={"provider": "eight", "choices": [], "usage": {"completion_tokens": 1}})

    app, _calls = performance_app(handler, _test_config([{
        "provider": "openrouter",
        "model": "author/model",
        "performance_routing": {},
        "routing": {"quantizations": ["fp8", "bf16"]},
    }]))
    assert (await _post(app)).status_code == 200
    logged = capsys.readouterr().err
    assert "quant=8bit" in logged
    assert "Nonebit" not in logged


@pytest.mark.asyncio
async def test_stale_only_rejection_retries_without_marking_down(performance_app):
    # A stale generated allowlist (catalog cached, provider stopped serving the
    # model) is our own selection being rejected, not an endpoint failure: the
    # request retries with the raw routing and nothing is cooled off.
    post_calls = 0

    def handler(request, _body):
        nonlocal post_calls
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": [_priced("stale", 400, 100)]}})
        post_calls += 1
        if post_calls == 1:
            return httpx.Response(404, json={"error": {"message": "No allowed providers are available for the selected model. Providers serving author/model: fast; but your request's provider.only preference permits only: stale.", "code": 404}})
        return httpx.Response(200, json={"provider": "stale", "choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    assert calls[1][2]["provider"]["only"] == ["stale"]
    assert "only" not in calls[2][2]["provider"]
    main = sys.modules["main"]
    assert main._stats["successes"][0] == 1
    assert main._stats["failures"][0] == 0
    assert main._cooloff_until.get(0, 0) == 0


@pytest.mark.asyncio
async def test_performance_routing_emits_derived_constraints_and_caches_catalog(performance_app):
    rows = [_priced("fast", 400, 100), _priced("near", 100, 90), _priced("slow", 100, 60)]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={"provider": "fast", "choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    assert (await _post(app)).status_code == 200

    assert [method for method, _, _ in calls] == ["GET", "POST", "POST"]
    provider = calls[1][2]["provider"]
    assert provider["only"] == ["fast", "near"]
    assert provider["allow_fallbacks"] is True
    assert provider["max_price"] == {"prompt": 2.3, "completion": 4.6}
    assert "quantizations" not in provider  # fp8-only catalog: the floor would be a no-op
    assert "sort" not in provider
    assert "order" not in provider
    assert provider["preferred_max_latency"] == {"p50": 1}


@pytest.mark.asyncio
async def test_pinned_via_stays_in_generated_only(performance_app):
    rows = [_priced("fast", 400, 100), _priced("slow", 100, 60)]

    def handler(request, _body):
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={"provider": "cached-provider", "choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    assert (await _post(app)).status_code == 200
    assert calls[2][2]["provider"]["only"] == ["fast", "cached-provider"]


@pytest.mark.asyncio
async def test_no_compatible_generated_only_retries_raw_routing(performance_app):
    post_calls = 0

    def handler(request, _body):
        nonlocal post_calls
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": [_priced("fast", 400, 100)]}})
        post_calls += 1
        if post_calls == 1:
            return httpx.Response(400, json={"error": {"message": "No providers available"}})
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    assert calls[1][2]["provider"]["only"] == ["fast"]
    assert calls[1][2]["provider"]["allow_fallbacks"] is True
    assert "only" not in calls[2][2]["provider"]
    assert calls[2][2]["provider"]["sort"] == "throughput"


@pytest.mark.asyncio
async def test_empty_derivation_skips_to_next_endpoint(performance_app):
    def handler(request, _body):
        if request.method == "GET":
            rows = (
                [_row("a", "1e-06", "9e-06", "fp8"), _row("b", "5e-06", "1e-06", "fp8")]
                if "author" in request.url.path
                else [_priced("ok", 100, 100)]
            )
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 1}})

    config = _test_config([
        {"provider": "openrouter", "model": "author/model", "performance_routing": {}},
        {"provider": "openrouter", "model": "other/model", "performance_routing": {}},
    ])
    app, calls = performance_app(handler, config)
    resp = await _post(app)
    assert resp.status_code == 200
    # The skipped endpoint sent nothing; the second one served the request.
    posts = [body for method, _, body in calls if method == "POST"]
    assert len(posts) == 1
    assert posts[0]["model"] == "other/model"
    assert posts[0]["provider"]["only"] == ["ok"]


@pytest.mark.asyncio
async def test_all_skip_group_names_empty_derivation_in_502(performance_app):
    def handler(request, _body):
        if request.method == "GET":
            rows = [_row("a", "1e-06", "9e-06", "fp8"), _row("b", "5e-06", "1e-06", "fp8")]
            return httpx.Response(200, json={"data": {"endpoints": rows}})
        raise AssertionError("no endpoint should be posted to")

    app, calls = performance_app(handler)
    resp = await _post(app)
    assert resp.status_code == 502
    assert "caps exclude every catalog row" in resp.json()["error"]
    assert [method for method, _, _ in calls] == ["GET"]


@pytest.mark.asyncio
async def test_unrankable_eligible_still_emits_constraints_and_retries(performance_app):
    """Eligible rows with no measurements: constraints are sent without `only`,
    and a cap rejection still triggers the single no-constraint retry."""
    post_calls = 0

    def handler(request, _body):
        nonlocal post_calls
        if request.method == "GET":
            return httpx.Response(200, json={"data": {"endpoints": [_row("unmeasured", "1e-06", "2e-06", "fp8")]}})
        post_calls += 1
        if post_calls == 1:
            return httpx.Response(400, json={"error": {"message": "no endpoints found matching max_price"}})
        return httpx.Response(200, json={"choices": [], "usage": {"completion_tokens": 1}})

    app, calls = performance_app(handler)
    assert (await _post(app)).status_code == 200
    first = calls[1][2]["provider"]
    assert first["max_price"] == {"prompt": 1.15, "completion": 2.3}
    assert "only" not in first
    assert "only" not in calls[2][2]["provider"]
