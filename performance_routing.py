"""Local OpenRouter provider selection from its public endpoint metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# Bit width of each quantization value OpenRouter's catalog reports. The
# quantizations vocabulary is the short forms; long forms fold into their short
# equivalent. 6-bit is a documented width but no catalog row reports it, so it
# is not an offerable floor tier.
_QUANT_BITS = {
    "int4": 4, "fp4": 4, "mxfp4": 4, "nvfp4": 4,
    "fp6": 6,
    "int8": 8, "fp8": 8, "mxfp8": 8,
    "fp16": 16, "bf16": 16,
    "fp32": 32,
}
_QUANT_TIERS = frozenset({4, 8, 16, 32})
_QUANT_TIER_LABELS = "auto (derived), 4, 8, 16, 32, or an explicit null for off"
_QUANT_FLOOR_AUTO: Literal["auto"] = "auto"
# The selector's short forms cover their long-form variants, so `fp4` admits a
# row reporting `mxfp4`. int4 is a separate family from fp4.
_QUANT_FAMILY = {"mxfp4": "fp4", "nvfp4": "fp4", "mxfp8": "fp8"}

# Catalog pricing fields the local cap can compare. `request` and `image` are
# passed through to OpenRouter, which enforces them.
_PRICE_FIELDS = frozenset({"prompt", "completion"})


@dataclass(frozen=True)
class PerformanceRouting:
    target_tokens: int = 10_000
    speed_tolerance: float = 0.15
    cache_ttl_seconds: float = 900.0
    price_cap_tolerance: float | None = 0.15  # None = no price constraint
    quantization_floor: int | Literal["auto"] | None = _QUANT_FLOOR_AUTO  # None = off, "auto" = derived from rows
    include_unknown_quantization: bool = True


def price_cap(row_price: float, tolerance: float) -> float:
    """OpenRouter max_price value for a per-token catalog price: the price
    converted to dollars per million tokens and padded by the tolerance."""
    return round(row_price * 1e6 * (1 + tolerance), 6)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _row_number(row: dict[str, Any], *path: str) -> float | None:
    """Numeric value at a nested path in a row, or None when it is absent or
    not a number."""
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_price(row: dict[str, Any], field: str) -> float | None:
    """Row's positive per-token price for the field, or None."""
    price = _row_number(row, "pricing", field)
    return price if price and price > 0 else None


def price_caps(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    """Unpadded per-token (prompt, completion) price medians over every
    positively-priced row, measured or not: the cap describes the market for
    the model. None when a field has no positive-priced row."""
    prompt = [p for row in rows if (p := _row_price(row, "prompt")) is not None]
    completion = [p for row in rows if (p := _row_price(row, "completion")) is not None]
    if not prompt or not completion:
        return None
    return _median(prompt), _median(completion)


def row_bits(row: dict[str, Any]) -> int | None:
    """Bit width of a row's reported quantization, or None when unrecognized."""
    quantization = row.get("quantization")
    return _QUANT_BITS.get(quantization) if isinstance(quantization, str) else None


def quant_family(value: str) -> str:
    """Fold a reported quantization into the short form the selector accepts."""
    return _QUANT_FAMILY.get(value, value)


def quant_allowed(reported: object, allowed: frozenset[str]) -> bool:
    """Whether a row's reported quantization satisfies the selector's list: an
    exact name, or the short form that covers it. A value of another type is not
    a name, so it matches nothing."""
    if not isinstance(reported, str):
        return False
    return reported in allowed or quant_family(reported) in allowed


def quantization_floor(rows: list[dict[str, Any]]) -> int | None:
    """Median observed bit width, rounded up to the next tier. None when no row
    reports a recognized width."""
    widths = [bits for row in rows if (bits := row_bits(row)) is not None]
    if not widths:
        return None
    median = _median(widths)
    return min(tier for tier in _QUANT_TIERS if tier >= median)


def quantization_values(floor: int, include_unknown: bool = True) -> list[str]:
    """Vocabulary entries at or above the floor, in OpenRouter's short form,
    plus `unknown` unless excluded."""
    values = [q for q, bits in _QUANT_BITS.items() if bits >= floor]
    values.sort(key=_QUANT_BITS.__getitem__)
    if include_unknown:
        values.append("unknown")
    return values


def row_passes(row: dict[str, Any], caps: dict[str, float] | None, quant_values: frozenset[str] | None) -> bool:
    """Whether a row meets the per-token price caps (every listed field must
    pass) and the quantization allowlist."""
    if caps is not None:
        for field, cap in caps.items():
            price = _row_number(row, "pricing", field)
            # A free row's zero price satisfies the cap.
            if price is None or not 0 <= price <= cap:
                return False
    return quant_values is None or quant_allowed(row.get("quantization"), quant_values)


def matches_provider_tag(tag: str, allowed: str) -> bool:
    """Whether an endpoint tag is covered by OpenRouter's provider selector."""
    return tag == allowed or tag.startswith(f"{allowed}/")


def fast_tags(endpoints: Iterable[dict[str, Any]], config: PerformanceRouting) -> list[str]:
    """Return endpoint tags within the configured projected-time tolerance.

    OpenRouter reports latency in milliseconds and throughput in tokens/second.
    Rows without usable measurements cannot be ranked.
    """
    scored: list[tuple[str, float]] = []
    for endpoint in endpoints:
        tag = endpoint.get("tag")
        latency = _row_number(endpoint, "latency_last_30m", "p50")
        throughput = _row_number(endpoint, "throughput_last_30m", "p50")
        if not isinstance(tag, str) or not tag:
            continue
        if latency is None or throughput is None:
            continue
        if latency < 0 or throughput <= 0:
            continue
        projected_seconds = latency / 1000 + config.target_tokens / throughput
        scored.append((tag, projected_seconds))

    if not scored:
        return []
    fastest = min(score for _, score in scored)
    limit = fastest * (1 + config.speed_tolerance)
    return [tag for tag, score in scored if score <= limit]


def derive_constraints(rows: list[dict[str, Any]], policy: PerformanceRouting, provider: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None] | None:
    """Derive the price cap and quantization floor from catalog rows, honoring
    author-set static values, and filter the rows to the eligible population.

    The emitted constraints are written into `provider` in place; returns
    (eligible rows, effective floor bit width or None), or None when the
    constraints exclude every row."""
    static_quant = provider.get("quantizations")
    # `routing` is a passthrough, so a value that is not a list is not a
    # constraint: it is ignored and the derived rule applies instead.
    # The author's list is literal: a short form covers its long variants and a
    # long form covers only itself, which is how the selector reads it too. Rows
    # are folded to their short form before matching.
    quant_values = frozenset(static_quant) if isinstance(static_quant, list) and static_quant else None
    floor = None
    static_price = provider.get("max_price") if isinstance(provider.get("max_price"), dict) else None
    if static_price is None and policy.price_cap_tolerance is not None:
        medians = price_caps(rows)
        if medians is not None:
            static_price = {
                "prompt": price_cap(medians[0], policy.price_cap_tolerance),
                "completion": price_cap(medians[1], policy.price_cap_tolerance),
            }
            provider["max_price"] = static_price
    # max_price is dollars per million tokens; rows are priced per token, so
    # enforce the sent values on the per-token scale. A static cap governs
    # filtering even when the derived rule is off.
    caps = (
        {field: amount / 1e6 for field, amount in static_price.items() if field in _PRICE_FIELDS}
        if static_price is not None
        else None
    )
    if quant_values is None:
        requested = policy.quantization_floor
        floor = quantization_floor(rows) if requested == _QUANT_FLOOR_AUTO else requested
        if floor is not None and requested == _QUANT_FLOOR_AUTO:
            lowest = min(bits for row in rows if (bits := row_bits(row)) is not None)
            # A derived floor equal to the lowest observed width excludes nothing,
            # so it is skipped rather than emitted as a no-op key. An explicit
            # tier is always emitted.
            if floor <= lowest:
                floor = None
        if floor is not None:
            emitted = quantization_values(floor, policy.include_unknown_quantization)
            quant_values = frozenset(emitted)
            provider["quantizations"] = emitted
    else:
        # A static list states its own floor, so the log can report it.
        floor = min((bits for row in rows if (bits := row_bits(row)) is not None and quant_allowed(row.get("quantization"), quant_values)), default=None)
    eligible = [row for row in rows if row_passes(row, caps, quant_values)]
    applied_floor = floor if quant_values is not None else None
    return (eligible, applied_floor) if eligible else None
