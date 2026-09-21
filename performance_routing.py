"""Local OpenRouter provider selection from its public endpoint metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceRouting:
    target_tokens: int = 10_000
    tolerance: float = 0.15
    cache_ttl_seconds: float = 900.0


def fast_tags(endpoints: Iterable[object], config: PerformanceRouting) -> list[str]:
    """Return endpoint tags within the configured projected-time tolerance.

    OpenRouter reports latency in milliseconds and throughput in tokens/second.
    Rows without usable measurements cannot be ranked.
    """
    scored: list[tuple[str, float]] = []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        tag = endpoint.get("tag")
        latency = (endpoint.get("latency_last_30m") or {}).get("p50")
        throughput = (endpoint.get("throughput_last_30m") or {}).get("p50")
        if not isinstance(tag, str) or not tag:
            continue
        if not isinstance(latency, (int, float)) or not isinstance(throughput, (int, float)):
            continue
        if latency < 0 or throughput <= 0:
            continue
        projected_seconds = latency / 1000 + config.target_tokens / throughput
        scored.append((tag, projected_seconds))

    if not scored:
        return []
    fastest = min(score for _, score in scored)
    limit = fastest * (1 + config.tolerance)
    return [tag for tag, score in scored if score <= limit]


def matches_provider_tag(tag: str, allowed: str) -> bool:
    """Whether an endpoint tag is covered by OpenRouter's provider selector."""
    return tag == allowed or tag.startswith(f"{allowed}/")
