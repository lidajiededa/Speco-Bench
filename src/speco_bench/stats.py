from __future__ import annotations

import math
from statistics import fmean
from typing import Any, Iterable

from .models import RequestResult


def percentile(values: Iterable[float], percent: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    rank = (len(ordered) - 1) * percent / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values) if values else 0.0,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
    }


def summarize_requests(
    results: list[RequestResult],
    *,
    duration_seconds: float,
    dataset: str,
    concurrency: int,
) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    ttfts = [result.ttft_ms for result in successful]
    tpots = [result.tpot_ms for result in successful if result.output_tokens > 1]
    e2es = [result.e2e_ms for result in successful]
    total_input = sum(result.input_tokens for result in successful)
    total_output = sum(result.output_tokens for result in successful)
    safe_duration = max(duration_seconds, 1e-12)
    return {
        "dataset": dataset,
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "benchmark_duration_seconds": duration_seconds,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(successful) / safe_duration,
        "output_throughput": total_output / safe_duration,
        "ttft_ms": _distribution(ttfts),
        "tpot_ms": _distribution(tpots),
        "e2e_ms": _distribution(e2es),
    }

