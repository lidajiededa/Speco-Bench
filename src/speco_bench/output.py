from __future__ import annotations

import json
from pathlib import Path

from .models import BenchmarkReport


def save_report(report: BenchmarkReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    requests_path = output_dir / "requests.jsonl"
    summary_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with requests_path.open("w", encoding="utf-8") as handle:
        for result in report.requests:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    return summary_path, requests_path


def format_report(report: BenchmarkReport) -> str:
    data = report.summary
    ttft = data["ttft_ms"]
    tpot = data["tpot_ms"]
    lines = [
        "============ Benchmark Result ============",
        f"Dataset:                  {data['dataset']}",
        f"Successful requests:      {data['successful_requests']} / {data['total_requests']}",
        f"Concurrency:              {data['concurrency']}",
        f"Benchmark duration:       {data['benchmark_duration_seconds']:.2f} s",
        f"Total input tokens:       {data['total_input_tokens']}",
        f"Total output tokens:      {data['total_output_tokens']}",
        f"Request throughput:       {data['request_throughput']:.2f} requests/s",
        f"Output throughput:        {data['output_throughput']:.2f} tokens/s",
        "",
        f"Mean TTFT:                {ttft['mean']:.2f} ms",
        f"P50 / P90 / P99 TTFT:     {ttft['p50']:.2f} / {ttft['p90']:.2f} / {ttft['p99']:.2f} ms",
        f"Mean TPOT:                {tpot['mean']:.2f} ms",
        f"P50 / P90 / P99 TPOT:     {tpot['p50']:.2f} / {tpot['p90']:.2f} / {tpot['p99']:.2f} ms",
        "",
        "========== Speculative Decoding ==========",
    ]
    spec = report.spec_decode
    if spec.available:
        acceptance = (
            f"{spec.acceptance_rate * 100:.2f}%"
            if spec.acceptance_rate is not None
            else "N/A"
        )
        lines.extend(
            [
                f"Draft rounds:              {spec.num_drafts}",
                f"Draft tokens:              {spec.draft_tokens}",
                f"Accepted draft tokens:     {spec.accepted_tokens}",
                f"Draft token acceptance:    {acceptance}",
                f"Mean acceptance length:    {spec.mean_acceptance_length:.2f}",
            ]
        )
        for index, rate in enumerate(spec.position_acceptance_rates, start=1):
            lines.append(f"Position {index:<3} acceptance:   {rate * 100:.2f}%")
    else:
        lines.append(f"Unavailable: {spec.warning or 'unknown reason'}")
    if report.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines)

