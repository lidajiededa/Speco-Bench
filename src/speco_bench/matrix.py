from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import BenchmarkConfig
from .models import BenchmarkReport
from .output import save_report
from .progress import TerminalProgress
from .service import BenchmarkService


DATASET_ALIASES = {
    "gsm-8k": "gsm8k",
    "human-eval": "humaneval",
    "human_eval": "humaneval",
    "math-500": "math500",
    "mt-bench": "mt_bench",
}

CSV_FIELDS = [
    "dataset",
    "dataset_path",
    "concurrency",
    "requested_num_prompts",
    "total_requests",
    "successful_requests",
    "failed_requests",
    "benchmark_duration_seconds",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput_req_s",
    "output_throughput_tok_s",
    "total_token_throughput_tok_s",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p99_ms",
    "e2e_mean_ms",
    "e2e_p50_ms",
    "e2e_p90_ms",
    "e2e_p99_ms",
    "spec_metrics_available",
    "draft_rounds",
    "draft_tokens",
    "accepted_draft_tokens",
    "draft_token_acceptance_rate",
    "mean_acceptance_length",
    "position_acceptance_rates",
    "warnings",
    "summary_json",
    "requests_jsonl",
    "error",
]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    path: Path


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def _split_values(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def parse_concurrencies(values: Sequence[str]) -> list[int]:
    parsed: list[int] = []
    for value in _split_values(values):
        try:
            concurrency = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid concurrency: {value}") from exc
        if concurrency < 1:
            raise ValueError("concurrency values must be at least 1")
        if concurrency in parsed:
            raise ValueError(f"duplicate concurrency value: {concurrency}")
        parsed.append(concurrency)
    if not parsed:
        raise ValueError("at least one concurrency value is required")
    return parsed


def parse_num_prompts(
    values: Sequence[str] | None,
    *,
    expected_count: int,
) -> list[int | None]:
    if values is None:
        return [None] * expected_count
    parsed: list[int] = []
    for value in _split_values(values):
        try:
            num_prompts = int(value)
        except ValueError as exc:
            raise ValueError(f"invalid num-prompts value: {value}") from exc
        if num_prompts < 1:
            raise ValueError("num-prompts values must be at least 1")
        parsed.append(num_prompts)
    if len(parsed) != expected_count:
        raise ValueError(
            "num-prompts must contain exactly one value per concurrency: "
            f"expected {expected_count}, got {len(parsed)}"
        )
    return parsed


def resolve_datasets(values: Sequence[str], dataset_root: Path) -> list[DatasetSpec]:
    datasets: list[DatasetSpec] = []
    seen: set[Path] = set()
    for value in _split_values(values):
        direct_path = Path(value).expanduser()
        if direct_path.is_file():
            name = (
                direct_path.parent.name
                if direct_path.name == "question.jsonl"
                else direct_path.stem
            )
            path = direct_path
        elif direct_path.suffix.lower() in {".json", ".jsonl"} or "/" in value:
            raise FileNotFoundError(f"dataset does not exist: {direct_path}")
        else:
            normalized = value.lower().replace(" ", "_")
            normalized = DATASET_ALIASES.get(normalized, normalized)
            name = normalized
            path = dataset_root / normalized / "question.jsonl"
            if not path.is_file():
                raise FileNotFoundError(
                    f"unknown dataset '{value}': expected {path}"
                )
        resolved = path.resolve()
        if resolved not in seen:
            datasets.append(DatasetSpec(name=name, path=resolved))
            seen.add(resolved)
    if not datasets:
        raise ValueError("at least one dataset is required")
    return datasets


def report_to_csv_row(
    dataset: DatasetSpec,
    report: BenchmarkReport,
    *,
    requested_num_prompts: int | None,
    summary_path: Path,
    requests_path: Path,
) -> dict[str, Any]:
    summary = report.summary
    duration = float(summary["benchmark_duration_seconds"])
    total_tokens = int(summary["total_input_tokens"]) + int(
        summary["total_output_tokens"]
    )
    spec = report.spec_decode
    row: dict[str, Any] = {
        "dataset": dataset.name,
        "dataset_path": str(dataset.path),
        "concurrency": summary["concurrency"],
        "requested_num_prompts": (
            requested_num_prompts if requested_num_prompts is not None else "all"
        ),
        "total_requests": summary["total_requests"],
        "successful_requests": summary["successful_requests"],
        "failed_requests": summary["failed_requests"],
        "benchmark_duration_seconds": duration,
        "total_input_tokens": summary["total_input_tokens"],
        "total_output_tokens": summary["total_output_tokens"],
        "request_throughput_req_s": summary["request_throughput"],
        "output_throughput_tok_s": summary["output_throughput"],
        "total_token_throughput_tok_s": (
            total_tokens / duration if duration > 0 else 0.0
        ),
        "spec_metrics_available": spec.available,
        "draft_rounds": spec.num_drafts,
        "draft_tokens": spec.draft_tokens,
        "accepted_draft_tokens": spec.accepted_tokens,
        "draft_token_acceptance_rate": spec.acceptance_rate,
        "mean_acceptance_length": spec.mean_acceptance_length,
        "position_acceptance_rates": json.dumps(spec.position_acceptance_rates),
        "warnings": " | ".join(report.warnings),
        "summary_json": str(summary_path),
        "requests_jsonl": str(requests_path),
        "error": "",
    }
    for metric in ("ttft", "tpot", "e2e"):
        distribution = summary[f"{metric}_ms"]
        for statistic in ("mean", "p50", "p90", "p99"):
            row[f"{metric}_{statistic}_ms"] = distribution[statistic]
    return row


def _error_row(
    dataset: DatasetSpec,
    concurrency: int,
    requested_num_prompts: int | None,
    error: Exception,
) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "dataset": dataset.name,
            "dataset_path": str(dataset.path),
            "concurrency": concurrency,
            "requested_num_prompts": (
                requested_num_prompts
                if requested_num_prompts is not None
                else "all"
            ),
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def write_csv(rows: Sequence[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "dataset"


async def run_matrix(args: argparse.Namespace) -> int:
    datasets = resolve_datasets(args.datasets, args.dataset_root)
    concurrencies = parse_concurrencies(args.concurrencies)
    num_prompts_values = parse_num_prompts(
        args.num_prompts,
        expected_count=len(concurrencies),
    )
    load_shapes = list(zip(concurrencies, num_prompts_values))
    csv_path = args.csv_file or args.output_dir / "matrix.csv"
    rows: list[dict[str, Any]] = []
    had_errors = False
    total_runs = len(datasets) * len(load_shapes)
    run_number = 0

    for dataset in datasets:
        for concurrency, num_prompts in load_shapes:
            run_number += 1
            prompt_label = str(num_prompts) if num_prompts is not None else "all"
            result_dir = (
                args.output_dir
                / _slug(dataset.name)
                / f"concurrency-{concurrency}-prompts-{prompt_label}"
            )
            print(
                f"[{run_number}/{total_runs}] "
                f"dataset={dataset.name} concurrency={concurrency} "
                f"num_prompts={prompt_label}",
                file=sys.stderr,
            )
            config = BenchmarkConfig(
                base_url=args.base_url,
                model=args.model,
                dataset_path=dataset.path,
                output_dir=result_dir,
                concurrency=concurrency,
                num_prompts=num_prompts,
                max_tokens=args.max_tokens,
                endpoint_type=args.endpoint_type,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                warmup_requests=args.warmup_requests,
                request_timeout_seconds=args.request_timeout,
                api_key=args.api_key,
                metrics_url=args.metrics_url,
                ignore_eos=args.ignore_eos,
                extra_body=args.extra_body,
            )
            progress = None if args.no_progress else TerminalProgress()
            try:
                report = await BenchmarkService().run(
                    config,
                    progress_callback=progress,
                    progress_interval_seconds=args.progress_interval,
                )
                summary_path, requests_path = save_report(report, result_dir)
                rows.append(
                    report_to_csv_row(
                        dataset,
                        report,
                        requested_num_prompts=num_prompts,
                        summary_path=summary_path,
                        requests_path=requests_path,
                    )
                )
                if report.summary["failed_requests"]:
                    had_errors = True
            except Exception as exc:
                had_errors = True
                rows.append(
                    _error_row(
                        dataset,
                        concurrency,
                        num_prompts,
                        exc,
                    )
                )
                print(f"Run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    write_csv(rows, csv_path)
                    raise
            finally:
                if progress is not None:
                    progress.close()
            write_csv(rows, csv_path)

    print(f"CSV saved: {csv_path}")
    return 2 if had_errors else 0


def add_matrix_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "matrix",
        help="benchmark dataset/concurrency combinations",
        description="Run every dataset/concurrency combination and write a CSV.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="dataset names or JSON/JSONL paths; commas are also accepted",
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        required=True,
        help="one or more positive integers; commas are also accepted",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/matrix"))
    parser.add_argument("--csv-file", type=Path)
    parser.add_argument("--endpoint-type", choices=("chat", "completions"), default="chat")
    parser.add_argument(
        "--num-prompts",
        nargs="+",
        help=(
            "one positive value per concurrency; commas are also accepted; "
            "omit to use each full dataset"
        ),
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--metrics-url")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--extra-body", type=_json_object, default={})
    parser.add_argument("--progress-interval", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
