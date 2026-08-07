from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .dataset import prepare_dataset_file
from .matrix import add_matrix_parser, run_matrix
from .output import format_report, save_report
from .progress import TerminalProgress
from .service import BenchmarkService


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speco-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="benchmark an OpenAI-compatible server")
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument(
        "--dataset-name",
        choices=("custom", "random"),
        default="custom",
        help="custom reads --dataset-path; random generates synthetic prompts",
    )
    run.add_argument("--dataset-path", type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("results"))
    run.add_argument("--endpoint-type", choices=("chat", "completions"), default="chat")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--num-prompts", type=int)
    run.add_argument("--max-tokens", type=int)
    run.add_argument("--random-input-len", type=int, default=1024)
    run.add_argument("--random-output-len", type=int, default=128)
    run.add_argument("--random-range-ratio", type=float, default=0.0)
    run.add_argument("--random-image-width", type=int)
    run.add_argument("--random-image-height", type=int)
    run.add_argument("--random-images-per-prompt", type=int, default=1)
    run.add_argument(
        "--tokenizer",
        help="tokenizer name or path for random prompts (default: --model)",
    )
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument(
        "--ignore-eos",
        action="store_true",
        help="ask vLLM to generate until the requested output length",
    )
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--warmup-requests", type=int, default=1)
    run.add_argument("--request-timeout", type=float, default=3600.0)
    run.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    run.add_argument("--metrics-url")
    run.add_argument("--extra-body", type=_json_object, default={})
    run.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="seconds between time-based progress updates (default: 1)",
    )
    run.add_argument(
        "--no-progress",
        action="store_true",
        help="disable terminal progress output",
    )

    prepare = subparsers.add_parser("prepare", help="normalize a JSON/JSONL dataset")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--format",
        choices=("auto", "prompt", "messages", "sharegpt", "alpaca"),
        default="auto",
    )
    prepare.add_argument("--default-max-tokens", type=int, default=256)
    prepare.add_argument("--limit", type=int)
    add_matrix_parser(subparsers)

    web = subparsers.add_parser(
        "web",
        help="start the local benchmark web console",
    )
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8080)
    web.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    web.add_argument("--output-dir", type=Path, default=Path("results/web"))
    return parser


async def _run_benchmark(args: argparse.Namespace) -> int:
    config = BenchmarkConfig(
        base_url=args.base_url,
        model=args.model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        endpoint_type=args.endpoint_type,
        concurrency=args.concurrency,
        num_prompts=args.num_prompts,
        max_tokens=args.max_tokens,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        random_range_ratio=args.random_range_ratio,
        random_image_width=args.random_image_width,
        random_image_height=args.random_image_height,
        random_images_per_prompt=args.random_images_per_prompt,
        tokenizer=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        ignore_eos=args.ignore_eos,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        warmup_requests=args.warmup_requests,
        request_timeout_seconds=args.request_timeout,
        api_key=args.api_key,
        metrics_url=args.metrics_url,
        extra_body=args.extra_body,
    )
    progress = None if args.no_progress else TerminalProgress()
    try:
        report = await BenchmarkService().run(
            config,
            progress_callback=progress,
            progress_interval_seconds=args.progress_interval,
        )
    finally:
        if progress is not None:
            progress.close()
    summary_path, requests_path = save_report(report, config.output_dir)
    print(format_report(report))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {requests_path}")
    return 0 if report.summary["failed_requests"] == 0 else 2


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        count = prepare_dataset_file(
            args.input,
            args.output,
            source_format=args.format,
            default_max_tokens=args.default_max_tokens,
            limit=args.limit,
        )
        print(f"Prepared {count} records: {args.output}")
        return
    if args.command == "matrix":
        try:
            exit_code = asyncio.run(run_matrix(args))
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"error: {exc}") from exc
        raise SystemExit(exit_code)
    if args.command == "web":
        from .web import run_web_server

        run_web_server(
            host=args.host,
            port=args.port,
            dataset_root=args.dataset_root,
            output_root=args.output_dir,
        )
        return
    raise SystemExit(asyncio.run(_run_benchmark(args)))
