#!/usr/bin/env python3
"""Download and convert speculative-decoding benchmark datasets.

The generated records contain ``prompt``/``max_tokens`` for Speco-Bench and
also retain ``turns`` for AngelSlim's ``tools/vllm_spec_benchmark.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


@dataclass(frozen=True)
class Source:
    filename: str
    url: str
    sha256: str


SOURCES = {
    "math500": Source(
        filename="math500.jsonl",
        url=(
            "https://raw.githubusercontent.com/notwitcheer/llm-bench-rig/"
            "252c038f7bc26e291fda2a746ea9b9cb30111bab/"
            "dataset/em/math500.jsonl"
        ),
        sha256="ddda84a0b9060832f5b014bb30dc1cf66d0ff99baafa78cda32f036084996c12",
    ),
    "humaneval": Source(
        filename="HumanEval.jsonl.gz",
        url=(
            "https://raw.githubusercontent.com/openai/human-eval/"
            "6d43fb980f9fee3c892a914eda09951f772ad10d/data/HumanEval.jsonl.gz"
        ),
        sha256="b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    ),
    "mbpp": Source(
        filename="sanitized-mbpp.json",
        url=(
            "https://raw.githubusercontent.com/google-research/google-research/"
            "ec7c3d346277b737bc2decffcd1b533d4b7ec105/"
            "mbpp/sanitized-mbpp.json"
        ),
        sha256="ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9",
    ),
    "mt_bench": Source(
        filename="mt_bench.jsonl",
        url=(
            "https://raw.githubusercontent.com/lm-sys/FastChat/"
            "b494d0c6b4e7935f1764f8439e75da3e66beccc7/"
            "fastchat/llm_judge/data/mt_bench/question.jsonl"
        ),
        sha256="119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
    ),
}

EXPECTED_COUNTS = {
    "math500": 500,
    "humaneval": 164,
    "mbpp": 257,
    "mt_bench": 80,
}

MATH_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
HUMANEVAL_TEMPLATE = (
    "Write a solution to the following problem and make sure that it passes "
    "the tests:\n```python\n{prompt}\n```"
)
DEFAULT_MAX_TOKENS = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source(path: Path, source: Source) -> None:
    actual = _sha256(path)
    if actual != source.sha256:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {source.sha256}, got {actual}"
        )


def _download(source: Source, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "Speco-Bench dataset preparer"},
    )

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            _validate_source(temporary, source)
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {source.url} after {retries} attempts"
                ) from exc
            time.sleep(attempt * 2)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def _convert_math500(path: Path) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(_read_jsonl(path)):
        yield {
            "question_id": index,
            "source_id": row["id"],
            "category": "math",
            "turns": [row["problem"] + MATH_INSTRUCTION],
            "reference": [row["answer"]],
        }


def _convert_humaneval(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            yield {
                "question_id": row["task_id"],
                "category": "code",
                "turns": [HUMANEVAL_TEMPLATE.format(prompt=row["prompt"])],
                "reference": [row["canonical_solution"]],
                "entry_point": row["entry_point"],
                "test": row["test"],
            }


def _convert_mbpp(path: Path) -> Iterator[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        task_id = row["task_id"]
        # This reproduces the Hugging Face sanitized/test split.
        if 11 <= task_id <= 510:
            yield {
                "question_id": task_id,
                "category": "code",
                "turns": [row["prompt"]],
                "reference": [row["code"]],
                "test_imports": row["test_imports"],
                "test_list": row["test_list"],
            }


def _convert_mt_bench(path: Path) -> Iterator[dict[str, Any]]:
    yield from _read_jsonl(path)


CONVERTERS: dict[str, Callable[[Path], Iterable[dict[str, Any]]]] = {
    "math500": _convert_math500,
    "humaneval": _convert_humaneval,
    "mbpp": _convert_mbpp,
    "mt_bench": _convert_mt_bench,
}


def _write_jsonl(rows: Iterable[dict[str, Any]], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as output:
        for row in rows:
            if not isinstance(row.get("turns"), list) or not row["turns"]:
                raise ValueError(f"Record {count} has no non-empty 'turns' list")
            row["prompt"] = row["turns"][0]
            row["max_tokens"] = DEFAULT_MAX_TOKENS
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(SOURCES),
        default=list(SOURCES),
        help="Datasets to prepare (default: all).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
        help="Directory containing downloaded source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing raw files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download all selected raw files again.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in args.datasets:
        source = SOURCES[name]
        raw_path = args.raw_dir / source.filename

        if args.force_download or (args.download and not raw_path.exists()):
            print(f"Downloading {name} from {source.url}", file=sys.stderr)
            _download(source, raw_path)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing {raw_path}; rerun with --download to fetch it"
            )
        _validate_source(raw_path, source)

        output_path = args.output_dir / name / "question.jsonl"
        count = _write_jsonl(CONVERTERS[name](raw_path), output_path)
        expected = EXPECTED_COUNTS[name]
        if count != expected:
            output_path.unlink(missing_ok=True)
            raise ValueError(f"{name}: expected {expected} records, converted {count}")
        print(f"{name}: {count} records -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
