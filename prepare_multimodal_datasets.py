#!/usr/bin/env python3
"""Download image-only VLM datasets and convert them for Speco-Bench."""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetDefinition:
    dataset_id: str
    split: str
    task: str


DATASETS = {
    "ai2d": DatasetDefinition("lmms-lab/ai2d", "test", "scientific diagrams"),
    "chartqa": DatasetDefinition("HuggingFaceM4/ChartQA", "test", "chart reasoning"),
    "textvqa": DatasetDefinition("lmms-lab/textvqa", "validation", "text in images"),
    "mmmu": DatasetDefinition("MMMU/MMMU", "validation", "multi-discipline reasoning"),
}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    return [] if value is None else [value]


def _lettered_options(options: Any) -> str:
    values = _as_list(options)
    return "\n".join(
        f"{chr(65 + index)}. {value}" for index, value in enumerate(values)
    )


def _build_prompt_and_metadata(
    name: str,
    row: dict[str, Any],
    *,
    source_index: int,
    subject: str | None = None,
) -> tuple[str, list[Any], dict[str, Any]]:
    definition = DATASETS[name]
    source_id = row.get("id", row.get("question_id", source_index))
    metadata: dict[str, Any] = {
        "source_dataset": definition.dataset_id,
        "source_split": definition.split,
        "source_id": source_id,
        "task": definition.task,
    }

    if name == "ai2d":
        options = _lettered_options(row.get("options"))
        prompt = f"{row['question']}\n\n{options}\n\nAnswer with the option letter only."
        images = [row.get("image")]
        metadata["reference"] = row.get("answer")
    elif name == "chartqa":
        prompt = f"{row['query']}\n\nAnswer the question using the chart."
        images = [row.get("image")]
        metadata["reference"] = row.get("label")
    elif name == "textvqa":
        prompt = f"{row['question']}\n\nAnswer using the text visible in the image."
        images = [row.get("image")]
        metadata["reference"] = row.get("answers")
    elif name == "mmmu":
        question = str(row["question"])
        question = re.sub(r"<image\s*(\d+)>", r"[Image \1]", question)
        options = _lettered_options(row.get("options"))
        option_block = f"\n\n{options}" if options else ""
        prompt = (
            "Images are supplied in numeric order.\n\n"
            f"{question}{option_block}\n\nAnswer with the option letter or a concise answer."
        )
        images = [row.get(f"image_{index}") for index in range(1, 8)]
        metadata["reference"] = row.get("answer")
        metadata["question_type"] = row.get("question_type")
        if subject:
            metadata["subject"] = subject
    else:
        raise ValueError(f"unsupported dataset: {name}")

    usable_images = [image for image in images if image is not None]
    if not usable_images:
        raise ValueError(f"{name} record {source_id} has no image")
    return prompt, usable_images, metadata


def _dataset_rows(name: str) -> Iterator[tuple[dict[str, Any], str | None]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "multimodal dataset preparation requires datasets and Pillow; "
            "install them with: pip install -e '.[multimodal]'"
        ) from exc

    definition = DATASETS[name]
    if name != "mmmu":
        rows = load_dataset(
            definition.dataset_id,
            split=definition.split,
            streaming=True,
        )
        for row in rows:
            yield dict(row), None
        return

    for subject in get_dataset_config_names(definition.dataset_id):
        rows = load_dataset(
            definition.dataset_id,
            subject,
            split=definition.split,
            streaming=True,
        )
        for row in rows:
            yield dict(row), subject


def _open_image(value: Any):
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "multimodal dataset preparation requires Pillow; "
            "install it with: pip install -e '.[multimodal]'"
        ) from exc

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            source_image = Image.open(io.BytesIO(value["bytes"]))
        elif value.get("path"):
            source_image = Image.open(value["path"])
        else:
            raise ValueError("image object contains neither bytes nor path")
    elif isinstance(value, (str, Path)):
        source_image = Image.open(value)
    elif hasattr(value, "convert"):
        source_image = value
    else:
        raise ValueError(f"unsupported image value: {type(value).__name__}")
    normalized = ImageOps.exif_transpose(source_image).convert("RGB")
    if normalized is not source_image:
        source_image.close()
    return normalized


def prepare_dataset(
    name: str,
    *,
    output_root: Path,
    limit: int | None,
    max_tokens: int,
    overwrite: bool,
) -> int:
    destination = output_root / name
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(destination)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True)
    temporary = destination / "question.jsonl.part"
    final_path = destination / "question.jsonl"

    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for source_index, (row, subject) in enumerate(_dataset_rows(name)):
                if limit is not None and count >= limit:
                    break
                prompt, images, metadata = _build_prompt_and_metadata(
                    name,
                    row,
                    source_index=source_index,
                    subject=subject,
                )
                paths = []
                for image_index, image_value in enumerate(images, start=1):
                    filename = f"{count:06d}_{image_index}.jpg"
                    image_path = image_dir / filename
                    image = _open_image(image_value)
                    image.save(image_path, format="JPEG", quality=95)
                    image.close()
                    paths.append(f"images/{filename}")
                record = {
                    "prompt": prompt,
                    "images": paths,
                    "max_tokens": max_tokens,
                    "metadata": metadata,
                }
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
                count += 1
                if count % 100 == 0:
                    print(f"{name}: prepared {count} records", file=sys.stderr)
        if count == 0:
            raise ValueError(f"{name} produced no records")
        temporary.replace(final_path)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
        help="Datasets to prepare (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
    )
    parser.add_argument("--limit", type=int, help="Maximum records per dataset.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    for name in args.datasets:
        count = prepare_dataset(
            name,
            output_root=args.output_dir,
            limit=args.limit,
            max_tokens=args.max_tokens,
            overwrite=args.overwrite,
        )
        print(f"{name}: {count} records -> {args.output_dir / name / 'question.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
