#!/usr/bin/env python3
"""Download image-only VLM datasets and convert them for Speco-Bench."""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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
    "ai2d": DatasetDefinition("lmms-lab-encoder/ai2d", "test", "scientific diagrams"),
    "chartqa": DatasetDefinition("vis-nlp/ChartQA", "test", "chart reasoning"),
    "textvqa": DatasetDefinition("facebook/TextVQA", "validation", "text in images"),
    "mmmu": DatasetDefinition("MMMU/MMMU", "validation", "multi-discipline reasoning"),
}

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
AI2D_FILES = (
    "data/test-00000-of-00002.parquet",
    "data/test-00001-of-00002.parquet",
)
MMMU_SUBJECTS = (
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
)
CHARTQA_ROOT = "https://github.com/vis-nlp/ChartQA/raw/refs/heads/main/ChartQA%20Dataset/test"
TEXTVQA_ANNOTATIONS = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
OPEN_IMAGES_ROOT = "https://open-images-dataset.s3.amazonaws.com/train"


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
    return "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(values))


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
        answer = str(row.get("answer", ""))
        metadata["reference"] = (
            chr(65 + int(answer)) if answer.isdigit() and int(answer) < 26 else answer
        )
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


def _download_file(
    url: str,
    destination: Path,
    *,
    attempts: int = 5,
    timeout: float = 60,
) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Speco-Bench/0.1"})
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"download: {url}", file=sys.stderr)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected_size = response.headers.get("Content-Length")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected_size and temporary.stat().st_size != int(expected_size):
                raise OSError(
                    f"incomplete response: expected {expected_size} bytes, "
                    f"received {temporary.stat().st_size}"
                )
            temporary.replace(destination)
            return destination
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 308:
                break
            if attempt == attempts:
                break
            time.sleep(min(2**attempt, 10))

    curl = shutil.which("curl")
    if curl:
        print(f"download with curl fallback: {url}", file=sys.stderr)
        result = subprocess.run(
            [
                curl,
                "--http1.1",
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                str(attempts),
                "--retry-all-errors",
                "--retry-delay",
                "1",
                "--output",
                str(temporary),
                url,
            ],
            check=False,
        )
        if result.returncode == 0 and temporary.is_file() and temporary.stat().st_size:
            temporary.replace(destination)
            return destination
        temporary.unlink(missing_ok=True)
        last_error = RuntimeError(f"curl exited with status {result.returncode}")
    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def _download_image(url: str, destination: Path, *, attempts: int = 3) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "multimodal dataset preparation requires Pillow; "
            "install it with: pip install -e '.[multimodal]'"
        ) from exc

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            image_path = _download_file(url, destination, attempts=1, timeout=30)
            with Image.open(image_path) as image:
                image.verify()
            return image_path
        except (OSError, RuntimeError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"failed to download a valid image from {url}: {last_error}")


def _parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise RuntimeError(
            "multimodal dataset preparation requires PyArrow and Pillow; "
            "install them with: pip install -e '.[multimodal]'"
        ) from exc
    table = parquet.read_table(path)
    yield from table.to_pylist()


def _hf_file(dataset_id: str, filename: str, destination: Path) -> Path:
    encoded_id = "/".join(urllib.parse.quote(part) for part in dataset_id.split("/"))
    encoded_filename = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    url = f"{HF_ENDPOINT}/datasets/{encoded_id}/resolve/main/{encoded_filename}"
    for attempt in range(2):
        path = _download_file(url, destination)
        try:
            from pyarrow import parquet

            parquet.read_metadata(path)
            return path
        except ImportError as exc:
            raise RuntimeError(
                "multimodal dataset preparation requires PyArrow; "
                "install it with: pip install -e '.[multimodal]'"
            ) from exc
        except Exception as exc:
            path.unlink(missing_ok=True)
            if attempt:
                raise RuntimeError(f"downloaded Parquet file is invalid: {url}") from exc
    raise AssertionError("unreachable")


def _chartqa_rows(cache_dir: Path) -> Iterator[dict[str, Any]]:
    for annotation_name in ("test_human.json", "test_augmented.json"):
        annotation_path = _download_file(
            f"{CHARTQA_ROOT}/{annotation_name}",
            cache_dir / "chartqa" / annotation_name,
        )
        with annotation_path.open(encoding="utf-8") as source:
            rows = json.load(source)
        for row in rows:
            image_name = str(row["imgname"])
            image_url = f"{CHARTQA_ROOT}/png/{urllib.parse.quote(image_name)}"
            image_path = _download_image(
                image_url,
                cache_dir / "chartqa" / "images" / image_name,
            )
            yield {**row, "image": image_path}


def _textvqa_rows(cache_dir: Path) -> Iterator[dict[str, Any]]:
    annotation_path = _download_file(
        TEXTVQA_ANNOTATIONS,
        cache_dir / "textvqa" / "TextVQA_0.5.1_val.json",
    )
    with annotation_path.open(encoding="utf-8") as source:
        rows = json.load(source)["data"]
    for row in rows:
        image_id = str(row["image_id"])
        image_path = cache_dir / "textvqa" / "images" / f"{image_id}.jpg"
        try:
            image_path = _download_image(
                f"{OPEN_IMAGES_ROOT}/{image_id}.jpg",
                image_path,
            )
        except RuntimeError:
            fallback_url = row.get("flickr_300k_url") or row.get("flickr_original_url")
            if not fallback_url:
                raise
            image_path = _download_image(str(fallback_url), image_path)
        yield {**row, "image": image_path}


def _dataset_rows(
    name: str,
    *,
    cache_dir: Path,
) -> Iterator[tuple[dict[str, Any], str | None]]:
    definition = DATASETS[name]
    if name == "ai2d":
        for filename in AI2D_FILES:
            path = _hf_file(
                definition.dataset_id,
                filename,
                cache_dir / "ai2d" / Path(filename).name,
            )
            for row in _parquet_rows(path):
                yield row, None
        return

    if name == "chartqa":
        for row in _chartqa_rows(cache_dir):
            yield row, None
        return

    if name == "textvqa":
        for row in _textvqa_rows(cache_dir):
            yield row, None
        return

    for subject in MMMU_SUBJECTS:
        filename = f"{subject}/validation-00000-of-00001.parquet"
        path = _hf_file(
            definition.dataset_id,
            filename,
            cache_dir / "mmmu" / f"{subject}.parquet",
        )
        for row in _parquet_rows(path):
            yield row, subject


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
    cache_dir: Path,
    limit: int | None,
    max_tokens: int,
    overwrite: bool,
) -> int:
    destination = output_root / name
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
        shutil.rmtree(destination)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True)
    temporary = destination / "question.jsonl.part"
    final_path = destination / "question.jsonl"

    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for source_index, (row, subject) in enumerate(_dataset_rows(name, cache_dir=cache_dir)):
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw" / "multimodal",
        help="Raw download cache (default: raw/multimodal).",
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
            cache_dir=args.cache_dir,
            limit=args.limit,
            max_tokens=args.max_tokens,
            overwrite=args.overwrite,
        )
        print(f"{name}: {count} records -> {args.output_dir / name / 'question.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
