from __future__ import annotations

import base64
import copy
import json
import mimetypes
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .models import BenchmarkRequest


class DatasetError(ValueError):
    pass


def _local_image_data_url(
    reference: str,
    *,
    base_dir: Path,
    request_id: int,
) -> str:
    if reference.startswith(("data:", "http://", "https://")):
        return reference

    if reference.startswith("file://"):
        parsed = urlparse(reference)
        image_path = Path(unquote(parsed.path))
    else:
        image_path = Path(reference).expanduser()
        if not image_path.is_absolute():
            image_path = base_dir / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise DatasetError(f"record {request_id} image does not exist: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise DatasetError(
            f"record {request_id} has an unsupported image type: {image_path}"
        )
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _normalize_image_reference(
    value: Any,
    *,
    base_dir: Path,
    request_id: int,
) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "url": _local_image_data_url(
                value,
                base_dir=base_dir,
                request_id=request_id,
            )
        }
    if not isinstance(value, dict) or not isinstance(value.get("url"), str):
        raise DatasetError(
            f"record {request_id} image_url must be a string or an object with a URL"
        )
    normalized = dict(value)
    normalized["url"] = _local_image_data_url(
        value["url"],
        base_dir=base_dir,
        request_id=request_id,
    )
    return normalized


def _normalize_messages(
    messages: Any,
    *,
    base_dir: Path,
    request_id: int,
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise DatasetError(f"record {request_id} messages must be a non-empty list")
    normalized = copy.deepcopy(messages)
    for message in normalized:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise DatasetError(f"record {request_id} contains an invalid message")
        content = message.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise DatasetError(
                f"record {request_id} message content must be a string or list"
            )
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("type"), str):
                raise DatasetError(
                    f"record {request_id} contains an invalid message content part"
                )
            if part["type"] == "image_url":
                part["image_url"] = _normalize_image_reference(
                    part.get("image_url"),
                    base_dir=base_dir,
                    request_id=request_id,
                )
    return normalized


def request_has_images(request: BenchmarkRequest) -> bool:
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for message in request.messages or []
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetError(f"dataset does not exist: {path}")

    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON in {path}: {exc}") from exc
        if isinstance(data, dict):
            data = data.get("data", [data])
        if not isinstance(data, list):
            raise DatasetError("JSON dataset must be a list or an object with a data list")
        records = data
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"invalid JSON at line {line_no}: {exc}") from exc
                if not isinstance(record, dict):
                    raise DatasetError(f"line {line_no} must be a JSON object")
                records.append(record)

    if not records:
        raise DatasetError("dataset is empty")
    if not all(isinstance(item, dict) for item in records):
        raise DatasetError("every dataset record must be a JSON object")
    return records


def _normalize_record(
    record: dict[str, Any],
    request_id: int,
    max_tokens_override: int | None,
    *,
    base_dir: Path | None = None,
) -> BenchmarkRequest:
    base_dir = (base_dir or Path.cwd()).resolve()
    prompt = record.get("prompt")
    messages = record.get("messages")
    image_value = record.get("image")
    images_value = record.get("images")
    if image_value is not None and images_value is not None:
        raise DatasetError(f"record {request_id} cannot contain both 'image' and 'images'")
    image_references = images_value if images_value is not None else image_value
    if image_references is not None:
        if messages is not None:
            raise DatasetError(
                f"record {request_id} cannot combine 'messages' with 'image' or 'images'"
            )
        if not isinstance(prompt, str):
            raise DatasetError(
                f"record {request_id} image shorthand requires a string prompt"
            )
        if not isinstance(image_references, list):
            image_references = [image_references]
        if not image_references:
            raise DatasetError(f"record {request_id} images must not be empty")
        content = [
            {
                "type": "image_url",
                "image_url": _normalize_image_reference(
                    reference,
                    base_dir=base_dir,
                    request_id=request_id,
                ),
            }
            for reference in image_references
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        prompt = None
    if (prompt is None) == (messages is None):
        raise DatasetError(
            f"record {request_id} must contain exactly one of 'prompt' or 'messages'"
        )
    if prompt is not None and not isinstance(prompt, str):
        raise DatasetError(f"record {request_id} prompt must be a string")
    if messages is not None:
        messages = _normalize_messages(
            messages,
            base_dir=base_dir,
            request_id=request_id,
        )

    row_max_tokens = record.get("max_tokens", 256)
    max_tokens = max_tokens_override if max_tokens_override is not None else row_max_tokens
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise DatasetError(f"record {request_id} max_tokens must be a positive integer")

    reserved = {"prompt", "messages", "image", "images", "max_tokens", "metadata"}
    metadata = dict(record.get("metadata") or {})
    metadata.update({key: value for key, value in record.items() if key not in reserved})
    return BenchmarkRequest(
        request_id=request_id,
        prompt=prompt,
        messages=messages,
        max_tokens=max_tokens,
        metadata=metadata,
    )


def load_dataset(
    path: Path,
    *,
    num_prompts: int | None = None,
    max_tokens: int | None = None,
    seed: int = 0,
) -> list[BenchmarkRequest]:
    """Load a JSON/JSONL dataset and deterministically sample or cycle it."""

    records = _read_records(path)
    target = num_prompts or len(records)
    rng = random.Random(seed)

    if target <= len(records):
        indices = list(range(len(records)))
        rng.shuffle(indices)
        selected = [records[index] for index in indices[:target]]
    else:
        indices = list(range(len(records)))
        rng.shuffle(indices)
        selected = [records[indices[index % len(indices)]] for index in range(target)]

    return [
        _normalize_record(
            record,
            request_id,
            max_tokens,
            base_dir=path.resolve().parent,
        )
        for request_id, record in enumerate(selected)
    ]


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}


def _sharegpt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    source = record.get("conversations")
    if not isinstance(source, list):
        raise DatasetError("ShareGPT record must contain a conversations list")
    messages: list[dict[str, str]] = []
    for item in source:
        if not isinstance(item, dict):
            raise DatasetError("ShareGPT conversation items must be objects")
        role = ROLE_MAP.get(str(item.get("from", "")).lower())
        content = item.get("value")
        if role is None or not isinstance(content, str):
            raise DatasetError("invalid ShareGPT role or content")
        messages.append({"role": role, "content": content})
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages:
        raise DatasetError("ShareGPT record has no usable prompt messages")
    return messages


def _alpaca_prompt(record: dict[str, Any]) -> str:
    instruction = record.get("instruction")
    input_text = record.get("input", "")
    if not isinstance(instruction, str) or not isinstance(input_text, str):
        raise DatasetError("Alpaca record requires string instruction and input fields")
    return instruction if not input_text else f"{instruction}\n\n{input_text}"


def prepare_records(
    records: Iterable[dict[str, Any]],
    *,
    source_format: str,
    default_max_tokens: int,
) -> list[dict[str, Any]]:
    """Convert common dataset schemas to Speco-Bench JSONL records."""

    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        detected = source_format
        if detected == "auto":
            if "conversations" in record:
                detected = "sharegpt"
            elif "instruction" in record:
                detected = "alpaca"
            elif "messages" in record:
                detected = "messages"
            elif "prompt" in record:
                detected = "prompt"
            else:
                raise DatasetError(f"cannot detect format for record {index}")

        if detected == "sharegpt":
            normalized: dict[str, Any] = {"messages": _sharegpt_messages(record)}
        elif detected == "alpaca":
            normalized = {"prompt": _alpaca_prompt(record)}
        elif detected == "messages":
            normalized = {"messages": record.get("messages")}
        elif detected == "prompt":
            normalized = {"prompt": record.get("prompt")}
        else:
            raise DatasetError(f"unsupported source format: {source_format}")

        normalized["max_tokens"] = int(record.get("max_tokens", default_max_tokens))
        normalized["metadata"] = {
            "source_index": index,
            **dict(record.get("metadata") or {}),
        }
        _normalize_record(normalized, index, None)
        prepared.append(normalized)
    return prepared


def prepare_dataset_file(
    input_path: Path,
    output_path: Path,
    *,
    source_format: str = "auto",
    default_max_tokens: int = 256,
    limit: int | None = None,
) -> int:
    records = _read_records(input_path)
    if limit is not None:
        if limit < 1:
            raise DatasetError("limit must be at least 1")
        records = records[:limit]
    prepared = prepare_records(
        records,
        source_format=source_format,
        default_max_tokens=default_max_tokens,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in prepared:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(prepared)
