from __future__ import annotations

import base64
import binascii
import math
import random
import struct
import zlib
from typing import Any, Protocol

from .models import BenchmarkRequest


class TokenizerLike(Protocol):
    vocab_size: int
    all_special_ids: list[int]

    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...


def load_tokenizer(name_or_path: str, *, trust_remote_code: bool) -> TokenizerLike:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "random dataset mode requires transformers; "
            "install it with: pip install -e '.[random]'"
        ) from exc

    return AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )


def _decode(tokenizer: TokenizerLike, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids)


def _encode(
    tokenizer: TokenizerLike,
    text: str,
    *,
    add_special_tokens: bool,
) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _allowed_token_ids(tokenizer: TokenizerLike) -> list[int]:
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        token_ids = sorted(set(get_vocab().values()))
    else:
        token_ids = list(range(int(tokenizer.vocab_size)))
    allowed = [token_id for token_id in token_ids if token_id not in special_ids]
    if not allowed:
        raise ValueError("tokenizer has no non-special tokens")
    return allowed


def _fit_prompt_to_length(
    tokenizer: TokenizerLike,
    token_ids: list[int],
    target_length: int,
    *,
    allowed_token_ids: list[int],
    rng: random.Random,
    max_retries: int = 20,
) -> tuple[str, int]:
    sequence = list(token_ids)
    for _ in range(max_retries + 1):
        prompt = _decode(tokenizer, sequence)
        encoded = _encode(tokenizer, prompt, add_special_tokens=False)
        if len(encoded) == target_length:
            return prompt, len(encoded)
        if len(encoded) > target_length:
            sequence = encoded[:target_length]
        else:
            sequence = encoded + [
                allowed_token_ids[rng.randrange(len(allowed_token_ids))]
                for _ in range(target_length - len(encoded))
            ]
    raise RuntimeError(
        f"could not create a prompt with exactly {target_length} tokens "
        f"after {max_retries} retries"
    )


def _length_bounds(length: int, range_ratio: float) -> tuple[int, int]:
    lower = max(1, math.floor(length * (1.0 - range_ratio)))
    upper = max(lower, math.ceil(length * (1.0 + range_ratio)))
    return lower, upper


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def generate_random_image_data_url(width: int, height: int, *, seed: int) -> str:
    """Build a deterministic RGB PNG without an image-library dependency."""

    if width < 1 or height < 1:
        raise ValueError("random image dimensions must be at least 1 pixel")
    if width > 16384 or height > 16384:
        raise ValueError("random image dimensions cannot exceed 16384 pixels")

    scanlines = bytearray()
    for y in range(height):
        red = (seed * 37 + y * 3) % 256
        green = (seed * 67 + y * 5) % 256
        blue = (seed * 97 + y * 7) % 256
        scanlines.extend(b"\x00")
        scanlines.extend(bytes((red, green, blue)) * width)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def generate_random_requests(
    tokenizer: TokenizerLike,
    *,
    num_prompts: int,
    input_length: int,
    output_length: int,
    range_ratio: float,
    seed: int,
    image_width: int | None = None,
    image_height: int | None = None,
    images_per_prompt: int = 1,
) -> list[BenchmarkRequest]:
    """Generate reproducible synthetic prompts with tokenizer-verified lengths."""

    if num_prompts < 1:
        raise ValueError("num_prompts must be at least 1")
    if input_length < 1:
        raise ValueError("random_input_len must be at least 1")
    if output_length < 1:
        raise ValueError("random_output_len must be at least 1")
    if not 0 <= range_ratio < 1:
        raise ValueError("random_range_ratio must be in [0, 1)")
    if (image_width is None) != (image_height is None):
        raise ValueError("random image width and height must be set together")
    if images_per_prompt < 1:
        raise ValueError("images_per_prompt must be at least 1")
    if image_width is None and images_per_prompt != 1:
        raise ValueError("images_per_prompt requires random image dimensions")

    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    input_lower, input_upper = _length_bounds(input_length, range_ratio)
    output_lower, output_upper = _length_bounds(output_length, range_ratio)
    if input_lower <= special_tokens:
        raise ValueError(
            "random_input_len is too small for this tokenizer and range ratio: "
            f"minimum input length {input_lower}, special tokens {special_tokens}"
        )

    rng = random.Random(seed)
    allowed_token_ids = _allowed_token_ids(tokenizer)
    requests: list[BenchmarkRequest] = []
    for request_id in range(num_prompts):
        requested_input_tokens = rng.randint(input_lower, input_upper)
        requested_output_tokens = rng.randint(output_lower, output_upper)
        prompt_token_target = requested_input_tokens - special_tokens
        offset = rng.randrange(len(allowed_token_ids))
        initial_ids = [
            allowed_token_ids[(offset + request_id + index) % len(allowed_token_ids)]
            for index in range(prompt_token_target)
        ]
        prompt, prompt_tokens = _fit_prompt_to_length(
            tokenizer,
            initial_ids,
            prompt_token_target,
            allowed_token_ids=allowed_token_ids,
            rng=rng,
        )
        total_tokens = len(
            _encode(tokenizer, prompt, add_special_tokens=True)
        )
        if total_tokens != requested_input_tokens:
            raise RuntimeError(
                "tokenizer special-token accounting changed while generating "
                f"request {request_id}: expected {requested_input_tokens}, got {total_tokens}"
            )
        metadata = {
            "dataset": "random",
            "requested_input_tokens": requested_input_tokens,
            "prompt_tokens_without_specials": prompt_tokens,
            "requested_output_tokens": requested_output_tokens,
            "seed": seed,
        }
        messages = None
        if image_width is not None and image_height is not None:
            content: list[dict[str, Any]] = []
            for image_index in range(images_per_prompt):
                image_seed = seed + request_id * images_per_prompt + image_index
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": generate_random_image_data_url(
                                image_width,
                                image_height,
                                seed=image_seed,
                            )
                        },
                    }
                )
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            metadata.update(
                {
                    "image_width": image_width,
                    "image_height": image_height,
                    "images_per_prompt": images_per_prompt,
                }
            )
        requests.append(
            BenchmarkRequest(
                request_id=request_id,
                prompt=prompt if messages is None else None,
                messages=messages,
                max_tokens=requested_output_tokens,
                metadata=metadata,
            )
        )
    return requests
