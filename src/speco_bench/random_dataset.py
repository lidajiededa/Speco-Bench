from __future__ import annotations

import math
import random
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


def generate_random_requests(
    tokenizer: TokenizerLike,
    *,
    num_prompts: int,
    input_length: int,
    output_length: int,
    range_ratio: float,
    seed: int,
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
        requests.append(
            BenchmarkRequest(
                request_id=request_id,
                prompt=prompt,
                max_tokens=requested_output_tokens,
                metadata={
                    "dataset": "random",
                    "requested_input_tokens": requested_input_tokens,
                    "prompt_tokens_without_specials": prompt_tokens,
                    "requested_output_tokens": requested_output_tokens,
                    "seed": seed,
                },
            )
        )
    return requests
