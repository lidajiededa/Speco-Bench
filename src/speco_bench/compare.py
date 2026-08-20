from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import aiohttp

from .client import _extract_text_delta


CompareEventEmitter = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class CompareModelTarget:
    base_url: str
    model: str
    api_key: str | None = None

    @property
    def endpoint_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"


def _stream_stats(
    *,
    status: str,
    started: float,
    first_token_at: float | None,
    ended: float | None,
    char_count: int,
    chunk_count: int,
    usage_output_tokens: int,
    finish_reason: str | None,
) -> dict[str, Any]:
    current = ended if ended is not None else time.perf_counter()
    elapsed_ms = max(0.0, (current - started) * 1000)
    ttft_ms = (
        max(0.0, (first_token_at - started) * 1000)
        if first_token_at is not None
        else None
    )
    decode_ms = (
        max(0.0, (current - first_token_at) * 1000)
        if first_token_at is not None
        else None
    )
    output_tokens = usage_output_tokens or chunk_count
    token_source = "usage" if usage_output_tokens > 0 else "stream_chunks"
    total_tokens_per_second = (
        output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
    )
    decode_token_count = max(0, output_tokens - 1)
    decode_tokens_per_second = (
        decode_token_count / (decode_ms / 1000)
        if decode_ms is not None and decode_ms > 0
        else 0.0
    )
    tpot_ms = (
        decode_ms / decode_token_count
        if decode_ms is not None and decode_token_count > 0
        else None
    )
    return {
        "status": status,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": elapsed_ms,
        "decode_ms": decode_ms,
        "char_count": char_count,
        "chunk_count": chunk_count,
        "output_tokens": output_tokens,
        "token_source": token_source,
        "total_tokens_per_second": total_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "chars_per_second": (
            char_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
        ),
        "chunks_per_second": (
            chunk_count / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
        ),
        "finish_reason": finish_reason,
    }


async def stream_model_comparison(
    *,
    side: str,
    target: CompareModelTarget,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    ignore_eos: bool,
    extra_body: dict[str, Any],
    request_timeout_seconds: float,
    emit: CompareEventEmitter,
) -> None:
    started = time.perf_counter()
    first_token_at: float | None = None
    ended: float | None = None
    char_count = 0
    chunk_count = 0
    usage_output_tokens = 0
    finish_reason: str | None = None

    def stats(status: str) -> dict[str, Any]:
        return _stream_stats(
            status=status,
            started=started,
            first_token_at=first_token_at,
            ended=ended,
            char_count=char_count,
            chunk_count=chunk_count,
            usage_output_tokens=usage_output_tokens,
            finish_reason=finish_reason,
        )

    await emit({"type": "status", "side": side, "stats": stats("connecting")})
    headers = {"Content-Type": "application/json"}
    if target.api_key:
        headers["Authorization"] = f"Bearer {target.api_key}"
    body: dict[str, Any] = {
        **extra_body,
        "model": target.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if ignore_eos:
        body["ignore_eos"] = True

    timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
    try:
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.post(target.endpoint_url, json=body) as response:
                if response.status >= 400:
                    response_body = await response.text()
                    raise RuntimeError(
                        f"HTTP {response.status}: {response_body[:1000]}"
                    )
                await emit(
                    {"type": "status", "side": side, "stats": stats("streaming")}
                )
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        usage_output_tokens = int(
                            usage.get("completion_tokens") or usage_output_tokens
                        )
                    text, chunk_finish_reason = _extract_text_delta(payload)
                    if chunk_finish_reason is not None:
                        finish_reason = chunk_finish_reason
                    if text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        chunk_count += 1
                        char_count += len(text)
                        await emit(
                            {
                                "type": "delta",
                                "side": side,
                                "text": text,
                                "stats": stats("streaming"),
                            }
                        )

        ended = time.perf_counter()
        if first_token_at is None:
            raise RuntimeError("stream completed without any generated text")
        await emit({"type": "done", "side": side, "stats": stats("completed")})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        ended = time.perf_counter()
        await emit(
            {
                "type": "error",
                "side": side,
                "error": f"{type(exc).__name__}: {exc}",
                "stats": stats("failed"),
            }
        )
