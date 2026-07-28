from __future__ import annotations

import json
import time
from typing import Any

import aiohttp

from .config import BenchmarkConfig
from .models import BenchmarkRequest, RequestResult


def _extract_text_delta(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices") or []
    if not choices:
        return "", None
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if "delta" in choice:
        delta = choice.get("delta") or {}
        for key in ("content", "reasoning_content"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value, finish_reason
        return "", finish_reason
    value = choice.get("text")
    return (value if isinstance(value, str) else ""), finish_reason


class OpenAIStreamingClient:
    """OpenAI-compatible streaming client used by the benchmark service."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "OpenAIStreamingClient":
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
        connector = aiohttp.TCPConnector(limit=max(8, self.config.concurrency * 2))
        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=connector,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("client must be used as an async context manager")
        return self._session

    @property
    def endpoint_url(self) -> str:
        base = self.config.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        suffix = "chat/completions" if self.config.endpoint_type == "chat" else "completions"
        return f"{base}/{suffix}"

    def _payload(self, request: BenchmarkRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": request.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.ignore_eos:
            payload["ignore_eos"] = True
        payload.update(self.config.extra_body)
        if self.config.endpoint_type == "chat":
            if request.messages is None:
                payload["messages"] = [{"role": "user", "content": request.prompt or ""}]
            else:
                payload["messages"] = request.messages
        else:
            if request.prompt is None:
                raise ValueError("completions endpoint requires prompt records")
            payload["prompt"] = request.prompt
        return payload

    async def generate(self, request: BenchmarkRequest) -> RequestResult:
        started = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        finish_reason: str | None = None
        input_tokens = output_tokens = 0
        nonempty_chunks = 0
        try:
            async with self.session.post(
                self.endpoint_url,
                json=self._payload(request),
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")

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
                        input_tokens = int(usage.get("prompt_tokens") or input_tokens)
                        output_tokens = int(usage.get("completion_tokens") or output_tokens)

                    delta_text, chunk_finish_reason = _extract_text_delta(payload)
                    if chunk_finish_reason is not None:
                        finish_reason = chunk_finish_reason
                    if delta_text:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        nonempty_chunks += 1
                        text_parts.append(delta_text)

            ended = time.perf_counter()
            if first_token_at is None:
                raise RuntimeError("stream completed without any generated text")
            token_count_source = "usage"
            if output_tokens <= 0:
                # A fallback for servers that ignore stream_options.include_usage.
                output_tokens = nonempty_chunks
                token_count_source = "stream_chunks"
            e2e_ms = (ended - started) * 1000
            ttft_ms = (first_token_at - started) * 1000
            tpot_ms = (
                (e2e_ms - ttft_ms) / (output_tokens - 1)
                if output_tokens > 1
                else 0.0
            )
            return RequestResult(
                request_id=request.request_id,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                e2e_ms=e2e_ms,
                generated_text="".join(text_parts),
                finish_reason=finish_reason,
                token_count_source=token_count_source,
            )
        except Exception as exc:
            return RequestResult(
                request_id=request.request_id,
                success=False,
                e2e_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def fetch_metrics(self) -> str:
        async with self.session.get(self.config.resolved_metrics_url) as response:
            if response.status >= 400:
                body = await response.text()
                raise RuntimeError(f"metrics HTTP {response.status}: {body[:500]}")
            return await response.text()
