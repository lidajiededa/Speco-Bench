from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


EndpointType = Literal["chat", "completions"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration shared by CLI and future web/API adapters."""

    base_url: str
    model: str
    dataset_path: Path
    output_dir: Path
    endpoint_type: EndpointType = "chat"
    concurrency: int = 1
    num_prompts: int | None = None
    max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    warmup_requests: int = 1
    request_timeout_seconds: float = 3600.0
    api_key: str | None = None
    metrics_url: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.num_prompts is not None and self.num_prompts < 1:
            raise ValueError("num_prompts must be at least 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests cannot be negative")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if not 0 <= self.temperature:
            raise ValueError("temperature cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    @property
    def resolved_metrics_url(self) -> str:
        if self.metrics_url:
            return self.metrics_url
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return f"{root}/metrics"

