from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    request_id: int
    max_tokens: int
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RequestResult:
    request_id: int
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    generated_text: str = ""
    finish_reason: str | None = None
    token_count_source: str = "usage"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """Structured benchmark progress for CLI and future web adapters."""

    phase: str
    completed: int
    total: int
    successful: int
    failed: int
    elapsed_seconds: float
    request_throughput: float
    eta_seconds: float | None

    @property
    def progress_percent(self) -> float:
        return self.completed / self.total * 100 if self.total else 100.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["progress_percent"] = self.progress_percent
        return data


@dataclass(frozen=True, slots=True)
class SpecDecodeSnapshot:
    num_drafts: float = 0.0
    draft_tokens: float = 0.0
    accepted_tokens: float = 0.0
    accepted_tokens_per_position: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpecDecodeStats:
    available: bool
    num_drafts: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    acceptance_rate: float | None = None
    mean_acceptance_length: float | None = None
    position_acceptance_rates: list[float] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BenchmarkReport:
    summary: dict[str, Any]
    requests: list[RequestResult]
    spec_decode: SpecDecodeStats
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.summary)
        data["spec_decode"] = self.spec_decode.to_dict()
        data["warnings"] = list(self.warnings)
        return data
