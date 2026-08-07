from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


EndpointType = Literal["chat", "completions"]
DatasetName = Literal["custom", "random"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration shared by CLI and future web/API adapters."""

    base_url: str
    model: str
    dataset_path: Path | None
    output_dir: Path
    dataset_name: DatasetName = "custom"
    endpoint_type: EndpointType = "chat"
    concurrency: int = 1
    num_prompts: int | None = None
    max_tokens: int | None = None
    random_input_len: int = 1024
    random_output_len: int = 128
    random_range_ratio: float = 0.0
    random_image_width: int | None = None
    random_image_height: int | None = None
    random_images_per_prompt: int = 1
    tokenizer: str | None = None
    trust_remote_code: bool = False
    ignore_eos: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    warmup_requests: int = 1
    request_timeout_seconds: float = 3600.0
    api_key: str | None = None
    metrics_url: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.dataset_name == "custom" and self.dataset_path is None:
            raise ValueError("dataset_path is required for a custom dataset")
        if self.dataset_name == "random" and self.dataset_path is not None:
            raise ValueError("dataset_path cannot be used with the random dataset")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.num_prompts is not None and self.num_prompts < 1:
            raise ValueError("num_prompts must be at least 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.dataset_name == "random" and self.max_tokens is not None:
            raise ValueError(
                "max_tokens cannot be used with the random dataset; "
                "use random_output_len"
            )
        if self.random_input_len < 1:
            raise ValueError("random_input_len must be at least 1")
        if self.random_output_len < 1:
            raise ValueError("random_output_len must be at least 1")
        if not 0 <= self.random_range_ratio < 1:
            raise ValueError("random_range_ratio must be in [0, 1)")
        image_dimensions = (self.random_image_width, self.random_image_height)
        if (image_dimensions[0] is None) != (image_dimensions[1] is None):
            raise ValueError(
                "random_image_width and random_image_height must be set together"
            )
        if any(value is not None and value < 1 for value in image_dimensions):
            raise ValueError("random image dimensions must be at least 1 pixel")
        if any(value is not None and value > 16384 for value in image_dimensions):
            raise ValueError("random image dimensions cannot exceed 16384 pixels")
        if self.random_images_per_prompt < 1:
            raise ValueError("random_images_per_prompt must be at least 1")
        if self.random_image_width is None and self.random_images_per_prompt != 1:
            raise ValueError(
                "random_images_per_prompt requires random image dimensions"
            )
        if (
            self.random_image_width is not None
            and self.random_image_height is not None
            and self.random_image_width
            * self.random_image_height
            * self.random_images_per_prompt
            > 100_000_000
        ):
            raise ValueError("random images cannot exceed 100 million pixels per prompt")
        if self.random_image_width is not None and self.dataset_name != "random":
            raise ValueError("random image dimensions require the random dataset")
        if self.random_image_width is not None and self.endpoint_type != "chat":
            raise ValueError("random multimodal requests require the chat endpoint")
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
