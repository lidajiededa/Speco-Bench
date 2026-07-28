from __future__ import annotations

from .client import OpenAIStreamingClient
from .config import BenchmarkConfig
from .dataset import load_dataset
from .models import BenchmarkReport
from .random_dataset import generate_random_requests, load_tokenizer
from .runner import BenchmarkRunner, ProgressCallback


class BenchmarkService:
    """Application service reusable from CLI, FastAPI, or a job worker."""

    async def run(
        self,
        config: BenchmarkConfig,
        *,
        progress_callback: ProgressCallback | None = None,
        progress_interval_seconds: float = 1.0,
    ) -> BenchmarkReport:
        config.validate()
        if progress_interval_seconds <= 0:
            raise ValueError("progress_interval_seconds must be positive")
        if config.dataset_name == "random":
            tokenizer = load_tokenizer(
                config.tokenizer or config.model,
                trust_remote_code=config.trust_remote_code,
            )
            requests = generate_random_requests(
                tokenizer,
                num_prompts=config.num_prompts or 1000,
                input_length=config.random_input_len,
                output_length=config.random_output_len,
                range_ratio=config.random_range_ratio,
                seed=config.seed,
            )
            dataset_label = (
                f"random(input={config.random_input_len},"
                f"output={config.random_output_len},"
                f"range={config.random_range_ratio})"
            )
        else:
            assert config.dataset_path is not None
            requests = load_dataset(
                config.dataset_path,
                num_prompts=config.num_prompts,
                max_tokens=config.max_tokens,
                seed=config.seed,
            )
            dataset_label = str(config.dataset_path)
        async with OpenAIStreamingClient(config) as client:
            runner = BenchmarkRunner(
                generate=client.generate,
                fetch_metrics=client.fetch_metrics,
                concurrency=config.concurrency,
                dataset_name=dataset_label,
                progress_callback=progress_callback,
                progress_interval_seconds=progress_interval_seconds,
            )
            return await runner.run(
                requests,
                warmup_requests=config.warmup_requests,
            )
