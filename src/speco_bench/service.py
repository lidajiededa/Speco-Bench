from __future__ import annotations

from .client import OpenAIStreamingClient
from .config import BenchmarkConfig
from .dataset import load_dataset
from .models import BenchmarkReport
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
        requests = load_dataset(
            config.dataset_path,
            num_prompts=config.num_prompts,
            max_tokens=config.max_tokens,
            seed=config.seed,
        )
        async with OpenAIStreamingClient(config) as client:
            runner = BenchmarkRunner(
                generate=client.generate,
                fetch_metrics=client.fetch_metrics,
                concurrency=config.concurrency,
                dataset_name=str(config.dataset_path),
                progress_callback=progress_callback,
                progress_interval_seconds=progress_interval_seconds,
            )
            return await runner.run(
                requests,
                warmup_requests=config.warmup_requests,
            )
