from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable

from .models import (
    BenchmarkReport,
    BenchmarkRequest,
    ProgressUpdate,
    RequestResult,
    SpecDecodeSnapshot,
    SpecDecodeStats,
)
from .prometheus import diff_spec_decode, snapshot_from_prometheus
from .stats import summarize_requests


Generate = Callable[[BenchmarkRequest], Awaitable[RequestResult]]
FetchMetrics = Callable[[], Awaitable[str]]
ProgressCallback = Callable[[ProgressUpdate], None]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        generate: Generate,
        fetch_metrics: FetchMetrics | None,
        concurrency: int,
        dataset_name: str,
        progress_callback: ProgressCallback | None = None,
        progress_interval_seconds: float = 1.0,
    ):
        self.generate = generate
        self.fetch_metrics = fetch_metrics
        self.concurrency = concurrency
        self.dataset_name = dataset_name
        self.progress_callback = progress_callback
        self.progress_interval_seconds = progress_interval_seconds

    async def _snapshot(self) -> tuple[SpecDecodeSnapshot | None, str | None]:
        if self.fetch_metrics is None:
            return None, "Speculative metrics collection is disabled."
        try:
            return snapshot_from_prometheus(await self.fetch_metrics()), None
        except Exception as exc:
            return None, f"Could not read speculative metrics: {type(exc).__name__}: {exc}"

    async def _run_concurrent(
        self,
        requests: list[BenchmarkRequest],
        *,
        phase: str,
    ) -> list[RequestResult]:
        queue: asyncio.Queue[BenchmarkRequest] = asyncio.Queue()
        for request in requests:
            queue.put_nowait(request)
        results: list[RequestResult] = []
        total = len(requests)
        completed = successful = failed = 0
        started = time.perf_counter()

        def emit_progress() -> None:
            if self.progress_callback is None:
                return
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (total - completed) / rate if rate > 0 and completed < total else None
            self.progress_callback(
                ProgressUpdate(
                    phase=phase,
                    completed=completed,
                    total=total,
                    successful=successful,
                    failed=failed,
                    elapsed_seconds=elapsed,
                    request_throughput=rate,
                    eta_seconds=eta,
                )
            )

        async def ticker() -> None:
            while completed < total:
                await asyncio.sleep(self.progress_interval_seconds)
                emit_progress()

        async def worker() -> None:
            nonlocal completed, successful, failed
            while True:
                try:
                    request = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    result = await self.generate(request)
                    results.append(result)
                    completed += 1
                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                    emit_progress()
                finally:
                    queue.task_done()

        emit_progress()
        ticker_task = (
            asyncio.create_task(ticker())
            if self.progress_callback is not None and total > 0
            else None
        )
        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self.concurrency, len(requests)))
        ]
        try:
            await asyncio.gather(*workers)
        finally:
            if ticker_task is not None:
                ticker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ticker_task
        return sorted(results, key=lambda result: result.request_id)

    async def run(
        self,
        requests: list[BenchmarkRequest],
        *,
        warmup_requests: int,
    ) -> BenchmarkReport:
        warnings: list[str] = []
        if warmup_requests:
            warmups = [
                BenchmarkRequest(
                    request_id=-(index + 1),
                    max_tokens=requests[index % len(requests)].max_tokens,
                    prompt=requests[index % len(requests)].prompt,
                    messages=requests[index % len(requests)].messages,
                    metadata={"warmup": True},
                )
                for index in range(warmup_requests)
            ]
            await self._run_concurrent(warmups, phase="warmup")

        before, before_warning = await self._snapshot()
        if before_warning:
            warnings.append(before_warning)

        started = time.perf_counter()
        results = await self._run_concurrent(requests, phase="benchmark")
        duration = time.perf_counter() - started

        after, after_warning = await self._snapshot()
        if after_warning and after_warning not in warnings:
            warnings.append(after_warning)

        if before is not None and after is not None:
            spec_decode = diff_spec_decode(before, after)
            if spec_decode.warning:
                warnings.append(spec_decode.warning)
        else:
            spec_decode = SpecDecodeStats(
                available=False,
                warning="Speculative metrics were not available for this run.",
            )

        if any(
            result.success and result.token_count_source != "usage"
            for result in results
        ):
            warnings.append(
                "The server did not return streaming usage for one or more requests; "
                "output token counts and TPOT used stream chunk counts as a fallback."
            )

        summary = summarize_requests(
            results,
            duration_seconds=duration,
            dataset=self.dataset_name,
            concurrency=self.concurrency,
        )
        return BenchmarkReport(
            summary=summary,
            requests=results,
            spec_decode=spec_decode,
            warnings=warnings,
        )
