import asyncio
import unittest

from speco_bench.models import BenchmarkRequest, RequestResult
from speco_bench.runner import BenchmarkRunner


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_concurrency_and_summary(self):
        active = 0
        max_active = 0

        async def generate(request):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return RequestResult(
                request_id=request.request_id,
                success=True,
                input_tokens=5,
                output_tokens=3,
                ttft_ms=2,
                tpot_ms=1,
                e2e_ms=4,
            )

        metrics_values = iter(
            [
                "vllm:spec_decode_num_drafts_total 1\n",
                (
                    "vllm:spec_decode_num_drafts_total 5\n"
                    "vllm:spec_decode_num_draft_tokens_total 12\n"
                    "vllm:spec_decode_num_accepted_tokens_total 8\n"
                ),
            ]
        )

        async def fetch_metrics():
            return next(metrics_values)

        runner = BenchmarkRunner(
            generate=generate,
            fetch_metrics=fetch_metrics,
            concurrency=2,
            dataset_name="test",
        )
        requests = [
            BenchmarkRequest(request_id=index, prompt="x", max_tokens=3)
            for index in range(5)
        ]
        report = await runner.run(requests, warmup_requests=0)
        self.assertEqual(max_active, 2)
        self.assertEqual(report.summary["successful_requests"], 5)
        self.assertEqual(report.summary["total_output_tokens"], 15)
        self.assertTrue(report.spec_decode.available)
        self.assertEqual(report.spec_decode.num_drafts, 4)

    async def test_emits_structured_progress(self):
        updates = []

        async def generate(request):
            await asyncio.sleep(0.001)
            return RequestResult(
                request_id=request.request_id,
                success=request.request_id != 1,
            )

        runner = BenchmarkRunner(
            generate=generate,
            fetch_metrics=None,
            concurrency=2,
            dataset_name="test",
            progress_callback=updates.append,
            progress_interval_seconds=0.01,
        )
        requests = [
            BenchmarkRequest(request_id=index, prompt="x", max_tokens=3)
            for index in range(3)
        ]
        await runner.run(requests, warmup_requests=0)

        self.assertEqual(updates[0].phase, "benchmark")
        self.assertEqual(updates[0].completed, 0)
        self.assertEqual(updates[-1].completed, 3)
        self.assertEqual(updates[-1].successful, 2)
        self.assertEqual(updates[-1].failed, 1)
        self.assertEqual(updates[-1].progress_percent, 100.0)
        self.assertIsNone(updates[-1].eta_seconds)


if __name__ == "__main__":
    unittest.main()
