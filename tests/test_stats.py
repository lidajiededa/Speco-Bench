import unittest

from speco_bench.models import RequestResult
from speco_bench.stats import percentile, summarize_requests


class StatsTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 50), 5)
        self.assertEqual(percentile([0, 10], 90), 9)

    def test_summary_matches_core_vllm_formulas(self):
        results = [
            RequestResult(
                request_id=0,
                success=True,
                input_tokens=10,
                output_tokens=5,
                ttft_ms=100,
                tpot_ms=20,
                e2e_ms=180,
            ),
            RequestResult(request_id=1, success=False, error="failed"),
        ]
        summary = summarize_requests(
            results,
            duration_seconds=2,
            dataset="data",
            concurrency=1,
        )
        self.assertEqual(summary["successful_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["request_throughput"], 0.5)
        self.assertEqual(summary["output_throughput"], 2.5)
        self.assertEqual(summary["tpot_ms"]["mean"], 20)


if __name__ == "__main__":
    unittest.main()

