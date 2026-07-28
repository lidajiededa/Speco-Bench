import unittest
from pathlib import Path

from speco_bench.client import OpenAIStreamingClient
from speco_bench.config import BenchmarkConfig
from speco_bench.models import BenchmarkRequest


class ClientTests(unittest.TestCase):
    def test_ignore_eos_is_sent_when_enabled(self):
        config = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=Path("dataset.jsonl"),
            output_dir=Path("results"),
            ignore_eos=True,
        )
        payload = OpenAIStreamingClient(config)._payload(
            BenchmarkRequest(request_id=0, prompt="hello", max_tokens=32)
        )

        self.assertTrue(payload["ignore_eos"])
        self.assertEqual(payload["max_tokens"], 32)

    def test_ignore_eos_is_omitted_by_default(self):
        config = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=Path("dataset.jsonl"),
            output_dir=Path("results"),
        )
        payload = OpenAIStreamingClient(config)._payload(
            BenchmarkRequest(request_id=0, prompt="hello", max_tokens=32)
        )

        self.assertNotIn("ignore_eos", payload)


if __name__ == "__main__":
    unittest.main()
