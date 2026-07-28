import unittest
from pathlib import Path

from speco_bench.client import OpenAIStreamingClient, _extract_text_delta
from speco_bench.config import BenchmarkConfig
from speco_bench.models import BenchmarkRequest


class ClientTests(unittest.TestCase):
    def test_extracts_current_reasoning_delta(self):
        text, finish_reason = _extract_text_delta(
            {
                "choices": [
                    {"delta": {"reasoning": "thinking"}, "finish_reason": None}
                ]
            }
        )

        self.assertEqual(text, "thinking")
        self.assertIsNone(finish_reason)

    def test_extracts_legacy_reasoning_content_delta(self):
        text, _ = _extract_text_delta(
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": "legacy thinking"},
                        "finish_reason": None,
                    }
                ]
            }
        )

        self.assertEqual(text, "legacy thinking")

    def test_preserves_reasoning_and_content_from_same_delta(self):
        text, finish_reason = _extract_text_delta(
            {
                "choices": [
                    {
                        "delta": {"reasoning": "thinking", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

        self.assertEqual(text, "thinkinganswer")
        self.assertEqual(finish_reason, "stop")

    def test_does_not_duplicate_reasoning_aliases(self):
        text, _ = _extract_text_delta(
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning": "thinking",
                            "reasoning_content": "thinking",
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )

        self.assertEqual(text, "thinking")

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
