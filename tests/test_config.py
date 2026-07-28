import unittest
from pathlib import Path

from speco_bench.config import BenchmarkConfig


class ConfigTests(unittest.TestCase):
    def test_custom_dataset_requires_path(self):
        config = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=None,
            output_dir=Path("results"),
        )
        with self.assertRaisesRegex(ValueError, "dataset_path is required"):
            config.validate()

    def test_random_dataset_rejects_path(self):
        config = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=Path("dataset.jsonl"),
            output_dir=Path("results"),
            dataset_name="random",
        )
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            config.validate()

    def test_random_dataset_accepts_fixed_lengths(self):
        config = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=None,
            output_dir=Path("results"),
            dataset_name="random",
            random_input_len=1024,
            random_output_len=128,
        )
        config.validate()


if __name__ == "__main__":
    unittest.main()
