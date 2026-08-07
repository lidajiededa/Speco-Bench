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

    def test_random_dataset_accepts_multimodal_dimensions(self):
        config = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=None,
            output_dir=Path("results"),
            dataset_name="random",
            random_image_width=1280,
            random_image_height=720,
            random_images_per_prompt=2,
        )
        config.validate()

    def test_multimodal_random_requires_both_dimensions_and_chat(self):
        missing_height = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=None,
            output_dir=Path("results"),
            dataset_name="random",
            random_image_width=640,
        )
        with self.assertRaisesRegex(ValueError, "must be set together"):
            missing_height.validate()

        completions = BenchmarkConfig(
            base_url="http://localhost:8000",
            model="model",
            dataset_path=None,
            output_dir=Path("results"),
            dataset_name="random",
            endpoint_type="completions",
            random_image_width=640,
            random_image_height=480,
        )
        with self.assertRaisesRegex(ValueError, "chat endpoint"):
            completions.validate()


if __name__ == "__main__":
    unittest.main()
