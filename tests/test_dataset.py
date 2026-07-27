import json
import tempfile
import unittest
from pathlib import Path

from speco_bench.dataset import DatasetError, load_dataset, prepare_dataset_file


class DatasetTests(unittest.TestCase):
    def test_load_and_cycle_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"prompt": "a", "max_tokens": 10}),
                        json.dumps({"prompt": "b", "max_tokens": 20}),
                    ]
                ),
                encoding="utf-8",
            )
            first = load_dataset(path, num_prompts=5, seed=7, max_tokens=30)
            second = load_dataset(path, num_prompts=5, seed=7, max_tokens=30)
            self.assertEqual([item.prompt for item in first], [item.prompt for item in second])
            self.assertEqual([item.request_id for item in first], list(range(5)))
            self.assertTrue(all(item.max_tokens == 30 for item in first))

    def test_invalid_record_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('{"prompt":"x","messages":[]}\n', encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_dataset(path)

    def test_prepare_sharegpt_removes_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            output = Path(directory) / "prepared.jsonl"
            source.write_text(
                json.dumps(
                    [
                        {
                            "conversations": [
                                {"from": "human", "value": "question"},
                                {"from": "gpt", "value": "reference answer"},
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            count = prepare_dataset_file(
                source,
                output,
                source_format="sharegpt",
                default_max_tokens=64,
            )
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(count, 1)
            self.assertEqual(record["messages"], [{"role": "user", "content": "question"}])
            self.assertEqual(record["max_tokens"], 64)


if __name__ == "__main__":
    unittest.main()

