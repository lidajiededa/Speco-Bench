import base64
import json
import tempfile
import unittest
from pathlib import Path

from speco_bench.dataset import (
    DatasetError,
    load_dataset,
    prepare_dataset_file,
    request_has_images,
)


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

    def test_loads_relative_images_with_prompt_shorthand(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            first_bytes = b"first-png"
            second_bytes = b"second-png"
            (images / "first.png").write_bytes(first_bytes)
            (images / "second.png").write_bytes(second_bytes)
            path = root / "question.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "Compare the images",
                        "images": ["images/first.png", "images/second.png"],
                        "max_tokens": 32,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            request = load_dataset(path)[0]

            self.assertIsNone(request.prompt)
            self.assertTrue(request_has_images(request))
            content = request.messages[0]["content"]
            self.assertEqual([part["type"] for part in content], ["image_url", "image_url", "text"])
            self.assertEqual(content[-1]["text"], "Compare the images")
            encoded = content[0]["image_url"]["url"].split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), first_bytes)
            self.assertNotIn("images", request.metadata)

    def test_preserves_remote_images_in_structured_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "https://example.com/image.jpg",
                                            "detail": "high",
                                        },
                                    },
                                    {"type": "text", "text": "What is shown?"},
                                ],
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            image_url = load_dataset(path)[0].messages[0]["content"][0]["image_url"]

            self.assertEqual(image_url["url"], "https://example.com/image.jpg")
            self.assertEqual(image_url["detail"], "high")

    def test_missing_local_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "question.jsonl"
            path.write_text(
                '{"prompt":"question","image":"missing.png"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetError, "image does not exist"):
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
