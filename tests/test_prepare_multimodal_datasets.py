import unittest

from prepare_multimodal_datasets import _build_prompt_and_metadata


class PrepareMultimodalDatasetTests(unittest.TestCase):
    def test_builds_ai2d_multiple_choice_prompt(self):
        prompt, images, metadata = _build_prompt_and_metadata(
            "ai2d",
            {
                "question": "Which label marks the nucleus?",
                "options": ["A1", "B2"],
                "answer": "A",
                "image": "diagram",
            },
            source_index=4,
        )

        self.assertIn("A. A1", prompt)
        self.assertIn("B. B2", prompt)
        self.assertEqual(images, ["diagram"])
        self.assertEqual(metadata["reference"], "A")

    def test_builds_mmmu_multi_image_prompt(self):
        prompt, images, metadata = _build_prompt_and_metadata(
            "mmmu",
            {
                "id": "sample-1",
                "question": "Compare <image 1> with <image 2>.",
                "options": "['first', 'second']",
                "answer": "B",
                "question_type": "multiple-choice",
                "image_1": "one",
                "image_2": "two",
                "image_3": None,
            },
            source_index=0,
            subject="physics",
        )

        self.assertIn("[Image 1]", prompt)
        self.assertIn("[Image 2]", prompt)
        self.assertEqual(images, ["one", "two"])
        self.assertEqual(metadata["subject"], "physics")


if __name__ == "__main__":
    unittest.main()
