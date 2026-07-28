import unittest

from speco_bench.random_dataset import generate_random_requests


class CharacterTokenizer:
    vocab_size = 27
    all_special_ids = [0]

    def get_vocab(self):
        return {"<bos>": 0, **{chr(96 + index): index for index in range(1, 27)}}

    def num_special_tokens_to_add(self, pair=False):
        return 1

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(96 + token_id) for token_id in token_ids if token_id)

    def encode(self, text, *, add_special_tokens):
        token_ids = [ord(character) - 96 for character in text]
        return ([0] if add_special_tokens else []) + token_ids


class RandomDatasetTests(unittest.TestCase):
    def test_generates_fixed_reproducible_lengths(self):
        tokenizer = CharacterTokenizer()
        first = generate_random_requests(
            tokenizer,
            num_prompts=5,
            input_length=16,
            output_length=8,
            range_ratio=0,
            seed=7,
        )
        second = generate_random_requests(
            tokenizer,
            num_prompts=5,
            input_length=16,
            output_length=8,
            range_ratio=0,
            seed=7,
        )

        self.assertEqual(
            [request.prompt for request in first],
            [request.prompt for request in second],
        )
        self.assertTrue(all(request.max_tokens == 8 for request in first))
        for request in first:
            self.assertEqual(
                len(tokenizer.encode(request.prompt, add_special_tokens=True)),
                16,
            )
            self.assertEqual(request.metadata["requested_input_tokens"], 16)

    def test_range_ratio_samples_within_bounds(self):
        requests = generate_random_requests(
            CharacterTokenizer(),
            num_prompts=100,
            input_length=20,
            output_length=10,
            range_ratio=0.2,
            seed=11,
        )

        self.assertTrue(
            all(
                16 <= request.metadata["requested_input_tokens"] <= 24
                for request in requests
            )
        )
        self.assertTrue(all(8 <= request.max_tokens <= 12 for request in requests))

    def test_rejects_invalid_range_ratio(self):
        with self.assertRaises(ValueError):
            generate_random_requests(
                CharacterTokenizer(),
                num_prompts=1,
                input_length=16,
                output_length=8,
                range_ratio=1,
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
