import unittest

from speco_bench.prometheus import diff_spec_decode, snapshot_from_prometheus


BEFORE = """
# HELP vllm:spec_decode_num_drafts Spec rounds
vllm:spec_decode_num_drafts_total{engine="0"} 10
vllm:spec_decode_num_draft_tokens_total{engine="0"} 40
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 20
vllm:spec_decode_num_accepted_tokens_per_pos{engine="0",position="0"} 8
vllm:spec_decode_num_accepted_tokens_per_pos{engine="0",position="1"} 6
"""

AFTER = """
vllm:spec_decode_num_drafts_total{engine="0"} 20
vllm:spec_decode_num_draft_tokens_total{engine="0"} 80
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 45
vllm:spec_decode_num_accepted_tokens_per_pos{engine="0",position="0"} 17
vllm:spec_decode_num_accepted_tokens_per_pos{engine="0",position="1"} 13
"""


class PrometheusTests(unittest.TestCase):
    def test_snapshot_and_delta(self):
        stats = diff_spec_decode(
            snapshot_from_prometheus(BEFORE),
            snapshot_from_prometheus(AFTER),
        )
        self.assertTrue(stats.available)
        self.assertEqual(stats.num_drafts, 10)
        self.assertEqual(stats.draft_tokens, 40)
        self.assertEqual(stats.accepted_tokens, 25)
        self.assertAlmostEqual(stats.acceptance_rate, 0.625)
        self.assertAlmostEqual(stats.mean_acceptance_length, 3.5)
        self.assertEqual(stats.position_acceptance_rates, [0.9, 0.7])

    def test_no_drafts_is_unavailable(self):
        stats = diff_spec_decode(
            snapshot_from_prometheus(""),
            snapshot_from_prometheus(""),
        )
        self.assertFalse(stats.available)


if __name__ == "__main__":
    unittest.main()

