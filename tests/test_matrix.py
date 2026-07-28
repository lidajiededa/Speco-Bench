import csv
import tempfile
import unittest
from pathlib import Path

from speco_bench.matrix import (
    DatasetSpec,
    parse_concurrencies,
    report_to_csv_row,
    resolve_datasets,
    write_csv,
)
from speco_bench.models import BenchmarkReport, SpecDecodeStats


class MatrixTests(unittest.TestCase):
    def test_parse_concurrencies_accepts_spaces_and_commas(self):
        self.assertEqual(parse_concurrencies(["1,2", "4", "2"]), [1, 2, 4])

    def test_resolve_dataset_names_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            named_path = root / "mt_bench" / "question.jsonl"
            named_path.parent.mkdir()
            named_path.write_text("{}\n", encoding="utf-8")
            direct_path = root / "custom.jsonl"
            direct_path.write_text("{}\n", encoding="utf-8")

            datasets = resolve_datasets(
                ["mt-bench", str(direct_path)],
                root,
            )

        self.assertEqual(
            [dataset.name for dataset in datasets],
            ["mt_bench", "custom"],
        )
        self.assertEqual(datasets[0].path, named_path.resolve())
        self.assertEqual(datasets[1].path, direct_path.resolve())

    def test_report_row_and_csv(self):
        report = BenchmarkReport(
            summary={
                "concurrency": 4,
                "total_requests": 2,
                "successful_requests": 2,
                "failed_requests": 0,
                "benchmark_duration_seconds": 2.0,
                "total_input_tokens": 20,
                "total_output_tokens": 10,
                "request_throughput": 1.0,
                "output_throughput": 5.0,
                "ttft_ms": {"mean": 1, "p50": 1, "p90": 2, "p99": 3},
                "tpot_ms": {"mean": 4, "p50": 4, "p90": 5, "p99": 6},
                "e2e_ms": {"mean": 7, "p50": 7, "p90": 8, "p99": 9},
            },
            requests=[],
            spec_decode=SpecDecodeStats(
                available=True,
                num_drafts=2,
                draft_tokens=4,
                accepted_tokens=3,
                acceptance_rate=0.75,
                mean_acceptance_length=2.5,
                position_acceptance_rates=[1.0, 0.5],
            ),
        )
        dataset = DatasetSpec("gsm8k", Path("/tmp/gsm8k.jsonl"))
        row = report_to_csv_row(
            dataset,
            report,
            summary_path=Path("/tmp/summary.json"),
            requests_path=Path("/tmp/requests.jsonl"),
        )

        self.assertEqual(row["total_token_throughput_tok_s"], 15.0)
        self.assertEqual(row["draft_token_acceptance_rate"], 0.75)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "matrix.csv"
            write_csv([row], destination)
            with destination.open(encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle))
        self.assertEqual(saved[0]["dataset"], "gsm8k")
        self.assertEqual(saved[0]["concurrency"], "4")


if __name__ == "__main__":
    unittest.main()
