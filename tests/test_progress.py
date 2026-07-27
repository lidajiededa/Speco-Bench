import io
import unittest

from speco_bench.models import ProgressUpdate
from speco_bench.progress import TerminalProgress


class ProgressTests(unittest.TestCase):
    def test_non_tty_progress_is_plain_text(self):
        stream = io.StringIO()
        progress = TerminalProgress(stream=stream)
        progress(
            ProgressUpdate(
                phase="benchmark",
                completed=10,
                total=10,
                successful=9,
                failed=1,
                elapsed_seconds=5.0,
                request_throughput=2.0,
                eta_seconds=None,
            )
        )

        output = stream.getvalue()
        self.assertIn("Benchmark", output)
        self.assertIn("100.00%", output)
        self.assertIn("10/10", output)
        self.assertIn("ok=9, fail=1", output)
        self.assertNotIn("\r", output)
        self.assertNotIn("\033", output)


if __name__ == "__main__":
    unittest.main()
