from __future__ import annotations

import sys
import time
from typing import TextIO

from .models import ProgressUpdate


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TerminalProgress:
    """Render progress without coupling the benchmark core to a terminal."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        min_interval_seconds: float = 0.2,
    ):
        self.stream = stream or sys.stderr
        self.min_interval_seconds = min_interval_seconds
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_phase: str | None = None
        self._last_rendered_at = 0.0
        self._line_open = False

    def __call__(self, update: ProgressUpdate) -> None:
        now = time.monotonic()
        phase_changed = update.phase != self._last_phase
        finished = update.completed >= update.total
        interval = self.min_interval_seconds if self.is_tty else max(
            5.0, self.min_interval_seconds
        )
        if (
            not phase_changed
            and not finished
            and now - self._last_rendered_at < interval
        ):
            return

        if phase_changed and self._line_open:
            self.stream.write("\n")

        label = "Warmup" if update.phase == "warmup" else "Benchmark"
        width = 24
        ratio = min(1.0, update.completed / update.total) if update.total else 1.0
        filled = round(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        line = (
            f"{label:<9} [{bar}] {update.progress_percent:6.2f}% "
            f"{update.completed}/{update.total} "
            f"(ok={update.successful}, fail={update.failed}) "
            f"elapsed={_format_duration(update.elapsed_seconds)} "
            f"rate={update.request_throughput:.2f} req/s "
            f"eta={_format_duration(update.eta_seconds)}"
        )

        if self.is_tty:
            self.stream.write(f"\r\033[2K{line}")
            self._line_open = not finished
            if finished:
                self.stream.write("\n")
        else:
            self.stream.write(f"{line}\n")
            self._line_open = False
        self.stream.flush()
        self._last_phase = update.phase
        self._last_rendered_at = now

    def close(self) -> None:
        if self._line_open:
            self.stream.write("\n")
            self.stream.flush()
            self._line_open = False
