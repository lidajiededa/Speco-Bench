from __future__ import annotations

import re
from dataclasses import dataclass

from .models import SpecDecodeSnapshot, SpecDecodeStats


_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)"
    r"(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')

_DRAFT_NAMES = {
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_drafts_total",
}
_DRAFT_TOKEN_NAMES = {
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_draft_tokens_total",
}
_ACCEPTED_NAMES = {
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_total",
}
_POSITION_NAMES = {
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
}
_POSITION_KEYS = ("position", "pos", "index", "token_position")


@dataclass(frozen=True, slots=True)
class PrometheusSample:
    name: str
    labels: dict[str, str]
    value: float


def parse_prometheus_text(text: str) -> list[PrometheusSample]:
    samples: list[PrometheusSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        raw_value = match.group("value")
        if raw_value in {"NaN", "Inf", "-Inf"}:
            continue
        labels = {
            key: bytes(value, "utf-8").decode("unicode_escape")
            for key, value in _LABEL_RE.findall(match.group("labels") or "")
        }
        samples.append(
            PrometheusSample(
                name=match.group("name"),
                labels=labels,
                value=float(raw_value),
            )
        )
    return samples


def _position_from_labels(labels: dict[str, str]) -> int | None:
    for key in _POSITION_KEYS:
        value = labels.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return None
    return None


def snapshot_from_prometheus(text: str) -> SpecDecodeSnapshot:
    drafts = draft_tokens = accepted = 0.0
    positions: dict[int, float] = {}
    for sample in parse_prometheus_text(text):
        if sample.name in _DRAFT_NAMES:
            drafts += sample.value
        elif sample.name in _DRAFT_TOKEN_NAMES:
            draft_tokens += sample.value
        elif sample.name in _ACCEPTED_NAMES:
            accepted += sample.value
        elif sample.name in _POSITION_NAMES:
            position = _position_from_labels(sample.labels)
            if position is not None:
                positions[position] = positions.get(position, 0.0) + sample.value
    return SpecDecodeSnapshot(
        num_drafts=drafts,
        draft_tokens=draft_tokens,
        accepted_tokens=accepted,
        accepted_tokens_per_position=positions,
    )


def _counter_delta(before: float, after: float) -> float:
    # Treat a decrease as a server restart/counter reset.
    return max(0.0, after - before) if after >= before else max(0.0, after)


def diff_spec_decode(
    before: SpecDecodeSnapshot,
    after: SpecDecodeSnapshot,
) -> SpecDecodeStats:
    drafts = _counter_delta(before.num_drafts, after.num_drafts)
    draft_tokens = _counter_delta(before.draft_tokens, after.draft_tokens)
    accepted = _counter_delta(before.accepted_tokens, after.accepted_tokens)
    all_positions = sorted(
        set(before.accepted_tokens_per_position)
        | set(after.accepted_tokens_per_position)
    )
    position_counts = {
        position: _counter_delta(
            before.accepted_tokens_per_position.get(position, 0.0),
            after.accepted_tokens_per_position.get(position, 0.0),
        )
        for position in all_positions
    }

    if drafts <= 0:
        return SpecDecodeStats(
            available=False,
            warning=(
                "No speculative draft rounds were observed. Check that speculative "
                "decoding is enabled and /metrics exposes vLLM spec-decode counters."
            ),
        )

    max_position = max(position_counts, default=-1)
    rates = [
        position_counts.get(position, 0.0) / drafts
        for position in range(max_position + 1)
    ]
    return SpecDecodeStats(
        available=True,
        num_drafts=round(drafts),
        draft_tokens=round(draft_tokens),
        accepted_tokens=round(accepted),
        acceptance_rate=(accepted / draft_tokens) if draft_tokens > 0 else None,
        mean_acceptance_length=1.0 + accepted / drafts,
        position_acceptance_rates=rates,
    )

