"""Certify the population executed by partitioned nextest shard logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


# The last complete #2705 regression census executed 558 tests. The workspace
# shards contain that regression binary plus the rest of the Rust test surface,
# so a smaller aggregate population is necessarily an incomplete measurement.
MIN_WORKSPACE_TESTS = 558

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SHARD = re.compile(r"^shard-(\d+)\.log$")
_SUMMARY = re.compile(
    r"^\s*Summary\s+\[[^\]]+\]\s+([0-9][0-9,]*)\s+tests?\s+run:"
)


class CoverageError(ValueError):
    """The logs cannot certify that the required test population ran."""


@dataclass(frozen=True)
class WorkspaceTestPopulation:
    total: int
    per_shard: tuple[tuple[int, int], ...]


def _summary_count(path: Path) -> int:
    matches = []
    for raw in path.read_text(errors="replace").splitlines():
        match = _SUMMARY.match(_ANSI.sub("", raw))
        if match:
            matches.append(int(match.group(1).replace(",", "")))
    if len(matches) != 1:
        raise CoverageError(
            f"{path.name} contains {len(matches)} terminal nextest summaries; expected exactly 1"
        )
    return matches[0]


def measure_workspace_test_population(
    paths: Iterable[str | Path],
    *,
    expected_shards: int,
    minimum: int = MIN_WORKSPACE_TESTS,
) -> WorkspaceTestPopulation:
    """Return a certified population or raise ``CoverageError``.

    Certification requires the exact numbered shard set, one terminal nextest
    summary per log, and a summed population at or above the checked-in floor.
    A failing test run still has a terminal summary and is a measurement; a
    partial or collapsed run is not.
    """

    if expected_shards < 1:
        raise CoverageError(f"invalid planned shard count {expected_shards}")
    if minimum < 1:
        raise CoverageError(f"invalid workspace test floor {minimum}")

    by_index: dict[int, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        match = _SHARD.match(path.name)
        if not match:
            raise CoverageError(f"unexpected workspace shard log name {path.name!r}")
        index = int(match.group(1))
        if index in by_index:
            raise CoverageError(f"duplicate workspace shard log for shard {index}")
        by_index[index] = path

    expected = set(range(1, expected_shards + 1))
    actual = set(by_index)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CoverageError(
            f"workspace shard set differs from 1..{expected_shards}: "
            f"missing={missing}, extra={extra}"
        )

    per_shard = tuple(
        (index, _summary_count(by_index[index])) for index in sorted(by_index)
    )
    total = sum(count for _, count in per_shard)
    if total < minimum:
        raise CoverageError(
            f"workspace shards ran {total} tests, below the required floor {minimum}"
        )
    return WorkspaceTestPopulation(total=total, per_shard=per_shard)
