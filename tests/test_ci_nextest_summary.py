"""Regression tests for the #2705 nextest population gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "nextest_summary.py"
_SPEC = importlib.util.spec_from_file_location("nextest_summary", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
nextest_summary = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = nextest_summary
_SPEC.loader.exec_module(nextest_summary)


def _write_shards(tmp_path: Path, counts: list[int]) -> list[Path]:
    paths = []
    for index, count in enumerate(counts, start=1):
        path = tmp_path / f"shard-{index}.log"
        path.write_text(
            f"Summary [1.23s] {count} tests run: {count - 1} passed, 1 failed\n"
        )
        paths.append(path)
    return paths


def test_complete_red_shards_are_a_measured_population(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [60] * 10)
    paths[0].write_text(
        "\x1b[31mSummary [1.23s] 60 tests run: 59 passed, 1 failed\x1b[0m\n"
    )

    measured = nextest_summary.measure_workspace_test_population(
        paths, expected_shards=10
    )

    assert measured.total == 600
    assert measured.per_shard[0] == (1, 60)


def test_missing_terminal_summary_is_not_measured(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [60] * 10)
    paths[-1].write_text("nextest process was interrupted before its summary\n")

    with pytest.raises(nextest_summary.CoverageError, match="0 terminal nextest summaries"):
        nextest_summary.measure_workspace_test_population(paths, expected_shards=10)


def test_duplicate_terminal_summary_is_not_measured(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [60] * 10)
    paths[-1].write_text(
        "Summary [1.0s] 60 tests run: 60 passed\n"
        "Summary [1.1s] 60 tests run: 60 passed\n"
    )

    with pytest.raises(nextest_summary.CoverageError, match="2 terminal nextest summaries"):
        nextest_summary.measure_workspace_test_population(paths, expected_shards=10)


def test_incomplete_numbered_shard_set_is_not_measured(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [70] * 9)

    with pytest.raises(nextest_summary.CoverageError, match=r"missing=\[10\]"):
        nextest_summary.measure_workspace_test_population(paths, expected_shards=10)


def test_population_below_checked_in_floor_is_not_measured(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [55] * 10)

    with pytest.raises(nextest_summary.CoverageError, match="550 tests, below.*558"):
        nextest_summary.measure_workspace_test_population(paths, expected_shards=10)
