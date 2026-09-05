"""Regression tests for the workspace test-population gate (#2705, #2818)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "nextest_summary.py"
_SPEC = importlib.util.spec_from_file_location("nextest_summary", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
nextest_summary = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = nextest_summary
_SPEC.loader.exec_module(nextest_summary)


def _archive(runnable: int, selected: int | None = None):
    return nextest_summary.ArchivePopulation(
        selected=runnable if selected is None else selected, runnable=runnable
    )


def _write_shards(tmp_path: Path, counts: list[int], *, skipped: int | None = None) -> list[Path]:
    """One log per shard, each with the terminal summary nextest emits."""
    total = sum(counts)
    paths = []
    for index, count in enumerate(counts, start=1):
        path = tmp_path / f"shard-{index}.log"
        tail = "" if skipped is None else f", {skipped if skipped is not None else 0} skipped"
        if skipped is None:
            tail = f", {total - count} skipped"
        path.write_text(
            f"Summary [1.23s] {count} tests run: {count - 1} passed, 1 failed{tail}\n"
        )
        paths.append(path)
    return paths


def _listing(cases: list[tuple[str, bool]], *, mismatched: int = 0) -> str:
    """A `cargo nextest list --message-format json` document."""
    testcases = {
        name: {"kind": "test", "ignored": ignored, "filter-match": {"status": "matches"}}
        for name, ignored in cases
    }
    for i in range(mismatched):
        testcases[f"filtered_out_{i}"] = {
            "kind": "test",
            "ignored": False,
            "filter-match": {"status": "mismatch", "reason": "expression"},
        }
    return json.dumps(
        {
            "test-count": len(testcases),
            "rust-suites": {"gam-linalg": {"binary-id": "gam-linalg", "testcases": testcases}},
        }
    )


# --- the population the archive says should run -----------------------------


def test_listing_counts_runnable_apart_from_ignored_and_filtered() -> None:
    listing = _listing([("a", False), ("b", False), ("c", True)], mismatched=4)

    population = nextest_summary.read_archive_population(listing)

    assert (population.selected, population.runnable) == (3, 2)


def test_listing_without_a_test_binary_is_not_a_population() -> None:
    with pytest.raises(nextest_summary.CoverageError, match="names no test binary"):
        nextest_summary.read_archive_population(json.dumps({"rust-suites": {}}))


def test_listing_that_selects_nothing_is_not_a_population() -> None:
    with pytest.raises(nextest_summary.CoverageError, match="selects no test at all"):
        nextest_summary.read_archive_population(_listing([], mismatched=3))


def test_truncated_listing_is_not_a_population() -> None:
    with pytest.raises(nextest_summary.CoverageError, match="not JSON"):
        nextest_summary.read_archive_population('{"rust-suites": {"gam')


# --- the population the shards actually executed ----------------------------


def test_complete_red_shards_are_a_measured_population(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [60] * 10)
    paths[0].write_text(
        "\x1b[31mSummary [1.23s] 60 tests run: 59 passed, 1 failed, 540 skipped\x1b[0m\n"
    )

    measured = nextest_summary.measure_workspace_test_population(
        paths, expected_shards=10, archive=_archive(600)
    )

    assert measured.total == 600
    assert measured.expected == 600
    assert measured.per_shard[0] == (1, 60)


def test_one_collapsed_shard_is_not_measured(tmp_path: Path) -> None:
    """The failure a floor of any size cannot catch: nine shards of ten report.

    This is the realistic loss (an OOM-killed runner, a truncated log that still
    carried a summary), and it is a tenth of the surface -- far above any floor
    that would ever be written down and far below a majority.
    """
    paths = _write_shards(tmp_path, [60] * 9 + [0])

    with pytest.raises(
        nextest_summary.CoverageError,
        match=r"ran 540 of the 600 runnable tests.*60 were never accounted for",
    ):
        nextest_summary.measure_workspace_test_population(
            paths, expected_shards=10, archive=_archive(600)
        )


def test_a_population_far_above_the_old_floor_is_still_refused(tmp_path: Path) -> None:
    """`MIN_WORKSPACE_TESTS = 558` passed this exact shape (#2818).

    9,458 tests ran against a workspace whose archive holds 9,460 runnable ones.
    A floor of 558 reports MEASURED; the derived expectation names the two.
    """
    paths = _write_shards(tmp_path, [946] * 9 + [944])

    with pytest.raises(nextest_summary.CoverageError, match=r"ran 9458 of the 9460 runnable"):
        nextest_summary.measure_workspace_test_population(
            paths, expected_shards=10, archive=_archive(9460)
        )


def test_a_shard_running_more_than_the_archive_holds_is_not_measured(tmp_path: Path) -> None:
    """A filterset that drifted between the listing and the run reads as a surplus."""
    paths = _write_shards(tmp_path, [60] * 10)

    with pytest.raises(nextest_summary.CoverageError, match="run more than once"):
        nextest_summary.measure_workspace_test_population(
            paths, expected_shards=10, archive=_archive(590)
        )


def test_missing_terminal_summary_is_not_measured(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [60] * 10)
    paths[-1].write_text("nextest process was interrupted before its summary\n")

    with pytest.raises(nextest_summary.CoverageError, match="0 terminal nextest summaries"):
        nextest_summary.measure_workspace_test_population(
            paths, expected_shards=10, archive=_archive(600)
        )


def test_duplicate_terminal_summary_is_not_measured(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [60] * 10)
    paths[-1].write_text(
        "Summary [1.0s] 60 tests run: 60 passed\n"
        "Summary [1.1s] 60 tests run: 60 passed\n"
    )

    with pytest.raises(nextest_summary.CoverageError, match="2 terminal nextest summaries"):
        nextest_summary.measure_workspace_test_population(
            paths, expected_shards=10, archive=_archive(600)
        )


def test_incomplete_numbered_shard_set_is_not_measured(tmp_path: Path) -> None:
    paths = _write_shards(tmp_path, [70] * 9)

    with pytest.raises(nextest_summary.CoverageError, match=r"missing=\[10\]"):
        nextest_summary.measure_workspace_test_population(
            paths, expected_shards=10, archive=_archive(630)
        )


def test_a_summary_without_a_skipped_clause_still_parses(tmp_path: Path) -> None:
    """A shard that ran everything it was given prints no `skipped` clause."""
    path = tmp_path / "shard-1.log"
    path.write_text("Summary [1.0s] 42 tests run: 42 passed\n")

    measured = nextest_summary.measure_workspace_test_population(
        [path], expected_shards=1, archive=_archive(42)
    )

    assert measured.total == 42


# --- the reducer the build job runs -----------------------------------------


def test_reducer_writes_the_two_counts(tmp_path: Path) -> None:
    listing = tmp_path / "archive-listing.json"
    listing.write_text(_listing([("a", False), ("b", True)], mismatched=1))
    output = tmp_path / "archive-population.json"

    assert nextest_summary.main(["--reduce-listing", str(listing), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == {"selected": 2, "runnable": 1}


def test_reducer_refuses_a_listing_it_cannot_read(tmp_path: Path) -> None:
    listing = tmp_path / "archive-listing.json"
    listing.write_text("")
    output = tmp_path / "archive-population.json"

    assert nextest_summary.main(["--reduce-listing", str(listing), "--output", str(output)]) == 1
    assert not output.exists()
