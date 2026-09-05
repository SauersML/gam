"""Certify the population executed by partitioned nextest shard logs.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
`scripts/test_census.py` answers "were tests deleted from source". This answers
a different question with a different blind spot: "did the shards actually RUN
what was there". A test can be present in every source tree, archived into
`nx.tar.zst`, assigned to a partition, and then never execute because that
shard's runner was OOM-killed or its log was truncated -- and every source-side
gate in this repository is byte-identical between that run and a complete one.

WHY THERE IS NO FLOOR HERE ANY MORE
This module used to certify the population against `MIN_WORKSPACE_TESTS = 558`,
a constant whose own comment named its provenance: "the last complete #2705
regression census executed 558 tests". That is the population of ONE test
binary, used as the floor for the WHOLE workspace. Measured against it, run
33946053992 reported `Workspace tests run: 9458 (required floor: 558)` -- 8,900
tests could have failed to execute and the ledger would still have printed
MEASURED.

Raising the number would not have fixed it. A floor of any size only catches
losing MOST of the surface, while the realistic failure loses a tenth: one
shard of ten dies and nine still report. And a hand-supplied constant goes stale
on the next test added, so the next person to see it red lowers it.

So the expectation is DERIVED instead, from the artifact the run already builds.
`cargo nextest archive` is followed by `cargo nextest list --message-format
json` under the identical profile and filterset, which is the exact set of tests
that should execute. The assertion is then the strong one and has no constant in
it: **every runnable test the archive contains was accounted for by a shard**.
`count:i/N` partitions the selected list, so the shards' run counts sum to the
listed runnable population exactly -- one shard losing a tenth of the surface is
a 10% shortfall, and it is named.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SHARD = re.compile(r"^shard-(\d+)\.log$")
_SUMMARY = re.compile(
    r"^\s*Summary\s+\[[^\]]+\]\s+([0-9][0-9,]*)\s+tests?\s+run:(?P<tail>.*)$"
)
_SKIPPED = re.compile(r"([0-9][0-9,]*)\s+skipped")


class CoverageError(ValueError):
    """The logs cannot certify that the required test population ran."""


@dataclass(frozen=True)
class ArchivePopulation:
    """What the built archive says should run, read from its own listing."""

    selected: int
    runnable: int


@dataclass(frozen=True)
class WorkspaceTestPopulation:
    total: int
    expected: int
    per_shard: tuple[tuple[int, int], ...]


def _count(text: str) -> int:
    return int(text.replace(",", ""))


def read_archive_population(listing: str) -> ArchivePopulation:
    """Reduce `cargo nextest list --message-format json` to its two counts.

    `selected` is every test the shard filterset admits; `runnable` is the
    subset nextest will actually execute, which is what the shards' summed run
    counts have to reproduce. An `#[ignore]`d test is excluded from `runnable`
    whether nextest reports it as a filter mismatch or as a selected-but-ignored
    case, because both spellings have been observed across versions.
    """
    try:
        document = json.loads(listing)
    except json.JSONDecodeError as exc:
        raise CoverageError(f"the nextest listing is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise CoverageError("the nextest listing is not a JSON object")
    suites = document.get("rust-suites")
    if not isinstance(suites, dict) or not suites:
        raise CoverageError("the nextest listing names no test binary")
    selected = 0
    runnable = 0
    for binary_id, suite in suites.items():
        if not isinstance(suite, dict) or not isinstance(suite.get("testcases"), dict):
            raise CoverageError(f"the nextest listing has no testcases for {binary_id!r}")
        for case in suite["testcases"].values():
            if not isinstance(case, dict):
                raise CoverageError(f"the nextest listing has a malformed testcase in {binary_id!r}")
            match = case.get("filter-match")
            if isinstance(match, dict) and match.get("status") != "matches":
                continue
            selected += 1
            if not case.get("ignored"):
                runnable += 1
    if selected < 1:
        raise CoverageError("the nextest listing selects no test at all")
    return ArchivePopulation(selected=selected, runnable=runnable)


def _summary_counts(path: Path) -> tuple[int, int]:
    """`(tests run, tests skipped)` from the log's single terminal summary."""
    matches: list[tuple[int, int]] = []
    for raw in path.read_text(errors="replace").splitlines():
        match = _SUMMARY.match(_ANSI.sub("", raw))
        if match:
            skipped = _SKIPPED.search(match.group("tail"))
            matches.append((_count(match.group(1)), _count(skipped.group(1)) if skipped else 0))
    if len(matches) != 1:
        raise CoverageError(
            f"{path.name} contains {len(matches)} terminal nextest summaries; expected exactly 1"
        )
    return matches[0]


def measure_workspace_test_population(
    paths: Iterable[str | Path],
    *,
    expected_shards: int,
    archive: ArchivePopulation,
) -> WorkspaceTestPopulation:
    """Return a certified population or raise ``CoverageError``.

    Certification requires the exact numbered shard set, one terminal nextest
    summary per log, and a summed run count equal to the archive's own runnable
    population. A failing test run still has a terminal summary and is a
    measurement; a partial or collapsed run is not.
    """

    if expected_shards < 1:
        raise CoverageError(f"invalid planned shard count {expected_shards}")

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

    counts = {index: _summary_counts(by_index[index]) for index in sorted(by_index)}
    per_shard = tuple((index, run) for index, (run, _) in counts.items())
    total = sum(run for _, run in per_shard)
    if total != archive.runnable:
        # Name the shards, not just the shortfall: `count:i/N` gives every shard
        # a comparable slice, so a single collapsed one is visible by inspection
        # and a filterset that drifted between listing and run is not.
        detail = ", ".join(
            f"shard {index}: {run} run / {skipped} skipped"
            for index, (run, skipped) in counts.items()
        )
        raise CoverageError(
            f"the shards ran {total} of the {archive.runnable} runnable tests the archive "
            f"listing contains ({archive.selected} selected by the shard filterset); "
            f"{abs(archive.runnable - total)} were "
            f"{'never accounted for by any shard' if total < archive.runnable else 'run more than once'}"
            f" [{detail}]"
        )
    return WorkspaceTestPopulation(total=total, expected=archive.runnable, per_shard=per_shard)


def main(argv: list[str] | None = None) -> int:
    """Reduce a nextest listing to the two counts the aggregate job consumes.

    The reduction lives here rather than in a shell one-liner in the workflow so
    that it is the same code path `tests/test_ci_nextest_summary.py` exercises.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduce-listing", required=True, type=Path,
                        help="cargo nextest list --message-format json output")
    parser.add_argument("--output", required=True, type=Path,
                        help="where to write the reduced population JSON")
    args = parser.parse_args(argv)
    try:
        population = read_archive_population(args.reduce_listing.read_text(errors="replace"))
    except (CoverageError, OSError) as error:
        print(f"NEXTEST LISTING NOT REDUCED: {error}", file=sys.stderr)
        return 1
    args.output.write_text(
        json.dumps({"selected": population.selected, "runnable": population.runnable}) + "\n"
    )
    print(
        f"archive listing: {population.selected} tests selected by the shard filterset, "
        f"{population.runnable} of them runnable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
