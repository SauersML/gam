"""Named calibration plans: which cells, how many reps, which libraries.

A cell is ``(family, n, null)``: one family, one sample size, one null
structure (see ``worker.NULLS``). Every rep of a cell draws one dataset under
the null and one under its matched alternative from the rep's seed, and fits
every library in ``libs`` to both.

``timeout_s`` and ``memcap_mb`` are a HARNESS SAFETY NET ONLY, as in
``bench/pygam_compare``. They stop one runaway chunk from stalling the plan.
They are not a solver budget. A chunk that trips one is recorded with that
status, rep by rep, and the report counts those reps as unusable. They are
never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .worker import FAMILIES, LIBS, NULLS

NS: tuple[int, ...] = (60, 200, 1_000, 5_000)


@dataclass(frozen=True)
class Cell:
    family: str
    n: int
    null: str

    @property
    def key(self) -> str:
        return f"{self.family}/n={self.n}/{self.null}"


@dataclass(frozen=True)
class Plan:
    name: str
    description: str
    cells: tuple[Cell, ...]
    reps: int
    # Reps per worker process. One process per rep would spend most of a small
    # cell importing gamfit and pyGAM; one process per cell would lose a whole
    # cell to one runaway rep. A chunk streams one RESULT line per rep, so a
    # chunk the safety net kills still keeps the reps it finished.
    chunk: int
    timeout_s: float
    libs: tuple[str, ...] = field(default=LIBS)
    # Whether this plan's verdict is an ASSERTION rather than a measurement.
    # A run of an asserting plan exits nonzero when any gamfit surface of any
    # of its cells comes out miscalibrated or produces no p-value at all
    # (``report.gate_failures``); that is what makes a scheduled run able to go
    # red, and ``report.FALSE_ALARM`` is the rate at which a fully calibrated
    # harness does so by chance. A plan asserts when its cells are a fixed
    # regression subset; the grid plans sweep every family x null in order to
    # PRODUCE the committed baseline and the docs table, so a flagged row there
    # is the finding the run exists to report, not a failure of the run.
    asserts_calibration: bool = False


def _grid(
    families: tuple[str, ...], ns: tuple[int, ...], nulls: tuple[str, ...]
) -> tuple[Cell, ...]:
    return tuple(Cell(f, n, s) for n in ns for f in families for s in nulls)


PLANS: dict[str, Plan] = {
    p.name: p
    for p in (
        Plan(
            name="ci",
            description=(
                "CI regression subset: a few cheap cells, 200 reps, fixed seeds; "
                "the smoke test fails if any gamfit surface rejects more often "
                "than a calibrated test can"
            ),
            cells=(
                Cell("gaussian", 200, "smooth"),
                Cell("poisson", 200, "smooth"),
                Cell("gaussian", 200, "linear"),
            ),
            reps=200,
            chunk=25,
            timeout_s=900.0,
            libs=("gamfit",),
            asserts_calibration=True,
        ),
        Plan(
            name="quick",
            description=(
                "every family x every null structure at n=200, except that the "
                "ti null runs for Gaussian only; 100 reps, gamfit and pyGAM "
                "(committed baseline behind docs/pvalues.md)"
            ),
            # smooth_significance on an s + s + ti model takes minutes per fit,
            # about ten times any other cell, so one ti cell stands in for the
            # five here. The nightly plan runs all of them.
            cells=tuple(c for c in _grid(FAMILIES, (200,), NULLS) if c.null != "ti")
            + (Cell("gaussian", 200, "ti"),),
            reps=100,
            chunk=10,
            timeout_s=3_600.0,
        ),
        Plan(
            name="nightly",
            description=(
                "the full grid: every family x n in {60, 200, 1000, 5000} x every "
                "null structure, 500 reps (MCSE <= 0.01 at 0.05)"
            ),
            cells=_grid(FAMILIES, NS, NULLS),
            reps=500,
            chunk=10,
            timeout_s=3_600.0,
        ),
    )
}
