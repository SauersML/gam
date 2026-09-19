"""Named benchmark plans: which cells to run, how many reps, which safety net.

A plan is a list of cells ``(family, n, design)`` crossed with ``libs`` and
``reps`` seeds. ``timeout_s`` and ``memcap_mb`` are a HARNESS SAFETY NET ONLY:
they stop one runaway rep from eating the whole run. They are not a solver
budget; a rep that hits either is recorded with status ``timeout`` / ``memcap``
and reported as a loss for that library, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .worker import DESIGNS as ALL_DESIGNS
from .worker import FAMILIES, LIBS

CORE_DESIGNS: tuple[str, ...] = ("p1", "p5", "te")
SMALL_N_DESIGNS: tuple[str, ...] = ("p1", "p3", "p5")


@dataclass(frozen=True)
class Cell:
    family: str
    n: int
    design: str

    @property
    def key(self) -> str:
        return f"{self.family}/n={self.n}/{self.design}"


@dataclass(frozen=True)
class Plan:
    name: str
    description: str
    cells: tuple[Cell, ...]
    reps: int
    timeout_s: float
    libs: tuple[str, ...] = field(default=LIBS)


def _grid(ns: tuple[int, ...], designs: tuple[str, ...]) -> tuple[Cell, ...]:
    # Ordered by n ascending so a timeout at small n can mark the larger n of
    # the same (lib, family, design) as not run instead of burning the net on
    # each one in turn.
    return tuple(Cell(f, n, d) for n in ns for f in FAMILIES for d in designs)


PLANS: dict[str, Plan] = {
    p.name: p
    for p in (
        Plan(
            name="smoke",
            description="CI smoke test: n=300, p1, all families, 1 rep",
            cells=_grid((300,), ("p1",)),
            reps=1,
            timeout_s=300.0,
        ),
        Plan(
            name="quick",
            description="n=1e3, every family x every design, 3 reps (committed baseline)",
            cells=_grid((1_000,), ALL_DESIGNS),
            reps=3,
            timeout_s=600.0,
        ),
        Plan(
            name="small_n",
            description=(
                "n in {50, 200, 500}, every family x {p1, p3, p5}, 3 reps:"
                " fixed per-fit overhead, cold and warm"
            ),
            cells=_grid((50, 200, 500), SMALL_N_DESIGNS),
            reps=3,
            timeout_s=600.0,
        ),
        Plan(
            name="n1e4_core",
            description="n=1e4, every family x {p1, p5, te}, 3 reps",
            cells=_grid((10_000,), CORE_DESIGNS),
            reps=3,
            timeout_s=1_200.0,
        ),
        Plan(
            name="n1e5_core",
            description="n=1e5, every family x {p1, p5, te}, 2 reps",
            cells=_grid((100_000,), CORE_DESIGNS),
            reps=2,
            timeout_s=3_600.0,
        ),
        Plan(
            name="full",
            description="n in {1e3, 1e4, 1e5}, every family x every design, 3 reps",
            cells=_grid((1_000, 10_000, 100_000), ALL_DESIGNS),
            reps=3,
            timeout_s=3_600.0,
        ),
    )
}
