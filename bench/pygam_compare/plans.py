"""Named benchmark plans: which cells to run, how many reps, which safety net.

A plan is a list of cells ``(family, n, design)`` crossed with ``libs`` and
``reps`` seeds. A cell may also fix ``n_predict``, the number of held-out rows
predicted on (default: ``n``), and a plan may set ``postfit`` to time the
fitted model's other post-fit operations as well (see ``worker.py``).
``timeout_s`` and ``memcap_mb`` are a HARNESS SAFETY NET ONLY: they stop one runaway rep from eating the whole run. They are not a solver
budget; a rep that hits either is recorded with status ``timeout`` / ``memcap``
and reported as a loss for that library, never dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .worker import DESIGNS as ALL_DESIGNS
from .worker import FAMILIES, LIBS

CORE_DESIGNS: tuple[str, ...] = ("p1", "p5", "te")
POSTFIT_DESIGNS: tuple[str, ...] = ("p5", "p20", "te")


@dataclass(frozen=True)
class Cell:
    family: str
    n: int
    design: str
    n_predict: int | None = None

    @property
    def key(self) -> str:
        key = f"{self.family}/n={self.n}/{self.design}"
        return key if self.n_predict is None else f"{key}/n_predict={self.n_predict}"


@dataclass(frozen=True)
class Plan:
    name: str
    description: str
    cells: tuple[Cell, ...]
    reps: int
    timeout_s: float
    libs: tuple[str, ...] = field(default=LIBS)
    postfit: bool = False


def _grid(ns: tuple[int, ...], designs: tuple[str, ...]) -> tuple[Cell, ...]:
    # Ordered by n ascending so a timeout at small n can mark the larger n of
    # the same (lib, family, design) as not run instead of burning the net on
    # each one in turn.
    return tuple(Cell(f, n, d) for n in ns for f in FAMILIES for d in designs)


def _postfit_grid(
    ns: tuple[int, ...], n_predicts: tuple[int, ...], designs: tuple[str, ...]
) -> tuple[Cell, ...]:
    # n_predict ascending inside each n, for the same timeout reason as _grid.
    return tuple(
        Cell(f, n, d, m) for n in ns for m in n_predicts for f in FAMILIES for d in designs
    )


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
        Plan(
            name="postfit",
            description=(
                "post-fit ops: n in {1e3, 1e5} x n_predict in {1e2, 1e4, 1e6},"
                " every family x {p5, p20, te}, gamfit vs pygam_gs, 1 rep"
            ),
            cells=_postfit_grid((1_000, 100_000), (100, 10_000, 1_000_000), POSTFIT_DESIGNS),
            reps=1,
            timeout_s=3_600.0,
            libs=("gamfit", "pygam_gs"),
            postfit=True,
        ),
    )
}
