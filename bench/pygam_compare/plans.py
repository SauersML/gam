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
from .worker import EXTRA_DESIGNS, FAMILIES, LIBS

CORE_DESIGNS: tuple[str, ...] = ("p1", "p5", "te")
SMALL_N_DESIGNS: tuple[str, ...] = ("p1", "p3", "p5")
# The Gaussian identity sweep (audit lane sweep-gaussian): every core design
# plus a tensor-with-additive-smooth and a factor-by smooth.
GAUSSIAN_SWEEP_DESIGNS: tuple[str, ...] = ALL_DESIGNS + EXTRA_DESIGNS


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


def _grid(
    ns: tuple[int, ...],
    designs: tuple[str, ...],
    families: tuple[str, ...] = FAMILIES,
) -> tuple[Cell, ...]:
    # Ordered by n ascending so a timeout at small n can mark the larger n of
    # the same (lib, family, design) as not run instead of burning the net on
    # each one in turn.
    return tuple(Cell(f, n, d) for n in ns for f in families for d in designs)


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
            name="n1e6_memory",
            description=(
                "n=1e6, {gaussian, poisson} x {p1, p5}, 1 rep: peak RSS and"
                " user/sys CPU against the dense design (pyGAM audit F11)"
            ),
            cells=tuple(
                Cell(f, 1_000_000, d) for f in ("gaussian", "poisson") for d in ("p1", "p5")
            ),
            reps=1,
            timeout_s=3_600.0,
        ),
        Plan(
            name="gaussian_small",
            description=(
                "Gaussian identity, n in {1e2, 1e3, 1e4} x {p1, p5, p20, te, te+s, by},"
                " 3 reps (the nightly Gaussian regression cells)"
            ),
            cells=_grid((100, 1_000, 10_000), GAUSSIAN_SWEEP_DESIGNS, ("gaussian",)),
            reps=3,
            timeout_s=1_200.0,
        ),
        Plan(
            name="gaussian_1e5",
            description="Gaussian identity, n=1e5 x {p1, p5, p20, te, te+s, by}, 3 reps",
            cells=_grid((100_000,), GAUSSIAN_SWEEP_DESIGNS, ("gaussian",)),
            reps=3,
            timeout_s=3_600.0,
        ),
        Plan(
            name="gaussian_1e6",
            description=(
                "Gaussian identity, n=1e6 x {p1, p5, p20, te, te+s, by}, 3 reps:"
                " wall, CPU and peak RSS at the largest scale"
            ),
            cells=_grid((1_000_000,), GAUSSIAN_SWEEP_DESIGNS, ("gaussian",)),
            reps=3,
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
