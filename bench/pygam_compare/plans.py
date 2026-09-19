"""Named benchmark plans: which cells to run, how many reps, which safety net.

A plan is a list of cells ``(family, n, design)`` crossed with ``libs`` and
``reps`` seeds. A cell may also fix ``n_predict``, the number of held-out rows
predicted on (default: ``n``), and a plan may set ``postfit`` to time the
fitted model's other post-fit operations as well (see ``worker.py``).
A cell may also set ``threads`` (the value every thread-pool
variable gets; ``None`` leaves them unset so each pool sizes itself to the
host) and ``concurrency`` (how many identical reps run at once, each its own
process, which is what ``joblib`` / ``multiprocessing`` / ``n_jobs=-1`` do).
The defaults, one thread and one process, are the single-core comparison.

``timeout_s`` and ``memcap_mb`` are a HARNESS SAFETY NET ONLY: they stop
one runaway rep from eating the whole run. They are not a solver
budget; a rep that hits either is recorded with status ``timeout`` / ``memcap``
and reported as a loss for that library, never dropped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .worker import DESIGNS as ALL_DESIGNS
from .worker import FAMILIES, LIBS

CORE_DESIGNS: tuple[str, ...] = ("p1", "p5", "te")
POSTFIT_DESIGNS: tuple[str, ...] = ("p5", "p20", "te")
SMALL_N_DESIGNS: tuple[str, ...] = ("p1", "p3", "p5")


@dataclass(frozen=True)
class Cell:
    family: str
    n: int
    design: str
    threads: int | None = 1
    concurrency: int = 1
    n_predict: int | None = None

    @property
    def key(self) -> str:
        key = f"{self.family}/n={self.n}/{self.design}"
        if self.n_predict is not None:
            key += f"/n_predict={self.n_predict}"
        return key + variant_suffix(self.threads, self.concurrency)


def threads_label(threads: int | None) -> str:
    return "auto" if threads is None else str(threads)


def variant_suffix(threads: int | None, concurrency: int) -> str:
    """Name suffix for a non-default thread setting; empty for the default, so
    single-core cells keep the names every existing baseline uses."""
    out = ""
    if threads != 1:
        out += f" threads={threads_label(threads)}"
    if concurrency != 1:
        out += f" x{concurrency}"
    return out


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
        Cell(f, n, d, n_predict=m) for n in ns for m in n_predicts for f in FAMILIES for d in designs
    )


SCALING_NS: tuple[int, ...] = (10_000, 100_000, 1_000_000)
SCALING_FAMILIES: tuple[str, ...] = ("gaussian", "binomial")
SCALING_DESIGNS: tuple[str, ...] = ("p5", "p20", "te")
# 8 exceeds the core count of most CI runners on purpose: it measures what an
# oversubscribed pool inside one process costs. ``None`` is the pool default.
SCALING_THREADS: tuple[int | None, ...] = (1, 2, 4, 8, None)
# One worker per logical CPU, which is what ``n_jobs=-1`` launches.
HOST_WORKERS: int = os.cpu_count() or 1


def _thread_grid() -> tuple[Cell, ...]:
    # n outermost, as in ``_grid``, so a timeout at small n stops the larger n
    # of the same (family, design, threads) rather than each in turn.
    return tuple(
        Cell(f, n, d, threads=t)
        for n in SCALING_NS
        for f in SCALING_FAMILIES
        for d in SCALING_DESIGNS
        for t in SCALING_THREADS
    )


def _oversubscription_grid() -> tuple[Cell, ...]:
    # Each shape runs alone and as HOST_WORKERS simultaneous processes, with
    # the thread pools pinned to one thread and at their default, so the report
    # can set the throughput of a full process fan-out against one fit.
    return tuple(
        Cell("gaussian", n, d, threads=t, concurrency=k)
        for n, d in ((20_000, "te"), (100_000, "p5"))
        for t in (1, None)
        for k in (1, HOST_WORKERS)
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
        Plan(
            name="threads",
            description=(
                "gamfit thread scaling: n in {1e4, 1e5, 1e6} x {gaussian, binomial}"
                " x {p5, p20, te} x threads {1, 2, 4, 8, auto}, 2 reps"
            ),
            cells=_thread_grid(),
            reps=2,
            timeout_s=3_600.0,
            libs=("gamfit",),
        ),
        Plan(
            name="oversubscribe",
            description=(
                "gamfit process fan-out: gaussian n=2e4 te and n=1e5 p5, alone and"
                " as one process per CPU, threads {1, auto}, 2 reps"
            ),
            cells=_oversubscription_grid(),
            reps=2,
            timeout_s=3_600.0,
            libs=("gamfit",),
        ),
    )
}
