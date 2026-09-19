"""Named benchmark plans: which cells to run, how many reps, which safety net.

A plan is a list of cells ``(family, n, design)`` crossed with ``libs`` and
``reps`` seeds. A cell may also set ``threads`` (the value every thread-pool
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

from .fuzz_families import FAMILY_LABELS, REGIMES, fuzz_design
from .worker import (
    BINOMIAL_FAMILIES,
    COUNT_FAMILIES,
    EXTRA_DESIGNS,
    FAMILIES,
    LIBS,
    POSITIVE_FAMILIES,
)
from .worker import DESIGNS as ALL_DESIGNS

CORE_DESIGNS: tuple[str, ...] = ("p1", "p5", "te")
SMALL_N_DESIGNS: tuple[str, ...] = ("p1", "p3", "p5")
# The Gaussian identity sweep (audit lane sweep-gaussian): every core design
# plus a tensor-with-additive-smooth and a factor-by smooth.
GAUSSIAN_SWEEP_DESIGNS: tuple[str, ...] = ALL_DESIGNS + EXTRA_DESIGNS

# Convergence fuzz over families and links (fuzz_families.py): every family
# label x every data regime x n in FAMILY_FUZZ_NS x FAMILY_FUZZ_REPS seeds ->
# 19 x 8 x 3 x 3 = 1368 gamfit fits.
FAMILY_FUZZ_NS: tuple[int, ...] = (50, 500, 5_000)
FAMILY_FUZZ_REPS = 3
# The quick mode: every family label at the base regime and at the edge of its
# support, at the two smaller n, one seed. It is a 0-failure regression test
# (test_fuzz_families_quick.py).
FAMILY_FUZZ_QUICK_REGIMES: tuple[str, ...] = ("base", "edge", "zeros", "lowdisp")
FAMILY_FUZZ_QUICK_NS: tuple[int, ...] = (50, 500)


@dataclass(frozen=True)
class Cell:
    family: str
    n: int
    design: str
    threads: int | None = 1
    concurrency: int = 1

    @property
    def key(self) -> str:
        return f"{self.family}/n={self.n}/{self.design}{variant_suffix(self.threads, self.concurrency)}"


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


# The binomial sweep (audit lane sweep-binomial): prevalence 0.5 / 0.1 / 0.01
# and a grouped binomial with 1..20 trials per row.
BINOMIAL_SWEEP: tuple[str, ...] = ("binomial", *BINOMIAL_FAMILIES)


def _grid(
    ns: tuple[int, ...],
    designs: tuple[str, ...],
    families: tuple[str, ...] = FAMILIES,
) -> tuple[Cell, ...]:
    # Ordered by n ascending so a timeout at small n can mark the larger n of
    # the same (lib, family, design) as not run instead of burning the net on
    # each one in turn.
    return tuple(Cell(f, n, d) for n in ns for f in families for d in designs)


def _family_fuzz_grid(ns: tuple[int, ...], regimes: tuple[str, ...]) -> tuple[Cell, ...]:
    # Ordered by n ascending for the same not-run-after-timeout rule as _grid.
    return tuple(
        Cell(f, n, fuzz_design(r)) for n in ns for f in FAMILY_LABELS for r in regimes
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
        # The positive-continuous speed/convergence sweep (audit lane
        # sweep-positive): Gamma on the log and inverse links, Gamma with heavy
        # right skew and near-zero responses, the inverse Gaussian, a
        # log-normal response fitted as Gaussian on the log scale and as
        # Gamma(log) on the raw scale, and the scaled Student-t. pyGAM runs the
        # Gamma cells only; the others report gamfit's absolute times and
        # certification.
        Plan(
            name="positive_small",
            description="n in {1e2, 1e3}, every positive family x every design, 3 reps",
            cells=_grid((100, 1_000), ALL_DESIGNS, POSITIVE_FAMILIES),
            reps=3,
            timeout_s=600.0,
        ),
        Plan(
            name="positive_1e4",
            description="n=1e4, every positive family x every design, 2 reps",
            cells=_grid((10_000,), ALL_DESIGNS, POSITIVE_FAMILIES),
            reps=2,
            timeout_s=1_800.0,
        ),
        Plan(
            name="positive_1e5",
            description="n=1e5, every positive family x {p1, p5, te}, 1 rep",
            cells=_grid((100_000,), CORE_DESIGNS, POSITIVE_FAMILIES),
            reps=1,
            timeout_s=3_600.0,
        ),
        Plan(
            name="binomial_small",
            description=(
                "n in {1e2, 1e3}, binomial at prevalence 0.5 / 0.1 / 0.01 and"
                " with trials x every design, 3 reps"
            ),
            cells=_grid((100, 1_000), ALL_DESIGNS, BINOMIAL_SWEEP),
            reps=3,
            timeout_s=600.0,
        ),
        Plan(
            name="binomial_1e4",
            description=(
                "n=1e4, binomial at prevalence 0.5 / 0.1 / 0.01 and with trials"
                " x every design, 2 reps"
            ),
            cells=_grid((10_000,), ALL_DESIGNS, BINOMIAL_SWEEP),
            reps=2,
            timeout_s=1_800.0,
        ),
        Plan(
            name="binomial_1e5",
            description=(
                "n=1e5, binomial at prevalence 0.5 / 0.1 / 0.01 and with trials"
                " x every design, 1 rep"
            ),
            cells=_grid((100_000,), ALL_DESIGNS, BINOMIAL_SWEEP),
            reps=1,
            timeout_s=3_600.0,
        ),
        # The count-family speed/convergence sweep (audit lane sweep-count):
        # Poisson at mean 0.3 / 5 / 500 and with a log-exposure offset, the
        # negative binomial with theta estimated, and Tweedie with phi
        # estimated. pyGAM runs the Poisson cells only (it has no NB or
        # Tweedie), so those two report gamfit's absolute times and status.
        Plan(
            name="count_small",
            description="n in {1e2, 1e3}, every count family x every design, 3 reps",
            cells=_grid((100, 1_000), ALL_DESIGNS, COUNT_FAMILIES),
            reps=3,
            timeout_s=600.0,
        ),
        Plan(
            name="count_1e4",
            description="n=1e4, every count family x every design, 2 reps",
            cells=_grid((10_000,), ALL_DESIGNS, COUNT_FAMILIES),
            reps=2,
            timeout_s=1_800.0,
        ),
        Plan(
            name="count_1e5",
            description="n=1e5, every count family x {p1, p5, te}, 1 rep",
            cells=_grid((100_000,), CORE_DESIGNS, COUNT_FAMILIES),
            reps=1,
            timeout_s=3_600.0,
        ),
        Plan(
            name="fuzz_families",
            description=(
                "convergence fuzz over families x links: every family label x"
                " every support-edge regime x n in {50, 500, 5000}, gamfit only,"
                f" {FAMILY_FUZZ_REPS} reps"
            ),
            cells=_family_fuzz_grid(FAMILY_FUZZ_NS, REGIMES),
            reps=FAMILY_FUZZ_REPS,
            timeout_s=1_800.0,
            libs=("gamfit",),
        ),
        Plan(
            name="fuzz_families_quick",
            description=(
                "family fuzz quick mode: every family label at the base, edge,"
                " zeros and lowdisp regimes, n in {50, 500}, 1 rep"
            ),
            cells=_family_fuzz_grid(FAMILY_FUZZ_QUICK_NS, FAMILY_FUZZ_QUICK_REGIMES),
            reps=1,
            timeout_s=600.0,
            libs=("gamfit",),
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
