"""Shared pytest fixtures for the Python test suite."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import erf, floor, sqrt
from statistics import NormalDist
from typing import Protocol, TypeVar, cast

import numpy as np
import pandas as pd

_Fixture = TypeVar("_Fixture", bound=Callable[..., object])


class _Pytest(Protocol):
    def fixture(self, fixture_function: _Fixture) -> _Fixture: ...


pytest = cast(_Pytest, import_module("pytest"))

SyntheticLargeScaleFactory = Callable[[int, int], pd.DataFrame]


def _build_synthetic_large_scale(seed: int = 0, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pc1 = rng.normal(0.0, 1.0, n)
    pc2 = 0.3 * pc1 + np.sqrt(1.0 - 0.3 ** 2) * rng.normal(0.0, 1.0, n)
    pc3 = rng.normal(0.0, 1.0, n)
    pc4 = rng.normal(0.0, 1.0, n)

    pgs = 0.4 * pc1 - 0.2 * pc2 + 0.15 * pc3 + rng.normal(0.0, 0.9, n)
    probs = np.array(
        [0.5 * (1.0 + erf((0.5 * z + 0.1 * p) / sqrt(2.0))) for z, p in zip(pgs, pc1)]
    )
    disease = (rng.uniform(0.0, 1.0, n) < probs).astype(np.float64)

    age_entry = rng.uniform(40.0, 70.0, n)
    lam = np.exp(-1.2 - 0.3 * pgs)
    time_to_event = rng.exponential(scale=1.0 / np.clip(lam, 1e-6, None), size=n)
    age_exit_raw = age_entry + time_to_event
    censor_age = 85.0
    age_exit = np.minimum(age_exit_raw, censor_age)
    event = (age_exit_raw < censor_age).astype(np.float64)
    eps = 0.01
    too_short = age_exit <= age_entry + eps
    age_exit = np.where(too_short, age_entry + eps, age_exit)
    event = np.where(too_short, 0.0, event)

    return pd.DataFrame(
        {
            "pc1": pc1,
            "pc2": pc2,
            "pc3": pc3,
            "pc4": pc4,
            "PGS": pgs,
            "disease": disease,
            "age_entry": age_entry,
            "age_exit": age_exit,
            "event": event,
        }
    )


@pytest.fixture
def synthetic_large_scale_factory() -> SyntheticLargeScaleFactory:
    def _factory(seed: int = 0, n: int = 200) -> pd.DataFrame:
        return _build_synthetic_large_scale(seed=seed, n=n)

    return _factory


# Mirror of `gam_test_support::calibration::audit_coverage` for the Python
# calibration tests, which cannot import the Rust harness. Same fixed
# false-positive rate, same two-sided Wilson score interval, same two-sided
# gate: a surface passes only when its nominal level lies inside the CI (#3534).
COVERAGE_FALSE_POSITIVE_RATE = 0.01


@dataclass(frozen=True)
class CoverageVerdict:
    nominal: float
    replications: int
    hits: int
    empirical: float
    ci_lo: float
    ci_hi: float

    @property
    def direction(self) -> str:
        if self.nominal > self.ci_hi:
            return "anti-conservative (nominal ABOVE the CI: under-covers / oversized test)"
        if self.nominal < self.ci_lo:
            return "conservative (nominal BELOW the CI: over-covers / undersized test)"
        return "calibrated"

    @property
    def passed(self) -> bool:
        return self.ci_lo <= self.nominal <= self.ci_hi

    def describe(self) -> str:
        return (
            f"empirical={self.empirical:.4f} (hits {self.hits}/{self.replications}), "
            f"Wilson CI=[{self.ci_lo:.4f},{self.ci_hi:.4f}] at FPR "
            f"{COVERAGE_FALSE_POSITIVE_RATE}, nominal {self.nominal} — {self.direction}"
        )


def _coverage_z() -> float:
    return NormalDist().inv_cdf(1.0 - COVERAGE_FALSE_POSITIVE_RATE / 2.0)


def audit_coverage(hits: int, replications: int, nominal: float) -> CoverageVerdict:
    """Classify `hits / replications` against `nominal` with the two-sided
    Wilson score interval at `COVERAGE_FALSE_POSITIVE_RATE`."""
    if replications <= 0:
        raise ValueError("coverage audit needs at least one replication")
    if not 0.0 < nominal < 1.0:
        raise ValueError("nominal coverage must lie in (0, 1)")
    if not 0 <= hits <= replications:
        raise ValueError("hit count must lie in [0, replications]")
    n = float(replications)
    p_hat = hits / n
    z = _coverage_z()
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = z * sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denom
    return CoverageVerdict(
        nominal=nominal,
        replications=replications,
        hits=hits,
        empirical=p_hat,
        ci_lo=max(center - half, 0.0),
        ci_hi=min(center + half, 1.0),
    )


def replications_resolving_both_tails(nominal: float) -> int:
    """Smallest replication count at which the two-sided verdict can fail the
    conservative tail at all: a surface that covers EVERY replication has Wilson
    lower bound `n / (n + z²)`, which exceeds `nominal` iff
    `n > z²·nominal / (1 − nominal)`. Below this count a test that never
    rejects (or an interval that always covers) is indistinguishable from a
    calibrated one, so a size or coverage audit must run at least this many."""
    z2 = _coverage_z() ** 2
    return floor(z2 * nominal / (1.0 - nominal)) + 1


@pytest.fixture
def coverage_audit() -> Callable[[int, int, float], CoverageVerdict]:
    return audit_coverage


@pytest.fixture
def coverage_replications() -> Callable[[float], int]:
    return replications_resolving_both_tails
