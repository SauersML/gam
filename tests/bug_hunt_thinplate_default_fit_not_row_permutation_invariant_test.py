"""Row-permutation invariance of the DEFAULT univariate thin-plate smooth (#1378).

A univariate thin-plate regression spline ``s(x, bs="tp")`` is a functional of
the unordered training sample ``{(x_i, y_i)}``: the radial kernel ``phi(x_i-x_j)``,
the polynomial nullspace ``{1, x}`` and the smoothness penalty are all symmetric
in the rows. So the fitted curve MUST be invariant to a pure permutation of the
training rows: permuting rows must give bit-identical (machine-precision)
predictions.

The defect (#1378): the DEFAULT univariate tp basis was sized by the generic
spatial center heuristic, which scales with ``n`` (≈75 centers / ≈62-coefficient
basis at n=300). That oversized basis carries two penalty blocks whose REML
ρ-surface has a weakly-identified FLAT VALLEY. The outer optimizer stalled on the
valley and returned a NON-converged ρ̂ whose landing point depended on row order,
so a pure row permutation moved the fitted curve by ~0.076 (≈3% of signal range).
``bs="cr"`` and ``bs="ps"`` (small local knot grids) converged cleanly and were
already row-permutation invariant to ≤1e-14.

The fix sizes the DEFAULT univariate tp basis to an mgcv-sized ``k = 10`` so the
ρ-surface is well-identified and the optimizer converges to a single ρ̂
regardless of row order.  The control below pins ``bs="cr"`` / ``bs="ps"`` as
already-invariant references that isolate the defect to the default tp path.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

# Drift ceiling: a row permutation is an exact symmetry, so the only slack is
# floating-point summation reordering. cr/ps land ≤1e-14; allow generous headroom
# below the ~0.076 defect signal while staying far stricter than any real drift.
DRIFT_CEILING = 1e-4

PERMUTATION_SEEDS = (1, 7, 19, 123)


def _signal(x: np.ndarray) -> np.ndarray:
    # A smooth, clearly non-polynomial target so the smooth genuinely bends.
    return np.sin(1.7 * x) + 0.5 * np.cos(0.9 * x)


def _make_data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    x = np.sort(rng.uniform(-3.0, 3.0, n))
    y = _signal(x) + 0.15 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y})


def _fit_predict(fr: pd.DataFrame, bs: str, grid: np.ndarray) -> np.ndarray:
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        m = gamfit.fit(fr, f'y ~ s(x, bs="{bs}")', family="gaussian")
    return np.asarray(m.predict(pd.DataFrame({"x": grid}))).ravel()


def _max_drift_under_permutation(bs: str) -> tuple[float, float]:
    """Max absolute prediction drift across row permutations vs. the canonical
    (row-sorted) fit, and the signal range over the evaluation grid."""
    base = _make_data()
    grid = np.linspace(-2.8, 2.8, 80)
    signal_range = float(_signal(grid).max() - _signal(grid).min())
    assert signal_range > 0.5, f"degenerate signal range {signal_range}"

    ref = _fit_predict(base, bs, grid)
    worst = 0.0
    for seed in PERMUTATION_SEEDS:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(base))
        permuted = base.iloc[perm].reset_index(drop=True)
        pred = _fit_predict(permuted, bs, grid)
        worst = max(worst, float(np.max(np.abs(pred - ref))))
    return worst, signal_range


def test_default_thinplate_fit_is_row_permutation_invariant() -> None:
    """The DEFAULT ``s(x, bs="tp")`` fit must not move under a row permutation."""
    drift, signal_range = _max_drift_under_permutation("tp")
    rel = drift / signal_range
    assert drift < DRIFT_CEILING, (
        f'default s(x, bs="tp") is not row-permutation invariant: '
        f"max drift {drift:.6g} ({100 * rel:.3f}% of signal range "
        f"{signal_range:.4g}) >= ceiling {DRIFT_CEILING:g}"
    )


def test_local_bases_are_row_permutation_invariant_control() -> None:
    """Control: cr/ps are already row-permutation invariant — pins the defect to
    the default tp path, not the harness."""
    for bs in ("cr", "ps"):
        drift, signal_range = _max_drift_under_permutation(bs)
        assert drift < DRIFT_CEILING, (
            f's(x, bs="{bs}") control is not row-permutation invariant: '
            f"max drift {drift:.6g} of signal range {signal_range:.4g}"
        )
