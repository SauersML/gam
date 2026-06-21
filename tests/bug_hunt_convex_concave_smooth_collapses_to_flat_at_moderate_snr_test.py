"""Regression test for #1380.

The convex / concave shape-constrained smooth silently collapsed to a flat
(linear-corner) fit and discarded the signal at moderate sample size / SNR, on
data where the shape constraint is *correct* and an *unconstrained* ``s(x)``
recovers the signal nearly perfectly.

Repro (from the issue): ``y = (x - 0.5)**2 + 0.05*N(0,1)`` on ``n = 400`` —
a clean convex parabola with true range 0.25. The unconstrained ``s(x)`` fit
recovers R² ≈ 0.99 (EDF ≈ 9). The ``shape="convex"`` fit collapsed to a flat
line: EDF pinned at exactly 3.0 (the maximally-smooth / linear corner of the
convex cone), fitted range ~0.01, R²-vs-truth ≈ 0.

Root cause: the box-reparameterised convex/concave smooth fits under an
active-set inequality constraint and the outer REML smoothing-parameter search
selected λ at the linear corner. There, every curvature coordinate is clamped,
so the constrained REML score's λ-dependence collapses (a flat plateau in ρ),
and the gradient-based outer search parks in that basin instead of descending to
the well-supported curved fit. A shape constraint that the truth satisfies can
only *remove* wrong candidates — it can never recover less signal than an
unconstrained ``s(x)`` — so this collapse was a genuine bug in the constrained
λ-selection path, not over-smoothing.

When the constrained REML stops parking at the flat corner, both tests below
pass with no edits.
"""

from __future__ import annotations

import importlib
import os
import contextlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

# R²-vs-truth floor the constrained fit must clear. The unconstrained control
# reaches ~0.99 on this data; 0.70 is a generous floor that the flat-corner
# collapse (R² ≈ 0) fails by a wide margin and a genuine curved recovery clears
# easily.
R2_FLOOR = 0.70


@contextlib.contextmanager
def _silence() -> Any:
    so, se = os.dup(1), os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    try:
        yield
    finally:
        os.dup2(so, 1)
        os.dup2(se, 2)
        os.close(dn)
        os.close(so)
        os.close(se)


def _fit_predict_truth(
    seed: int, n: int, formula: str, truth_fn: Any
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    truth = truth_fn(x)
    y = truth + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "x": x})
    with _silence():
        model = gamfit.fit(df, formula, family="gaussian")
        pred = np.asarray(model.predict(pd.DataFrame({"x": x})), dtype=float).ravel()
        edf = float(model.summary().edf_total)
    return truth, pred, edf


def _r2_vs_truth(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.var(truth - pred) / np.var(truth))


def test_unconstrained_control_recovers_convex_signal() -> None:
    """Control: the unconstrained smooth recovers the convex parabola (~0.99)."""
    truth, pred, edf = _fit_predict_truth(0, 400, "y ~ s(x)", lambda x: (x - 0.5) ** 2)
    assert np.all(np.isfinite(pred))
    r2 = _r2_vs_truth(truth, pred)
    assert r2 > 0.95, f"unconstrained control should recover the signal, got R²={r2:.4f}"


def test_convex_smooth_recovers_clean_convex_signal() -> None:
    """The headline repro: convex fit must recover the clean convex signal."""
    for seed in range(4):
        truth, pred, edf = _fit_predict_truth(
            seed, 400, 'y ~ s(x, shape="convex")', lambda x: (x - 0.5) ** 2
        )
        assert np.all(np.isfinite(pred)), f"seed {seed}: prediction must be finite"
        span = pred.max() - pred.min()
        # Recovery of the convex signal is the bug's acceptance metric: the
        # flat-corner collapse gives R²-vs-truth ≈ 0; a genuine curved fit clears
        # the 0.70 floor (the unconstrained control reaches ~0.99).
        r2 = _r2_vs_truth(truth, pred)
        assert r2 > R2_FLOOR, (
            f"seed {seed}: convex fit collapsed to the flat corner "
            f"(R²-vs-truth={r2:.4f} below floor {R2_FLOOR}, EDF={edf:.2f}, "
            f"fitted range={span:.4f})"
        )
        # And the fit is genuinely curved, not the linear corner: its overall
        # curvature is convex (the endpoints sit above the chord midpoint of a
        # convex parabola). The shape constraint itself is enforced on the basis
        # coefficients; this is a robust shape sanity check that tolerates the
        # finite-difference noise of differentiating a dense prediction grid.
        mid = pred[len(pred) // 2]
        chord = 0.5 * (pred[0] + pred[-1])
        assert mid < chord, (
            f"seed {seed}: convex fit is not bowl-shaped "
            f"(midpoint {mid:.4f} not below chord {chord:.4f})"
        )


def test_concave_smooth_recovers_clean_concave_signal() -> None:
    """Concavity is symmetric: -(x-0.5)**2 with shape=concave must recover too."""
    for seed in range(4):
        truth, pred, edf = _fit_predict_truth(
            seed, 400, 'y ~ s(x, shape="concave")', lambda x: -((x - 0.5) ** 2)
        )
        assert np.all(np.isfinite(pred)), f"seed {seed}: prediction must be finite"
        span = pred.max() - pred.min()
        r2 = _r2_vs_truth(truth, pred)
        assert r2 > R2_FLOOR, (
            f"seed {seed}: concave fit collapsed to the flat corner "
            f"(R²-vs-truth={r2:.4f} below floor {R2_FLOOR}, EDF={edf:.2f}, "
            f"fitted range={span:.4f})"
        )
        # Genuinely dome-shaped (concave), not the linear corner.
        mid = pred[len(pred) // 2]
        chord = 0.5 * (pred[0] + pred[-1])
        assert mid > chord, (
            f"seed {seed}: concave fit is not dome-shaped "
            f"(midpoint {mid:.4f} not above chord {chord:.4f})"
        )
