"""Bug hunt (#1548): the default P-spline smooth ``s(x)`` / ``s(x, bs="ps")``
must be INVARIANT to reflecting the covariate ``x -> -x``.

Reflection is an *exact* symmetry of the uniform B-spline basis and its
difference penalty.  It maps the uniform knot lattice ``[min x, max x]`` to the
mirrored lattice ``[-max x, -min x]`` (same spacing), reverses the basis columns
(``B_j(-x) = B_{K+1-j}(x)``), and leaves both the order-``m`` difference penalty
and the null-space ``{1, x}`` ridge invariant.  So the penalized least-squares
problem AND the REML/LAML smoothing-parameter objective are identical up to a
fixed column permutation; the fitted curve must satisfy
``f_reflected(-t) = f_original(t)`` to machine precision, and the two selected
smoothing-parameter vectors must agree.

Root cause (pre-fix): the default double-penalty ``ps`` smooth whose null space
(the linear trend) is genuinely SUPPORTED has a bimodal REML profile along the
null-space coordinate ``rho_null`` — a deep "keep" basin at a moderate
``lambda_null`` and a flat high-``lambda_null`` "annihilation shelf".  The
objective-grid prepass seeds on the shelf corner; whether the outer optimizer
then crosses the flat shelf into the keep basin was decided by sub-ULP gradient
signs that the column-reversal of ``x -> -x`` flips.  One orientation rolled off
into the keep basin (the global REML optimum) while the mirror orientation
stalled on the shelf and certified that strictly-worse point as converged,
annihilating the supported linear trend and drifting the curve ~3.4 % of the
signal range.

The fix makes the outer smoothing-parameter selection escape the high-lambda
null-space shelf whenever a strictly-better (lower-REML) keep basin exists, so
both orientations converge to the SAME global optimum.

R-free / tool-free: the contract is a pure internal symmetry of the gam fit.
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


def _signal(t: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * t) + 0.3 * t**2


def _fit_pred(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"y": y, "x": x})
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        m = gamfit.fit(df, 'y ~ s(x, bs="ps")', family="gaussian")
        pred = np.asarray(m.predict(pd.DataFrame({"x": grid}))).ravel()
        lambdas = np.asarray(m.summary().lambdas, dtype=float)
    return pred, lambdas


def test_pspline_default_fit_is_reflection_invariant():
    # Legacy MT19937 seeds, stable across versions. Seed 2 is the originally
    # reported failure (lambda_null jumped 11 orders of magnitude under
    # reflection, curve drifted ~3.4 % of the signal range); the others are
    # stable controls that must keep behaving.
    grid = np.linspace(-1.8, 1.8, 60)
    signal_range = float(np.ptp(_signal(grid)))
    n = 400

    worst = 0.0
    for seed in (0, 1, 2, 3, 4):
        rng = np.random.RandomState(seed)
        x = rng.uniform(-2.0, 2.0, n)
        y = _signal(x) + 0.2 * rng.standard_normal(n)

        p1, l1 = _fit_pred(x, y, grid)
        p2, l2 = _fit_pred(-x, y, -grid)  # mirror orientation, read at mirrored grid

        drift = float(np.max(np.abs(p1 - p2)))
        worst = max(worst, drift)

        # The fitted curves must agree to far below the 0.076 (3.4 %) defect and
        # well above genuine column-reordering rounding noise (~1e-8).
        assert drift < 1e-3 * signal_range + 1e-6, (
            f"seed {seed}: reflected P-spline fit drifts {drift:.4g} "
            f"({100 * drift / signal_range:.2f} % of the {signal_range:.3f} signal range) — "
            f"the default ps smooth is not invariant to x -> -x. "
            f"lambdas original={l1}, reflected={l2} (#1548)."
        )

        # The selected smoothing parameters themselves must match: a flipped
        # landing shoulder shows up as a multi-order-of-magnitude lambda_null gap.
        assert l1.shape == l2.shape
        for a, b in zip(l1, l2):
            rel = abs(a - b) / (abs(a) + abs(b) + 1e-12)
            assert rel < 1e-3, (
                f"seed {seed}: selected smoothing parameters differ under reflection "
                f"(original={l1}, reflected={l2}); rel gap {rel:.3g} on a component (#1548)."
            )

    # Sanity: the test grid actually exercises a non-trivial signal.
    assert signal_range > 1.0
