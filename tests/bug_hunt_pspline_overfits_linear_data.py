"""Bug hunt: the default smooth ``s(x)`` / ``s(x, bs="ps")`` OVER-FITS data whose
signal lies entirely in the penalty null space (a straight line).

For a P-spline with an order-2 difference penalty, the penalty null space is
exactly the linear functions ``{1, x}`` (verified separately by
``test_clamped_bspline_curvature_penalty_null_space_is_exactly_linear``). When
the response is genuinely linear plus noise, REML should drive ``lambda -> inf``
and collapse the smooth's effective degrees of freedom (EDF) to the null-space
dimension (~2). mgcv does exactly this; its ``cr`` basis in *this* library does
too (EDF = 2.000, zero residual wiggle).

But ``bs="ps"`` (and the default ``s(x)``, which routes to ps) plateaus at
EDF ~= 4.9-5.0 with a comparatively *small* selected lambda (~4e4 vs ~6.5e7 for
cr), introducing spurious curvature on a dead-straight signal and measurably
worse predictive accuracy than the cr basis on the identical data. ``bs="tp"``
is worse still (EDF ~= 8). The cr basis reaching EDF = 2.000 on the same data
proves that "collapse to the line" is the correct, achievable answer, so ps/tp
are genuinely under-smoothing.

Root-cause hypothesis: the 1-D ``bs="ps"`` path ships its bending penalty
*un-normalized* (normalization_scale = 1.0; see
``bspline_penalty_candidates`` in src/terms/basis/bspline_build.rs), whereas
cr / duchon / tensor penalties are Frobenius-normalized. An un-normalized
penalty puts ``lambda`` on a basis-dependent scale, miscalibrating REML's
lambda-selection so it stops at a smaller effective lambda and the null-space
directions are not fully shrunk.

This is the ``bs="ps"`` sibling of #1271 (tp over-fits linear). Filed separately
because ps was not addressed by the tp data-metric reparam and the cr control
isolates it cleanly.
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


def _fit_edf_and_wiggle(bs: str, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = 400
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y_true = 1.0 + 3.0 * x
    y = y_true + 0.3 * rng.standard_normal(n)
    fr = pd.DataFrame({"x": x, "y": y})
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        m = gamfit.fit(fr, f'y ~ s(x, bs="{bs}")', family="gaussian")
    edf = float(m.summary().edf_total)

    grid = np.linspace(0.02, 0.98, 50)
    pred = np.asarray(m.predict(pd.DataFrame({"x": grid}))).ravel()
    # Residual wiggle = RMS deviation of the fitted curve from its OWN best line.
    a = np.vstack([np.ones_like(grid), grid]).T
    coef, *_ = np.linalg.lstsq(a, pred, rcond=None)
    wiggle = float(np.sqrt(np.mean((pred - a @ coef) ** 2)))
    rmse_truth = float(np.sqrt(np.mean((pred - (1.0 + 3.0 * grid)) ** 2)))
    return edf, wiggle, rmse_truth


def test_pspline_collapses_to_line_on_linear_data():
    # cr reaches the correct answer (EDF ~= 2, zero wiggle) on the SAME data,
    # so it is the control that proves the target is achievable.
    cr_edfs = [_fit_edf_and_wiggle("cr", s)[0] for s in (0, 1, 2)]
    assert max(cr_edfs) < 3.0, f"control failed: cr should collapse to ~2 on linear data, got {cr_edfs}"

    ps = [_fit_edf_and_wiggle("ps", s) for s in (0, 1, 2)]
    ps_edf = [e for e, _, _ in ps]
    ps_wiggle = [w for _, w, _ in ps]

    # A correctly-smoothed P-spline on linear+noise data should land near the
    # null-space dimension. Allow generous slack (3.0) before flagging.
    worst_edf = max(ps_edf)
    worst_wiggle = max(ps_wiggle)
    assert worst_edf < 3.0, (
        f"bs='ps' OVER-FITS linear data: EDF {ps_edf} (expected ~2, cr control {cr_edfs}); "
        f"residual wiggle {ps_wiggle} on a dead-straight signal. The default smooth "
        f"under-smooths signal that lies in the penalty null space."
    )
    assert worst_wiggle < 2e-3, (
        f"bs='ps' introduces spurious curvature on linear data: wiggle {ps_wiggle}"
    )
