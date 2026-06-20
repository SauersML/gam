"""Bug hunt (#1366): a cyclic P-spline smooth (``s(x, periodic=true)``)
UNDER-SMOOTHS data whose signal lies entirely in the cyclic penalty null space —
the end-to-end consequence of shipping the cyclic bending penalty
*un-normalized*.

This is the cyclic sibling of #1365 (``bs="ps"`` over-fits a straight line).
For a cyclic spline with an order-2 difference penalty the penalty null space is
the constant function: a constant signal is annihilated by the roughness
penalty, so when the response is a constant plus noise REML should drive
``lambda -> inf`` and collapse the smooth's effective degrees of freedom (EDF)
to the null-space dimension (~1), leaving a flat fit with no spurious harmonic
wiggle.

Root cause (#1366): the cyclic ``bs="cc"`` path shipped its
``create_cyclic_difference_penalty_matrix`` wiggliness penalty un-normalized
(``normalization_scale = 1.0``), and so did the periodic 1-D Duchon path
(``build_cyclic_duchon_basis_1dwithworkspace``).
``filter_active_penalty_candidates_with_ops`` does NOT normalize — it ships the
operator verbatim and only records the scale — so an un-normalized operator puts
``lambda`` on a basis-dependent scale and miscalibrates REML's lambda-selection
heuristics, stopping at a smaller effective lambda that fails to fully shrink the
null-space directions (the same mechanism as #1364/#1365). After the
Frobenius-normalization fix the cyclic smooth must collapse to the constant on
constant-plus-noise data, exactly as the already-normalized bases do.

This complements the construction-level certification
(``tests/audit_penalty_normalization_scale_equivariance.rs``, which asserts the
shipped penalty is unit-Frobenius) with the observable end-to-end fit
consequence: the corrected cyclic smooth no longer under-smooths a null-space
signal.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

import gamfit


def _fit_cyclic_edf_and_wiggle(seed: int) -> tuple[float, float]:
    """Fit a cyclic P-spline on constant-plus-noise periodic data and return the
    fit's total EDF and the residual harmonic wiggle of the fitted curve."""
    rng = np.random.default_rng(seed)
    n = 400
    period = 2.0 * np.pi
    # A periodic covariate on [0, period); the response is a flat constant plus
    # noise, so its entire signal lies in the cyclic penalty null space.
    x = np.sort(rng.uniform(0.0, period, n))
    y_true = 2.0
    y = y_true + 0.3 * rng.standard_normal(n)
    fr = pd.DataFrame({"x": x, "y": y})
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        # `periodic=true, period=2*pi` routes to the cyclic P-spline basis (the
        # `create_cyclic_difference_penalty_matrix` wiggliness penalty fixed by
        # #1366).
        m = gamfit.fit(fr, "y ~ s(x, periodic=true, period=2*pi)", family="gaussian")
    edf = float(m.summary().edf_total)

    grid = np.linspace(0.05, period - 0.05, 60)
    pred = np.asarray(m.predict(pd.DataFrame({"x": grid}))).ravel()
    # Residual wiggle = RMS deviation of the fitted curve from its OWN mean (the
    # constant is the null space, so a correctly-smoothed fit is flat).
    wiggle = float(np.sqrt(np.mean((pred - pred.mean()) ** 2)))
    return edf, wiggle


def test_cyclic_pspline_collapses_to_constant_on_flat_data():
    # Cyclic P-spline, order-2 difference penalty, null space = {const}. On a
    # flat (constant + noise) signal a correctly-smoothed cyclic spline should
    # land near the null-space dimension and leave no harmonic wiggle.
    results = [_fit_cyclic_edf_and_wiggle(s) for s in (0, 1, 2)]
    edfs = [e for e, _ in results]
    wiggles = [w for _, w in results]

    worst_edf = max(edfs)
    worst_wiggle = max(wiggles)
    # Generous slack (2.5) before flagging an under-smooth; the correct answer is
    # ~1 (the intercept / constant null space).
    assert worst_edf < 2.5, (
        f"cyclic P-spline UNDER-SMOOTHS data whose signal is in the cyclic "
        f"penalty null space: EDF {edfs} (expected ~1 on a flat signal). The "
        f"cyclic bending penalty was shipped un-normalized (#1366), "
        f"miscalibrating REML's lambda-selection."
    )
    assert worst_wiggle < 5e-3, (
        f"cyclic P-spline introduces spurious harmonic wiggle on a flat signal: "
        f"wiggle {wiggles} (expected ~0 once the smooth collapses to the constant)."
    )
