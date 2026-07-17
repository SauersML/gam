"""#2346 python acceptance: competing-risks interval prediction at the DEFAULT
covariance mode.

Per the #2296 provenance contract the default covariance mode is the
smoothing-corrected matrix with no silent fallback; #2346 landed the fit-side
first-order rho-uncertainty inflation (C = A.V_rho.A^T with
FirstOrderIdentifiedSubspace provenance) for custom-family (cause-specific)
fits, so the omitted mode and the explicit spelling must both succeed and
report the corrected provenance — never relabel the conditional matrix.

Self-contained on purpose: this contract lives in its own module so it cannot
be swept away by unrelated rewrites of the general survival API suite.
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def make_competing_risks(n: int = 320, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40.0, 75.0, n)
    x = (age - 55.0) / 10.0
    t1 = rng.exponential(scale=1.0 / np.exp(-3.0 + 0.25 * x), size=n)
    t2 = rng.exponential(scale=1.0 / np.exp(-3.2 - 0.20 * x), size=n)
    censor = rng.exponential(scale=22.0, size=n)
    exit_time = np.minimum.reduce([t1, t2, censor]) + 0.1
    event = np.where((t1 < t2) & (t1 < censor), 1.0, 0.0)
    event = np.where((t2 < t1) & (t2 < censor), 2.0, event)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": exit_time,
            "event": event,
            "age": age,
        }
    )


def prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry": [0.0, 0.0, 0.0],
            "exit": [5.0, 10.0, 20.0],
            "event": [1, 1, 1],
            "age": [45.0, 55.0, 65.0],
        }
    )


def test_competing_risks_default_mode_uses_smoothing_corrected_covariance() -> None:
    model = gamfit.fit(
        make_competing_risks(),
        "Surv(entry, exit, event) ~ age",
        survival_likelihood="weibull",
    )
    rows = prediction_rows()

    # (1) DEFAULT mode: no covariance_mode argument. Must succeed and carry
    # the corrected provenance, with usable uncertainty on every surface.
    pred = model.predict(rows, interval=0.9)
    assert isinstance(pred, gamfit.CompetingRisksPrediction)
    assert pred.interval_level == 0.9
    assert pred.covariance_source == "smoothing-corrected", (
        "default covariance mode must be the smoothing-corrected matrix "
        f"(no silent fallback, #2296/#2346); got {pred.covariance_source!r}"
    )
    cif_se = np.asarray(pred.cif_se, dtype=float)
    assert np.all(np.isfinite(cif_se))
    assert np.any(cif_se > 0.0), "corrected CIF SEs must not be all zero"
    eta = np.asarray(pred.linear_predictor, dtype=float)
    eta_lower = np.asarray(pred.eta_lower, dtype=float)
    eta_upper = np.asarray(pred.eta_upper, dtype=float)
    assert np.all(eta_lower <= eta) and np.all(eta <= eta_upper)

    # (2) Explicit spelling agrees with the default bitwise on the point
    # surfaces and reports the same provenance.
    explicit = model.predict(rows, interval=0.9, covariance_mode="smoothing")
    assert explicit.covariance_source == "smoothing-corrected"
    np.testing.assert_array_equal(
        np.asarray(explicit.cif, dtype=float), np.asarray(pred.cif, dtype=float)
    )

    # (3) The corrected matrix strictly inflates over the conditional one
    # somewhere: conditional-mode SEs must never exceed corrected SEs beyond
    # roundoff, and at least one must strictly grow.
    conditional = model.predict(rows, interval=0.9, covariance_mode="conditional")
    assert conditional.covariance_source == "conditional"
    cond_se = np.asarray(conditional.cif_se, dtype=float)
    mask = np.isfinite(cond_se) & np.isfinite(cif_se)
    assert np.all(cif_se[mask] >= cond_se[mask] - 1e-10 * (1.0 + cond_se[mask]))
    assert np.any(cif_se[mask] > cond_se[mask] * (1.0 + 1e-9) + 1e-14), (
        "rho-uncertainty inflation must strictly widen at least one CIF SE"
    )
