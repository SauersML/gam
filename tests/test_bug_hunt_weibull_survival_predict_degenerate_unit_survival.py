"""Bug hunt: single-cause ``survival_likelihood="weibull"`` predicts a
degenerate survival surface ``S(t) ≡ 1`` (cumulative hazard ≈ 0).

On data where roughly 87% of subjects have an event and the median observed
time is ~2.5, the Weibull survival mode predicts ``S(t) = 1.0`` for every time
out to t=12 (cumulative hazard ~1e-12, i.e. essentially nobody ever fails).
Fitting the *same* data with ``survival_likelihood="transformation"`` recovers
a sensible survival curve that tracks the empirical survival closely, so the
data and formula are fine — the defect is specific to the Weibull mode.

Root cause (files read, no patch):

* ``src/families/survival_predict.rs`` evaluates Transformation and Weibull
  through the *same* ``evaluate_rp_row`` (lines ~540-553), differing only in the
  baseline time offset ``r_eta_exit`` / ``r_deriv_exit``. Weibull is explicitly
  exempted from carrying a structural (spline) time basis (line ~383:
  ``if saved_likelihood_mode != Weibull && !has_baseline_time_wiggle()``), so
  for Weibull the *entire* baseline log-cumulative-hazard comes from
  ``build_survival_time_offsets_for_likelihood(..., &baseline_cfg, Weibull, ...)``
  with ``baseline_cfg = saved_survival_runtime_baseline_config(model)``
  (lines ~387, ~502-509), which routes to ``build_survival_baseline_offsets``
  (``src/families/survival_construction.rs:3085``).
* The resulting predicted cumulative hazard collapses to ~0 (linear predictor
  pinned around -28), so ``S = exp(-H) = 1`` everywhere. This is the same class
  of "reconstructed from a non-fitted baseline config collapses to null" defect
  as the closed competing-risks issues #689 / #690, but on the single-cause
  Weibull prediction path, which has no test asserting non-degenerate survival
  values (the existing ``test_survival_location_scale_regressor_prediction_does_not_saturate``
  guards only the location-scale mode).

A large ``exit`` placeholder is used in the prediction frame so the requested
query times are inside the surface grid (isolating this from the separate
"surface truncated at the prediction row's exit" defect, Related: #896).

When the Weibull predicted survival reflects the fitted baseline, this test
passes without edits.
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _make_dataset() -> pd.DataFrame:
    """Deterministic right-censored data with high mortality.

    ~87% of subjects have an event and the median observed time is ~2.5, so any
    correctly-fitted survival model must predict survival well below 1 within
    the observed time range.
    """
    rng = np.random.default_rng(11)
    n = 500
    age = rng.uniform(40.0, 75.0, n)
    shape = 1.5
    eta = -2.0 + 0.05 * (age - 55.0)
    u = rng.uniform(1e-9, 1.0, n)
    t_lat = np.exp(-eta / shape) * (-np.log(u)) ** (1.0 / shape)
    cens = np.minimum(rng.exponential(20.0, n), 30.0)
    exit_t = np.minimum(t_lat, cens)
    event = (t_lat <= cens).astype(int)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": exit_t,
            "event": event,
            "age": age,
        }
    )


def test_weibull_survival_prediction_is_not_degenerate_unit_survival() -> None:
    df = _make_dataset()
    # The data genuinely has substantial mortality inside the query range.
    assert df["event"].mean() > 0.5
    assert float(np.median(df["exit"])) < 6.0

    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ s(age)",
        survival_likelihood="weibull",
    )

    # Large `exit` placeholder so every query time is inside the surface grid
    # (this isolates the Weibull degeneracy from the grid-truncation bug #896).
    big_exit = float(df["exit"].max()) + 5.0
    grid = [1.0, 3.0, 6.0, 12.0]  # all within the observed range
    pred = model.predict(
        pd.DataFrame(
            {"entry": [0.0], "exit": [big_exit], "event": [1], "age": [57.0]}
        )
    )
    surv = np.asarray(pred.survival_at(grid))[0]

    assert np.all(np.isfinite(surv))

    # With ~87% events and median observed time ~2.5, the survival surface must
    # be far from the degenerate S(t) == 1. A correctly-fitted Weibull baseline
    # predicts survival well below 1 within the observed range; the bug returns
    # S == 1.0 (cumulative hazard ~ 0) at every time.
    assert float(np.min(surv)) < 0.85, (
        "Weibull single-cause prediction collapsed to a degenerate unit "
        f"survival surface: S(age=57) at {grid} = {np.round(surv, 4).tolist()} "
        "(cumulative hazard ~ 0). The same data fits a sensible curve under "
        "survival_likelihood='transformation'."
    )

    # The surface must also be (weakly) decreasing in time.
    assert np.all(np.diff(surv) <= 1e-9), (
        f"Weibull survival surface is not monotone non-increasing: {surv.tolist()}"
    )
