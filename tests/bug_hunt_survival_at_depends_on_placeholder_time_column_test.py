"""Bug hunt: ``SurvivalPrediction.survival_at`` is not invariant to the
placeholder response (``time``) column carried in the prediction frame.

A fitted survival model defines a survival surface ``S(t | x)`` that is a pure
function of the covariates ``x`` and the query time ``t``. To call
``model.predict(frame)`` the formula parser requires the ``Surv(time, event)``
*response* columns to be present in ``frame``, but those values are semantically
meaningless for a covariate-only survival query: ``survival_at(query_times)``
supplies its own times, so two prediction frames with identical covariates and
queried at identical times **must** return identical survival probabilities.

They do not. ``crates/gam-pyffi/src/model/model_ffi.rs::default_survival_time_grid``
builds the internal surface grid as 64 *uniform* points spanning
``[lo, max(prediction-frame max exit, training_hi)]`` (lines ~1140 and ~1167):

    hi = hi.max(exit_value);          // prediction-frame placeholder wins...
    ...
    let step = (hi_padded - lo) / 63.0;
    (0..64).map(|i| lo + step * (i as f64))

``survival_at`` then *linearly interpolates* that 64-cell surface (see
``crates/gam-pyffi/src/io/survival_surface_io.rs::interpolate_survival_surface``).
Issue #896 fixed the case where a *small* placeholder ``exit`` shrank the grid
below the fitted range (the surface was truncated and ``S`` fell to its
``t -> inf`` asymptote). But a *large* placeholder ``exit`` does the opposite:
it stretches the 64 uniform points far past the training support, coarsening
every grid cell inside the fitted range. An in-range query time then falls in a
much wider linear-interpolation cell and the returned ``S(t | x)`` drifts.

This directly contradicts the design invariant stated in that same file
(model_ffi.rs:927): "The default surface grid must be a property of the FITTED
model, not of the ``exit`` placeholder a caller happens to put in the prediction
frame." The #896 anchoring only covers the lower edge; the upper edge is still
taken from the prediction frame.

Reproduction (deterministic): fit a transformation survival model whose training
exit times top out near ~13.7, then query the *same* covariates at the *same*
in-range times ``[0.5, 1, 2, 3, 4]`` from two frames that differ only in the
placeholder ``time`` column (``6.0`` vs ``120.0``). The two ``survival_at``
results disagree by ~3% of the probability scale — a moderate, perfectly
reasonable follow-up horizon silently changes the answer.

When the grid is anchored to the fitted model's time support (independent of the
placeholder), both frames yield the same surface and this test passes without
edits. Related: #1595 (survival extrapolation past the grid), #896.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _survival_frame(seed: int = 20240630, n: int = 3000) -> pd.DataFrame:
    """Weibull survival with a single covariate effect; right-censored."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    scale = np.exp(0.5 + 1.0 * x)  # longer scale -> longer survival
    shape = 1.3
    event_time = scale * rng.weibull(shape, n)
    censor_time = rng.exponential(6.0, n)
    observed = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return pd.DataFrame({"time": observed, "event": event, "x": x})


def test_survival_at_is_invariant_to_placeholder_time_column() -> None:
    df = _survival_frame()
    model = gamfit.fit(
        df, "Surv(time, event) ~ x", survival_likelihood="transformation"
    )

    # Query times all sit well inside the fitted range (training exit max ~13.7).
    query_times = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    covariates = [0.25, 0.75]

    # Two prediction frames: identical covariates, identical query times, but the
    # required-yet-ignored placeholder `time` column differs. Both placeholders
    # are ordinary, finite follow-up times.
    frame_small = pd.DataFrame(
        {"time": [6.0, 6.0], "event": [0.0, 0.0], "x": covariates}
    )
    frame_large = pd.DataFrame(
        {"time": [120.0, 120.0], "event": [0.0, 0.0], "x": covariates}
    )

    s_small = np.asarray(
        model.predict(frame_small).survival_at(query_times), dtype=float
    )
    s_large = np.asarray(
        model.predict(frame_large).survival_at(query_times), dtype=float
    )

    assert s_small.shape == (2, query_times.size)
    assert np.all(np.isfinite(s_small)) and np.all(np.isfinite(s_large))

    # The defining invariant: S(t | x) is a property of the fitted model and the
    # query time, not of the placeholder response column. The two surfaces must
    # agree to tight tolerance. They currently differ by ~3% of probability.
    max_abs_diff = float(np.max(np.abs(s_small - s_large)))
    assert max_abs_diff < 1e-3, (
        "survival_at must not depend on the placeholder `time` column; "
        f"max |S_small - S_large| = {max_abs_diff:.4f} over in-range query "
        f"times {query_times.tolist()}\nS_small=\n{s_small}\nS_large=\n{s_large}"
    )
