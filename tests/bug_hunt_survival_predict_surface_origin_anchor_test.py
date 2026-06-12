"""Regression for #1024 from a different angle: the survival prediction surface
must be evaluable at its natural origin ``t = 0``.

The original repro (``bug_hunt_royston_parmar_predict_fails_on_valid_data_test``)
shows ``family="royston-parmar"`` aborting on ``predict`` because the default
surface grid's first node is the origin and the probit-survival baseline used by
the (location-scale) Royston-Parmar model had no ``age == 0`` handling, so the
shared ``age <= 0`` hazard guard rejected it.

This test attacks the same root cause from two angles the original does not:

1. It also drives ``survival_likelihood="location-scale"`` directly — the issue
   thread notes the same guard fires through that documented entry point, on the
   same shared predict-path defect.
2. It asserts the *origin anchor itself*: ``S(0) == 1`` and ``H(0) == 0``
   exactly (everyone is at risk at ``t = 0``), and that ``survival_at(0)`` and
   ``cumulative_hazard_at(0)`` return those exact values — rather than only
   probing strictly-positive query times.

Both modes must produce a well-posed survival surface: finite, in ``[0, 1]``,
monotone non-increasing in time, and ordered correctly across the covariate
(a longer Weibull scale gives higher survival).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _weibull_survival_frame(seed: int = 20260611, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    scale = np.exp(0.5 + 1.0 * x)  # longer scale -> longer survival
    shape = 1.3
    event_time = scale * rng.weibull(shape, n)
    censor_time = rng.exponential(6.0, n)
    observed = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return pd.DataFrame({"time": observed, "event": event, "x": x})


@pytest.mark.parametrize(
    "fit_kwargs",
    [
        {"family": "royston-parmar"},
        {"survival_likelihood": "location-scale"},
    ],
)
def test_survival_surface_is_anchored_at_the_origin(fit_kwargs: dict[str, Any]) -> None:
    df = _weibull_survival_frame()
    new_data = pd.DataFrame(
        {"time": [3.0, 3.0], "event": [1.0, 1.0], "x": [0.2, 0.8]}
    )

    model = gamfit.fit(df, "Surv(time, event) ~ x", **fit_kwargs)
    prediction = model.predict(new_data)

    # --- The origin anchor: S(0) = 1 and H(0) = 0 exactly. ---
    survival_at_origin = np.asarray(prediction.survival_at([0.0]), dtype=float)
    cumulative_at_origin = np.asarray(
        prediction.cumulative_hazard_at([0.0]), dtype=float
    )
    assert survival_at_origin.shape == (2, 1)
    assert np.allclose(survival_at_origin, 1.0, atol=1e-12)
    assert np.allclose(cumulative_at_origin, 0.0, atol=1e-12)

    # --- The full surface, including the origin column, is well-posed. ---
    query_times = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    survival = np.asarray(prediction.survival_at(query_times), dtype=float)
    assert survival.shape == (2, query_times.size)
    assert np.all(np.isfinite(survival))
    assert np.all(survival >= -1e-9)
    assert np.all(survival <= 1.0 + 1e-9)
    # S(t) is monotone non-increasing in t for every subject, and S(0) = 1.
    assert np.allclose(survival[:, 0], 1.0, atol=1e-12)
    assert np.all(np.diff(survival, axis=1) <= 1e-9)

    # --- The covariate effect: a longer Weibull scale -> higher survival. ---
    # Compare at an interior time (index 4 -> t = 2.0).
    assert survival[1, 4] > survival[0, 4]
