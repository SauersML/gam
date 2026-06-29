"""Regression for #965: survival predict-query times must accept negative and
``+inf`` values instead of rejecting them as invalid.

The bug: ``survival_coerce_times`` rejected any non-finite OR negative query
time with "survival prediction times must be finite and non-negative". But
``S(t) = P(T > t)`` is a well-defined function on ALL of the reals:

  * ``t <= 0``  -> ``S = 1`` (everyone is still at risk before the origin),
  * ``t = +inf`` -> ``S = 0`` (the genuine right asymptote for any model with
    positive total hazard).

Rejecting these is wrong: a user asking "what fraction survive past the start"
or "what is the limiting survival" gets an exception instead of ``1.0`` / ``0.0``.

This is DISTINCT from the finite past-grid flat-clamp of #1595 (covered by
``test_bug_hunt_survival_at_inconsistent_with_cumulative_hazard_beyond_support_test``):
a finite ``t > t_max`` flat-clamps to ``S(t_max) > 0``; only the genuine ``+inf``
query reaches the asymptote ``0``.

The fix (``gamfit/_survival.py`` + ``crates/gam-pyffi/.../survival_surface_io.rs``):
  * ``survival_coerce_times`` rejects only NaN now,
  * the interpolation kernels carry a separate ``inf_value`` so ``+inf`` hits the
    asymptote while finite past-grid queries still flat-clamp.
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

from gamfit._survival import SurvivalPrediction


def _toy_prediction() -> SurvivalPrediction:
    # A single individual whose survival curve drops 0.8 -> 0.5 over t in [1, 2].
    return SurvivalPrediction(
        model_class="survival",
        parameters=np.zeros((1, 2), dtype=float),
        times=np.array([1.0, 2.0], dtype=float),
        survival=np.array([[0.8, 0.5]], dtype=float),
    )


def test_negative_and_infinite_survival_query_times_are_accepted() -> None:
    surv = _toy_prediction()

    # The whole point of #965: these must NOT raise. Previously a ValueError
    # "survival prediction times must be finite and non-negative" fired here.
    row = np.asarray(surv.survival_at(np.array([-5.0, -2.0, 0.0, np.inf], dtype=float)))[0]

    assert row[0] == pytest.approx(1.0), "S(t=-5) must be 1.0 (before the origin)."
    assert row[1] == pytest.approx(1.0), "S(t=-2) must be 1.0 (before the origin)."
    assert row[2] == pytest.approx(1.0), "S(t=0) must be 1.0 at the origin."
    assert row[3] == pytest.approx(0.0), "S(+inf) must be exactly 0.0 (asymptote)."


def test_finite_past_grid_does_not_collapse_to_the_infinite_asymptote() -> None:
    # #965 must not regress #1595: a FINITE time past the last grid node
    # flat-clamps to the last grid value (0.5 here), it does NOT jump to the
    # +inf asymptote 0.0. Only the genuine +inf query reaches 0.0.
    surv = _toy_prediction()
    row = np.asarray(surv.survival_at(np.array([100.0, np.inf], dtype=float)))[0]
    assert row[0] == pytest.approx(0.5), (
        "finite past-grid time must flat-clamp to S(t_max)=0.5, not jump to 0.0"
    )
    assert row[1] == pytest.approx(0.0), "only +inf reaches the asymptote 0.0"


def test_cumulative_hazard_infinite_query_is_the_plus_infinity_asymptote() -> None:
    # The dual view: H(+inf) = +inf for any model with positive total hazard,
    # while a finite past-grid time flat-clamps to the finite H(t_max).
    surv = _toy_prediction()
    cum = np.asarray(surv.cumulative_hazard_at(np.array([100.0, np.inf], dtype=float)))[0]
    assert np.isfinite(cum[0]), "finite past-grid H must stay finite (flat-clamp)"
    assert np.isinf(cum[1]) and cum[1] > 0.0, "H(+inf) must be +inf"


def test_nan_survival_query_time_is_still_rejected() -> None:
    # NaN remains invalid: it is not a point on the real line, so there is no
    # survival value to return. The coercion must still reject it.
    surv = _toy_prediction()
    with pytest.raises((ValueError, RuntimeError), match="(?i)nan"):
        surv.survival_at(np.array([np.nan], dtype=float))
