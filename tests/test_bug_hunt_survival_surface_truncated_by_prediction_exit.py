"""Bug hunt: the predicted survival surface is silently truncated at each
prediction row's ``exit`` placeholder.

``Model.predict(test_df)`` returns a ``SurvivalPrediction`` whose time grid is
chosen by ``default_survival_time_grid`` (crates/gam-pyffi/src/lib.rs around
line 2112-2171) as ``[min entry, max exit]`` of the *prediction rows*, sampled
at 64 points. The user then asks for the survival curve at times chosen
*after* predict via ``pred.survival_at(times)`` / ``cumulative_hazard_at(times)``.

Any requested time beyond that grid is handed to the asymptotic-extrapolation
logic in ``gamfit/_survival.py`` (``_SURVIVAL_EXTRAPOLATION`` lines 730-743),
which forces ``survival -> 0.0`` (the ``t -> inf`` asymptote) and clamps the
cumulative hazard to its last grid value. So a perfectly ordinary query time
well inside the *fitted* data range is mistreated as "t = infinity" purely
because the prediction frame happened to carry a small ``exit`` placeholder.

Concretely: fitting on data with exit times up to ~20 and asking for the
survival of a 60-year-old at t = 2, 5, 10 returns ``[0, 0, 0]`` when the
prediction row's ``exit`` is 1.0, but a sensible decreasing curve when ``exit``
is 18.0 — even though the covariates and query times are identical. The
predicted survival surface must not depend on the ``exit`` placeholder column
of the prediction frame.

When the bug is fixed (the surface evaluated for ``survival_at`` must cover the
requested query times regardless of the prediction frame's placeholder ``exit``
values, rather than silently extrapolating interior times to the t->inf
asymptote), this test passes without edits.
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _make_dataset() -> pd.DataFrame:
    """Deterministic right-censored Weibull-ish survival data.

    Hazard rises with age; exit times span roughly [0, 20] with a healthy mix
    of events and censoring, so a smooth-baseline transformation fit converges.
    """
    rng = np.random.default_rng(0)
    n = 600
    age = rng.uniform(40.0, 80.0, n)
    shape = 1.5
    scale = np.exp(-(age - 60.0) / 20.0) * 10.0
    u = rng.uniform(0.0, 1.0, n)
    t_lat = scale * (-np.log(u)) ** (1.0 / shape)
    cens = rng.uniform(0.0, 20.0, n)
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


def test_survival_surface_is_invariant_to_prediction_exit_placeholder() -> None:
    df = _make_dataset()
    # Sanity: the fitted data genuinely covers the query times below.
    assert df["exit"].max() > 12.0

    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ s(age)",
        survival_likelihood="transformation",
    )

    # Query times are all strictly inside the fitted time range (max exit ~20),
    # so the model has a well-defined, non-degenerate survival surface here.
    times = [2.0, 5.0, 10.0]

    # Identical covariates; only the placeholder `exit` differs. The first row's
    # placeholder is smaller than the smallest query time; the second's is
    # larger than the largest query time.
    covariate = 60.0
    small_exit = pd.DataFrame(
        {"entry": [0.0], "exit": [1.0], "event": [1], "age": [covariate]}
    )
    large_exit = pd.DataFrame(
        {"entry": [0.0], "exit": [18.0], "event": [1], "age": [covariate]}
    )

    s_small = np.asarray(model.predict(small_exit).survival_at(times))[0]
    s_large = np.asarray(model.predict(large_exit).survival_at(times))[0]

    # The survival curve for fixed covariates at fixed times must not depend on
    # the prediction frame's placeholder `exit`.
    assert np.max(np.abs(s_small - s_large)) < 0.02, (
        "predicted survival surface depends on the prediction row's `exit` "
        f"placeholder: S(exit=1)={np.round(s_small, 4).tolist()} vs "
        f"S(exit=18)={np.round(s_large, 4).tolist()} at times {times}"
    )

    # And the survival at an interior time (t=5, well within the data range)
    # must be a sensible probability, not collapsed to the t->inf asymptote 0.
    assert s_small[1] > 0.05, (
        "survival at t=5 (inside the fitted range) collapsed to ~0 because the "
        f"prediction row carried exit=1.0: S={np.round(s_small, 4).tolist()}"
    )
