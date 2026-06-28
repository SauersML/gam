"""Regression for #1595: ``survival_at`` and ``cumulative_hazard_at`` must stay
mutually consistent past the fitted time grid.

A survival curve ``S`` and a cumulative hazard ``H`` are two views of the SAME
fitted curve, tied by the exact identity ``S(t) = exp(-H(t))`` at every ``t``.
On one ``SurvivalPrediction`` object the two accessors must therefore agree
under this identity for any query time -- including times past the last grid
node ``t_max``.

Previously the two surfaces extrapolated with contradictory rules
(``gamfit/_survival.py`` ``_SURVIVAL_EXTRAPOLATION``): ``survival`` was forced
to its ``t -> inf`` asymptote ``0.0`` immediately past ``t_max`` while
``cumulative_hazard`` flat-clamped to its last finite value ``H(t_max)``. The
two then satisfied ``exp(-H) = exp(-H(t_max)) > 0`` but ``S = 0`` -- a gross
contradiction, plus a spurious jump discontinuity in ``S`` at the grid edge.

The fix flat-clamps BOTH surfaces past the grid, so ``S`` and ``H`` keep
mirroring each other: ``S(t > t_max) = S(t_max) = exp(-H(t_max)) > 0``. This
test fits a Weibull right-censored model, predicts, and asserts the identity
holds at times spanning below / at / beyond ``t_max`` -- and that the right
asymptotes are consistent (S does not collapse to 0 while H stays finite).
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _make_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    age = rng.uniform(40.0, 75.0, n)
    eta = -2.0 + 0.04 * (age - 50.0)
    shape = 1.5
    u = rng.uniform(1e-9, 1.0, n)
    t_lat = np.exp(-eta / shape) * (-np.log(u)) ** (1.0 / shape) * 10.0
    cens = np.minimum(rng.exponential(25.0 / np.log(2), n), 25.0)
    exit_t = np.minimum(t_lat, cens)
    event = (t_lat <= cens).astype(int)
    return pd.DataFrame(
        {"entry": np.zeros(n), "exit": exit_t, "event": event, "age": age}
    )


@pytest.mark.parametrize(
    "likelihood", ["weibull", "transformation", "location-scale"]
)
def test_survival_at_consistent_with_cumulative_hazard_beyond_support(
    likelihood: str,
) -> None:
    df = _make_dataset()
    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ age",
        survival_likelihood=likelihood,
    )

    pred = model.predict(
        pd.DataFrame({"entry": [0.0], "exit": [5.0], "event": [1], "age": [50.0]})
    )

    # Top of the fitted grid is anchored to the TRAINING time support (#896),
    # not the exit=5 placeholder.
    t_max = float(np.asarray(pred.times).max())
    assert t_max > 12.0, f"unexpectedly small fitted grid: t_max={t_max}"

    # Times straddling below / at / just past / well past the last grid node.
    ts = np.array([0.5 * t_max, t_max, t_max + 1e-3, t_max + 5.0, t_max + 20.0])

    survival = np.asarray(pred.survival_at(ts))[0]
    cum_haz = np.asarray(pred.cumulative_hazard_at(ts))[0]
    survival_from_haz = np.exp(-cum_haz)

    gap = np.abs(survival - survival_from_haz)
    max_gap = float(np.max(gap))

    # (1) Core identity S(t) == exp(-H(t)) at EVERY query time, including the
    # three past-grid times. Tight tol -- the identity holds to ~1e-5 within the
    # grid and must not blow up past it.
    assert max_gap < 1e-4, (
        f"[{likelihood}] survival_at and cumulative_hazard_at violate "
        f"S(t) = exp(-H(t)) past the grid: max gap = {max_gap:.6f} at times "
        f"{ts.tolist()}; S={np.round(survival, 6).tolist()}, "
        f"exp(-H)={np.round(survival_from_haz, 6).tolist()}"
    )

    # (2) Right asymptotes must be consistent: S must NOT collapse to 0 while H
    # stays finite. Past a finite, flat-clamped H, survival stays strictly
    # positive.
    past_grid = ts > t_max
    assert np.all(np.isfinite(cum_haz[past_grid])), (
        f"[{likelihood}] cumulative hazard went non-finite past the grid: "
        f"H={cum_haz.tolist()}"
    )
    assert np.all(survival[past_grid] > 1e-6), (
        f"[{likelihood}] survival collapsed to ~0 past the grid while H stayed "
        f"finite ({cum_haz[past_grid].tolist()}): S={survival.tolist()}"
    )

    # (3) No jump discontinuity in S at the grid edge: the value just past
    # t_max matches the value at t_max.
    assert abs(survival[1] - survival[2]) < 1e-4, (
        f"[{likelihood}] survival jumps at the grid edge: "
        f"S(t_max)={survival[1]:.6f} -> S(t_max+eps)={survival[2]:.6f}"
    )
