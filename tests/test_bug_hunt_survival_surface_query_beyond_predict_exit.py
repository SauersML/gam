"""Regression for #896: the predicted survival surface must not be truncated at
the prediction frame's ``exit`` placeholder.

``Model.predict(test_df)`` builds a survival surface on a default time grid. That
grid was previously bounded above by ``max(exit)`` of the *prediction rows*, so a
small placeholder ``exit`` shrank the grid and any later ``survival_at(t)`` query
past that placeholder fell through to the ``t -> inf`` asymptote (``S = 0``) even
for ``t`` well inside the fitted time range.

The fix (crates/gam-pyffi/src/lib.rs ``default_survival_time_grid`` +
``saved_survival_training_time_upper_bound``) anchors the grid's upper edge to
the fitted model's TRAINING time support, so the surface a caller queries is a
property of the model, not of the placeholder ``exit``. This test fits a model,
predicts with a deliberately tiny placeholder ``exit``, and asserts that
``survival_at`` at times far BEYOND that placeholder (but inside the fitted
range) keeps decreasing instead of collapsing to the asymptote.
"""

import importlib
from typing import Any, cast

pytest = cast(Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _make_dataset() -> pd.DataFrame:
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
        {"entry": np.zeros(n), "exit": exit_t, "event": event, "age": age}
    )


@pytest.mark.parametrize("likelihood", ["transformation", "weibull"])
def test_survival_surface_not_truncated_by_small_predict_exit(likelihood: str) -> None:
    df = _make_dataset()
    # The fitted data genuinely covers the query times below.
    assert df["exit"].max() > 12.0

    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ s(age)",
        survival_likelihood=likelihood,
    )

    # Tiny placeholder exit (1.0), far below the query times. Before the fix the
    # surface grid was [0, 1] and survival_at([2,5,10]) returned [0, 0, 0].
    small = pd.DataFrame(
        {"entry": [0.0], "exit": [1.0], "event": [1], "age": [60.0]}
    )
    big = pd.DataFrame(
        {"entry": [0.0], "exit": [18.0], "event": [1], "age": [60.0]}
    )

    times = [2.0, 5.0, 10.0]
    s_small = np.asarray(model.predict(small).survival_at(times))[0]
    s_big = np.asarray(model.predict(big).survival_at(times))[0]

    # (1) The surface must not depend on the placeholder exit.
    assert np.max(np.abs(s_small - s_big)) < 0.02, (
        f"[{likelihood}] survival surface depends on the predict-row `exit` "
        f"placeholder: S(exit=1)={np.round(s_small, 4).tolist()} vs "
        f"S(exit=18)={np.round(s_big, 4).tolist()} at {times}"
    )

    # (2) In-range query times past the placeholder must be real probabilities,
    # not the collapsed t->inf asymptote.
    assert np.all(s_small > 0.05), (
        f"[{likelihood}] survival at in-range times collapsed to ~0 because the "
        f"predict row carried exit=1.0: S={np.round(s_small, 4).tolist()}"
    )

    # (3) The curve must keep DECREASING across times that straddle the
    # placeholder exit -- the signature of a non-truncated surface.
    assert s_small[0] > s_small[1] > s_small[2], (
        f"[{likelihood}] survival surface is not strictly decreasing past the "
        f"placeholder exit: S={np.round(s_small, 4).tolist()} at {times}"
    )
