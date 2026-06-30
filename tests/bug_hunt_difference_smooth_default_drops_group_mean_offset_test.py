"""Regression test for #1121.

`Model.difference_smooth(...)` with its default arguments must NOT silently drop
the constant between-group mean offset. A bare additive factor `+ g` is fitted
internally as a `random_effect` term; the default `marginalise_random=True` used
to zero `g`'s coefficient columns, overriding the default `group_means=True`
which is supposed to keep exactly those columns. The two defaults collided on
the same columns and the wrong one won, so the returned difference curve was
wrong by the entire between-group level offset.

With the fix, `group_means=True` is honoured over `marginalise_random` for the
compared factor's own columns, so the default difference smooth recovers BOTH
the level difference and the wiggle difference, matching the model's own
predicted between-group difference `predict(B) - predict(A)`.

Sign convention: a pair `(level_1, level_2)` yields the contrast
`ŝ(level_1) - ŝ(level_2)` (mgcv's `plot_diff(model, level_1, level_2)`; see the
companion `bug_hunt_difference_smooth_sign_inverted_test.py`). We therefore
request `pairs=[("B", "A")]` so the recovered `diff` equals `predict(B) -
predict(A)` (≈ +offset here); the magnitude — not the chosen direction — is what
the #1121 offset-recovery contract pins.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _frame(seed: int = 11, n: int = 2000, offset: float = 2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    g = np.array(["A", "B"])[rng.integers(0, 2, n)]
    # Shared smooth + a pure constant offset: B sits `offset` above A.
    y = np.sin(2 * np.pi * x) + np.where(g == "B", offset, 0.0) + rng.normal(0.0, 0.2, n)
    return pd.DataFrame({"x": x, "g": g, "y": y})


def test_default_difference_smooth_recovers_group_mean_offset() -> None:
    offset = 2.0
    df = _frame(offset=offset)
    model = gamfit.fit(df, "y ~ s(x) + g")

    # Default arguments: marginalise_random=True, group_means=True.
    # Pair (B, A) ⇒ diff = ŝ(B) − ŝ(A) = predict(B) − predict(A) ≈ +offset.
    d = model.difference_smooth(view="x", group="g", pairs=[("B", "A")], n=30)
    grid = d["x"].to_numpy()
    diff = d["diff"].to_numpy()

    # The model's own predicted between-group difference is the truth the
    # difference smooth must reproduce.
    pred_a = np.asarray(
        model.predict(pd.DataFrame({"x": grid, "g": ["A"] * len(grid)}))
    ).ravel()
    pred_b = np.asarray(
        model.predict(pd.DataFrame({"x": grid, "g": ["B"] * len(grid)}))
    ).ravel()
    predicted_diff = pred_b - pred_a

    assert np.all(np.isfinite(diff))

    # The default difference smooth must equal the model's own predicted
    # between-group difference at every grid point (level + wiggle), not just the
    # wiggle. predict(B) - predict(A) is ~ +offset across the grid here since the
    # data has only a constant offset and a shared smooth.
    np.testing.assert_allclose(diff, predicted_diff, atol=1e-6)

    # And that level difference must be the true constant offset (recovered, not
    # dropped to ~0 as the bug produced).
    assert abs(float(diff.mean()) - offset) < 0.1
    assert abs(float(predicted_diff.mean()) - offset) < 0.1


def test_marginalise_random_false_matches_default() -> None:
    # With the fix, marginalise_random no longer changes the compared factor's
    # group-mean columns when group_means is True, so the default (True) and the
    # explicit False must agree for this single-factor model.
    df = _frame()
    model = gamfit.fit(df, "y ~ s(x) + g")

    default = model.difference_smooth(view="x", group="g", pairs=[("B", "A")], n=30)
    explicit = model.difference_smooth(
        view="x",
        group="g",
        pairs=[("B", "A")],
        n=30,
        marginalise_random=False,
    )

    np.testing.assert_allclose(
        default["diff"].to_numpy(), explicit["diff"].to_numpy(), atol=1e-6
    )
