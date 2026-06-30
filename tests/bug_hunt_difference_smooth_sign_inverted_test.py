"""Bug hunt: ``Model.difference_smooth`` returns the contrast with the WRONG
SIGN — for a requested pair ``(level_1, level_2)`` it reports ``f(level_2) −
f(level_1)`` while labelling the row ``level_1`` / ``level_2``, i.e. the
negation of the natural "level_1 minus level_2" reading.

``difference_smooth(view="x", group="g", pairs=[("B", "A")], ...)`` is the
covariance-aware contrast between the by-group smooths of ``s(x, by=g)``. For a
pair ``("B", "A")`` it returns rows tagged ``level_1="B"``, ``level_2="A"`` with
a ``diff`` column, which a user reads as ``ŝ_B(x) − ŝ_A(x)`` (the standard mgcv
``plot_diff(model, "B", "A")`` convention).

We build data where the true group-B minus group-A surface is a known, strictly
*increasing* function ``Δ(x) = 1.5·x`` (group B's response sits 1.5·x above group
A's). The full-model contrast ``predict(g="B") − predict(g="A")`` recovers this
(positive slope ~+1.5). ``difference_smooth(pairs=[("B","A")])`` should agree.

Observed: the returned ``diff`` is the exact negation — its correlation with
``predict("B") − predict("A")`` is **−1.000** across seeds and for both pair
orders, with the same magnitude. So requesting ("B","A") yields ŝ_A − ŝ_B while
the row still says ``level_1="B", level_2="A"``. The reported confidence band
inherits the flipped centre.

Root cause: ``crates/gam-pyffi/src/manifold/manifold_and_posterior_ffi.rs``,
``difference_smooth`` loop — ``row_left`` holds ``level_1`` and ``row_right``
holds ``level_2`` (lines ~1509-1514), but the design difference is formed as
``let mut xd = &xr - &xl;`` (line ~1532), i.e. ``design(level_2) −
design(level_1)``, so ``diff = xd·beta = f(level_2) − f(level_1)`` — the negation
of the labelled (level_1 − level_2) contrast. It should be ``&xl - &xr``.

Expected: ``diff`` for pair ``("B","A")`` equals ``ŝ_B − ŝ_A`` (positive slope
here), matching ``predict("B") − predict("A")``.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
pd: Any = pytest.importorskip("pandas")

import gamfit


def _fit(seed: int):
    rng = np.random.default_rng(seed)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    g = rng.choice(["A", "B"], n)
    # Group B sits Δ(x) = 1.5·x above group A (strictly increasing contrast).
    y = np.sin(2.0 * np.pi * x) + (g == "B") * (1.5 * x) + rng.normal(0.0, 0.2, n)
    df = pd.DataFrame({"y": y, "x": x, "g": g})
    return gamfit.fit(df, "y ~ g + s(x, by=g)")


def _diff_smooth_grid(model, pair):
    ds = model.difference_smooth(
        view="x", group="g", pairs=[pair], n=40, data=None
    )
    d = ds.to_dict() if hasattr(ds, "to_dict") else dict(ds)
    xs = np.array([d["x"][i] for i in sorted(d["x"])], dtype=float)
    diff = np.array([d["diff"][i] for i in sorted(d["diff"])], dtype=float)
    return xs, diff


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_difference_smooth_sign_matches_pair_order(seed: int) -> None:
    model = _fit(seed)
    xs, diff = _diff_smooth_grid(model, ("B", "A"))

    # Reference: the full-model contrast B - A on the same grid.
    pred_B = np.asarray(model.predict(pd.DataFrame({"x": xs, "g": ["B"] * len(xs)})))
    pred_A = np.asarray(model.predict(pd.DataFrame({"x": xs, "g": ["A"] * len(xs)})))
    contrast_BA = pred_B - pred_A  # this is the true ŝ_B − ŝ_A (positive slope)

    # Sanity: the reference contrast is strongly increasing (B above A at large x).
    assert contrast_BA[-1] - contrast_BA[0] > 0.5, (
        f"seed {seed}: reference B−A contrast is not increasing as designed "
        f"({contrast_BA[0]:.3f} -> {contrast_BA[-1]:.3f})"
    )

    corr = float(np.corrcoef(diff, contrast_BA)[0, 1])
    # difference_smooth("B","A") must equal predict(B)−predict(A), not its negation.
    assert corr > 0.99, (
        f"seed {seed}: difference_smooth(pairs=[('B','A')]) has the WRONG SIGN — "
        f"its diff correlates {corr:+.3f} with predict('B')−predict('A') "
        f"(diff[-1]={diff[-1]:+.3f} vs expected {contrast_BA[-1]:+.3f}). "
        "The pair (level_1, level_2) should report level_1 − level_2."
    )
