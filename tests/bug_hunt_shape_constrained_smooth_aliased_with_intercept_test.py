"""Regression test: a shape-constrained ``s()`` was aliased with the intercept.

The monotone / convex / concave chart ``β = T·δ`` kept ``T[:, 0] = ±1`` as a
free "level" coordinate. A clamped B-spline basis is a partition of unity, so
that level column reproduces the intercept column exactly. The model then had
two coordinates for one constant direction. The only thing that pinned the
split between them was the double penalty's null-space ridge. So the
"intercept" was not ``mean(y)`` and the term did not sum to zero, unlike every
unconstrained smooth.

The chart now drops the level and subtracts each increment column's weighted
training mean. Coefficient differences, and so the cone ``δ ≥ 0``, are
unchanged. For a Gaussian identity fit with a centred term, the intercept's
conditional posterior mean given ``δ`` is ``mean(y)`` for every ``δ``, so the
posterior mean of the intercept is exactly ``mean(y)``. The term's
training-row mean is exactly zero.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _data(truth: str, seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    if truth == "inc_step":
        f = 1.0 / (1.0 + np.exp(-30.0 * (x - 0.5)))
    elif truth == "dec_exp":
        f = np.exp(-4.0 * x)
    elif truth == "convex":
        f = (x - 0.3) ** 2
    else:
        f = np.sqrt(x)
    return pd.DataFrame({"x": x, "y": 2.0 + f + rng.normal(0.0, 0.3, n)})


def _intercept_and_term(model: Any, df: pd.DataFrame) -> tuple[float, np.ndarray]:
    design = model.design_matrix(df)
    blocks = {b.kind: b for b in model.term_blocks}
    assert set(blocks) == {"intercept", "smooth_bspline1d"}, f"unexpected terms {blocks}"
    beta = np.asarray(design.coefficients, dtype=float)
    x_mat = np.asarray(design.matrix, dtype=float)
    icpt = blocks["intercept"]
    assert icpt.end - icpt.start == 1
    smooth = slice(blocks["smooth_bspline1d"].start, blocks["smooth_bspline1d"].end)
    return float(beta[icpt.start]), x_mat[:, smooth] @ beta[smooth]


# Each seed converges before and after this change. Truths with a long flat
# tail (``exp(-4x)``) often stop on a dominated certified plateau on either
# build; that outer-convergence failure is tracked on its own.
@pytest.mark.parametrize(
    ("truth", "shape", "seed"),
    [
        ("inc_step", "monotone_increasing", 102),
        ("dec_exp", "monotone_decreasing", 107),
        ("convex", "convex", 102),
        ("sqrt", "concave", 103),
    ],
)
def test_shape_constrained_term_is_centred_and_intercept_is_mean_y(
    truth: str, shape: str, seed: int
) -> None:
    df = _data(truth, seed=seed)
    model = gamfit.fit(df, f"y ~ s(x, shape={shape})")
    intercept, term = _intercept_and_term(model, df)
    y_mean = float(df["y"].mean())
    spread = float(np.ptp(term))
    assert spread > 0.1, f"{shape} term collapsed to a constant (range {spread:.3g})"
    assert abs(float(term.mean())) <= 1e-8 * max(1.0, spread), (
        f"{shape} term is not centred: training mean {term.mean():.3e}"
    )
    assert abs(intercept - y_mean) <= 1e-6 * max(1.0, abs(y_mean)), (
        f"{shape} intercept {intercept:.6f} is not mean(y) = {y_mean:.6f}; "
        "the smooth still carries a level aliased with the intercept"
    )


def test_unconstrained_control_has_the_same_gauge() -> None:
    """Control: the unconstrained smooth already had this gauge."""
    df = _data("inc_step", seed=102)
    model = gamfit.fit(df, "y ~ s(x)")
    intercept, term = _intercept_and_term(model, df)
    assert abs(float(term.mean())) <= 1e-8 * max(1.0, float(np.ptp(term)))
    assert abs(intercept - float(df["y"].mean())) <= 1e-6 * max(1.0, abs(float(df["y"].mean())))
