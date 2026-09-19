"""The ``Smooth.shape_constraint`` docstring must describe what the engine does.

The docstring used to say the shape-constraint matrix was "generated from the
basis on a dense 1D grid". The engine never used a grid: the constraint is the
exact control-polygon cone on the raw B-spline coefficients
(``crates/gam-terms/src/basis/shape_constraints.rs``), and it is admitted only
for open, non-periodic B-spline smooths. These tests pin the documented
contract to the observed behaviour:

* the docstring states the exact cone and no grid construction;
* a monotone / convex fit holds its shape on an evaluation grid far finer
  than any construction grid would be, including between clustered knots;
* the smooth kinds the docstring says are rejected actually raise.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

from gamfit.smooth import BSpline, Duchon, Smooth


def _shape_constraint_doc() -> str:
    doc = Smooth.__doc__ or ""
    start = doc.index("shape_constraint :")
    return doc[start:]


def test_shape_constraint_docstring_describes_exact_cone_not_a_grid() -> None:
    doc = " ".join(_shape_constraint_doc().split())
    assert "dense 1D grid" not in doc
    assert "generated from the basis" not in doc
    assert "no evaluation grid" in doc
    assert "control points" in doc
    assert "γ_j ≥ 0" in doc
    assert "periodic=False" in doc


def _clustered_frame(seed: int = 7, n: int = 400) -> pd.DataFrame:
    # Half the data packed into [0, 0.1] so quantile knots are strongly
    # non-uniform; the truth has a sharp rise there and a flat tail.
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(0.0, 0.1, n // 2), rng.uniform(0.1, 1.0, n // 2)])
    y = np.tanh(40.0 * (x - 0.05)) + rng.normal(0.0, 0.15, x.size)
    return pd.DataFrame({"x": x, "y": y})


def _fine_grid(df: pd.DataFrame, n: int = 20001) -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(df["x"].min(), df["x"].max(), n)})


def test_monotone_fit_is_monotone_on_a_grid_finer_than_any_construction_grid() -> None:
    pytest.importorskip("gamfit._rust")
    import gamfit

    df = _clustered_frame()
    model = gamfit.fit(
        df,
        "y ~ s(x)",
        smooths={"x": BSpline(shape_constraint="monotone_increasing")},
    )
    pred = np.asarray(model.predict(_fine_grid(df)), dtype=float)
    scale = float(np.ptp(pred))
    assert scale > 0.5, "fit collapsed; the shape check below would be vacuous"
    # The cone certifies f' ≥ 0 on every knot span, so no step of a 20001-point
    # grid may decrease beyond floating-point rounding of the prediction.
    assert np.min(np.diff(pred)) >= -1e-10 * scale


def test_convex_fit_is_convex_on_a_grid_finer_than_any_construction_grid() -> None:
    pytest.importorskip("gamfit._rust")
    import gamfit

    rng = np.random.default_rng(11)
    x = np.concatenate([rng.uniform(0.0, 0.15, 200), rng.uniform(0.15, 1.0, 200)])
    y = np.abs(x - 0.4) + 0.3 * np.sin(12.0 * x) + rng.normal(0.0, 0.05, x.size)
    df = pd.DataFrame({"x": x, "y": y})
    model = gamfit.fit(
        df,
        "y ~ s(x)",
        smooths={"x": BSpline(shape_constraint="convex")},
    )
    grid = _fine_grid(df)
    pred = np.asarray(model.predict(grid), dtype=float)
    h = float(grid["x"].iloc[1] - grid["x"].iloc[0])
    second = np.diff(pred, 2) / (h * h)
    curvature_scale = float(np.max(np.abs(second)))
    assert curvature_scale > 0.0
    # Rounding in a second difference is ~eps·|f|/h²; allow a few ulps of that.
    rounding = 16.0 * np.finfo(float).eps * float(np.max(np.abs(pred))) / (h * h)
    assert np.min(second) >= -rounding


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(
            lambda: Duchon(
                centers=np.linspace(0.0, 1.0, 8).reshape(-1, 1),
                m=2,
                shape_constraint="monotone_increasing",
            ),
            id="duchon",
        ),
        pytest.param(
            lambda: BSpline(periodic=True, shape_constraint="monotone_increasing"),
            id="periodic_bspline",
        ),
    ],
)
def test_documented_unsupported_kinds_reject_shape_constraint(spec: Any) -> None:
    pytest.importorskip("gamfit._rust")
    import gamfit

    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, 200)
    df = pd.DataFrame({"x": x, "y": x + rng.normal(0.0, 0.1, x.size)})
    with pytest.raises(Exception, match=r"(?i)shape"):
        gamfit.fit(df, "y ~ s(x)", smooths={"x": spec()})
