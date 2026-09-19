"""Shape constraints on tensor products, conjunctions and ``by=`` smooths.

Before this change a ``shape=`` option was accepted only on a plain 1-D
B-spline smooth: ``te(x, z, shape=...)``, ``s(x, by=g, shape=...)`` and any
combination such as ``shape=[monotone_increasing, concave]`` were refused.

* ``te()`` takes one entry per margin; each constrained margin becomes the
  exact B-spline cone Kroneckered with identities on the other margins, so
  the fitted surface is monotone (or convex/concave) along that margin
  everywhere, not only at sample points.
* A list of atoms on one term is a conjunction: every property holds, the
  redundant monotone rows are dropped, and contradictions (increasing and
  decreasing, convex and concave) are refused with an explanation.
* ``by=<factor>`` repeats the cone for every level's curve; ``by=<numeric>``
  constrains ``f`` in ``z·f(x)``.

Every assertion is on the fitted function over a dense grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit
from gamfit.smooth import shape_constraint_text

TOL = 1e-6


def _surface_data(seed: int = 3, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    # Increasing in x for every z, but with z-dependent slope; oscillating in z.
    f = np.tanh(3.0 * (x - 0.5)) * (1.0 + 0.5 * z) + 0.6 * np.sin(2.0 * np.pi * z)
    y = f + rng.normal(0.0, 0.25, n)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def _grid(n: int = 41) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    gx = np.linspace(0.0, 1.0, n)
    gz = np.linspace(0.0, 1.0, n)
    xx, zz = np.meshgrid(gx, gz, indexing="ij")
    return gx, gz, pd.DataFrame({"x": xx.ravel(), "z": zz.ravel()})


def test_tensor_surface_is_monotone_along_the_constrained_margin() -> None:
    df = _surface_data()
    model = gamfit.fit(df, "y ~ te(x, z, shape=[monotone_increasing, none])")
    gx, gz, grid = _grid()
    surf = np.asarray(model.predict(grid), dtype=float).reshape(gx.size, gz.size)
    assert np.all(np.isfinite(surf))
    steps_x = np.diff(surf, axis=0)
    assert steps_x.min() >= -TOL, f"surface decreases along x: {steps_x.min():.3e}"
    # The unconstrained margin keeps its oscillation.
    assert np.ptp(surf[gx.size // 2, :]) > 0.5
    assert np.diff(surf[gx.size // 2, :]).min() < -1e-3


def test_tensor_margin_curvature_constraint_holds_on_the_grid() -> None:
    rng = np.random.default_rng(8)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = (x - 0.4) ** 2 * (1.0 + z) + 0.5 * np.cos(3.0 * z) + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    model = gamfit.fit(df, "y ~ te(x, z, shape=[convex, monotone_decreasing])")
    gx, gz, grid = _grid()
    surf = np.asarray(model.predict(grid), dtype=float).reshape(gx.size, gz.size)
    second_x = np.diff(surf, n=2, axis=0)
    steps_z = np.diff(surf, axis=1)
    assert second_x.min() >= -TOL, f"surface not convex in x: {second_x.min():.3e}"
    assert steps_z.max() <= TOL, f"surface increases along z: {steps_z.max():.3e}"


def test_monotone_and_concave_conjunction_holds_both_properties() -> None:
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    # Increasing and concave, with noise large enough that an unconstrained
    # fit wiggles.
    y = np.sqrt(x) + rng.normal(0.0, 0.15, n)
    df = pd.DataFrame({"x": x, "y": y})
    model = gamfit.fit(df, "y ~ s(x, shape=[monotone_increasing, concave])")
    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 401)})
    pred = np.asarray(model.predict(grid), dtype=float)
    assert np.diff(pred).min() >= -TOL, "conjunction fit is not monotone"
    assert np.diff(pred, n=2).max() <= TOL, "conjunction fit is not concave"
    assert pred[-1] - pred[0] > 0.5


def test_constraints_mapping_accepts_a_list_value() -> None:
    rng = np.random.default_rng(12)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    y = np.sqrt(x) + rng.normal(0.0, 0.15, n)
    df = pd.DataFrame({"x": x, "y": y})
    via_mapping = gamfit.fit(
        df, "y ~ s(x)", constraints={"s(x)": ["monotone_increasing", "concave"]}
    )
    via_formula = gamfit.fit(df, "y ~ s(x, shape=[monotone_increasing, concave])")
    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 101)})
    np.testing.assert_allclose(
        np.asarray(via_mapping.predict(grid)),
        np.asarray(via_formula.predict(grid)),
        rtol=1e-8,
        atol=1e-8,
    )


def test_every_factor_by_level_curve_satisfies_its_constraint() -> None:
    rng = np.random.default_rng(21)
    n_per = 250
    frames = []
    truths = {
        "a": lambda x: np.log1p(4.0 * x),
        "b": lambda x: 2.0 * x**3,
        "c": lambda x: np.tanh(6.0 * (x - 0.5)),
    }
    for level, f in truths.items():
        x = rng.uniform(0.0, 1.0, n_per)
        frames.append(
            pd.DataFrame({"x": x, "g": level, "y": f(x) + rng.normal(0.0, 0.3, n_per)})
        )
    df = pd.concat(frames, ignore_index=True)
    df["g"] = df["g"].astype("category")
    model = gamfit.fit(df, "y ~ g + s(x, by=g, shape=monotone_increasing)")
    gx = np.linspace(0.0, 1.0, 301)
    for level, f in truths.items():
        grid = pd.DataFrame(
            {"x": gx, "g": pd.Categorical([level] * gx.size, categories=list(truths))}
        )
        pred = np.asarray(model.predict(grid), dtype=float)
        assert np.all(np.isfinite(pred))
        assert np.diff(pred).min() >= -TOL, f"level {level} curve decreases"
        # Each level's curve tracks its own truth up to a level offset.
        truth = f(gx)
        centred_err = (pred - pred.mean()) - (truth - truth.mean())
        assert np.sqrt(np.mean(centred_err**2)) < 0.2, f"level {level} curve is off"


def test_numeric_by_constrains_the_multiplied_function() -> None:
    rng = np.random.default_rng(31)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    w = rng.uniform(0.5, 2.0, n)
    y = w * np.sqrt(x) + rng.normal(0.0, 0.2, n)
    df = pd.DataFrame({"x": x, "w": w, "y": y})
    model = gamfit.fit(df, "y ~ s(x, by=w, shape=monotone_increasing)")
    gx = np.linspace(0.0, 1.0, 301)
    # z·f(x) inherits the shape of f exactly where z ≥ 0.
    for wv in (0.5, 1.0, 2.0):
        pred = np.asarray(model.predict(pd.DataFrame({"x": gx, "w": wv})), dtype=float)
        assert np.diff(pred).min() >= -TOL, f"w={wv}: z·f(x) decreases"


@pytest.mark.parametrize(
    ("formula", "needle"),
    [
        ("y ~ s(x, shape=[monotone_increasing, monotone_decreasing])", "constant"),
        ("y ~ s(x, shape=[convex, concave])", "affine"),
        ("y ~ te(x, z, shape=[monotone_increasing])", "2 margins"),
        ("y ~ ti(x, z, shape=[monotone_increasing, none])", "ti() removes"),
    ],
)
def test_unsatisfiable_or_ill_formed_shapes_are_refused_by_name(
    formula: str, needle: str
) -> None:
    df = _surface_data(n=200)
    with pytest.raises(Exception) as excinfo:
        gamfit.fit(df, formula)
    assert needle in str(excinfo.value), str(excinfo.value)


def test_python_shape_values_render_in_the_formula_grammar() -> None:
    assert shape_constraint_text(None) == "none"
    assert shape_constraint_text("convex") == "convex"
    assert (
        shape_constraint_text(["monotone_increasing", None])
        == "[monotone_increasing, none]"
    )
    assert (
        shape_constraint_text([["monotone_increasing", "concave"], "none"])
        == "[[monotone_increasing, concave], none]"
    )
