"""``Model.partial_dependence`` builds its grid from the saved term specification
(#2899 P8), so it takes no reference table.

- The control for deleting ``data``: a term's design block reads only the term's own
  axes, plus a by column its specification records. Frames that agree on those axes
  and differ in every other column give bitwise identical block columns, and the
  curve equals that block times its coefficients.
- A linear term sweeps its own column. The old label parser found no axis in ``"x"``
  and refused a default grid.
- A numeric ``by=z`` smooth reports its coefficient function ``f(x)``, which is the
  block at ``z = 1``. Refitting on reordered training rows gives the same curve.
"""

import numpy as np
import pandas as pd

import gamfit


def _beta_and_cov(model):
    summary = model.summary()
    beta = np.asarray([c["estimate"] for c in summary.coefficients], dtype=float)
    cov = summary.covariance
    return beta, cov


def _block_columns(model, term, frame):
    block = next(b for b in model.term_blocks if b.name == term)
    design = np.asarray(model.design_matrix(frame).matrix, dtype=float)
    return block, design[:, block.start : block.end]


def test_other_columns_never_reach_a_terms_design_block():
    rng = np.random.default_rng(7)
    n = 300
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 1.0, n),
            "z": rng.uniform(-1.0, 1.0, n),
            "g": [f"g{i % 3}" for i in range(n)],
        }
    )
    frame["y"] = np.sin(2 * np.pi * frame["x"]) + 0.5 * frame["z"] + 0.1 * rng.standard_normal(n)
    model = gamfit.fit(frame, "y ~ s(x) + s(z) + g")

    result = model.partial_dependence("s(x)", n_points=17)
    assert result.axes == ("x",)
    assert result.held == {}
    assert result.scale == "linear_predictor"
    assert result.quantity == "term_contribution"
    grid = result.x

    one = pd.DataFrame({"x": grid, "z": np.full(grid.size, -0.9), "g": ["g0"] * grid.size})
    two = pd.DataFrame(
        {"x": grid, "z": np.linspace(0.3, 0.95, grid.size), "g": ["g2"] * grid.size}
    )
    block, columns_one = _block_columns(model, "s(x)", one)
    _, columns_two = _block_columns(model, "s(x)", two)
    np.testing.assert_array_equal(columns_one, columns_two)

    beta, cov = _beta_and_cov(model)
    sub = cov[block.start : block.end, block.start : block.end]
    np.testing.assert_allclose(
        result.fit, columns_one @ beta[block.start : block.end], atol=1e-12
    )
    np.testing.assert_allclose(
        result.se,
        np.sqrt(np.einsum("ij,jk,ik->i", columns_one, sub, columns_one)),
        atol=1e-12,
    )


def test_a_linear_term_sweeps_its_own_column():
    rng = np.random.default_rng(11)
    n = 200
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 2.0, n), "z": rng.uniform(0.0, 1.0, n)})
    frame["y"] = 1.5 * frame["x"] + np.sin(2 * np.pi * frame["z"]) + 0.1 * rng.standard_normal(n)
    model = gamfit.fit(frame, "y ~ x + s(z)")

    result = model.partial_dependence("x", n_points=5)
    assert result.axes == ("x",)
    grid = result.x
    block, columns = _block_columns(
        model, "x", pd.DataFrame({"x": grid, "z": np.full(grid.size, 0.5)})
    )
    beta, _ = _beta_and_cov(model)
    np.testing.assert_allclose(
        result.fit, columns @ beta[block.start : block.end], atol=1e-12
    )


def _numeric_by_frame(reverse):
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.5, 2.0, n)
    y = z * np.sin(2 * np.pi * x) + 0.05 * rng.standard_normal(n)
    frame = pd.DataFrame({"x": x, "z": z, "y": y})
    return frame.iloc[::-1].reset_index(drop=True) if reverse else frame


def test_a_numeric_by_smooth_reports_its_coefficient_function():
    term = "s(x, by=z, k=8)"
    model = gamfit.fit(_numeric_by_frame(reverse=False), f"y ~ {term}")
    result = model.partial_dependence(term, n_points=21)
    assert result.quantity == "coefficient_function"
    assert result.contribution == "z * f(x)"
    assert result.held == {"z": 1.0}
    grid = result.x

    block, columns = _block_columns(
        model, term, pd.DataFrame({"x": grid, "z": np.ones(grid.size)})
    )
    beta, _ = _beta_and_cov(model)
    np.testing.assert_allclose(
        result.fit, columns @ beta[block.start : block.end], atol=1e-12
    )
    assert np.max(np.abs(result.fit)) > 0.5

    reordered = gamfit.fit(_numeric_by_frame(reverse=True), f"y ~ {term}")
    again = reordered.partial_dependence(term, n_points=21)
    np.testing.assert_array_equal(again.x, grid)
    np.testing.assert_allclose(again.fit, result.fit, rtol=0.0, atol=1e-8)
