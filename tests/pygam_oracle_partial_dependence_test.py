"""pyGAM oracle: partial dependence (pygam/tests/test_partial_dependence.py).

pyGAM asserts that the term partial dependences plus the intercept add up to
the linear predictor, that the default grid is a linspace over the training
range, that a 2-D tensor term accepts a meshgrid, that an unknown term raises,
and that ``width=`` intervals come from the model covariance.

gamfit's ``partial_dependence`` is the term contribution ``X_t(x) beta_t`` on the
linear-predictor scale, so the additive identity

    eta(x_i) == intercept + sum_t pdep_t(x_i)

must hold to rounding for every training row. Nothing else in the suite ties
``partial_dependence`` to ``predict``: shape and finiteness checks pass for a
curve that is shifted by a constant or priced off the wrong block.

All data are seeded synthetic stand-ins for pyGAM's datasets; pyGAM is never
imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gamfit


def _eta(model: Any, data: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(model.predict(data, interval=0.95)["linear_predictor_plugin"], float)


def _intercept(model: Any, data: dict[str, np.ndarray]) -> float:
    design = model.design_matrix(data)
    (block,) = [b for b in model.term_blocks if b.kind == "intercept"]
    assert block.end - block.start == 1
    return float(design.coefficients[block.start])


def _pdep_at_rows(model: Any, term: str, *columns: np.ndarray) -> np.ndarray:
    grid = columns[0] if len(columns) == 1 else np.column_stack(columns)
    return np.asarray(model.partial_dependence(term, grid=grid)["predicted"], float)


def _scale_tol(eta: np.ndarray) -> float:
    # Rounding of a length-p dot product, relative to the predictor's size.
    return 1e-10 * max(1.0, float(np.max(np.abs(eta))))


@pytest.fixture(scope="module")
def wage_like() -> dict[str, np.ndarray]:
    """pyGAM's wage shape: s(year) + s(age) + a 5-level education factor."""
    rng = np.random.default_rng(20)
    n = 600
    year = rng.integers(2003, 2010, n).astype(float)
    age = rng.uniform(18.0, 80.0, n)
    levels = np.array(["hs_drop", "hs", "some_college", "college", "advanced"])
    edu = levels[rng.integers(0, levels.size, n)]
    edu_effect = dict(zip(levels, [-12.0, -4.0, 0.0, 9.0, 20.0]))
    y = (
        100.0
        + 1.5 * (year - 2006.0)
        + 30.0 * np.sin((age - 18.0) / 62.0 * np.pi)
        + np.array([edu_effect[e] for e in edu])
        + rng.normal(0.0, 8.0, n)
    )
    return {"year": year, "age": age, "edu": edu, "y": y}


@pytest.fixture(scope="module")
def counts_like() -> dict[str, np.ndarray]:
    """pyGAM's chicago shape: Poisson counts over a trend and a temperature curve."""
    rng = np.random.default_rng(21)
    n = 800
    time = np.sort(rng.uniform(0.0, 100.0, n))
    tmpd = 50.0 + 25.0 * np.sin(2.0 * np.pi * time / 25.0) + rng.normal(0.0, 5.0, n)
    eta = 3.0 + 0.3 * np.sin(time / 15.0) + 0.004 * (tmpd - 50.0) ** 2 / 10.0
    y = rng.poisson(np.exp(eta)).astype(float)
    return {"time": time, "tmpd": tmpd, "y": y}


@pytest.fixture(scope="module")
def smooth_1d() -> tuple[dict[str, np.ndarray], Any]:
    """pyGAM's mcycle shape: one strongly nonlinear Gaussian smooth."""
    rng = np.random.default_rng(22)
    n = 150
    x = rng.uniform(2.0, 58.0, n)
    y = 5.0 + 40.0 * np.sin(x / 7.0) * np.exp(-x / 30.0) + rng.normal(0.0, 5.0, n)
    data = {"x": x, "y": y}
    return data, gamfit.fit(data, "y ~ s(x)")


# test_partial_dependence_on_univar_data: pdep + intercept == prediction.
def test_univariate_pdep_plus_intercept_equals_linear_predictor(
    smooth_1d: tuple[dict[str, np.ndarray], Any],
) -> None:
    data, m = smooth_1d
    eta = _eta(m, data)
    intercept = _intercept(m, data)
    # The intercept is far from zero, so a pdep that silently absorbed it (or
    # an identity that forgot it) cannot pass.
    assert abs(intercept) > 1.0
    total = intercept + _pdep_at_rows(m, "s(x)", data["x"])
    np.testing.assert_allclose(total, eta, rtol=0.0, atol=_scale_tol(eta))


def test_multi_smooth_poisson_additive_identity(counts_like: dict[str, np.ndarray]) -> None:
    d = counts_like
    m = gamfit.fit(d, "y ~ s(time) + s(tmpd)", family="poisson")
    eta = _eta(m, d)
    intercept = _intercept(m, d)
    assert abs(intercept) > 1.0
    total = (
        intercept
        + _pdep_at_rows(m, "s(time)", d["time"])
        + _pdep_at_rows(m, "s(tmpd)", d["tmpd"])
    )
    np.testing.assert_allclose(total, eta, rtol=0.0, atol=_scale_tol(eta))
    # Each smooth carries real signal, so dropping either one breaks the sum.
    for term, col in (("s(time)", "time"), ("s(tmpd)", "tmpd")):
        assert np.ptp(_pdep_at_rows(m, term, d[col])) > 100.0 * _scale_tol(eta)


def test_additive_identity_with_a_factor(wage_like: dict[str, np.ndarray]) -> None:
    """With a factor in the model, eta minus the intercept and the smooth pdeps
    must be a pure function of the level, and must equal the factor block's own
    design contribution."""
    d = wage_like
    m = gamfit.fit(d, "y ~ s(year) + s(age) + factor(edu)")
    eta = _eta(m, d)
    tol = _scale_tol(eta)
    rest = (
        eta
        - _intercept(m, d)
        - _pdep_at_rows(m, "s(year)", d["year"])
        - _pdep_at_rows(m, "s(age)", d["age"])
    )
    per_level = {}
    for level in np.unique(d["edu"]):
        r = rest[d["edu"] == level]
        assert np.ptp(r) <= tol, (level, np.ptp(r))
        per_level[level] = float(r[0])
    # The factor carries a 32-unit spread in truth; the remainder must see it.
    assert max(per_level.values()) - min(per_level.values()) > 10.0

    design = m.design_matrix(d)
    (block,) = [b for b in m.term_blocks if "edu" in b.name]
    factor_part = (
        np.asarray(design.matrix)[:, block.start : block.end]
        @ np.asarray(design.coefficients)[block.start : block.end]
    )
    np.testing.assert_allclose(rest, factor_part, rtol=0.0, atol=tol)


# test_partial_dependence_gives_correct_shape_no_meshgrid
def test_default_grid_is_linspace_over_training_range(
    smooth_1d: tuple[dict[str, np.ndarray], Any],
) -> None:
    data, m = smooth_1d
    default = m.partial_dependence("s(x)")
    expected_grid = np.linspace(data["x"].min(), data["x"].max(), 100)
    np.testing.assert_allclose(default["grid"], expected_grid, rtol=1e-12)
    explicit = m.partial_dependence("s(x)", grid=expected_grid)
    np.testing.assert_array_equal(default["predicted"], explicit["predicted"])
    np.testing.assert_array_equal(default["standard_error"], explicit["standard_error"])
    assert default["predicted"].shape == (100,)
    assert np.all(default["standard_error"] > 0.0)
    assert m.partial_dependence("s(x)", n_points=37)["predicted"].shape == (37,)


# test_partial_dependence_gives_correct_shape_with_meshgrid
def test_tensor_pdep_on_meshgrid_matches_rowwise_evaluation() -> None:
    rng = np.random.default_rng(23)
    n = 500
    a = rng.uniform(0.0, 1.0, n)
    b = rng.uniform(0.0, 1.0, n)
    y = 1.0 + np.sin(3.0 * a) * np.cos(3.0 * b) + rng.normal(0.0, 0.1, n)
    d = {"a": a, "b": b, "y": y}
    m = gamfit.fit(d, "y ~ te(a, b)")
    ga, gb = np.meshgrid(np.linspace(0.0, 1.0, 7), np.linspace(0.0, 1.0, 5))
    grid = np.column_stack([ga.ravel(), gb.ravel()])
    out = m.partial_dependence("te(a, b)", grid=grid)
    assert out["predicted"].shape == (35,)
    assert np.all(np.isfinite(out["standard_error"]))
    assert list(out["axes"]) == ["a", "b"]
    # The meshgrid curve is the same function the rows see: at the training
    # rows it closes the additive identity.
    eta = _eta(m, d)
    total = _intercept(m, d) + _pdep_at_rows(m, "te(a, b)", a, b)
    np.testing.assert_allclose(total, eta, rtol=0.0, atol=_scale_tol(eta))


# test_partial_dependence_raises_error_with_bad_term
def test_unknown_term_raises(smooth_1d: tuple[dict[str, np.ndarray], Any]) -> None:
    _, m = smooth_1d
    with pytest.raises(ValueError):
        m.partial_dependence("s(nope)")


# test_partial_dependence_width: intervals come from the model covariance.
def test_pdep_se_is_priced_off_the_published_covariance(
    smooth_1d: tuple[dict[str, np.ndarray], Any],
) -> None:
    data, m = smooth_1d
    grid = np.linspace(data["x"].min(), data["x"].max(), 25)
    pdep = m.partial_dependence("s(x)", grid=grid)
    pred = m.predict({"x": grid}, interval=0.95)
    assert pdep["covariance_source"] == pred["covariance_source"]

    # Delta-method SE sqrt(diag(X_t V_tt X_t')) from the same published matrix.
    design = m.design_matrix({"x": grid, "y": np.zeros_like(grid)})
    source = pdep["covariance_source"]
    cov = (
        design.covariance_smoothing_corrected
        if source == "smoothing-corrected"
        else design.covariance_conditional
    )
    assert cov is not None
    (block,) = [b for b in m.term_blocks if b.name == "s(x)"]
    xt = np.asarray(design.matrix)[:, block.start : block.end]
    vt = np.asarray(cov)[block.start : block.end, block.start : block.end]
    se = np.sqrt(np.einsum("ij,jk,ik->i", xt, vt, xt))
    np.testing.assert_allclose(pdep["standard_error"], se, rtol=1e-8)
