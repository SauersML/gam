"""End-to-end coverage of the formula DSL surface through ``gamfit.fit``.

Pins the user-visible contract of:

* intercept removal (``0 + ...``, ``... + 0``, ``... - 1``): an unpenalized
  slope without the intercept is ordinary least squares through the origin,
  and ``0 + g`` is the cell-means model;
* backtick-quoted, non-identifier column names, and patsy's ``C()`` refused
  with an error that names ``factor()``;
* ``domain=[a, b]`` on ``s()`` (validated against the data, linear
  extrapolation past it at predict time);
* strict option parsing: a malformed value, an unknown option, or
  ``penalty_order`` above the spline degree raises ``gamfit.errors.FormulaError``
  naming the term and the option;
* a scalar ``bs=`` on ``te()`` applying to every margin.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

N = 40


def _linear_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(1, N + 1) * 2.0 / N
    wobble = 0.1 * (((np.arange(N) * 7) % 11) - 5) / 5
    return x, 1.5 + 0.8 * x + wobble


def _predict(model, data) -> np.ndarray:
    return np.asarray(model.predict(data), dtype=float)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ 0 + linear(x, double_penalty=false)",
        "y ~ linear(x, double_penalty=false) + 0",
        "y ~ linear(x, double_penalty=false) - 1",
    ],
)
def test_unpenalized_slope_without_intercept_is_origin_least_squares(formula):
    x, y = _linear_data()
    slope = float(x @ y / (x @ x))
    model = gamfit.fit({"y": y, "x": x}, formula, family="gaussian")
    fitted = _predict(model, {"x": np.append(x, 0.0)})
    np.testing.assert_allclose(fitted[:-1], slope * x, rtol=1e-9, atol=1e-12)
    assert fitted[-1] == 0.0


def test_default_slope_without_intercept_passes_through_the_origin():
    # The implicit linear term keeps its REML shrinkage ridge (as it does in
    # `y ~ x`), so the slope may sit toward zero from OLS, but the model has
    # no constant: every prediction is a single slope times x.
    x, y = _linear_data()
    slope = float(x @ y / (x @ x))
    model = gamfit.fit({"y": y, "x": x}, "y ~ 0 + x", family="gaussian")
    grid = np.array([0.0, 0.5, 1.0, 3.0])
    fitted = _predict(model, {"x": grid})
    assert fitted[0] == 0.0
    ratio = fitted[1:] / grid[1:]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
    assert 0.0 < ratio[0] <= slope * (1.0 + 1e-12)

    with_intercept = gamfit.fit({"y": y, "x": x}, "y ~ x", family="gaussian")
    assert abs(_predict(with_intercept, {"x": np.array([0.0])})[0]) > 1.0


def test_zero_plus_factor_is_the_cell_means_model():
    levels = np.array(["a", "b", "c"])
    g = np.resize(levels, N)
    means = {"a": 1.0, "b": 2.0, "c": 4.0}
    y = np.array([means[v] for v in g]) + 0.1 * np.sin(np.arange(N))
    model = gamfit.fit({"y": y, "g": g}, "y ~ 0 + g", family="gaussian")
    fitted = _predict(model, {"g": levels})
    expected = [y[g == level].mean() for level in levels]
    np.testing.assert_allclose(fitted, expected, rtol=1e-9)


def test_no_intercept_model_round_trips_through_save_and_load(tmp_path):
    x, y = _linear_data()
    model = gamfit.fit({"y": y, "x": x}, "y ~ 0 + s(x)", family="gaussian")
    grid = {"x": np.linspace(0.1, 2.0, 9)}
    path = tmp_path / "no_intercept.gam"
    model.save(path)
    reloaded = gamfit.load(path)
    np.testing.assert_allclose(_predict(reloaded, grid), _predict(model, grid), rtol=1e-12)


def test_backtick_columns_match_plain_names():
    x, y = _linear_data()
    site = np.resize(np.array(["north", "south", "east"]), N)
    y = y + np.where(site == "south", 0.5, 0.0)
    quoted = gamfit.fit(
        {"y": y, "dose (mg)": x, "site-id": site},
        "y ~ `dose (mg)` + factor(`site-id`)",
        family="gaussian",
    )
    plain = gamfit.fit(
        {"y": y, "dose": x, "site": site},
        "y ~ dose + factor(site)",
        family="gaussian",
    )
    np.testing.assert_allclose(
        _predict(quoted, {"dose (mg)": x, "site-id": site}),
        _predict(plain, {"dose": x, "site": site}),
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    ("formula", "column"), [("y ~ x + C(x2)", "x2"), ("y ~ x + C(`site-id`)", "`site-id`")]
)
def test_patsy_c_is_refused_with_a_pointer_to_factor(formula, column):
    x, y = _linear_data()
    site = np.resize(np.array(["north", "south", "east"]), N)
    with pytest.raises(gamfit.errors.FormulaError) as excinfo:
        gamfit.fit({"y": y, "x": x, "x2": site, "site-id": site}, formula, family="gaussian")
    message = str(excinfo.value)
    assert f"factor({column})" in message and f"group({column})" in message, message


def test_domain_must_contain_the_data():
    x, y = _linear_data()
    with pytest.raises(gamfit.errors.FormulaError, match=r"in term s\(x, domain=\[0.5, 1.0\]\).*does not contain the data"):
        gamfit.fit({"y": y, "x": x}, "y ~ s(x, domain=[0.5, 1.0])", family="gaussian")


def test_domain_extends_the_basis_and_extrapolates_linearly_past_it():
    x, y = _linear_data()
    y = y + np.sin(3.0 * x)
    model = gamfit.fit({"y": y, "x": x}, "y ~ s(x, domain=[0, 3])", family="gaussian")
    inside = _predict(model, {"x": np.array([2.5, 2.9])})
    assert np.all(np.isfinite(inside))
    beyond = _predict(model, {"x": np.array([3.5, 4.0, 4.5])})
    # Past the declared domain the spline continues as a straight line.
    second_difference = beyond[2] - 2.0 * beyond[1] + beyond[0]
    assert abs(second_difference) <= 1e-8 * (1.0 + np.max(np.abs(beyond)))


@pytest.mark.parametrize(
    ("formula", "needles"),
    [
        ("y ~ s(x, k=ten)", ["in term s(x, k=ten)", "`k=ten`"]),
        (
            "y ~ s(x, degree=2, penalty_order=3)",
            ["in term s(x, degree=2, penalty_order=3)", "penalty_order=3 exceeds the spline degree 2"],
        ),
        ("y ~ s(x, bogus=1)", ["in term s(x, bogus=1)", "option `bogus`"]),
        ("y ~ s(x, double_penalty=maybe)", ["in term s(x, double_penalty=maybe)", "double_penalty"]),
    ],
)
def test_malformed_options_raise_formula_error_naming_term_and_option(formula, needles):
    x, y = _linear_data()
    with pytest.raises(gamfit.errors.FormulaError) as raised:
        gamfit.fit({"y": y, "x": x}, formula, family="gaussian")
    message = str(raised.value)
    for needle in needles:
        assert needle in message, message


def test_scalar_bs_on_te_applies_to_every_margin():
    x, y = _linear_data()
    z = x[::-1] ** 2
    model = gamfit.fit({"y": y, "x": x, "z": z}, "y ~ te(x, z, bs=cr)", family="gaussian")
    assert np.all(np.isfinite(_predict(model, {"x": x, "z": z})))
    with pytest.raises(gamfit.errors.FormulaError, match="not a supported penalized-spline margin"):
        gamfit.fit({"y": y, "x": x, "z": z}, "y ~ te(x, z, bs=re)", family="gaussian")
