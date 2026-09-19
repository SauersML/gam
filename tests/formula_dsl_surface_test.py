"""End-to-end coverage of the formula DSL surface through ``gamfit.fit``.

Pins the user-visible contract of:

* intercept removal (``0 + ...``, ``... + 0``, ``... - 1``): an unpenalized
  slope without the intercept is ordinary least squares through the origin,
  and a term that spans the constant (``0 + g``, ``0 + g:h``, ``0 + s(x)``)
  keeps the intercept, so only the constant is unpenalized;
* backtick-quoted, non-identifier column names and ``C()`` as a ``factor()``
  alias;
* ``domain=[a, b]`` on ``s()`` (validated against the data, linear
  extrapolation past it at predict time);
* strict option parsing: a malformed value, an unknown option, or
  ``penalty_order`` above the spline degree raises ``gamfit.errors.FormulaError``
  naming the term and the option;
* a scalar ``bs=`` on ``te()`` applying to every margin;
* a formula that does not parse (unbalanced parentheses, no ``~``) raising
  ``gamfit.errors.FormulaError``, not a configuration error.
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


_CONSTANT_SPANNING = [
    ("y ~ 0 + g", "y ~ g"),
    ("y ~ g - 1", "y ~ g"),
    ("y ~ 0 + g:h", "y ~ g:h"),
    ("y ~ 0 + s(x)", "y ~ s(x)"),
]


def _factor_noise_data(seed: int, n: int = 240) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "y": 3.0 + rng.standard_normal(n),
        "g": np.resize(np.array(["a", "b", "c"]), n),
        "h": np.resize(np.array(["u", "v"]), n + n // 3)[n // 3 :],
        "x": rng.uniform(0.0, 1.0, n),
    }


@pytest.mark.parametrize(("formula", "with_intercept"), _CONSTANT_SPANNING)
def test_a_term_spanning_the_constant_keeps_the_intercept(formula, with_intercept):
    # The constant is the one unpenalized direction; the term keeps every
    # other penalty, so the fit is the intercept formula's.
    data = _factor_noise_data(0)
    data["y"] = data["y"] + np.where(data["g"] == "b", 1.0, 0.0) + np.sin(4.0 * data["x"])
    dropped = gamfit.fit(data, formula, family="gaussian")
    kept = gamfit.fit(data, with_intercept, family="gaussian")
    np.testing.assert_allclose(_predict(dropped, data), _predict(kept, data), rtol=1e-10)


@pytest.mark.parametrize("formula", ["y ~ 0 + g", "y ~ 0 + g:h", "y ~ 0 + s(x)"])
def test_pure_noise_recovers_only_the_constant(formula):
    # Under a constant response every non-constant direction has a penalty
    # REML can drive to the null. Freeing the level by stripping the term's
    # penalties left unpenalized non-constant directions (the level contrasts
    # of `g`, every cell of `g:h`, the slope of `s(x)`), so its excess edf over
    # the constant was at least one in every replicate. A penalized fit's
    # excess is random, with an atom at zero where REML puts every smoothing
    # parameter on its rail: the null must be reachable.
    excess = []
    for seed in range(8):
        data = _factor_noise_data(seed)
        model = gamfit.fit(data, formula, family="gaussian")
        excess.append(float(model.summary().edf_total) - 1.0)
        fitted = _predict(model, data)
        assert abs(float(np.mean(fitted)) - float(np.mean(data["y"]))) < 1e-8
    assert min(excess) < 1e-6, excess


def test_no_intercept_model_round_trips_through_save_and_load(tmp_path):
    x, y = _linear_data()
    model = gamfit.fit({"y": y, "x": x}, "y ~ 0 + s(x)", family="gaussian")
    grid = {"x": np.linspace(0.1, 2.0, 9)}
    path = tmp_path / "no_intercept.gam"
    model.save(path)
    reloaded = gamfit.load(path)
    np.testing.assert_allclose(_predict(reloaded, grid), _predict(model, grid), rtol=1e-12)


def test_backtick_columns_and_c_alias_match_plain_names():
    x, y = _linear_data()
    site = np.resize(np.array(["north", "south", "east"]), N)
    y = y + np.where(site == "south", 0.5, 0.0)
    quoted = gamfit.fit(
        {"y": y, "dose (mg)": x, "site-id": site},
        "y ~ `dose (mg)` + C(`site-id`)",
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


@pytest.mark.parametrize(
    "formula",
    ["y ~ s(x, k=10", "y ~ s(x))", "y s(x)"],
)
def test_formula_syntax_error_raises_formula_error(formula):
    x, y = _linear_data()
    with pytest.raises(gamfit.errors.FormulaError, match="invalid formula syntax"):
        gamfit.fit({"y": y, "x": x}, formula, family="gaussian")
