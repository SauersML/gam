"""The fixed categorical factor contract (pyGAM audit F1, F2, F3).

``factor(g)`` and a bare categorical ``+ g`` are FIXED effects in the R
``factor()`` / patsy ``C()`` sense: treatment-coded, so an ``L``-level factor
adds ``L-1`` columns against a reference level that the intercept carries,
with no penalty and therefore no smoothing parameter. ``group(g)`` and
``re(g)`` are the penalized random effects.

Before this contract was pinned, every categorical spelling lowered to the same
REML-penalized full one-hot ridge, so a "fixed" factor was shrunk toward the
intercept. On a rare-event binomial the shrinkage is severe: the audit's
credit-default fit reported a student effect of about ``-2.7e-5`` where the
unpenalized fit gives about ``-0.41``. These tests check the fitted factor
against closed forms that hold only for an unpenalized fixed effect:

* a Gaussian ``y ~ factor(g)`` reproduces each group's sample mean exactly;
* a binomial ``y ~ factor(g)`` reproduces each group's empirical log-odds, so
  the contrast is the empirical log odds ratio;
* with a smooth alongside, the factor contrast equals the coefficient of the
  same 0/1 indicator entered as an unpenalized ``linear(..., double_penalty=false)``.

They also pin the two input contracts that keep categorical columns out of the
wrong terms: a categorical column in a numeric-axis term (``s()``, ``linear()``,
``te()``, ...) is refused with a pointer to ``factor()``/``group()`` (F2), and
the categorical wrappers take no options, so a typo such as
``factor(g, foo=1)`` is an error instead of being silently ignored (F3).
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

_LEVELS = np.array(["alpha", "beta", "gamma", "delta"])
_GROUP_MEANS = np.array([1.0, 2.5, -0.7, 0.3])


def _gaussian_frame(seed: int) -> tuple[Any, Any]:
    rng = np.random.default_rng(seed)
    gi = rng.integers(0, _LEVELS.size, 400)
    y = _GROUP_MEANS[gi] + rng.normal(0.0, 1.0, gi.size)
    return pd.DataFrame({"g": _LEVELS[gi], "y": y}), gi


def _g_block(model: Any) -> Any:
    blocks = [block for block in model.term_blocks if block.name == "g"]
    assert len(blocks) == 1, f"expected exactly one `g` block, got {model.term_blocks}"
    return blocks[0]


def _linear_predictor(model: Any, rows: Any) -> Any:
    design = model.design_matrix(rows)
    return np.asarray(design.offset + design.matrix @ design.coefficients, dtype=float)


@pytest.mark.parametrize("formula", ["y ~ factor(g)", "y ~ g"])
def test_gaussian_fixed_factor_reproduces_group_means_with_l_minus_one_columns(
    formula: str,
) -> None:
    data, gi = _gaussian_frame(seed=0)
    model = gamfit.fit(data, formula)

    block = _g_block(model)
    assert block.kind == "factor", f"{formula}: `g` must be a fixed factor, got {block}"
    assert block.end - block.start == _LEVELS.size - 1, (
        f"{formula}: an {_LEVELS.size}-level fixed factor must be treatment-coded "
        f"with {_LEVELS.size - 1} columns, got {block}"
    )
    assert model.smoothing_parameters() == {}, (
        f"{formula}: a fixed factor is unpenalized and must carry no smoothing "
        f"parameter, got {model.smoothing_parameters()}"
    )

    fitted = np.asarray(model.predict(pd.DataFrame({"g": _LEVELS})), dtype=float).reshape(-1)
    sample_means = np.array([data["y"].to_numpy()[gi == k].mean() for k in range(_LEVELS.size)])
    np.testing.assert_allclose(
        fitted,
        sample_means,
        rtol=0.0,
        atol=1e-8,
        err_msg=f"{formula}: an unpenalized factor fit must reproduce the group means",
    )


def test_group_and_re_remain_penalized_random_effects() -> None:
    data, _ = _gaussian_frame(seed=1)
    for formula in ["y ~ group(g)", "y ~ re(g)"]:
        model = gamfit.fit(data, formula)
        block = _g_block(model)
        assert block.kind == "random_effect", f"{formula}: got {block}"
        assert block.end - block.start == _LEVELS.size, (
            f"{formula}: a random effect keeps one column per level, got {block}"
        )
        assert len(model.smoothing_parameters()) == 1, (
            f"{formula}: a random effect has one REML variance parameter, got "
            f"{model.smoothing_parameters()}"
        )


def _rare_event_frame(seed: int) -> Any:
    # A credit-default-shaped problem: about 3% events and a binary student
    # indicator whose log odds ratio sits several standard errors from 0. The
    # balance covariate drives the outcome but is left out of the saturated
    # fits, which therefore target the marginal cell log-odds.
    rng = np.random.default_rng(seed)
    n = 8000
    student = rng.integers(0, 2, n)
    balance = rng.uniform(-1.0, 1.0, n)
    eta = -3.6 - 0.8 * student + 1.2 * balance
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return pd.DataFrame({"student": np.array(["No", "Yes"])[student], "y": y})


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


@pytest.mark.parametrize("formula", ["y ~ factor(student)", "y ~ student"])
def test_rare_event_binomial_factor_is_the_unshrunk_empirical_log_odds_ratio(
    formula: str,
) -> None:
    data = _rare_event_frame(seed=0)
    y = data["y"].to_numpy()
    yes = data["student"].to_numpy() == "Yes"
    assert 0.0 < y.mean() < 0.06, "the fixture must be a rare-event problem"

    model = gamfit.fit(data, formula, family="binomial")
    assert model.smoothing_parameters() == {}, model.smoothing_parameters()
    eta = _linear_predictor(model, pd.DataFrame({"student": ["No", "Yes"]}))

    # A saturated unpenalized logistic fit reproduces each cell's empirical
    # log-odds, so the treatment contrast is the empirical log odds ratio.
    expected_no = _logit(y[~yes].mean())
    expected_yes = _logit(y[yes].mean())
    np.testing.assert_allclose(eta, [expected_no, expected_yes], rtol=0.0, atol=1e-6)
    contrast = eta[1] - eta[0]
    assert contrast < -0.2, (
        f"{formula}: the student contrast {contrast:.3e} must not be shrunk toward 0"
    )


def _indicator_beside_smooth_frame(seed: int) -> Any:
    # The saturated rare-event case is pinned above; here the point is the
    # factor's equivalence to an unpenalized 0/1 column when a smooth shares
    # the fit, so the smooth carries a genuine curve.
    rng = np.random.default_rng(seed)
    n = 8000
    student = rng.integers(0, 2, n)
    balance = rng.uniform(-1.0, 1.0, n)
    eta = -0.5 - 0.8 * student + 1.5 * np.sin(2.5 * balance)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return pd.DataFrame(
        {
            "student": np.array(["No", "Yes"])[student],
            "student01": student.astype(float),
            "balance": balance,
            "y": y,
        }
    )


def test_fixed_factor_matches_an_unpenalized_indicator_beside_a_smooth() -> None:
    data = _indicator_beside_smooth_frame(seed=1)
    factor_model = gamfit.fit(data, "y ~ factor(student) + s(balance)", family="binomial")
    indicator_model = gamfit.fit(
        data,
        "y ~ linear(student01, double_penalty=false) + s(balance)",
        family="binomial",
    )
    assert len(factor_model.smoothing_parameters()) == len(
        indicator_model.smoothing_parameters()
    ), "the factor must add no smoothing parameter beyond the smooth's"

    rows = pd.DataFrame(
        {"student": ["No", "Yes"], "student01": [0.0, 1.0], "balance": [0.0, 0.0]}
    )
    factor_contrast = float(np.diff(_linear_predictor(factor_model, rows))[0])
    indicator_contrast = float(np.diff(_linear_predictor(indicator_model, rows))[0])
    assert factor_contrast < -0.2, f"factor contrast {factor_contrast:.3e} was shrunk"
    np.testing.assert_allclose(factor_contrast, indicator_contrast, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ s(g)",
        "y ~ linear(g)",
        "y ~ te(x, g)",
        'y ~ s(g, bs="cc")',
        "y ~ thinplate(x, g)",
        "y ~ matern(g)",
    ],
)
def test_categorical_column_in_a_numeric_axis_term_is_refused(formula: str) -> None:
    data, _ = _gaussian_frame(seed=2)
    data["x"] = np.linspace(0.0, 1.0, len(data))
    for column in (data["g"], data["g"].astype("category")):
        frame = data.assign(g=column)
        with pytest.raises(gamfit.GamError) as excinfo:
            gamfit.fit(frame, formula)
        message = str(excinfo.value)
        assert "'g' is categorical" in message, message
        assert "factor(g)" in message and "group(g)" in message, message


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ factor(g, foo=1)",
        "y ~ factor(g, double_penalty=false)",
        "y ~ group(g, bogus=3)",
        "y ~ re(g, k=4)",
    ],
)
def test_categorical_wrappers_reject_unknown_options(formula: str) -> None:
    data, _ = _gaussian_frame(seed=3)
    with pytest.raises(gamfit.FormulaError, match="does not accept option"):
        gamfit.fit(data, formula)


def test_fixed_factor_rejects_an_unseen_level_and_accepts_the_reference() -> None:
    data, gi = _gaussian_frame(seed=4)
    model = gamfit.fit(data, "y ~ factor(g)")
    reference = sorted(_LEVELS)[0]
    fitted = np.asarray(model.predict(pd.DataFrame({"g": [reference]})), dtype=float)
    expected = data["y"].to_numpy()[data["g"].to_numpy() == reference].mean()
    np.testing.assert_allclose(fitted.reshape(-1), [expected], rtol=0.0, atol=1e-8)
    with pytest.raises(gamfit.GamError):
        model.predict(pd.DataFrame({"g": ["never-seen"]}))
