"""Data validation at the Rust fit boundary (pyGAM audit F3 + F7).

Every rule below lives in Rust (``gam-data`` ``validate_fit_boundary``, the
family-owned ``ResponseFamily`` support/degeneracy rules in ``gam-spec``, and
the prior-weight check in the formula materializer), so the CLI and Python
share one definition. Python only forwards the table; these tests pin what the
caller sees.

Policies pinned here:

* **Scale invariance (F3).** A Gaussian fit is equivariant under ``y -> c*y``:
  the fitted mean scales by ``c`` and the smoothing parameters do not move.
  There is no absolute spread floor on the response.
* **Family support.** Negative binomial takes non-negative *integer* counts
  (non-integer data is pointed at Poisson and Tweedie); an explicit Poisson
  family takes any non-negative real (gam#4572); binomial takes
  ``y`` in ``[0, 1]``, Gamma takes ``y > 0``. Errors are ``DataError`` naming
  the column, the family and the first offending 1-based row.
* **Weights.** Finite, non-negative and not all zero. A zero weight excludes
  its row from the likelihood, so the support and degeneracy rules judge only
  positive-weight rows.
* **Missing values are rejected**, never dropped (the scikit-learn policy).
"""
from __future__ import annotations

import numpy as np
import pytest

import gamfit

N = 300


def _base(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, N)
    y = np.sin(6.0 * x) + rng.normal(0.0, 0.3, N)
    return x, y


def _mean(model: gamfit.Model, x: np.ndarray) -> np.ndarray:
    out = model.predict({"x": x})
    mean = out if isinstance(out, np.ndarray) else out["posterior_mean"]
    return np.asarray(mean, dtype=float).ravel()


# ---------------------------------------------------------------------------
# F3: scale invariance of the Gaussian fit
# ---------------------------------------------------------------------------

GRID = np.linspace(0.05, 0.95, 25)


def _lambdas(model: gamfit.Model) -> np.ndarray:
    params = model.smoothing_parameters()
    return np.array([params[key] for key in sorted(params)], dtype=float)


@pytest.fixture(scope="module")
def unit_scale_fit() -> tuple[np.ndarray, np.ndarray]:
    x, y = _base()
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)")
    return _mean(model, GRID), _lambdas(model)


@pytest.mark.parametrize("k", [-12, -6, 0, 6, 12])
def test_gaussian_fit_scales_with_the_response(k: int, unit_scale_fit) -> None:
    unit_mean, unit_lambda = unit_scale_fit
    x, y = _base()
    c = 10.0**k
    model = gamfit.fit({"x": x, "y": y * c}, "y ~ s(x)")
    # Fitted mean is c times the unit-scale fitted mean ...
    np.testing.assert_allclose(_mean(model, GRID) / c, unit_mean, rtol=1e-6, atol=0.0)
    # ... and the smoothing parameters are scale-free.
    np.testing.assert_allclose(_lambdas(model), unit_lambda, rtol=1e-6)


def test_tiny_scale_gaussian_response_is_fit_not_refused() -> None:
    # The #332 reproducer: pure noise at scale 1e-13. It was once refused by an
    # absolute ``sd <= 1e-10`` floor; the fit is the unit-scale fit times 1e-13.
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 200)
    noise = rng.normal(size=200)
    unit = gamfit.fit({"x": x, "y": noise}, "y ~ s(x)", family="gaussian")
    tiny = gamfit.fit({"x": x, "y": noise * 1e-13}, "y ~ s(x)", family="gaussian")
    np.testing.assert_allclose(_mean(tiny, GRID) / 1e-13, _mean(unit, GRID), rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Family support
# ---------------------------------------------------------------------------


def _counts(seed: int = 12) -> np.ndarray:
    return np.random.default_rng(seed).poisson(3.0, N).astype(float)


def _binary(seed: int = 13) -> np.ndarray:
    return (np.random.default_rng(seed).uniform(size=N) < 0.5).astype(float)


def _positive(seed: int = 14) -> np.ndarray:
    return np.random.default_rng(seed).gamma(2.0, 1.0, N)


def _with(values: np.ndarray, row0: int, value: float) -> np.ndarray:
    out = values.copy()
    out[row0] = value
    return out


SUPPORT_CASES = [
    # (id, family, response, 1-based offending row, value text, family label)
    ("poisson-negative", "poisson", _with(_counts(), 6, -1.0), 7, "-1", "Poisson"),
    ("nb-non-integer", "negative-binomial", _with(_counts(), 9, 0.5), 10, "0.5", "Negative-Binomial"),
    ("binomial-above-one", "binomial", _with(_binary(), 3, 2.0), 4, "2", "Binomial"),
    ("binomial-negative", "binomial", _with(_binary(), 0, -0.25), 1, "-0.25", "Binomial"),
    ("gamma-zero", "gamma", _with(_positive(), 4, 0.0), 5, "0", "Gamma"),
    ("gamma-negative", "gamma", _with(_positive(), 8, -2.0), 9, "-2", "Gamma"),
]


@pytest.mark.parametrize(
    "family, y, row, value, label",
    [case[1:] for case in SUPPORT_CASES],
    ids=[case[0] for case in SUPPORT_CASES],
)
def test_response_outside_family_support_is_a_data_error(family, y, row, value, label) -> None:
    x, _ = _base()
    with pytest.raises(gamfit.errors.DataError) as excinfo:
        gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family=family)
    message = str(excinfo.value)
    assert message.startswith("column 'y' "), message
    assert f"{label} family" in message, message
    assert f"first offending row {row} has value {value}" in message, message


def test_non_integer_counts_point_negative_binomial_at_poisson_and_tweedie() -> None:
    x, _ = _base()
    with pytest.raises(gamfit.errors.DataError, match="tweedie"):
        gamfit.fit({"x": x, "y": _counts() + 0.5}, "y ~ s(x)", family="negative-binomial")


def test_explicit_poisson_fits_a_non_negative_real_response() -> None:
    # gam#4572: the Poisson family's support is the non-negative reals; a
    # non-integer y is the ln-Gamma continuation of the log-mass (the Poisson
    # quasi-likelihood at dispersion one). See test_poisson_real_response_4572.py
    # for the X'y sufficiency this buys.
    x, _ = _base()
    model = gamfit.fit({"x": x, "y": _counts() + 0.5}, "y ~ s(x)", family="poisson")
    assert np.all(np.isfinite(_mean(model, GRID)))


def test_non_integer_non_negative_data_fits_under_tweedie() -> None:
    x, _ = _base()
    model = gamfit.fit({"x": x, "y": _counts() + 0.5}, "y ~ s(x)", family="tweedie(p=1.5)")
    assert np.all(np.isfinite(_mean(model, GRID)))


@pytest.mark.parametrize(
    "family, y, needle",
    [
        ("binomial", np.zeros(N), "all values are 0"),
        ("binomial", np.ones(N), "all values are 1"),
        ("poisson", np.zeros(N), "all counts are 0"),
    ],
    ids=["binomial-all-zero", "binomial-all-one", "poisson-all-zero"],
)
def test_degenerate_response_is_a_data_error(family, y, needle) -> None:
    x, _ = _base()
    with pytest.raises(gamfit.errors.DataError, match=needle):
        gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family=family)


# ---------------------------------------------------------------------------
# Weights: zero weight excludes the row
# ---------------------------------------------------------------------------


def test_zero_weight_row_is_not_judged_against_the_support() -> None:
    x, _ = _base()
    y = _with(_positive(), 4, 0.0)
    w = _with(np.ones(N), 4, 0.0)
    model = gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x)", family="gamma", weights="w")
    assert np.all(np.isfinite(_mean(model, GRID)))


def test_degeneracy_is_judged_over_positive_weight_rows() -> None:
    # The only 0 sits on an excluded row, so the likelihood sees all ones.
    x, _ = _base()
    y = _with(np.ones(N), 5, 0.0)
    w = _with(np.ones(N), 5, 0.0)
    with pytest.raises(gamfit.errors.DataError, match="all values are 1"):
        gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x)", family="binomial", weights="w")


@pytest.mark.parametrize(
    "w, needle",
    [
        (_with(np.ones(N), 7, -1.0), "must be non-negative; found -1 at row 8"),
        (np.zeros(N), "no positive weight"),
        (_with(np.ones(N), 7, np.nan), "row 8"),
        (_with(np.ones(N), 7, np.inf), "row 8"),
    ],
    ids=["negative", "all-zero", "nan", "inf"],
)
def test_invalid_weights_are_data_errors(w, needle) -> None:
    x, y = _base()
    with pytest.raises(gamfit.errors.DataError) as excinfo:
        gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x)", weights="w")
    message = str(excinfo.value)
    assert message.startswith("column 'w' "), message
    assert needle in message, message


def test_weights_of_the_wrong_length_are_refused() -> None:
    x, y = _base()
    with pytest.raises(ValueError):
        gamfit.fit({"x": x, "y": y, "w": np.ones(N - 1)}, "y ~ s(x)", weights="w")


# ---------------------------------------------------------------------------
# Degenerate tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formula", ["y ~ 1", "y ~ x", "y ~ s(x)"])
def test_single_observation_is_a_data_error(formula: str) -> None:
    with pytest.raises(gamfit.errors.DataError, match="too few rows"):
        gamfit.fit({"x": np.array([0.5]), "y": np.array([0.3])}, formula)


def test_all_nan_predictor_is_a_data_error() -> None:
    x, y = _base()
    with pytest.raises(gamfit.errors.DataError, match="column 'x' has no finite values"):
        gamfit.fit({"x": np.full(N, np.nan), "y": y}, "y ~ s(x)")


@pytest.mark.parametrize(
    "column, value, needle",
    [
        ("x", np.inf, "column 'x' has non-finite value inf at row 10"),
        ("x", np.nan, "column 'x' has non-finite value NaN at row 10"),
        ("y", np.nan, "column 'y' has non-finite value NaN at row 10"),
        ("y", -np.inf, "column 'y' has non-finite value -inf at row 10"),
    ],
    ids=["inf-x", "nan-x", "nan-y", "neg-inf-y"],
)
def test_non_finite_cells_are_rejected_not_dropped(column, value, needle) -> None:
    x, y = _base()
    data = {"x": x.copy(), "y": y.copy()}
    data[column][9] = value
    with pytest.raises(gamfit.errors.DataError) as excinfo:
        gamfit.fit(data, "y ~ s(x)")
    assert needle in str(excinfo.value), str(excinfo.value)
