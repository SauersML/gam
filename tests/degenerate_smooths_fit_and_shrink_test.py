"""Irrelevant, noiseless and degenerate smooths must fit, shrink and predict.

Two defects made these fits abort or degrade:

1. When REML drives smoothing parameters onto their box rails (an irrelevant
   smooth with the double penalty rails every lambda at +inf), the sigma-point
   smoothing-correction cubature built its nodes along eigen-axes of the full
   rho covariance. At a corner of the rho box such an axis mixes railed
   coordinates with opposite signs, so it leaves the box in both directions
   and has zero width: "smoothing cubature proposal has no positive width".
   The fix conditions railed coordinates on their face (at a rail
   dbeta/drho -> 0, so the integrand is a point mass there) and integrates only
   the free block of the rho covariance.

2. The default B-spline basis floors at 4 internal knots, so a covariate with
   2 or 3 unique values got an 8-column basis that its support cannot identify.
   The default basis dimension is now capped by the number of unique values.
"""

import numpy as np
import pytest

import gamfit


def _smooth_edfs(model):
    return [term["edf"] for term in model.summary().smooth_terms]


def _block_width(model, name):
    for block in model.term_blocks:
        if block.name == name:
            return block.end - block.start
    raise AssertionError(f"no term block named {name!r} in {model.term_blocks}")


@pytest.mark.parametrize("seed", [1, 6, 10])
def test_two_irrelevant_smooths_rail_and_correct_without_a_cubature_failure(seed, capfd):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(200, 2))
    y = rng.normal(size=200)
    model = gamfit.fit({"x0": x[:, 0], "x1": x[:, 1], "y": y}, "y ~ s(x0) + s(x1)")
    captured = capfd.readouterr()
    log = captured.out + captured.err

    assert "no positive width" not in log
    assert "numerical-failure" not in log
    summary = model.summary()
    assert summary.convergence["certified"]
    for edf in _smooth_edfs(model):
        assert 0.0 <= edf < 1e-3
    prediction = np.asarray(
        model.predict({"x0": np.array([-1.0, 0.0, 1.0]), "x1": np.array([0.5, 0.0, -0.5])})
    ).ravel()
    assert np.all(np.isfinite(prediction))
    np.testing.assert_allclose(prediction, np.full(3, y.mean()), atol=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_pure_noise_smooth_shrinks_toward_its_null_space(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=200)
    y = rng.normal(size=200)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)")
    assert model.summary().convergence["certified"]
    (edf,) = _smooth_edfs(model)
    assert 0.0 <= edf < 1.0
    prediction = np.asarray(model.predict({"x": np.array([0.2, 0.5, 0.8])})).ravel()
    assert np.all(np.isfinite(prediction))
    assert np.all(np.abs(prediction - y.mean()) < 0.3)


def test_constant_response_smooth_shrinks_to_zero():
    x = np.linspace(0.0, 1.0, 50)
    model = gamfit.fit({"x": x, "y": np.full(50, 3.0)}, "y ~ s(x)")
    (edf,) = _smooth_edfs(model)
    assert 0.0 <= edf < 1e-3
    prediction = np.asarray(model.predict({"x": np.array([0.2, 0.5])})).ravel()
    np.testing.assert_allclose(prediction, [3.0, 3.0], atol=0.05)


def test_exact_linear_response_is_fit_by_the_null_space():
    x = np.linspace(0.0, 1.0, 50)
    model = gamfit.fit({"x": x, "y": 2.0 * x + 1.0}, "y ~ s(x)")
    (edf,) = _smooth_edfs(model)
    assert edf == pytest.approx(1.0, abs=1e-3)
    prediction = np.asarray(model.predict({"x": np.array([0.2, 0.5])})).ravel()
    np.testing.assert_allclose(prediction, [1.4, 2.0], atol=1e-6)


@pytest.mark.parametrize(
    ("unique", "expected_width"),
    [(2, 1), (3, 2), (4, 3)],
)
def test_default_basis_is_capped_by_the_covariate_support(unique, expected_width):
    rng = np.random.default_rng(4)
    x = rng.integers(0, unique, 300).astype(float)
    truth = (x - (unique - 1) / 2.0) ** 2
    y = truth + rng.normal(0.0, 0.5, 300)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)")

    # The centered basis has one column fewer than the unique values it spans.
    assert _block_width(model, "s(x)") == expected_width
    assert model.summary().convergence["certified"]
    (edf,) = _smooth_edfs(model)
    assert 0.0 <= edf <= expected_width + 1e-6
    grid = np.arange(unique, dtype=float)
    prediction = np.asarray(model.predict({"x": grid})).ravel()
    group_means = np.array([y[x == value].mean() for value in grid])
    np.testing.assert_allclose(prediction, group_means, atol=0.25)


def test_explicit_basis_dimension_larger_than_n_fits_and_predicts():
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 1.0, 12)
    y = np.sin(6.0 * x) + rng.normal(0.0, 0.1, 12)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x, k=20)")
    assert model.summary().convergence["certified"]
    (edf,) = _smooth_edfs(model)
    assert 0.0 <= edf < 11.0
    prediction = np.asarray(model.predict({"x": np.array([0.2, 0.5])})).ravel()
    np.testing.assert_allclose(prediction, np.sin(6.0 * np.array([0.2, 0.5])), atol=0.3)


def test_tensor_smooth_on_fifteen_rows_fits_and_predicts():
    rng = np.random.default_rng(7)
    x = rng.uniform(size=15)
    z = rng.uniform(size=15)
    y = x + z + rng.normal(0.0, 0.1, 15)
    model = gamfit.fit({"x": x, "z": z, "y": y}, "y ~ te(x, z)")
    assert model.summary().convergence["certified"]
    (edf,) = _smooth_edfs(model)
    assert 0.0 <= edf < 14.0
    prediction = np.asarray(model.predict({"x": np.array([0.3]), "z": np.array([0.6])})).ravel()
    np.testing.assert_allclose(prediction, [0.9], atol=0.2)
