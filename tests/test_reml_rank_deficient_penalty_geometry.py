"""Public-API regressions for smoothing penalties with genuine null spaces."""

import numpy as np
import pytest

import gamfit


def spline_block(t, centered=False):
    degree, knots_inside = 3, 4
    design = np.asarray(gamfit.bspline_basis(t, knots_inside, degree=degree))
    interior = np.quantile(t, np.linspace(0, 1, knots_inside + 2)[1:-1])
    knots = np.r_[np.repeat(t.min(), degree + 1), interior, np.repeat(t.max(), degree + 1)]
    penalty = np.asarray(gamfit.smoothness_penalty(knots, degree=degree, order=2)[0])
    if centered:
        q, _ = np.linalg.qr(design.sum(axis=0).reshape(-1, 1), mode="complete")
        tangent = q[:, 1:]
        design, penalty = design @ tangent, tangent.T @ penalty @ tangent
    return design, penalty


@pytest.fixture(scope="module")
def block_problem():
    rng = np.random.default_rng(5)
    x = np.sort(rng.uniform(size=60))
    z = rng.uniform(size=60)
    first, s_first = spline_block(x)
    second, s_second = spline_block(z, centered=True)
    y = np.sin(4 * x) + 0.5 * z**2 + 0.15 * rng.normal(size=60)
    weights = rng.uniform(0.5, 2.0, size=60)
    args = ([first, second], [s_first, s_second], y)
    base = gamfit.gaussian_reml_fit_blocks_forward(*args, weights=weights)
    return args, weights, base


@pytest.mark.parametrize("row", range(60))
@pytest.mark.parametrize("sign", [-1, 1])
def test_block_reml_response_perturbation_is_continuous(block_problem, row, sign):
    """#2830: a tiny response perturbation cannot destroy a well-posed fit."""
    (designs, penalties, y), weights, base = block_problem
    shifted = y.copy()
    shifted[row] += sign * 1e-6
    fit = gamfit.gaussian_reml_fit_blocks_forward(designs, penalties, shifted, weights=weights)
    assert np.max(np.abs(np.asarray(fit["fitted"]) - np.asarray(base["fitted"]))) < 1e-4


@pytest.mark.parametrize("initial_rho", [5.0, 10.0, 15.0, 20.0, 25.0, 28.0, 30.0, 32.0, 35.0, 40.0])
def test_block_reml_initial_strength_does_not_change_the_fit(block_problem, initial_rho):
    args, weights, base = block_problem
    fit = gamfit.gaussian_reml_fit_blocks_forward(
        *args, weights=weights, init_rhos=np.array([-7.37, initial_rho])
    )
    assert np.max(np.abs(np.asarray(fit["fitted"]) - np.asarray(base["fitted"]))) < 1e-4


@pytest.fixture(scope="module")
def constrained_problem():
    rng = np.random.default_rng(9)
    t = np.sort(rng.uniform(size=50))
    design, penalty = spline_block(t)
    y = np.sin(5 * t) + 0.2 * rng.normal(size=50)
    weights = rng.uniform(0.5, 2.0, size=50)
    free = gamfit.gaussian_reml_fit_with_constraints_forward(design, y, penalty, weights=weights)
    return design, penalty, y, weights, np.asarray(free["coefficients"]).ravel()


@pytest.mark.parametrize("column", range(8))
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_binding_constraint_with_derivative_penalty(constrained_problem, column, sign):
    """#2831: both half-space orientations bind and survive a negligible ridge."""
    design, penalty, y, weights, free = constrained_problem
    a = np.zeros((1, design.shape[1]))
    a[0, column] = sign
    b = np.array([sign * free[column] + 0.3])
    kwargs = dict(weights=weights, a_inequality=a, b_inequality=b)
    fit = gamfit.gaussian_reml_fit_with_constraints_forward(design, y, penalty, **kwargs)
    beta = np.asarray(fit["coefficients"]).ravel()
    assert np.isfinite(beta).all()
    assert np.all(a @ beta >= b - 1e-8)
    np.testing.assert_array_equal(fit["active_indices"], [0])
    np.testing.assert_allclose(a @ beta, b, atol=1e-8, rtol=0)
    perturbed = gamfit.gaussian_reml_fit_with_constraints_forward(
        design, y, penalty + 1e-10 * np.eye(penalty.shape[0]), **kwargs
    )
    np.testing.assert_allclose(fit["fitted"], perturbed["fitted"], atol=1e-3, rtol=0)


@pytest.mark.parametrize("seed", range(40))
def test_shared_and_group_smooth_recovers_signal(seed):
    """#2834: irrelevant group deviations may shrink away without losing the shared curve."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(size=120))
    group = rng.integers(0, 3, size=120).astype(str)
    truth = np.sin(6 * x)
    data = {"x": x, "g": group, "y": truth + 0.3 * rng.normal(size=120)}
    fit = gamfit.fit(data, "y ~ s(x) + s(x, g, bs='fs')")
    predicted = np.asarray(fit.predict(data, return_type="pandas")["posterior_mean"])
    assert np.corrcoef(predicted, truth)[0, 1] > 0.95


@pytest.mark.parametrize("seed", range(30))
def test_firth_two_smooths_recovers_probability(seed):
    """#2835: Firth must retain both independently penalized likelihood directions."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(size=120))
    z = rng.uniform(size=120)
    probability = 1 / (1 + np.exp(-2 * (np.sin(5 * x) + 0.6 * np.cos(4 * z))))
    data = {"x": x, "z": z, "yb": (rng.uniform(size=120) < probability).astype(float)}
    fit = gamfit.fit(data, "yb ~ s(x, k=8) + s(z, k=8)", family="binomial", firth=True)
    predicted = np.asarray(fit.predict(data, return_type="pandas")["posterior_mean"])
    assert np.corrcoef(predicted, probability)[0, 1] > 0.80


@pytest.mark.parametrize("seed", [0, 2, 9])
def test_firth_preserves_fittability_of_the_plain_binomial_model(seed):
    """The same non-separated data must fit with and without the Jeffreys term."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(size=120))
    z = rng.uniform(size=120)
    probability = 1 / (1 + np.exp(-2 * (np.sin(5 * x) + 0.6 * np.cos(4 * z))))
    data = {"x": x, "z": z, "yb": (rng.uniform(size=120) < probability).astype(float)}
    for firth in (False, True):
        fit = gamfit.fit(
            data, "yb ~ s(x, k=8) + s(z, k=8)", family="binomial", firth=firth
        )
        predicted = np.asarray(fit.predict(data, return_type="pandas")["posterior_mean"])
        assert np.isfinite(predicted).all()
        assert np.corrcoef(predicted, probability)[0, 1] > 0.80
