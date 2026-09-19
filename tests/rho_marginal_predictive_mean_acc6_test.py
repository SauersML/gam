"""ACC-6 (pyGAM audit): the point prediction's estimand, and the rho-marginal
alternative the audit proposed.

The audit asked whether ``predict`` should integrate over the smoothing
parameters, ``E[mu(x) | y] = ∫ E[mu(x) | rho, y] p(rho | y) d rho``, when the
LAML surface over ``rho`` is flat (coal-mining disasters, Poisson). It was
measured over the whole ``bench_accuracy.py`` battery and on independent
replicates (``bench/pygam_audit/accuracy/acc6/README.md``). The first-order
integral changes accuracy by sub-percent amounts of either sign, and the full
integral is worse wherever its effect is measurable. So the shipped estimand
stays the conditional posterior mean at ``rho_hat``:

    posterior_mean = E[g^{-1}(eta)],  eta ~ N(x' beta_hat, x' V_beta x)

with ``V_beta`` the conditional covariance, not the smoothing-corrected ``V_p``
(#398). The evidence compared against this exact reconstruction, so the tests
pin it through the public affine design: the log link in closed form
``exp(eta + s^2 / 2)`` and the logit link by trapezoid quadrature.

The second half of each test checks that the pin can tell the two estimands
apart. The first-order rho-marginal mean replaces ``V_beta`` by ``V_p`` in the
same integral, and on these fits it moves the prediction far beyond the
tolerance used above.
"""

from __future__ import annotations

import numpy as np

import gamfit

# Trapezoid rule for E[logistic(eta + s z)], z ~ N(0, 1). The integrand is
# analytic in a strip around the real axis, so the rule converges
# geometrically; the normal mass outside [-12, 12] is below 1e-32.
_Z = np.linspace(-12.0, 12.0, 8001)
_W = np.exp(-0.5 * _Z**2) / np.sqrt(2.0 * np.pi) * (_Z[1] - _Z[0])
_W[[0, -1]] *= 0.5


def _logistic_expectation(eta: np.ndarray, var: np.ndarray) -> np.ndarray:
    z = eta[:, None] + np.sqrt(var)[:, None] * _Z[None, :]
    return (0.5 * (1.0 + np.tanh(0.5 * z))) @ _W


def _eta_and_variances(model: gamfit.Model, new: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    affine = model.design_matrix(new)
    eta = np.asarray(affine.offset) + np.asarray(affine.matrix) @ np.asarray(affine.coefficients)
    grad = np.asarray(affine.eta_gradient)
    conditional = affine.covariance_conditional
    corrected = affine.covariance_smoothing_corrected
    assert conditional is not None
    assert corrected is not None
    var_b = np.einsum("ij,jk,ik->i", grad, np.asarray(conditional), grad)
    var_p = np.einsum("ij,jk,ik->i", grad, np.asarray(corrected), grad)
    return eta, var_b, var_p


def test_poisson_posterior_mean_is_the_conditional_lognormal_mean() -> None:
    # Low counts on a rate that is flat, then falls: the coal-mining shape
    # whose broad LAML surface motivated ACC-6.
    rng = np.random.default_rng(1851)
    x = np.sort(rng.uniform(0.0, 1.0, 150))
    y = rng.poisson(np.exp(1.0 - 2.2 / (1.0 + np.exp(-12.0 * (x - 0.35))))).astype(float)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="poisson")
    new = {"x": np.linspace(0.0, 1.0, 101)}

    shipped = np.asarray(model.predict(new), dtype=float).ravel()
    eta, var_b, var_p = _eta_and_variances(model, new)
    conditional = np.exp(eta + 0.5 * var_b)
    rho_marginal = np.exp(eta + 0.5 * var_p)

    np.testing.assert_allclose(shipped, conditional, rtol=1e-12, atol=0.0)
    assert np.max(np.abs(rho_marginal / conditional - 1.0)) > 1e-3


def test_binomial_posterior_mean_is_the_conditional_logistic_normal_mean() -> None:
    rng = np.random.default_rng(3005)
    x = rng.uniform(0.0, 1.0, 500)
    y = (rng.uniform(size=500) < 1.0 / (1.0 + np.exp(-2.0 * np.sin(4.0 * np.pi * x)))).astype(float)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="binomial")
    new = {"x": np.linspace(0.0, 1.0, 101)}

    shipped = np.asarray(model.predict(new), dtype=float).ravel()
    eta, var_b, var_p = _eta_and_variances(model, new)
    conditional = _logistic_expectation(eta, var_b)
    rho_marginal = _logistic_expectation(eta, var_p)

    np.testing.assert_allclose(shipped, conditional, rtol=1e-8, atol=0.0)
    assert np.max(np.abs(rho_marginal / conditional - 1.0)) > 1e-3
