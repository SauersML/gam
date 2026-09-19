"""Generic variance-function x link cells (families F2, F3, F9).

pyGAM ties each family to a hand-picked set of links and has no identity
Poisson, log Gaussian or log-binomial (relative-risk) model at all. gamfit
composes any exponential-dispersion variance function with any inverse link
whose range meets the family's mean domain; a link whose range overshoots the
domain carries a feasibility set on the linear predictor, enforced by step
rejection inside the solver. These tests pin:

* an unpenalized parametric fit in every generic cell is exactly the GLM
  maximum-likelihood fit, checked against an independent Fisher-scoring
  reference written here in numpy from the textbook ``V(mu)`` and ``mu(eta)``;
* identity-Poisson, log-Gaussian, inverse-Gamma and log-binomial GAMs each
  recover a smooth truth drawn from that cell, and their pointwise credible
  bands cover the truth;
* a likelihood whose maximum sits on the feasibility boundary is a typed
  error, not a clamped fit;
* binomial + identity stays illegal, with the message that says what to write
  instead.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

# Variance function V(mu) up to the dispersion.
_VARIANCE = {
    "gaussian": lambda mu: np.ones_like(mu),
    "poisson": lambda mu: mu,
    "gamma": lambda mu: mu * mu,
    "inverse-gaussian": lambda mu: mu**3,
    "binomial": lambda mu: mu * (1.0 - mu),
}

# Inverse link mu(eta) and its derivative dmu/deta.
_INVERSE_LINK = {
    "identity": (lambda eta: eta, lambda eta: np.ones_like(eta)),
    "log": (np.exp, np.exp),
    "sqrt": (lambda eta: eta * eta, lambda eta: 2.0 * eta),
    "inverse": (lambda eta: 1.0 / eta, lambda eta: -1.0 / (eta * eta)),
    "inverse-squared": (lambda eta: eta**-0.5, lambda eta: -0.5 * eta**-1.5),
}

# The link function g(mu) = eta, used only to seed the reference solve.
_LINK = {
    "identity": lambda mu: mu,
    "log": np.log,
    "sqrt": np.sqrt,
    "inverse": lambda mu: 1.0 / mu,
    "inverse-squared": lambda mu: mu**-2.0,
}


def _feasible(eta: np.ndarray, family: str, link: str) -> bool:
    """Whether every ``eta`` lies in the cell's feasibility set."""
    if family == "binomial" and link == "log":
        return bool(np.all(eta < 0.0))
    if link in ("sqrt", "inverse", "inverse-squared") or (
        link == "identity" and family != "gaussian"
    ):
        return bool(np.all(eta > 0.0))
    return True


def _fisher_scoring_glm(X: np.ndarray, y: np.ndarray, family: str, link: str) -> np.ndarray:
    """Reference GLM MLE by Fisher scoring on ``V(mu)`` and ``mu(eta)``.

    The dispersion cancels from the scoring step, so the MLE of ``beta`` does
    not depend on it. A scoring step that leaves the feasibility set is halved
    back into it. Iterates to a fixed point of the normal equations.
    """
    variance = _VARIANCE[family]
    mean, dmean = _INVERSE_LINK[link]
    start = np.clip(y, 0.05, 0.95) if family == "binomial" else np.maximum(y, 0.1)
    beta = np.linalg.lstsq(X, _LINK[link](start), rcond=None)[0]
    assert _feasible(X @ beta, family, link), "reference seed is infeasible"
    for _ in range(500):
        eta = X @ beta
        mu = mean(eta)
        dmu = dmean(eta)
        weight = dmu * dmu / variance(mu)
        working = eta + (y - mu) / dmu
        sqrt_w = np.sqrt(weight)
        new_beta = np.linalg.lstsq(X * sqrt_w[:, None], working * sqrt_w, rcond=None)[0]
        while not _feasible(X @ new_beta, family, link):
            new_beta = 0.5 * (beta + new_beta)
        if np.max(np.abs(new_beta - beta)) <= 1e-14 * (1.0 + np.max(np.abs(beta))):
            return new_beta
        beta = new_beta
    raise AssertionError("reference Fisher scoring did not reach its fixed point")


def _draw(rng: np.random.Generator, family: str, mu: np.ndarray, phi: float) -> np.ndarray:
    if family == "poisson":
        return rng.poisson(mu).astype(float)
    if family == "binomial":
        return (rng.uniform(size=mu.shape) < mu).astype(float)
    if family == "gamma":
        return rng.gamma(1.0 / phi, mu * phi)
    if family == "inverse-gaussian":
        return rng.wald(mu, 1.0 / phi)
    return mu + np.sqrt(phi) * rng.standard_normal(mu.shape)


# Every generic (non-fast-path) cell, with a positive-intercept linear
# predictor that keeps the truth strictly inside the cell's feasibility set.
_GENERIC_CELLS = [
    ("gaussian", "log", (0.5, 0.8), 0.05),
    ("gaussian", "sqrt", (1.0, 0.8), 0.05),
    ("gaussian", "inverse-squared", (1.0, 0.8), 0.001),
    ("poisson", "identity", (3.0, 4.0), 1.0),
    ("poisson", "sqrt", (1.5, 1.0), 1.0),
    ("poisson", "inverse", (0.3, 0.4), 1.0),
    ("poisson", "inverse-squared", (0.1, 0.1), 1.0),
    ("gamma", "identity", (2.0, 3.0), 0.2),
    ("gamma", "sqrt", (1.0, 0.8), 0.2),
    ("gamma", "inverse-squared", (1.0, 0.8), 0.2),
    ("inverse-gaussian", "identity", (1.0, 1.0), 0.2),
    ("inverse-gaussian", "sqrt", (1.0, 0.5), 0.2),
    ("inverse-gaussian", "inverse", (1.0, 0.8), 0.2),
    ("binomial", "log", (-1.5, 1.0), 1.0),
]


@pytest.mark.parametrize(("family", "link", "beta_true", "phi"), _GENERIC_CELLS)
def test_parametric_generic_cell_fit_is_the_glm_mle(family, link, beta_true, phi) -> None:
    rng = np.random.default_rng(20260919)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    eta = beta_true[0] + beta_true[1] * x
    mu = _INVERSE_LINK[link][0](eta)
    y = _draw(rng, family, mu, phi)

    # A linear term carries a REML-estimated shrinkage penalty by default;
    # switching it off leaves the unpenalized GLM likelihood.
    model = gamfit.fit(
        {"x": x, "y": y}, "y ~ linear(x, double_penalty=false)", family=family, link=link
    )
    estimates = np.array([row["estimate"] for row in model.summary().coefficients])

    reference = _fisher_scoring_glm(np.column_stack([np.ones(n), x]), y, family, link)
    np.testing.assert_allclose(estimates, reference, rtol=1e-6, atol=1e-8)


# Recovery cells: (family, link, truth mu(x), dispersion, n).
_RECOVERY_CELLS = [
    ("poisson", "identity", lambda x: 6.0 + 4.0 * np.sin(2.0 * np.pi * x), 1.0, 3000),
    ("gaussian", "log", lambda x: np.exp(0.5 + 0.6 * np.sin(2.0 * np.pi * x)), 0.04, 2000),
    ("gamma", "inverse", lambda x: 1.0 / (0.6 + 0.3 * np.sin(2.0 * np.pi * x)), 0.1, 3000),
    ("binomial", "log", lambda x: np.exp(-1.2 + 0.6 * np.sin(2.0 * np.pi * x)), 1.0, 6000),
]


@pytest.mark.parametrize(("family", "link", "truth", "phi", "n"), _RECOVERY_CELLS)
def test_generic_cell_gam_recovers_truth_and_bands_cover_it(family, link, truth, phi, n) -> None:
    rng = np.random.default_rng(101)
    x = rng.uniform(0.0, 1.0, n)
    y = _draw(rng, family, truth(x), phi)

    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family=family, link=link)

    grid = np.linspace(0.02, 0.98, 97)
    mu_grid = truth(grid)
    pred = model.predict({"x": grid}, interval=0.95)
    fitted = np.asarray(pred["mean_plugin"], dtype=float).reshape(-1)
    lower = np.asarray(pred["posterior_mean_lower"], dtype=float).reshape(-1)
    upper = np.asarray(pred["posterior_mean_upper"], dtype=float).reshape(-1)

    # The fit explains essentially all of the truth's variation.
    explained = 1.0 - np.mean((fitted - mu_grid) ** 2) / np.var(mu_grid)
    assert explained > 0.95, f"{family}/{link}: explained share {explained:.4f}"

    # Bayesian credible bands have across-the-function coverage near nominal
    # (Nychka 1988; Marra & Wood 2012); a band built on the wrong curvature
    # or the wrong dispersion falls far short of it.
    covered = np.mean((lower <= mu_grid) & (mu_grid <= upper))
    assert covered >= 0.85, f"{family}/{link}: band covers {covered:.3f} of the truth"


def test_identity_poisson_optimum_on_the_feasibility_boundary_is_a_typed_error() -> None:
    # Half the rows see only zero counts at a common design point: the
    # identity-Poisson likelihood increases without bound as that group's mean
    # falls to zero, so its maximum lies on the boundary eta = 0 of the
    # feasible set. A clamped or jittered fit would report a spurious mean.
    rng = np.random.default_rng(5)
    n = 400
    group = np.repeat([0.0, 1.0], n // 2)
    y = np.where(group == 0.0, 0.0, rng.poisson(5.0, n)).astype(float)
    with pytest.raises(Exception, match=r"feasib"):
        gamfit.fit(
            {"g": group, "y": y},
            "y ~ linear(g, double_penalty=false)",
            family="poisson",
            link="identity",
        )


def test_binomial_identity_is_spelled_as_a_gaussian_model() -> None:
    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, 100)
    y = (rng.uniform(size=100) < 0.4).astype(float)
    with pytest.raises(Exception) as caught:
        gamfit.fit({"x": x, "y": y}, "y ~ x", family="binomial", link="identity")
    message = str(caught.value)
    assert "a linear probability GAM is a Gaussian model and should be spelled as one" in message
