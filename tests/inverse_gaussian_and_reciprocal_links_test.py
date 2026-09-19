"""Inverse Gaussian family and the reciprocal (inverse / inverse-squared) links.

pyGAM ships an inverse Gaussian family whose scale is stored as the square
root of the dispersion, and a Gamma inverse link that fails on ordinary data.
These tests pin the gamfit behaviour that replaces both:

* an unpenalized parametric fit on a reciprocal link is exactly the GLM
  maximum-likelihood fit (the no-smooth limit of the GAM), checked against an
  independent Fisher-scoring reference written here in numpy;
* the inverse Gaussian recovers a smooth truth on both of its links, and its
  estimated dispersion is the variance parameter ``phi`` of ``V = phi mu^3``
  (not ``sqrt(phi)``), inside the sampling band of the maximum-likelihood
  estimator;
* the inverse Gaussian refuses a non-positive response;
* an illegal family/link cell names the links the family does admit, listed
  from the engine's own legality table.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import gamfit

# Variance power ``p`` of ``V(mu) = mu^p`` and the reciprocal-power exponent
# ``a`` of ``mu = eta^(-a)`` for each reciprocal-link cell the engine admits.
_VARIANCE_POWER = {"gaussian": 0.0, "gamma": 2.0, "inverse-gaussian": 3.0}
_LINK_EXPONENT = {"inverse": 1.0, "inverse-squared": 0.5}


def _fisher_scoring_glm(X: np.ndarray, y: np.ndarray, family: str, link: str) -> np.ndarray:
    """Reference GLM MLE for ``mu = eta^(-a)``, ``V(mu) = mu^p``, by Fisher scoring.

    The dispersion cancels from the scoring step, so the MLE of ``beta`` does
    not depend on it. Iterates to a fixed point of the normal equations.
    """
    p = _VARIANCE_POWER[family]
    a = _LINK_EXPONENT[link]
    eta = y ** (-1.0 / a)
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    for _ in range(200):
        eta = X @ beta
        assert np.all(eta > 0.0), "reference iterate left the link domain"
        mu = eta ** (-a)
        dmu = -a * eta ** (-a - 1.0)
        weight = dmu * dmu / mu**p
        working = eta + (y - mu) / dmu
        sqrt_w = np.sqrt(weight)
        new_beta = np.linalg.lstsq(X * sqrt_w[:, None], working * sqrt_w, rcond=None)[0]
        if np.max(np.abs(new_beta - beta)) <= 1e-14 * (1.0 + np.max(np.abs(beta))):
            return new_beta
        beta = new_beta
    raise AssertionError("reference Fisher scoring did not reach its fixed point")


def _draw(rng: np.random.Generator, family: str, mu: np.ndarray, phi: float) -> np.ndarray:
    if family == "inverse-gaussian":
        # numpy's Wald(mean, scale) has variance mean^3 / scale, so scale = 1/phi.
        return rng.wald(mu, 1.0 / phi)
    if family == "gamma":
        return rng.gamma(1.0 / phi, mu * phi)
    return mu + np.sqrt(phi) * rng.standard_normal(mu.shape)


def _find_key(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _estimated_dispersion(model) -> float:
    scale = _find_key(json.loads(model.dumps()), "likelihood_scale")
    assert isinstance(scale, dict) and "EstimatedDispersion" in scale, scale
    return float(scale["EstimatedDispersion"]["phi"])


@pytest.mark.parametrize(
    ("family", "link"),
    [("gamma", "inverse"), ("gaussian", "inverse"), ("inverse-gaussian", "inverse-squared")],
)
def test_parametric_reciprocal_link_fit_is_the_glm_mle(family: str, link: str) -> None:
    rng = np.random.default_rng(20260919)
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    beta_true = np.array([1.0, 0.8])
    eta = beta_true[0] + beta_true[1] * x
    mu = eta ** (-_LINK_EXPONENT[link])
    y = _draw(rng, family, mu, 0.2 if family != "gaussian" else 0.01)

    # A linear term carries a REML-estimated shrinkage penalty by default;
    # switching it off leaves the unpenalized GLM likelihood.
    model = gamfit.fit(
        {"x": x, "y": y}, "y ~ linear(x, double_penalty=false)", family=family, link=link
    )
    estimates = np.array([row["estimate"] for row in model.summary().coefficients])

    reference = _fisher_scoring_glm(np.column_stack([np.ones(n), x]), y, family, link)
    np.testing.assert_allclose(estimates, reference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("link", ["log", "inverse-squared"])
def test_inverse_gaussian_recovers_smooth_truth_and_dispersion(link: str) -> None:
    rng = np.random.default_rng(7)
    n = 3000
    phi = 0.3
    x = rng.uniform(0.0, 1.0, n)
    mu = np.exp(0.3 + 0.5 * np.sin(2.0 * np.pi * x))
    y = rng.wald(mu, 1.0 / phi)

    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="inverse-gaussian", link=link)
    fitted = np.asarray(model.predict({"x": x}), dtype=float).reshape(-1)

    # The fit explains essentially all of the truth's variation.
    explained = 1.0 - np.mean((fitted - mu) ** 2) / np.var(mu)
    assert explained > 0.97, f"{link}: explained share of truth variance {explained:.4f}"

    # The MLE of the inverse Gaussian dispersion has asymptotic variance
    # 2 phi^2 / n; phi_hat must sit inside four standard errors of phi. A
    # square-root scale (pyGAM's convention) would report ~0.55 here.
    phi_hat = _estimated_dispersion(model)
    band = 4.0 * np.sqrt(2.0 / n) * phi
    assert abs(phi_hat - phi) < band, f"{link}: phi_hat={phi_hat} phi={phi} band={band}"


def test_inverse_gaussian_refuses_a_nonpositive_response() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, 200)
    y = rng.wald(np.ones(200), 2.0)
    y[17] = 0.0
    with pytest.raises(Exception, match=r"Inverse-Gaussian family requires strictly positive response values \(y > 0\)"):
        gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="inverse-gaussian")


@pytest.mark.parametrize(
    ("family", "link", "legal"),
    [
        # Every positive-mean family composes with every link whose range
        # meets its mean domain; a probability link is not one of them.
        ("gamma", "logit", "identity|log|sqrt|inverse|inverse-squared"),
        ("inverse-gaussian", "probit", "identity|log|sqrt|inverse|inverse-squared"),
        ("gaussian", "cloglog", "identity|log|sqrt|inverse|inverse-squared"),
    ],
)
def test_illegal_link_error_lists_the_family_legal_links(family: str, link: str, legal: str) -> None:
    rng = np.random.default_rng(11)
    x = rng.uniform(0.0, 1.0, 100)
    y = rng.gamma(4.0, 0.25, 100)
    with pytest.raises(Exception) as caught:
        gamfit.fit({"x": x, "y": y}, "y ~ x", family=family, link=link)
    assert f"legal links for `{family}`: {legal}" in str(caught.value), str(caught.value)
