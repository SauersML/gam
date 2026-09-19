"""pyGAM oracle: families (pygam/tests/test_GAM_methods.py, test_distributions.py).

pyGAM fits LinearGAM, LogisticGAM, PoissonGAM and GammaGAM on its bundled
datasets and checks that the fit completes and the predictions are in the
response domain. Here each family is fitted on seeded synthetic data drawn from
that family, and every fit must also:

* report a certified convergence,
* predict means inside the family's support, and
* recover the generating mean curve better than the intercept-only model.

pyGAM is never imported.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import gamfit


def _truth(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x)


# family -> (draw y given eta, inverse link, open/closed support check)
FAMILIES: dict[str, tuple[Callable[[np.random.Generator, np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], bool]]] = {
    "gaussian": (
        lambda rng, eta: eta + rng.normal(0.0, 0.3, eta.size),
        lambda eta: eta,
        lambda mu: bool(np.all(np.isfinite(mu))),
    ),
    "binomial": (
        lambda rng, eta: (rng.uniform(size=eta.size) < 1.0 / (1.0 + np.exp(-eta))).astype(float),
        lambda eta: 1.0 / (1.0 + np.exp(-eta)),
        lambda mu: bool(np.all((mu > 0.0) & (mu < 1.0))),
    ),
    "poisson": (
        lambda rng, eta: rng.poisson(np.exp(eta)).astype(float),
        np.exp,
        lambda mu: bool(np.all(mu > 0.0)),
    ),
    "gamma": (
        lambda rng, eta: rng.gamma(5.0, np.exp(eta) / 5.0),
        np.exp,
        lambda mu: bool(np.all(mu > 0.0)),
    ),
}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_fit_is_certified_in_domain_and_informative(family: str) -> None:
    draw, inverse_link, in_support = FAMILIES[family]
    rng = np.random.default_rng(70 + list(FAMILIES).index(family))
    n = 800
    x = rng.uniform(0.0, 1.0, n)
    eta = 1.0 + _truth(x)
    d = {"x": x, "y": draw(rng, eta)}
    m = gamfit.fit(d, "y ~ s(x)", family=family)
    assert "certified" in list(m.summary().convergence), m.summary().convergence

    grid = np.linspace(0.0, 1.0, 50)
    mu = np.asarray(m.predict({"x": grid}), float)
    assert in_support(mu), (family, mu.min(), mu.max())

    target = inverse_link(1.0 + _truth(grid))
    null = np.full_like(target, float(np.mean(d["y"])))
    err_fit = float(np.sqrt(np.mean((mu - target) ** 2)))
    err_null = float(np.sqrt(np.mean((null - target) ** 2)))
    assert err_fit < 0.25 * err_null, (family, err_fit, err_null)


def test_binomial_two_smooth_fit_finds_the_signal() -> None:
    """pyGAM's LogisticGAM on the default dataset: here a balanced synthetic
    stand-in with a monotone signal and a nuisance covariate."""
    rng = np.random.default_rng(75)
    n = 1000
    x = rng.uniform(-2.0, 2.0, n)
    z = rng.uniform(-2.0, 2.0, n)
    p = 1.0 / (1.0 + np.exp(-(1.5 * x)))
    d = {"x": x, "z": z, "y": (rng.uniform(size=n) < p).astype(float)}
    m = gamfit.fit(d, "y ~ s(x) + s(z)", family="binomial")
    mu = np.asarray(m.predict(d), float)
    assert np.all((mu > 0.0) & (mu < 1.0))
    # The fitted probability is increasing in x on average (the signal is found).
    order = np.argsort(x)
    lo, hi = mu[order[: n // 4]].mean(), mu[order[-n // 4 :]].mean()
    assert hi - lo > 0.5
