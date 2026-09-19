"""Prediction uncertainty is the posterior of η and of μ = g⁻¹(η), for every
family (pyGAM audit robustness F5 / inference G3).

``Model.predict(interval=...)`` reports

* ``linear_predictor_standard_error`` — the posterior SD of η,
  ``√diag(G V Gᵀ)`` under the covariance the band names;
* ``posterior_mean_standard_error`` — the posterior SD of μ, from the same
  Gaussian η integral as the ``posterior_mean`` point (the delta method
  ``|dμ/dη|·SE(η)`` collapses to zero wherever the inverse link saturates,
  while the posterior of μ stays wide);
* ``posterior_mean_lower`` / ``posterior_mean_upper`` — the inverse link of the
  η credible quantiles.

``Model.sample`` draws carry the smoothing-parameter uncertainty for every
family (non-Gaussian draws used to be conditional on ρ̂), and its μ draws are
the inverse link of its η draws.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

_LEVEL = 0.95
# Two-sided 95% standard-normal quantile.
_Z = 1.959963984540054


def _expit(eta: Any) -> Any:
    return 0.5 * (1.0 + np.tanh(0.5 * eta))


_INVERSE_LINK = {"gaussian": lambda eta: eta, "poisson": np.exp, "binomial": _expit}


def _truth(x: Any) -> Any:
    return 0.3 + 0.9 * np.sin(2.0 * np.pi * x)


def _simulate(family: str, seed: int, n: int = 300) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    eta = _truth(x)
    if family == "gaussian":
        y = eta + 0.5 * rng.standard_normal(n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(float)
    else:
        y = rng.binomial(1, _expit(eta)).astype(float)
    return {"x": x, "y": y}


def _predict(model: Any, grid: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    table = model.predict(grid, interval=_LEVEL, return_type="dict", **kwargs)
    return {key: np.asarray(table[key]) for key in table.keys()}


def _gauss_hermite_moments(link: Any, mean: Any, sd: Any) -> tuple[Any, Any]:
    """``E[g⁻¹(η)]`` and ``Var[g⁻¹(η)]`` for ``η ~ N(mean, sd²)`` per row."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(200)
    weights = weights / weights.sum()
    values = link(mean[:, None] + sd[:, None] * nodes[None, :])
    first = values @ weights
    centred = values - first[:, None]
    return first, (centred * centred) @ weights


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial"])
def test_eta_standard_error_is_the_covariance_quadratic_form(family: str) -> None:
    data = _simulate(family, seed=11)
    model = gamfit.fit(data, "y ~ s(x, k=10)", family=family)
    grid = {"x": np.linspace(0.02, 0.98, 25)}
    design = model.design_matrix(grid)
    gradient = np.asarray(design.eta_gradient, dtype=float)

    for mode, covariance, source in (
        (None, design.covariance_smoothing_corrected, "smoothing-corrected"),
        ("conditional", design.covariance_conditional, "conditional"),
    ):
        table = model.predict(
            grid, interval=_LEVEL, return_type="dict", covariance_mode=mode
        )
        assert table.covariance_source == source
        expected = np.sqrt(
            np.einsum("ij,jk,ik->i", gradient, np.asarray(covariance), gradient)
        )
        np.testing.assert_allclose(
            np.asarray(table["linear_predictor_standard_error"], dtype=float),
            expected,
            rtol=1e-8,
            err_msg=f"{family}/{source}: SE(η) is not √diag(G V Gᵀ)",
        )
        eta_hat = np.asarray(design.offset) + np.asarray(design.matrix) @ np.asarray(
            design.coefficients
        )
        np.testing.assert_allclose(
            np.asarray(table["linear_predictor_plugin"], dtype=float), eta_hat, rtol=1e-10
        )


@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_response_band_is_the_inverse_link_of_the_eta_quantiles(family: str) -> None:
    model = gamfit.fit(_simulate(family, seed=12), "y ~ s(x, k=10)", family=family)
    out = _predict(model, {"x": np.linspace(0.02, 0.98, 25)})
    eta = out["linear_predictor_plugin"].astype(float)
    sd = out["linear_predictor_standard_error"].astype(float)
    link = _INVERSE_LINK[family]
    np.testing.assert_allclose(out["posterior_mean_lower"], link(eta - _Z * sd), rtol=1e-9)
    np.testing.assert_allclose(out["posterior_mean_upper"], link(eta + _Z * sd), rtol=1e-9)


@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_posterior_sd_of_mean_is_the_same_integral_as_the_mean(family: str) -> None:
    model = gamfit.fit(_simulate(family, seed=13), "y ~ s(x, k=10)", family=family)
    # Conditional mode: the band and the posterior-mean point integrate one
    # covariance, so both moments come from one η posterior.
    out = _predict(model, {"x": np.linspace(0.02, 0.98, 25)}, covariance_mode="conditional")
    eta = out["linear_predictor_plugin"].astype(float)
    sd = out["linear_predictor_standard_error"].astype(float)
    mean, variance = _gauss_hermite_moments(_INVERSE_LINK[family], eta, sd)
    np.testing.assert_allclose(out["posterior_mean"], mean, rtol=1e-7)
    np.testing.assert_allclose(out["posterior_mean_standard_error"], np.sqrt(variance), rtol=1e-6)


def test_separated_binomial_posterior_sd_does_not_collapse() -> None:
    """Under separation η̂ runs off to where σ'(η̂) ≈ 0, so the delta-method SD
    underflows (the audit measured 1.3e-22) while the posterior of μ keeps a
    wide spread from the large SE(η)."""
    x = np.linspace(-1.0, 1.0, 120)
    data = {"x": x, "y": (x > 0.0).astype(float)}
    model = gamfit.fit(data, "y ~ x", family="binomial")
    out = _predict(
        model, {"x": np.array([-0.9, -0.5, -0.1, 0.1, 0.5, 0.9])}, covariance_mode="conditional"
    )
    eta = out["linear_predictor_plugin"].astype(float)
    sd = out["linear_predictor_standard_error"].astype(float)
    mean, variance = _gauss_hermite_moments(_expit, eta, sd)
    reported = out["posterior_mean_standard_error"].astype(float)
    np.testing.assert_allclose(out["posterior_mean"], mean, rtol=1e-6)
    np.testing.assert_allclose(reported, np.sqrt(variance), rtol=1e-6)
    mu_hat = _expit(eta)
    delta_method = mu_hat * (1.0 - mu_hat) * sd
    assert np.all(reported > 1e3 * delta_method), (reported, delta_method)
    assert np.all(reported > 1e-2)


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial"])
def test_eta_and_mean_bands_cover_the_truth_at_the_nominal_rate(family: str) -> None:
    grid_x = np.linspace(0.05, 0.95, 40)
    eta_true = _truth(grid_x)
    mu_true = _INVERSE_LINK[family](eta_true)
    eta_hits = []
    mu_hits = []
    for seed in range(40):
        model = gamfit.fit(_simulate(family, seed=1000 + seed), "y ~ s(x, k=10)", family=family)
        out = _predict(model, {"x": grid_x})
        eta = out["linear_predictor_plugin"].astype(float)
        sd = out["linear_predictor_standard_error"].astype(float)
        eta_hits.append(np.abs(eta_true - eta) <= _Z * sd)
        mu_hits.append(
            (out["posterior_mean_lower"] <= mu_true) & (mu_true <= out["posterior_mean_upper"])
        )
    eta_coverage = float(np.mean(eta_hits))
    mu_coverage = float(np.mean(mu_hits))
    # Across-the-function Bayesian coverage (Nychka 1988) averages to the
    # nominal level; 40 fits × 40 correlated grid points resolve it to a few
    # percent.
    assert abs(eta_coverage - _LEVEL) <= 0.04, eta_coverage
    assert abs(mu_coverage - _LEVEL) <= 0.04, mu_coverage


@pytest.mark.parametrize("family", ["poisson", "binomial"])
def test_sample_draws_carry_the_smoothing_correction(family: str) -> None:
    data = _simulate(family, seed=21, n=400)
    model = gamfit.fit(data, "y ~ s(x, k=8)", family=family)
    design = model.design_matrix(data)
    corrected = np.asarray(design.covariance_smoothing_corrected, dtype=float)
    conditional = np.asarray(design.covariance_conditional, dtype=float)

    posterior = model.sample(data, seed=5, samples=4000)
    assert posterior.covariance_source == "smoothing-corrected"

    draws = np.asarray(posterior.samples, dtype=float)
    drawn_sd = draws.std(axis=0, ddof=1)
    np.testing.assert_allclose(np.asarray(posterior.std, dtype=float), drawn_sd, rtol=1e-10)
    corrected_sd = np.sqrt(np.diag(corrected))
    conditional_sd = np.sqrt(np.diag(conditional))
    # Wiggly-basis directions are the ones the smoothing parameter moves.
    moved = corrected_sd > 1.05 * conditional_sd
    assert np.any(moved)
    ratio = drawn_sd / corrected_sd
    assert np.all(np.abs(ratio - 1.0) <= 0.1), ratio
    closer = np.abs(drawn_sd - corrected_sd) < np.abs(drawn_sd - conditional_sd)
    assert np.all(closer[moved]), (drawn_sd, corrected_sd, conditional_sd)

    grid = {"x": np.linspace(0.05, 0.95, 9)}
    predictive = posterior.predict_draws(grid)
    np.testing.assert_allclose(
        predictive.mean, _INVERSE_LINK[family](predictive.eta), rtol=1e-12, atol=0.0
    )
