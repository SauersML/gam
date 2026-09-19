"""The Firth/Jeffreys estimator is never adopted silently.

* An ordinary fit that fails for a reason other than proven separation is not
  retried under the Jeffreys prior: the non-separated binomial MC cell that
  used to switch estimators (pyGAM audit, inference cell ``binom``, rep 21)
  is fitted by plain penalized likelihood and says so.
* When the estimator does change, the model says which estimator ran and
  why, in the same words in ``summary()``, the model ``repr`` and the CLI's
  ``gam summary``: all three print the line rendered by the engine.
* Posterior draws under the Jeffreys prior come from the exact Pólya-Gamma
  kernel with a Metropolis step for the ``|I(beta)|^(1/2)`` factor, and match
  the posterior computed by quadrature within Monte Carlo error.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit

PLAIN = "penalized likelihood"
JEFFREYS = "penalized likelihood with Jeffreys prior"


def _gam_bin() -> str:
    candidates = [
        os.environ.get("GAM_BIN"),
        "target/release/gam",
        "target/debug/gam",
        shutil.which("gam"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    pytest.skip("gam binary not built")


def _estimator_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("Estimator")]
    assert len(lines) == 1, f"expected one Estimator line in:\n{text}"
    return lines[0]


def _mc_binom_rep(rep: int) -> dict[str, np.ndarray]:
    # bench/pygam_audit/inference/mc.py, cell "binom": n=400, eta = 1.5 sin(2 pi x1)
    # + 0.6 cos(2 pi x3), Bernoulli draws, seeded by 1000 + rep.
    rng = np.random.default_rng(1000 + rep)
    x = rng.uniform(0, 1, (400, 3))
    eta = 1.5 * np.sin(2 * np.pi * x[:, 0]) + 0.6 * np.cos(2 * np.pi * x[:, 2])
    y = (rng.uniform(size=400) < 1 / (1 + np.exp(-eta))).astype(float)
    return {"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "y": y}


def _separated() -> dict[str, np.ndarray]:
    x = np.array([-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
    return {"x": x, "y": (x > 0).astype(float)}


def test_non_separated_binomial_rep21_keeps_plain_penalized_likelihood() -> None:
    model = gamfit.fit(_mc_binom_rep(21), "y ~ s(x1) + s(x2) + s(x3)", family="binomial")
    estimator = model.summary().convergence["estimator"]
    assert estimator["name"] == PLAIN, estimator
    assert estimator["reason"] is None, estimator
    assert f"estimator={PLAIN!r}" in repr(model)


def test_separated_fit_names_its_estimator_identically_everywhere(tmp_path) -> None:
    model = gamfit.fit(_separated(), "y ~ x", family="binomial")
    estimator = model.summary().convergence["estimator"]
    assert estimator["name"] == JEFFREYS, estimator
    assert "separat" in estimator["reason"], estimator

    python_line = _estimator_line(str(model.summary()))
    assert python_line == f"Estimator: {estimator['text']}"
    assert f"estimator={estimator['text']!r}" in repr(model)

    path = tmp_path / "separated.gam.json"
    model.save(path)
    cli = subprocess.run(
        [_gam_bin(), "summary", str(path)], check=True, capture_output=True, text=True
    )
    assert _estimator_line(cli.stdout) == python_line


def _fitted_penalty(model) -> np.ndarray:
    """The fit's own lambda * S: penalized Hessian minus the weighted Gram at the mode."""
    inference = json.loads(model.dumps())["payload"]["fit_result"]["inference"]
    hessian = np.asarray(inference["penalized_hessian"]["data"]).reshape(2, 2)
    gram = np.asarray(inference["weighted_gram"]["data"]).reshape(2, 2)
    return hessian - gram


def _jeffreys_quadrature(x: np.ndarray, y: np.ndarray, penalty: np.ndarray):
    """Marginals of exp(loglik - b'Sb/2) * |X'WX|^(1/2) for y ~ 1 + x on a dense grid."""
    g0 = np.linspace(-15.0, 15.0, 801)
    g1 = np.linspace(-4.0, 12.0, 801)
    b0, b1 = np.meshgrid(g0, g1, indexing="ij")
    eta = b0[..., None] + b1[..., None] * x
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1.0 - mu)
    i00, i01, i11 = w.sum(-1), (w * x).sum(-1), (w * x * x).sum(-1)
    quadratic = penalty[0, 0] * b0**2 + 2.0 * penalty[0, 1] * b0 * b1 + penalty[1, 1] * b1**2
    log_density = (
        (y * eta - np.logaddexp(0.0, eta)).sum(-1)
        + 0.5 * np.log(i00 * i11 - i01 * i01)
        - 0.5 * quadratic
    )
    mass = np.exp(log_density - log_density.max())
    mass /= mass.sum()
    return (g0, mass.sum(axis=1)), (g1, mass.sum(axis=0))


def test_firth_sample_is_exact_polya_gamma_with_reported_acceptance() -> None:
    data = _separated()
    model = gamfit.fit(data, "y ~ x", family="binomial", firth=True)
    draws = model.sample(data, samples=4000, seed=7)

    assert draws.method == "polya-gamma-jeffreys"
    assert draws.is_exact
    assert draws.acceptance_rate is not None and 0.0 < draws.acceptance_rate <= 1.0
    assert draws.summary().extras["acceptance_rate"] == draws.acceptance_rate
    assert draws.converged, (draws.rhat, draws.ess)

    samples = np.asarray(draws.samples)
    reference = _jeffreys_quadrature(data["x"], data["y"], _fitted_penalty(model))
    for column, (grid, marginal) in enumerate(reference):
        reference_mean = float(marginal @ grid)
        reference_sd = float(np.sqrt(marginal @ (grid - reference_mean) ** 2))
        mc_se = reference_sd / np.sqrt(draws.ess)
        assert abs(samples[:, column].mean() - reference_mean) <= 4.0 * mc_se, (
            column, samples[:, column].mean(), reference_mean, mc_se
        )
        cdf = np.cumsum(marginal)
        for level in (0.1, 0.5, 0.9):
            reference_quantile = grid[np.searchsorted(cdf, level)]
            below = float(np.mean(samples[:, column] <= reference_quantile))
            quantile_se = np.sqrt(level * (1.0 - level) / draws.ess)
            assert abs(below - level) <= 4.0 * quantile_se, (
                column, level, below, reference_quantile
            )
