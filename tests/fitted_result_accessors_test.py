"""Fitted-result accessors are Rust summary fields, checked against closed forms.

pyGAM / mgcv users read the fitted coefficients, EDF, smoothing parameters,
dispersion ``phi_hat``, log-likelihood, deviance, residuals, iteration counts
and ``n_obs`` straight off the fitted model. ``gamfit`` computes every one of
them in Rust: the scalars are fields of the one ``SummaryPayload`` that
``Model.summary()`` and ``gam summary`` both serialize, and residuals are
``Model.residuals(data, type=...)`` / ``gam residuals`` over the same Rust
kernel. Before this change the dispersion was never surfaced (the loglik used
it internally), there were no residuals, and no inner-iteration count.

Every accessor is checked here against a closed form evaluated in the test:

* Gaussian identity with a fixed (empty) smoothing vector is ordinary least
  squares, so ``coefficients`` solve the normal equations, ``edf_total = p``,
  every residual type is ``y - X beta``, and ``scale = RSS / (n - p)``.
* A penalized, prior-weighted Gaussian smooth at its reported ``lambda`` is a
  linear smoother ``mu = F y`` with ``edf = tr F``. Then
  ``E||y - mu||_w^2 = sigma^2 (n - 2 tr F + tr F'F) ~ sigma^2 (n - tr F)``,
  the residual-df estimator mgcv's ``gam.scale`` uses, where ``n`` counts only the
  rows with positive prior weight (a zero-weight row carries no information).
* Poisson and Gamma log-link fits check each residual type against its textbook
  formula, ``sum(deviance residuals**2) == deviance``, and the log-likelihood
  against the closed-form density at the reported ``scale``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

pytest = cast(Any, import_module("pytest"))
gamfit = cast(Any, import_module("gamfit"))
special = cast(Any, import_module("scipy.special"))

REPO = Path(__file__).resolve().parent.parent
GAM_BIN = Path(os.environ.get("GAM_BIN", REPO / "target" / "release" / "gam"))

RTOL = 1e-8


def _linear_predictor(model: Any, data: dict[str, Any]) -> np.ndarray:
    design = model.design_matrix(data)
    return np.asarray(design.offset) + np.asarray(design.matrix) @ model.coefficients


def _gaussian_linear_data(n: int = 80) -> dict[str, list[float]]:
    rng = np.random.default_rng(11)
    x = rng.uniform(-2.0, 2.0, n)
    y = 1.5 - 0.7 * x + rng.normal(0.0, 0.4, n)
    return {"x": x.tolist(), "y": y.tolist()}


def test_gaussian_ols_accessors_match_normal_equations() -> None:
    data = _gaussian_linear_data()
    y = np.asarray(data["y"])
    n = y.size
    model = gamfit.fit(data, "y ~ x", family="gaussian")

    x_design = np.asarray(model.design_matrix(data).matrix)
    p = x_design.shape[1]
    beta_ols, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    np.testing.assert_allclose(model.coefficients, beta_ols, rtol=RTOL, atol=1e-10)

    # No smoothing coordinate: the fit is at a fixed (empty) lambda.
    assert model.smoothing_parameters() == {}
    assert model.n_obs == n
    np.testing.assert_allclose(model.edf_total, p, rtol=RTOL)

    resid = y - x_design @ beta_ols
    rss = float(resid @ resid)
    sigma2 = rss / (n - p)
    np.testing.assert_allclose(model.scale, sigma2, rtol=RTOL)
    np.testing.assert_allclose(model.deviance, rss, rtol=RTOL)
    loglik = -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * rss / sigma2
    np.testing.assert_allclose(model.log_likelihood, loglik, rtol=RTOL)

    # Identity link, unit variance function: every residual type is y - mu.
    for kind in ("response", "working", "deviance", "pearson"):
        np.testing.assert_allclose(
            model.residuals(data, type=kind), resid, rtol=0, atol=1e-9, err_msg=kind
        )

    convergence = model.convergence
    assert convergence is not None and convergence["certified"]
    assert model.outer_iterations == convergence["outer_iterations"]
    assert model.inner_iterations == convergence["inner_iterations"]
    assert model.inner_iterations >= 1


def test_weighted_gaussian_smooth_scale_is_rss_over_residual_df() -> None:
    rng = np.random.default_rng(5)
    n = 160
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, n)
    w = rng.uniform(0.5, 2.0, n)
    w[::10] = 0.0  # zero prior weight: excluded from the residual df
    data = {"x": x.tolist(), "y": y.tolist(), "w": w.tolist()}
    model = gamfit.fit(data, "y ~ s(x)", family="gaussian", weights="w")

    mu = _linear_predictor(model, data)
    response = model.residuals(data, type="response")
    np.testing.assert_allclose(response, y - mu, rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        model.residuals(data, type="pearson"), np.sqrt(w) * (y - mu), rtol=0, atol=1e-9
    )
    deviance_resid = model.residuals(data, type="deviance")
    np.testing.assert_allclose(deviance_resid, np.sqrt(w) * (y - mu), rtol=0, atol=1e-9)

    rss_w = float(np.sum(w * (y - mu) ** 2))
    np.testing.assert_allclose(model.deviance, rss_w, rtol=1e-7)
    np.testing.assert_allclose(float(deviance_resid @ deviance_resid), model.deviance, rtol=1e-7)

    edf = model.edf_total
    smooth_edf = model.smooth_edf
    assert set(smooth_edf) == {"s(x)"}
    # One unpenalized intercept plus the smooth's own trace.
    np.testing.assert_allclose(edf, 1.0 + smooth_edf["s(x)"], rtol=1e-7)
    assert 2.0 < edf < 12.0

    n_informative = int(np.count_nonzero(w > 0.0))
    np.testing.assert_allclose(model.scale, rss_w / (n_informative - edf), rtol=1e-7)

    lambdas = model.smoothing_parameters()
    assert len(lambdas) >= 1 and all(value > 0.0 for value in lambdas.values())
    np.testing.assert_allclose(list(lambdas.values()), model.summary().lambdas, rtol=0)


def test_poisson_residuals_and_loglik_closed_forms() -> None:
    rng = np.random.default_rng(7)
    n = 120
    x = rng.uniform(-1.0, 1.0, n)
    y = rng.poisson(np.exp(0.4 + 0.9 * x)).astype(float)
    data = {"x": x.tolist(), "y": y.tolist()}
    model = gamfit.fit(data, "y ~ x", family="poisson")

    x_design = np.asarray(model.design_matrix(data).matrix)
    mu = np.exp(_linear_predictor(model, data))
    # Unpenalized Poisson MLE: the score equations X'(y - mu) = 0.
    np.testing.assert_allclose(x_design.T @ (y - mu), 0.0, atol=1e-6)

    unit_dev = 2.0 * (special.xlogy(y, y / mu) - (y - mu))
    np.testing.assert_allclose(model.residuals(data, type="response"), y - mu, atol=1e-9)
    np.testing.assert_allclose(model.residuals(data, type="working"), (y - mu) / mu, atol=1e-9)
    np.testing.assert_allclose(
        model.residuals(data, type="pearson"), (y - mu) / np.sqrt(mu), atol=1e-9
    )
    deviance_resid = model.residuals(data, type="deviance")
    np.testing.assert_allclose(
        deviance_resid, np.sign(y - mu) * np.sqrt(unit_dev), atol=1e-9
    )
    np.testing.assert_allclose(model.deviance, unit_dev.sum(), rtol=1e-8)
    np.testing.assert_allclose(float(deviance_resid @ deviance_resid), model.deviance, rtol=1e-8)

    assert model.scale == 1.0
    loglik = float(np.sum(y * np.log(mu) - mu - special.gammaln(y + 1.0)))
    np.testing.assert_allclose(model.log_likelihood, loglik, rtol=1e-8)
    np.testing.assert_allclose(model.edf_total, x_design.shape[1], rtol=1e-8)
    assert model.n_obs == n
    # A non-Gaussian fit needs more than one P-IRLS step.
    assert model.inner_iterations > 1


def test_gamma_scale_is_the_dispersion_the_loglik_uses() -> None:
    rng = np.random.default_rng(13)
    n = 150
    x = rng.uniform(-1.0, 1.0, n)
    mean = np.exp(1.0 + 0.5 * x)
    shape = 4.0
    y = rng.gamma(shape, mean / shape)
    data = {"x": x.tolist(), "y": y.tolist()}
    model = gamfit.fit(data, "y ~ x", family="gamma")

    mu = np.exp(_linear_predictor(model, data))
    np.testing.assert_allclose(model.residuals(data, type="response"), y - mu, atol=1e-9)
    np.testing.assert_allclose(model.residuals(data, type="working"), (y - mu) / mu, atol=1e-9)
    np.testing.assert_allclose(model.residuals(data, type="pearson"), (y - mu) / mu, atol=1e-9)
    unit_dev = 2.0 * (-np.log(y / mu) + (y - mu) / mu)
    deviance_resid = model.residuals(data, type="deviance")
    np.testing.assert_allclose(deviance_resid, np.sign(y - mu) * np.sqrt(unit_dev), atol=1e-9)
    np.testing.assert_allclose(model.deviance, unit_dev.sum(), rtol=1e-8)

    phi = model.scale
    assert phi is not None and 0.0 < phi < 1.0
    k = 1.0 / phi
    loglik = float(
        np.sum(k * np.log(k * y / mu) - k * y / mu - np.log(y) - special.gammaln(k))
    )
    np.testing.assert_allclose(model.log_likelihood, loglik, rtol=1e-8)


def test_residual_type_is_validated_in_rust() -> None:
    data = _gaussian_linear_data()
    model = gamfit.fit(data, "y ~ x", family="gaussian")
    with pytest.raises(ValueError, match="unknown residual type 'raw'"):
        model.residuals(data, type="raw")


def test_sklearn_fitted_attributes_are_the_model_accessors() -> None:
    sklearn = import_module("gamfit.sklearn")
    data = _gaussian_linear_data()
    features = {"x": data["x"]}
    reg = sklearn.GAMRegressor(formula="y ~ x").fit(features, np.asarray(data["y"]))
    np.testing.assert_array_equal(reg.coef_, reg.model_.coefficients)
    assert reg.edf_ == reg.model_.edf_total
    assert reg.n_iter_ == reg.model_.outer_iterations


def test_cli_summary_and_residuals_match_python(tmp_path: Path) -> None:
    assert GAM_BIN.exists(), (
        f"the `gam` CLI binary is required at {GAM_BIN} "
        "(build it with `cargo build --release -p gam-cli`, or set GAM_BIN)"
    )
    rng = np.random.default_rng(3)
    n = 90
    x = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.5 + np.cos(3.0 * x))).astype(float)
    data = {"x": x.tolist(), "y": y.tolist()}
    model = gamfit.fit(data, "y ~ s(x)", family="poisson")
    model_path = tmp_path / "model.gam"
    model.save(model_path)
    data_path = tmp_path / "data.csv"
    with data_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        writer.writerows(zip(data["x"], data["y"]))

    def run(*args: str) -> Any:
        proc = subprocess.run(
            [str(GAM_BIN), *args], capture_output=True, text=True, timeout=600
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    cli_summary = run("summary", str(model_path))
    summary = model.summary()
    for key in ("n_obs", "scale", "edf_total", "log_likelihood", "deviance", "lambdas"):
        assert cli_summary[key] == summary[key], key
    assert cli_summary["convergence"] == summary.convergence
    assert [c["estimate"] for c in cli_summary["coefficients"]] == model.coefficients.tolist()

    for kind in ("response", "working", "deviance", "pearson"):
        cli = run("residuals", str(model_path), str(data_path), "--type", kind)
        assert cli["type"] == kind
        # CSV round-trips the inputs exactly (shortest repr), so the values
        # agree to the last bit.
        assert cli["residuals"] == model.residuals(data, type=kind).tolist(), kind
