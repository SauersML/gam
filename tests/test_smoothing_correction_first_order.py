"""The smoothing-parameter-uncertainty correction is analytic and budget-free.

``Vp = Vb + J V_rho J^T`` (Wood, Pya & Saefken 2016), with ``J = d beta_hat /
d rho`` from the implicit function theorem and ``V_rho`` the certified inverse
outer Hessian on its identified subspace, is the published corrected
covariance for any number of smoothing parameters. No sampling budget, no
dimension gate and no cubature decides whether a fit gets it.

A small repeated-data study pins the coverage the correction exists for:
95% intervals for the mean and for partial dependence must cover at their
nominal rate for Gaussian, Poisson and binomial responses.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import gamfit

TWO_PI = 2.0 * np.pi
Z95 = 1.959963984540054


def _saved_payload(model, tmp_path):
    path = tmp_path / "model.gam"
    model.save(path)
    return json.loads(path.read_text())["model"]["fit_result"]


@pytest.mark.parametrize("family", ["binomial", "poisson", "gaussian"])
def test_six_smooth_glm_publishes_the_first_order_correction(family, tmp_path) -> None:
    rng = np.random.default_rng(20260919)
    n = 1500
    X = rng.uniform(0.0, 1.0, (n, 6))
    eta = (
        np.sin(TWO_PI * X[:, 0])
        + 0.5 * np.cos(TWO_PI * X[:, 1])
        + 3.0 * (X[:, 2] - 0.5) ** 2
        + 0.3 * np.sin(2.0 * TWO_PI * X[:, 3])
        + 0.4 * X[:, 5]
    )
    if family == "binomial":
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    elif family == "poisson":
        y = rng.poisson(np.exp(0.3 + 0.5 * eta)).astype(float)
    else:
        y = eta + rng.normal(0.0, 1.0, n)
    data = {f"x{j + 1}": X[:, j] for j in range(6)}
    data["y"] = y
    formula = "y ~ " + " + ".join(f"s(x{j + 1})" for j in range(6))
    model = gamfit.fit(data, formula, family=family)

    fit_result = _saved_payload(model, tmp_path)
    inference = fit_result["inference"]
    method = inference["smoothing_correction_method"]
    assert set(method) == {"FirstOrderIdentifiedSubspace"}, method
    geometry = method["FirstOrderIdentifiedSubspace"]
    # Every smoothing parameter is in the correction: each smooth carries a
    # wiggliness and a null-space penalty, so six smooths are twelve rho.
    n_rho = len(fit_result["lambdas"]["data"])
    assert n_rho == 12, n_rho
    assert geometry["rho_dimension"] == n_rho, geometry
    assert 1 <= geometry["active_rank"] <= n_rho, geometry
    assert inference["smoothing_correction"] is not None
    assert inference["smoothing_correction_method_first_order"] == method

    grid = {k: v[:40] for k, v in data.items() if k != "y"}
    corrected = model.predict(grid, interval=0.95, return_type="dict")
    conditional = model.predict(
        grid, interval=0.95, covariance_mode="conditional", return_type="dict"
    )
    assert corrected.covariance_source == "smoothing-corrected"
    se_corr = np.asarray(corrected.posterior_mean_standard_error, dtype=float)
    se_cond = np.asarray(conditional.posterior_mean_standard_error, dtype=float)
    assert np.all(np.isfinite(se_corr))
    # J V_rho J^T is a Gram: the correction only adds variance.
    assert np.all(se_corr >= se_cond * (1.0 - 1e-9))


# ---------------------------------------------------------------------------
# Repeated-data coverage
# ---------------------------------------------------------------------------

N_REP = 40
N_TEST = 60
GRID = np.linspace(0.02, 0.98, 25)
NOMINAL = 0.95
# Across-the-function coverage of Bayesian GAM intervals is nominal on average
# over replicate data sets (Nychka 1988; Marra & Wood 2012), so the gate is on
# the replicate mean against its own Monte-Carlo SE. At exactly nominal
# coverage a |z| > 4 gate fails with probability 2(1 - Phi(4)) = 6.3e-5 per
# assertion.
Z_GATE = 4.0
CELLS = {
    # family: (n, intercept, a1, a3, sigma)
    "gaussian": (200, 0.0, 1.0, 0.30, 1.0),
    "poisson": (200, 0.5, 0.8, 0.25, None),
    "binomial": (400, 0.0, 1.5, 0.60, None),
}


def _inv_link(family, eta):
    if family == "gaussian":
        return eta
    if family == "binomial":
        return 1.0 / (1.0 + np.exp(-eta))
    return np.exp(eta)


def _draw(family, mu, sigma, rng):
    if family == "gaussian":
        return mu + rng.normal(0.0, sigma, mu.shape)
    if family == "binomial":
        return (rng.uniform(size=mu.shape) < mu).astype(float)
    return rng.poisson(mu).astype(float)


def _replicate_mean_and_mcse(per_rep):
    hits = np.asarray(per_rep, dtype=float)
    return float(hits.mean()), float(hits.std(ddof=1) / np.sqrt(hits.size))


def _assert_nominal_coverage(label, per_rep):
    """Two-sided: over-coverage is a calibration failure too.

    The smoothing correction ``J V_rho J^T`` is a Gram, so it can only add
    variance. Its one way to be wrong is to add too much, which gives bands
    that over-cover. A lower bar alone cannot see that.
    """
    cover, mcse = _replicate_mean_and_mcse(per_rep)
    assert abs(cover - NOMINAL) <= Z_GATE * mcse, (
        f"{label}: coverage {cover:.4f} is outside {NOMINAL} +/- {Z_GATE}*MCSE "
        f"(MCSE {mcse:.4f} over {len(per_rep)} replicates)"
    )


@pytest.mark.parametrize("family", ["gaussian", "poisson", "binomial"])
def test_corrected_intervals_cover_the_mean_and_partial_dependence(family) -> None:
    n, b0, a1, a3, sigma = CELLS[family]
    Xt = np.random.default_rng(12345).uniform(0.02, 0.98, (N_TEST, 3))
    mu_t = _inv_link(family, b0 + a1 * np.sin(TWO_PI * Xt[:, 0]) + a3 * np.cos(TWO_PI * Xt[:, 2]))
    test = {"x1": Xt[:, 0], "x2": Xt[:, 1], "x3": Xt[:, 2]}
    mean_hits = []
    pd_hits = {"x1": [], "x2": [], "x3": []}
    for rep in range(N_REP):
        rng = np.random.default_rng(1000 + rep)
        X = rng.uniform(0.0, 1.0, (n, 3))
        f1, f3 = a1 * np.sin(TWO_PI * X[:, 0]), a3 * np.cos(TWO_PI * X[:, 2])
        y = _draw(family, _inv_link(family, b0 + f1 + f3), sigma, rng)
        data = {"x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "y": y}
        model = gamfit.fit(data, "y ~ s(x1) + s(x2) + s(x3)", family=family)
        pred = model.predict(test, interval=0.95)
        assert pred["covariance_source"] == "smoothing-corrected"
        lo = np.asarray(pred["posterior_mean_lower"], dtype=float)
        hi = np.asarray(pred["posterior_mean_upper"], dtype=float)
        mean_hits.append(np.mean((mu_t >= lo) & (mu_t <= hi)))
        truth_pd = {
            "x1": a1 * np.sin(TWO_PI * GRID) - f1.mean(),
            "x2": np.zeros_like(GRID),
            "x3": a3 * np.cos(TWO_PI * GRID) - f3.mean(),
        }
        for v, truth in truth_pd.items():
            pd = model.partial_dependence(f"s({v})", grid=GRID)
            est = np.asarray(pd["predicted"], dtype=float)
            se = np.asarray(pd["standard_error"], dtype=float)
            pd_hits[v].append(np.mean(np.abs(truth - est) <= Z95 * se))
    _assert_nominal_coverage(f"{family} mean", mean_hits)
    # The two terms that carry signal are gated separately, so a shortfall in
    # one cannot be averaged away against an excess in the other.
    _assert_nominal_coverage(f"{family} s(x1)", pd_hits["x1"])
    _assert_nominal_coverage(f"{family} s(x3)", pd_hits["x3"])
    # s(x2) has truth identically zero, which lies in the double penalty's
    # infinite-lambda limit. REML shrinks it there, where the band contains
    # zero at every grid point by construction, so its coverage is 1 whenever
    # the selection is right. Only a band that excludes the zero truth is a
    # failure for this term.
    null_cover, null_mcse = _replicate_mean_and_mcse(pd_hits["x2"])
    assert null_cover >= NOMINAL - Z_GATE * null_mcse, (
        f"{family} s(x2): band covers the zero truth at {null_cover:.4f} "
        f"(MCSE {null_mcse:.4f})"
    )
