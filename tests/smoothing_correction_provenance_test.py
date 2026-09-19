"""pyGAM audit lane ``smoothing-correction-provenance``.

Every fold below is one the audit saw fall back from the smoothing-corrected
covariance, or raise ``smoothing cubature could not calibrate its nodes`` in
the released wheel. Each must fit, and ``summary().convergence`` must say which
covariance the standard errors came from and why it is not the full cubature
upgrade whenever it is not. The Tier-0 PSIS ``k̂`` of the rho-posterior Laplace
proposal is published, and a ``K <= 4`` fit it grades ``escalate`` carries its
Tier-1 Gauss-Hermite record.

Folds are ``sklearn.model_selection.KFold(5, shuffle=True, random_state=0)``,
reproduced exactly so the test does not depend on scikit-learn: the test folds
are consecutive runs of ``RandomState(0).permutation(n)`` (the first ``n % 5``
one longer) and the training set is the sorted complement.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest

import gamfit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data", "pygam_smoothing_provenance")
OUTER_CERTIFY = os.path.join(HERE, "data", "pygam_outer_certify")

SOURCES = {"conditional", "smoothing-corrected"}
SEVERITIES = {"routine", "numerical_failure"}
STATUSES = {"assessed", "refused", "not_computed", "not_applicable"}
QUADRATURE_MAX_DIM = 4


def kfold_train(n, fold):
    perm = np.random.RandomState(0).permutation(n)
    sizes = np.full(5, n // 5)
    sizes[: n % 5] += 1
    start = int(sizes[:fold].sum())
    test = perm[start : start + sizes[fold]]
    return np.setdiff1d(np.arange(n), test)


def mc_pois():
    rng = np.random.default_rng(1000)
    x = rng.uniform(0, 1, (200, 3))
    eta = 0.5 + 0.8 * np.sin(2 * np.pi * x[:, 0]) + 0.25 * np.cos(2 * np.pi * x[:, 2])
    y = rng.poisson(np.exp(eta)).astype(float)
    return dict(x1=x[:, 0], x2=x[:, 1], x3=x[:, 2], y=y)


def pois_add2(n, fold):
    rng = np.random.default_rng(n + 8)
    x = rng.uniform(0, 1, (n, 2))
    eta = 1 + np.sin(2 * np.pi * 1.5 * x[:, 0]) + 0.8 * np.cos(2 * np.pi * x[:, 1])
    y = rng.poisson(np.exp(eta)).astype(float)
    tr = kfold_train(n, fold)
    return dict(x0=x[tr, 0], x1=x[tr, 1], y=y[tr])


def csv_fold(path, fold=None):
    frame = pd.read_csv(path).astype(float)
    if fold is not None:
        frame = frame.iloc[kfold_train(len(frame), fold)]
    return {column: frame[column].to_numpy() for column in frame.columns}


def city_temp(fold):
    frame = pd.read_csv(os.path.join(REPO, "bench", "datasets", "global_major_city_temp.csv"))
    tr = kfold_train(len(frame), fold)
    return dict(
        lat=frame["lat"].to_numpy(float)[tr],
        lon=frame["lon"].to_numpy(float)[tr],
        y=frame["temp"].to_numpy(float)[tr],
    )


def hepatitis():
    frame = pd.read_csv(os.path.join(DATA, "hepatitis_A_bulgaria.csv")).astype(float)
    frame = frame[frame.total > 0]
    return dict(x=frame.age.to_numpy(), y=(frame.hepatitis_A_positive / frame.total).to_numpy())


WAGE = "y ~ s(year) + s(age) + factor(edu)"
CAKE = "y ~ factor(recipe) + factor(replicate) + s(temp)"
CITY = "y ~ s(lat, k=30) + s(lon, k=30)"
HABERMAN = "y ~ s(age, k=20) + s(year, k=20) + s(nodes, k=20)"

FIXTURES = {
    "mc_pois_rep0": (mc_pois, "y ~ s(x1) + s(x2) + s(x3)", "poisson"),
    "wage_fold2": (lambda: csv_fold(os.path.join(DATA, "wage.csv"), 2), WAGE, "gaussian"),
    "wage_fold3": (lambda: csv_fold(os.path.join(DATA, "wage.csv"), 3), WAGE, "gaussian"),
    "cake_fold0": (lambda: csv_fold(os.path.join(DATA, "cake.csv"), 0), CAKE, "gaussian"),
    "cake_fold3": (lambda: csv_fold(os.path.join(DATA, "cake.csv"), 3), CAKE, "gaussian"),
    "pois_add2_n300_fold3": (lambda: pois_add2(300, 3), "y ~ s(x0) + s(x1)", "poisson"),
    "pois_add2_n2000_fold4": (lambda: pois_add2(2000, 4), "y ~ s(x0) + s(x1)", "poisson"),
    **{
        f"city_temp_k30_fold{fold}": ((lambda fold=fold: city_temp(fold)), CITY, "gaussian")
        for fold in range(5)
    },
    "haberman_k20_fold0": (
        lambda: csv_fold(os.path.join(DATA, "haberman_fold0.csv")),
        HABERMAN,
        "binomial",
    ),
    "haberman_k20_fold2": (
        lambda: csv_fold(os.path.join(OUTER_CERTIFY, "haberman_fold2.csv")),
        HABERMAN,
        "binomial",
    ),
    "haberman_k20_fold4": (
        lambda: csv_fold(os.path.join(OUTER_CERTIFY, "haberman_fold4.csv")),
        HABERMAN,
        "binomial",
    ),
    "nearsep_n200_fold0": (
        lambda: csv_fold(os.path.join(OUTER_CERTIFY, "nearsep_n200_fold0.csv")),
        "y ~ s(x) + s(z)",
        "binomial",
    ),
    **{
        f"hepatitis_{shape}": (hepatitis, f"y ~ s(x, shape={shape})", "gaussian")
        for shape in ("monotone_decreasing", "convex", "concave")
    },
}


def fit_summary(name):
    make, formula, family = FIXTURES[name]
    model = gamfit.fit(make(), formula, family=family)
    summary = model.summary()
    return summary, summary.convergence


def nonempty(text):
    return isinstance(text, str) and text.strip() != ""


def finite_matrix(rows):
    return all(math.isfinite(value) for row in rows for value in row)


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_covariance_source_names_itself_and_its_reason(name):
    summary, conv = fit_summary(name)
    source = conv["covariance_source"]
    assert source in SOURCES
    assert source == summary.coefficient_se_source
    fallback = conv["smoothing_correction_fallback"]
    if source == "smoothing-corrected":
        assert conv["smoothing_correction_method"] in {
            "sigma_point_cubature",
            "first_order_identified_subspace",
        }
    else:
        assert conv["smoothing_correction_method"] is None
    if conv["smoothing_correction_method"] == "first_order_identified_subspace":
        assert fallback is not None, conv
    if fallback is not None:
        assert nonempty(fallback["reason"])
        assert fallback["severity"] in SEVERITIES
    if source != "smoothing-corrected" or fallback is not None:
        assert nonempty(conv["covariance_source_reason"]), conv
    if conv["smoothing_correction_method"] == "sigma_point_cubature":
        assert conv["covariance_source_reason"] is None


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_rho_posterior_khat_is_published(name):
    summary, conv = fit_summary(name)
    status = conv["rho_posterior_status"]
    assert status in STATUSES
    assert status == "assessed", conv
    khat = conv["rho_posterior_khat"]
    assert isinstance(khat, float) and math.isfinite(khat)
    assert conv["rho_posterior_adequacy"] in {
        "plug_in_adequate",
        "importance_correct",
        "escalate",
    }
    assert conv["rho_posterior_samples"] > 0
    assert conv["rho_posterior_reason"] is None
    n_rho = len(summary.lambdas)
    escalation = conv["rho_posterior_escalation"]
    if khat > 0.7:
        assert conv["rho_posterior_adequacy"] == "escalate"
    if conv["rho_posterior_adequacy"] == "escalate" and n_rho <= QUADRATURE_MAX_DIM:
        assert escalation is not None and escalation["tier"] == "quadrature", conv
        assert escalation["n_nodes"] > 0
        assert all(math.isfinite(value) for value in escalation["mean"])
        assert finite_matrix(escalation["covariance"])
        assert math.isfinite(escalation["effective_sample_size"])
    elif conv["rho_posterior_adequacy"] != "escalate":
        assert escalation is None


def test_monotone_decreasing_names_its_source_like_the_other_shapes():
    reports = {}
    for shape in ("monotone_decreasing", "convex", "concave"):
        summary, conv = fit_summary(f"hepatitis_{shape}")
        reports[shape] = (summary.coefficient_se_source, conv)
    source, conv = reports["monotone_decreasing"]
    assert source == conv["covariance_source"]
    # F16: the shape-constrained fit must not publish a bare "conditional".
    if source == "conditional":
        assert nonempty(conv["covariance_source_reason"])
    else:
        assert source == "smoothing-corrected"
        assert conv["smoothing_correction_method"] is not None
        if conv["smoothing_correction_method"] != "sigma_point_cubature":
            assert nonempty(conv["covariance_source_reason"])
    for other in ("convex", "concave"):
        other_source, other_conv = reports[other]
        assert other_source == other_conv["covariance_source"]
