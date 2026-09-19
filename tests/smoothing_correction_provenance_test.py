"""pyGAM audit lane ``smoothing-correction-provenance``.

Every fold below is one the audit saw fall back from the smoothing-corrected
covariance, or raise ``smoothing cubature could not calibrate its nodes``, in
the released wheel. Each must fit with the full sigma-point cubature upgrade:
a cubature that cannot integrate its nodes is a typed ``IntegrationError``, never
a downgraded covariance. The two folds whose criterion latches the #784
block-local correction have no analytic outer rho-Hessian (#3139) and must say
so. ``summary().convergence`` publishes the Tier-0 PSIS ``k̂`` of the
rho-posterior Laplace proposal, or the typed reason it could not be formed.

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


# The criterion latches the #784 block-local correction, whose rho-Hessian is
# not implemented (#3139): the published covariance is the conditional one and
# the reason says why.
BLOCK_CORRECTION_FOLDS = {"haberman_k20_fold0", "nearsep_n200_fold0"}
# The certified rho covariance identifies no direction, so the first-order
# correction is exact (and zero): there is nothing for a cubature to integrate.
NO_IDENTIFIED_DIRECTION = {"hepatitis_monotone_decreasing"}
CUBATURE_FOLDS = sorted(set(FIXTURES) - BLOCK_CORRECTION_FOLDS - NO_IDENTIFIED_DIRECTION)
# Folds whose Laplace Gaussian puts under 1/M of its mass inside the rho domain
# (an identified but nearly flat direction): the truncated proposal cannot
# supply M draws within M^2 and says so (#3010).
PROPOSAL_OUTSIDE_SUPPORT = {
    "mc_pois_rep0",
    "cake_fold0",
    "cake_fold3",
    "haberman_k20_fold2",
    "haberman_k20_fold4",
}
OUTSIDE_SUPPORT_REASON = "of its mass inside the rho domain"


@pytest.mark.parametrize("name", CUBATURE_FOLDS)
def test_fold_fits_with_the_sigma_point_cubature(name):
    summary, conv = fit_summary(name)
    assert conv["covariance_source"] == "smoothing-corrected", conv
    assert summary.coefficient_se_source == "smoothing-corrected"
    assert conv["smoothing_correction_method"] == "sigma_point_cubature", conv
    assert conv["smoothing_correction_fallback"] is None, conv
    assert conv["covariance_source_reason"] is None, conv


@pytest.mark.parametrize("name", sorted(BLOCK_CORRECTION_FOLDS))
def test_block_correction_fold_names_its_missing_hessian(name):
    summary, conv = fit_summary(name)
    assert conv["covariance_source"] == "conditional", conv
    assert summary.coefficient_se_source == "conditional"
    assert conv["smoothing_correction_method"] is None
    reason = conv["covariance_source_reason"]
    assert nonempty(reason) and "no analytic" in reason and "#784" in reason, conv
    assert conv["rho_posterior_status"] == "not_computed", conv
    assert "#784" in conv["rho_posterior_reason"], conv


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_rho_posterior_khat_is_published(name):
    _, conv = fit_summary(name)
    status = conv["rho_posterior_status"]
    # The default fit never runs a heavier tier on its own (no dimension gate).
    assert conv["rho_posterior_escalation"] is None, conv
    if name in BLOCK_CORRECTION_FOLDS or name in NO_IDENTIFIED_DIRECTION:
        assert status == "not_computed", conv
        assert conv["rho_posterior_khat"] is None
        assert nonempty(conv["rho_posterior_reason"])
        return
    if name in PROPOSAL_OUTSIDE_SUPPORT and status == "refused":
        assert conv["rho_posterior_khat"] is None
        assert OUTSIDE_SUPPORT_REASON in conv["rho_posterior_reason"], conv
        return
    assert status == "assessed", conv
    khat = conv["rho_posterior_khat"]
    assert isinstance(khat, float) and math.isfinite(khat)
    expected = (
        "plug_in_adequate" if khat < 0.5 else "importance_correct" if khat < 0.7 else "escalate"
    )
    assert conv["rho_posterior_adequacy"] == expected, conv
    assert conv["rho_posterior_samples"] > 0
    ess = conv["rho_posterior_effective_sample_size"]
    assert 0.0 < ess <= conv["rho_posterior_samples"]
    assert conv["rho_posterior_reason"] is None


def test_monotone_decreasing_names_its_source_like_the_other_shapes():
    """F16: the shape-constrained fit's source is smoothing-corrected and says why
    its correction is the first-order one, while convex and concave integrate."""
    summary, conv = fit_summary("hepatitis_monotone_decreasing")
    assert summary.coefficient_se_source == conv["covariance_source"] == "smoothing-corrected"
    assert conv["smoothing_correction_method"] == "first_order_identified_subspace", conv
    fallback = conv["smoothing_correction_fallback"]
    assert fallback is not None and "no identified rho direction" in fallback["reason"]
    assert set(fallback) == {"reason"}
    assert fallback["reason"] in conv["covariance_source_reason"]
    for other in ("convex", "concave"):
        other_summary, other_conv = fit_summary(f"hepatitis_{other}")
        assert other_summary.coefficient_se_source == other_conv["covariance_source"]
        assert other_conv["smoothing_correction_method"] == "sigma_point_cubature", other_conv
