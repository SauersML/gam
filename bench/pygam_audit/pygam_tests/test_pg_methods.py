"""Translation of pygam/tests/test_GAM_methods.py (statistically meaningful, SPEC-compatible parts).

Each test names the pyGAM test it mirrors.
"""
import numpy as np
import pytest
from scipy import stats

import gamfit
from bench.pygam_audit.pygam_tests.pg_helpers import eta_of


def _summary_row(m, name_part):
    s = m.summary()
    for row in s.smooth_terms:
        rd = row if isinstance(row, dict) else getattr(row, "__dict__", {})
        nm = rd.get("name") or rd.get("term") or str(row)
        if name_part in str(nm):
            return rd
    raise AssertionError(f"no smooth row containing {name_part!r}: {s.smooth_terms!r}")


def _pvalue(row):
    for k in ("p_value", "pvalue", "p"):
        if k in row:
            return float(row[k])
    raise AssertionError(f"no p-value key in {row}")


def _edf(row):
    for k in ("edf", "effective_df"):
        if k in row:
            return float(row[k])
    raise AssertionError(f"no edf key in {row}")


# test_LinearGAM_prediction / test_LogisticGAM_accuracy / test_PoissonGAM_exposure ...
def test_prediction_shape(mcycle_model, mcycle):
    p = mcycle_model.predict(mcycle)
    assert np.shape(p) == (len(mcycle["y"]),)
    assert np.all(np.isfinite(p))


def test_classifier_accuracy_score_matches_definition(default):
    from gamfit.sklearn import GAMClassifier
    X = np.column_stack([default["balance"], default["income"]])
    y = default["y"].astype(int)
    clf = GAMClassifier(formula="y ~ s(x0) + s(x1)").fit(X, y)
    acc = clf.score(X, y)
    assert acc == pytest.approx(np.mean(clf.predict(X) == y))
    assert 0.9 < acc <= 1.0


# test_PoissonGAM_exposure: predicting with exposure doubles the rate
def test_poisson_offset_scales_prediction(coal):
    d = dict(coal)
    d["off"] = np.zeros_like(d["x"])
    m = gamfit.fit(d, "y ~ s(x)", family="poisson", offset="off")
    base = m.predict(d)
    d2 = dict(d)
    d2["off"] = np.full_like(d["x"], np.log(2.0))
    np.testing.assert_allclose(m.predict(d2), 2.0 * base, rtol=1e-8)


# test_PoissonGAM_loglike: non-integer exposure still gives a finite log likelihood
def test_poisson_non_integer_offset_finite_loglik(coal):
    rng = np.random.default_rng(0)
    d = dict(coal)
    d["off"] = np.log(rng.uniform(0.5, 2.5, len(d["x"])))
    m = gamfit.fit(d, "y ~ s(x)", family="poisson", offset="off")
    assert np.isfinite(m.summary().log_likelihood)


# test_large_GAM: n=100k fits and converges
def test_large_n_converges():
    rng = np.random.default_rng(1)
    n = 100_000
    x = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + rng.normal(0, 0.3, n)
    m = gamfit.fit({"x": x, "y": y}, "y ~ s(x)")
    s = m.summary()
    assert s.convergence["certified"] and s.convergence["inner_status"] == "Converged", s.convergence


# test_summary / test_summary_returns_12_lines-ish: one row per smooth term
def test_summary_has_one_row_per_smooth(wage):
    m = gamfit.fit(wage, "y ~ s(year) + s(age) + factor(edu)")
    rows = m.summary().smooth_terms
    names = " ".join(str(r) for r in rows)
    assert "year" in names and "age" in names


# test_more_splines_than_samples: k = n+1 must still fit, with edf < n
def test_more_basis_than_samples(mcycle):
    n = len(mcycle["y"])
    m = gamfit.fit(mcycle, f"y ~ s(x, k={n + 1})")
    assert m.summary().edf_total < n
    assert np.all(np.isfinite(m.predict(mcycle)))


# test_prediction_interval_known_scale / test_conf_intervals_*: ordering & nesting
def test_interval_ordering_and_nesting(mcycle_model, mcycle):
    r95 = mcycle_model.predict(mcycle, interval=0.95, observation_interval=True)
    r90 = mcycle_model.predict(mcycle, interval=0.90, observation_interval=True)
    lo, mu, hi = (np.asarray(r95[k]) for k in ("posterior_mean_lower", "posterior_mean", "posterior_mean_upper"))
    assert np.all(lo <= mu) and np.all(mu <= hi)
    assert np.all(np.asarray(r90["posterior_mean_upper"]) - np.asarray(r90["posterior_mean_lower"]) <= hi - lo + 1e-12)
    olo, ohi = np.asarray(r95["observation_lower"]), np.asarray(r95["observation_upper"])
    assert np.all(olo <= lo + 1e-12) and np.all(ohi >= hi - 1e-12)


# test_prediction_interval_unknown_scale: y ~ N(0,1), 80% observation interval ~ +-1.2816
def test_observation_interval_calibration():
    rng = np.random.default_rng(2)
    n = 100_000
    x = rng.uniform(0, 1, n)
    y = rng.normal(0, 1, n)
    m = gamfit.fit({"x": x, "y": y}, "y ~ s(x)")
    g = {"x": np.linspace(0.05, 0.95, 5)}
    r = m.predict(g, interval=0.8, observation_interval=True)
    z = stats.norm.ppf(0.9)
    np.testing.assert_allclose(r["observation_lower"], -z, atol=0.03)
    np.testing.assert_allclose(r["observation_upper"], z, atol=0.03)
    # and empirical coverage on fresh data
    xt = rng.uniform(0, 1, 20000)
    yt = rng.normal(0, 1, 20000)
    rt = m.predict({"x": xt}, interval=0.8, observation_interval=True)
    cov = np.mean((yt >= rt["observation_lower"]) & (yt <= rt["observation_upper"]))
    assert abs(cov - 0.8) < 0.015, cov


# test_sample / test_sample_predict: posterior draws have the right shapes
def test_posterior_sample_shapes(mcycle_model, mcycle):
    post = mcycle_model.sample(mcycle, samples=50, seed=0)
    k = len(mcycle_model._coefficient_state()["beta"])
    assert np.asarray(post.samples).shape[1] == k
    ndraw = np.asarray(post.samples).shape[0]
    g = {"x": np.linspace(mcycle["x"].min(), mcycle["x"].max(), 7)}
    pd_ = post.predict_draws(g)
    assert np.asarray(pd_.mean).shape == (ndraw, 7)
    yrep = mcycle_model.sample_replicates(mcycle, 11, seed=0)
    assert np.asarray(yrep).shape == (11, len(mcycle["y"]))


# test_pvalue_invariant_to_scale / test_pvalue_sig_impt: noise feature has a large p-value
def test_noise_feature_has_large_pvalue():
    rng = np.random.default_rng(3)
    n = 1000
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + rng.normal(0, 0.5, n)
    m = gamfit.fit({"x": x, "z": z, "y": y}, "y ~ s(x) + s(z)")
    assert _pvalue(_summary_row(m, "z")) > 0.05
    assert _pvalue(_summary_row(m, "x")) < 1e-6


def test_pvalue_and_edf_invariant_to_response_scale():
    rng = np.random.default_rng(4)
    n = 500
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + 0.2 * z + rng.normal(0, 0.5, n)
    d1 = {"x": x, "z": z, "y": y}
    d2 = {"x": x, "z": z, "y": y * 1e6}
    m1 = gamfit.fit(d1, "y ~ s(x) + s(z)")
    m2 = gamfit.fit(d2, "y ~ s(x) + s(z)")
    for t in ("x", "z"):
        r1, r2 = _summary_row(m1, t), _summary_row(m2, t)
        assert _edf(r1) == pytest.approx(_edf(r2), rel=1e-3, abs=1e-3)
        assert _pvalue(r1) == pytest.approx(_pvalue(r2), rel=1e-3, abs=1e-8)
    np.testing.assert_allclose(m2.predict(d2) / 1e6, m1.predict(d1), rtol=1e-5, atol=1e-8)


# test_integer_y: integer binary targets fit
def test_integer_binary_target(default):
    d = {"balance": default["balance"], "y": default["y"].astype(np.int64)}
    m = gamfit.fit(d, "y ~ s(balance)", family="binomial")
    p = m.predict(d)
    assert np.all((p > 0) & (p < 1))


# test_score_* : R^2-type score is <= 1
def test_regressor_score_at_most_one(mcycle):
    from gamfit.sklearn import GAMRegressor
    X = mcycle["x"].reshape(-1, 1)
    r = GAMRegressor(formula="y ~ s(x0)").fit(X, mcycle["y"])
    assert r.score(X, mcycle["y"]) <= 1.0


# test_get_params / test_scale estimation: sigma ~ 3 on N(0, 3^2) noise
def test_scale_estimate():
    rng = np.random.default_rng(5)
    n = 5000
    x = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + rng.normal(0, 3.0, n)
    m = gamfit.fit({"x": x, "y": y}, "y ~ s(x)")
    s = m.summary()
    fields = dict(s.extras or {})
    fields.update({a: getattr(s, a) for a in dir(s) if not a.startswith("_")})
    cand = {k: v for k, v in fields.items() if any(t in k.lower() for t in ("scale", "sigma", "dispersion", "phi"))}
    assert cand, f"no scale/sigma/dispersion field on Summary: {sorted(fields)}"
    vals = [float(v) for v in cand.values() if np.isscalar(v)]
    assert any(abs(v - 3.0) < 0.15 for v in vals) or any(abs(v - 9.0) < 0.5 for v in vals), cand


# test_loglikelihood ordering: s(x) > linear(x) > intercept-only on sine data
def test_loglik_ordering():
    rng = np.random.default_rng(6)
    n = 400
    x = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + rng.normal(0, 0.3, n)
    d = {"x": x, "y": y}
    ll = [gamfit.fit(d, f).summary().log_likelihood for f in ("y ~ s(x)", "y ~ linear(x)", "y ~ 1")]
    assert ll[0] > ll[1] > ll[2]


# test_expectiles: tau validation and ordering (pyGAM ExpectileGAM)
@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_expectile_tau_validation(mcycle, tau):
    with pytest.raises(Exception):
        gamfit.fit(mcycle, "y ~ s(x)", expectile_tau=tau)


def test_expectile_ordering_and_median_equals_gaussian(mcycle):
    m50 = gamfit.fit(mcycle, "y ~ s(x)", expectile_tau=0.5)
    m90 = gamfit.fit(mcycle, "y ~ s(x)", expectile_tau=0.9)
    mg = gamfit.fit(mcycle, "y ~ s(x)")
    p50, p90, pg = m50.predict(mcycle), m90.predict(mcycle), mg.predict(mcycle)
    assert np.mean(p90 > p50) > 0.95
    np.testing.assert_allclose(p50, pg, rtol=1e-4, atol=1e-4 * np.std(mcycle["y"]))
    # expectile 0.9: ~ proportion of y above curve sits well below 0.5
    assert np.mean(mcycle["y"] > p90) < 0.35


# test_weights (pyGAM test_GAM_methods weights/exposure tests). gamfit's contract
# (tests/weighted_gaussian_is_prior_weight_not_frequency_weight_test.py): for fixed-dispersion
# families an integer weight == row duplication; for Gaussian weights are prior weights and
# a global rescale w -> c*w leaves the fit invariant.
def test_poisson_weights_equal_row_duplication():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(0, 1, n)
    y = rng.poisson(np.exp(0.5 + np.sin(6 * x))).astype(float)
    w = rng.integers(1, 4, n).astype(float)
    mw = gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x)", family="poisson", weights="w")
    idx = np.repeat(np.arange(n), w.astype(int))
    md = gamfit.fit({"x": x[idx], "y": y[idx]}, "y ~ s(x)", family="poisson")
    g = {"x": np.linspace(0.02, 0.98, 50)}
    np.testing.assert_allclose(mw.predict(g), md.predict(g), rtol=1e-4)


def test_gaussian_weight_rescale_invariance():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(0, 1, n)
    y = np.sin(6 * x) + rng.normal(0, 0.3, n)
    w = rng.uniform(0.5, 2.0, n)
    m1 = gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x)", weights="w")
    m2 = gamfit.fit({"x": x, "y": y, "w": 7.0 * w}, "y ~ s(x)", weights="w")
    g = {"x": np.linspace(0.02, 0.98, 50)}
    r1, r2 = m1.predict(g, interval=0.95), m2.predict(g, interval=0.95)
    np.testing.assert_allclose(r1["posterior_mean"], r2["posterior_mean"], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(r1["posterior_mean_standard_error"], r2["posterior_mean_standard_error"], rtol=1e-4)


@pytest.mark.parametrize("fam", ["poisson", "gaussian"])
def test_zero_weights_equal_row_deletion(fam):
    """A zero prior weight carries no information: the fit must equal deleting the row."""
    rng = np.random.default_rng(8)
    n = 300
    x = rng.uniform(0, 1, n)
    if fam == "poisson":
        y = rng.poisson(np.exp(0.5 + np.sin(6 * x))).astype(float)
    else:
        y = np.sin(6 * x) + rng.normal(0, 0.3, n)
    w = (rng.uniform(size=n) > 0.3).astype(float)
    # keep x-range identical so default knot placement cannot differ for range reasons
    w[np.argmin(x)] = 1.0
    w[np.argmax(x)] = 1.0
    mw = gamfit.fit({"x": x, "y": y, "w": w}, "y ~ s(x, knot_placement=uniform)", family=fam, weights="w")
    keep = w > 0
    md = gamfit.fit({"x": x[keep], "y": y[keep]}, "y ~ s(x, knot_placement=uniform)", family=fam)
    g = {"x": np.linspace(0.05, 0.95, 50)}
    np.testing.assert_allclose(mw.predict(g), md.predict(g), rtol=1e-4, atol=1e-5)
