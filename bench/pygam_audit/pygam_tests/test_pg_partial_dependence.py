"""Translation of pygam/tests/test_partial_dependence.py (+ pdep parts of test_GAM_methods.py).

pyGAM asserts:
  * sum of term partial dependences (+ intercept) == linear predictor   (test_partial_dependence_on_univar_data*)
  * default grid == explicit linspace grid over training range          (test_partial_dependence_gives_correct_shape_*)
  * 2-D tensor term partial dependence works                             (test_partial_dependence_gives_correct_shape_with_meshgrid)
  * bad term index raises                                                (test_partial_dependence_raises_error_*)
  * width => confidence intervals with shape (n, 2)                      (test_partial_dependence_width)
"""
import numpy as np
import pytest

import gamfit
from bench.pygam_audit.pygam_tests.pg_helpers import eta_of, intercept, pdep


def test_univariate_pdep_plus_intercept_equals_prediction(mcycle, mcycle_model):
    m = mcycle_model
    p = pdep(m, "s(x)", mcycle, grid=mcycle["x"])
    eta = eta_of(m, mcycle)
    np.testing.assert_allclose(eta, intercept(m, mcycle) + p["predicted"], atol=1e-8)


def test_multi_term_additive_identity_numeric(chicago):
    d = chicago
    m = gamfit.fit(d, "y ~ s(time) + s(tmpd) + te(pm10, o3)", family="poisson")
    eta = eta_of(m, d)
    total = intercept(m, d)
    total = total + pdep(m, "s(time)", d, grid=d["time"])["predicted"]
    total = total + pdep(m, "s(tmpd)", d, grid=d["tmpd"])["predicted"]
    total = total + pdep(m, "te(pm10, o3)", d, grid=np.column_stack([d["pm10"], d["o3"]]))["predicted"]
    np.testing.assert_allclose(eta, total, atol=1e-7)


def test_multi_term_additive_identity_with_factor(wage):
    d = wage
    m = gamfit.fit(d, "y ~ s(year) + s(age) + factor(edu)")
    eta = eta_of(m, d)
    resid = eta - intercept(m, d)
    resid = resid - pdep(m, "s(year)", d, grid=d["year"])["predicted"]
    resid = resid - pdep(m, "s(age)", d, grid=d["age"])["predicted"]
    # what remains must be a pure function of the factor level
    for lev in np.unique(d["edu"]):
        r = resid[d["edu"] == lev]
        assert np.ptp(r) < 1e-8, (lev, np.ptp(r))


def test_factor_term_partial_dependence_supported(wage):
    """pyGAM: partial_dependence works for a factor term (plot bar per level)."""
    d = wage
    m = gamfit.fit(d, "y ~ s(age) + factor(edu)")
    names = [b.name for b in m.term_blocks]
    fterm = next(n for n in names if "edu" in n)
    out = pdep(m, fterm, d)
    assert np.all(np.isfinite(out["predicted"]))


def test_default_grid_equals_explicit_linspace(mcycle, mcycle_model):
    m = mcycle_model
    a = pdep(m, "s(x)", mcycle)
    g = np.linspace(mcycle["x"].min(), mcycle["x"].max(), 100)
    b = pdep(m, "s(x)", mcycle, grid=g)
    np.testing.assert_allclose(a["grid"], g)
    np.testing.assert_allclose(a["predicted"], b["predicted"])
    assert a["predicted"].shape == (100,)
    assert a["standard_error"].shape == (100,)
    assert np.all(a["standard_error"] > 0)


def test_n_points_controls_grid(mcycle, mcycle_model):
    assert pdep(mcycle_model, "s(x)", mcycle, n_points=37)["predicted"].shape == (37,)


def test_tensor_pdep_2d_grid(chicago):
    d = chicago
    m = gamfit.fit(d, "y ~ te(pm10, o3)", family="poisson")
    gx, gy = np.meshgrid(np.linspace(d["pm10"].min(), d["pm10"].max(), 7),
                         np.linspace(d["o3"].min(), d["o3"].max(), 5))
    out = pdep(m, "te(pm10, o3)", d, grid=np.column_stack([gx.ravel(), gy.ravel()]))
    assert out["predicted"].shape == (35,)
    assert np.all(np.isfinite(out["standard_error"]))


def test_bad_term_raises(mcycle, mcycle_model):
    with pytest.raises(ValueError):
        pdep(mcycle_model, "s(nope)", mcycle)


def test_pdep_se_priced_off_published_covariance(mcycle, mcycle_model):
    """pyGAM: partial_dependence(width=) intervals come from the model covariance.
    gamfit: the pdep SE must be priced off the same covariance predict() reports."""
    m = mcycle_model
    p = pdep(m, "s(x)", mcycle)
    r = m.predict(mcycle, interval=0.95)
    assert p["covariance_source"] == r["covariance_source"]
    assert np.all(np.isfinite(p["standard_error"])) and np.all(p["standard_error"] > 0)
