"""Translation of pygam/tests/test_terms.py + test_GAM_params.py (SPEC-compatible parts)."""
import numpy as np
import pytest

import gamfit
from conftest import pdep


def _width(m, pred):
    b = [b for b in m.term_blocks if pred(b.name)]
    assert len(b) == 1, m.term_blocks
    return b[0].end - b[0].start


# test_terms::test_n_coefs / test_GAM_params::test_n_splines: block widths
def test_block_width_spline(mcycle):
    m = gamfit.fit(mcycle, "y ~ s(x, k=10)")
    assert _width(m, lambda n: n.startswith("s(")) == 9  # k minus 1 identifiability constraint


def test_block_width_factor(wage):
    m = gamfit.fit(wage, "y ~ factor(edu)")
    nlev = len(np.unique(wage["edu"]))
    w = sum(b.end - b.start for b in m.term_blocks if b.kind != "intercept")
    assert w == nlev - 1, m.term_blocks


def test_block_width_tensor(chicago):
    d = {k: v[:1000] for k, v in chicago.items()}
    m = gamfit.fit(d, "y ~ te(pm10, o3, k=[5, 4])", family="poisson")
    assert _width(m, lambda n: n.startswith("te(")) == 5 * 4 - 1


# test_GAM_params::test_intercept: intercept-only model is the mean
def test_intercept_only_is_mean(mcycle):
    m = gamfit.fit(mcycle, "y ~ 1")
    np.testing.assert_allclose(m.predict(mcycle), np.mean(mcycle["y"]), rtol=1e-10)


# test_GAM_params::test_linear_term: an unpenalized linear term reproduces OLS
def test_linear_term_is_ols(mcycle):
    m = gamfit.fit(mcycle, "y ~ linear(x)")
    b = np.polyfit(mcycle["x"], mcycle["y"], 1)
    np.testing.assert_allclose(m.predict(mcycle), np.polyval(b, mcycle["x"]), rtol=1e-6, atol=1e-6)


# test_GAM_params::test_fit_intercept=False / test_partial_dependence_on_univar_data2:
# a model without an intercept. pyGAM: GAM(fit_intercept=False).
@pytest.mark.parametrize("formula", ["y ~ 0 + linear(x)", "y ~ linear(x) - 1"])
def test_no_intercept_formula(formula):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    y = 3 * x + rng.normal(0, 0.1, 200)
    m = gamfit.fit({"x": x, "y": y}, formula)
    assert not any(b.kind == "intercept" for b in m.term_blocks)
    np.testing.assert_allclose(m.predict({"x": np.array([0.0])}), 0.0, atol=1e-10)


# test_terms::test_tensor_invariance_to_scaling (skipped in pyGAM, meaningful here):
def test_tensor_invariance_to_covariate_rescaling(chicago):
    d = {k: v[:1500] for k, v in chicago.items()}
    m1 = gamfit.fit(d, "y ~ te(tmpd, o3)", family="poisson")
    d2 = dict(d)
    d2["o3"] = d["o3"] * 100.0
    m2 = gamfit.fit(d2, "y ~ te(tmpd, o3)", family="poisson")
    np.testing.assert_allclose(m1.predict(d), m2.predict(d2), rtol=2e-3)


# test_terms::test_by_variable / test_by_variable_doesnt_exist
def test_by_numeric_close_to_tensor_with_linear_margin(toy_interaction):
    d = toy_interaction
    mby = gamfit.fit(d, "y ~ s(x1, by=x0)")
    mte = gamfit.fit(d, "y ~ te(x0, x1, degree=[1, 3])")
    ss = np.sum((d["y"] - d["y"].mean()) ** 2)
    r2by = 1 - np.sum((d["y"] - mby.predict(d)) ** 2) / ss
    r2te = 1 - np.sum((d["y"] - mte.predict(d)) ** 2) / ss
    assert r2by > 0.99 and r2te > 0.99
    assert abs(r2by - r2te) < 0.01


def test_by_variable_missing_raises(mcycle):
    with pytest.raises(Exception):
        gamfit.fit(mcycle, "y ~ s(x, by=nope)")


# test_terms::test_correct_smoothing_in_tensors: REML puts the larger lambda on the
# linear direction of x0 * sin(x1), and the fit explains the signal.
def test_reml_tensor_explains_interaction(toy_interaction):
    d = toy_interaction
    m = gamfit.fit(d, "y ~ te(x0, x1)")
    ss = np.sum((d["y"] - d["y"].mean()) ** 2)
    r2 = 1 - np.sum((d["y"] - m.predict(d)) ** 2) / ss
    assert r2 > 0.9


# test_terms::test_cyclic / test_GAM_params::test_cyclic_basis: periodic smooth
def test_cyclic_is_periodic():
    rng = np.random.default_rng(9)
    n = 500
    x = rng.uniform(0, 24, n)
    y = np.sin(2 * np.pi * x / 24) + rng.normal(0, 0.2, n)
    m = gamfit.fit({"x": x, "y": y}, "y ~ cyclic(x, period=24)")
    f0 = m.predict({"x": np.array([0.0, 6.0])})
    f1 = m.predict({"x": np.array([24.0, 30.0])})
    np.testing.assert_allclose(f0, f1, atol=1e-8)


def test_cyclic_worse_than_free_smooth_on_aperiodic_data(hepatitis):
    """pyGAM test_cyclic_basis_on_non_cyclic: periodic basis must fit worse on monotone data."""
    d = hepatitis
    mc = gamfit.fit(d, "y ~ cyclic(x)")
    ms = gamfit.fit(d, "y ~ s(x)")
    assert ms.summary().deviance < mc.summary().deviance


# test_terms::test_tensor_with_constraints: shape constraint on a tensor margin
def test_tensor_margin_shape_constraint(chicago):
    d = {k: v[:1500] for k, v in chicago.items()}
    m = gamfit.fit(d, "y ~ te(tmpd, o3, shape=[monotone_increasing, none])", family="poisson")
    assert np.all(np.isfinite(m.predict(d)))


# test_terms::test_build_from_info / test_GAM_methods::test_save_load: round-trip
def test_save_load_roundtrip(tmp_path, mcycle_model, mcycle):
    p = tmp_path / "m.gam"
    mcycle_model.save(str(p))
    m2 = gamfit.load(str(p))
    np.testing.assert_array_equal(m2.predict(mcycle), mcycle_model.predict(mcycle))
    m3 = gamfit.loads(mcycle_model.dumps())
    np.testing.assert_array_equal(m3.predict(mcycle), mcycle_model.predict(mcycle))


# test_terms::test_tensor_terms: pyGAM raises on a length-1 n_splines for a 2-D tensor;
# gamfit documents broadcasting (docs/formulas.md "Margins requested as a single value are
# broadcast"), so assert the documented equivalence instead.
def test_tensor_k_single_value_broadcasts(chicago):
    d = {k: v[:500] for k, v in chicago.items()}
    a = gamfit.fit(d, "y ~ te(pm10, o3, k=[5])", family="poisson")
    b = gamfit.fit(d, "y ~ te(pm10, o3, k=5)", family="poisson")
    np.testing.assert_allclose(a.predict(d), b.predict(d), rtol=1e-8)


# test_utils::test_check_X_categorical_prediction_exceeds_training
def test_unseen_factor_level_at_predict_raises(wage):
    m = gamfit.fit(wage, "y ~ s(age) + factor(edu)")
    new = {"age": np.array([40.0]), "year": np.array([2005.0]), "edu": np.array(["e99"])}
    with pytest.raises(Exception):
        m.predict(new)


# test_utils::test_input_data_after_fitting / check_X numeric: s() of a string column must raise
def test_smooth_of_string_column_raises():
    rng = np.random.default_rng(0)
    x = np.array([f"v{i % 7}" for i in range(200)])
    y = rng.normal(size=200)
    with pytest.raises(Exception):
        gamfit.fit({"x": x, "y": y}, "y ~ s(x)")
