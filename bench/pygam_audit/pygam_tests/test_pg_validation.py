"""Translation of pygam/tests/test_utils.py + validation asserts in test_GAM_methods/test_core.

pyGAM asserts check_X/check_y reject NaN/inf, wrong shapes, out-of-domain y, and that
fitting/predicting on bad input raises cleanly.
"""
import numpy as np
import pytest

import gamfit


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    return {"x": x, "y": np.sin(6 * x) + rng.normal(0, 0.3, 200)}


@pytest.mark.parametrize("col,val", [("x", np.nan), ("y", np.nan), ("x", np.inf), ("y", -np.inf)])
def test_nonfinite_fit_raises(xy, col, val):
    d = {k: v.copy() for k, v in xy.items()}
    d[col][3] = val
    with pytest.raises(Exception):
        gamfit.fit(d, "y ~ s(x)")


@pytest.mark.parametrize("val", [np.nan, np.inf])
def test_nonfinite_predict_raises(xy, val):
    m = gamfit.fit(xy, "y ~ s(x)")
    with pytest.raises(Exception):
        m.predict({"x": np.array([0.5, val])})


@pytest.mark.parametrize("fam,bad", [("binomial", 2.0), ("poisson", -1.0), ("gamma", 0.0)])
def test_out_of_domain_y_raises(xy, fam, bad):
    d = {"x": xy["x"], "y": np.abs(xy["y"]) + 0.1}
    if fam == "binomial":
        d["y"] = (xy["y"] > 0).astype(float)
    if fam == "poisson":
        d["y"] = np.round(d["y"] * 3)
    d["y"][0] = bad
    with pytest.raises(Exception):
        gamfit.fit(d, "y ~ s(x)", family=fam)


def test_length_mismatch_raises(xy):
    with pytest.raises(Exception):
        gamfit.fit({"x": xy["x"][:-1], "y": xy["y"]}, "y ~ s(x)")


def test_negative_weights_raise(xy):
    d = dict(xy)
    d["w"] = np.ones_like(xy["x"])
    d["w"][0] = -1
    with pytest.raises(Exception):
        gamfit.fit(d, "y ~ s(x)", weights="w")


def test_no_terms_raises(xy):
    with pytest.raises(Exception):
        gamfit.fit(xy, "y ~ ")


def test_extrapolation_finite(xy):
    """test_utils / test_GAM_methods::test_extrapolation: predictions beyond the range are finite."""
    m = gamfit.fit(xy, "y ~ s(x)")
    p = m.predict({"x": np.array([-1.0, 2.0])})
    assert np.all(np.isfinite(p))


def test_extrapolation_is_linear(xy):
    """pyGAM's B-spline basis extrapolates linearly (second-difference penalty null space);
    the posterior-mean curve beyond the boundary should have zero curvature."""
    m = gamfit.fit(xy, "y ~ s(x)")
    xs = np.array([1.5, 2.0, 2.5, 3.0])
    p = m.predict({"x": xs})
    np.testing.assert_allclose(np.diff(p, 2), 0.0, atol=1e-8 * max(1, np.max(np.abs(p))))


def test_pandas_inputs(xy):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(xy)
    m = gamfit.fit(df, "y ~ s(x)")
    np.testing.assert_allclose(m.predict(df), gamfit.fit(xy, "y ~ s(x)").predict(xy), rtol=1e-10)


def test_fit_array_1d_design(xy):
    """pyGAM accepts a 1-D X; gamfit.fit_array should accept (n,) or document (n,1)."""
    m = gamfit.fit_array(xy["x"].reshape(-1, 1), xy["y"], "y ~ s(x0)")
    assert np.all(np.isfinite(m.predict_array(xy["x"].reshape(-1, 1))))
