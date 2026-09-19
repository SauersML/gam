"""Translation of pygam/tests/test_penalties.py::test_monotonic_* / test_convex_* / test_concave_*.

pyGAM asserts, on hepatitis with a sorted grid:
  * constraints='monotonic_inc'  -> diff(pred) >= 0
  * constraints='monotonic_dec'  -> diff(pred) <= 0   (even though the data increase)
  * constraints='convex'         -> diff(diff(pred)) >= 0
  * constraints='concave'        -> diff(diff(pred)) <= 0
gamfit spelling: s(x, shape=monotone_increasing|monotone_decreasing|convex|concave).
gamfit must be stronger: the guarantee must hold for the plug-in, the reported
posterior mean, the partial-dependence curve AND for posterior draws.
"""
import numpy as np
import pytest

import gamfit
from conftest import pdep

SHAPES = {
    "monotone_increasing": (1, +1),
    "monotone_decreasing": (1, -1),
    "convex": (2, +1),
    "concave": (2, -1),
}
TOL = 1e-8


def _grid(d, n=200):
    return np.linspace(d["x"].min(), d["x"].max(), n)


def _check(values, order, sign, tol=TOL):
    dv = np.diff(np.asarray(values, float), n=order, axis=-1)
    assert np.all(sign * dv >= -tol * max(1.0, np.max(np.abs(values)))), (
        f"shape violated: worst {np.min(sign * dv):.3g}"
    )


@pytest.fixture(scope="module", params=list(SHAPES))
def shaped(request, hepatitis):
    shape = request.param
    m = gamfit.fit(hepatitis, f"y ~ s(x, shape={shape})")
    return shape, m


def test_plugin_prediction_respects_shape(shaped, hepatitis):
    shape, m = shaped
    g = _grid(hepatitis)
    r = m.predict({"x": g}, interval=0.95)
    _check(r["mean_plugin"], *SHAPES[shape])


def test_posterior_mean_respects_shape(shaped, hepatitis):
    shape, m = shaped
    g = _grid(hepatitis)
    r = m.predict({"x": g}, interval=0.95)
    _check(r["posterior_mean"], *SHAPES[shape])


def test_partial_dependence_respects_shape(shaped, hepatitis):
    shape, m = shaped
    names = [b.name for b in m.term_blocks if b.kind != "intercept"]
    p = pdep(m, names[0], hepatitis, grid=_grid(hepatitis))
    _check(p["predicted"], *SHAPES[shape])


def test_posterior_draws_respect_shape(shaped, hepatitis):
    shape, m = shaped
    g = _grid(hepatitis, 60)
    post = m.sample(hepatitis, samples=200, seed=0)
    draws = np.asarray(post.predict_draws({"x": g}).eta, float)
    order, sign = SHAPES[shape]
    dv = sign * np.diff(draws, n=order, axis=1)
    frac_bad = np.mean(np.any(dv < -1e-8 * np.max(np.abs(draws)), axis=1))
    assert frac_bad == 0.0, f"{frac_bad:.1%} of posterior draws violate {shape}"


def test_interval_bounds_respect_monotone(hepatitis):
    """If every draw is monotone the pointwise quantile band endpoints are monotone too."""
    m = gamfit.fit(hepatitis, "y ~ s(x, shape=monotone_increasing)")
    g = _grid(hepatitis)
    r = m.predict({"x": g}, interval=0.95)
    _check(r["posterior_mean_lower"], 1, +1, tol=1e-6)
    _check(r["posterior_mean_upper"], 1, +1, tol=1e-6)


def test_monotone_binomial_and_poisson(chicago):
    """pyGAM test_terms::test_tensor_with_constraints-style: Poisson with a monotone term fits."""
    d = {k: v[:1500] for k, v in chicago.items()}
    m = gamfit.fit(d, "y ~ s(tmpd, shape=monotone_decreasing) + s(time)", family="poisson")
    g = np.linspace(d["tmpd"].min(), d["tmpd"].max(), 100)
    p = pdep(m, "s(tmpd, shape=monotone_decreasing)" if any(
        b.name == "s(tmpd, shape=monotone_decreasing)" for b in m.term_blocks) else
        next(b.name for b in m.term_blocks if "tmpd" in b.name), d, grid=g)
    _check(p["predicted"], 1, -1)
