"""Translation of pygam/tests/test_GAMs.py + test_gen_imgs.py.

pyGAM fits each family on a canonical dataset (LinearGAM/mcycle, LogisticGAM/default,
PoissonGAM/coal, GammaGAM/trees, InvGaussGAM/trees, GAM(gamma, link='inverse')) and
renders plots for each. We assert the fits exist, converge and predict on the support.
"""
import numpy as np
import pytest

import gamfit


def _converged(m):
    c = m.summary().convergence
    return bool(c and (c.get("certified") or c.get("inner_status") == "Converged"))


def test_gaussian_mcycle(mcycle_model):
    assert _converged(mcycle_model)


def test_binomial_default(default):
    m = gamfit.fit(default, "y ~ factor(student) + s(balance) + s(income)", family="binomial")
    p = m.predict(default)
    assert _converged(m) and np.all((p > 0) & (p < 1))


def test_poisson_coal(coal):
    m = gamfit.fit(coal, "y ~ s(x)", family="poisson")
    assert _converged(m) and np.all(m.predict(coal) > 0)


def test_gamma_trees(trees):
    m = gamfit.fit(trees, "y ~ s(girth) + s(height)", family="gamma")
    assert _converged(m) and np.all(m.predict(trees) > 0)


def test_inverse_gaussian_trees(trees):
    """pyGAM InvGaussGAM."""
    m = gamfit.fit(trees, "y ~ s(girth) + s(height)", family="inverse_gaussian")
    assert _converged(m)


def test_gamma_inverse_link(trees):
    """pyGAM GAM(distribution='gamma', link='inverse') -- the canonical Gamma link."""
    m = gamfit.fit(trees, "y ~ s(girth) + s(height)", family="gamma", link="inverse")
    assert _converged(m)


def test_gamma_identity_link(trees):
    """pyGAM GAM(distribution='gamma', link='identity')."""
    m = gamfit.fit(trees, "y ~ s(girth) + s(height)", family="gamma", link="identity")
    assert _converged(m)


def test_head_circumference_big_gaussian():
    import pygam.datasets as ds
    X, y = ds.head_circumference(return_X_y=True)
    d = {"x": X[:, 0].astype(float), "y": np.asarray(y, float)}
    m = gamfit.fit(d, "y ~ s(x)")
    assert _converged(m)


# test_gen_imgs.py: plotting smoke tests (pyGAM draws pdep + CI per term)
@pytest.mark.parametrize("formula,fam,fixture", [
    ("y ~ s(x)", "gaussian", "mcycle"),
    ("y ~ s(x)", "poisson", "coal"),
    ("y ~ s(year) + s(age) + factor(edu)", "gaussian", "wage"),
])
def test_plot_renders(formula, fam, fixture, request, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    d = request.getfixturevalue(fixture)
    m = gamfit.fit(d, formula, family=fam)
    out = m.plot(d)
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.savefig(tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000
    plt.close("all")


def test_plot_tensor_renders(chicago, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = {k: v[:1500] for k, v in chicago.items()}
    m = gamfit.fit(d, "y ~ te(tmpd, o3)", family="poisson")
    m.plot(d)
    plt.gcf().savefig(tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 1000
    plt.close("all")
