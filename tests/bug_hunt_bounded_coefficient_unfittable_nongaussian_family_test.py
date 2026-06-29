"""Bug hunt: the documented ``bounded(x, min, max)`` interval-bounded-coefficient
term is unfittable on the non-Gaussian GLM families (Poisson / Gamma /
negative-binomial). Fitting one deterministically aborts with

    IntegrationError: Custom-family fit failed: custom-family invalid input in
    custom-family string boundary: Invalid input:
    bounded linear terms are not supported for PoissonLog fits

even though a plain ``y ~ x`` linear term fits the same Poisson data, and
``bounded(x, min, max)`` fits fine under the Gaussian and Binomial families.

Root cause (same defect as the Matérn-GP failure in the sibling issue): the
bounded-linear-term observation-state builder in
``crates/gam-models/src/fit_orchestration/drivers/design_construction.rs``
(the family match around lines 3401-3425) implements ONLY the Gaussian and
Binomial arms; every other ``ResponseFamily`` is a hard
``bail_invalid_estim!("bounded linear terms are not supported for <Family> fits")``:

    (ResponseFamily::Poisson, _)         => bail "...PoissonLog fits"
    (ResponseFamily::Tweedie { .. }, _)  => bail "...Tweedie fits"
    (ResponseFamily::NegativeBinomial..) => bail "...NegativeBinomial fits"
    (ResponseFamily::Beta { .. }, _)     => bail "...BetaLogit fits"
    (ResponseFamily::Gamma, _)           => bail "...GammaLog fits"

A ``bounded()`` term is realized as a custom-family box-constrained ("string
boundary") coefficient whose inner solve evaluates the observation state through
exactly that builder, so the fit aborts the moment the family is not
Gaussian/Binomial. Unlike the Matérn manifestation (which depends on the spatial
κ search reaching the path), this one is fully deterministic: every
``bounded(...)`` fit under Poisson/Gamma/NB bails.

This test fits a Poisson count model whose log-mean increases linearly in ``x``
with a true slope (0.7) interior to the requested bound ``[0, 2]``, confirms the
data IS fittable via a plain ``y ~ x`` control, and then asserts the
``bounded(x, min=0, max=2)`` term fits and recovers the increasing trend. It
currently fails (the bounded fit raises before producing any result); once the
bounded-linear builder handles the non-Gaussian families, the fit completes and
the assertion holds without edits.

Related: #1615
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _poisson_linear(seed: int = 0, n: int = 500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    mu = np.exp(0.5 + 0.7 * x)  # true slope 0.7 is interior to [0, 2]
    y = rng.poisson(mu)
    df = pd.DataFrame({"x": x, "y": y})
    return df, mu


def test_bounded_coefficient_is_fittable_under_poisson_family() -> None:
    df, mu = _poisson_linear(seed=0)

    # Control: the same Poisson data fits cleanly with an ordinary linear term,
    # so the data / family / formula machinery is sound — only the bounded
    # (box-constrained) coefficient path is at fault.
    ctrl = gamfit.fit(df, "y ~ x", family="poisson")
    ctrl_pred = np.asarray(ctrl.predict(df), dtype=float)
    assert np.all(np.isfinite(ctrl_pred)), "linear control produced non-finite predictions"
    ctrl_corr = float(np.corrcoef(ctrl_pred, mu)[0, 1])
    assert ctrl_corr > 0.5, f"sanity: y~x should track the trend, got corr={ctrl_corr:.3f}"

    # The documented bounded() interval-coefficient term must fit the same data
    # under the same family. Today this raises IntegrationError ("bounded linear
    # terms are not supported for PoissonLog fits") before any prediction exists.
    bnd = gamfit.fit(df, "y ~ bounded(x, min=0, max=2)", family="poisson")
    bnd_pred = np.asarray(bnd.predict(df), dtype=float)
    assert np.all(np.isfinite(bnd_pred)), "bounded fit produced non-finite predictions"
    bnd_corr = float(np.corrcoef(bnd_pred, mu)[0, 1])
    assert bnd_corr > 0.5, (
        f"bounded(x, min=0, max=2) under Poisson must recover the increasing "
        f"trend (true slope 0.7 is interior to the bound), got corr={bnd_corr:.3f}"
    )
