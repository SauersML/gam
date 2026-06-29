"""Bug hunt: a 2-D ``matern(x, z)`` isotropic-Matérn GP smooth is unfittable on
the common non-Gaussian GLM families (Gamma / Poisson / negative-binomial) — it
deterministically aborts the REML fit with

    IntegrationError: REML smoothing optimization failed to converge:
    spatial kappa optimization failed: Invalid input:
    bounded linear terms are not supported for GammaLog fits

even though the *same data* fits fine with ``te(x, z)`` and ``thinplate(x, z)``,
and ``matern(x, z)`` itself fits fine under the Gaussian and Binomial families.

Root cause: the Matérn isotropic length-scale (κ / range) optimization (the
spatial-smooth outer search) evaluates its candidate κ via a *bounded-linear-term*
inner observation-state builder. That builder
(``crates/gam-models/src/fit_orchestration/drivers/design_construction.rs``,
the ``build_*observation_state`` match around lines 3401-3425) only implements
the Gaussian and Binomial arms; every other ``ResponseFamily`` arm is a hard
``bail_invalid_estim!("bounded linear terms are not supported for <Family> fits")``:

    (ResponseFamily::Poisson, _)            => bail "...PoissonLog fits"
    (ResponseFamily::Tweedie { .. }, _)     => bail "...Tweedie fits"
    (ResponseFamily::NegativeBinomial..)    => bail "...NegativeBinomial fits"
    (ResponseFamily::Beta { .. }, _)        => bail "...BetaLogit fits"
    (ResponseFamily::Gamma, _)              => bail "...GammaLog fits"

So whenever the κ search reaches that path under a non-Gaussian family the whole
fit aborts. This is distinct from the Gaussian-only Matérn defects already filed
(#1270 penalty-topology staging, #1357 Gaussian κ collapse, #1379 univariate
Gaussian range penalty): here the forward fit itself is refused purely because of
the *family*, while the identical design fits under `te`/`thinplate`.

This test builds a Gamma-distributed 2-D surface (a configuration that triggers
the bail deterministically — most seeds at this size/shape hit the
bounded-linear κ path), confirms the surface IS recoverable via a control
``te(x, z)`` fit, and then asserts that the documented ``matern(x, z)`` smooth
also fits and recovers it. It currently fails (the matern fit raises before
producing any result); once the κ optimizer no longer routes non-Gaussian
families through the Gaussian/Binomial-only bounded-linear builder, the fit
completes and the assertion holds without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _gamma_surface(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    eta = 0.8 * np.sin(3.0 * x) + 0.5 * z
    mu = np.exp(eta)
    shape = 5.0
    y = rng.gamma(shape, mu / shape)  # Gamma mean=mu, var=mu^2/shape
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    return df, mu


def test_matern_2d_smooth_is_fittable_under_gamma_family() -> None:
    df, mu = _gamma_surface(seed=0)

    # Control: the SAME data fits cleanly with a tensor-product smooth under the
    # Gamma family, so the data / family / formula machinery is sound — only the
    # matern κ optimizer's family handling is at fault.
    te_model = gamfit.fit(df, "y ~ te(x, z)", family="gamma")
    te_pred = np.asarray(te_model.predict(df), dtype=float)
    assert np.all(np.isfinite(te_pred)), "te control produced non-finite predictions"
    te_corr = float(np.corrcoef(te_pred, mu)[0, 1])
    assert te_corr > 0.4, f"sanity: te(x,z) should recover the surface, got corr={te_corr:.3f}"

    # The documented matern smooth must fit the same data under the same family.
    # Today this raises IntegrationError ("bounded linear terms are not supported
    # for GammaLog fits") from the κ optimization before any prediction exists.
    matern_model = gamfit.fit(df, "y ~ matern(x, z)", family="gamma")
    matern_pred = np.asarray(matern_model.predict(df), dtype=float)
    assert np.all(np.isfinite(matern_pred)), "matern produced non-finite predictions"
    matern_corr = float(np.corrcoef(matern_pred, mu)[0, 1])
    assert matern_corr > 0.5, (
        f"matern(x,z) under Gamma must recover the surface like te/thinplate do, "
        f"got corr={matern_corr:.3f}"
    )
