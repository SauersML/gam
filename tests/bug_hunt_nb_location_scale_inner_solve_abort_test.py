"""Bug hunt: a negative-binomial location-scale (dispersion) fit aborts on
textbook heteroscedastic count data that every sibling path fits.

``docs/location-scale.md`` / ``noise_formula=`` advertises GAMLSS-style joint
modelling of the mean and a per-row dispersion for the dispersion families
(gamma / nb / beta / tweedie). For the negative binomial this is

    gamfit.fit(df, "y ~ s(x)", family="nb", noise_formula="s(x)")

On well-posed data — a smooth log-mean ``mu(x) = exp(1 + 0.5 x)`` and a smoothly
varying NB size/dispersion ``size(x) = exp(1.5 - 0.7 x)`` at ``n = 10000`` — this
aborts at fit time:

    IntegrationError: exact two-block spatial optimization failed:
      ... REML smoothing optimization failed to converge: custom-family inner
      solve did not converge after 44 cycle(s); refusing to expose profile
      objective derivatives for theta_dim=5 (rho_dim=5, psi_dim=0). The analytic
      outer gradient/Hessian require the inner KKT equation F_beta(beta,theta)=0;
      returning a value with zero or shape-only derivatives is mathematically
      inconsistent.

The two-block inner P-IRLS coordinate solve fails to reach KKT stationarity, and
the non-convergence is escalated to a fatal error (the custom-family profile
objective refuses to return derivatives at a non-converged inner state — see
``crates/gam-custom-family/src/psi_hyper.rs`` ~lines 945-958, reached through the
NB two-block dispersion path).

The data is well-posed and the failure is specific to the NB location-scale path:

  * a plain ``family="nb"`` fit (no ``noise_formula``) on the SAME data fits,
  * a Gaussian location-scale fit (same design, same ``n``) fits,
  * a Gamma location-scale fit on analogous positive data fits.

So this is neither bad data nor a generic two-block defect — it is an
NB-specific inner-solver robustness failure. (Sibling of the cause-specific
``custom-family inner solve did not converge`` escalation in the same file;
both stem from the inner blockwise solve stalling and being promoted to fatal
instead of degrading gracefully.)

This test fits the deterministic dataset (and asserts two well-posed controls
fit it) and requires the NB location-scale fit to succeed and predict finite
per-row means. It fails today at the NB ``fit`` call. When the two-block NB
inner solve converges (or its non-convergence is handled without aborting the
fit), the test passes without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

N = 10000
SEED = 22


def _nb_heteroscedastic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-1.5, 1.5, N)
    mu = np.exp(1.0 + 0.5 * x)  # smooth log-mean
    size = np.exp(1.5 - 0.7 * x)  # smoothly varying NB size / dispersion
    y = rng.negative_binomial(size, size / (size + mu))
    return pd.DataFrame({"x": x, "y": y})


def test_nb_location_scale_fit_succeeds_on_heteroscedastic_counts() -> None:
    df = _nb_heteroscedastic_frame()

    # Control 1: plain NB (no dispersion block) must fit this data — proves the
    # mean structure / data scale are fine.
    plain = gamfit.fit(df, "y ~ s(x)", family="nb")
    assert np.all(np.isfinite(np.asarray(plain.predict(df), dtype=float)))

    # Control 2: a Gaussian location-scale fit on the same design and n fits,
    # so the two-block dispersion machinery itself is not the problem.
    gauss_ls = gamfit.fit(df, "y ~ s(x)", family="gaussian", noise_formula="s(x)")
    assert np.all(np.isfinite(np.asarray(gauss_ls.predict(df), dtype=float)))

    # The documented NB location-scale fit. This is the line that currently
    # raises IntegrationError ("custom-family inner solve did not converge").
    model = gamfit.fit(df, "y ~ s(x)", family="nb", noise_formula="s(x)")

    mean = np.asarray(model.predict(df), dtype=float)
    assert mean.shape[0] == len(df)
    assert np.all(np.isfinite(mean))
    assert np.all(mean > 0.0)  # NB mean is strictly positive
