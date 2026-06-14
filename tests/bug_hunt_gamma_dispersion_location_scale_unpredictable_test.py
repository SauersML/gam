"""#1119: the genuine-dispersion mean families NB / Gamma / Tweedie, when
fitted as a location-scale (GAMLSS) model via ``noise_formula``, must assemble
a joint posterior covariance and a finite EDF and be predictable — exactly like
their Beta sibling and the Gaussian ``gaulss`` location-scale model.

Root cause (now fixed): the ``DispersionGlmLocationScale`` engine (#913) only
built the joint coefficient Hessian for Beta (the one member with a nonzero
mean/precision Fisher cross block). The Fisher-orthogonal members
(NegativeBinomial / Gamma / Tweedie) returned ``None`` for the joint Hessian,
so the multi-block outer-REML path failed the "multi-block families must
provide a joint outer path" gate, the fit silently escalated to a degraded
ρ-seed refit, and the result carried no covariance and no EDF. Every
``predict`` mode then aborted with::

    posterior-mean prediction failed: ... requires covariance or penalized
    Hessian for posterior-mean prediction

The fix assembles the (block-diagonal, zero-cross) joint Hessian for all four
members and declares ``likelihood_blocks_uncoupled`` for the orthogonal ones so
the directional-derivative / Jeffreys dispatch accepts it. This file pins the
end-to-end behaviour from the Python surface, the angle a regression of the
same root cause would re-break.
"""

from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _gamma_location_scale_frame(n: int = 1500, seed: int = 20260614) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    mean = np.exp(0.5 + 0.6 * x)
    shape = np.exp(0.7 + 0.4 * np.cos(2 * x))  # genuine varying dispersion
    y = rng.gamma(shape=shape, scale=mean / shape)
    return pd.DataFrame({"y": y, "x": x})


def _assert_joint_covariance_and_predictable(
    m: typing.Any,
    *,
    x_grid: np.ndarray,
    true_mean: np.ndarray,
    family: str,
) -> None:
    """The shared contract: a location-scale fit exposes a joint (mean +
    log-precision) posterior covariance and a finite EDF, and ``predict``
    returns finite means that recover the conditional-mean trend."""
    s = m.summary()
    n_coef = len(s.coefficients)
    assert n_coef > 0, f"{family}: fit produced no coefficients"

    # (1) Joint posterior covariance is assembled over the FULL coefficient
    #     vector (mean block + log-precision block), not left unset.
    assert s.covariance_n is not None, (
        f"{family} location-scale fit must expose a joint posterior covariance; "
        f"got covariance_n=None (the orthogonal-family joint-Hessian regression)"
    )
    assert s.covariance_n == n_coef, (
        f"{family}: covariance must span all {n_coef} coefficients, "
        f"got covariance_n={s.covariance_n}"
    )

    # (2) A finite effective degrees of freedom.
    assert s.edf_total is not None and np.isfinite(s.edf_total), (
        f"{family}: edf_total must be finite, got {s.edf_total}"
    )
    assert 0.0 < s.edf_total <= n_coef + 1e-6, (
        f"{family}: edf_total={s.edf_total} out of (0, n_coef] range"
    )

    # (3) Every predict mode returns finite means that recover the trend.
    pred_df = pd.DataFrame({"x": x_grid})
    point = np.asarray(m.predict(pred_df))
    assert point.shape[0] == x_grid.shape[0]
    assert np.all(np.isfinite(point)), f"{family}: plain predict produced non-finite means"

    # SE-only interval and an observation interval must also run (these are the
    # modes that hard-failed in the bug report).
    se = m.predict(pred_df, interval=0.95)
    obs = m.predict(pred_df, interval=0.95, observation_interval=True)
    for label, frame in (("interval", se), ("observation_interval", obs)):
        mean_col = np.asarray(frame["mean"])
        assert np.all(np.isfinite(mean_col)), (
            f"{family}: predict({label}) produced non-finite means"
        )

    # (4) Truth recovery of the conditional mean: strong positive correlation
    #     and a small relative RMSE against the KNOWN exp(0.5 + 0.6x) trend.
    corr = float(np.corrcoef(point, true_mean)[0, 1])
    rel_rmse = float(np.sqrt(np.mean((point - true_mean) ** 2)) / np.mean(true_mean))
    assert corr > 0.9, f"{family}: mean corr with truth too low ({corr:.3f})"
    assert rel_rmse < 0.25, f"{family}: mean rel-RMSE vs truth too high ({rel_rmse:.3f})"


def test_gamma_dispersion_location_scale_is_predictable() -> None:
    """The headline #1119 repro: a Gamma location-scale fit carries a joint
    covariance / finite EDF and predicts the known mean trend."""
    df = _gamma_location_scale_frame()
    m = gamfit.fit(df, "y ~ s(x)", family="gamma", noise_formula="s(x)")

    x_grid = np.linspace(-1.5, 1.5, 40)
    true_mean = np.exp(0.5 + 0.6 * x_grid)
    _assert_joint_covariance_and_predictable(
        m, x_grid=x_grid, true_mean=true_mean, family="gamma"
    )


def test_negbin_dispersion_location_scale_is_predictable() -> None:
    """Same root cause, NB2 angle: an overdispersed-count location-scale fit
    must also assemble a joint covariance and predict its mean trend."""
    rng = np.random.default_rng(20260614)
    n = 1500
    x = rng.uniform(-2, 2, n)
    mean = np.exp(0.5 + 0.6 * x)
    theta = np.exp(0.7 + 0.4 * np.cos(2 * x))  # NB size (precision) varies
    lam = rng.gamma(shape=theta, scale=mean / theta)  # Gamma-Poisson mixture
    y = rng.poisson(lam).astype(float)
    df = pd.DataFrame({"y": y, "x": x})

    m = gamfit.fit(df, "y ~ s(x)", family="nb", noise_formula="s(x)")
    x_grid = np.linspace(-1.5, 1.5, 40)
    true_mean = np.exp(0.5 + 0.6 * x_grid)
    _assert_joint_covariance_and_predictable(
        m, x_grid=x_grid, true_mean=true_mean, family="nb"
    )


def test_tweedie_dispersion_location_scale_is_predictable() -> None:
    """Same root cause, Tweedie angle (compound-Poisson-Gamma, mass at 0)."""
    rng = np.random.default_rng(20260614)
    n = 1500
    x = rng.uniform(-2, 2, n)
    mean = np.exp(0.5 + 0.6 * x)
    # Compound Poisson-Gamma draw with power p=1.5 and unit-ish dispersion.
    p = 1.5
    phi = 0.6
    lam = mean ** (2 - p) / (phi * (2 - p))
    alpha = (2 - p) / (p - 1)
    beta = phi * (p - 1) * mean ** (p - 1)
    counts = rng.poisson(lam)
    y = np.array(
        [0.0 if c == 0 else float(rng.gamma(shape=alpha * c, scale=beta[i]))
         for i, c in enumerate(counts)]
    )
    df = pd.DataFrame({"y": y, "x": x})

    m = gamfit.fit(df, "y ~ s(x)", family="tweedie", noise_formula="s(x)")
    x_grid = np.linspace(-1.5, 1.5, 40)
    true_mean = np.exp(0.5 + 0.6 * x_grid)
    _assert_joint_covariance_and_predictable(
        m, x_grid=x_grid, true_mean=true_mean, family="tweedie"
    )


def test_beta_dispersion_location_scale_still_predictable() -> None:
    """Regression guard for the sibling that always worked: the fix must not
    disturb Beta (the coupled-cross-block member)."""
    rng = np.random.default_rng(7)
    n = 1500
    x = rng.uniform(-2, 2, n)
    mu = 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * x)))
    phi = np.exp(2.0 + 0.4 * np.cos(2 * x))
    a = mu * phi
    b = (1.0 - mu) * phi
    y = np.clip(rng.beta(a, b), 1e-4, 1.0 - 1e-4)
    df = pd.DataFrame({"y": y, "x": x})

    m = gamfit.fit(df, "y ~ s(x)", family="beta", noise_formula="s(x)")
    s = m.summary()
    n_coef = len(s.coefficients)
    assert s.covariance_n == n_coef
    assert s.edf_total is not None and np.isfinite(s.edf_total)
    point = np.asarray(m.predict(pd.DataFrame({"x": np.linspace(-1.5, 1.5, 40)})))
    assert np.all(np.isfinite(point))
    # Beta mean is on a logit scale; recovered mean must track the true logit
    # trend in rank (rising in x).
    true_mu = 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * np.linspace(-1.5, 1.5, 40))))
    assert float(np.corrcoef(point, true_mu)[0, 1]) > 0.9
