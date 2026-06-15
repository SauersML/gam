"""Regression for #1124 (Python `Model.sample_replicates` path).

The generative observation model must draw Negative-Binomial replicate counts
with the **estimated** overdispersion ``theta_hat``, not the construction seed
``theta = 1.0``. With the seed, replicate counts carry ``Var = mu + mu^2``
instead of ``Var = mu + mu^2 / theta_hat`` — far too much overdispersion — and
posterior-predictive p-values are wrong.

The CLI ``gam generate`` path was fixed by unifying the dispersion picker, but
the Python ``sample_replicates`` path kept a *separate inline copy* in
``gam-pyffi`` whose NB arm returned the seed ``theta`` (``Some(*theta)``) rather
than ``likelihood_scale.negbin_theta()`` — so this exact bug stayed live in the
front-end the issue names. Both paths now route through the single canonical
``gam::generative::family_noise_parameter``.

This test fits overdispersed NB data with a true ``theta = 3`` (so the seed
``theta = 1`` is a grossly different, easily distinguished value), draws many
``sample_replicates``, and asserts the empirically implied overdispersion
``theta_implied = mu^2 / (var - mu)`` is far above the seed ``1`` and near the
truth — at every covariate level. Pre-fix it pins at ``~1.0``.
"""
from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def test_negative_binomial_sample_replicates_recovers_estimated_theta() -> None:
    rng = np.random.default_rng(5)
    n = 2000
    x = rng.uniform(0.0, 4.0, n)
    mu = np.exp(0.5 + 0.3 * x)
    true_theta = 3.0
    # NB2 via the gamma-Poisson mixture: E[y]=mu, Var=mu+mu^2/theta.
    y = rng.negative_binomial(true_theta, true_theta / (true_theta + mu))
    rows = [{"y": int(y[i]), "x": float(x[i])} for i in range(n)]

    model = gamfit.fit(rows, "y ~ s(x)", family="negative-binomial")

    # Replicate-sample at four covariate levels and estimate the implied
    # overdispersion from the per-row replicate moments.
    probe_x = [0.5, 1.5, 2.5, 3.5]
    probe_rows = [{"x": float(xx)} for xx in probe_x]
    reps = np.asarray(model.sample_replicates(probe_rows, 6000, seed=11), dtype=float)
    assert reps.shape == (6000, len(probe_x)), f"unexpected replicate shape {reps.shape}"

    for j, xx in enumerate(probe_x):
        col = reps[:, j]
        m = col.mean()
        v = col.var()
        # theta_implied = mu^2 / (Var - mu). With the seed bug Var≈mu+mu^2 so
        # theta_implied≈1; with the fix Var≈mu+mu^2/theta_hat so theta_implied≈3.
        assert v > m, (
            f"x={xx}: replicate variance {v:.3f} <= mean {m:.3f}; NB draws are not "
            f"overdispersed at all"
        )
        theta_implied = m * m / (v - m)
        assert theta_implied > 1.8, (
            f"x={xx}: NB sample_replicates implies theta={theta_implied:.2f} (mean={m:.2f}, "
            f"var={v:.2f}) — pins near the seed theta=1 instead of the fitted "
            f"theta_hat (~3). The Python generative path is drawing at the seed "
            f"overdispersion (#1124)."
        )
        # And not absurdly far above the truth either (sanity on the moment estimator).
        assert theta_implied < 6.0, (
            f"x={xx}: implied theta={theta_implied:.2f} unreasonably large for true 3"
        )
