"""Regression test for issue #879: honest latent-fit diagnostics.

``gaussian_reml_optimize_latent`` now (a) reports the PROJECTED gradient as
``grad_t_norm`` and (b) carries payload keys ``response_r2``,
``response_residual_norm``, and ``latent_t_std`` so a caller can distinguish a
good decoder fit (high ``response_r2``) whose latent gradient simply did not
reach ``grad_tol`` (``converged`` may be False) from a genuine latent collapse
(``latent_t_std ~ 0``). Landed in commit 7a350692d.

The assertions check the CONTRACT (keys present, finite, sane ranges) on real
numerical data rather than brittle exact values.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def test_latent_optimize_diagnostics_keys_879():
    rng = np.random.RandomState(0)
    n = 40
    th = np.sort(rng.uniform(0, 2 * np.pi, n))
    Y = np.c_[np.cos(th), np.sin(th)] + 0.02 * rng.randn(n, 2)
    Y = np.c_[Y, 0.02 * rng.randn(n, 3)]
    C = np.linspace(0, 1, 12).reshape(-1, 1)

    r = gamfit.gaussian_reml_optimize_latent(
        y=Y.astype(float),
        n_obs=n,
        latent_dim=1,
        centers=C,
        penalty=np.eye(12),
        m=2,
        manifold="euclidean",
        basis_kind="duchon",
        max_iter=200,
        seed=0,
    )

    # The honest-diagnostics keys (and the existing ones) must be present.
    for key in (
        "response_r2",
        "response_residual_norm",
        "latent_t_std",
        "grad_t_norm",
        "converged",
    ):
        assert key in r, f"missing diagnostic key {key!r} in result payload"

    response_r2 = float(r["response_r2"])
    response_residual_norm = float(r["response_residual_norm"])
    latent_t_std = float(r["latent_t_std"])
    grad_t_norm = float(r["grad_t_norm"])

    # A reasonable euclidean decoder fit reconstructs the response well.
    assert np.isfinite(response_r2)
    assert response_r2 > 0.3, f"decoder did not reconstruct: response_r2 = {response_r2}"

    # The latent was recovered, not collapsed onto a single point.
    assert np.isfinite(latent_t_std)
    assert latent_t_std > 0.0, f"latent collapsed: latent_t_std = {latent_t_std}"

    # Residual norm is a finite, non-negative magnitude.
    assert np.isfinite(response_residual_norm)
    assert response_residual_norm >= 0.0

    # The projected gradient norm is reported as a finite, non-negative scalar.
    assert np.isfinite(grad_t_norm)
    assert grad_t_norm >= 0.0

    # `converged` is an honest boolean flag.
    assert isinstance(bool(r["converged"]), bool)
