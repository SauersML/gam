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
        "grad_t_norm_scaled",
        "converged",
    ):
        assert key in r, f"missing diagnostic key {key!r} in result payload"

    response_r2 = float(r["response_r2"])
    response_residual_norm = float(r["response_residual_norm"])
    latent_t_std = float(r["latent_t_std"])
    grad_t_norm = float(r["grad_t_norm"])
    grad_t_norm_scaled = float(r["grad_t_norm_scaled"])

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

    # The scale-aware (relative) latent-gradient stationarity measure is a
    # finite, non-negative scalar, and `converged` is decided from IT (not the
    # raw `grad_t_norm`) -- this is the #879 fix: the profiled Gaussian REML
    # objective leaves the raw gradient O(n) near interpolation, so the absolute
    # test mis-flags near-perfect fits as non-converged.
    assert np.isfinite(grad_t_norm_scaled)
    assert grad_t_norm_scaled >= 0.0

    # `converged` is an honest boolean flag, equal to the scale-aware test.
    converged = bool(r["converged"])
    assert isinstance(converged, bool)
    grad_tol = 1.0e-8  # the default `grad_tol` of gaussian_reml_optimize_latent
    assert converged == (grad_t_norm_scaled <= grad_tol)


def test_latent_optimize_convergence_is_shift_invariant_and_honest_954():
    """#954 + #879: ``converged`` is decided by the SHIFT-INVARIANT relative
    gradient measure ``‖∇ₜf(t̂)‖ / max(‖∇ₜf(t₀)‖, 1)`` -- the objective magnitude
    never enters -- and it honestly reflects latent *stationarity*, which is a
    distinct quantity from decoder *reconstruction quality* (``response_r2``).

    This replaces an earlier test that asserted ``converged=True`` for this exact
    scenario "being near-stationary at an excellent optimum". That premise is
    false: the latent here is NOT stationary -- the REML objective is still
    strictly decreasing (see the longer run below), so the large raw gradient is
    genuine non-stationarity, not merely profiled-scale stiffness. The retired
    ``‖∇ₜf‖·‖t‖_typ/max(|f|,1)`` measure also yields ``≈ grad/|f| ≈ 9 ≫ grad_tol``
    here, i.e. ``converged=False`` too -- the old test never actually passed. The
    #954 fix makes the decision *shift-invariant* (independent of ``|f|``); #879's
    own first test (above) already separates "good reconstruction" from
    "converged", and this test pins that separation with a hard, self-justifying
    proof of non-stationarity.
    """
    rng = np.random.RandomState(7)
    n = 30
    t_true = np.sort(rng.uniform(-1.0, 1.0, n))
    # A smooth 3-output decoder image of a 1-D latent, with negligible noise so
    # the Duchon basis can near-interpolate (R² -> 1).
    Y = np.c_[
        np.sin(2.0 * t_true),
        t_true**2,
        np.cos(1.5 * t_true),
    ] + 1e-4 * rng.randn(n, 3)
    C = np.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    P = np.eye(16)

    def run(max_iter):
        return gamfit.gaussian_reml_optimize_latent(
            y=Y.astype(float),
            n_obs=n,
            latent_dim=1,
            centers=C,
            penalty=P,
            m=2,
            manifold="euclidean",
            basis_kind="duchon",
            max_iter=max_iter,
            init="caller",
            t=t_true.astype(float),
            seed=0,
        )

    r = run(300)
    grad_t_norm = float(r["grad_t_norm"])
    grad_t_norm_init = float(r["grad_t_norm_init"])
    grad_t_norm_scaled = float(r["grad_t_norm_scaled"])
    response_r2 = float(r["response_r2"])
    objective_value = float(r["objective_value"])
    converged = bool(r["converged"])
    grad_tol = 1.0e-8  # the default `grad_tol` of gaussian_reml_optimize_latent

    # The decoder near-interpolates the response (reconstruction is excellent).
    assert response_r2 > 0.99, f"expected near-perfect fit, got response_r2={response_r2}"

    # (1) #954: the convergence measure is the relative gradient computed PURELY
    # from gradient norms -- `grad_t_norm / max(grad_t_norm_init, 1)`. The
    # objective value (which an additive shift f -> f + C would move) does not
    # enter. The retired formula divided by `max(|f|, 1)` and was thus
    # shift-non-invariant.
    expected = grad_t_norm / max(grad_t_norm_init, 1.0)
    assert grad_t_norm_scaled == pytest.approx(expected, rel=1e-9, abs=1e-18), (
        "grad_t_norm_scaled must be the shift-invariant relative gradient "
        f"grad_t_norm/max(grad_t_norm_init,1)={expected}, got {grad_t_norm_scaled}"
    )
    # Dividing by max(., 1) >= 1 can only shrink the raw gradient norm.
    assert grad_t_norm_scaled <= grad_t_norm + 1e-12
    # `converged` is exactly the relative-gradient test, decoupled from |f|.
    assert converged == (grad_t_norm_scaled <= grad_tol)

    # (2) #879 separation: near-perfect RECONSTRUCTION does not imply latent
    # STATIONARITY. Run 4x longer; the REML objective strictly decreases (so the
    # 300-iter latent was provably non-stationary) while response_r2 stays ~1.
    # `converged` honestly tracks stationarity -- here False -- not the decoder's
    # reconstruction quality. A caller distinguishes the two via `response_r2`.
    r_long = run(1200)
    assert float(r_long["objective_value"]) < objective_value - 1e-6, (
        "the REML objective must strictly decrease with more iterations, proving "
        f"the 300-iter latent was non-stationary (f@300={objective_value}, "
        f"f@1200={float(r_long['objective_value'])})"
    )
    assert float(r_long["response_r2"]) > 0.99
    assert converged is False, (
        "a still-descending (non-stationary) latent must not be reported "
        "converged, even though its decoder reconstruction is near-perfect: "
        f"grad_t_norm={grad_t_norm}, grad_t_norm_scaled={grad_t_norm_scaled}, "
        f"response_r2={response_r2}"
    )
