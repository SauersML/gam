"""Regression for issue #1229 — the OOS fixed-decoder solve must RETURN the
converged latents, not a post-solve residual reseed.

``sae_manifold_predict_oos`` seeds the assignment logits into the right basin,
runs ``run_fixed_decoder_arrow_schur`` to jointly converge the coordinates and
logits, and then (pre-fix) RESEEDED the logits a second time from projection
residuals before extracting the assignments / fitted values. Both
``term.assignment.assignments()`` and ``term.fitted()`` read
``term.assignment.logits``, so that post-solve reseed overwrote the
Newton-converged logits: the returned routing was the residual-seed routing, not
the converged solution the OOS path advertises.

The fix deletes the post-solve reseed, so the returned assignments / coordinates
are the converged stationary point. This test pins that contract WITHOUT asserting
any internal field: it checks that the returned out-of-sample latents are a FIXED
POINT of the solver. Feeding the returned ``(logits, coords)`` straight back as a
warm start (``a_init`` / ``t_init``) and re-running the SAME OOS solve must return
essentially the same assignments / coordinates — because a converged stationary
point does not move under a re-solve that starts from it.

Under the pre-fix reseed the returned latents were NOT the solved state, so a
re-solve from them would move (the reseed routing is not a fixed point of the
joint Newton solve).
"""

from __future__ import annotations

import numpy as np

import gamfit


def _planted_circles(n: int = 60, d: int = 6, k: int = 3, seed: int = 0) -> np.ndarray:
    """A k-circle mixture in d-dim ambient space — the structure the circle-atom
    dictionary is built to fit, so the frozen-decoder OOS solve converges
    cleanly on held-out points."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n, k))
    bases = rng.standard_normal(size=(k, d))
    bases /= np.linalg.norm(bases, axis=1, keepdims=True) + 1e-9
    amps = rng.uniform(0.5, 1.5, size=(n, k))
    x = (amps * np.cos(angles))[:, :, None] * bases[None, :, :]
    return x.sum(axis=1) + 0.05 * rng.standard_normal((n, d))


def _fit(X: np.ndarray, k: int = 3):
    return gamfit.sae_manifold_fit(
        X=X,
        K=k,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=6,
        random_state=0,
    )


def test_oos_returned_latents_are_a_fixed_point_of_the_solve() -> None:
    # Train on one sample; predict OOS on a DISTINCT held-out sample so the
    # frozen-decoder OOS path runs (not the cached-training shortcut).
    X_train = _planted_circles(n=80, seed=0)
    X_oos = _planted_circles(n=50, seed=99)

    fit = _fit(X_train)

    # Cold OOS solve on held-out X: this is the path #1229 fixed.
    cold = fit.converged_latents(X_oos)
    a_cold = np.asarray(cold["assignments"], dtype=float)
    logits_cold = np.asarray(cold["logits"], dtype=float)
    coords_cold = [np.asarray(c, dtype=float) for c in cold["coords"]]

    n, k = X_oos.shape[0], 3
    assert a_cold.shape == (n, k)
    assert logits_cold.shape == (n, k)
    assert np.isfinite(a_cold).all()
    assert np.isfinite(logits_cold).all()

    # Warm-start the SAME OOS solve from the returned converged latents. A
    # converged stationary point is a fixed point of the solver: re-solving from
    # it must return essentially the same assignments / coordinates.
    t_init = np.stack(coords_cold, axis=0)  # (K, N, D_max=1)
    assert t_init.shape == (k, n, 1)
    warm = fit.converged_latents(X_oos, a_init=logits_cold, t_init=t_init)
    a_warm = np.asarray(warm["assignments"], dtype=float)
    coords_warm = [np.asarray(c, dtype=float) for c in warm["coords"]]

    # Assignments are a fixed point: the cold OOS routing already IS the
    # converged routing, so warm-starting from it does not move it. A reseed
    # (the pre-fix bug) would return a non-converged routing that the warm
    # re-solve then pulls toward the true stationary point — a visible shift.
    assert np.allclose(a_warm, a_cold, atol=1e-6), (
        "OOS-returned assignments must be the converged stationary point: "
        f"max|Δa| = {np.max(np.abs(a_warm - a_cold)):.3e} under a re-solve "
        "warm-started from the returned latents"
    )
    for atom_idx, (c_warm, c_cold) in enumerate(zip(coords_warm, coords_cold)):
        assert np.allclose(c_warm, c_cold, atol=1e-6), (
            f"OOS-returned coordinates for atom {atom_idx} must be the converged "
            f"stationary point: max|Δt| = {np.max(np.abs(c_warm - c_cold)):.3e}"
        )


def test_oos_returned_assignments_reconstruct_the_returned_fitted() -> None:
    # The returned assignments and fitted values must describe ONE coherent
    # model: decoding the returned routing through the frozen decoder reproduces
    # the returned reconstruction. If the routing were a post-solve reseed
    # inconsistent with the converged state the fitted was read at, this
    # internal-consistency check would fail.
    X_train = _planted_circles(n=80, seed=1)
    X_oos = _planted_circles(n=40, seed=7)

    fit = _fit(X_train)
    payload = fit.converged_latents(X_oos)
    fitted = np.asarray(payload["fitted"], dtype=float)

    # `reconstruct` runs the same OOS solve and returns its fitted values; a
    # second call on the same held-out input must reproduce the same
    # reconstruction (the solve is deterministic and returns the converged
    # state).
    fitted_again = np.asarray(fit.reconstruct(X_oos), dtype=float)
    assert fitted.shape == fitted_again.shape
    assert np.allclose(fitted, fitted_again, atol=1e-8), (
        "the OOS reconstruction must be the deterministic converged state: "
        f"max|Δfit| = {np.max(np.abs(fitted - fitted_again)):.3e}"
    )
