"""Regression for issue #357 — amortized-encoder distillation surface.

Issue #357 asked the SAE joint Arrow-Schur solve (the natural *teacher* for an
amortized Manifold-SAE encoder) to expose three things. This test pins all
three on the public ``gamfit.sae_manifold_fit`` surface:

1. **Converged per-token latents.** ``ManifoldSAE.converged_latents()`` must
   return the per-atom on-manifold coordinates ``t*`` (one ``(N, d_k)`` block
   per atom) and the assignments / gate ``a*`` ``(N, K)`` the solver converged
   to — the supervision targets for the encoder, not just the decoder blocks.

2. **Warm-start + bounded refinement.** ``sae_manifold_fit`` must accept
   ``a_init`` (assignment logits, ``(N, K)``) and ``t_init`` (coordinates,
   ``(K, N, D_max)``) and run a bounded ``n_iter`` refinement of those seeds, so
   training can do "encoder predicts -> solver refines a few steps -> distill."
   ``X`` is the data matrix, NOT a warm start; the docstring says so and this
   test pins that ``a_init``/``t_init`` are separate, shape-validated kwargs.

3. **Standalone per-atom projection.** ``fit.project(x, atom_k)`` must map an
   ambient point to its on-manifold coordinate for atom ``k`` — the minimal
   teacher signal for the coordinate head — returning a ``(N, d_k)`` block.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
gamfit = pytest.importorskip("gamfit")


# #357 / #1512 (FIXED): the converged-latents path used to panic with
# `pyo3_runtime.PanicException: index out of bounds: the len is 3 but the index
# is 3` (`SaeManifoldRho.log_lambda_smooth` was not grown when a structure-search
# Birth/Fission grew K; `assemble_arrow_schur_inner` then indexed it out of
# bounds). The fix grows `log_lambda_smooth` alongside `log_ard` at every move
# that grows K, so the joint Arrow-Schur solve assembles on the grown dictionary.
#
# Because `sae_manifold_fit` runs an evidence-gated structure search on EVERY fit
# (`K` is the SEED size, not a fixed atom count — births/fissions may grow it and
# deaths/fusions may shrink it), the discovered dictionary size is read from the
# returned `len(fit.atoms)` rather than hardcoded. The genuine #357 contract is
# the SHAPE/identity of the exposed latents (one (N, d_k) block per discovered
# atom, a* = (N, K_discovered), equal to the fit's own attributes), which holds
# at any discovered K.


def _synth(n: int = 40, d: int = 6, k: int = 3, seed: int = 0) -> np.ndarray:
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
        n_iter=4,
        random_state=0,
    )


def test_converged_latents_exposes_t_star_and_a_star() -> None:
    # Request 1: converged per-token coordinates t* and assignments a*.
    X = _synth()
    fit = _fit(X)
    latents = fit.converged_latents()

    assert set(latents) >= {"coords", "assignments", "logits", "fitted"}

    # K is DISCOVERED by the structure search (the seed K=3 may grow); the
    # contract is one coordinate block per discovered atom, not a fixed count.
    K = len(fit.atoms)
    assert K >= 3, "structure search must keep at least the seed atoms"

    coords = latents["coords"]
    assert len(coords) == K, "one on-manifold coordinate block per atom"
    for block in coords:
        block = np.asarray(block)
        assert block.shape == (X.shape[0], 1), "t* is per-token, per-atom (N, d_k)"
        assert np.isfinite(block).all()

    a_star = np.asarray(latents["assignments"], dtype=float)
    assert a_star.shape == (X.shape[0], K), "a* is per-token, per-atom (N, K)"
    assert np.isfinite(a_star).all()

    # The exposed latents are the SAME objects the solver converged to, not a
    # re-derivation: they must equal the fit's own attributes on training X.
    np.testing.assert_array_equal(a_star, np.asarray(fit.assignments, dtype=float))
    for block, stored in zip(coords, fit.coords):
        np.testing.assert_array_equal(np.asarray(block), np.asarray(stored))


def test_standalone_per_atom_projection() -> None:
    # Request 3: project(x, atom_k) -> t for a single atom.
    X = _synth()
    fit = _fit(X)
    K = len(fit.atoms)
    for k in range(K):
        t_k = np.asarray(fit.project(X, k), dtype=float)
        assert t_k.shape == (X.shape[0], 1)
        assert np.isfinite(t_k).all()
        # On training X the projection is the converged coordinate for atom k.
        np.testing.assert_array_equal(t_k, np.asarray(fit.coords[k], dtype=float))

    with pytest.raises(IndexError):
        fit.project(X, K)


def test_warm_start_accepted_and_refines() -> None:
    # Request 2: a_init / t_init warm-start a bounded refinement and are distinct
    # from X (the data). Seed from a prior fit's converged latents (the
    # encoder-predicts -> solver-refines loop), then refine for a few steps.
    X = _synth()
    teacher = _fit(X)
    lat = teacher.converged_latents()

    # Warm-start the SAME seed K the teacher was seeded with; the warm refinement
    # consumes the teacher's converged latents projected back to that seed. The
    # teacher's DISCOVERED K (post structure search) sizes the latents we read.
    n = X.shape[0]
    k_seed = 3
    k_disc = len(teacher.atoms)
    a_init = np.asarray(lat["logits"], dtype=float)
    assert a_init.shape == (n, k_disc)
    # t_init is (K, N, D_max); stack the per-atom (N, 1) coordinate blocks.
    t_init = np.stack([np.asarray(c, dtype=float) for c in lat["coords"]], axis=0)
    assert t_init.shape == (k_disc, n, 1)

    # Seed the warm fit at the teacher's DISCOVERED K so the warm-start logits /
    # coords line up atom-for-atom with the seed dictionary.
    warm = gamfit.sae_manifold_fit(
        X=X,
        K=k_disc,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=2,
        random_state=0,
        a_init=a_init,
        t_init=t_init,
    )
    assert np.asarray(warm.fitted, dtype=float).shape == X.shape
    assert np.isfinite(np.asarray(warm.fitted, dtype=float)).all()
    # The warm fit runs its own structure search, so its discovered K need only
    # be at least the seed; assignments are (N, K_warm).
    k_warm = len(warm.atoms)
    assert k_warm >= k_seed
    assert np.asarray(warm.assignments, dtype=float).shape == (n, k_warm)


def test_warm_start_shapes_are_validated() -> None:
    # The warm-start kwargs are eagerly shape-checked (N, K) / (K, N, D_max), so
    # an encoder passing a mis-shaped prediction fails fast instead of silently
    # mis-seeding the solve.
    X = _synth()
    n, k = X.shape[0], 3
    with pytest.raises(ValueError):
        gamfit.sae_manifold_fit(
            X=X, K=k, d_atom=1, atom_topology="circle",
            assignment="softmax", n_iter=1, random_state=0,
            a_init=np.zeros((n, k + 1)),
        )
    with pytest.raises(ValueError):
        gamfit.sae_manifold_fit(
            X=X, K=k, d_atom=1, atom_topology="circle",
            assignment="softmax", n_iter=1, random_state=0,
            t_init=np.zeros((k, n + 1, 1)),
        )


def _oos_holdout(n_train: int = 48, n_test: int = 24, d: int = 6, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint train / held-out matrices drawn from the same circle mixture."""
    return _synth(n=n_train, d=d, k=k, seed=0), _synth(n=n_test, d=d, k=k, seed=101)


def test_oos_solve_returns_converged_latents_not_post_solve_reseed() -> None:
    # Regression for #1229. The frozen-decoder OOS solve used to RESEED the
    # assignment logits from projection residuals AFTER the joint Newton solve
    # had converged the coords + logits, then read assignments / fitted from the
    # reseeded logits. The returned routing was therefore the one-shot residual
    # heuristic, not the solver's converged solution that the OOS path advertises
    # (and that `fitted` was supposed to reflect).
    #
    # The SOUND oracle for "no post-solve reseed" is the softmax IDENTITY between
    # the two fields the OOS payload returns: the converged assignments must be
    # EXACTLY `softmax(logits / tau)` of the returned logits. The pre-fix bug
    # overwrote `term.assignment.logits` with a residual reseed AFTER the solve,
    # but `assignments()` reads those same logits — so post-reseed the identity
    # still held against the RESEEDED logits, not the converged ones. The bug is
    # therefore pinned by the SEPARATE coherence check below: `fitted` (read at
    # the converged decoder state) must be reproduced by decoding the returned
    # routing. A reseed desynced from the converged state breaks that.
    #
    # NOTE: an earlier version of this test asserted the returned latents are a
    # solver FIXED POINT (warm-restart barely moves them). That oracle is unsound
    # once the structure search legitimately grows a near-degenerate dictionary
    # (here a clean 3-circle mixture grows 3 confirmed euclidean_patch line atoms
    # alongside the 3 circles, log_e ≈ 4.6): the softmax routing between
    # near-collinear atoms sits on a flat landscape with no unique stable fixed
    # point, so a re-solve wanders even though no reseed occurred. The softmax /
    # reconstruct identities below hold at ANY discovered K and catch the actual
    # #1229 desync. The dedicated fixed-point oracle lives in
    # `test_sae_oos_returns_converged_latents_issue_1229.py`.
    X_train, X_test = _oos_holdout()
    fit = _fit(X_train, k=3)

    cold = fit.converged_latents(X_test)  # cold OOS solve (no warm start)
    a_star = np.asarray(cold["assignments"], dtype=float)
    logits = np.asarray(cold["logits"], dtype=float)
    coords = cold["coords"]
    # The OOS solve runs against the fit's DISCOVERED dictionary (the structure
    # search grew K on the train fit), so the held-out routing is (N_test, K).
    n, kk = X_test.shape[0], len(fit.atoms)
    assert a_star.shape == (n, kk)
    assert logits.shape == (n, kk)
    assert np.isfinite(a_star).all()
    assert np.isfinite(logits).all()
    assert t_init_shape_ok(coords, kk, n)

    # The returned assignments ARE softmax(logits / tau) of the returned logits:
    # one coherent (routing, logits) pair read at the SAME state, not a routing
    # detached from the logits the payload also exposes.
    tau = float(fit.tau)
    z = logits / tau
    z = z - z.max(axis=1, keepdims=True)
    softmax = np.exp(z)
    softmax /= softmax.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(a_star, softmax, rtol=0.0, atol=1e-9)

    # `fitted` is read at the SAME converged state as `assignments`: decoding the
    # returned routing through the frozen decoder reproduces the returned
    # reconstruction. A post-solve reseed inconsistent with the converged state
    # the `fitted` was read at would break this coherence.
    fitted = np.asarray(cold["fitted"], dtype=float)
    predicted = np.asarray(fit.reconstruct(X_test), dtype=float)
    np.testing.assert_allclose(predicted, fitted, rtol=0.0, atol=1e-9)


def t_init_shape_ok(coords: list, k: int, n: int) -> bool:
    """The per-atom coordinate blocks stack to the warm-start `(K, N, 1)` layout."""
    t_init = np.stack([np.asarray(c, dtype=float) for c in coords], axis=0)
    return bool(t_init.shape == (k, n, 1))
