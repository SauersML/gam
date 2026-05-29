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
   ``Z`` is the data alias, NOT a warm start; the docstring says so and this
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
        Z=X,
        n_atoms=k,
        atom_dim=1,
        atom_topology="circle",
        assignment="softmax",
        max_iter=4,
        random_state=0,
    )


def test_converged_latents_exposes_t_star_and_a_star() -> None:
    # Request 1: converged per-token coordinates t* and assignments a*.
    X = _synth()
    fit = _fit(X)
    latents = fit.converged_latents()

    assert set(latents) >= {"coords", "assignments", "logits", "fitted"}

    coords = latents["coords"]
    assert len(coords) == 3, "one on-manifold coordinate block per atom"
    for block in coords:
        block = np.asarray(block)
        assert block.shape == (X.shape[0], 1), "t* is per-token, per-atom (N, d_k)"
        assert np.isfinite(block).all()

    a_star = np.asarray(latents["assignments"], dtype=float)
    assert a_star.shape == (X.shape[0], 3), "a* is per-token, per-atom (N, K)"
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
    for k in range(3):
        t_k = np.asarray(fit.project(X, k), dtype=float)
        assert t_k.shape == (X.shape[0], 1)
        assert np.isfinite(t_k).all()
        # On training X the projection is the converged coordinate for atom k.
        np.testing.assert_array_equal(t_k, np.asarray(fit.coords[k], dtype=float))

    with pytest.raises(IndexError):
        fit.project(X, 3)


def test_warm_start_accepted_and_refines() -> None:
    # Request 2: a_init / t_init warm-start a bounded refinement and are distinct
    # from Z (the data). Seed from a prior fit's converged latents (the
    # encoder-predicts -> solver-refines loop), then refine for a few steps.
    X = _synth()
    teacher = _fit(X)
    lat = teacher.converged_latents()

    n, k = X.shape[0], 3
    a_init = np.asarray(lat["logits"], dtype=float)
    assert a_init.shape == (n, k)
    # t_init is (K, N, D_max); stack the per-atom (N, 1) coordinate blocks.
    t_init = np.stack([np.asarray(c, dtype=float) for c in lat["coords"]], axis=0)
    assert t_init.shape == (k, n, 1)

    warm = gamfit.sae_manifold_fit(
        Z=X,
        n_atoms=k,
        atom_dim=1,
        atom_topology="circle",
        assignment="softmax",
        n_iter=2,
        random_state=0,
        a_init=a_init,
        t_init=t_init,
    )
    assert np.asarray(warm.fitted, dtype=float).shape == X.shape
    assert np.isfinite(np.asarray(warm.fitted, dtype=float)).all()
    assert np.asarray(warm.assignments, dtype=float).shape == (n, k)


def test_warm_start_shapes_are_validated() -> None:
    # The warm-start kwargs are eagerly shape-checked (N, K) / (K, N, D_max), so
    # an encoder passing a mis-shaped prediction fails fast instead of silently
    # mis-seeding the solve.
    X = _synth()
    n, k = X.shape[0], 3
    with pytest.raises(ValueError):
        gamfit.sae_manifold_fit(
            Z=X, n_atoms=k, atom_dim=1, atom_topology="circle",
            assignment="softmax", n_iter=1, random_state=0,
            a_init=np.zeros((n, k + 1)),
        )
    with pytest.raises(ValueError):
        gamfit.sae_manifold_fit(
            Z=X, n_atoms=k, atom_dim=1, atom_topology="circle",
            assignment="softmax", n_iter=1, random_state=0,
            t_init=np.zeros((k, n + 1, 1)),
        )
