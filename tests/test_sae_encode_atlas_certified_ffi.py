"""Parity + honesty tests for the Kantorovich-certified encode-atlas FFI (#1010).

The Rust ``EncodeAtlas`` (crates/gam-sae/src/encode.rs) carries a per-row
``h <= 1/2`` Newton-Kantorovich certificate for a frozen-dictionary encode. Until
now it was unreachable from Python: the amortized-encoder honesty gate
(``gamfit.distill.encode_with_fallback``) paid a *cold exact multi-start probe*
per row instead of reading the in-kernel certificate.

``ManifoldSAE.build_encode_atlas`` now wires it through:
``atlas.certified_encode(X, amplitudes, atom_index)`` returns the recovered
coordinates AND the per-row certificate flag. These tests pin the contract:

  1. The certificate is HONEST — every row it certifies encodes to the SAME
     reconstruction the existing Python exact path (:meth:`converged_latents` /
     :meth:`reconstruct`) produces (parity in ambient space, so the coordinate
     gauge / periodic wrap is irrelevant).
  2. The flag bookkeeping is consistent (``n_uncertified`` == count of ``False``)
     and uncertified rows are surfaced, never silently approximated.
"""

from __future__ import annotations

import numpy as np

import gamfit


def _planted_circle(n: int, p: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """On-manifold circle data ``x_i = z_i * (cos u + sin v)`` in R^p."""
    rng = np.random.default_rng(seed)
    # Two orthonormal ambient directions the circle lives in.
    a = rng.standard_normal(p)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(p)
    b -= b.dot(a) * a
    b /= np.linalg.norm(b)
    angles = rng.uniform(0.0, 1.0, size=n)  # periodic coordinate in [0, 1)
    amplitudes = rng.uniform(0.8, 1.2, size=n)
    circle = np.cos(2.0 * np.pi * angles)[:, None] * a[None, :] + np.sin(
        2.0 * np.pi * angles
    )[:, None] * b[None, :]
    x = amplitudes[:, None] * circle
    return np.ascontiguousarray(x), amplitudes


def _fit_circle(x: np.ndarray) -> gamfit._sae_manifold.ManifoldSAE:
    return gamfit.sae_manifold_fit(
        X=x,
        K=1,
        atom_basis="periodic",
        d_atom=1,
        assignment="ibp_map",
        n_iter=80,
        random_state=0,
    )


def test_certified_encode_matches_exact_path_on_certified_rows() -> None:
    """Every certified row reconstructs identically to the exact Python path."""
    x, _ = _planted_circle(n=200, p=6, seed=0)
    fit = _fit_circle(x)

    exact = fit.converged_latents(x)
    z = np.ascontiguousarray(np.asarray(exact["assignments"], dtype=float)[:, 0])

    atlas = fit.build_encode_atlas()
    assert int(atlas.k_atoms) == 1
    result = atlas.certified_encode(x, z, 0)

    coords = np.asarray(result["coords"], dtype=float)
    certified = np.asarray(result["certified"], dtype=bool)
    assert coords.shape == (x.shape[0], int(result["latent_dim"]))
    assert certified.shape == (x.shape[0],)

    # Honesty bookkeeping: the reported uncertified count is exactly the flags.
    assert int(result["n_uncertified"]) == int(np.count_nonzero(~certified))

    # The certificate must actually fire on clean on-manifold data (otherwise the
    # parity check below is vacuous).
    assert int(np.count_nonzero(certified)) > 0, (
        "the h<=1/2 certificate must fire for at least one clean on-manifold row"
    )

    # PARITY vs the existing Python exact path: on every CERTIFIED row, the
    # certified encode's reconstruction (z * Phi(t) * B) equals the exact fitted
    # reconstruction. Ambient space, so no coordinate-gauge / periodic-wrap issue.
    exact_recon = np.asarray(fit.reconstruct(x), dtype=float)
    atlas_recon = np.asarray(atlas.reconstruct(coords, z, 0), dtype=float)
    delta = np.max(np.abs(atlas_recon[certified] - exact_recon[certified]))
    assert delta < 5.0e-3, (
        f"certified rows must match the exact encode's reconstruction; max abs "
        f"delta = {delta:.3e}"
    )


def test_atom_index_out_of_range_is_a_clean_error() -> None:
    """Encoding against a non-existent atom raises, never panics."""
    x, z = _planted_circle(n=32, p=5, seed=1)
    fit = _fit_circle(x)
    atlas = fit.build_encode_atlas()
    try:
        atlas.certified_encode(x, np.ascontiguousarray(z), 7)
    except ValueError:
        return
    raise AssertionError("out-of-range atom_index must raise ValueError")
