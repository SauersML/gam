"""Ground-truth synthetic-quality oracles for the manifold SAE training path.

These tests plant a *known* generative structure (disjoint periodic atoms
with one-hot routing) and require the public ``gamfit.sae_manifold_fit``
path to recover it: reconstruction, planted atom routing, and the absence
of leakage onto the inactive atom. Unlike looser reconstruction checks,
this pins the dictionary-quality claim — a manifold SAE that cannot fit two
disjoint periodic atoms drawn from its own basis family is not accurate
enough to be called a dictionary learner (issue #629).
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _periodic_phi(t: np.ndarray, n_harmonics: int = 1) -> np.ndarray:
    """Periodic harmonic design using the engine's own convention so the
    oracle and the fit speak the same basis (``[1, sin, cos, ...]``)."""
    from gamfit._binding import rust_module

    phi, _jet, _pen = rust_module().basis_with_jet(
        "periodic",
        np.asarray(t, float).reshape(-1, 1),
        {"n_harmonics": n_harmonics},
    )
    return np.asarray(phi)


def _make_disjoint_periodic_oracle(n: int = 48, p: int = 8, seed: int = 0):
    """Two periodic atoms living in disjoint output blocks with balanced,
    shuffled one-hot routing. Atom 0 drives columns ``[0:p/2]``, atom 1
    drives ``[p/2:p]``; the planted harmonic coefficients skip the constant
    column so each block has ~zero mean."""
    rng = np.random.default_rng(seed)
    half = p // 2
    assign = np.zeros(n, dtype=int)
    assign[n // 2:] = 1
    rng.shuffle(assign)
    t = rng.uniform(0.0, 1.0, n)
    phi = _periodic_phi(t, 1)  # (n, 3): [1, sin, cos]
    m = phi.shape[1]
    blocks = [np.zeros((m, p)) for _ in range(2)]
    blocks[0][1, 0] = 1.5
    blocks[0][2, 1] = -1.2
    blocks[0][1, 2] = 0.8
    blocks[0][2, 3] = 1.0
    blocks[1][1, half + 0] = 1.3
    blocks[1][2, half + 1] = 0.9
    blocks[1][1, half + 2] = -1.1
    blocks[1][2, half + 3] = 1.4
    x = np.zeros((n, p))
    for i in range(n):
        x[i] = phi[i] @ blocks[assign[i]]
    return x, assign


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def test_fit_learns_disjoint_periodic_atoms_without_inactive_leakage():
    x, assign = _make_disjoint_periodic_oracle()

    fit = gamfit.sae_manifold_fit(
        Z=x,
        n_atoms=2,
        atom_basis="periodic",
        atom_dim=1,
        assignment="softmax",
        top_k=1,
        max_iter=50,
        random_state=0,
    )

    fitted = np.asarray(fit.fitted)
    r2 = _r2(x, fitted)

    a = np.asarray(fit.assignments)
    assert a.shape == (x.shape[0], 2)
    hard = a.argmax(axis=1)

    # Match the learned hard routing to the planted one-hot up to the trivial
    # atom-label permutation (the model has no preferred atom ordering).
    acc_direct = float((hard == assign).mean())
    acc_swap = float((hard == (1 - assign)).mean())
    routing_acc = max(acc_direct, acc_swap)
    swapped = acc_swap > acc_direct
    matched = (1 - assign) if swapped else assign

    # Inactive leakage: the soft assignment mass placed on the atom that did
    # NOT generate each row. With top_k=1 and clean disjoint atoms this should
    # be ~0; we require it small in the mean.
    inactive_mass = np.array(
        [a[i, 1 - matched[i]] for i in range(x.shape[0])]
    )
    leakage = float(inactive_mass.mean())

    assert r2 >= 0.95, (
        f"manifold SAE failed to reconstruct two disjoint periodic atoms: "
        f"R^2 = {r2:.4f} (need >= 0.95). reconstruction_r2 attr = "
        f"{getattr(fit, 'reconstruction_r2', float('nan'))!r}."
    )
    assert routing_acc >= 0.90, (
        f"learned routing does not match the planted one-hot assignment: "
        f"accuracy (up to permutation) = {routing_acc:.4f} (need >= 0.90)."
    )
    assert leakage <= 0.02, (
        f"too much assignment mass leaks onto the inactive atom: "
        f"mean inactive weight = {leakage:.4f} (need <= 0.02)."
    )
