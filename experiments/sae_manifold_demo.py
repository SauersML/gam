"""SAE-manifold un-shatter demo.

This script is intentionally small and deterministic. It builds one curved
feature in a high-dimensional observation space, fits the gamfit SAE-manifold
configuration, and compares the REML evidence of a single curved atom against
ten linear shards.

Intended target path from the task:
``/Users/user/Manifold-SAE/experiments/sae_manifold_demo.py``.
The current sandbox only permits writes under ``/Users/user/gam``, so this
repo-local copy carries the same demo content.
"""

from __future__ import annotations

import numpy as np

import gamfit


def make_curved_feature(n: int = 480, p: int = 32, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    harmonics = np.stack(
        [
            np.sin(theta),
            np.cos(theta),
            np.sin(2.0 * theta),
            np.cos(2.0 * theta),
            0.35 * np.sin(3.0 * theta + 0.2),
            0.35 * np.cos(3.0 * theta - 0.1),
        ],
        axis=1,
    )
    mixing = rng.normal(size=(harmonics.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harmonics @ mixing + 0.035 * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z, theta


def linear_shard_baseline(z: np.ndarray, k: int = 10) -> dict[str, float]:
    """K linear atoms as a zero-dimensional SAE baseline."""
    fit = gamfit.sae_manifold_fit(
        z,
        n_atoms=k,
        atom_basis="duchon",
        atom_dim=0,
        sparsity_strength=1.0,
        smoothness=1.0,
        max_iter=8,
        learning_rate=0.04,
    )
    return {"reml_score": fit.reml_score, "chosen_k": float(fit.chosen_k)}


def manifold_atom(z: np.ndarray) -> dict[str, float]:
    fit = gamfit.sae_manifold_fit(
        z,
        n_atoms=1,
        atom_basis="periodic",
        atom_dim=1,
        sparsity_strength=1.0,
        smoothness="auto",
        max_iter=12,
        learning_rate=0.04,
    )
    return {"reml_score": fit.reml_score, "chosen_k": float(fit.chosen_k)}


def main() -> None:
    z, theta = make_curved_feature()
    curved = manifold_atom(z)
    shards = linear_shard_baseline(z, k=10)
    delta = curved["reml_score"] - shards["reml_score"]
    comparison = gamfit.compare_models(
        [curved, shards],
        names=["K=1 periodic manifold atom", "K=10 linear atoms"],
    )
    print(f"synthetic Z shape: {z.shape}; theta[120]={theta[120]:.6f}")
    print(f"K=1 periodic manifold atom REML: {curved['reml_score']:.3f}")
    print(f"K=10 linear atoms REML: {shards['reml_score']:.3f}")
    print(f"delta REML (curved - linear shards): {delta:.3f}")
    print(comparison["evidence_summary"])


if __name__ == "__main__":
    main()
