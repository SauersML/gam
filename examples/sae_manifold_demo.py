"""SAE-manifold un-shatter demo.

This script is intentionally small and deterministic. It builds one curved
feature in a high-dimensional observation space, fits the gamfit SAE-manifold
configuration with a single periodic (circle) atom, and compares its
reconstruction R^2 against a fit using several Euclidean linear atoms. The
single curved atom should explain the data with substantially better R^2 than
the linear shards because it can wrap around the closed manifold.
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


def reconstruction_r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def linear_shard_baseline(z: np.ndarray, k: int = 10) -> dict[str, float]:
    """K linear (0-d) Euclidean atoms as a flat-shard SAE baseline."""
    fit = gamfit.sae_manifold_fit(
        z,
        K=k,
        d_atom=1,
        atom_topology="euclidean",
        assignment="topk",
        top_k=max(1, k // 2),
        isometry_weight=1.0,
        ard_per_atom=True,
        n_iter=8,
        learning_rate=0.04,
        random_state=0,
    )
    fitted = fit.reconstruct(z)
    return {"r2": reconstruction_r2(z, fitted), "K": float(k)}


def manifold_atom(z: np.ndarray) -> dict[str, float]:
    fit = gamfit.sae_manifold_fit(
        z,
        K=1,
        d_atom=2,
        atom_topology="circle",
        assignment="ibp",
        isometry_weight=1.0,
        ard_per_atom=True,
        n_iter=12,
        learning_rate=0.04,
        random_state=0,
    )
    fitted = fit.reconstruct(z)
    return {"r2": reconstruction_r2(z, fitted), "K": 1.0}


def main() -> None:
    z, theta = make_curved_feature()
    curved = manifold_atom(z)
    shards = linear_shard_baseline(z, k=10)
    delta = curved["r2"] - shards["r2"]
    print(f"synthetic Z shape: {z.shape}; theta[120]={theta[120]:.6f}")
    print(f"K=1 periodic manifold atom R^2: {curved['r2']:.3f}")
    print(f"K=10 linear atoms R^2: {shards['r2']:.3f}")
    print(f"delta R^2 (curved - linear shards): {delta:+.3f}")
    if delta > 0.0:
        print("verdict: curved atom wins (un-shatter)")
    else:
        print("verdict: linear shards match or exceed curved atom on this draw")


if __name__ == "__main__":
    main()
