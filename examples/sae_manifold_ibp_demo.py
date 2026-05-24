"""Softmax-vs-IBP SAE-manifold assignment demo."""

from __future__ import annotations

import numpy as np

import gamfit


def make_coexisting_manifolds(
    n_per: int = 260, p: int = 32, seed: int = 11
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_per, endpoint=False)
    line = np.linspace(-1.0, 1.0, n_per)
    circle_latent = np.stack(
        [np.sin(theta), np.cos(theta), np.sin(2.0 * theta), np.cos(2.0 * theta)], axis=1
    )
    line_latent = np.stack([line, line**2 - np.mean(line**2), np.sin(np.pi * line)], axis=1)
    mix_circle = rng.normal(size=(circle_latent.shape[1], p))
    mix_line = rng.normal(size=(line_latent.shape[1], p))
    z_circle = circle_latent @ mix_circle
    z_line = line_latent @ mix_line + 0.25
    z = np.concatenate([z_circle, z_line], axis=0)
    z += 0.04 * rng.normal(size=z.shape)
    z -= z.mean(axis=0, keepdims=True)
    labels = np.concatenate([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    return z, labels


def r2_score(z: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((z - fitted) ** 2))
    ss_tot = float(np.sum((z - z.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def summarize(name: str, fit: gamfit.SaeManifoldFitResult, z: np.ndarray) -> dict[str, float]:
    assignments = fit.assignments
    if name == "softmax":
        active_per_row = np.sum(assignments > 1e-3, axis=1)
    else:
        active_per_row = np.sum(assignments > 0.5, axis=1)
    binaryish = np.mean((assignments < 0.05) | (assignments > 0.95))
    return {
        "avg_active_atoms": float(np.mean(active_per_row)),
        "reml_score": float(fit.reml_score),
        "reconstruction_r2": r2_score(z, fit.fitted),
        "binaryish_fraction": float(binaryish),
        "mean_assignment_mass": float(np.mean(assignments)),
    }


def main() -> None:
    z, labels = make_coexisting_manifolds()
    shared = dict(
        n_atoms=4,
        atom_basis="duchon",
        atom_dim=2,
        smoothness=1.0,
        max_iter=10,
        learning_rate=0.04,
        random_state=19,
    )
    softmax = gamfit.sae_manifold_fit(
        z,
        assignment_prior="softmax",
        sparsity_strength=1.0,
        tau=0.5,
        **shared,
    )
    ibp = gamfit.sae_manifold_fit(
        z,
        assignment_prior="ibp_map",
        alpha="auto",
        tau=0.5,
        **shared,
    )
    print(f"synthetic Z shape: {z.shape}; class balance={np.bincount(labels).tolist()}")
    for name, fit in [("softmax", softmax), ("ibp_map", ibp)]:
        summary = summarize(name, fit, z)
        print(
            f"{name}: avg_active_atoms={summary['avg_active_atoms']:.3f}, "
            f"REML={summary['reml_score']:.3f}, "
            f"R2={summary['reconstruction_r2']:.4f}, "
            f"binaryish={summary['binaryish_fraction']:.3f}, "
            f"mean_assignment_mass={summary['mean_assignment_mass']:.3f}"
        )


if __name__ == "__main__":
    main()
