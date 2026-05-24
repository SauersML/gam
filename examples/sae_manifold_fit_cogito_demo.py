"""Cogito-style manifold SAE fit demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gamfit


def load_cogito_or_synthetic() -> np.ndarray:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "Manifold-SAE" / "X_L40.npy",
        here.parents[2] / "Manifold-SAE" / "data" / "X_L40.npy",
        here.parents[1] / "X_L40.npy",
    ]
    for path in candidates:
        if path.exists():
            x = np.load(path)
            return np.asarray(x, dtype=float)
    return synthetic_circle()


def synthetic_circle(n: int = 768, p: int = 96, seed: int = 40) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    latent = np.stack(
        [
            np.sin(theta),
            np.cos(theta),
            np.sin(2.0 * theta + 0.3),
            np.cos(2.0 * theta - 0.2),
            0.35 * np.sin(3.0 * theta),
            0.35 * np.cos(3.0 * theta),
        ],
        axis=1,
    )
    mixing = rng.normal(size=(latent.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    x = latent @ mixing + 0.035 * rng.normal(size=(n, p))
    return x - x.mean(axis=0, keepdims=True)


def r2_score(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> None:
    x = load_cogito_or_synthetic()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    schedule = gamfit.GumbelTemperatureSchedule(
        tau_start=1.0,
        tau_min=0.25,
        decay="geometric",
        rate=0.94,
    )
    model = gamfit.sae_manifold_fit(
        x,
        K=64,
        d_atom=2,
        atom_topology="circle",
        assignment="ibp",
        schedule=schedule,
        isometry_weight=1.0,
        ard_per_atom=True,
        n_iter=50,
        random_state=40,
    )
    fitted = model.reconstruct(x)
    active = model.per_atom_active_set(x)
    latents = model.per_atom_latent_for(x)
    anchors = model.get_anchors()
    summary = model.summary()
    print(f"X shape: {x.shape}")
    print(f"K={summary['K']} d_atom={summary['d_atom']} topology={summary['atom_topology']}")
    print(f"reconstruction R2: {r2_score(x, fitted):.4f}")
    print(f"avg active atoms/row: {np.mean(np.sum(active, axis=1)):.3f}")
    print(f"mean assignment mass: {np.mean(model.assignments):.4f}")
    print(f"latent blocks: {len(latents)}; first latent shape: {latents[0].shape}")
    print(f"anchor blocks: {len(anchors)}; first anchor shape: {anchors[0].shape}")
    print(f"visualization outputs: assignments={model.assignments.shape}, fitted={fitted.shape}")


if __name__ == "__main__":
    main()
