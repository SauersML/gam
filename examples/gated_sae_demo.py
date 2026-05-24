"""Gated SAE decoder demo."""

from __future__ import annotations

import numpy as np

import gamfit


def make_data(n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 1.0, n)
    x = np.c_[
        np.sin(2.0 * np.pi * t),
        np.cos(2.0 * np.pi * t),
        (t > 0.5).astype(float) * np.sin(6.0 * np.pi * t),
    ]
    return x + 0.03 * rng.standard_normal(x.shape)


def main() -> None:
    x = make_data()
    decoder = gamfit.GatedSAEDecoder(np.eye(3), np.eye(3))
    fit = gamfit.sae_manifold_fit(
        x,
        K=3,
        d_atom=2,
        atom_topology="circle",
        assignment="gated",
        n_iter=8,
        random_state=5,
    )
    decoded = decoder.decode(np.array([0.8, -0.2, 0.4]))
    print("assignment=gated")
    print("decoder sample:", np.round(decoded, 3))
    print(f"K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")


if __name__ == "__main__":
    main()
