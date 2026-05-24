"""JumpReLU SAE-manifold demo."""

from __future__ import annotations

import numpy as np

import gamfit


def make_data(n: int = 100) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.linspace(-1.0, 1.0, n)
    x = np.c_[
        t,
        np.maximum(t - 0.15, 0.0),
        np.maximum(-t - 0.2, 0.0),
    ]
    return x + 0.03 * rng.standard_normal(x.shape)


def main() -> None:
    x = make_data()
    penalty = gamfit.JumpReLUPenalty(np.full(3, 0.5), weight=1.0, target="t")
    fit = gamfit.sae_manifold_fit(
        x,
        K=3,
        d_atom=2,
        atom_topology="euclidean",
        assignment="jumprelu",
        n_iter=8,
        random_state=9,
    )
    print("assignment=jumprelu")
    print("descriptor:", penalty.to_rust_descriptor()["kind"])
    print(f"K={len(fit.atoms)} r2={fit.reconstruction_r2:.3f}")


if __name__ == "__main__":
    main()
