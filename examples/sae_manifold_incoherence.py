"""Use decoder incoherence to separate co-active superposed manifold atoms.

The data are the sum of two simultaneously active curved atoms in overlapping
ambient channels. The script fits the same IBP SAE twice, once with the
cross-atom decoder incoherence penalty disabled and once with it enabled, then
reports the normalized decoder cross-Gram score.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import gamfit


def _decoder_overlap(decoder_a: np.ndarray, decoder_b: np.ndarray) -> float:
    numerator = np.linalg.norm(decoder_a.T @ decoder_b, ord="fro")
    denominator = np.linalg.norm(decoder_a, ord="fro") * np.linalg.norm(decoder_b, ord="fro")
    return float(numerator / max(denominator, np.finfo(float).eps))


def _fit(x: np.ndarray, weight: float, seed: int) -> gamfit.ManifoldSAE:
    return gamfit.sae_manifold_fit(
        x,
        K=2,
        d_atom=1,
        atom_topology="circle",
        assignment="ordered_beta_bernoulli",
        decoder_incoherence_weight=weight,
        sparsity_weight=0.35,
        n_iter=18,
        random_state=seed,
    )


def main() -> None:
    rng = np.random.default_rng(671)
    n = 260
    theta_a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta_b = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + 0.75

    atom_a = np.c_[
        np.cos(theta_a),
        np.sin(theta_a),
        0.20 * np.cos(2.0 * theta_a),
        0.20 * np.sin(2.0 * theta_a),
        np.zeros(n),
        np.zeros(n),
    ]
    atom_b = np.c_[
        0.45 * np.cos(theta_b),
        -0.25 * np.sin(theta_b),
        np.cos(2.0 * theta_b),
        np.sin(2.0 * theta_b),
        0.40 * np.cos(theta_b),
        0.40 * np.sin(theta_b),
    ]
    gates = np.ones((n, 2), dtype=float)
    gates[:60, 1] = 0.0
    gates[-60:, 0] = 0.0
    x = gates[:, [0]] * atom_a + gates[:, [1]] * atom_b
    x += 0.035 * rng.standard_normal(x.shape)

    plain = _fit(x, weight=0.0, seed=671)
    incoherent = _fit(x, weight=8.0, seed=671)

    plain_overlap = _decoder_overlap(plain.decoder_blocks[0], plain.decoder_blocks[1])
    incoherent_overlap = _decoder_overlap(incoherent.decoder_blocks[0], incoherent.decoder_blocks[1])
    print("Decoder incoherence penalty demo")
    print(f"without penalty: r2={plain.reconstruction_r2:.3f} overlap={plain_overlap:.3f}")
    print(
        f"with penalty:    r2={incoherent.reconstruction_r2:.3f} "
        f"overlap={incoherent_overlap:.3f}"
    )
    print("enabled primitives:", incoherent.summary()["primitives"])

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), constrained_layout=True)
    axes[0].plot(gates[:, 0], label="atom 0 planted")
    axes[0].plot(gates[:, 1], label="atom 1 planted")
    axes[0].set_title("planted co-activation")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="best")
    axes[1].plot(plain.assignments)
    axes[1].set_title(f"no incoherence ({plain_overlap:.2f})")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].plot(incoherent.assignments)
    axes[2].set_title(f"incoherence weight 8 ({incoherent_overlap:.2f})")
    axes[2].set_ylim(-0.05, 1.05)
    plt.show()


if __name__ == "__main__":
    main()
