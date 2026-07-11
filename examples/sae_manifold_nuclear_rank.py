"""Select a low decoder embedding rank with the SAE nuclear-norm penalty.

The planted torus is observed through an ambient 8D map whose decoder loading
has only a few dominant directions. Fitting with ``nuclear_norm_weight`` shrinks
the singular spectrum of each per-atom decoder matrix, making the selected
embedding rank visible in the printed singular values and spectrum plot.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import gamfit


def _effective_rank(values: np.ndarray, relative_cutoff: float = 0.08) -> int:
    if values.size == 0 or values[0] <= 0.0:
        return 0
    return int(np.sum(values >= relative_cutoff * values[0]))


def main() -> None:
    rng = np.random.default_rng(672)
    n = 260
    u = rng.uniform(0.0, 2.0 * np.pi, size=n)
    v = rng.uniform(0.0, 2.0 * np.pi, size=n)
    torus_features = np.c_[
        np.cos(u),
        np.sin(u),
        np.cos(v),
        np.sin(v),
        0.35 * np.cos(u + v),
        0.35 * np.sin(u - v),
    ]
    low_rank_loading = rng.normal(size=(6, 2)) @ rng.normal(size=(2, 8))
    x = torus_features @ low_rank_loading
    x += 0.04 * rng.standard_normal(x.shape)

    baseline = gamfit.sae_manifold_fit(
        x,
        K=1,
        d_atom=2,
        atom_topology="torus",
        assignment="ordered_beta_bernoulli",
        nuclear_norm_weight=0.0,
        n_iter=18,
        random_state=672,
    )
    ranked = gamfit.sae_manifold_fit(
        x,
        K=1,
        d_atom=2,
        atom_topology="torus",
        assignment="ordered_beta_bernoulli",
        nuclear_norm_weight=2.5,
        nuclear_norm_max_rank=4,
        n_iter=18,
        random_state=672,
    )

    spectra = {
        "baseline": np.linalg.svd(baseline.decoder_blocks[0], compute_uv=False),
        "nuclear norm": np.linalg.svd(ranked.decoder_blocks[0], compute_uv=False),
    }
    print("Nuclear-norm decoder rank selection")
    for label, values in spectra.items():
        print(
            f"{label}: r2="
            f"{(baseline if label == 'baseline' else ranked).reconstruction_r2:.3f} "
            f"effective_rank={_effective_rank(values)} "
            f"singular_values={np.round(values[:8], 3)}"
        )
    print("enabled primitives:", ranked.summary()["primitives"])

    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    for label, values in spectra.items():
        ax.plot(np.arange(1, len(values) + 1), values, marker="o", label=label)
    ax.set_title("decoder singular spectrum")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    main()
