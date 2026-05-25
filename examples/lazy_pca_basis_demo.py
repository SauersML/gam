"""Fit a GAM with lazily loaded PCA scores."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(7)
    n = 128
    k = 8
    z = np.linspace(0.0, 1.0, n)
    y = 0.4 * np.sin(8.0 * z) + rng.normal(scale=0.05, size=n)

    with tempfile.TemporaryDirectory() as tmp:
        scores_path = Path(tmp) / "pca_scores.npy"
        np.save(scores_path, rng.normal(size=(n, k)))
        model = gamfit.fit(
            {"y": y, "z": z},
            f'y ~ pca(z, lazy_path="{scores_path}", k={k}, chunk_size=32)',
            family="gaussian",
        )
        print(f"lazy PCA fit: {model}")


if __name__ == "__main__":
    main()
