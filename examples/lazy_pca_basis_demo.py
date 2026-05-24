"""Lazy PCA basis demo."""

from __future__ import annotations

import resource
import sys
import tempfile
from pathlib import Path

import numpy as np

import gamfit


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def main() -> None:
    n = 50_000
    k = 64
    rng = np.random.default_rng(7)
    baseline = rss_mb()

    with tempfile.TemporaryDirectory() as tmp:
        scores_path = Path(tmp) / "pca_scores.npy"
        scores = np.lib.format.open_memmap(
            scores_path,
            mode="w+",
            dtype=np.float64,
            shape=(n, k),
        )
        for start in range(0, n, 4096):
            end = min(start + 4096, n)
            scores[start:end] = rng.normal(size=(end - start, k))
        scores.flush()
        del scores

        z = np.linspace(0.0, 1.0, n)
        y = 0.4 * np.sin(8.0 * z) + rng.normal(scale=0.05, size=n)
        data = {"y": y, "z": z}
        latent = gamfit.LatentCoord(n=n, d=1, init=z[:, None], aux_prior={"u": z[:, None]})
        model = gamfit.fit(
            data,
            f'y ~ pca(z, lazy_path="{scores_path}", k={k}, chunk_size=4096)',
            family="gaussian",
            latents={"z": latent},
        )
        print(model.summary())

    peak = rss_mb()
    delta = peak - baseline
    print(f"peak RSS delta: {delta:.1f} MB")
    assert delta < 100.0


if __name__ == "__main__":
    main()
