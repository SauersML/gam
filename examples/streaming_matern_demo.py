"""Streaming Matérn basis demo."""

from __future__ import annotations

import resource
import sys

import numpy as np

import gamfit


def rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


def matern52(r: np.ndarray, length_scale: float) -> np.ndarray:
    z = np.sqrt(5.0) * r / length_scale
    return (1.0 + z + z * z / 3.0) * np.exp(-z)


def main() -> None:
    n = 20_000
    p = 4
    chunk = 2048
    rng = np.random.default_rng(17)
    baseline = rss_mb()

    x = rng.normal(size=(n, p))
    centers = rng.normal(size=(24, p))
    coef = rng.normal(scale=0.7, size=centers.shape[0])
    y = np.zeros(n)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = x[start:end, None, :] - centers[None, :, :]
        y[start:end] = matern52(np.sqrt(np.sum(diff * diff, axis=2)), 1.25) @ coef
    y += rng.normal(scale=0.05, size=n)

    data = {"y": y} | {f"x{i}": x[:, i] for i in range(p)}
    formula = (
        "y ~ matern(x0, x1, x2, x3, centers=96, nu=5/2, "
        "length_scale=1.25, streaming_chunk_size=2048)"
    )
    model = gamfit.fit(data, formula, family="gaussian")
    print(model.summary())

    delta = rss_mb() - baseline
    print(f"peak RSS delta: {delta:.1f} MB")
    assert delta < 512.0


if __name__ == "__main__":
    main()
