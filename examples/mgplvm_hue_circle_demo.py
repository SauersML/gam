"""Fit a circular latent coordinate to hue-like spherical responses."""

from __future__ import annotations

import numpy as np

import gamfit


def main() -> None:
    rng = np.random.default_rng(7)
    n = 96
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    y = np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
            0.35 * np.cos(2.0 * theta + 0.2),
        ]
    )
    y += 0.03 * rng.standard_normal(y.shape)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    data = {f"y{j}": y[:, j] for j in range(y.shape[1])}

    model = gamfit.fit(
        data,
        "y0 ~ s(t, type='periodic', k=16, period=6.283185307179586, origin=0)",
        family="gaussian",
        latents={
            "t": gamfit.LatentCoord(
                n=n,
                d=1,
                init=theta[:, None],
                manifold="circle",
                retraction="circle",
                aux_prior={"u": theta[:, None], "family": "ridge", "strength": "auto"},
            )
        },
        response_geometry="spherical",
        response_columns=["y0", "y1", "y2"],
    )

    shared_fit = model.summary()["shared_fit"]
    if not isinstance(shared_fit, dict):
        raise RuntimeError("response geometry fit did not produce a shared tangent fit")
    print(f"hue circle fit: lambda={float(shared_fit['lambda']):.4g}")


if __name__ == "__main__":
    main()
