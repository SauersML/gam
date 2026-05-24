"""Fit wrapped phases on a product torus."""

import numpy as np

import gamfit


def wrap(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def main():
    rng = np.random.default_rng(11)
    target = np.array([1.0, -2.2, 0.4])
    theta = rng.normal(size=3)
    for _ in range(60):
        grad = wrap(theta - target)
        theta = wrap(theta - 0.2 * grad)
    manifold = gamfit.ProductManifold(
        gamfit.CircleManifold(),
        gamfit.CircleManifold(),
        gamfit.CircleManifold(),
    )
    print(gamfit.TorusManifold(3).to_json())
    print(manifold.to_json())
    print("phase error", np.round(wrap(theta - target), 6))


if __name__ == "__main__":
    main()
