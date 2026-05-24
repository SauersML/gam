"""Optimize a quadratic objective directly on S^2."""

import numpy as np

import gamfit


def sphere_exp(x, u):
    u = u - x * float(x @ u)
    theta = float(np.linalg.norm(u))
    if theta < 1e-12:
        y = x + u
        return y / np.linalg.norm(y)
    return np.cos(theta) * x + np.sin(theta) * u / theta


def main():
    manifold = gamfit.SphereManifold(2)
    target = np.array([0.2, -0.4, 0.8944271909999159])
    x = np.array([1.0, 0.0, 0.0])
    for _ in range(40):
        grad = -(target - x * float(x @ target))
        x = sphere_exp(x, -0.25 * grad)
    print(manifold.to_json())
    print("solution", np.round(x, 6), "cosine", round(float(x @ target), 6))


if __name__ == "__main__":
    main()
