"""Recover a principal subspace as a point on Gr(k, n)."""

import numpy as np

import gamfit


def qf(a):
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def main():
    rng = np.random.default_rng(7)
    n, k = 6, 2
    data = rng.normal(size=(400, n))
    data[:, :k] *= 4.0
    cov = data.T @ data / data.shape[0]
    y = qf(rng.normal(size=(n, k)))
    for _ in range(80):
        grad = -2.0 * cov @ y
        tangent = grad - y @ (y.T @ grad)
        y = qf(y - 0.04 * tangent)
    score = np.trace(y.T @ cov @ y)
    print(gamfit.GrassmannManifold(k, n).to_json())
    print("variance score", round(float(score), 6))


if __name__ == "__main__":
    main()
