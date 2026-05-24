"""Affine-invariant SPD metric interpolation."""

import numpy as np

import gamfit


def spd_power(a, power):
    vals, vecs = np.linalg.eigh((a + a.T) * 0.5)
    return (vecs * (vals**power)) @ vecs.T


def spd_log(a):
    vals, vecs = np.linalg.eigh((a + a.T) * 0.5)
    return (vecs * np.log(vals)) @ vecs.T


def spd_exp(a):
    vals, vecs = np.linalg.eigh((a + a.T) * 0.5)
    return (vecs * np.exp(vals)) @ vecs.T


def geodesic(p, q, t):
    half = spd_power(p, 0.5)
    inv_half = spd_power(p, -0.5)
    middle = inv_half @ q @ inv_half
    return half @ spd_exp(t * spd_log(middle)) @ half


def main():
    p = np.array([[2.0, 0.3], [0.3, 1.0]])
    q = np.array([[0.8, -0.2], [-0.2, 2.4]])
    learned = geodesic(p, q, 0.5)
    print(gamfit.SpdManifold(2).to_json())
    print("midpoint eigenvalues", np.round(np.linalg.eigvalsh(learned), 6))


if __name__ == "__main__":
    main()
