"""Compute a spherical Frechet mean."""

import numpy as np

import gamfit


def main() -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.99498743710662],
            [0.0, 0.1, 0.99498743710662],
            [-0.1, 0.0, 0.99498743710662],
        ]
    )
    mean = gamfit.sphere_frechet_mean(points)
    print("sphere mean", np.round(mean, 6))


if __name__ == "__main__":
    main()
