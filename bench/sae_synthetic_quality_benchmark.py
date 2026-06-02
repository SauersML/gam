"""Synthetic SAE-manifold quality benchmark.

Runs compact planted-periodic SAE fits with known atom assignments and prints
JSON metrics for reconstruction, assignment recovery, inactive leakage, and
runtime. This is a benchmark/reporting companion to
``tests/test_sae_manifold_synthetic_quality_ground_truth.py``; it has no pass
thresholds by design.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

import gamfit


@dataclass(frozen=True)
class SaeQualityMetrics:
    seed: int
    n_train: int
    n_test: int
    train_r2: float
    test_r2: float
    train_assignment_accuracy: float
    test_assignment_accuracy: float
    train_inactive_leakage: float
    test_inactive_leakage: float
    fit_seconds: float
    predict_seconds: float


def _periodic_phi(t: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi * np.asarray(t, dtype=float)
    return np.column_stack([np.ones_like(angle), np.sin(angle), np.cos(angle)])


def _decoder() -> list[np.ndarray]:
    return [
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.35, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.80, 0.45, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.90, -0.30, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.65, 1.05],
            ],
            dtype=float,
        ),
    ]


def _data(n: int, seed: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = np.arange(n, dtype=int) % 2
    rng.shuffle(truth)
    coords = rng.uniform(0.0, 1.0, size=(2, n))
    blocks = _decoder()
    x = np.zeros((n, 6), dtype=float)
    for atom in (0, 1):
        rows = truth == atom
        x[rows] = _periodic_phi(coords[atom, rows]) @ blocks[atom]
    x += noise * rng.standard_normal(size=x.shape)
    return x, truth


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _assignment_accuracy(assignments: np.ndarray, truth: np.ndarray) -> float:
    pred = np.argmax(assignments, axis=1)
    return max(float(np.mean(pred == truth)), float(np.mean((1 - pred) == truth)))


def run_one(
    seed: int,
    n_train: int,
    n_test: int,
    noise: float,
    max_iter: int,
) -> SaeQualityMetrics:
    train, train_truth = _data(n_train, seed=seed, noise=noise)
    test, test_truth = _data(n_test, seed=seed + 10_000, noise=noise)

    t0 = time.perf_counter()
    fit = gamfit.sae_manifold_fit(
        Z=train,
        n_atoms=2,
        atom_basis="periodic",
        atom_dim=1,
        assignment="softmax",
        top_k=1,
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        max_iter=int(max_iter),
        learning_rate=1.0,
        random_state=seed,
    )
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = fit.reconstruct(test)
    test_assign = fit.encode(test)
    predict_seconds = time.perf_counter() - t1

    train_assign = np.asarray(fit.assignments, dtype=float)
    return SaeQualityMetrics(
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        train_r2=_r2(train, np.asarray(fit.fitted, dtype=float)),
        test_r2=_r2(test, np.asarray(pred, dtype=float)),
        train_assignment_accuracy=_assignment_accuracy(train_assign, train_truth),
        test_assignment_accuracy=_assignment_accuracy(np.asarray(test_assign, dtype=float), test_truth),
        train_inactive_leakage=float(np.mean(np.min(train_assign, axis=1))),
        test_inactive_leakage=float(np.mean(np.min(np.asarray(test_assign, dtype=float), axis=1))),
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=48)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=6)
    args = parser.parse_args()

    metrics = [
        run_one(seed, args.n_train, args.n_test, args.noise, args.max_iter)
        for seed in args.seeds
    ]
    print(json.dumps([asdict(item) for item in metrics], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
