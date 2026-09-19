"""Multiclass GAM: gamfit's joint multinomial logit vs pyGAM one-vs-rest.

pyGAM has no multiclass GAM; the usual workaround is K independent binary
``LogisticGAM`` fits (one-vs-rest) whose probabilities are renormalised. This
script fits both on simulated 3- and 5-class data with known smooth
class-probability surfaces and reports, on a held-out test set:

* log loss and Brier score against the observed labels;
* mean |p̂ − p| against the TRUE class probabilities;
* expected calibration error (ECE, 10 equal-width bins over all K·n
  predicted probabilities);
* how far the raw one-vs-rest probability rows are from summing to one before
  the renormalisation a user has to add by hand.

pyGAM is run with its documented defaults (``gridsearch`` over lam), which is
what a user doing one-vs-rest would run. Usage::

    python bench/multiclass_vs_pygam_ovr.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from pygam import LogisticGAM, s

from gamfit.sklearn import GAMClassifier


def softmax(eta: np.ndarray) -> np.ndarray:
    weights = np.exp(eta - eta.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


SURFACES = {
    3: lambda x1, x2: np.column_stack(
        [np.zeros_like(x1), 1.6 * np.sin(np.pi * x1) + 0.4 * x2, 1.2 * x2**2 - 0.8 - 0.6 * x1]
    ),
    5: lambda x1, x2: np.column_stack(
        [
            np.zeros_like(x1),
            1.5 * np.sin(np.pi * x1),
            1.4 * x2,
            1.3 * np.cos(np.pi * x2) - 0.3,
            -1.2 * x1 + 0.8 * x1 * x1 - 0.2,
        ]
    ),
}


def simulate(rng, n, n_classes):
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    p = softmax(SURFACES[n_classes](x1, x2))
    y = (rng.uniform(size=(n, 1)) > p.cumsum(axis=1)).sum(axis=1)
    return np.column_stack([x1, x2]), y, p


def metrics(p_hat, y, p_true):
    n, k = p_hat.shape
    clipped = np.clip(p_hat, 1e-15, 1.0)
    onehot = np.eye(k)[y]
    flat_p, flat_y = p_hat.ravel(), onehot.ravel()
    bins = np.minimum((flat_p * 10).astype(int), 9)
    ece = sum(
        abs(flat_p[bins == b].mean() - flat_y[bins == b].mean()) * np.mean(bins == b)
        for b in range(10)
        if np.any(bins == b)
    )
    return {
        "logloss": float(-np.log(clipped[np.arange(n), y]).mean()),
        "brier": float(((p_hat - onehot) ** 2).sum(axis=1).mean()),
        "mae_true": float(np.abs(p_hat - p_true).mean()),
        "ece": float(ece),
    }


def fit_gamfit(X, y):
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1]})
    return GAMClassifier(formula="y ~ s(x1) + s(x2)").fit(frame, y)


def predict_gamfit(clf, X):
    return clf.predict_proba(pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1]}))


def fit_pygam_ovr(X, y, n_classes):
    return [
        LogisticGAM(s(0) + s(1)).gridsearch(X, (y == k).astype(int), progress=False)
        for k in range(n_classes)
    ]


def predict_pygam_ovr(models, X):
    raw = np.column_stack([m.predict_proba(X) for m in models])
    return raw / raw.sum(axis=1, keepdims=True), raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-test", type=int, default=5000)
    args = parser.parse_args()

    rows = []
    for n_classes in (3, 5):
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 * n_classes + seed)
            X, y, _ = simulate(rng, args.n_train, n_classes)
            X_test, y_test, p_test = simulate(rng, args.n_test, n_classes)

            t0 = time.perf_counter()
            clf = fit_gamfit(X, y)
            p_gamfit = predict_gamfit(clf, X_test)
            t_gamfit = time.perf_counter() - t0

            t0 = time.perf_counter()
            models = fit_pygam_ovr(X, y, n_classes)
            p_pygam, raw = predict_pygam_ovr(models, X_test)
            t_pygam = time.perf_counter() - t0

            for name, p_hat, seconds in (
                ("gamfit multinomial", p_gamfit, t_gamfit),
                ("pyGAM one-vs-rest", p_pygam, t_pygam),
            ):
                row = {"K": n_classes, "seed": seed, "method": name, "seconds": seconds}
                row.update(metrics(p_hat, y_test, p_test))
                row["raw_row_sum_dev"] = (
                    float(np.abs(raw.sum(axis=1) - 1.0).mean())
                    if name.startswith("pyGAM")
                    else float(np.abs(p_hat.sum(axis=1) - 1.0).mean())
                )
                rows.append(row)
                print(row, flush=True)

    table = pd.DataFrame(rows)
    summary = table.groupby(["K", "method"]).mean(numeric_only=True).drop(columns="seed")
    print()
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
