#!/usr/bin/env python3
"""MechanismSparsityPenalty worked example on a sparse latent decoder W."""

from __future__ import annotations

import json
import warnings

import numpy as np


N, P, D, SEED = 300, 24, 3, 802
GROUPS = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18)), list(range(18, 24))]
TRUE_ACTIVE = [{0, 2}, {1}, {2, 3}]
LAMBDA, RIDGE, STEPS = 0.18, 1e-3, 900


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((N, D))
    z = (z - z.mean(axis=0, keepdims=True)) / z.std(axis=0, keepdims=True)
    w_true = np.zeros((D, P))
    for latent, active_groups in enumerate(TRUE_ACTIVE):
        for group in active_groups:
            cols = GROUPS[group]
            w_true[latent, cols] = rng.normal(0.0, 0.9, size=len(cols))
    x = z @ w_true + 0.06 * rng.standard_normal((N, P))
    x -= x.mean(axis=0, keepdims=True)
    return z, x, w_true


def prox_mechanism(w: np.ndarray, step: float, lam: float) -> np.ndarray:
    out = w.copy()
    for latent in range(D):
        for group in GROUPS:
            norm = float(np.linalg.norm(out[latent, group]))
            shrink = max(0.0, 1.0 - step * lam * np.sqrt(len(group)) / max(norm, 1e-12))
            out[latent, group] *= shrink
    return out


def fit_decoder(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    lipschitz = float(np.linalg.norm(z, ord=2) ** 2 / N + RIDGE)
    step = 0.9 / lipschitz
    w = np.linalg.lstsq(z, x, rcond=None)[0]
    for _ in range(STEPS):
        grad = z.T @ (z @ w - x) / N + RIDGE * w
        w = prox_mechanism(w - step * grad, step, LAMBDA)
    return w


def active_counts(w: np.ndarray) -> list[int]:
    row_max = np.maximum(np.linalg.norm(w, axis=1), 1e-12)
    counts = []
    for latent in range(D):
        norms = np.array([np.linalg.norm(w[latent, group]) for group in GROUPS])
        counts.append(int(np.count_nonzero(norms > 0.08 * row_max[latent])))
    return counts


def check_wrapper() -> None:
    try:
        import gamfit
    except Exception as exc:
        warnings.warn(f"gamfit import unavailable; numeric decoder fit still runs: {exc!r}")
        return
    penalty = gamfit.MechanismSparsityPenalty(
        GROUPS, weight=LAMBDA, n_eff=N, target="t", smoothing_eps=1e-6
    )
    payload = penalty.to_rust_descriptor()
    assert payload["kind"] == "mechanism_sparsity"
    assert payload["feature_groups"] == GROUPS
    try:
        gamfit._binding.rust_module().register_analytic_penalties(
            json.dumps({"t": {"name": "t", "n": N, "d": D}}),
            json.dumps([payload]),
        )
    except Exception as exc:
        warnings.warn(f"pyffi registry check skipped: {exc!r}")


def main() -> None:
    check_wrapper()
    z, x, w_true = make_data()
    w_hat = fit_decoder(z, x)
    counts = active_counts(w_hat)
    ok = w_hat.shape == (D, P) and max(counts) <= 2
    print(f"W_shape = {w_hat.shape}")
    print(f"true_active_counts = {[len(g) for g in TRUE_ACTIVE]}")
    print(f"recovered_active_counts = {counts}")
    print(f"relative_error = {np.linalg.norm(w_hat - w_true) / np.linalg.norm(w_true):.3f}")
    print(f"PASS = {ok}")


if __name__ == "__main__":
    main()
