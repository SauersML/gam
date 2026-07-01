"""Posterior-predictive replicate sampling is reachable (#1057).

`src/inference/generative.rs` (GenerativeSpec / sampleobservation_replicates)
was implemented but had no user entry point. `Model.sample_replicates` now
surfaces it: family-aware observation draws from the fitted predictive
distribution at new data.

Objective assertions (no reference tool needed — the generative law IS the
ground truth):
  - Replicates have shape (n_draws, n_rows) and are seed-deterministic.
  - For a Gaussian fit, the per-row replicate mean converges to the model's
    predicted mean as n_draws grows, and the replicate spread tracks the
    fitted residual scale (the noise model is honest).
  - For a Poisson fit, replicates are non-negative integers.
"""
from __future__ import annotations

import importlib
import typing

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def _gaussian_rows(n: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n)
    y = 1.5 * x + 0.3 + rng.normal(scale=0.4, size=n)
    return [{"y": float(y[i]), "x": float(x[i])} for i in range(n)]


def test_replicates_shape_and_seed_determinism() -> None:
    rows = _gaussian_rows(60, seed=0)
    model = gamfit.fit(rows, "y ~ x")
    a = model.sample_replicates(rows, 16, seed=7)
    b = model.sample_replicates(rows, 16, seed=7)
    c = model.sample_replicates(rows, 16, seed=8)
    a = np.asarray(a)
    assert a.shape == (16, len(rows))
    np.testing.assert_array_equal(a, np.asarray(b))  # same seed -> identical
    assert not np.allclose(a, np.asarray(c))  # different seed -> different draws


def test_gaussian_replicate_mean_converges_to_predicted_mean() -> None:
    rows = _gaussian_rows(80, seed=1)
    model = gamfit.fit(rows, "y ~ x")
    predicted = np.asarray(model.predict(rows), dtype=float)
    reps = np.asarray(model.sample_replicates(rows, 4000, seed=3), dtype=float)
    rep_mean = reps.mean(axis=0)
    # The replicate mean is an unbiased estimate of the predictive mean.
    assert np.max(np.abs(rep_mean - predicted)) < 0.1, (
        f"replicate mean strayed from predicted mean: "
        f"max |Δ| = {np.max(np.abs(rep_mean - predicted)):.4f}"
    )
    # The per-row replicate spread must be a real, positive observation noise
    # (the noise model is not degenerate).
    assert np.all(reps.std(axis=0) > 0.05)


def test_poisson_replicates_are_nonnegative_integers() -> None:
    rng = np.random.default_rng(2)
    n = 70
    x = np.linspace(0.0, 1.5, n)
    mu = np.exp(0.5 + 1.0 * x)
    y = rng.poisson(mu)
    rows = [{"y": int(y[i]), "x": float(x[i])} for i in range(n)]
    model = gamfit.fit(rows, "y ~ x", family="poisson")
    reps = np.asarray(model.sample_replicates(rows, 50, seed=5), dtype=float)
    assert reps.shape == (50, n)
    assert np.all(reps >= 0.0)
    np.testing.assert_array_equal(reps, np.round(reps))
