"""Held-out periodic atom accuracy across observation-noise levels.

For a unit-scale one-harmonic circle signal with i.i.d. Gaussian noise,
the rough signal-to-noise limit on achievable R^2 is about
``1 - noise**2``. Each case allows 0.05 finite-sample slack while keeping
a 0.70 floor so the highest-noise case still has to clear a real OOS bar.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# #1512: this fit exceeds the standard Python-API CI runner budget (>60s in
# triage), so it is tagged slow and excluded from the directory-level
# `-m "not slow"` CI step while still being collected (run by a bare pytest).
pytestmark = pytest.mark.slow

gamfit = pytest.importorskip("gamfit")


def _circle_data(
    n: int, p: int, noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z, theta


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@pytest.mark.parametrize("noise", [0.01, 0.05, 0.1, 0.2])
def test_periodic_atom_oos_r2_noise_sweep(noise: float):
    z, _ = _circle_data(n=600, p=64, noise=noise, seed=42)
    z_train = z[:300]
    z_test = z[300:]

    fit = gamfit.sae_manifold_fit(
        X=z_train,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ordered_beta_bernoulli",
        n_iter=50,
        learning_rate=0.04,
        random_state=0,
    )

    assert hasattr(fit, "reconstruct") or hasattr(fit, "predict"), (
        "OOS scoring regression: periodic atom fit must expose "
        "`reconstruct` or `predict` for held-out reconstruction."
    )
    if hasattr(fit, "reconstruct"):
        oos_fitted = fit.reconstruct(z_test)
    else:
        oos_fitted = fit.predict(z_test)

    oos_r2 = _r2(z_test, oos_fitted)
    threshold = max(1.0 - noise**2 - 0.05, 0.70)
    assert oos_r2 >= threshold, (
        f"noise={noise}: held-out R^2 = {oos_r2:.4f}, "
        f"expected >= {threshold:.4f}"
    )
