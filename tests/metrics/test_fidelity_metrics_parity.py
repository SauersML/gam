"""Parity + correctness contract for the model-currency fidelity reductions.

``gamfit._fidelity_metrics`` is the ONE canonical numeric core for scorecard axis
1 (loss-recovered, Euclidean R², categorical KL over token rows, distortion-floor
R²), shared verbatim by CONTROL / EVAL / RUNG1. It has two execution paths: a
pure-numpy fallback (so it runs on MSI without a fresh wheel) and a dispatch to
the Rust source of truth ``gam_math::fidelity_metrics`` exposed over FFI as
``gamfit._rust.fidelity_*``. This test pins BOTH:

  * the numpy fallback is exercised against an INDEPENDENT inline reference on
    random inputs and known edge cases (runs everywhere, no extension needed);
  * when the compiled extension exposes the ``fidelity_*`` kernels, the Rust path
    is pinned to that same reference to double precision (1e-12) — so the fallback
    can never silently diverge from the source of truth.

Rust stays the source of truth; the numpy path is only a portability fallback.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

import gamfit._fidelity_metrics as fm  # noqa: E402


# --------------------------------------------------------------------------- #
# Independent inline references (deliberately NOT importing fm's implementations).
# --------------------------------------------------------------------------- #


def _ref_loss_recovered(l_clean: float, l_recon: float, l_ablate: float) -> float:
    denom = l_ablate - l_clean
    if abs(denom) < 1e-12:
        return float("nan")
    return (l_ablate - l_recon) / denom


def _ref_r2(clean: np.ndarray, approx: np.ndarray) -> float:
    clean = clean.astype(np.float64)
    approx = approx.astype(np.float64)
    rss = float(np.sum((clean - approx) ** 2))
    tss = float(np.sum((clean - clean.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0 else 0.0


def _ref_kl(clean_lp: np.ndarray, other_lp: np.ndarray) -> float:
    p = np.exp(clean_lp.astype(np.float64))
    return float(np.mean(np.sum(p * (clean_lp - other_lp), axis=-1)))


def _ref_floor(r2s: np.ndarray, lrs: np.ndarray, tol: float = 0.05):
    order = sorted(range(len(r2s)), key=lambda i: (-r2s[i], i))
    plateau = float(lrs[order[0]])
    thr = plateau - tol * abs(plateau)
    for idx in order:
        if lrs[idx] < thr:
            return float(r2s[idx])
    return float(r2s[order[-1]])


def _logprobs(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    logits = rng.standard_normal((rows, cols))
    return logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))


def _force_numpy(monkeypatch) -> None:
    monkeypatch.setattr(fm, "_rust_metrics", lambda: None)


# --------------------------------------------------------------------------- #
# numpy-fallback correctness (runs without the compiled extension).
# --------------------------------------------------------------------------- #


def test_loss_recovered_endpoints_numpy(monkeypatch) -> None:
    _force_numpy(monkeypatch)
    assert fm.loss_recovered(1.0, 1.0, 3.0) == pytest.approx(1.0)
    assert fm.loss_recovered(1.0, 3.0, 3.0) == pytest.approx(0.0)
    assert fm.loss_recovered(1.0, 2.0, 3.0) == pytest.approx(0.5)
    assert np.isnan(fm.loss_recovered(2.0, 2.0, 2.0))


def test_r2_endpoints_numpy(monkeypatch) -> None:
    _force_numpy(monkeypatch)
    clean = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert fm.r2_score(clean, clean) == pytest.approx(1.0)
    mean_pred = np.tile(clean.mean(axis=0, keepdims=True), (3, 1))
    assert fm.r2_score(clean, mean_pred) == pytest.approx(0.0, abs=1e-12)
    const = np.full((2, 2), 7.0)
    assert fm.r2_score(const, np.arange(4).reshape(2, 2).astype(float)) == 0.0


def test_kl_identical_zero_numpy(monkeypatch) -> None:
    _force_numpy(monkeypatch)
    rng = np.random.default_rng(0)
    lp = _logprobs(rng, 4, 6)
    assert fm.kl_categorical_rows(lp, lp) == pytest.approx(0.0, abs=1e-12)
    assert fm.kl_categorical_rows(lp, _logprobs(rng, 4, 6)) > 0.0


def test_floor_numpy(monkeypatch) -> None:
    _force_numpy(monkeypatch)
    # flat plateau -> coarsest R²
    assert fm.distortion_floor_r2([0.99, 0.90, 0.50], [1.0, 1.0, 1.0]) == pytest.approx(0.50)
    # drop detected at R²=0.60
    assert fm.distortion_floor_r2([0.99, 0.95, 0.60, 0.30], [1.0, 0.99, 0.80, 0.40]) == pytest.approx(0.60)
    assert fm.distortion_floor_r2([], []) is None


def test_numpy_matches_reference_random(monkeypatch) -> None:
    _force_numpy(monkeypatch)
    rng = np.random.default_rng(1234)
    for _ in range(20):
        lc, lr, la = rng.standard_normal(3) * 2.0
        assert fm.loss_recovered(lc, lr, la) == pytest.approx(_ref_loss_recovered(lc, lr, la), nan_ok=True)
    clean = rng.standard_normal((7, 5))
    approx = clean + 0.1 * rng.standard_normal((7, 5))
    assert fm.r2_score(clean, approx) == pytest.approx(_ref_r2(clean, approx), abs=1e-12)
    a, b = _logprobs(rng, 5, 9), _logprobs(rng, 5, 9)
    assert fm.kl_categorical_rows(a, b) == pytest.approx(_ref_kl(a, b), abs=1e-12)
    r2s = rng.random(8)
    lrs = rng.random(8)
    assert fm.distortion_floor_r2(r2s, lrs) == pytest.approx(_ref_floor(r2s, lrs), abs=1e-12)


# --------------------------------------------------------------------------- #
# Rust parity (skips when the extension lacks the fidelity_* kernels).
# --------------------------------------------------------------------------- #


def test_rust_matches_reference_when_available() -> None:
    if fm._rust_metrics() is None:
        pytest.skip("gamfit._rust fidelity_* kernels not present in this build")
    rng = np.random.default_rng(99)
    # loss_recovered
    for _ in range(20):
        lc, lr, la = rng.standard_normal(3) * 2.0
        assert fm.loss_recovered(lc, lr, la) == pytest.approx(_ref_loss_recovered(lc, lr, la), nan_ok=True, abs=1e-12)
    # r2 + kl on random matrices
    clean = rng.standard_normal((11, 6))
    approx = clean + 0.2 * rng.standard_normal((11, 6))
    assert fm.r2_score(clean, approx) == pytest.approx(_ref_r2(clean, approx), abs=1e-12)
    a, b = _logprobs(rng, 8, 13), _logprobs(rng, 8, 13)
    assert fm.kl_categorical_rows(a, b) == pytest.approx(_ref_kl(a, b), abs=1e-12)
    # floor
    r2s, lrs = rng.random(12), rng.random(12)
    assert fm.distortion_floor_r2(r2s, lrs) == pytest.approx(_ref_floor(r2s, lrs), abs=1e-12)
