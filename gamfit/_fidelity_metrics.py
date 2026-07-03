"""Model-currency fidelity metric reductions (scorecard axis 1).

The canonical numeric core for the fidelity harness: ``loss_recovered``,
Euclidean ``r2_score``, categorical ``kl_categorical_rows`` over token rows, and
the ``distortion_floor_r2``. Held-out EV prices reconstruction error in Euclidean
distance, but the model reads directions unequally; these price reconstruction in
the model's own loss/output currency (see METRICS.md axis 1). The *patching* that
produces the losses/logprobs lives at the torch/HF boundary in the harness; these
are the pure reductions, shared verbatim by every consumer (CONTROL composed-dict
run, EVAL scorecard, RUNG1 iid-vs-Fisher-GLS A/B) so the program has ONE currency.

Pure-numpy so it runs on MSI without a fresh wheel. Each function transparently
dispatches to the Rust source of truth ``gam_math::fidelity_metrics`` (via
``gamfit._rust``) once that extension exposes the ``fidelity_*`` entry points;
``tests/metrics/test_fidelity_metrics_parity.py`` pins the numpy path to the Rust
kernel to double precision and asserts they agree, so the fallback can never
silently diverge from the source of truth.

Definitions (model loss units / nats):
    loss_recovered = (L_ablate − L_recon) / (L_ablate − L_clean)
    r2_score       = 1 − Σ(clean − approx)² / Σ(clean − colmean(clean))²
    kl_categorical_rows = mean_row Σ_v p·(logp_clean − logp_other), p = exp(logp_clean)
    distortion_floor_r2  = highest quant-R² at which loss_recovered has fallen
                           more than ``tol_frac`` below the finest-precision plateau
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

_EPS = 1e-12


def _rust_metrics():
    """Return the compiled extension iff it exposes the fidelity kernels, else None.

    A soft probe: if the wheel predates the ``fidelity_*`` FFI exposure (or the
    extension is a pure-Python shim), we silently use the numpy path. The parity
    test guarantees the two agree when the kernels ARE present.
    """
    try:
        from gamfit._binding import rust_module

        module = rust_module()
    except Exception:
        return None
    return module if hasattr(module, "fidelity_loss_recovered") else None


def loss_recovered(l_clean: float, l_recon: float, l_ablate: float) -> float:
    """(L_ablate − L_recon) / (L_ablate − L_clean); NaN when the gap is degenerate."""
    rust = _rust_metrics()
    if rust is not None:
        return float(rust.fidelity_loss_recovered(float(l_clean), float(l_recon), float(l_ablate)))
    denom = l_ablate - l_clean
    if abs(denom) < _EPS:
        return float("nan")
    return (l_ablate - l_recon) / denom


def r2_score(clean: np.ndarray, approx: np.ndarray) -> float:
    """Euclidean R² of ``approx`` vs ``clean``; TSS about the per-column mean of clean."""
    clean = np.ascontiguousarray(clean, dtype=np.float64)
    approx = np.ascontiguousarray(approx, dtype=np.float64)
    if clean.shape != approx.shape:
        raise ValueError(f"r2_score shape mismatch: {clean.shape} vs {approx.shape}")
    clean2 = clean.reshape(clean.shape[0], -1) if clean.ndim > 1 else clean.reshape(1, -1)
    approx2 = approx.reshape(clean2.shape)
    rust = _rust_metrics()
    if rust is not None:
        n_rows, n_cols = clean2.shape
        return float(
            rust.fidelity_r2_score(
                clean2.ravel().tolist(), approx2.ravel().tolist(), int(n_rows), int(n_cols)
            )
        )
    rss = float(np.sum((clean2 - approx2) ** 2))
    tss = float(np.sum((clean2 - clean2.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0.0 else 0.0


def kl_categorical_rows(clean_logprobs: np.ndarray, other_logprobs: np.ndarray) -> float:
    """Mean over rows of KL(clean ‖ other) from natural-log probability rows (nats)."""
    clean_lp = np.ascontiguousarray(clean_logprobs, dtype=np.float64)
    other_lp = np.ascontiguousarray(other_logprobs, dtype=np.float64)
    if clean_lp.shape != other_lp.shape:
        raise ValueError(f"kl shape mismatch: {clean_lp.shape} vs {other_lp.shape}")
    clean2 = clean_lp.reshape(clean_lp.shape[0], -1) if clean_lp.ndim > 1 else clean_lp.reshape(1, -1)
    other2 = other_lp.reshape(clean2.shape)
    rust = _rust_metrics()
    if rust is not None:
        n_rows, n_cols = clean2.shape
        return float(
            rust.fidelity_kl_categorical_rows(
                clean2.ravel().tolist(), other2.ravel().tolist(), int(n_rows), int(n_cols)
            )
        )
    p = np.exp(clean2)
    kl = np.sum(p * (clean2 - other2), axis=-1)
    return float(np.mean(kl))


def distortion_floor_r2(
    r2s: Sequence[float], loss_recovereds: Sequence[float], tol_frac: float = 0.05
) -> Optional[float]:
    """Highest-R² point at which loss_recovered drops >``tol_frac`` below the plateau.

    ``r2s`` and ``loss_recovereds`` are parallel per-quantisation-level arrays
    (order irrelevant; sorted by R² descending internally). Above the returned R²
    extra precision buys ~no model-usable fidelity, so all fidelity is read AT the
    floor. Returns ``None`` only for an empty sweep.
    """
    r2_arr = np.asarray(r2s, dtype=np.float64)
    lr_arr = np.asarray(loss_recovereds, dtype=np.float64)
    if r2_arr.shape != lr_arr.shape:
        raise ValueError("distortion_floor_r2: parallel arrays length mismatch")
    if r2_arr.size == 0:
        return None
    rust = _rust_metrics()
    if rust is not None:
        out = rust.fidelity_distortion_floor_r2(
            r2_arr.tolist(), lr_arr.tolist(), float(tol_frac)
        )
        return None if out is None else float(out)
    # Descending by R² (finest precision first), stable tie-break on original index.
    order = sorted(range(r2_arr.size), key=lambda i: (-r2_arr[i], i))
    plateau = float(lr_arr[order[0]])
    threshold = plateau - tol_frac * abs(plateau)
    for idx in order:
        if lr_arr[idx] < threshold:
            return float(r2_arr[idx])
    return float(r2_arr[order[-1]])
