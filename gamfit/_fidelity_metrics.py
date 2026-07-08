"""Model-currency fidelity metric reductions (scorecard axis 1).

Thin wrappers over the Rust source of truth ``gam_math::fidelity_metrics``
(exposed through ``gamfit._rust`` as the ``fidelity_*`` FFI): ``loss_recovered``,
Euclidean ``r2_score``, categorical ``kl_categorical_rows`` over token rows, and
the ``distortion_floor_r2``. Held-out EV prices reconstruction error in Euclidean
distance, but the model reads directions unequally; these price reconstruction in
the model's own loss/output currency (see METRICS.md axis 1). The *patching* that
produces the losses/logprobs lives at the torch/HF boundary in the harness; these
are the pure reductions, shared verbatim by every consumer (CONTROL composed-dict
run, EVAL scorecard, RUNG1 iid-vs-Fisher-GLS A/B) so the program has ONE currency.

Every function routes unconditionally to the Rust kernels — there is no numpy
twin. numpy here is marshalling only (shape validation, flattening to the
row-major flat layout the FFI consumes). If the compiled extension is missing the
``fidelity_*`` entry points that is an ImportError bug in the build, not a reason
to reimplement the math.

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


def _rust_metrics():
    """Return the compiled extension, requiring the ``fidelity_*`` FFI kernels.

    Hard dependency: the metric math is owned by Rust. A missing extension or a
    wheel that predates the ``fidelity_*`` exposure is a build bug and surfaces as
    an ImportError rather than a silent numpy reimplementation.
    """
    from gamfit._binding import rust_module

    module = rust_module()
    if not hasattr(module, "fidelity_loss_recovered"):
        raise ImportError(
            "gamfit._rust is missing the fidelity_* FFI kernels; rebuild the wheel "
            "(gam_math::fidelity_metrics must be exposed through gam-pyffi)"
        )
    return module


def loss_recovered(l_clean: float, l_recon: float, l_ablate: float) -> float:
    """(L_ablate − L_recon) / (L_ablate − L_clean); NaN when the gap is degenerate."""
    return float(
        _rust_metrics().fidelity_loss_recovered(float(l_clean), float(l_recon), float(l_ablate))
    )


def r2_score(clean: np.ndarray, approx: np.ndarray) -> float:
    """Euclidean R² of ``approx`` vs ``clean``; TSS about the per-column mean of clean."""
    clean = np.ascontiguousarray(clean, dtype=np.float64)
    approx = np.ascontiguousarray(approx, dtype=np.float64)
    if clean.shape != approx.shape:
        raise ValueError(f"r2_score shape mismatch: {clean.shape} vs {approx.shape}")
    clean2 = clean.reshape(clean.shape[0], -1) if clean.ndim > 1 else clean.reshape(1, -1)
    approx2 = approx.reshape(clean2.shape)
    n_rows, n_cols = clean2.shape
    return float(
        _rust_metrics().fidelity_r2_score(
            clean2.ravel().tolist(), approx2.ravel().tolist(), int(n_rows), int(n_cols)
        )
    )


def kl_categorical_rows(clean_logprobs: np.ndarray, other_logprobs: np.ndarray) -> float:
    """Mean over rows of KL(clean ‖ other) from natural-log probability rows (nats)."""
    clean_lp = np.ascontiguousarray(clean_logprobs, dtype=np.float64)
    other_lp = np.ascontiguousarray(other_logprobs, dtype=np.float64)
    if clean_lp.shape != other_lp.shape:
        raise ValueError(f"kl shape mismatch: {clean_lp.shape} vs {other_lp.shape}")
    clean2 = clean_lp.reshape(clean_lp.shape[0], -1) if clean_lp.ndim > 1 else clean_lp.reshape(1, -1)
    other2 = other_lp.reshape(clean2.shape)
    n_rows, n_cols = clean2.shape
    return float(
        _rust_metrics().fidelity_kl_categorical_rows(
            clean2.ravel().tolist(), other2.ravel().tolist(), int(n_rows), int(n_cols)
        )
    )


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
    out = _rust_metrics().fidelity_distortion_floor_r2(
        r2_arr.tolist(), lr_arr.tolist(), float(tol_frac)
    )
    return None if out is None else float(out)
