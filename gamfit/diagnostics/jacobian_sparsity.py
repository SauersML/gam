"""Decoder-Jacobian sparsity diagnostic for nonlinear-ICA identifiability.

Reference: Hyvarinen, A., Morioka, H. (2017). "Nonlinear ICA of temporally
dependent stationary sources." AISTATS 2017. See also
Lachapelle et al. (2024) "Disentanglement via mechanism sparsity"
arXiv:2401.04890.

The numeric kernel — sparsity fraction and per-sample column rank — lives
in the Rust ``gam::identifiability::kernel::jacobian_sparsity_metrics`` function. This
Python file is a thin extraction-and-reporting wrapper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._binding import rust_module
from ._report import IdentifiabilityReport

__all__ = ["check_jacobian_sparsity"]


def _extract_jacobians(model: Any) -> tuple[np.ndarray, str]:
    """Return ``(jacobians, source)`` for a fitted model.

    ``jacobians`` has shape ``(N_samples, P, latent_dim)``. Stacking is
    delegated to Rust when the model only exposes a list of per-atom
    decoder blocks.
    """
    if hasattr(model, "decoder") and isinstance(getattr(model, "decoder"), np.ndarray):
        dec = np.ascontiguousarray(np.asarray(model.decoder, dtype=float))
        if dec.ndim != 2:
            raise TypeError(
                f"`decoder` must be a 2-D (P, latent_dim) ndarray; got shape {dec.shape}"
            )
        return dec[np.newaxis, :, :].copy(), "linear_decoder_attr"

    if hasattr(model, "decoder_blocks"):
        blocks = [
            np.ascontiguousarray(np.asarray(b, dtype=float)) for b in model.decoder_blocks
        ]
        if not blocks:
            raise TypeError("`decoder_blocks` is empty; no Jacobian available.")
        rust = rust_module()
        J = rust.diagnostics_concat_decoder_blocks(blocks)
        return np.asarray(J)[np.newaxis, :, :].copy(), "decoder_blocks_concat"

    raise TypeError(
        f"Cannot extract a decoder Jacobian from object of type "
        f"{type(model).__name__}; pass an explicit `jacobians=` array or "
        f"a model with a `decoder` / `decoder_blocks` attribute."
    )


def check_jacobian_sparsity(
    model: Any = None,
    X: Any = None,
    *,
    jacobians: Any = None,
    sparsity_threshold: float = 0.5,
    zero_threshold: float = 1.0e-3,
) -> IdentifiabilityReport:
    """Check the Hyvarinen-Morioka / Lachapelle Jacobian-sparsity precondition.

    Parameters
    ----------
    model : object, optional
        Fitted model with a linear ``decoder`` or ``decoder_blocks``.
    X : array-like, optional
        Reserved for future nonlinear-decoder Jacobian evaluation.
    jacobians : array-like of shape ``(N, P, latent_dim)`` or ``(P, latent_dim)``, optional
        Explicit Jacobians; takes precedence over ``model``.
    sparsity_threshold : float, default ``0.5``
        Minimum required sparsity fraction.
    zero_threshold : float, default ``1e-3``
        Relative cutoff: entries below ``zero_threshold * max|J|`` count as zero.

    Returns
    -------
    IdentifiabilityReport
    """
    del X  # currently unused; reserved for nonlinear extension.
    name = "jacobian_sparsity"
    theorem = "Hyvarinen-Morioka 2017; Lachapelle et al. 2024 mechanism sparsity"

    preconditions: dict[str, bool] = {}
    violations: list[str] = []
    recommendations: list[str] = []

    if jacobians is not None:
        J = np.ascontiguousarray(np.asarray(jacobians, dtype=float))
        source = "explicit_jacobians"
    else:
        if model is None:
            preconditions["jacobian_available"] = False
            violations.append("No model or `jacobians=` argument supplied.")
            recommendations.append(
                "Pass either a fitted model with a `decoder` attribute or "
                "an explicit `jacobians` array of shape (N, P, latent_dim)."
            )
            return IdentifiabilityReport(
                name=name, theorem=theorem,
                preconditions=preconditions, violations=violations,
                recommendations=recommendations,
            )
        try:
            J, source = _extract_jacobians(model)
        except TypeError as exc:
            preconditions["jacobian_available"] = False
            violations.append(str(exc))
            recommendations.append(
                "Add a `decoder` attribute (P x latent_dim) to the model, "
                "or pass `jacobians=` directly."
            )
            return IdentifiabilityReport(
                name=name, theorem=theorem,
                preconditions=preconditions, violations=violations,
                recommendations=recommendations,
            )

    if J.ndim == 2:
        J = J[np.newaxis, :, :]
    if J.ndim != 3:
        preconditions["jacobian_available"] = False
        violations.append(
            f"jacobians must be shape (N, P, latent_dim); got {J.shape}."
        )
        recommendations.append(
            "Reshape your Jacobian to (N_samples, n_features, latent_dim)."
        )
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations,
            recommendations=recommendations,
        )
    preconditions["jacobian_available"] = True

    n_samples, p_feat, latent_dim = J.shape
    # Flatten the (N, P, K) stack to (N*P, K) for the Rust kernel.
    flat = np.ascontiguousarray(J.reshape(n_samples * p_feat, latent_dim))

    rust = rust_module()
    metrics = rust.diagnostics_jacobian_sparsity(
        flat, int(n_samples), float(zero_threshold)
    )

    max_abs = float(metrics["max_abs"])
    if max_abs <= 0.0:
        preconditions["jacobian_nonzero"] = False
        violations.append(
            "Decoder Jacobian is identically zero; the latent->observation map is degenerate."
        )
        recommendations.append(
            "Refit the model with a non-collapsed decoder (check learning rate, "
            "penalty weights, and initialization)."
        )
        details = {"source": source, "max_abs": max_abs}
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations,
            recommendations=recommendations, details=details,
        )
    preconditions["jacobian_nonzero"] = True

    mean_sparsity = float(metrics["mean_sparsity"])
    sparsity_ok = mean_sparsity >= sparsity_threshold
    preconditions["sparsity_above_threshold"] = sparsity_ok
    if not sparsity_ok:
        violations.append(
            f"Decoder Jacobian sparsity {mean_sparsity:.3f} is below required "
            f"threshold {sparsity_threshold:.3f}; mechanism-sparsity "
            f"identifiability does not hold."
        )
        recommendations.append(
            f"Increase the mechanism-sparsity penalty weight until at least "
            f"{int(100 * sparsity_threshold)}% of decoder entries fall below "
            f"{zero_threshold:.0e} * max|J|."
        )

    ranks = [int(r) for r in metrics["ranks"]]
    rank_ok = all(r >= latent_dim for r in ranks)
    preconditions["jacobian_full_column_rank"] = rank_ok
    if not rank_ok:
        bad = sum(1 for r in ranks if r < latent_dim)
        min_rank = min(ranks)
        violations.append(
            f"Decoder Jacobian is rank-deficient at {bad}/{n_samples} sample "
            f"point(s); minimum rank {min_rank} < latent_dim {latent_dim}."
        )
        recommendations.append(
            f"Reduce latent_dim to <= {min_rank} or rebalance the sparsity "
            f"penalty so that all latent columns remain active."
        )

    details = {
        "source": source,
        "n_samples": int(metrics["n_samples"]),
        "p_features": int(metrics["p_features"]),
        "latent_dim": int(metrics["latent_dim"]),
        "mean_sparsity": mean_sparsity,
        "min_rank": min(ranks) if ranks else 0,
    }
    return IdentifiabilityReport(
        name=name, theorem=theorem,
        preconditions=preconditions, violations=violations,
        recommendations=recommendations, details=details,
    )
