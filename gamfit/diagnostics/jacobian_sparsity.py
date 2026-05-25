"""Decoder-Jacobian sparsity diagnostic for nonlinear-ICA identifiability.

Reference: Hyvarinen, A., Morioka, H. (2017). "Nonlinear ICA of temporally
dependent stationary sources." AISTATS 2017. See also
Lachapelle et al. (2024) "Disentanglement via mechanism sparsity"
arXiv:2401.04890 for the modern mechanism-sparsity restatement.

The mechanism-sparsity / structured-Jacobian identifiability theorem
requires the Jacobian ``J(x) = ∂ f^{-1}(x) / ∂ x`` (or, in a linear-decoder
setting, the decoder matrix itself) to be sparse — a constant fraction of
its entries must be exactly zero (or below a small threshold). Without
this structure, the latent variables are only identified up to an
arbitrary invertible linear transform (Hyvarinen-Morioka Lemma 1).

The diagnostic accepts either:

* a fitted model with a linear ``decoder`` attribute (shape
  ``(P, latent_dim)``), in which case ``decoder`` itself is the Jacobian,
  or
* an explicit ``jacobians`` array of shape ``(N_samples, P, latent_dim)``
  giving the per-row Jacobian.

Sparsity is measured as the fraction of entries with absolute value
below ``zero_threshold * max|J|``. The check passes when that fraction is
at least ``sparsity_threshold`` *and* the Jacobian has full column rank
(rank == ``latent_dim``) at every sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._report import IdentifiabilityReport

__all__ = ["check_jacobian_sparsity"]


def _extract_jacobians(model: Any, X: Any) -> tuple[np.ndarray, str]:
    """Best-effort extraction of decoder Jacobians from a fitted model.

    Returns ``(jacobians, source)`` where ``jacobians`` has shape
    ``(N_samples, P, latent_dim)`` (broadcasting a constant linear decoder
    to a single sample if appropriate) and ``source`` is a short string
    describing how the Jacobian was obtained.

    Raises :class:`TypeError` if no Jacobian can be inferred.
    """
    # 1. Identifiable factor fit result: decoder shape (P, n_supervised + n_free).
    if hasattr(model, "decoder") and isinstance(getattr(model, "decoder"), np.ndarray):
        dec = np.asarray(model.decoder, dtype=float)
        return dec[np.newaxis, :, :].copy(), "linear_decoder_attr"

    # 2. Manifold-SAE-style: decoder_blocks is a list of (basis_size, P) arrays.
    if hasattr(model, "decoder_blocks"):
        blocks = list(model.decoder_blocks)
        if blocks:
            # Concatenate columns: per-atom decoder block has shape
            # (basis_size, P); transpose to (P, basis_size) and stack.
            cols = [np.asarray(b, dtype=float).T for b in blocks]
            J = np.concatenate(cols, axis=1)
            return J[np.newaxis, :, :].copy(), "decoder_blocks_concat"

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
        A fitted model with a linear ``decoder`` (shape
        ``(P, latent_dim)``) or ``decoder_blocks`` attribute. Ignored if
        ``jacobians`` is supplied directly.
    X : array-like, optional
        Sample of input points. Currently used only to broadcast a linear
        Jacobian; reserved for future nonlinear-decoder support.
    jacobians : array-like of shape ``(N_samples, P, latent_dim)``, optional
        Explicit per-sample decoder Jacobians. Takes precedence over
        ``model``.
    sparsity_threshold : float, default ``0.5``
        Minimum fraction of near-zero entries required across the
        averaged Jacobian.
    zero_threshold : float, default ``1e-3``
        An entry is "near-zero" if its absolute value is below
        ``zero_threshold * max|J|``.

    Returns
    -------
    IdentifiabilityReport
    """
    name = "jacobian_sparsity"
    theorem = "Hyvarinen-Morioka 2017; Lachapelle et al. 2024 mechanism sparsity"

    preconditions: dict[str, bool] = {}
    violations: list[str] = []
    recommendations: list[str] = []

    if jacobians is not None:
        J = np.asarray(jacobians, dtype=float)
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
            J, source = _extract_jacobians(model, X)
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
            preconditions=preconditions, violations=violations, recommendations=recommendations,
        )
    preconditions["jacobian_available"] = True

    n_samples, p_feat, latent_dim = J.shape

    # Compute sparsity per sample then aggregate.
    abs_J = np.abs(J)
    max_abs = float(abs_J.max())
    if max_abs <= 0.0:
        # All-zero Jacobian: technically maximally sparse, but the decoder
        # carries no information about latents — that is a worse failure.
        preconditions["jacobian_nonzero"] = False
        violations.append(
            "Decoder Jacobian is identically zero; the latent->observation map is degenerate."
        )
        recommendations.append(
            "Refit the model with a non-collapsed decoder (check learning rate, "
            "penalty weights, and initialization)."
        )
        return IdentifiabilityReport(
            name=name, theorem=theorem,
            preconditions=preconditions, violations=violations, recommendations=recommendations,
        )
    preconditions["jacobian_nonzero"] = True

    cutoff = zero_threshold * max_abs
    near_zero = abs_J < cutoff
    sparsity_per_sample = near_zero.reshape(n_samples, -1).mean(axis=1)
    mean_sparsity = float(sparsity_per_sample.mean())

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

    # Per-sample column rank must equal latent_dim.
    ranks = np.array(
        [int(np.linalg.matrix_rank(J[i], tol=cutoff)) for i in range(n_samples)],
        dtype=int,
    )
    rank_ok = bool(np.all(ranks >= latent_dim))
    preconditions["jacobian_full_column_rank"] = rank_ok
    if not rank_ok:
        bad = int(np.sum(ranks < latent_dim))
        min_rank = int(ranks.min())
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
        "n_samples": n_samples,
        "p_features": p_feat,
        "latent_dim": latent_dim,
        "mean_sparsity": mean_sparsity,
        "min_rank": int(ranks.min()),
    }
    return IdentifiabilityReport(
        name=name, theorem=theorem,
        preconditions=preconditions, violations=violations, recommendations=recommendations,
        details=details,
    )
