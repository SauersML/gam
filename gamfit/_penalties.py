"""Analytic structured penalties: isometry, sparsity, ARD-per-latent-dim.

Thin Python configuration wrappers around the analytic primitives
implemented in `src/terms/analytic_penalties.rs`. Each wrapper is a
pure dataclass — no computation runs at construction; the Rust side
materializes the penalty's value / gradient / Hessian-vector product
analytically inside the inner loop.

See `proposals/composition_engine.md` §3-§4 and
`proposals/latent_coord.md` §2.3 for the motivation. These three
penalties span the identifiability tools the impossibility theorem
says a principal-manifold / SAE / SAE-manifold engine needs:

* `IsometryPenalty` lives on ψ (the per-observation latent field
  produced by `LatentCoord`). Pulls the decoder's pullback metric
  toward a reference Riemannian metric — gauge fix for the
  diffeomorphism gauge that bare `LatentCoord` carries.
* `SparsityPenalty` lives on β (SAE codes) or ψ (soft atom
  assignments). Smoothed L¹ by default, with `ε` itself optionally
  REML-selected.
* `ARDPenalty` lives on ψ. One strength per latent axis, all
  REML-selectable. The Occam factor in the marginal likelihood
  prunes unused axes; the count of finite strengths at convergence
  is the intrinsic dimension.

All three compose with the existing smoothness penalty (`S(ρ)`),
they slot into the same REML outer loop, and their strengths are
"just another hyperparameter" to that loop. Pass `strength="auto"`
(the default) to let REML choose; pass an explicit float to pin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

__all__ = [
    "IsometryPenalty",
    "SparsityPenalty",
    "ARDPenalty",
    "Penalty",
]


# Strength specification: either "auto" (REML-selected) or a positive float
# (held fixed at that value throughout the fit).
StrengthSpec = "str | float"


def _validate_strength(strength: Any, name: str) -> None:
    if isinstance(strength, str):
        if strength != "auto":
            raise ValueError(
                f"{name}.strength: only 'auto' is accepted as a string, got {strength!r}"
            )
    elif isinstance(strength, (int, float)):
        if float(strength) <= 0.0:
            raise ValueError(f"{name}.strength must be > 0, got {strength}")
    else:
        raise TypeError(
            f"{name}.strength must be 'auto' or a positive float, got {type(strength).__name__}"
        )


@dataclass
class IsometryPenalty:
    """Pull the decoder's pullback metric toward a reference metric on the
    latent manifold.

    For a decoder ``T : ℝ^d → ℝ^p`` and per-row Jacobian
    ``J_n = ∂T/∂t |_{t = t_n}``, the induced pullback metric is
    ``g_n = J_n^T J_n``. The penalty is

    .. math::

        P_\\mathrm{iso}(t; \\rho) \\;=\\; \\tfrac12\\, e^{\\rho_\\mathrm{iso}}
            \\sum_n \\bigl\\| J_n^T J_n - g^\\mathrm{ref}_n \\bigr\\|_F^2

    with ``g^\\mathrm{ref}`` either the identity (``reference="euclidean"``,
    pulling toward a local isometry) or a per-row reference metric supplied
    by the caller. ``e^{\\rho_\\mathrm{iso}}`` is REML-selectable like any
    other smoothing strength.

    **When to use.** Whenever a ``LatentCoord`` block is in play without an
    ``AuxPrior`` or active ARD to break the diffeomorphism gauge. The bare
    data-fit loss is invariant under any smooth invertible reparameterization
    of ``t``; the isometry penalty breaks that symmetry by pinning the local
    geometry of the decoder. The analytic gradient w.r.t. ``t`` reuses the
    radial-derivative ``∂Φ/∂t`` jet that ``LatentCoord`` already computes
    via ``latent_coord::LatentCoordValues::design_gradient_wrt_t``.

    Parameters
    ----------
    target
        Either the name of a ``LatentCoord`` block (``"t"``) or the
        ``LatentCoord`` object itself.
    reference
        ``"euclidean"`` for the identity reference metric, or a callable /
        array yielding per-row ``(d, d)`` reference metrics.
    strength
        ``"auto"`` (REML-selected; the default) or a fixed positive float.
    """

    target: Any
    reference: Any = "euclidean"
    strength: Any = "auto"

    def __post_init__(self) -> None:
        _validate_strength(self.strength, "IsometryPenalty")
        if isinstance(self.reference, str) and self.reference != "euclidean":
            raise ValueError(
                "IsometryPenalty.reference: only 'euclidean' is supported as a string; "
                "pass an (N, d, d) array or callable for user-supplied references."
            )

    def _to_rust_payload(self) -> dict[str, Any]:
        """Spec for the Rust `AnalyticPenaltyRegistry::push` builder.

        The Rust side reads this dict and constructs the corresponding
        `terms::analytic_penalties::IsometryPenalty` instance. Kept as a dict
        rather than a typed handle so the Python surface stays import-light.
        """
        return {
            "kind": "isometry",
            "target": self.target if isinstance(self.target, str) else "__latent_object__",
            "reference": self.reference if isinstance(self.reference, str) else "user_supplied",
            "strength": self.strength,
        }


@dataclass
class SparsityPenalty:
    """Smoothed-L¹ / Hoyer / Log sparsifier on a β or ψ slice.

    The smoothed-L¹ default is

    .. math::

        P(\\beta; \\rho, \\varepsilon) \\;=\\; e^{\\rho_\\mathrm{spars}}
            \\sum_i \\sqrt{\\beta_i^2 + \\varepsilon^2},

    with analytic gradient ``β_i / sqrt(β_i^2 + ε²)`` (smoothed sign) and
    diagonal Hessian ``ε² / (β_i^2 + ε²)^{3/2}``. The strength
    ``e^{\\rho_\\mathrm{spars}}`` is REML-selectable; ``ε`` may *also* be
    REML-selected (``eps_strength="auto"``), in which case the Occam factor
    of the marginal likelihood shrinks ``ε`` only as far as the data warrants.

    Alternatives: ``kind="hoyer"`` (scale-invariant; no diagonal HVP) and
    ``kind="log"`` (``log(1 + x²/δ²)``; aggressively sparsifying).

    **When to use.** SAE codes on a β slice; soft atom amplitudes on a ψ
    slice; any time the inductive bias is "this coefficient block should be
    sparse" without giving up differentiability for an active-set solver.

    Parameters
    ----------
    target
        The name of the β block (or ψ slice) to apply the penalty to.
    kind
        ``"smooth_l1"`` (the default), ``"hoyer"``, or ``"log"``.
    strength
        ``"auto"`` (REML) or a fixed positive float.
    eps
        Smoothing scale for ``"smooth_l1"`` / ``"log"`` kernels. Default
        ``1e-3``.
    eps_strength
        ``"auto"`` to let REML select ``ε`` as well; ``"fixed"`` to pin
        it at ``eps``. Defaults to ``"fixed"``.
    """

    target: str
    kind: Literal["smooth_l1", "hoyer", "log"] = "smooth_l1"
    strength: Any = "auto"
    eps: float = 1e-3
    eps_strength: Literal["auto", "fixed"] = "fixed"

    def __post_init__(self) -> None:
        _validate_strength(self.strength, "SparsityPenalty")
        if self.kind not in ("smooth_l1", "hoyer", "log"):
            raise ValueError(
                f"SparsityPenalty.kind must be one of 'smooth_l1' | 'hoyer' | 'log', "
                f"got {self.kind!r}"
            )
        if self.kind == "hoyer" and self.eps_strength == "auto":
            raise ValueError(
                "SparsityPenalty(kind='hoyer'): Hoyer has no smoothing scale, "
                "so eps_strength='auto' is not meaningful."
            )
        if self.eps <= 0.0:
            raise ValueError(f"SparsityPenalty.eps must be > 0, got {self.eps}")
        if self.eps_strength not in ("auto", "fixed"):
            raise ValueError(
                f"SparsityPenalty.eps_strength must be 'auto' or 'fixed', "
                f"got {self.eps_strength!r}"
            )

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "sparsity",
            "target": self.target,
            "sparsity_kind": self.kind,
            "strength": self.strength,
            "eps": float(self.eps),
            "eps_strength": self.eps_strength,
        }


@dataclass
class ARDPenalty:
    """Automatic Relevance Determination over latent axes.

    For a ``LatentCoord`` block ``t ∈ ℝ^{N × d}``, applies one independent
    ridge penalty per axis with its own REML-selectable log-precision:

    .. math::

        P_\\mathrm{ARD}(t; \\rho) \\;=\\; \\tfrac12 \\sum_{j=0}^{d-1}
            e^{\\rho_j}\\, \\|t_{:,j}\\|^2.

    REML's marginal-likelihood selection drives ``ρ_j → +∞`` (precision → ∞,
    coefficients → 0) on axes whose data evidence does not justify them.
    The intrinsic dimension is read off post-fit as the number of finite
    ``ρ_j``. Solves two problems in one penalty: the rotation gauge (the
    eigenbasis is fixed because each axis has its own strength), and
    intrinsic-dim discovery (unused axes prune themselves).

    **When to use.** Any ``LatentCoord`` block where the intrinsic dimension
    is unknown. Compose with ``IsometryPenalty`` for full gauge fixing on
    higher-dim manifolds; ARD alone fixes the rotation, isometry fixes the
    remaining diffeomorphisms.

    Parameters
    ----------
    target
        The ``LatentCoord`` block (or its name).
    strength_per_dim
        ``"auto"`` (REML-selected per axis; the default) or a length-``d``
        sequence of fixed positive floats.
    """

    target: Any
    strength_per_dim: Any = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.strength_per_dim, str):
            if self.strength_per_dim != "auto":
                raise ValueError(
                    "ARDPenalty.strength_per_dim: only 'auto' is accepted as a string"
                )
        else:
            try:
                vals = [float(v) for v in self.strength_per_dim]
            except TypeError as exc:
                raise TypeError(
                    "ARDPenalty.strength_per_dim must be 'auto' or a sequence of floats"
                ) from exc
            if not vals:
                raise ValueError("ARDPenalty.strength_per_dim must have at least one entry")
            if any(v <= 0.0 for v in vals):
                raise ValueError("ARDPenalty.strength_per_dim entries must be > 0")

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "ard",
            "target": self.target if isinstance(self.target, str) else "__latent_object__",
            "strength_per_dim": self.strength_per_dim,
        }


# Sum type for type hints on `gamfit.fit(..., penalties=...)` and similar.
Penalty = "IsometryPenalty | SparsityPenalty | ARDPenalty"
