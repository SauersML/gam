"""Analytic structured penalties: isometry, sparsity, ARD, TV, nuclear norm, block sparsity, orthogonality.

Thin Python configuration wrappers around the analytic primitives
implemented in `src/terms/analytic_penalties.rs`. Each wrapper is a
pure dataclass — no computation runs at construction; the Rust side
materializes the penalty's value / gradient / Hessian-vector product
analytically inside the inner loop.

See `proposals/composition_engine.md` §3-§4 and
`proposals/latent_coord.md` §2.3 for the motivation. These structured
penalties span the identifiability tools the impossibility theorem
says a principal-manifold / SAE / SAE-manifold engine needs:

* `IsometryPenalty` lives on t (the per-observation latent field
  produced by `LatentCoord`). Pulls the decoder's pullback metric
  toward a reference Riemannian metric — gauge fix for the
  diffeomorphism gauge that bare `LatentCoord` carries.
* `SparsityPenalty` lives on β (SAE codes) or t (soft atom
  assignments). Smoothed L¹ by default, with `ε` itself optionally
  REML-selected.
* `ARDPenalty` lives on t. One weight per latent axis, all
  REML-selectable. The Occam factor in the marginal likelihood
  prunes unused axes only after an AuxPrior or Isometry-style gauge
  fix pins rotations/reparameterisations.
* `TotalVariationPenalty` lives on t. Smoothed L¹ on first differences
  promotes piecewise-constant latent atom maps over ordered contexts or
  graph adjacencies.
* `NuclearNormPenalty` lives on t. Smoothed L¹ on singular values
  encourages low rank in a basis-free way; pair with ARD/Orthogonality
  depending on whether canonical-axis pruning or gauge fixing is also needed.
* `BlockSparsityPenalty` lives on t. Group-lasso smoothed L¹ on predefined
  latent-axis blocks, shrinking whole groups rather than individual entries
  or single ARD axes.
* `AuxConditionalPriorPenalty` lives on t. Fixed-precomputed iVAE-style
  row-conditional precision, the auxiliary-supervised sibling to ARD/Ortho
  from `proposals/composition_engine.md` §4(c).
* `OrthogonalityPenalty` lives on t. It fixes the rotation gauge by
  penalizing latent-axis correlations; pair it with ARD so pruned axes
  are identifiable.

All eight compose with the existing smoothness penalty (`S(ρ)`),
they slot into the same REML outer loop, and their weights are
"just another hyperparameter" to that loop. Pass `weight="auto"`
(the default) to let REML choose; pass an explicit float to pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any, Literal, TypeAlias

import numpy as np

__all__ = [
    "IsometryPenalty",
    "SparsityPenalty",
    "ARDPenalty",
    "TotalVariationPenalty",
    "NuclearNormPenalty",
    "BlockSparsityPenalty",
    "AuxConditionalPriorPenalty",
    "OrthogonalityPenalty",
    "IBPAssignmentPenalty",
    "SoftmaxAssignmentSparsityPenalty",
    "Penalty",
]


# Weight specification: either "auto" (REML-selected) or a positive float
# (held fixed at that value throughout the fit).
WeightSpec: TypeAlias = str | float
TargetSpec: TypeAlias = str | int | Any


def _validate_weight(weight: Any, name: str) -> None:
    if isinstance(weight, str):
        if weight != "auto":
            raise ValueError(
                f"{name}.weight: only 'auto' is accepted as a string, got {weight!r}"
            )
    elif isinstance(weight, (int, float)):
        if float(weight) <= 0.0:
            raise ValueError(f"{name}.weight must be > 0, got {weight}")
    else:
        raise TypeError(
            f"{name}.weight must be 'auto' or a positive float, got {type(weight).__name__}"
        )


def _target_descriptor(target: Any) -> str | int:
    if isinstance(target, (str, int)):
        return target
    name = getattr(target, "name", None)
    if name is not None:
        return str(name)
    raise TypeError(
        "analytic penalty target must be a latent block name, latent block index, "
        "or object exposing a 'name' attribute"
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
    other smoothing weight.

    **When to use.** Whenever a ``LatentCoord`` block is in play and there is
    no auxiliary prior to break the diffeomorphism gauge. The bare data-fit
    loss is invariant under any smooth invertible reparameterization of ``t``;
    the isometry penalty breaks that symmetry by pinning the local geometry
    of the decoder. The analytic gradient w.r.t. ``t`` reuses the
    radial-derivative ``∂Φ/∂t`` jet that ``LatentCoord`` already computes via
    ``latent_coord::LatentCoordValues::design_gradient_wrt_t``.

    Parameters
    ----------
    target
        Either the name of a ``LatentCoord`` block (``"t"``) or the
        ``LatentCoord`` object itself.
    reference
        ``"euclidean"`` for the identity reference metric, or a callable /
        array yielding per-row ``(d, d)`` reference metrics.
    weight
        ``"auto"`` (REML-selected; the default) or a fixed positive float.
    """

    target: TargetSpec
    reference: Any = "euclidean"
    weight: WeightSpec = "auto"

    def __post_init__(self) -> None:
        _validate_weight(self.weight, "IsometryPenalty")
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
            "target": _target_descriptor(self.target),
            "reference": self.reference if isinstance(self.reference, str) else "user_supplied",
            "weight": self.weight,
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass
class SparsityPenalty:
    """Smoothed-L¹ / Hoyer / Log sparsifier on a β or t slice.

    The smoothed-L¹ default is

    .. math::

        P(\\beta; \\rho, \\varepsilon) \\;=\\; e^{\\rho_\\mathrm{spars}}
            \\sum_i \\sqrt{\\beta_i^2 + \\varepsilon^2},

    with analytic gradient ``β_i / sqrt(β_i^2 + ε²)`` (smoothed sign) and
    diagonal Hessian ``ε² / (β_i^2 + ε²)^{3/2}``. The weight
    ``e^{\\rho_\\mathrm{spars}}`` is REML-selectable; ``ε`` may *also* be
    REML-selected (``eps_weight="auto"``), in which case the Occam factor
    of the marginal likelihood shrinks ``ε`` only as far as the data warrants.

    Alternatives: ``kind="hoyer"`` (scale-invariant; no diagonal HVP) and
    ``kind="log"`` (``log(1 + x²/δ²)``; aggressively sparsifying).

    **When to use.** SAE codes on a β slice; soft atom amplitudes on a
    design-moving ext-coordinate slice; any time the inductive bias is
    "this coefficient block should be sparse" without giving up
    differentiability for an active-set solver.

    Parameters
    ----------
    target
        The name of the β block (or ext-coordinate slice) to apply the
        penalty to.
    kind
        ``"smooth_l1"`` (the default), ``"hoyer"``, or ``"log"``.
    weight
        ``"auto"`` (REML) or a fixed positive float.
    eps
        Smoothing scale for ``"smooth_l1"`` / ``"log"`` kernels. Default
        ``1e-3``.
    eps_weight
        ``"auto"`` to let REML select ``ε`` as well; ``"fixed"`` to pin
        it at ``eps``. Defaults to ``"fixed"``.
    """

    target: TargetSpec
    kind: Literal["smooth_l1", "hoyer", "log"] = "smooth_l1"
    weight: WeightSpec = "auto"
    eps: float = 1e-3
    eps_weight: Literal["auto", "fixed"] = "fixed"

    def __post_init__(self) -> None:
        _validate_weight(self.weight, "SparsityPenalty")
        if self.kind not in ("smooth_l1", "hoyer", "log"):
            raise ValueError(
                f"SparsityPenalty.kind must be one of 'smooth_l1' | 'hoyer' | 'log', "
                f"got {self.kind!r}"
            )
        if self.kind == "hoyer" and self.eps_weight == "auto":
            raise ValueError(
                "SparsityPenalty(kind='hoyer'): Hoyer has no smoothing scale, "
                "so eps_weight='auto' is not meaningful."
            )
        if self.eps <= 0.0:
            raise ValueError(f"SparsityPenalty.eps must be > 0, got {self.eps}")
        if self.eps_weight not in ("auto", "fixed"):
            raise ValueError(
                f"SparsityPenalty.eps_weight must be 'auto' or 'fixed', "
                f"got {self.eps_weight!r}"
            )

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "sparsity",
            "target": _target_descriptor(self.target),
            "sparsity_kind": self.kind,
            "weight": self.weight,
            "eps": float(self.eps),
            "eps_weight": self.eps_weight,
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


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
    ``ρ_j`` only after a separate gauge fix has pinned the latent axes.

    **When to use.** Any ``LatentCoord`` block where the intrinsic dimension
    is unknown. Fixes the audit-revised claim: compose with
    ``IsometryPenalty`` or an auxiliary prior; ARD alone is rotation-symmetric.

    Parameters
    ----------
    target
        The ``LatentCoord`` block (or its name).
    weight
        ``"auto"`` (REML-selected per axis; the default) or a length-``d``
        sequence of fixed positive floats.
    """

    target: TargetSpec
    weight: Any = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.weight, str):
            if self.weight != "auto":
                raise ValueError(
                    "ARDPenalty.weight: only 'auto' is accepted as a string"
                )
        else:
            try:
                vals = [float(v) for v in self.weight]
            except TypeError as exc:
                raise TypeError(
                    "ARDPenalty.weight must be 'auto' or a sequence of floats"
                ) from exc
            if not vals:
                raise ValueError("ARDPenalty.weight must have at least one entry")
            if any(v <= 0.0 for v in vals):
                raise ValueError("ARDPenalty.weight entries must be > 0")

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "ard",
            "target": _target_descriptor(self.target),
            "weight": self.weight,
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass(init=False)
class TotalVariationPenalty:
    """Smoothed-L¹ total variation on first differences of a latent block.

    Uses ``φ(x) = sqrt(x² + ε²) - ε`` on ``D @ T``. ``difference_op`` is either
    ``"forward_1d"`` for adjacent ordered rows or a graph edge list
    ``[(from_row, to_row), ...]``. Pair with ``OrthogonalityPenalty`` when
    piecewise-constant atoms should be interpreted in a gauge-fixed basis.

    Parameters
    ----------
    weight
        Fixed base weight, or the base multiplier when ``learnable=True``.
    n_eff
        Number of rows in the row-major latent coefficient block.
    difference_op
        ``"forward_1d"`` or a sequence of ``(from_row, to_row)`` graph edges.
    smoothing_eps
        Positive smoothing scale ``ε`` for the smoothed-L¹ kernel.
    learnable
        If true, expose one REML-selectable log-weight ``ρ``.
    target
        The ``LatentCoord`` block name/object. Defaults to ``"t"``.
    """

    target: TargetSpec
    weight: float
    n_eff: int
    difference_op: Any
    smoothing_eps: float
    learnable: bool
    _edges: list[tuple[int, int]] | None

    def __init__(
        self,
        weight: float,
        n_eff: int,
        difference_op: Any = "forward_1d",
        smoothing_eps: float = 1e-6,
        learnable: bool = False,
        *,
        target: TargetSpec = "t",
    ) -> None:
        self.target = target
        self.weight = float(weight)
        self.n_eff = int(n_eff)
        self.difference_op = difference_op
        self.smoothing_eps = float(smoothing_eps)
        self.learnable = bool(learnable)
        self._edges = None
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.weight > 0.0:
            raise ValueError(f"TotalVariationPenalty.weight must be > 0, got {self.weight}")
        if self.n_eff <= 0:
            raise ValueError(f"TotalVariationPenalty.n_eff must be > 0, got {self.n_eff}")
        if not self.smoothing_eps > 0.0:
            raise ValueError(
                "TotalVariationPenalty.smoothing_eps must be > 0, "
                f"got {self.smoothing_eps}"
            )
        if isinstance(self.difference_op, str):
            if self.difference_op != "forward_1d":
                raise ValueError(
                    "TotalVariationPenalty.difference_op string must be 'forward_1d', "
                    f"got {self.difference_op!r}"
                )
            self._edges = None
            return

        try:
            edges = [(index(a), index(b)) for a, b in self.difference_op]
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "TotalVariationPenalty.difference_op must be 'forward_1d' "
                "or a sequence of (from_row, to_row) edges"
            ) from exc
        if not edges:
            raise ValueError("TotalVariationPenalty graph edges must not be empty")
        for a, b in edges:
            if a < 0 or b < 0 or a >= self.n_eff or b >= self.n_eff:
                raise ValueError(
                    "TotalVariationPenalty graph edges must be within "
                    f"[0, n_eff); got {(a, b)} for n_eff={self.n_eff}"
                )
            if a == b:
                raise ValueError(f"TotalVariationPenalty graph edge {(a, b)} is self-referential")
        self._edges = edges

    def _to_rust_payload(self) -> dict[str, Any]:
        payload = {
            "kind": "total_variation",
            "target": _target_descriptor(self.target),
            "weight": self.weight,
            "n_eff": self.n_eff,
            "smoothing_eps": self.smoothing_eps,
            "learnable": self.learnable,
        }
        if self._edges is None:
            payload["difference_op"] = "forward_1d"
        else:
            payload["difference_op"] = "graph_edges"
            payload["edges"] = self._edges
        return payload

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass(init=False)
class NuclearNormPenalty:
    """Smoothed nuclear norm on a matrix-valued latent block.

    Uses ``sum_i sqrt(σ_i² + ε²) - ε`` on the singular values of the row-major
    latent matrix. This encourages low rank without requiring useful axes to
    align with the canonical basis, complementing ARD's per-axis shrinkage and
    Orthogonality's gauge-fixing role.

    Parameters
    ----------
    weight
        Fixed base weight, or the base multiplier when ``learnable=True``.
    n_eff
        Number of rows in the row-major latent coefficient block.
    smoothing_eps
        Positive smoothing scale ``ε`` for singular values near zero.
    max_rank
        Optional cap on retained singular values.
    learnable
        If true, expose one REML-selectable log-weight ``ρ``.
    target
        The ``LatentCoord`` block name/object. Defaults to ``"t"``.
    """

    target: TargetSpec
    weight: float
    n_eff: int
    smoothing_eps: float
    max_rank: int | None
    learnable: bool

    def __init__(
        self,
        weight: float,
        n_eff: int,
        smoothing_eps: float = 1e-6,
        max_rank: int | None = None,
        learnable: bool = False,
        *,
        target: TargetSpec = "t",
    ) -> None:
        self.target = target
        self.weight = float(weight)
        self.n_eff = int(n_eff)
        self.smoothing_eps = float(smoothing_eps)
        self.max_rank = None if max_rank is None else index(max_rank)
        self.learnable = bool(learnable)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.weight > 0.0:
            raise ValueError(f"NuclearNormPenalty.weight must be > 0, got {self.weight}")
        if self.n_eff <= 0:
            raise ValueError(f"NuclearNormPenalty.n_eff must be > 0, got {self.n_eff}")
        if not self.smoothing_eps > 0.0:
            raise ValueError(
                "NuclearNormPenalty.smoothing_eps must be > 0, "
                f"got {self.smoothing_eps}"
            )
        if self.max_rank is not None and self.max_rank <= 0:
            raise ValueError(f"NuclearNormPenalty.max_rank must be > 0, got {self.max_rank}")

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "nuclear_norm",
            "target": _target_descriptor(self.target),
            "weight": self.weight,
            "n_eff": self.n_eff,
            "smoothing_eps": self.smoothing_eps,
            "max_rank": self.max_rank,
            "learnable": self.learnable,
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass(init=False)
class BlockSparsityPenalty:
    """Group-lasso sparsity over predefined latent-axis groups.

    Uses ``sum_g sqrt(||T[:, G_g]||² + ε²)``. This differs from per-element
    L¹ by selecting whole groups, and from ARD by applying L¹ to each group's
    L² norm rather than an independent axis precision. Pair with
    ``LatentIdMode.AuxPriorDimSelection`` when auxiliary classes should select
    active latent groups.

    Parameters
    ----------
    groups
        Partition of latent column indices, e.g. ``[[0, 1], [2, 3]]``.
    weight
        Fixed base weight, or the base multiplier when ``learnable=True``.
    n_eff
        Number of rows in the row-major latent coefficient block.
    smoothing_eps
        Positive smoothing scale ``ε`` for group norms near zero.
    learnable
        If true, expose one REML-selectable log-weight ``ρ``.
    target
        The ``LatentCoord`` block name/object. Defaults to ``"t"``.
    """

    target: TargetSpec
    groups: list[list[int]]
    weight: float
    n_eff: int
    smoothing_eps: float
    learnable: bool

    def __init__(
        self,
        groups: Any,
        weight: float,
        n_eff: int,
        smoothing_eps: float = 1e-6,
        learnable: bool = False,
        *,
        target: TargetSpec = "t",
    ) -> None:
        self.target = target
        self.groups = self._coerce_groups(groups)
        self.weight = float(weight)
        self.n_eff = int(n_eff)
        self.smoothing_eps = float(smoothing_eps)
        self.learnable = bool(learnable)
        self.__post_init__()

    @staticmethod
    def _coerce_groups(groups: Any) -> list[list[int]]:
        try:
            coerced = [[index(axis) for axis in group] for group in groups]
        except TypeError as exc:
            raise TypeError(
                "BlockSparsityPenalty.groups must be a sequence of integer sequences"
            ) from exc
        if not coerced:
            raise ValueError("BlockSparsityPenalty.groups must not be empty")
        seen: set[int] = set()
        for group_idx, group in enumerate(coerced):
            if not group:
                raise ValueError(
                    f"BlockSparsityPenalty.groups[{group_idx}] must not be empty"
                )
            for axis in group:
                if axis < 0:
                    raise ValueError(
                        "BlockSparsityPenalty.groups entries must be non-negative; "
                        f"got {axis}"
                    )
                if axis in seen:
                    raise ValueError(
                        f"BlockSparsityPenalty.groups axis {axis} appears more than once"
                    )
                seen.add(axis)
        max_axis = max(seen)
        missing = set(range(max_axis + 1)) - seen
        if missing:
            first = min(missing)
            raise ValueError(
                "BlockSparsityPenalty.groups must partition contiguous axes from 0; "
                f"missing axis {first}"
            )
        return coerced

    def __post_init__(self) -> None:
        if not self.weight > 0.0:
            raise ValueError(f"BlockSparsityPenalty.weight must be > 0, got {self.weight}")
        if self.n_eff <= 0:
            raise ValueError(f"BlockSparsityPenalty.n_eff must be > 0, got {self.n_eff}")
        if not self.smoothing_eps > 0.0:
            raise ValueError(
                "BlockSparsityPenalty.smoothing_eps must be > 0, "
                f"got {self.smoothing_eps}"
            )

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "block_sparsity",
            "target": _target_descriptor(self.target),
            "groups": self.groups,
            "weight": self.weight,
            "n_eff": self.n_eff,
            "smoothing_eps": self.smoothing_eps,
            "learnable": self.learnable,
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass(init=False)
class AuxConditionalPriorPenalty:
    """Fixed-precomputed iVAE-style auxiliary-conditional prior on t.

    Applies ``0.5 * weight * sum_n t_n.T @ Lambda_n @ t_n`` with one
    caller-supplied positive-definite precision matrix per latent row. Python
    validates shape, finiteness, and symmetry; Rust performs the PD eigensolve.
    This is the fixed variant of the aux-conditional prior family identified
    in ``proposals/composition_engine.md`` §4(c).

    Parameters
    ----------
    lambda_per_row
        Numeric ndarray of shape ``(N, d, d)`` containing one symmetric
        precision matrix per latent row.
    weight
        Fixed base weight, or the base multiplier when ``learnable=True``.
    n_eff
        Number of rows in the row-major latent coefficient block.
    learnable
        If true, expose one REML-selectable log-weight ``ρ``.
    target
        The ``LatentCoord`` block name/object. Defaults to ``"t"``.
    """

    target: TargetSpec
    lambda_per_row: np.ndarray
    weight: float
    n_eff: int
    learnable: bool

    def __init__(
        self,
        lambda_per_row: Any,
        weight: float,
        n_eff: int,
        learnable: bool = False,
        *,
        target: TargetSpec = "t",
    ) -> None:
        self.target = target
        self.lambda_per_row = np.asarray(lambda_per_row, dtype=float)
        self.weight = float(weight)
        self.n_eff = int(n_eff)
        self.learnable = bool(learnable)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.weight > 0.0:
            raise ValueError(
                f"AuxConditionalPriorPenalty.weight must be > 0, got {self.weight}"
            )
        if self.n_eff <= 0:
            raise ValueError(
                f"AuxConditionalPriorPenalty.n_eff must be > 0, got {self.n_eff}"
            )
        if self.lambda_per_row.ndim != 3:
            raise ValueError(
                "AuxConditionalPriorPenalty.lambda_per_row must have shape (N, d, d), "
                f"got ndim={self.lambda_per_row.ndim}"
            )
        n_obs, rows, cols = self.lambda_per_row.shape
        if n_obs != self.n_eff:
            raise ValueError(
                "AuxConditionalPriorPenalty.lambda_per_row first dimension must equal "
                f"n_eff={self.n_eff}, got {n_obs}"
            )
        if rows == 0 or cols == 0 or rows != cols:
            raise ValueError(
                "AuxConditionalPriorPenalty.lambda_per_row must have square non-empty "
                f"row matrices, got shape {self.lambda_per_row.shape}"
            )
        if not np.isfinite(self.lambda_per_row).all():
            raise ValueError("AuxConditionalPriorPenalty.lambda_per_row must be finite")
        max_asym = float(
            np.max(np.abs(self.lambda_per_row - np.swapaxes(self.lambda_per_row, 1, 2)))
        )
        if max_asym >= 1.0e-10:
            raise ValueError(
                "AuxConditionalPriorPenalty.lambda_per_row matrices must be symmetric "
                f"within 1e-10; max asymmetry is {max_asym:.3e}"
            )

    def _to_rust_payload(self) -> dict[str, Any]:
        arr = np.ascontiguousarray(self.lambda_per_row, dtype=float)
        return {
            "kind": "aux_conditional_prior",
            "target": _target_descriptor(self.target),
            "lambda_per_row": arr.reshape(-1).tolist(),
            "lambda_per_row_shape": list(arr.shape),
            "weight": self.weight,
            "n_eff": self.n_eff,
            "learnable": self.learnable,
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass(init=False)
class OrthogonalityPenalty:
    """Gauge-fixing penalty for latent-axis identifiability.

    Applies a Frobenius orthogonality penalty to the column-normalized latent
    coordinate matrix, so it breaks the rotation gauge while leaving ARD free
    to shrink individual axis norms.

    **When to use.** Pair with ``ARDPenalty`` when learning intrinsic
    dimension. ARD alone is rotation-invariant; Orthogonality locks the basis
    so an ARD-pruned axis has a stable meaning.

    Parameters
    ----------
    weight
        Fixed base weight, or the base multiplier when ``learnable=True``.
    n_eff
        Effective observation count used for normalization.
    learnable
        If true, expose one REML-selectable log-weight ``ρ``.
    target
        The ``LatentCoord`` block name/object. Defaults to ``"t"``.
    """

    target: TargetSpec
    weight: float
    n_eff: int
    learnable: bool

    def __init__(
        self,
        weight: float,
        n_eff: int,
        learnable: bool = False,
        *,
        target: TargetSpec = "t",
    ) -> None:
        self.target = target
        self.weight = float(weight)
        self.n_eff = int(n_eff)
        self.learnable = bool(learnable)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.weight > 0.0:
            raise ValueError(f"OrthogonalityPenalty.weight must be > 0, got {self.weight}")
        if self.n_eff <= 0:
            raise ValueError(f"OrthogonalityPenalty.n_eff must be > 0, got {self.n_eff}")

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "orthogonality",
            "target": _target_descriptor(self.target),
            "weight": self.weight,
            "n_eff": self.n_eff,
            "learnable": self.learnable,
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass
class IBPAssignmentPenalty:
    target: TargetSpec
    k_max: int
    alpha: float = 1.0
    tau: float = 1.0
    learnable_alpha: bool = False

    def __post_init__(self) -> None:
        if self.k_max <= 0:
            raise ValueError(f"IBPAssignmentPenalty.k_max must be > 0, got {self.k_max}")
        if self.alpha <= 0.0:
            raise ValueError(f"IBPAssignmentPenalty.alpha must be > 0, got {self.alpha}")
        if self.tau <= 0.0:
            raise ValueError(f"IBPAssignmentPenalty.tau must be > 0, got {self.tau}")

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "ibp_assignment",
            "target": _target_descriptor(self.target),
            "k_max": int(self.k_max),
            "alpha": float(self.alpha),
            "tau": float(self.tau),
            "learnable_alpha": bool(self.learnable_alpha),
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


@dataclass
class SoftmaxAssignmentSparsityPenalty:
    target: TargetSpec
    k_atoms: int
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.k_atoms <= 0:
            raise ValueError(
                "SoftmaxAssignmentSparsityPenalty.k_atoms must be > 0, "
                f"got {self.k_atoms}"
            )
        if self.temperature <= 0.0:
            raise ValueError(
                "SoftmaxAssignmentSparsityPenalty.temperature must be > 0, "
                f"got {self.temperature}"
            )

    def _to_rust_payload(self) -> dict[str, Any]:
        return {
            "kind": "softmax_assignment_sparsity",
            "target": _target_descriptor(self.target),
            "k_atoms": int(self.k_atoms),
            "temperature": float(self.temperature),
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()


# Sum type for type hints on `gamfit.fit(..., penalties=...)` and similar.
Penalty = (
    "IsometryPenalty | SparsityPenalty | ARDPenalty | "
    "TotalVariationPenalty | NuclearNormPenalty | BlockSparsityPenalty | "
    "AuxConditionalPriorPenalty | OrthogonalityPenalty | IBPAssignmentPenalty | "
    "SoftmaxAssignmentSparsityPenalty"
)
