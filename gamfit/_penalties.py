"""Analytic structured penalties: isometry, sparsity, SCAD/MCP, ARD, TV, nuclear norm, block sparsity, mechanism sparsity, block orthogonality, orthogonality.

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
* `ScadMcpPenalty` lives on t. Concave element-wise sparsity that shrinks
  noise near zero while flattening the gradient for large true signals.
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
* `MechanismSparsityPenalty` lives on decoder W. Per-latent group-lasso
  smoothed L¹ over feature groups.
* `BlockOrthogonalityPenalty` lives on t. Penalizes only between-block
  cross-products, leaving each block internally free; this is the
  supervised-block-plus-free-discovery-block pattern: an
  auxiliary-supervised gauge-fix block pins one subspace while a free block
  discovers the residual structure.
* `AuxConditionalPriorPenalty` lives on t. Fixed-precomputed iVAE-style
  row-conditional precision, the auxiliary-supervised sibling to ARD/Ortho
  from `proposals/composition_engine.md` §4(c).
* `IvaeRidgeMeanGauge` lives on t. It fixes the iVAE conditional-mean gauge
  by penalizing the component of t not explained by a ridge fit against
  auxiliary covariates u.
* `ParametricAuxConditionalPriorPenalty` lives on t. Parametric iVAE-style
  diagonal row precision learned from auxiliary covariates through a
  distance-kernel map.
* `OrthogonalityPenalty` lives on t. It fixes the rotation gauge by
  penalizing latent-axis correlations; pair it with ARD so pruned axes
  are identifiable.

All analytic penalties compose with the existing smoothness penalty (`S(ρ)`),
they slot into the same REML outer loop, and their weights are
"just another hyperparameter" to that loop. Pass `weight="auto"`
(the default) to let REML choose; pass an explicit float to pin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from operator import index
from typing import Any, Literal, TypeAlias

import numpy as np

from ._penalties_manifest import PENALTY_MANIFEST

__all__ = [
    "PENALTY_MANIFEST",
    "AnalyticPenaltyKind",
    "IsometryPenalty",
    "SparsityPenalty",
    "ScadMcpPenalty",
    "ARDPenalty",
    "TopKActivationPenalty",
    "JumpReLUPenalty",
    "GatedSAEDecoder",
    "TotalVariationPenalty",
    "NuclearNormPenalty",
    "BlockSparsityPenalty",
    "MechanismSparsityPenalty",
    "BlockOrthogonalityPenalty",
    "AuxConditionalPriorPenalty",
    "IvaeRidgeMeanGauge",
    "ParametricAuxConditionalPriorPenalty",
    "OrthogonalityPenalty",
    "IBPAssignmentPenalty",
    "SoftmaxAssignmentSparsityPenalty",
    "ScalarWeightSchedule",
    "Penalty",
]

from ._binding import rust_module as _rust_module; BlockSparsityPenalty = _rust_module().BlockSparsityPenalty
from ._binding import rust_module as _rust_module; ParametricAuxConditionalPriorPenalty = _rust_module().ParametricAuxConditionalPriorPenalty
from ._binding import rust_module as _rust_module; IBPAssignmentPenalty = _rust_module().IBPAssignmentPenalty
from ._binding import rust_module as _rust_module; TotalVariationPenalty = _rust_module().TotalVariationPenalty
from ._binding import rust_module as _rust_module; SoftmaxAssignmentSparsityPenalty = _rust_module().SoftmaxAssignmentSparsityPenalty
from ._binding import rust_module as _rust_module; OrthogonalityPenalty = _rust_module().OrthogonalityPenalty
from ._binding import rust_module as _rust_module; IvaeRidgeMeanGauge = _rust_module().IvaeRidgeMeanGauge
from ._binding import rust_module as _rust_module; MechanismSparsityPenalty = _rust_module().MechanismSparsityPenalty


class AnalyticPenaltyKind(str, Enum):
    """Stable Python names for Rust analytic penalty descriptor kinds."""

    ISOMETRY = "isometry"
    SPARSITY = "sparsity"
    SOFTMAX_ASSIGNMENT_SPARSITY = "softmax_assignment_sparsity"
    IBP_ASSIGNMENT = "ibp_assignment"
    ARD = "ard"
    TOPK_ACTIVATION = "topk_activation"
    JUMPRELU = "jumprelu"
    TOTAL_VARIATION = "total_variation"
    NUCLEAR_NORM = "nuclear_norm"
    BLOCK_SPARSITY = "block_sparsity"
    MECHANISM_SPARSITY = "mechanism_sparsity"
    NESTED_PREFIX = "nested_prefix"
    ROW_PRECISION_PRIOR = "row_precision_prior"
    IVAE_RIDGE_MEAN_GAUGE = "ivae_ridge_mean_gauge"
    PARAMETRIC_ROW_PRECISION_PRIOR = "parametric_row_precision_prior"
    SCAD_MCP = "scad_mcp"
    BLOCK_ORTHOGONALITY = "block_orthogonality"
    ORTHOGONALITY = "orthogonality"


# Weight specification: either "auto" (REML-selected) or a positive float
# (held fixed at that value throughout the fit).
WeightSpec: TypeAlias = str | float
TargetSpec: TypeAlias = str | int | Any


@dataclass(frozen=True, slots=True)
class ScalarWeightSchedule:
    """Temperature-style annealing schedule for analytic penalty weights."""

    w_start: float
    w_end: float
    kind: Literal["geometric", "linear", "reciprocal_iter"] = "geometric"
    rate: float | None = 0.9
    steps: int | None = None
    iter_count: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.w_start) or self.w_start < 0.0:
            raise ValueError(
                f"ScalarWeightSchedule.w_start must be finite and >= 0, got {self.w_start}"
            )
        if not np.isfinite(self.w_end) or self.w_end < 0.0:
            raise ValueError(
                f"ScalarWeightSchedule.w_end must be finite and >= 0, got {self.w_end}"
            )
        if self.kind == "geometric":
            if self.rate is None or not np.isfinite(self.rate) or not 0.0 < self.rate < 1.0:
                raise ValueError("ScalarWeightSchedule geometric rate must be in (0, 1)")
        elif self.kind == "linear":
            if self.steps is None or self.steps <= 0:
                raise ValueError("ScalarWeightSchedule linear steps must be positive")
        elif self.kind != "reciprocal_iter":
            raise ValueError(
                "ScalarWeightSchedule.kind must be 'geometric', 'linear', or 'reciprocal_iter'"
            )
        if self.iter_count < 0:
            raise ValueError("ScalarWeightSchedule.iter_count must be non-negative")

    def to_rust_descriptor(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "w_start": float(self.w_start),
            "w_end": float(self.w_end),
            "kind": self.kind,
            "iter_count": int(self.iter_count),
        }
        if self.kind == "geometric":
            if self.rate is None:
                raise ValueError("ScalarWeightSchedule geometric rate must be in (0, 1)")
            payload["rate"] = float(self.rate)
        if self.kind == "linear":
            if self.steps is None:
                raise ValueError("ScalarWeightSchedule linear steps must be positive")
            payload["steps"] = int(self.steps)
        return payload


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


def _weight_schedule_descriptor(schedule: Any) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, ScalarWeightSchedule):
        return schedule.to_rust_descriptor()
    if isinstance(schedule, dict):
        return dict(schedule)
    raise TypeError(
        "weight_schedule must be ScalarWeightSchedule, a mapping, or None"
    )


def _add_weight_schedule(payload: dict[str, Any], owner: Any) -> dict[str, Any]:
    schedule = _weight_schedule_descriptor(getattr(owner, "weight_schedule", None))
    if schedule is not None:
        payload["weight_schedule"] = schedule
    return payload


def _temperature_schedule_descriptor(schedule: Any) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if hasattr(schedule, "to_rust_descriptor"):
        raw = schedule.to_rust_descriptor()
    elif isinstance(schedule, Mapping):
        raw = dict(schedule)
    else:
        if not hasattr(schedule, "tau_start") or not hasattr(schedule, "decay"):
            raise TypeError(
                "temperature_schedule must be GumbelTemperatureSchedule, a mapping, or None"
            )
        raw = {
            "tau_start": getattr(schedule, "tau_start"),
            "tau_min": getattr(schedule, "tau_min", getattr(schedule, "tau_end", None)),
            "decay": getattr(schedule, "decay"),
            "rate": getattr(schedule, "rate", None),
            "steps": getattr(schedule, "steps", None),
            "iter_count": getattr(schedule, "iter_count", 0),
        }
    decay = str(raw.get("decay", "geometric")).lower().replace("-", "_")
    if decay == "exponential":
        decay = "geometric"
    tau_start = float(raw["tau_start"])
    raw_tau_min = raw.get("tau_min", raw.get("tau_end"))
    if raw_tau_min is None:
        raise ValueError("temperature_schedule requires tau_min or tau_end")
    tau_min = float(raw_tau_min)
    if not np.isfinite(tau_start) or tau_start <= 0.0:
        raise ValueError("temperature_schedule.tau_start must be finite and > 0")
    if not np.isfinite(tau_min) or tau_min <= 0.0:
        raise ValueError("temperature_schedule.tau_min must be finite and > 0")
    if tau_min > tau_start:
        raise ValueError("temperature_schedule.tau_min cannot exceed tau_start")
    out: dict[str, Any] = {
        "tau_start": tau_start,
        "tau_min": tau_min,
        "decay": decay,
        "iter_count": int(raw.get("iter_count", 0)),
    }
    if out["iter_count"] < 0:
        raise ValueError("temperature_schedule.iter_count must be non-negative")
    if decay == "geometric":
        rate = float(raw.get("rate", 0.9))
        if not np.isfinite(rate) or not 0.0 < rate < 1.0:
            raise ValueError("temperature_schedule.rate must be in (0, 1)")
        out["rate"] = rate
    elif decay == "linear":
        steps = raw.get("steps")
        if steps is None or int(steps) <= 0:
            raise ValueError("temperature_schedule.steps must be positive")
        out["steps"] = int(steps)
    elif decay != "reciprocal_iter":
        raise ValueError(
            "temperature_schedule.decay must be 'geometric', 'exponential', 'linear', or 'reciprocal_iter'"
        )
    return out


def _add_temperature_schedule(payload: dict[str, Any], owner: Any) -> dict[str, Any]:
    schedule = _temperature_schedule_descriptor(getattr(owner, "temperature_schedule", None))
    if schedule is not None:
        payload["temperature_schedule"] = schedule
    return payload


def _set_weight_schedule(self: Any, schedule: ScalarWeightSchedule | dict[str, Any]) -> Any:
    self.weight_schedule = schedule
    return self


class _AnalyticPenalty:
    """Shared boilerplate for analytic penalty dataclasses.

    Subclasses declare ``KIND_TAG`` (the string Rust expects under the ``"kind"``
    descriptor key) and override ``_payload_extras`` to return the
    kind-specific descriptor fields. The base class handles the common
    ``target``/``kind``/``weight_schedule`` plumbing and exposes
    ``set_weight_schedule`` so every analytic penalty supports fluent
    weight-schedule assignment without per-class patching.
    """

    KIND_TAG = ""

    def _payload_extras(self) -> dict[str, Any]:
        raise NotImplementedError

    def _to_rust_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": type(self).KIND_TAG,
            "target": _target_descriptor(getattr(self, "target")),
        }
        payload.update(self._payload_extras())
        return _add_weight_schedule(payload, self)

    def to_rust_descriptor(self) -> dict[str, Any]:
        return self._to_rust_payload()

    def set_weight_schedule(
        self, schedule: ScalarWeightSchedule | dict[str, Any]
    ) -> "_AnalyticPenalty":
        return _set_weight_schedule(self, schedule)


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


from ._binding import rust_module as _rust_module; ARDPenalty = _rust_module().ARDPenalty
from ._binding import rust_module as _rust_module; TopKActivationPenalty = _rust_module().TopKActivationPenalty
from ._binding import rust_module as _rust_module; JumpReLUPenalty = _rust_module().JumpReLUPenalty


def _inverse_softplus(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    large = x > 30.0
    out[large] = x[large]
    out[~large] = np.log(np.expm1(x[~large]))
    return out


from ._binding import rust_module as _rust_module; SparsityPenalty = _rust_module().SparsityPenalty
from ._binding import rust_module as _rust_module; AuxConditionalPriorPenalty = _rust_module().AuxConditionalPriorPenalty
from ._binding import rust_module as _rust_module; BlockOrthogonalityPenalty = _rust_module().BlockOrthogonalityPenalty
from ._binding import rust_module as _rust_module; IsometryPenalty = _rust_module().IsometryPenalty
from ._binding import rust_module as _rust_module; ScadMcpPenalty = _rust_module().ScadMcpPenalty


@dataclass(frozen=True, slots=True)
class GatedSAEDecoder:
    """Standalone gated SAE decoder with gate and amplitude weights."""

    w_gate: Any
    w_amp: Any

    def __post_init__(self) -> None:
        gate = np.asarray(self.w_gate, dtype=float)
        amp = np.asarray(self.w_amp, dtype=float)
        if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
            raise ValueError("GatedSAEDecoder.w_gate must be square")
        if amp.ndim != 2 or amp.shape[1] != gate.shape[1]:
            raise ValueError("GatedSAEDecoder.w_amp columns must match w_gate input")
        if not np.all(np.isfinite(gate)) or not np.all(np.isfinite(amp)):
            raise ValueError("GatedSAEDecoder weights must be finite")
        object.__setattr__(self, "w_gate", gate)
        object.__setattr__(self, "w_amp", amp)

    def decode(self, x: Any) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        single = x_arr.ndim == 1
        x2 = x_arr.reshape(1, -1) if single else x_arr
        if x2.ndim != 2 or x2.shape[1] != self.w_gate.shape[1]:
            raise ValueError(
                f"GatedSAEDecoder.decode expected input dimension {self.w_gate.shape[1]}"
            )
        gates = (_sigmoid_numpy(x2 @ self.w_gate.T) > 0.5).astype(float)
        out = (gates * x2) @ self.w_amp.T
        return out[0] if single else out

    def to_rust_descriptor(self) -> dict[str, Any]:
        return {
            "kind": "gated_sae_decoder",
            "w_gate": self.w_gate.tolist(),
            "w_amp": self.w_amp.tolist(),
        }


def _sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


@dataclass(init=False, slots=True)
class NuclearNormPenalty(_AnalyticPenalty):
    """Smoothed nuclear norm on a matrix-valued latent block.
    KIND_TAG = "nuclear_norm"

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
    weight_schedule: ScalarWeightSchedule | dict[str, Any] | None

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
        self.weight_schedule = None
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

    def _payload_extras(self) -> dict[str, Any]:
        return {
            "weight": self.weight,
            "n_eff": self.n_eff,
            "smoothing_eps": self.smoothing_eps,
            "max_rank": self.max_rank,
            "learnable": self.learnable,
        }

@dataclass(init=False, slots=True)
class MechanismSparsityPenalty(_AnalyticPenalty):
    """Per-latent group-lasso sparsity over decoder feature groups."""
    KIND_TAG = "mechanism_sparsity"

    target: TargetSpec
    feature_groups: list[list[int]]
    weight: float
    smoothing_eps: float
    n_eff: float
    learnable: bool
    weight_schedule: ScalarWeightSchedule | dict[str, Any] | None

    def __init__(
        self,
        feature_groups: Any,
        weight: float,
        n_eff: float,
        smoothing_eps: float = 1e-6,
        learnable: bool = False,
        *,
        target: TargetSpec = "t",
    ) -> None:
        self.target = target
        self.feature_groups = self._coerce_feature_groups(feature_groups)
        self.weight = float(weight)
        self.smoothing_eps = float(smoothing_eps)
        self.n_eff = float(n_eff)
        self.learnable = bool(learnable)
        self.weight_schedule = None
        self.__post_init__()

    @staticmethod
    def _coerce_feature_groups(feature_groups: Any) -> list[list[int]]:
        try:
            coerced = [[index(feature) for feature in group] for group in feature_groups]
        except TypeError as exc:
            raise TypeError(
                "MechanismSparsityPenalty.feature_groups must be a sequence of integer sequences"
            ) from exc
        if not coerced:
            raise ValueError("MechanismSparsityPenalty.feature_groups must not be empty")
        seen: set[int] = set()
        for group_idx, group in enumerate(coerced):
            if not group:
                raise ValueError(
                    f"MechanismSparsityPenalty.feature_groups[{group_idx}] must not be empty"
                )
            for feature in group:
                if feature < 0:
                    raise ValueError(
                        "MechanismSparsityPenalty.feature_groups entries must be non-negative; "
                        f"got {feature}"
                    )
                if feature in seen:
                    raise ValueError(
                        "MechanismSparsityPenalty.feature_groups feature "
                        f"{feature} appears more than once"
                    )
                seen.add(feature)
        missing = set(range(max(seen) + 1)) - seen
        if missing:
            first = min(missing)
            raise ValueError(
                "MechanismSparsityPenalty.feature_groups must partition contiguous features from 0; "
                f"missing feature {first}"
            )
        return coerced

    def __post_init__(self) -> None:
        if not self.weight > 0.0:
            raise ValueError(f"MechanismSparsityPenalty.weight must be > 0, got {self.weight}")
        if not self.smoothing_eps > 0.0:
            raise ValueError(
                "MechanismSparsityPenalty.smoothing_eps must be > 0, "
                f"got {self.smoothing_eps}"
            )
        if not np.isfinite(self.n_eff) or self.n_eff <= 0.0:
            raise ValueError(f"MechanismSparsityPenalty.n_eff must be > 0, got {self.n_eff}")

    def _payload_extras(self) -> dict[str, Any]:
        return {
            "feature_groups": self.feature_groups,
            "weight": self.weight,
            "smoothing_eps": self.smoothing_eps,
            "n_eff": self.n_eff,
            "learnable": self.learnable,
        }


# Sum type for type hints on `gamfit.fit(..., penalties=...)` and similar.
Penalty = (
    "IsometryPenalty | SparsityPenalty | ScadMcpPenalty | ARDPenalty | "
    "TopKActivationPenalty | JumpReLUPenalty | TotalVariationPenalty | "
    "NuclearNormPenalty | BlockSparsityPenalty | "
    "MechanismSparsityPenalty | AuxConditionalPriorPenalty | IvaeRidgeMeanGauge | "
    "ParametricAuxConditionalPriorPenalty | OrthogonalityPenalty | IBPAssignmentPenalty | "
    "SoftmaxAssignmentSparsityPenalty"
)
