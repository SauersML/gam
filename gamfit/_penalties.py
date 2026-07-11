"""Analytic structured penalties and SAE assignment priors.

Thin Python configuration wrappers around the analytic primitives
implemented in `src/terms/analytic_penalties.rs`. Some wrappers are Python
dataclasses and others compose the Rust pyclass descriptor directly; in both
cases construction only builds a descriptor. Penalty value / gradient /
Hessian-vector products are evaluated by the Rust analytic kernels when the
descriptor is passed into a fit or when ``value_grad`` / ``hvp`` is called.

These structured
penalties span the identifiability tools the impossibility theorem
says a principal-manifold / SAE / SAE-manifold engine needs:

* `IsometryPenalty` lives on t (the per-observation latent field
  produced by `LatentCoord`). Pulls the decoder's pullback metric
  toward a reference Riemannian metric — gauge fix for the
  diffeomorphism gauge that bare `LatentCoord` carries.
* `SparsityPenalty` lives on β (SAE codes) or t (soft atom assignments).
  It is a smoothed L¹ penalty; the smoothing scale may be fixed by the
  Rust descriptor.
* `ScadMcpPenalty` lives on t. Concave element-wise sparsity that shrinks
  noise near zero while flattening the gradient for large true signals.
  The Rust descriptor accepts SCAD/MCP variants (for example
  ``variant="mcp"`` in the composition-engine examples).
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
* `SoftmaxAssignmentSparsityPenalty` and `OrderedBetaBernoulliPenalty` live on
  row-wise assignment logits. They serialize to the Rust assignment-prior
  registry used by SAE-style latent blocks.
* `TopKActivationPenalty`, `SmoothThresholdPenalty`, and `GatedSAEDecoder` support
  the newer SAE assignment family. `TopKActivationPenalty` / `SmoothThresholdPenalty`
  are Rust-backed analytic descriptors; `GatedSAEDecoder` is a config-only
  descriptor that serializes to the Rust gated decoder contract.
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
  row-conditional precision, the auxiliary-supervised sibling to ARD/Ortho.
* `IvaeRidgeMeanGauge` lives on t. It fixes the iVAE conditional-mean gauge
  by penalizing the component of t not explained by a ridge fit against
  auxiliary covariates u.
* `ParametricAuxConditionalPriorPenalty` lives on t. Parametric iVAE-style
  diagonal row precision learned from auxiliary covariates through a
  distance-kernel map.
* `OrthogonalityPenalty` lives on t. It fixes the rotation gauge by
  penalizing latent-axis correlations; pair it with ARD so pruned axes
  are identifiable.

The generated :data:`PENALTY_MANIFEST` may include engine penalty kinds before
this module binds a top-level Python wrapper for them. In this source tree the
manifest records ``decoder_incoherence`` /
``DecoderIncoherencePenalty`` engine support, while this module exposes only
the wrapper names listed in :data:`__all__`.

All analytic penalties compose with the existing smoothness penalty (`S(ρ)`),
they slot into the same REML outer loop, and their weights are
"just another hyperparameter" to that loop. Pass `weight="auto"`
(the default) to let REML choose; pass an explicit float to pin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeAlias

import numpy as np

from ._penalties_manifest import PENALTY_MANIFEST
from ._binding import rust_module as _rust_module

__all__ = [
    "PENALTY_MANIFEST",
    "AnalyticPenaltyKind",
    "IsometryPenalty",
    "SparsityPenalty",
    "ScadMcpPenalty",
    "ARDPenalty",
    "TopKActivationPenalty",
    "SmoothThresholdPenalty",
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
    "OrderedBetaBernoulliPenalty",
    "SoftmaxAssignmentSparsityPenalty",
    "SheafConsistencyPenalty",
    "ScalarWeightSchedule",
    "Penalty",
]


def _rust_descriptor_class(name: str) -> type[Any]:
    """Return a Python wrapper class around the Rust pyclass `name`.

    The Rust pyclass is constructed via `__init__` exactly as before, but the
    Python wrapper composes (not inherits) the inner descriptor and adds two
    composition-engine entry points required by non-gamfit hosts (torch
    trainers, JAX, Stan-style HMC):

    * `value_grad(t)` — scalar penalty value and ∂P/∂t. Routes through the
      same `analytic_penalty_value_grad` Rust kernel that REML uses
      internally, so the returned `(value, grad)` is bit-for-bit identical
      to what `gamfit.fit(..., penalties=[pen])` scores during PIRLS.
    * `hvp(t, v)` — Hessian-vector product (∂²P/∂t²)·v. Routes through the
      `analytic_penalty_hvp` kernel; every analytic penalty in this module
      has either a closed-form or analytic-matvec Hessian, so this never
      falls back to finite differences.

    All other attributes (target, weight, weight_schedule, to_rust_descriptor,
    __repr__, …) forward to the Rust descriptor via `__getattr__`, so the
    wrapper is duck-typed-equivalent to the underlying Rust class for every
    existing consumer.
    """
    module = _rust_module()
    rust_cls = getattr(module, name)
    wrapper = _build_penalty_wrapper(name, rust_cls)
    return wrapper


def _build_penalty_wrapper(name: str, rust_cls: type[Any]) -> type[Any]:
    """Build a Python composition wrapper around `rust_cls`.

    The wrapper holds an inner Rust descriptor and proxies every attribute
    access through `__getattr__` so callers see no behavioral difference for
    the existing surface. The wrapper exposes `value_grad` and `hvp` which
    forward into the polymorphic Rust evaluators.
    """

    class _PenaltyWrapper:
        __slots__ = ("_inner",)
        _rust_cls = rust_cls
        _penalty_name = name

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            object.__setattr__(self, "_inner", rust_cls(*args, **kwargs))

        @property
        def inner(self) -> Any:
            return self._inner

        def to_rust_descriptor(self) -> dict[str, Any]:
            return self._inner.to_rust_descriptor()

        def set_weight_schedule(self, schedule: Any) -> "_PenaltyWrapper":
            self._inner.set_weight_schedule(schedule)
            return self

        def value_grad(self, t: Any) -> tuple[Any, Any]:
            """Compute ``(P(t), ∂P/∂t)`` in the frame of ``t``.

            Frame is auto-detected from ``t`` (NumPy / Torch / JAX). The
            same Rust kernel runs in every frame; only the output wrapping
            differs. Torch and JAX outputs carry an autograd graph so that
            ``torch.autograd.grad`` and ``jax.grad`` of ``value`` return
            ``grad`` consistently with the analytic Rust gradient.
            """
            from ._frame import Frame, detect_frame

            frame = detect_frame(t)
            if frame is Frame.NUMPY:
                return _penalty_value_grad_via_rust(self, t)
            if frame is Frame.TORCH:
                from ._penalty_frames import torch_penalty_value_grad

                return torch_penalty_value_grad(self, t)
            from ._penalty_frames import jax_penalty_value_grad

            return jax_penalty_value_grad(self, t)

        def hvp(self, t: Any, v: Any) -> Any:
            """Compute ``∂²P/∂t² · v`` in the frame shared by ``t`` and ``v``.

            Mixed frames raise :class:`TypeError` — both arguments must be
            in the same frame.
            """
            from ._frame import Frame, detect_frame

            frame = detect_frame(t, v)
            if frame is Frame.NUMPY:
                return _penalty_hvp_via_rust(self, t, v)
            if frame is Frame.TORCH:
                from ._penalty_frames import torch_penalty_hvp

                return torch_penalty_hvp(self, t, v)
            from ._penalty_frames import jax_penalty_hvp

            return jax_penalty_hvp(self, t, v)

        def __getattr__(self, item: str) -> Any:
            # Called only when normal attribute resolution fails — forward to
            # the inner Rust descriptor so the wrapper is duck-typed
            # equivalent for every existing consumer.
            return getattr(object.__getattribute__(self, "_inner"), item)

        def __repr__(self) -> str:
            return repr(self._inner)

    _PenaltyWrapper.__name__ = name
    _PenaltyWrapper.__qualname__ = name
    return _PenaltyWrapper


def _penalty_t_shape(t_array: np.ndarray, name: str) -> tuple[int, int]:
    """Auto-derive `(n, d)` from `t`'s shape.

    Most analytic penalties live on a latent block `t` of shape
    `(n_obs, latent_dim)`. `MechanismSparsityPenalty` is the exception:
    it lives on a decoder weight matrix of shape `(d_latent, p_features)`,
    so we pass `(n=1, d=d_latent)` and let the Rust dispatch read
    `p_features` off the descriptor's `feature_groups`.
    """
    if name == "MechanismSparsityPenalty":
        if t_array.ndim != 2:
            raise ValueError(
                f"{name}.value_grad expects a 2D decoder weight matrix (d_latent, p_features)"
            )
        return 1, int(t_array.shape[0])
    if t_array.ndim == 2:
        return int(t_array.shape[0]), int(t_array.shape[1])
    if t_array.ndim == 1:
        return int(t_array.shape[0]), 1
    raise ValueError(f"{name}.value_grad expects t with ndim 1 or 2; got ndim={t_array.ndim}")


def _penalty_latents_and_descriptor(
    wrapper: Any, t_array: np.ndarray
) -> tuple[str, str]:
    """Serialize a single-penalty registry payload for the Rust evaluator.

    Returns `(latents_json, penalties_json)` where the latents block is a
    synthetic single-entry dict keyed by the descriptor's `target` name
    (defaulting to `"t"`), and the penalties block is a one-element list
    containing the descriptor produced by the inner Rust class.
    """
    import json as _json

    descriptor = wrapper.to_rust_descriptor()
    if not isinstance(descriptor, Mapping):
        raise TypeError(
            f"{type(wrapper).__name__}.to_rust_descriptor() must return a mapping"
        )
    descriptor = dict(descriptor)
    target = descriptor.get("target", "t")
    target_name = str(target) if isinstance(target, (str, int)) else str(target)
    if not isinstance(target_name, str) or not target_name:
        target_name = "t"
    descriptor["target"] = target_name
    n, d = _penalty_t_shape(t_array, type(wrapper).__name__)
    latents = {target_name: {"name": target_name, "n": int(n), "d": int(d)}}
    return _json.dumps(latents), _json.dumps([descriptor])


def _penalty_value_grad_via_rust(
    wrapper: Any, t: Any
) -> tuple[float, np.ndarray]:
    """Evaluate `(P(t), ∂P/∂t)` through the Rust kernel REML uses internally."""
    rust = _rust_module()
    eval_fn = getattr(rust, "analytic_penalty_value_grad", None)
    if eval_fn is None:
        raise AttributeError(
            "gamfit._rust does not expose analytic_penalty_value_grad; "
            "rebuild the local Rust extension"
        )
    t_array = np.ascontiguousarray(t, dtype=float)
    original_shape = t_array.shape
    flat = t_array.reshape(-1)
    latents_json, penalties_json = _penalty_latents_and_descriptor(wrapper, t_array)
    value, grad_target, _grad_rho, _grad_jac = eval_fn(
        latents_json,
        penalties_json,
        flat,
        None,
    )
    grad = np.asarray(grad_target, dtype=float).reshape(original_shape)
    return float(value), grad


def _penalty_hvp_via_rust(wrapper: Any, t: Any, v: Any) -> np.ndarray:
    """Evaluate `H · v = ∂²P/∂t² · v` through the Rust kernel."""
    rust = _rust_module()
    eval_fn = getattr(rust, "analytic_penalty_hvp", None)
    if eval_fn is None:
        raise NotImplementedError(
            f"hvp not available for {type(wrapper).__name__}: gamfit._rust does not "
            "expose analytic_penalty_hvp; rebuild the local Rust extension"
        )
    t_array = np.ascontiguousarray(t, dtype=float)
    v_array = np.ascontiguousarray(v, dtype=float)
    if v_array.shape != t_array.shape:
        raise ValueError(
            f"{type(wrapper).__name__}.hvp: v shape {v_array.shape} does not match "
            f"t shape {t_array.shape}"
        )
    original_shape = t_array.shape
    flat_t = t_array.reshape(-1)
    flat_v = v_array.reshape(-1)
    latents_json, penalties_json = _penalty_latents_and_descriptor(wrapper, t_array)
    hv = eval_fn(
        latents_json,
        penalties_json,
        flat_t,
        flat_v,
        None,
    )
    return np.asarray(hv, dtype=float).reshape(original_shape)


from ._sheaf import SheafConsistencyPenalty as SheafConsistencyPenalty

BlockSparsityPenalty = _rust_descriptor_class("BlockSparsityPenalty")
ParametricAuxConditionalPriorPenalty = _rust_descriptor_class("ParametricAuxConditionalPriorPenalty")
OrderedBetaBernoulliPenalty = _rust_descriptor_class("OrderedBetaBernoulliPenalty")
TotalVariationPenalty = _rust_descriptor_class("TotalVariationPenalty")
SoftmaxAssignmentSparsityPenalty = _rust_descriptor_class("SoftmaxAssignmentSparsityPenalty")
OrthogonalityPenalty = _rust_descriptor_class("OrthogonalityPenalty")
IvaeRidgeMeanGauge = _rust_descriptor_class("IvaeRidgeMeanGauge")
MechanismSparsityPenalty = _rust_descriptor_class("MechanismSparsityPenalty")


class AnalyticPenaltyKind(str, Enum):
    """Stable Python names for Rust analytic penalty descriptor kinds."""

    ISOMETRY = "isometry"
    SPARSITY = "sparsity"
    SOFTMAX_ASSIGNMENT_SPARSITY = "softmax_assignment_sparsity"
    ORDERED_BETA_BERNOULLI = "ordered_beta_bernoulli"
    ARD = "ard"
    TOPK_ACTIVATION = "topk_activation"
    SMOOTH_THRESHOLD = "smooth_threshold"
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
    """Annealing schedule for a scalar analytic-penalty weight.

    Parameters
    ----------
    w_start, w_end:
        Non-negative endpoints of the schedule.
    kind:
        ``"geometric"``, ``"linear"``, or ``"reciprocal_iter"``.
    rate:
        Geometric decay rate in ``(0, 1)`` when ``kind="geometric"``.
    steps:
        Positive number of linear interpolation steps when ``kind="linear"``.
    iter_count:
        Initial iteration counter, forwarded to the Rust descriptor.

    Returns
    -------
    ScalarWeightSchedule
        Use directly as ``weight_schedule`` or through
        ``penalty.set_weight_schedule(schedule)``.

    Raises
    ------
    ValueError
        If endpoints or schedule-specific fields are invalid.
    """

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
            "tau_min": getattr(schedule, "tau_min"),
            "decay": getattr(schedule, "decay"),
            "rate": getattr(schedule, "rate", None),
            "steps": getattr(schedule, "steps", None),
            "iter_count": getattr(schedule, "iter_count", 0),
        }
    decay = str(raw.get("decay", "geometric")).lower().replace("-", "_")
    tau_start = float(raw["tau_start"])
    tau_min = float(raw["tau_min"])
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
            "temperature_schedule.decay must be 'geometric', 'linear', or 'reciprocal_iter'"
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


ARDPenalty = _rust_descriptor_class("ARDPenalty")
TopKActivationPenalty = _rust_descriptor_class("TopKActivationPenalty")
SmoothThresholdPenalty = _rust_descriptor_class("SmoothThresholdPenalty")


SparsityPenalty = _rust_descriptor_class("SparsityPenalty")
AuxConditionalPriorPenalty = _rust_descriptor_class("AuxConditionalPriorPenalty")
BlockOrthogonalityPenalty = _rust_descriptor_class("BlockOrthogonalityPenalty")
IsometryPenalty = _rust_descriptor_class("IsometryPenalty")
ScadMcpPenalty = _rust_descriptor_class("ScadMcpPenalty")
NuclearNormPenalty = _rust_descriptor_class("NuclearNormPenalty")

_RUST_WRAPPER_DOCS: dict[str, str] = {
    "ARDPenalty": """Automatic relevance determination penalty over latent axes.

The Rust descriptor accepts the constructor fields supported by the compiled
extension, including ``target`` and optional weight controls. It serializes to
``kind="ard"`` and can be passed to ``gamfit.fit(..., penalties=[...])`` or
evaluated directly with ``value_grad(t)`` / ``hvp(t, v)``.
""",
    "TopKActivationPenalty": """Analytic SAE top-k activation penalty descriptor.

Targets the row-wise activation/assignment block named by ``target`` and
serializes to ``kind="topk_activation"``. Direct evaluation uses the same Rust
``value_grad`` / ``hvp`` kernels as the formula pipeline.
""",
    "SmoothThresholdPenalty": """Analytic smooth-threshold sparsity penalty descriptor.

Represents a sigmoid-smoothed coordinate threshold. The Python wrapper forwards
constructor arguments to the Rust descriptor and
supports ``value_grad(t)`` / ``hvp(t, v)`` in NumPy, Torch, and JAX frames.
""",
    "SparsityPenalty": """Smoothed element-wise sparsity penalty descriptor.

Use for latent coordinates, SAE codes, or assignment-like blocks as accepted by
the Rust descriptor. The wrapper is configuration-only until passed to a fit or
evaluated through ``value_grad`` / ``hvp``.
""",
    "AuxConditionalPriorPenalty": """Fixed row-conditional iVAE precision prior.

Lives on a latent block ``t`` and serializes to the Rust row-precision-prior
descriptor. Use when the per-row conditional precision is precomputed outside
the fit.
""",
    "BlockOrthogonalityPenalty": """Between-block latent-axis orthogonality penalty.

Penalizes cross-products between declared axis groups while leaving axes within
each group free. Commonly used to separate supervised and residual discovery
blocks.
""",
    "IsometryPenalty": """Pullback-metric isometry penalty for latent coordinates.

Use as a gauge fix for latent-coordinate decoders: the Rust side compares the
decoder pullback metric against the configured reference metric and supplies
analytic gradients/HVPs.
""",
    "ScadMcpPenalty": """Concave SCAD/MCP element-wise sparsity penalty.

Targets latent coordinates and forwards fields such as ``target``, ``weight``,
``n_eff``, and the Rust-supported ``variant`` selector. It shrinks small
coordinates while flattening the derivative for large coordinates.
""",
    "NuclearNormPenalty": """Smoothed nuclear-norm penalty for latent-coordinate matrices.

Encourages low-rank structure in the target block by penalizing singular
values. It is wired through both the formula ``penalties=`` bridge and the
direct Rust ``value_grad`` / ``hvp`` evaluators.
""",
    "BlockSparsityPenalty": """Group-lasso style block sparsity penalty.

Shrinks declared latent-axis groups as units rather than individual entries.
The exact accepted constructor fields are defined by the compiled Rust
descriptor.
""",
    "ParametricAuxConditionalPriorPenalty": """Parametric iVAE row-precision prior.

Learns a diagonal row-conditional precision map from auxiliary covariates via
the Rust descriptor and applies it to the targeted latent block.
""",
    "OrderedBetaBernoulliPenalty": """Ordered independent Beta--Bernoulli prior over row-wise assignment logits.

The public Rust-backed wrapper accepts fields such as ``k_max``, ``alpha``,
``tau``, ``learnable``, and ``target`` where supported by the extension, then
serializes to the ordered independent Beta--Bernoulli descriptor.
""",
    "TotalVariationPenalty": """Smoothed total-variation penalty on ordered or graph-linked rows.

Applies an L1-like penalty to first differences of the targeted latent block,
using the Rust descriptor's configured difference operator.
""",
    "SoftmaxAssignmentSparsityPenalty": """Softmax assignment sparsity penalty descriptor.

Targets row-wise assignment logits for the competitive softmax assignment
family and serializes to ``kind="softmax_assignment_sparsity"``.
""",
    "OrthogonalityPenalty": """Latent-axis correlation penalty.

Penalizes cross-axis correlations to fix the rotation gauge. Pair with ARD when
axes should also be pruned by evidence.
""",
    "IvaeRidgeMeanGauge": """iVAE conditional-mean ridge gauge.

Penalizes the component of a latent block not explained by a ridge map from
auxiliary covariates, fixing the conditional-mean gauge used by iVAE-style
identifiability arguments.
""",
    "MechanismSparsityPenalty": """Per-latent group-lasso sparsity on decoder weights.

Unlike most penalties in this module, direct evaluation expects a decoder
weight matrix ``(d_latent, p_features)`` and derives the latent dimension from
its row count.
""",
}

for _name, _doc in _RUST_WRAPPER_DOCS.items():
    globals()[_name].__doc__ = _doc


@dataclass(frozen=True, slots=True)
class GatedSAEDecoder:
    """Gated SAE decoder descriptor (gate + amplitude weights).

    Configuration-only wrapper: it validates the weight shapes and
    serializes them to the Rust ``gated_sae_decoder`` descriptor via
    :meth:`to_rust_descriptor`. All decode math (the gate threshold and the
    amplitude projection) is evaluated by the Rust kernel when the descriptor
    is passed into a fit.

    Parameters
    ----------
    w_gate:
        Square gate matrix of shape ``(F, F)``.
    w_amp:
        Amplitude matrix of shape ``(D, F)``.

    Raises
    ------
    ValueError
        If shapes are incompatible or any weight is non-finite.
    """

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

    def to_rust_descriptor(self) -> dict[str, Any]:
        return {
            "kind": "gated_sae_decoder",
            "w_gate": self.w_gate.tolist(),
            "w_amp": self.w_amp.tolist(),
        }


# Sum type for type hints on `gamfit.fit(..., penalties=...)` and similar.
Penalty = (
    "IsometryPenalty | SparsityPenalty | ScadMcpPenalty | ARDPenalty | "
    "TopKActivationPenalty | SmoothThresholdPenalty | TotalVariationPenalty | "
    "NuclearNormPenalty | BlockSparsityPenalty | "
    "MechanismSparsityPenalty | AuxConditionalPriorPenalty | IvaeRidgeMeanGauge | "
    "ParametricAuxConditionalPriorPenalty | OrthogonalityPenalty | OrderedBetaBernoulliPenalty | "
    "SoftmaxAssignmentSparsityPenalty"
)
