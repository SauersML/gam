"""Single source of truth for the Python ↔ Rust analytic-penalty bridge.

Every penalty surface in gamfit — the composable frame-aware descriptors in
:mod:`gamfit._penalty_descriptors`, the ``nn.Module`` autograd shells in
:mod:`gamfit.torch.penalties`, the dataclass-wrapper adapters in
:mod:`gamfit._penalty_frames`, and the SAE regularizer payload in
:mod:`gamfit._sae_manifold` — shares the same plumbing:

* build the latent / penalty JSON the Rust registry consumes
  (:func:`latent_json`, :func:`penalty_json`, :func:`fixed_weight_schedule`),
* run ``analytic_penalty_value_grad`` / ``analytic_penalty_hvp`` once
  (:func:`call_value_grad`, :func:`call_hvp`, :func:`call_rust_value_grad`
  for the isometry-Jacobian variant),
* wrap the result for the active backend (the JAX ``custom_vjp`` core lives
  in :func:`jax_value_grad_from_rust`, re-exported here),
* construct the per-kind descriptor dict (:func:`*_descriptor`),
* and anneal the Gumbel temperature through exactly one
  :class:`GumbelTemperatureSchedule`.

No penalty math is reimplemented here. The Rust trait in
``src/terms/analytic_penalties.rs`` and ``src/terms/sae_manifold.rs`` are the
single source of truth for the math; this module only marshals JSON / arrays.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Sequence

import numpy as np

from ._binding import rust_module as _rust_module
from ._penalty_jax_vjp import jax_value_grad_from_rust

__all__ = [
    "latent_json",
    "fixed_weight_schedule",
    "penalty_json",
    "call_value_grad",
    "call_hvp",
    "call_rust_value_grad",
    "jax_value_grad_from_rust",
    "ard_descriptor",
    "ibp_assignment_descriptor",
    "block_orthogonality_descriptor",
    "mechanism_sparsity_descriptor",
    "GumbelTemperatureSchedule",
    "validate_gumbel_schedule_fields",
]


# ---------------------------------------------------------------------------
# JSON construction
# ---------------------------------------------------------------------------


def latent_json(n: int, d: int, *, name: str = "t") -> str:
    """Serialize the single-latent-block registry descriptor."""
    return json.dumps({name: {"name": name, "n": int(n), "d": int(d)}})


def fixed_weight_schedule(weight: float) -> dict[str, Any] | None:
    """A constant ``weight_schedule`` payload, or ``None`` at unit weight."""
    if float(weight) == 1.0:
        return None
    return {
        "w_start": float(weight),
        "w_end": float(weight),
        "kind": "linear",
        "steps": 1,
        "iter_count": 1,
    }


def penalty_json(descriptor: dict[str, Any]) -> str:
    """Serialize a single penalty descriptor as the one-element registry list."""
    return json.dumps([descriptor])


# ---------------------------------------------------------------------------
# Rust value / grad / HVP calls
# ---------------------------------------------------------------------------


def _torch() -> Any:
    from ._protocol import _require_torch

    return _require_torch()


def call_value_grad(
    t_flat: np.ndarray,
    n: int,
    d: int,
    target_name: str,
    descriptor: dict[str, Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    """``(value, ∂value/∂t, ∂value/∂rho)`` for one descriptor at ``t_flat``.

    ``rho=None`` lets the FFI fill a default zero rho sized to the registry's
    total rho count; the descriptor-frame callers never manage rho themselves.
    """
    value, grad, grad_rho, _grad_jac = _rust_module().analytic_penalty_value_grad(
        latent_json(n, d, name=target_name),
        penalty_json(descriptor),
        np.ascontiguousarray(t_flat, dtype=np.float64),
        None,
    )
    return (
        float(value),
        np.asarray(grad, dtype=np.float64),
        np.asarray(grad_rho, dtype=np.float64),
    )


def call_hvp(
    t_flat: np.ndarray,
    v_flat: np.ndarray,
    n: int,
    d: int,
    target_name: str,
    descriptor: dict[str, Any],
) -> np.ndarray:
    """Hessian-vector product ``H · v`` for one descriptor at ``t_flat``."""
    out = _rust_module().analytic_penalty_hvp(
        latent_json(n, d, name=target_name),
        penalty_json(descriptor),
        np.ascontiguousarray(t_flat, dtype=np.float64),
        np.ascontiguousarray(v_flat, dtype=np.float64),
        None,
    )
    return np.asarray(out, dtype=np.float64)


def call_rust_value_grad(
    target: Any,
    rho: Any,
    latents_json: str,
    penalties_json: str,
    *,
    isometry_jacobian: Any = None,
    isometry_jacobian_second: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """Torch-typed ``(value, ∂P/∂t, ∂P/∂rho, ∂P/∂J)`` for an explicit-rho call.

    ``target`` / ``rho`` are torch tensors; the return tensors share their
    dtype / device. The optional isometry-Jacobian arguments route the
    ``IsometryPenalty`` variant that also returns ``∂P/∂J`` as the fourth
    element (``None`` when no Jacobian was supplied).
    """
    from .torch._coerce import from_numpy_like, to_numpy_f64

    kwargs: dict[str, Any] = {}
    if isometry_jacobian is not None:
        kwargs["isometry_jacobian"] = to_numpy_f64(isometry_jacobian)
    if isometry_jacobian_second is not None:
        kwargs["isometry_jacobian_second"] = to_numpy_f64(isometry_jacobian_second)
    value, grad, grad_rho, grad_jac = _rust_module().analytic_penalty_value_grad(
        latents_json,
        penalties_json,
        to_numpy_f64(target.contiguous().reshape(-1)),
        to_numpy_f64(rho.reshape(-1)),
        **kwargs,
    )
    value_t = _torch().as_tensor(value, dtype=target.dtype, device=target.device)
    grad_t = from_numpy_like(grad, target).reshape_as(target)
    grad_rho_t = from_numpy_like(grad_rho, rho).reshape_as(rho)
    grad_jac_t = None if grad_jac is None else from_numpy_like(grad_jac, target)
    return value_t, grad_t, grad_rho_t, grad_jac_t


# ---------------------------------------------------------------------------
# Per-kind descriptor builders (single source of truth for the JSON dicts)
# ---------------------------------------------------------------------------


def ard_descriptor(target: str, weight: float) -> dict[str, Any]:
    """ARD descriptor; folds a constant weight into ``weight_schedule``."""
    desc: dict[str, Any] = {"kind": "ard", "target": str(target)}
    schedule = fixed_weight_schedule(weight)
    if schedule is not None:
        desc["weight_schedule"] = schedule
    return desc


def ibp_assignment_descriptor(
    target: str,
    k_max: int,
    alpha: float,
    tau: float,
    *,
    learnable: bool = False,
) -> dict[str, Any]:
    """Finite-IBP assignment-logit descriptor."""
    return {
        "kind": "ibp_assignment",
        "target": str(target),
        "k_max": int(k_max),
        "alpha": float(alpha),
        "tau": float(tau),
        "learnable": bool(learnable),
    }


def block_orthogonality_descriptor(
    target: str,
    groups: Sequence[Sequence[int]],
    weight: float,
    n_eff: int,
    *,
    learnable: bool = False,
) -> dict[str, Any]:
    """Between-block orthogonality descriptor over latent-axis groups."""
    return {
        "kind": "block_orthogonality",
        "target": str(target),
        "groups": [[int(axis) for axis in group] for group in groups],
        "weight": float(weight),
        "n_eff": int(n_eff),
        "learnable": bool(learnable),
    }


def mechanism_sparsity_descriptor(
    target: str,
    feature_groups: Sequence[Sequence[int]],
    weight: float,
    smoothing_eps: float,
    n_eff: float,
    *,
    learnable: bool = False,
) -> dict[str, Any]:
    """Per-latent group-lasso sparsity descriptor over decoder feature groups."""
    return {
        "kind": "mechanism_sparsity",
        "target": str(target),
        "feature_groups": [[int(f) for f in group] for group in feature_groups],
        "weight": float(weight),
        "smoothing_eps": float(smoothing_eps),
        "n_eff": float(n_eff),
        "learnable": bool(learnable),
    }


# ---------------------------------------------------------------------------
# Gumbel temperature schedule (one class, both interfaces)
# ---------------------------------------------------------------------------


def validate_gumbel_schedule_fields(
    *,
    tau_start: float,
    tau_min: float,
    decay: str,
    rate: float | None,
    steps: int | None,
    iter_count: int,
) -> None:
    """Validate Gumbel schedule fields; raises ``ValueError`` on bad input."""
    if not (np.isfinite(tau_start) and tau_start > 0.0):
        raise ValueError(
            f"GumbelTemperatureSchedule: tau_start must be finite and positive; got {tau_start}"
        )
    if not (np.isfinite(tau_min) and tau_min > 0.0):
        raise ValueError(
            f"GumbelTemperatureSchedule: tau_min must be finite and positive; got {tau_min}"
        )
    if tau_min > tau_start:
        raise ValueError(
            f"GumbelTemperatureSchedule: tau_min ({tau_min}) cannot exceed tau_start ({tau_start})"
        )
    if decay not in {"geometric", "linear", "reciprocal_iter"}:
        raise ValueError(f"GumbelTemperatureSchedule: unknown decay {decay!r}")
    if rate is not None and (not np.isfinite(rate) or rate <= 0.0 or rate >= 1.0):
        raise ValueError(f"GumbelTemperatureSchedule: rate must be in (0, 1); got {rate}")
    if steps is not None and int(steps) < 1:
        raise ValueError(f"GumbelTemperatureSchedule: steps must be >= 1; got {steps}")
    if int(iter_count) < 0:
        raise ValueError(f"GumbelTemperatureSchedule: iter_count must be >= 0; got {iter_count}")


class GumbelTemperatureSchedule:
    """Deterministic Gumbel temperature schedule descriptor.

    Holds the schedule fields plus an iteration counter. Supports both call
    styles in the codebase:

    * stateless — ``current_tau(iter_count)`` evaluates the decay at an
      explicit step without mutating ``self``;
    * stateful — ``step()`` advances the internal ``iter_count`` and returns
      the temperature at the new step.

    The decay arithmetic lives in exactly one place: the Rust
    ``gam.terms.sae_manifold.GumbelTemperatureSchedule`` reached through the
    ``gumbel_schedule_tau`` FFI.
    """

    __slots__ = ("tau_start", "tau_min", "decay", "rate", "steps", "iter_count")

    decay: Literal["geometric", "linear", "reciprocal_iter"]

    def __init__(
        self,
        tau_start: float,
        tau_min: float,
        decay: str = "geometric",
        rate: float | None = None,
        steps: int | None = None,
        iter_count: int = 0,
    ) -> None:
        name = str(decay).lower().replace("-", "_")
        validate_gumbel_schedule_fields(
            tau_start=float(tau_start),
            tau_min=float(tau_min),
            decay=name,
            rate=rate,
            steps=steps,
            iter_count=int(iter_count),
        )
        self.tau_start = float(tau_start)
        self.tau_min = float(tau_min)
        self.decay = name  # type: ignore[assignment]
        # A geometric schedule may be specified either by an explicit `rate` or
        # by the (tau_start, tau_min, steps) endpoints spec (Rust derives the
        # rate). Only fall back to the 0.9 default when neither is given.
        self.rate = (
            0.9
            if (rate is None and steps is None and name == "geometric")
            else rate
        )
        self.steps = steps
        self.iter_count = int(iter_count)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tau_start": self.tau_start,
            "tau_min": self.tau_min,
            "decay": self.decay,
            "iter_count": self.iter_count,
        }
        if self.rate is not None:
            out["rate"] = float(self.rate)
        if self.steps is not None:
            out["steps"] = int(self.steps)
        return out

    def current_tau(self, iter_count: int | None = None) -> float:
        """Temperature at ``iter_count`` (defaults to the internal counter),
        evaluated by the Rust ``GumbelTemperatureSchedule``."""
        step = self.iter_count if iter_count is None else int(iter_count)
        return float(
            _rust_module().gumbel_schedule_tau(self.to_rust_descriptor(), int(step))
        )

    def step(self) -> float:
        """Advance the internal counter and return the temperature there."""
        self.iter_count += 1
        return self.current_tau()
