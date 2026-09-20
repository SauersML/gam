"""Single source of truth for the Python ↔ Rust analytic-penalty bridge.

Every penalty surface in gamfit — the ``nn.Module`` autograd shells in
:mod:`gamfit.torch.penalties`, the penalty-wrapper adapters in
:mod:`gamfit._penalty_frames`, and the SAE regularizer payload in
:mod:`gamfit._sae_manifold` — shares the same plumbing:

* build the latent / penalty JSON the Rust registry consumes
  (:func:`latent_json`, :func:`penalty_json`),
* run ``analytic_penalty_value_grad`` once at an explicit rho
  (:func:`call_rust_value_grad`, including the isometry-Jacobian variant),
* wrap the result for the active backend (the JAX ``custom_vjp`` core lives
  in :func:`jax_value_grad_from_rust`, re-exported here),
* and anneal the Gumbel temperature through exactly one
  :class:`GumbelTemperatureSchedule`.

No penalty math is reimplemented here. The Rust trait in
``src/terms/analytic_penalties.rs`` and ``src/terms/sae_manifold.rs`` are the
single source of truth for the math; this module only marshals JSON / arrays.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from ._binding import rust_module as _rust_module
from ._penalty_jax_vjp import jax_value_grad_from_rust

__all__ = [
    "latent_json",
    "penalty_json",
    "torch_value_grad_from_rust",
    "call_rust_value_grad",
    "jax_value_grad_from_rust",
    "GumbelTemperatureSchedule",
]


# ---------------------------------------------------------------------------
# JSON construction
# ---------------------------------------------------------------------------


def latent_json(n: int, d: int, *, name: str = "t") -> str:
    """Serialize the single-latent-block registry descriptor."""
    return json.dumps({name: {"name": name, "n": int(n), "d": int(d)}})


def penalty_json(descriptor: dict[str, Any]) -> str:
    """Serialize a single penalty descriptor as the one-element registry list."""
    return json.dumps([descriptor])


# ---------------------------------------------------------------------------
# Rust value / grad / HVP calls
# ---------------------------------------------------------------------------


def _torch() -> Any:
    from ._protocol import _require_torch

    return _require_torch()


def torch_value_grad_from_rust(
    t: Any,
    value_grad_np: Any,
    hvp_np: Any,
) -> tuple[Any, Any]:
    """Torch ``(value, grad)`` at ``t`` from the Rust gradient and HVP kernels.

    ``value_grad_np(x) -> (value, grad)`` and ``hvp_np(x, v) -> H·v`` take and
    return NumPy arrays shaped like ``t``. ``value`` is autograd-connected to
    ``t`` through the Rust gradient, and that gradient is itself connected
    through the Rust Hessian-vector product, so differentiating a
    ``create_graph=True`` gradient of ``value`` reaches the analytic Hessian
    instead of a detached constant.
    """
    from ._frame_torch import from_numpy_like, to_numpy_f64

    _torch()  # clean ImportError when torch is missing
    import torch

    value_np, grad_np = value_grad_np(to_numpy_f64(t))

    class _Grad(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            ctx.save_for_backward(x)
            _value, grad = value_grad_np(to_numpy_f64(x))
            return from_numpy_like(grad, x)

        @staticmethod
        def backward(ctx: Any, v: Any) -> Any:
            (x,) = ctx.saved_tensors
            return from_numpy_like(hvp_np(to_numpy_f64(x), to_numpy_f64(v)), x)

    class _Value(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            ctx.save_for_backward(x)
            return torch.as_tensor(float(value_np), dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:
            (x,) = ctx.saved_tensors
            return _Grad.apply(x) * grad_out

    return _Value.apply(t), from_numpy_like(grad_np, t)


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
# Gumbel temperature schedule (one class, both interfaces)
# ---------------------------------------------------------------------------


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
        self.tau_start = float(tau_start)
        self.tau_min = float(tau_min)
        self.decay = str(decay).lower().replace("-", "_")  # type: ignore[assignment]
        # A geometric schedule takes exactly one of an explicit `rate` or the
        # (tau_start, tau_min, steps) endpoints; there is no default rate. The
        # Rust descriptor parser derives the rate from `steps` and validates
        # every field; parsing it once here refuses a bad schedule at
        # construction.
        self.rate = rate
        self.steps = steps
        self.iter_count = int(iter_count)
        self.current_tau()

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
