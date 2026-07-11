"""Torch-aware :class:`PenaltyDescriptor` wrappers around the Rust analytic
penalty registry.

Every penalty's ``value / value_grad / hvp / hessian_diag`` call routes
through the existing Rust pyfunctions ``analytic_penalty_value_grad`` and
``analytic_penalty_hvp`` (see ``crates/gam-pyffi/src/lib.rs``). No penalty
math is reimplemented in Python — these wrappers are thin torch.autograd
adapters that produce tensors with grad flowing back through ``t``.

The Rust trait in ``src/terms/analytic_penalties.rs`` is the single source
of truth for the math: ``value(target, rho)``, ``gradient(target, rho)``,
``hvp(target, rho, v)``, optional ``hessian_diag(target, rho)``.

Each descriptor is a torch-aware *callable*: ``ARDPenalty(0.1)(t)`` returns
the scalar penalty value as a torch tensor; ``+`` composes two penalties
into a :class:`CompositePenalty`.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ._penalty_bridge import (
    ard_descriptor,
    block_orthogonality_descriptor,
    call_hvp as _call_hvp,
    call_value_grad as _call_value_grad,
    ordered_beta_bernoulli_descriptor,
    mechanism_sparsity_descriptor,
)
from ._protocol import PenaltyDescriptor, _require_torch


def _to_numpy_f64(t: Any) -> np.ndarray:
    torch = _require_torch()
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().to(torch.float64).contiguous().numpy().reshape(-1).astype(np.float64, copy=False)
    arr = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
    return arr.reshape(-1)


def _infer_shape(t: Any) -> tuple[int, int]:
    # Avoid importing torch eagerly: most non-torch frames (numpy / jax)
    # expose a stable ``.shape`` tuple we can read directly.
    shape = getattr(t, "shape", None)
    if shape is None:
        arr = np.asarray(t)
        shape = arr.shape
    ndim = len(shape)
    if ndim == 1:
        return int(shape[0]), 1
    if ndim == 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"penalty target must be 1-D or 2-D, got {ndim}-D")


class _PenaltyValueFn:
    """Wrap the value/grad/hvp Rust calls behind a torch autograd.Function.

    Implemented as a class factory because ``torch`` may not be importable
    at module load time; we build the autograd.Function on first use.
    """

    _impl = None

    @classmethod
    def get(cls) -> Any:
        if cls._impl is not None:
            return cls._impl
        torch = _require_torch()

        class _Impl(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, t: Any, n: int, d: int, target_name: str, descriptor_json: str) -> Any:
                descriptor = json.loads(descriptor_json)
                t_np = _to_numpy_f64(t)
                value, grad, _ = _call_value_grad(t_np, n, d, target_name, descriptor)
                ctx.save_for_backward(t.detach())
                ctx.n = n
                ctx.d = d
                ctx.target_name = target_name
                ctx.descriptor_json = descriptor_json
                ctx.grad_cache = grad
                return torch.as_tensor(value, dtype=t.dtype, device=t.device)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, None, None, None, None]:
                (t,) = ctx.saved_tensors
                grad = ctx.grad_cache
                grad_t = torch.as_tensor(grad, dtype=t.dtype, device=t.device).reshape_as(t)
                return grad_output.to(dtype=t.dtype, device=t.device) * grad_t, None, None, None, None

        cls._impl = _Impl
        return cls._impl


class _RustPenaltyDescriptor(PenaltyDescriptor):
    """Mixin for analytic penalties whose ``value / value_grad / hvp`` is
    delegated to ``analytic_penalty_value_grad`` / ``analytic_penalty_hvp``.

    Subclasses override :meth:`_descriptor` to return the Rust JSON descriptor.
    """

    target_name: str = "t"

    def _descriptor(self, n: int, d: int) -> dict[str, Any]:
        raise NotImplementedError

    def value(self, t: Any) -> Any:
        """Penalty value ``P(t)`` in the frame of ``t``.

        Frame is auto-detected (numpy / torch / jax). The Rust kernel runs
        once; the output type matches the input frame. Torch / JAX outputs
        carry an autograd graph back to ``t``.
        """
        from ._frame import Frame, detect_frame

        frame = detect_frame(t)
        if frame is Frame.NUMPY:
            v, _g = self.value_grad(t)
            return v
        if frame is Frame.JAX:
            v, _g = self.value_grad(t)
            return v
        torch = _require_torch()
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float64)
        if not torch.is_floating_point(t):
            t = t.to(torch.float64)
        n, d = _infer_shape(t)
        descriptor = self._descriptor(n, d)
        fn = _PenaltyValueFn.get()
        return fn.apply(t.contiguous(), n, d, self.target_name, json.dumps(descriptor))

    def value_grad(self, t: Any) -> tuple[Any, Any]:
        """``(value, ∂value/∂t)`` in the frame of ``t``.

        See :meth:`value` for the frame-dispatch contract.
        """
        from ._frame import Frame, detect_frame

        frame = detect_frame(t)
        if frame is Frame.NUMPY:
            t_np = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
            n, d = _infer_shape(t_np)
            descriptor = self._descriptor(n, d)
            value, grad, _ = _call_value_grad(
                t_np.reshape(-1), n, d, self.target_name, descriptor
            )
            return float(value), grad.reshape(t_np.shape)
        if frame is Frame.JAX:
            return _jax_value_grad_via_rust(self, t)
        torch = _require_torch()
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float64)
        if not torch.is_floating_point(t):
            t = t.to(torch.float64)
        n, d = _infer_shape(t)
        descriptor = self._descriptor(n, d)
        t_np = _to_numpy_f64(t)
        value, grad, _ = _call_value_grad(t_np, n, d, self.target_name, descriptor)
        v = torch.as_tensor(value, dtype=t.dtype, device=t.device)
        g = torch.as_tensor(grad, dtype=t.dtype, device=t.device).reshape_as(t)
        return v, g

    def to_rust_descriptor(self) -> dict[str, Any]:
        """Return the JSON descriptor consumed by the Rust analytic-penalty
        registry. Mirrors the formula-API contract used by existing penalty
        dataclasses so this descriptor can drop into ``gamfit.fit(...,
        penalties=[...])`` flows."""
        # Default shape (n=1, d=1) is a placeholder; the formula pipeline
        # supplies the real shape when the registry is materialized.
        return self._descriptor(1, 1)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict serialization (alias of ``to_rust_descriptor``)."""
        return self.to_rust_descriptor()

    def hvp(self, t: Any, v: Any) -> Any:
        """Hessian-vector product ``H · v`` in the frame shared by ``t`` and ``v``.

        Both arguments must be in the same frame; a mismatch raises
        :class:`TypeError` (see :func:`gamfit._frame.detect_frame`).
        """
        from ._frame import Frame, detect_frame

        frame = detect_frame(t, v)
        if frame is Frame.NUMPY:
            t_np = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
            v_np = np.ascontiguousarray(np.asarray(v, dtype=np.float64))
            if t_np.shape != v_np.shape:
                raise ValueError(
                    f"hvp target shape {t_np.shape} must match v shape {v_np.shape}"
                )
            n, d = _infer_shape(t_np)
            descriptor = self._descriptor(n, d)
            out = _call_hvp(
                t_np.reshape(-1), v_np.reshape(-1), n, d, self.target_name, descriptor
            )
            return out.reshape(t_np.shape)
        if frame is Frame.JAX:
            from ._frame_jax import from_numpy_like, to_numpy_f64

            t_np = to_numpy_f64(t).reshape(-1)
            v_np = to_numpy_f64(v).reshape(-1)
            shape = tuple(int(s) for s in getattr(t, "shape", (t_np.size,)))
            n, d = _infer_shape(np.asarray(t, dtype=np.float64))
            descriptor = self._descriptor(n, d)
            out = _call_hvp(t_np, v_np, n, d, self.target_name, descriptor)
            return from_numpy_like(out.reshape(shape), t)
        torch = _require_torch()
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float64)
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v, dtype=t.dtype)
        if t.shape != v.shape:
            raise ValueError(
                f"hvp target shape {tuple(t.shape)} must match v shape {tuple(v.shape)}"
            )
        n, d = _infer_shape(t)
        descriptor = self._descriptor(n, d)
        t_np = _to_numpy_f64(t)
        v_np = _to_numpy_f64(v)
        out = _call_hvp(t_np, v_np, n, d, self.target_name, descriptor)
        return torch.as_tensor(out, dtype=t.dtype, device=t.device).reshape_as(t)


def _jax_value_grad_via_rust(descriptor_obj: Any, t: Any) -> tuple[Any, Any]:
    """JAX-frame ``(value, grad)`` for a :class:`_RustPenaltyDescriptor`.

    Thin adapter over :func:`jax_value_grad_from_rust`: the only
    descriptor-specific piece is the host callback that runs
    ``analytic_penalty_value_grad`` for this descriptor's JSON.
    """
    from ._penalty_bridge import jax_value_grad_from_rust

    shape = tuple(int(s) for s in t.shape)
    n, d = _infer_shape(np.asarray(t, dtype=np.float64))
    descriptor = descriptor_obj._descriptor(n, d)
    target_name = descriptor_obj.target_name

    def _callback(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad, _grad_rho = _call_value_grad(
            x_np.reshape(-1), n, d, target_name, descriptor
        )
        return value, grad.reshape(shape)

    return jax_value_grad_from_rust(
        descriptor.get("kind", "penalty"), shape, _callback, ref=t
    )


class ARDPenalty(_RustPenaltyDescriptor):
    """Automatic relevance determination penalty over latent axes.

    The Rust analytic ARD penalty is ``½ exp(rho) · ‖t‖²`` summed over axes
    (with per-axis ``rho``). At construction we pin ``rho = log(weight)``
    uniformly across axes; users who need REML-selected weights should use
    the formula API / ``gamfit.fit`` instead.
    """

    def __init__(self, weight: float = 1.0, *, target: str = "t") -> None:
        if float(weight) <= 0.0:
            raise ValueError("ARDPenalty.weight must be > 0")
        self.weight = float(weight)
        self.target_name = str(target)

    def _descriptor(self, n: int, d: int) -> dict[str, Any]:
        return ard_descriptor(self.target_name, self.weight)

    def __repr__(self) -> str:
        return f"ARDPenalty(weight={self.weight})"


class IBPPenalty(_RustPenaltyDescriptor):
    """Finite IBP prior over row-wise assignment logits."""

    def __init__(self, alpha: float = 1.0, *, tau: float = 1.0, k_max: int | None = None, target: str = "t") -> None:
        if float(alpha) <= 0.0:
            raise ValueError("IBPPenalty.alpha must be > 0")
        if float(tau) <= 0.0:
            raise ValueError("IBPPenalty.tau must be > 0")
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.k_max = None if k_max is None else int(k_max)
        self.target_name = str(target)

    def _descriptor(self, n: int, d: int) -> dict[str, Any]:
        k_max = self.k_max if self.k_max is not None else d
        return ordered_beta_bernoulli_descriptor(
            self.target_name, int(k_max), self.alpha, self.tau
        )

    def __repr__(self) -> str:
        return f"IBPPenalty(alpha={self.alpha}, tau={self.tau})"


class BlockOrthogonalityDescriptor(_RustPenaltyDescriptor):
    """Penalize between-block cross-products of latent axes."""

    def __init__(
        self,
        groups: list[list[int]],
        weight: float = 1.0,
        *,
        n_eff: int | None = None,
        target: str = "t",
    ) -> None:
        if float(weight) <= 0.0:
            raise ValueError("BlockOrthogonality.weight must be > 0")
        self.groups = [[int(a) for a in g] for g in groups]
        self.weight = float(weight)
        self.n_eff = None if n_eff is None else int(n_eff)
        self.target_name = str(target)

    def _descriptor(self, n: int, d: int) -> dict[str, Any]:
        return block_orthogonality_descriptor(
            self.target_name, self.groups, self.weight, int(self.n_eff or n)
        )


class MechanismSparsityDescriptor(_RustPenaltyDescriptor):
    """Per-latent group-lasso sparsity on decoder feature groups."""

    def __init__(
        self,
        feature_groups: list[list[int]],
        weight: float = 1.0,
        *,
        n_eff: float | None = None,
        smoothing_eps: float = 1e-6,
        target: str = "t",
    ) -> None:
        if float(weight) <= 0.0:
            raise ValueError("MechanismSparsity.weight must be > 0")
        self.feature_groups = [[int(f) for f in g] for g in feature_groups]
        self.weight = float(weight)
        self.n_eff = None if n_eff is None else float(n_eff)
        self.smoothing_eps = float(smoothing_eps)
        self.target_name = str(target)

    def _descriptor(self, n: int, d: int) -> dict[str, Any]:
        return mechanism_sparsity_descriptor(
            self.target_name,
            self.feature_groups,
            self.weight,
            self.smoothing_eps,
            float(self.n_eff if self.n_eff is not None else n),
        )


__all__ = [
    "ARDPenalty",
    "IBPPenalty",
    "BlockOrthogonalityDescriptor",
    "MechanismSparsityDescriptor",
]
