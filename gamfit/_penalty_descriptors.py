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
import math
from typing import Any

import numpy as np

from ._binding import rust_module as _rust_module
from ._protocol import PenaltyDescriptor, _require_torch


def _latent_json(n: int, d: int, *, name: str = "t") -> str:
    return json.dumps({name: {"name": name, "n": int(n), "d": int(d)}})


def _fixed_weight_schedule(weight: float) -> dict[str, Any] | None:
    if float(weight) == 1.0:
        return None
    return {
        "w_start": float(weight),
        "w_end": float(weight),
        "kind": "linear",
        "steps": 1,
        "iter_count": 1,
    }


def _penalty_json(descriptor: dict[str, Any]) -> str:
    return json.dumps([descriptor])


def _to_numpy_f64(t: Any) -> np.ndarray:
    torch = _require_torch()
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().to(torch.float64).contiguous().numpy().reshape(-1).astype(np.float64, copy=False)
    arr = np.ascontiguousarray(np.asarray(t, dtype=np.float64))
    return arr.reshape(-1)


def _call_value_grad(
    t_flat: np.ndarray, n: int, d: int, target_name: str, descriptor: dict[str, Any]
) -> tuple[float, np.ndarray, np.ndarray]:
    latents_json = _latent_json(n, d, name=target_name)
    penalties_json = _penalty_json(descriptor)
    rho = np.zeros(0, dtype=np.float64)
    value, grad, grad_rho = _rust_module().analytic_penalty_value_grad(
        latents_json, penalties_json, t_flat, rho,
    )
    return float(value), np.asarray(grad, dtype=np.float64), np.asarray(grad_rho, dtype=np.float64)


def _call_hvp(
    t_flat: np.ndarray, v_flat: np.ndarray, n: int, d: int, target_name: str, descriptor: dict[str, Any]
) -> np.ndarray:
    latents_json = _latent_json(n, d, name=target_name)
    penalties_json = _penalty_json(descriptor)
    rho = np.zeros(0, dtype=np.float64)
    out = _rust_module().analytic_penalty_hvp(
        latents_json, penalties_json, t_flat, v_flat, rho,
    )
    return np.asarray(out, dtype=np.float64)


def _infer_shape(t: Any) -> tuple[int, int]:
    torch = _require_torch()
    if isinstance(t, torch.Tensor):
        if t.dim() == 1:
            return int(t.shape[0]), 1
        if t.dim() == 2:
            return int(t.shape[0]), int(t.shape[1])
        raise ValueError(f"penalty target must be 1-D or 2-D, got {t.dim()}-D")
    arr = np.asarray(t)
    if arr.ndim == 1:
        return int(arr.shape[0]), 1
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError(f"penalty target must be 1-D or 2-D, got {arr.ndim}-D")


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

    def hvp(self, t: Any, v: Any) -> Any:
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
        desc: dict[str, Any] = {"kind": "ard", "target": self.target_name}
        sched = _fixed_weight_schedule(self.weight)
        if sched is not None:
            desc["weight_schedule"] = sched
        return desc

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
        return {
            "kind": "ibp_assignment",
            "target": self.target_name,
            "k_max": int(k_max),
            "alpha": self.alpha,
            "tau": self.tau,
            "learnable": False,
        }

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
        return {
            "kind": "block_orthogonality",
            "target": self.target_name,
            "groups": self.groups,
            "weight": self.weight,
            "n_eff": int(self.n_eff or n),
            "learnable": False,
        }


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
        return {
            "kind": "mechanism_sparsity",
            "target": self.target_name,
            "feature_groups": self.feature_groups,
            "weight": self.weight,
            "smoothing_eps": self.smoothing_eps,
            "n_eff": float(self.n_eff if self.n_eff is not None else n),
            "learnable": False,
        }


__all__ = [
    "ARDPenalty",
    "IBPPenalty",
    "BlockOrthogonalityDescriptor",
    "MechanismSparsityDescriptor",
]
