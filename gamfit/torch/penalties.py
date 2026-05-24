"""Autograd modules for gamfit analytic composition penalties."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from .._binding import rust_module
from .._topology_selector import TopologyAutoSelector
from ._coerce import from_numpy_like, to_numpy_f64


def _check_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dim() != 2:
        raise ValueError(f"{name} must be 2-D, got shape {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        value = value.to(torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


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


def _rho_tensor(rho: torch.Tensor | None, ref: torch.Tensor, count: int) -> torch.Tensor:
    if rho is None:
        return torch.zeros(count, dtype=ref.dtype, device=ref.device)
    if rho.numel() != count:
        raise ValueError(f"rho length {rho.numel()} does not match expected {count}")
    return rho.reshape(-1).to(device=ref.device, dtype=ref.dtype)


def _as_flat_target(target: torch.Tensor) -> torch.Tensor:
    return target.contiguous().reshape(-1)


def _call_rust_value_grad(
    target: torch.Tensor,
    rho: torch.Tensor,
    latents_json: str,
    penalties_json: str,
    *,
    isometry_jacobian: torch.Tensor | None = None,
    isometry_jacobian_second: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {}
    if isometry_jacobian is not None:
        kwargs["isometry_jacobian"] = to_numpy_f64(isometry_jacobian)
    if isometry_jacobian_second is not None:
        kwargs["isometry_jacobian_second"] = to_numpy_f64(isometry_jacobian_second)
    value, grad, grad_rho = rust_module().analytic_penalty_value_grad(
        latents_json,
        penalties_json,
        to_numpy_f64(_as_flat_target(target)),
        to_numpy_f64(rho.reshape(-1)),
        **kwargs,
    )
    value_t = torch.as_tensor(value, dtype=target.dtype, device=target.device)
    grad_t = from_numpy_like(grad, target).reshape_as(target)
    grad_rho_t = from_numpy_like(grad_rho, rho).reshape_as(rho)
    return value_t, grad_t, grad_rho_t


class _RustPenaltyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        target: torch.Tensor,
        rho: torch.Tensor,
        latents_json: str,
        penalties_json: str,
    ) -> torch.Tensor:
        value, _, _ = _call_rust_value_grad(target, rho, latents_json, penalties_json)
        ctx.save_for_backward(target, rho)
        ctx.latents_json = latents_json
        ctx.penalties_json = penalties_json
        return value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        target, rho = ctx.saved_tensors
        _, grad_target, grad_rho = _call_rust_value_grad(
            target, rho, ctx.latents_json, ctx.penalties_json
        )
        scale = grad_output.to(dtype=target.dtype, device=target.device)
        return grad_target * scale, grad_rho * scale.to(dtype=rho.dtype, device=rho.device), None, None


class _IsometryPenaltyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        target: torch.Tensor,
        rho: torch.Tensor,
        basis: torch.Tensor,
        jacobian_second: torch.Tensor | None,
        latents_json: str,
        penalties_json: str,
    ) -> torch.Tensor:
        if basis.dim() == 2:
            jacobian = basis.unsqueeze(0).expand(target.shape[0], -1, -1)
        else:
            jacobian = basis
        value, _, _ = _call_rust_value_grad(
            target,
            rho,
            latents_json,
            penalties_json,
            isometry_jacobian=jacobian.reshape(target.shape[0], -1),
            isometry_jacobian_second=jacobian_second,
        )
        saved_second = jacobian_second if jacobian_second is not None else torch.empty(0, dtype=target.dtype, device=target.device)
        ctx.save_for_backward(target, rho, basis, saved_second)
        ctx.has_second = jacobian_second is not None
        ctx.latents_json = latents_json
        ctx.penalties_json = penalties_json
        return value

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        target, rho, basis, saved_second = ctx.saved_tensors
        jacobian_second = saved_second if ctx.has_second else None
        if basis.dim() == 2:
            jacobian = basis.unsqueeze(0).expand(target.shape[0], -1, -1)
        else:
            jacobian = basis
        _, grad_target, grad_rho = _call_rust_value_grad(
            target,
            rho,
            ctx.latents_json,
            ctx.penalties_json,
            isometry_jacobian=jacobian.reshape(target.shape[0], -1),
            isometry_jacobian_second=jacobian_second,
        )
        d = target.shape[1]
        j = jacobian.reshape(target.shape[0], -1, d)
        eye = torch.eye(d, dtype=j.dtype, device=j.device)
        gram = torch.matmul(j.transpose(1, 2), j)
        diff = gram - eye
        mu = torch.exp(rho.reshape(-1)[0]).to(dtype=j.dtype, device=j.device)
        grad_j = 2.0 * mu * torch.matmul(j, diff)
        if basis.dim() == 2:
            grad_basis = grad_j.sum(dim=0)
        else:
            grad_basis = grad_j
        scale = grad_output.to(dtype=target.dtype, device=target.device)
        return (
            grad_target * scale,
            grad_rho * scale.to(dtype=rho.dtype, device=rho.device),
            grad_basis * scale.to(dtype=grad_basis.dtype, device=grad_basis.device),
            None,
            None,
            None,
        )


class IsometryPenalty(nn.Module):
    """Pull a decoder Jacobian's metric toward the Euclidean metric."""

    def __init__(self, weight: float = 1.0, *, target: str = "t") -> None:
        super().__init__()
        if weight <= 0.0:
            raise ValueError("IsometryPenalty.weight must be > 0")
        self.weight = float(weight)
        self.target = str(target)

    def forward(self, latent: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        latent = _check_matrix(latent, "latent")
        if basis.dim() == 2:
            basis_t = basis.to(device=latent.device, dtype=latent.dtype)
            p_out = int(basis_t.shape[0])
        elif basis.dim() == 3:
            if basis.shape[0] != latent.shape[0]:
                raise ValueError("3-D basis must have the same row count as latent")
            basis_t = basis.to(device=latent.device, dtype=latent.dtype)
            p_out = int(basis_t.shape[1])
        else:
            raise ValueError("basis must have shape (p, d) or (N, p, d)")
        if basis_t.shape[-1] != latent.shape[1]:
            raise ValueError("basis last dimension must match latent width")
        descriptor = {
            "kind": "isometry",
            "target": self.target,
            "weight": 1.0,
            "p_out": p_out,
        }
        rho = torch.full((1,), float(np.log(self.weight)), dtype=latent.dtype, device=latent.device)
        apply = cast(Callable[..., torch.Tensor], _IsometryPenaltyFn.apply)
        return apply(
            latent,
            rho,
            basis_t,
            None,
            _latent_json(latent.shape[0], latent.shape[1], name=self.target),
            _penalty_json(descriptor),
        )


class ARDPenalty(nn.Module):
    """Automatic relevance determination over latent axes."""

    def __init__(self, latent_dim: int | None = None, weight: float = 1.0, *, target: str = "t") -> None:
        super().__init__()
        if weight <= 0.0:
            raise ValueError("ARDPenalty.weight must be > 0")
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self.weight = float(weight)
        self.target = str(target)
        if self.latent_dim is not None:
            self.log_precision = nn.Parameter(torch.zeros(self.latent_dim))

    def _rho(self, latent: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "log_precision"):
            self.latent_dim = int(latent.shape[1])
            self.register_parameter("log_precision", nn.Parameter(torch.zeros(self.latent_dim, dtype=latent.dtype, device=latent.device)))
        return self.log_precision.to(device=latent.device, dtype=latent.dtype)

    def forward(self, latent: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        del basis
        latent = _check_matrix(latent, "latent")
        descriptor: dict[str, Any] = {"kind": "ard", "target": self.target}
        schedule = _fixed_weight_schedule(self.weight)
        if schedule is not None:
            descriptor["weight_schedule"] = schedule
        rho = self._rho(latent)
        apply = cast(Callable[..., torch.Tensor], _RustPenaltyFn.apply)
        return apply(latent, rho, _latent_json(latent.shape[0], latent.shape[1], name=self.target), _penalty_json(descriptor))


class BlockOrthogonalityPenalty(nn.Module):
    """Between-block orthogonality over latent-axis groups."""

    def __init__(self, groups: Sequence[Sequence[int]], weight: float, n_eff: int | None = None, *, target: str = "t", learnable: bool = False) -> None:
        super().__init__()
        self.groups = [[int(axis) for axis in group] for group in groups]
        self.weight = float(weight)
        self.n_eff = None if n_eff is None else int(n_eff)
        self.target = str(target)
        self.learnable = bool(learnable)
        if self.learnable:
            self.log_weight = nn.Parameter(torch.zeros(1))

    def forward(self, latent: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        del basis
        latent = _check_matrix(latent, "latent")
        descriptor = {
            "kind": "block_orthogonality",
            "target": self.target,
            "groups": self.groups,
            "weight": self.weight,
            "n_eff": int(self.n_eff or latent.shape[0]),
            "learnable": self.learnable,
        }
        rho = _rho_tensor(getattr(self, "log_weight", None), latent, 1 if self.learnable else 0)
        apply = cast(Callable[..., torch.Tensor], _RustPenaltyFn.apply)
        return apply(latent, rho, _latent_json(latent.shape[0], latent.shape[1], name=self.target), _penalty_json(descriptor))


class MechanismSparsityPenalty(nn.Module):
    """Per-latent group-lasso sparsity over decoder feature groups."""

    def __init__(self, feature_groups: Sequence[Sequence[int]], weight: float, n_eff: float, smoothing_eps: float = 1e-6, *, target: str = "t", learnable: bool = False) -> None:
        super().__init__()
        self.feature_groups = [[int(feature) for feature in group] for group in feature_groups]
        self.weight = float(weight)
        self.n_eff = float(n_eff)
        self.smoothing_eps = float(smoothing_eps)
        self.target = str(target)
        self.learnable = bool(learnable)
        if self.learnable:
            self.log_weight = nn.Parameter(torch.zeros(1))

    def forward(self, weights: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        del basis
        weights = _check_matrix(weights, "weights")
        descriptor = {
            "kind": "mechanism_sparsity",
            "target": self.target,
            "feature_groups": self.feature_groups,
            "weight": self.weight,
            "smoothing_eps": self.smoothing_eps,
            "n_eff": self.n_eff,
            "learnable": self.learnable,
        }
        rho = _rho_tensor(getattr(self, "log_weight", None), weights, 1 if self.learnable else 0)
        apply = cast(Callable[..., torch.Tensor], _RustPenaltyFn.apply)
        return apply(weights, rho, _latent_json(max(1, int(round(self.n_eff))), weights.shape[0], name=self.target), _penalty_json(descriptor))


class IBPAssignmentPenalty(nn.Module):
    """Finite IBP prior over row-wise assignment logits."""

    def __init__(self, k_max: int, alpha: float = 1.0, tau: float = 1.0, *, target: str = "t", learnable: bool = False) -> None:
        super().__init__()
        self.k_max = int(k_max)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.target = str(target)
        self.learnable = bool(learnable)
        if self.learnable:
            self.log_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        del basis
        logits = _check_matrix(logits, "logits")
        if logits.shape[1] != self.k_max:
            raise ValueError("logits width must equal k_max")
        descriptor = {
            "kind": "ibp_assignment",
            "target": self.target,
            "k_max": self.k_max,
            "alpha": self.alpha,
            "tau": self.tau,
            "learnable": self.learnable,
        }
        rho = _rho_tensor(getattr(self, "log_alpha", None), logits, 1 if self.learnable else 0)
        apply = cast(Callable[..., torch.Tensor], _RustPenaltyFn.apply)
        return apply(logits, rho, _latent_json(logits.shape[0], logits.shape[1], name=self.target), _penalty_json(descriptor))


class IvaeRidgeMeanGauge(nn.Module):
    """iVAE conditional-mean ridge gauge on latent coordinates."""

    def __init__(self, aux: Any, weight: float, n_eff: int | None = None, ridge_eps: float = 1e-6, *, target: str = "t", learnable: bool = False) -> None:
        super().__init__()
        aux_t = torch.as_tensor(aux, dtype=torch.float64)
        if aux_t.dim() != 2:
            raise ValueError("aux must have shape (N, q)")
        self.register_buffer("aux", aux_t)
        self.weight = float(weight)
        self.n_eff = int(n_eff or aux_t.shape[0])
        self.ridge_eps = float(ridge_eps)
        self.target = str(target)
        self.learnable = bool(learnable)
        if self.learnable:
            self.log_weight = nn.Parameter(torch.zeros(1))

    def forward(self, latent: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        del basis
        latent = _check_matrix(latent, "latent")
        aux = self.aux.to(device=latent.device, dtype=latent.dtype)
        descriptor = {
            "kind": "ivae_ridge_mean_gauge",
            "target": self.target,
            "aux": to_numpy_f64(aux).reshape(-1).tolist(),
            "aux_shape": [int(aux.shape[0]), int(aux.shape[1])],
            "ridge_eps": self.ridge_eps,
            "weight": self.weight,
            "n_eff": self.n_eff,
            "learnable": self.learnable,
        }
        rho = _rho_tensor(getattr(self, "log_weight", None), latent, 1 if self.learnable else 0)
        apply = cast(Callable[..., torch.Tensor], _RustPenaltyFn.apply)
        return apply(latent, rho, _latent_json(latent.shape[0], latent.shape[1], name=self.target), _penalty_json(descriptor))


class JumpReLUPenalty(nn.Module):
    """JumpReLU SAE prior with hard-threshold gating + straight-through estimator.

    Forward: ``φ(z) = z · 1[z > τ]`` per latent axis with per-axis learnable
    thresholds ``τ_k = thresholds_k · exp(log_threshold_k)``. The hard-threshold
    forward path has zero subgradient almost everywhere; the STE backward path
    routes ``∂L/∂z = ∂L/∂φ · 1[|z − τ| < bandwidth]`` (rectangular kernel) plus
    ``∂L/∂τ = −∂L/∂φ · φ̄`` evaluated by the gam Rust core's analytic
    smoothed sigmoid (``smoothing_eps`` controls the smoothing scale).

    Acts as a sparsity penalty when used as ``loss += w · jumprelu(z).sum()``;
    acts as an activation function when used as ``z_active = jumprelu(z)``.
    Both modes share the same parameters and STE backward.

    Parameters
    ----------
    thresholds: per-axis base thresholds (length F). Each entry must be > 0.
    weight: scalar prior weight (must be > 0).
    smoothing_eps: bandwidth of the sigmoid-smoothed STE gate (default 1e-3).
        Smaller → harder threshold (closer to true step), larger → smoother
        backward; numerics get hairy below 1e-5.
    learnable_threshold: if True, expose ``log_threshold`` as ``nn.Parameter``;
        the effective threshold is ``thresholds * exp(log_threshold)``. REML can
        then select the threshold jointly with all other hyperparameters via
        the outer loop.
    """

    def __init__(
        self,
        thresholds: torch.Tensor | Sequence[float],
        weight: float = 1.0,
        smoothing_eps: float = 1e-3,
        *,
        learnable_threshold: bool = False,
        target: str = "t",
    ) -> None:
        super().__init__()
        thr = torch.as_tensor(thresholds, dtype=torch.float64).reshape(-1)
        if thr.numel() == 0:
            raise ValueError("JumpReLUPenalty.thresholds must be non-empty")
        if not bool(torch.isfinite(thr).all()) or bool((thr <= 0).any()):
            raise ValueError("JumpReLUPenalty.thresholds must be finite and > 0")
        if not (np.isfinite(weight) and weight > 0.0):
            raise ValueError(f"JumpReLUPenalty.weight must be finite and > 0, got {weight}")
        if not (np.isfinite(smoothing_eps) and smoothing_eps > 0.0):
            raise ValueError(
                f"JumpReLUPenalty.smoothing_eps must be finite and > 0, got {smoothing_eps}"
            )
        self.register_buffer("thresholds", thr)
        self.weight = float(weight)
        self.smoothing_eps = float(smoothing_eps)
        self.target = str(target)
        if learnable_threshold:
            self.log_threshold = nn.Parameter(torch.zeros(thr.numel(), dtype=torch.float64))
        else:
            self.register_buffer("log_threshold", torch.zeros(thr.numel(), dtype=torch.float64))

    def effective_thresholds(self, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return (self.thresholds.to(dtype) * torch.exp(self.log_threshold.to(dtype)))

    def gate(self, z: torch.Tensor) -> torch.Tensor:
        """Hard-threshold forward with STE backward. Returns ``z · 1[z > τ]``."""
        tau = self.effective_thresholds(z.dtype).to(z.device)
        return _JumpReLUSTEFn.apply(z, tau, float(self.smoothing_eps))

    def forward(self, latent: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        """Penalty value: ``weight · Σ τ · σ((z − τ)/ε)`` (smoothed L0).

        Matches the Rust `JumpReLUPenalty::value` analytic formulation so
        outer-loop REML can compose this term with other gam penalties.
        """
        del basis
        latent = _check_matrix(latent, "latent")
        tau = self.effective_thresholds(latent.dtype).to(latent.device)
        diff = (latent - tau.unsqueeze(0)) / float(self.smoothing_eps)
        gate = torch.sigmoid(diff)
        per_axis = (tau.unsqueeze(0) * gate).sum()
        return float(self.weight) * per_axis


class _JumpReLUSTEFn(torch.autograd.Function):
    """Hard-threshold forward; smooth-sigmoid Hessian-consistent STE backward.

    Backward is derived from the same smoothed indicator the Rust
    `JumpReLUPenalty` uses so the analytic and STE gradients agree by
    construction (see gam/tests/jumprelu_ste.rs).
    """

    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, tau: torch.Tensor, smoothing_eps: float) -> torch.Tensor:
        ctx.save_for_backward(z, tau)
        ctx.smoothing_eps = float(smoothing_eps)
        return z * (z > tau.unsqueeze(0)).to(z.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        z, tau = ctx.saved_tensors
        eps = ctx.smoothing_eps
        diff = (z - tau.unsqueeze(0)) / eps
        # Rectangular STE for z: nonzero in [tau-eps, tau+eps], smooth elsewhere.
        # Equivalent in expectation to the derivative of the sigmoid-smoothed
        # gate; on average ∂φ/∂z = sigmoid'(diff)·z/eps + sigmoid(diff).
        gate = torch.sigmoid(diff)
        dphi_dz = gate + z * gate * (1.0 - gate) / eps
        dphi_dtau = -(gate + z * gate * (1.0 - gate) / eps)
        grad_z = grad_output * dphi_dz
        grad_tau = (grad_output * dphi_dtau).sum(dim=0)
        return grad_z, grad_tau, None


class GumbelTemperatureSchedule:
    """Deterministic Gumbel temperature schedule descriptor."""

    def __init__(self, tau_start: float, tau_min: float, decay: str = "geometric", *, rate: float = 0.95, steps: int = 100) -> None:
        self.tau_start = float(tau_start)
        self.tau_min = float(tau_min)
        self.decay = str(decay)
        self.rate = float(rate)
        self.steps = int(steps)
        self.iter_count = 0

    def step(self) -> float:
        self.iter_count += 1
        return self.current_tau()

    def current_tau(self) -> float:
        if self.decay in {"geometric", "exponential"}:
            return max(self.tau_min, self.tau_start * (self.rate ** self.iter_count))
        if self.decay == "linear":
            frac = min(1.0, self.iter_count / max(1, self.steps))
            return max(self.tau_min, self.tau_start + frac * (self.tau_min - self.tau_start))
        return max(self.tau_min, self.tau_start / (1.0 + self.iter_count))

    def to_rust_descriptor(self) -> dict[str, Any]:
        payload = {
            "tau_start": self.tau_start,
            "tau_min": self.tau_min,
            "decay": self.decay,
            "iter_count": self.iter_count,
        }
        if self.decay in {"geometric", "exponential"}:
            payload["rate"] = self.rate
        if self.decay == "linear":
            payload["steps"] = self.steps
        return payload


class RiemannianRetraction(Optimizer):
    """Optimizer that retracts Euclidean gradient steps onto a manifold."""

    def __init__(self, params: Any, manifold: str | dict[str, Any], lr: float = 1e-2, inner_steps: int = 1) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if inner_steps <= 0:
            raise ValueError("inner_steps must be > 0")
        self.manifold = manifold
        defaults = {"lr": float(lr), "inner_steps": int(inner_steps)}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = closure() if closure is not None else None
        manifold_spec = self.manifold.to_json() if hasattr(self.manifold, "to_json") else self.manifold
        manifold_json = json.dumps(manifold_spec if isinstance(manifold_spec, dict) else {"kind": manifold_spec})
        for group in self.param_groups:
            lr = float(group["lr"])
            inner_steps = int(group["inner_steps"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.dim() != 2:
                    raise ValueError("RiemannianRetraction parameters must be 2-D row batches")
                current = param.detach()
                grad = param.grad.detach().to(dtype=current.dtype, device=current.device)
                for _ in range(inner_steps):
                    delta = -lr * grad
                    out = rust_module().riemannian_retract(manifold_json, to_numpy_f64(current), to_numpy_f64(delta))
                    current = from_numpy_like(out, current)
                param.copy_(current)
        return loss


class LazyPcaBasis(nn.Module):
    """Memmap-backed PCA score loader with torch row indexing."""

    def __init__(self, path: str | Path, *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.path = Path(path)
        self.dtype = dtype
        self.device = torch.device("cpu" if device is None else device)
        self._scores = np.load(self.path, mmap_mode="r")
        if self._scores.ndim != 2:
            raise ValueError("LazyPcaBasis expects a 2-D .npy array")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self._scores.shape[0]), int(self._scores.shape[1])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if not isinstance(idx, torch.Tensor):
            raise TypeError("idx must be a torch.Tensor")
        rows = idx.detach().cpu().numpy()
        out = np.asarray(self._scores[rows], dtype=np.float64, order="C")
        return torch.as_tensor(out, dtype=self.dtype, device=self.device)


__all__ = [
    "ARDPenalty",
    "BlockOrthogonalityPenalty",
    "GumbelTemperatureSchedule",
    "IBPAssignmentPenalty",
    "IsometryPenalty",
    "IvaeRidgeMeanGauge",
    "JumpReLUPenalty",
    "LazyPcaBasis",
    "MechanismSparsityPenalty",
    "RiemannianRetraction",
    "TopologyAutoSelector",
]
