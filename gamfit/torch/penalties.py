"""Autograd modules for gamfit analytic composition penalties."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, cast, runtime_checkable

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from .._binding import rust_module
from .._penalty_bridge import (
    GumbelTemperatureSchedule,
    ard_descriptor,
    block_orthogonality_descriptor,
    call_rust_value_grad as _call_rust_value_grad,
    ibp_assignment_descriptor,
    latent_json as _latent_json,
    mechanism_sparsity_descriptor,
    penalty_json as _penalty_json,
)
from .._select_topology import TopologyAutoSelector
from ._coerce import from_numpy_like, to_numpy_f64


@runtime_checkable
class _ManifoldJson(Protocol):
    def to_json(self) -> dict[str, Any]: ...


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


def _rho_tensor(rho: torch.Tensor | None, ref: torch.Tensor, count: int) -> torch.Tensor:
    if rho is None:
        return torch.zeros(count, dtype=ref.dtype, device=ref.device)
    if rho.numel() != count:
        raise ValueError(f"rho length {rho.numel()} does not match expected {count}")
    return rho.reshape(-1).to(device=ref.device, dtype=ref.dtype)


class _RustPenaltyFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        target: torch.Tensor,
        rho: torch.Tensor,
        latents_json: str,
        penalties_json: str,
    ) -> torch.Tensor:
        value = _call_rust_value_grad(target, rho, latents_json, penalties_json)[0]
        ctx.save_for_backward(target, rho)
        ctx.latents_json = latents_json
        ctx.penalties_json = penalties_json
        return value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        target, rho = ctx.saved_tensors
        _, grad_target, grad_rho, _ = _call_rust_value_grad(
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
        value = _call_rust_value_grad(
            target,
            rho,
            latents_json,
            penalties_json,
            isometry_jacobian=jacobian.reshape(target.shape[0], -1),
            isometry_jacobian_second=jacobian_second,
        )[0]
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
        _, grad_target, grad_rho, grad_jac_flat = _call_rust_value_grad(
            target,
            rho,
            ctx.latents_json,
            ctx.penalties_json,
            isometry_jacobian=jacobian.reshape(target.shape[0], -1),
            isometry_jacobian_second=jacobian_second,
        )
        if grad_jac_flat is None:
            raise RuntimeError(
                "analytic_penalty_value_grad did not return an isometry Jacobian "
                "gradient even though an isometry Jacobian was supplied"
            )
        # Rust returns ∂P/∂J flattened as (N, p*d), W-/reference-/weight-aware
        # (src/terms/analytic_penalties.rs IsometryPenalty::grad_jacobian).
        d = target.shape[1]
        grad_j = grad_jac_flat.reshape(target.shape[0], -1, d)
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


@dataclass(slots=True)
class _PenaltyCall:
    """Materialized arguments for one Rust analytic-penalty evaluation.

    Subclasses of :class:`_RustPenaltyModule` build this from their parameters;
    the base class then routes it through either the autograd ``Function`` (for
    ``forward``) or the closed-form helper (for ``rust_value``).
    """

    target: torch.Tensor
    rho: torch.Tensor
    latents_json: str
    penalties_json: str
    isometry_basis: torch.Tensor | None = None
    isometry_jacobian_second: torch.Tensor | None = None


class _RustPenaltyModule(nn.Module):
    """Base class for penalty modules backed by Rust's analytic penalty registry.

    Subclasses implement :meth:`_prepare`. The base class provides:

    * ``forward(primary, basis=None)`` — routes through ``_RustPenaltyFn`` or,
      when the prep returns an isometry basis, through ``_IsometryPenaltyFn``.
    * ``rust_value(primary, **kwargs)`` — graph-free numpy entry point that
      calls the same ``_call_rust_value_grad`` helper ``forward`` uses, so the
      two paths cannot drift.
    """

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        raise NotImplementedError

    def forward(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> torch.Tensor:
        call = self._prepare(primary, basis=basis)
        if call.isometry_basis is not None:
            apply = cast(Callable[..., torch.Tensor], _IsometryPenaltyFn.apply)
            return apply(
                call.target,
                call.rho,
                call.isometry_basis,
                call.isometry_jacobian_second,
                call.latents_json,
                call.penalties_json,
            )
        apply = cast(Callable[..., torch.Tensor], _RustPenaltyFn.apply)
        return apply(call.target, call.rho, call.latents_json, call.penalties_json)

    def rust_value(
        self,
        primary: np.ndarray | torch.Tensor,
        basis: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        """Closed-form penalty value (no autograd graph)."""
        primary_t = torch.as_tensor(primary, dtype=torch.float64)
        basis_t = (
            None if basis is None else torch.as_tensor(basis, dtype=torch.float64)
        )
        call = self._prepare(primary_t, basis=basis_t)
        extras: dict[str, torch.Tensor] = {}
        if call.isometry_basis is not None:
            n_rows = int(call.target.shape[0])
            jacobian = (
                call.isometry_basis.unsqueeze(0).expand(n_rows, -1, -1)
                if call.isometry_basis.dim() == 2
                else call.isometry_basis
            )
            extras["isometry_jacobian"] = jacobian.reshape(n_rows, -1)
        if call.isometry_jacobian_second is not None:
            extras["isometry_jacobian_second"] = call.isometry_jacobian_second
        value_t = _call_rust_value_grad(
            call.target,
            call.rho,
            call.latents_json,
            call.penalties_json,
            **extras,
        )[0]
        return float(value_t.item())


class IsometryPenalty(_RustPenaltyModule):
    """Penalize the normalized decoder pullback metric.

    The Rust isometry penalty compares ``JᵀJ / gbar`` with the reference metric,
    where ``gbar`` is the mean pullback trace per latent dimension. This pins a
    unit-average-speed chart without coupling the penalty to decoder scale.
    """

    def __init__(self, weight: float = 1.0, *, target: str = "t") -> None:
        super().__init__()
        if weight <= 0.0:
            raise ValueError("IsometryPenalty.weight must be > 0")
        self.weight = float(weight)
        self.target = str(target)

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        if basis is None:
            raise ValueError("IsometryPenalty requires `basis`")
        latent = _check_matrix(primary, "latent")
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
        rho = torch.full(
            (1,), float(np.log(self.weight)), dtype=latent.dtype, device=latent.device
        )
        return _PenaltyCall(
            target=latent,
            rho=rho,
            latents_json=_latent_json(latent.shape[0], latent.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
            isometry_basis=basis_t,
        )


class ARDPenalty(_RustPenaltyModule):
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

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        del basis
        latent = _check_matrix(primary, "latent")
        descriptor = ard_descriptor(self.target, self.weight)
        return _PenaltyCall(
            target=latent,
            rho=self._rho(latent),
            latents_json=_latent_json(latent.shape[0], latent.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
        )


class BlockOrthogonalityPenalty(_RustPenaltyModule):
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

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        del basis
        latent = _check_matrix(primary, "latent")
        descriptor = block_orthogonality_descriptor(
            self.target,
            self.groups,
            self.weight,
            int(self.n_eff or latent.shape[0]),
            learnable=self.learnable,
        )
        rho = _rho_tensor(getattr(self, "log_weight", None), latent, 1 if self.learnable else 0)
        return _PenaltyCall(
            target=latent,
            rho=rho,
            latents_json=_latent_json(latent.shape[0], latent.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
        )


class MonotonicityPenalty(_RustPenaltyModule):
    """Soft monotonicity penalty along the leading axis of a latent block.

    Routes through the Rust ``monotonicity`` analytic-penalty descriptor.
    """

    def __init__(
        self,
        weight: float,
        n_eff: int,
        *,
        direction: float = 1.0,
        smoothing_eps: float = 1.0e-3,
        target: str = "t",
        learnable: bool = False,
    ) -> None:
        super().__init__()
        if not (np.isfinite(weight) and weight > 0.0):
            raise ValueError("MonotonicityPenalty.weight must be finite and > 0")
        if n_eff <= 0:
            raise ValueError("MonotonicityPenalty.n_eff must be > 0")
        if not (np.isfinite(direction) and direction != 0.0):
            raise ValueError("MonotonicityPenalty.direction must be finite and non-zero")
        if not (np.isfinite(smoothing_eps) and smoothing_eps > 0.0):
            raise ValueError("MonotonicityPenalty.smoothing_eps must be finite and > 0")
        self.weight = float(weight)
        self.n_eff = int(n_eff)
        self.direction = float(direction)
        self.smoothing_eps = float(smoothing_eps)
        self.target = str(target)
        self.learnable = bool(learnable)
        if self.learnable:
            self.log_weight = nn.Parameter(torch.zeros(1))

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        del basis
        latent = _check_matrix(primary, "latent")
        descriptor = {
            "kind": "monotonicity",
            "target": self.target,
            "weight": self.weight,
            "n_eff": self.n_eff,
            "direction": self.direction,
            "smoothing_eps": self.smoothing_eps,
            "learnable": self.learnable,
        }
        rho = _rho_tensor(getattr(self, "log_weight", None), latent, 1 if self.learnable else 0)
        return _PenaltyCall(
            target=latent,
            rho=rho,
            latents_json=_latent_json(latent.shape[0], latent.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
        )


class HarmonicRoughnessPenalty(_RustPenaltyModule):
    """Graduated periodic-basis roughness prior over a decoder block (#1282).

    Routes through the Rust ``harmonic_roughness`` analytic-penalty descriptor,
    which evaluates ``weight · Σ_r row_weights[r mod period] · Σ_j target[r,j]²``
    on a row-major ``(n_eff, d)`` block. ``row_weights`` is one atom's per-basis
    weight vector (``h⁴`` on harmonic rows ``h ≥ 2``, ``0`` on DC / fundamental)
    and is tiled across the ``F`` atoms of a stacked ``(F·K, D)`` decoder.

    ``weight`` is the evidence-optimal roughness precision refreshed from the
    Rust REML machinery during training (see
    :func:`ManifoldSAE.decoder_harmonic_penalty`); it is a plain float held off
    the autograd tape, so mutating it between forwards is safe.
    """

    def __init__(
        self,
        row_weights: Sequence[float],
        n_eff: int,
        weight: float = 1.0,
        *,
        target: str = "t",
        learnable: bool = False,
    ) -> None:
        super().__init__()
        rows = [float(w) for w in row_weights]
        if not rows:
            raise ValueError("HarmonicRoughnessPenalty.row_weights must be non-empty")
        if not all(np.isfinite(w) and w >= 0.0 for w in rows):
            raise ValueError(
                "HarmonicRoughnessPenalty.row_weights must be finite and non-negative"
            )
        if n_eff <= 0:
            raise ValueError("HarmonicRoughnessPenalty.n_eff must be > 0")
        if n_eff % len(rows) != 0:
            raise ValueError(
                "HarmonicRoughnessPenalty.n_eff must be a multiple of len(row_weights)"
            )
        self.row_weights = rows
        self.n_eff = int(n_eff)
        self.weight = float(weight)
        self.target = str(target)
        self.learnable = bool(learnable)
        if self.learnable:
            self.log_weight = nn.Parameter(torch.zeros(1))

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        del basis
        coeffs = _check_matrix(primary, "decoder_block")
        descriptor = {
            "kind": "harmonic_roughness",
            "target": self.target,
            "weight": self.weight,
            "n_eff": self.n_eff,
            "row_weights": list(self.row_weights),
            "learnable": self.learnable,
        }
        rho = _rho_tensor(
            getattr(self, "log_weight", None), coeffs, 1 if self.learnable else 0
        )
        return _PenaltyCall(
            target=coeffs,
            rho=rho,
            latents_json=_latent_json(coeffs.shape[0], coeffs.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
        )


class MechanismSparsityPenalty(_RustPenaltyModule):
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

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        del basis
        weights = _check_matrix(primary, "weights")
        descriptor = mechanism_sparsity_descriptor(
            self.target,
            self.feature_groups,
            self.weight,
            self.smoothing_eps,
            self.n_eff,
            learnable=self.learnable,
        )
        rho = _rho_tensor(getattr(self, "log_weight", None), weights, 1 if self.learnable else 0)
        return _PenaltyCall(
            target=weights,
            rho=rho,
            latents_json=_latent_json(
                max(1, int(round(self.n_eff))), weights.shape[0], name=self.target
            ),
            penalties_json=_penalty_json(descriptor),
        )


class IBPAssignmentPenalty(_RustPenaltyModule):
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

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        del basis
        logits = _check_matrix(primary, "logits")
        if logits.shape[1] != self.k_max:
            raise ValueError("logits width must equal k_max")
        descriptor = ibp_assignment_descriptor(
            self.target,
            self.k_max,
            self.alpha,
            self.tau,
            learnable=self.learnable,
        )
        rho = _rho_tensor(getattr(self, "log_alpha", None), logits, 1 if self.learnable else 0)
        return _PenaltyCall(
            target=logits,
            rho=rho,
            latents_json=_latent_json(logits.shape[0], logits.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
        )


class IvaeRidgeMeanGauge(_RustPenaltyModule):
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

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        del basis
        latent = _check_matrix(primary, "latent")
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
        return _PenaltyCall(
            target=latent,
            rho=rho,
            latents_json=_latent_json(latent.shape[0], latent.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
        )


class JumpReLUPenalty(_RustPenaltyModule):
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

    def _prepare(
        self, primary: torch.Tensor, basis: torch.Tensor | None = None
    ) -> _PenaltyCall:
        """Penalty value: ``weight · Σ τ · σ((z − τ)/ε)`` (smoothed L0).

        Matches the Rust ``JumpReLUPenalty::value`` analytic formulation so
        outer-loop REML can compose this term with other gam penalties.
        """
        del basis
        latent = _check_matrix(primary, "latent")
        descriptor = {
            "kind": "jumprelu",
            "target": self.target,
            "thresholds": to_numpy_f64(self.thresholds).reshape(-1).tolist(),
            "weight": self.weight,
            "smoothing_eps": self.smoothing_eps,
        }
        rho = self.log_threshold.to(device=latent.device, dtype=latent.dtype)
        return _PenaltyCall(
            target=latent,
            rho=rho,
            latents_json=_latent_json(latent.shape[0], latent.shape[1], name=self.target),
            penalties_json=_penalty_json(descriptor),
        )


class _JumpReLUSTEFn(torch.autograd.Function):
    """Hard-threshold JumpReLU gate — a pure-torch, on-device transcription of Rust.

    Forward returns the hard gate ``φ(z) = z · 1[z > τ]``; backward uses the
    smooth surrogate ``φ̃(z) = z · σ((z − τ)/ε)`` so the activation keeps a usable
    subgradient inside the smoothing band ``|z − τ| ≲ ε`` (a straight-through
    estimator). The math is transcribed element-for-element from the Rust source
    of truth ``gam::terms::analytic_penalties::jumprelu_gate_value_grad``
    (crates/gam-terms/src/analytic_penalties/sparsity.rs), computed directly in
    torch so no ``(N, F)`` matrix ever crosses the Python↔Rust boundary and
    dtype/device are preserved (works on GPU without a CPU/float64 round-trip).
    Parity is pinned by ``tests/torch/test_jumprelu_ste_parity.py``.
    """

    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, tau: torch.Tensor, smoothing_eps: float) -> torch.Tensor:
        # Pure-torch, on-device transcription of the Rust source of truth
        # `gam::terms::analytic_penalties::jumprelu_gate_value_grad`. Per element,
        # with per-column threshold ``τ`` broadcast over rows (matching the Rust
        # divide order for parity):
        #   ``g         = σ((z − τ)/ε)`` (stable_logistic ≡ torch.sigmoid)
        #   ``value     = z`` where ``z > τ`` else ``0`` (hard gate)
        #   ``slope     = z·g·(1 − g)/ε``
        #   ``dphi_dz   = g + slope``   ``dphi_dtau = −slope`` (smooth surrogate STE)
        rows = z.reshape(z.shape[0], -1)
        tau_row = tau.reshape(1, -1).to(device=rows.device, dtype=rows.dtype)
        g = torch.sigmoid((rows - tau_row) / smoothing_eps)
        value = torch.where(rows > tau_row, rows, torch.zeros_like(rows))
        slope = rows * g * (1.0 - g) / smoothing_eps
        dphi_dz = g + slope
        dphi_dtau = -slope
        ctx.save_for_backward(dphi_dz.reshape_as(z), dphi_dtau.reshape_as(z))
        return value.reshape_as(z)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        dphi_dz, dphi_dtau = ctx.saved_tensors
        grad_z = grad_output * dphi_dz
        grad_tau = (grad_output * dphi_dtau).sum(dim=0)
        return grad_z, grad_tau, None


class _IBPMapFn(torch.autograd.Function):
    """IBP-MAP concrete relaxation, value+grad from the Rust source of truth.

    Forward returns ``z_k = σ(l_k/τ) · π_k`` where ``π_k = (α/(α+1))^(k+1)`` is
    the consistent truncated stick-breaking prior mean (every atom shrunk by one
    Beta(α,1) stick, #614); backward multiplies the upstream gradient by
    the diagonal logit Jacobian ``∂z_k/∂l_k`` that Rust returns. Replacing the
    bare ``sigmoid(logits/τ)`` torch path makes torch IBP-Gumbel agree with the
    closed-form ``SaeAssignment`` IBP-MAP assignments (see
    ``src/terms/sae_manifold.rs`` ``ibp_map_row``).
    """

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, temperature: float, alpha: float) -> torch.Tensor:
        # Pure-torch, on-device transcription of the Rust source of truth
        # `gam::terms::sae::assignment::ibp_map_row_value_grad`
        # (crates/gam-sae/src/assignment.rs:1033). Per atom `k` (0-indexed),
        # with ``inv_tau = 1/τ`` (matching the Rust multiply order for parity):
        #   ``π_k = max((α/(α+1))^{k+1}, tiny)`` — the truncated stick-breaking
        #     prior mean, accumulated in LOG space then floored at the smallest
        #     positive normal so large-K atoms keep a live (never hard-masked)
        #     gradient path, exactly as `ordered_geometric_shrinkage_prior`
        #     (assignment.rs:966) does with `f64::MIN_POSITIVE`.
        #   ``sig   = σ(l_k·inv_tau)`` (stable_logistic ≡ torch.sigmoid)
        #   ``value = sig · π_k``
        #   ``grad  = sig·(1 − sig)·inv_tau·π_k`` (diagonal logit Jacobian)
        # No (N, K) matrix crosses the FFI boundary; dtype and device are
        # preserved (runs on GPU without a CPU/float64 round-trip).
        rows = logits.reshape(logits.shape[0], -1)
        k_atoms = rows.shape[1]
        inv_tau = 1.0 / temperature
        log_ratio = math.log(alpha / (alpha + 1.0))
        atom_index = torch.arange(1, k_atoms + 1, device=rows.device, dtype=rows.dtype)
        tiny = torch.finfo(rows.dtype).tiny
        pi = torch.exp(atom_index * log_ratio).clamp_min(tiny).reshape(1, -1)
        sig = torch.sigmoid(rows * inv_tau)
        value = sig * pi
        grad = sig * (1.0 - sig) * inv_tau * pi
        ctx.save_for_backward(grad.reshape_as(logits))
        return value.reshape_as(logits)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (jac_diag,) = ctx.saved_tensors
        return grad_output * jac_diag, None, None


def ibp_map(logits: torch.Tensor, temperature: float, alpha: float) -> torch.Tensor:
    """Differentiable IBP-MAP assignments via the Rust value+grad kernel."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("ibp_map logits must be a torch.Tensor")
    apply = cast(Callable[..., torch.Tensor], _IBPMapFn.apply)
    return apply(logits, float(temperature), float(alpha))


class _JumpReLUBoundedGateFn(torch.autograd.Function):
    """Bounded threshold gate — a pure-torch, on-device transcription of Rust.

    Forward returns ``a_k = σ((l_k − θ_k)/τ) · 1[l_k > θ_k]`` — the SAME bounded
    ``[0, 1)`` gate the closed-form ``SaeAssignment`` jumprelu / threshold_gate
    path evaluates (``jumprelu_row``; magnitude lives in the decoder). The math
    is transcribed from the Rust source of truth
    ``gam_sae::assignment::jumprelu_row_value_grad`` and computed directly in
    torch, so no ``(N, K)`` matrix ever crosses the Python↔Rust boundary and
    dtype/device are preserved (works on GPU without a CPU/float64 round-trip).
    Backward multiplies the upstream gradient by the smooth surrogate's diagonal
    derivative ``da/dl_k = σ'((l_k − θ_k)/τ)/τ`` (a straight-through estimator,
    alive on both sides of the jump so gated-off atoms keep a training signal);
    the threshold gradient is its negated row-sum
    (``∂a_k/∂θ_k = −da/dl_k``); callers negate and accumulate.
    """

    @staticmethod
    def forward(
        ctx: Any, logits: torch.Tensor, thresholds: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        # Pure-torch, on-device transcription of the Rust source of truth
        # `gam_sae::assignment::jumprelu_row_value_grad` (crates/gam-sae/src/
        # assignment.rs). Per atom, with ``inv_tau = 1/τ`` (matching the Rust
        # multiply order for bit-parity):
        #   ``sig = σ((l − θ)·inv_tau)`` (stable_logistic ≡ torch.sigmoid)
        #   ``value = sig`` where ``l > θ`` else ``0`` (hard jump)
        #   ``grad  = sig·(1 − sig)·inv_tau`` (straight-through, both sides)
        # No CPU/float64 round-trip: dtype and device are preserved.
        rows = logits.reshape(logits.shape[0], -1)
        thr = thresholds.reshape(1, -1)
        inv_tau = 1.0 / temperature
        sig = torch.sigmoid((rows - thr) * inv_tau)
        value = torch.where(rows > thr, sig, torch.zeros_like(sig))
        grad = sig * (1.0 - sig) * inv_tau
        ctx.save_for_backward(grad.reshape_as(logits))
        return value.reshape_as(logits)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        (jac_diag,) = ctx.saved_tensors
        upstream = grad_output * jac_diag
        return upstream, -upstream.sum(dim=0), None


def jumprelu_bounded_gate(
    logits: torch.Tensor, thresholds: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Differentiable bounded threshold gate via the Rust value+grad kernel."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("jumprelu_bounded_gate logits must be a torch.Tensor")
    if not isinstance(thresholds, torch.Tensor):
        raise TypeError("jumprelu_bounded_gate thresholds must be a torch.Tensor")
    apply = cast(Callable[..., torch.Tensor], _JumpReLUBoundedGateFn.apply)
    return apply(logits, thresholds, float(temperature))


class _TopKActivationFn(torch.autograd.Function):
    """Top-k SAE activation — a pure-torch, on-device transcription of Rust.

    Forward returns the per-atom **independent**, strictly non-negative
    activation ``a_k = τ·softplus(l_k/τ)`` the ``softmax_topk`` gate scores atoms
    with. The math is transcribed from the Rust source of truth
    ``gam_sae::assignment::topk_activation_row_value_grad`` and computed directly
    in torch, so no ``(N, K)`` matrix ever crosses the Python↔Rust boundary and
    dtype/device are preserved (works on GPU without a CPU/float64 round-trip).
    Backward multiplies the upstream gradient by the diagonal derivative
    ``da/dl_k = σ(l_k/τ)`` (the temperature cancels in the chain rule since
    ``a = τ·softplus(l/τ)``). The hard top-k *selection* and its masked gradient
    are applied by the caller on the tape — this Function owns only the smooth
    activation and its exact derivative, the single-source-of-truth counterpart
    to the closed-form family's ``ibp_map`` / ``jumprelu`` activations.
    """

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        # Pure-torch, on-device transcription of the Rust source of truth
        # `gam_sae::assignment::topk_activation_row_value_grad`
        # (crates/gam-sae/src/assignment.rs). With ``inv_tau = 1/τ`` (matching the
        # Rust multiply order for bit-parity):
        #   ``scaled = l · inv_tau``
        #   ``value  = τ · softplus(scaled)`` (stable_softplus ≡ softplus)
        #   ``grad   = σ(scaled)``          (stable_logistic ≡ torch.sigmoid)
        # No CPU/float64 round-trip: dtype and device are preserved.
        inv_tau = 1.0 / temperature
        scaled = logits * inv_tau
        value = temperature * torch.nn.functional.softplus(scaled)
        grad = torch.sigmoid(scaled)
        ctx.save_for_backward(grad)
        return value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (jac_diag,) = ctx.saved_tensors
        return grad_output * jac_diag, None


def topk_activation(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Differentiable top-k SAE activation via the Rust value+grad kernel.

    Returns ``τ·softplus(logits/τ)`` with the Rust-defined diagonal derivative
    ``σ(logits/τ)`` on the backward. The hard top-k mask/STE stays on the caller's
    tape; this owns only the smooth non-negative activation.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("topk_activation logits must be a torch.Tensor")
    apply = cast(Callable[..., torch.Tensor], _TopKActivationFn.apply)
    return apply(logits, float(temperature))


class RiemannianRetraction(Optimizer):
    """Optimizer that retracts Euclidean gradient steps onto a manifold."""

    def __init__(self, params: Any, manifold: str | dict[str, Any] | _ManifoldJson, lr: float = 1e-2, inner_steps: int = 1) -> None:
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
        manifold_spec = self.manifold.to_json() if isinstance(self.manifold, _ManifoldJson) else self.manifold
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
    "HarmonicRoughnessPenalty",
    "IBPAssignmentPenalty",
    "IsometryPenalty",
    "IvaeRidgeMeanGauge",
    "JumpReLUPenalty",
    "LazyPcaBasis",
    "MechanismSparsityPenalty",
    "RiemannianRetraction",
    "TopologyAutoSelector",
    "ibp_map",
]
