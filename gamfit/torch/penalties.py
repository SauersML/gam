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
    latent_json as _latent_json,
    mechanism_sparsity_descriptor,
    ordered_beta_bernoulli_descriptor,
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


class OrderedBetaBernoulliPenalty(_RustPenaltyModule):
    """Ordered independent Beta--Bernoulli prior over assignment logits."""

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
        descriptor = ordered_beta_bernoulli_descriptor(
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
    """Hard-threshold JumpReLU gate backed by the Rust value/gradient kernel.

    Forward returns the hard gate ``φ(z) = z · 1[z > τ]``; backward uses the
    smooth surrogate ``φ̃(z) = z · σ((z − τ)/ε)`` so the activation keeps a usable
    subgradient inside the smoothing band ``|z − τ| ≲ ε`` (a straight-through
    estimator). Rust computes the value and both per-element derivatives through
    ``jumprelu_gate_value_grad``; Python only caches those derivatives and applies
    the upstream gradient on Torch's tape.
    """

    @staticmethod
    def forward(ctx: Any, z: torch.Tensor, tau: torch.Tensor, smoothing_eps: float) -> torch.Tensor:
        value_np, dphi_dz_np, dphi_dtau_np = rust_module().jumprelu_gate_value_grad(
            to_numpy_f64(z.reshape(z.shape[0], -1)),
            to_numpy_f64(tau.reshape(-1)),
            float(smoothing_eps),
        )
        value = from_numpy_like(value_np, z).reshape_as(z)
        dphi_dz = from_numpy_like(dphi_dz_np, z).reshape_as(z)
        dphi_dtau = from_numpy_like(dphi_dtau_np, z).reshape_as(z)
        ctx.save_for_backward(dphi_dz, dphi_dtau)
        return value

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        dphi_dz, dphi_dtau = ctx.saved_tensors
        grad_z = grad_output * dphi_dz
        grad_tau = (grad_output * dphi_dtau).sum(dim=0)
        return grad_z, grad_tau, None


class RiemannianRetraction(Optimizer):
    """One-step Riemannian gradient optimizer using the manifold metric."""

    def __init__(
        self,
        params: Any,
        manifold: str | dict[str, Any] | _ManifoldJson,
        lr: float = 1e-2,
    ) -> None:
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and > 0")
        self.manifold = manifold
        defaults = {"lr": float(lr)}
        super().__init__(params, defaults)

    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        manifold_spec = (
            self.manifold.to_json()
            if isinstance(self.manifold, _ManifoldJson)
            else self.manifold
        )
        manifold_json = json.dumps(
            manifold_spec
            if isinstance(manifold_spec, dict)
            else {"kind": manifold_spec}
        )
        with torch.no_grad():
            for group in self.param_groups:
                lr = float(group["lr"])
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    if param.dim() != 2:
                        raise ValueError(
                            "RiemannianRetraction parameters must be 2-D row batches"
                        )
                    out = rust_module().riemannian_gradient_step(
                        manifold_json,
                        to_numpy_f64(param),
                        to_numpy_f64(param.grad),
                        lr,
                    )
                    param.copy_(from_numpy_like(out, param))
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
    "OrderedBetaBernoulliPenalty",
    "IsometryPenalty",
    "IvaeRidgeMeanGauge",
    "JumpReLUPenalty",
    "LazyPcaBasis",
    "MechanismSparsityPenalty",
    "RiemannianRetraction",
    "TopologyAutoSelector",
]
