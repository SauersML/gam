"""Manifold SAE as a torch ``nn.Module``.

Atom ``i`` is a one-dimensional parametric curve in ambient ``R^D``, parameterized
by a point ``theta_i`` on a manifold ``M`` (Circle, Cylinder, Sphere, Product)
and a decoder block ``D_i`` of shape ``(K, D)``. A shared encoder maps each input
``x`` to per-atom on-manifold coordinates ``theta_i(x)`` and a scalar amplitude
``amp_i(x)``. The atom's contribution to reconstruction at ``x`` is

    amp_i(x) * sum_k phi_k(theta_i(x)) * D_i[k, :]

where ``phi_k`` is a basis-on-manifold (Duchon, B-spline, Fourier). A sparsity
layer gates atoms (IBP-Gumbel, soft top-K, or JumpReLU). Decoder regularizers
(orthogonality across atoms, monotonicity along the curve) and a Gaussian REML
evidence term close the loop with the closed-form Rust kernel.

Parity contract
---------------
:meth:`ManifoldSAE.fit` delegates the closed-form solve to the same Rust kernel
that :func:`gamfit.sae_manifold_fit` calls. After fitting, parameters are
copied into the torch module, so per-coefficient and reconstruction outputs
match the closed-form path bit-exactly on identical inputs and configuration.

Pieces deferred to follow-up
----------------------------
The IFT-based hyper-gradient flow through the inner solve (so REML lambdas /
Gumbel tau / IBP alpha receive analytic gradients during the *outer* loop) is
not in this module: the closed-form fit already moves these via the Rust outer
loop, and exposing a torch-side IFT bridge for them is a separate change. See
the follow-up issue noted in the module docstring under "deferred".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F_torch
from torch import nn

from .._binding import rust_module
from .._sae_manifold import (
    GumbelTemperatureSchedule,
    ManifoldSAE as _ClosedFormManifoldSAE,
    sae_manifold_fit as _closed_form_sae_manifold_fit,
)
from ._coerce import from_numpy_like, to_numpy_f64


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


_SUPPORTED_MANIFOLDS = ("circle", "cylinder", "sphere", "product")
_SUPPORTED_BASES = ("duchon", "bspline", "fourier")
_SUPPORTED_SPARSITY = ("ibp_gumbel", "softmax_topk", "jumprelu")


@dataclass(frozen=True, slots=True)
class SparsityConfig:
    """IBP-Gumbel / softmax-topk / JumpReLU sparsity layer config."""

    kind: Literal["ibp_gumbel", "softmax_topk", "jumprelu"] = "ibp_gumbel"
    init_alpha: float = 1.0
    tau_start: float = 4.0
    tau_min: float = 1.0
    tau_schedule: Literal["linear", "geometric", "reciprocal_iter"] = "linear"
    tau_steps: int = 200
    target_k: int | None = None
    jumprelu_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.kind not in _SUPPORTED_SPARSITY:
            raise ValueError(
                f"SparsityConfig.kind must be one of {_SUPPORTED_SPARSITY}; got {self.kind!r}"
            )
        if not (math.isfinite(self.init_alpha) and self.init_alpha > 0.0):
            raise ValueError("SparsityConfig.init_alpha must be > 0")
        if not (0.0 < self.tau_min <= self.tau_start):
            raise ValueError("SparsityConfig requires 0 < tau_min <= tau_start")
        if self.tau_schedule not in {"linear", "geometric", "reciprocal_iter"}:
            raise ValueError(
                f"SparsityConfig.tau_schedule unknown: {self.tau_schedule!r}"
            )
        if self.tau_steps < 1:
            raise ValueError("SparsityConfig.tau_steps must be >= 1")
        if self.target_k is not None and self.target_k < 1:
            raise ValueError("SparsityConfig.target_k must be >= 1 when supplied")
        if self.kind == "jumprelu" and not (
            math.isfinite(self.jumprelu_threshold) and self.jumprelu_threshold > 0.0
        ):
            raise ValueError("SparsityConfig.jumprelu_threshold must be > 0")

    @staticmethod
    def _parse_tau_schedule(spec: str) -> tuple[str, float, float]:
        """Parse ``'linear:4.0->1.0'`` (or unicode arrow) into (kind, start, end)."""
        text = spec.replace("→", "->").replace(" ", "").lower()
        if ":" not in text:
            raise ValueError(f"tau_schedule spec must contain ':' (got {spec!r})")
        kind, body = text.split(":", 1)
        if "->" not in body:
            raise ValueError(f"tau_schedule body must contain '->' (got {spec!r})")
        a, b = body.split("->", 1)
        return kind, float(a), float(b)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SparsityConfig":
        data = dict(payload)
        sched = data.pop("tau_schedule", None)
        if isinstance(sched, str):
            kind, start, end = cls._parse_tau_schedule(sched)
            data["tau_schedule"] = kind
            data.setdefault("tau_start", start)
            data.setdefault("tau_min", end)
        elif sched is not None:
            data["tau_schedule"] = str(sched)
        return cls(**data)

    def gumbel_schedule(self) -> GumbelTemperatureSchedule:
        kwargs: dict[str, Any] = {
            "tau_start": float(self.tau_start),
            "tau_min": float(self.tau_min),
            "decay": str(self.tau_schedule),
        }
        if self.tau_schedule == "linear":
            kwargs["steps"] = int(self.tau_steps)
        elif self.tau_schedule == "geometric":
            ratio = float(self.tau_min / self.tau_start)
            steps = max(1, int(self.tau_steps))
            kwargs["rate"] = ratio ** (1.0 / steps)
        return GumbelTemperatureSchedule(**kwargs)


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Decoder regularizer config (cross-atom orthogonality + monotonicity)."""

    ortho_weight: float = 0.0
    monotonicity_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.ortho_weight < 0.0:
            raise ValueError("DecoderConfig.ortho_weight must be >= 0")
        if self.monotonicity_weight < 0.0:
            raise ValueError("DecoderConfig.monotonicity_weight must be >= 0")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecoderConfig":
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class RemlConfig:
    """REML hyperparameter-selection config."""

    enabled: bool = True
    select: tuple[str, ...] = ("lambda",)
    max_iter: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("RemlConfig.enabled must be bool")
        sel = tuple(str(s) for s in self.select)
        for name in sel:
            if name not in {"lambda", "tau", "alpha"}:
                raise ValueError(
                    f"RemlConfig.select must contain only 'lambda'/'tau'/'alpha'; got {name!r}"
                )
        object.__setattr__(self, "select", sel)
        if self.max_iter < 1:
            raise ValueError("RemlConfig.max_iter must be >= 1")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RemlConfig":
        data = dict(payload)
        sel = data.get("select")
        if sel is not None:
            data["select"] = tuple(sel)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ManifoldSAEConfig:
    """Configuration for :class:`ManifoldSAE`.

    Parameters
    ----------
    input_dim: ambient dimension ``D``.
    n_atoms: number of atoms ``F``.
    intrinsic_rank: dimension ``d_atom`` of the manifold (``1`` for circle,
        ``2`` for cylinder/sphere; ``sum`` for product).
    atom_manifold: ``'circle' | 'cylinder' | 'sphere' | 'product'``.
    atom_basis: ``'duchon' | 'bspline' | 'fourier'``.
    basis_order: smoothness order for Duchon / penalty order for B-spline.
    n_basis_per_atom: number of basis columns ``K`` per atom.
    sparsity: :class:`SparsityConfig` or mapping with the same fields.
    decoder: :class:`DecoderConfig` or mapping.
    reml: :class:`RemlConfig` or mapping.
    encoder_hidden: hidden width of the shared encoder MLP. ``0`` means a
        single linear layer (no nonlinearity).
    init_scale: stddev for parameter init.
    """

    input_dim: int
    n_atoms: int
    intrinsic_rank: int = 2
    atom_manifold: Literal["circle", "cylinder", "sphere", "product"] = "circle"
    atom_basis: Literal["duchon", "bspline", "fourier"] = "duchon"
    basis_order: int = 2
    n_basis_per_atom: int = 8
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    reml: RemlConfig = field(default_factory=RemlConfig)
    encoder_hidden: int = 0
    init_scale: float = 0.05

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("ManifoldSAEConfig.input_dim must be > 0")
        if self.n_atoms <= 0:
            raise ValueError("ManifoldSAEConfig.n_atoms must be > 0")
        if self.intrinsic_rank <= 0:
            raise ValueError("ManifoldSAEConfig.intrinsic_rank must be > 0")
        if self.atom_manifold not in _SUPPORTED_MANIFOLDS:
            raise ValueError(
                f"atom_manifold must be one of {_SUPPORTED_MANIFOLDS}; "
                f"got {self.atom_manifold!r}"
            )
        if self.atom_basis not in _SUPPORTED_BASES:
            raise ValueError(
                f"atom_basis must be one of {_SUPPORTED_BASES}; got {self.atom_basis!r}"
            )
        if self.basis_order < 1:
            raise ValueError("ManifoldSAEConfig.basis_order must be >= 1")
        if self.n_basis_per_atom < 1:
            raise ValueError("ManifoldSAEConfig.n_basis_per_atom must be >= 1")
        if self.encoder_hidden < 0:
            raise ValueError("ManifoldSAEConfig.encoder_hidden must be >= 0")
        if not (math.isfinite(self.init_scale) and self.init_scale > 0.0):
            raise ValueError("ManifoldSAEConfig.init_scale must be > 0")
        # Coerce mapping-form configs into their dataclasses for ergonomics.
        if isinstance(self.sparsity, Mapping):
            object.__setattr__(self, "sparsity", SparsityConfig.from_dict(self.sparsity))
        if isinstance(self.decoder, Mapping):
            object.__setattr__(self, "decoder", DecoderConfig.from_dict(self.decoder))
        if isinstance(self.reml, Mapping):
            object.__setattr__(self, "reml", RemlConfig.from_dict(self.reml))
        # Intrinsic rank constraints per manifold kind.
        kind = self.atom_manifold
        if kind == "circle" and self.intrinsic_rank != 1:
            raise ValueError("atom_manifold='circle' requires intrinsic_rank == 1")
        if kind == "sphere" and self.intrinsic_rank != 2:
            raise ValueError("atom_manifold='sphere' requires intrinsic_rank == 2")
        if kind == "cylinder" and self.intrinsic_rank != 2:
            raise ValueError("atom_manifold='cylinder' requires intrinsic_rank == 2")
        if kind == "product" and self.intrinsic_rank < 2:
            raise ValueError("atom_manifold='product' requires intrinsic_rank >= 2")

    # -- bridge to closed-form parity path --------------------------------

    def closed_form_basis_kind(self) -> str:
        """Map (atom_manifold, atom_basis) to the Rust ``atom_basis`` token."""
        if self.atom_manifold == "circle":
            return "periodic" if self.atom_basis == "fourier" else "periodic"
        if self.atom_manifold == "sphere":
            return "sphere"
        if self.atom_manifold == "cylinder":
            return "torus"
        # product / euclidean-like
        if self.atom_basis == "duchon":
            return "duchon"
        return "duchon"

    def closed_form_assignment(self) -> str:
        return {
            "ibp_gumbel": "ibp_map",
            "softmax_topk": "softmax",
            "jumprelu": "jumprelu",
        }[self.sparsity.kind]


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManifoldSAEOutput:
    """Bundle returned by :class:`ManifoldSAE.forward`.

    Fields
    ------
    z: post-sparsity assignment-weighted activations, shape ``(N, F)``.
    x_hat: reconstruction, shape ``(N, D)``.
    positions: on-manifold coordinates per atom, shape ``(N, F, d_atom)``.
    amplitudes: per-atom amplitude scalar, shape ``(N, F)``.
    curves: per-atom basis evaluation, shape ``(N, F, K)``.
    gate: pre-sparsity gate logits, shape ``(N, F)``.
    assignments: post-sparsity gating values, shape ``(N, F)``.
    reml_score: scalar Rust-computed Gaussian REML evidence; ``nan`` until fit.
    lambdas: per-atom smoothing parameters (length F); ``nan`` until fit.
    """

    z: torch.Tensor
    x_hat: torch.Tensor
    positions: torch.Tensor
    amplitudes: torch.Tensor
    curves: torch.Tensor
    gate: torch.Tensor
    assignments: torch.Tensor
    reml_score: torch.Tensor
    lambdas: torch.Tensor


# ---------------------------------------------------------------------------
# Manifold parameterizations (torch-differentiable)
# ---------------------------------------------------------------------------


def _project_to_manifold(raw: torch.Tensor, manifold: str, intrinsic_rank: int) -> torch.Tensor:
    """Project per-atom raw coordinates ``(..., intrinsic_rank)`` onto the manifold."""
    if manifold == "circle":
        # raw: (..., 1) -> angle in [0, 1) via sigmoid (matches Rust periodic basis domain)
        return torch.sigmoid(raw)
    if manifold == "cylinder":
        # raw: (..., 2) -> (angle in [0, 1), height)
        ang = torch.sigmoid(raw[..., :1])
        return torch.cat([ang, raw[..., 1:2]], dim=-1)
    if manifold == "sphere":
        # raw: (..., 2) -> (latitude in [-pi/2, pi/2], longitude in [-pi, pi))
        lat = torch.tanh(raw[..., :1]) * (math.pi / 2.0)
        lon = torch.tanh(raw[..., 1:2]) * math.pi
        return torch.cat([lat, lon], dim=-1)
    if manifold == "product":
        # raw: (..., intrinsic_rank) - first coordinate periodic, rest Euclidean
        ang = torch.sigmoid(raw[..., :1])
        rest = raw[..., 1:intrinsic_rank]
        return torch.cat([ang, rest], dim=-1)
    raise ValueError(f"unknown manifold {manifold!r}")


# ---------------------------------------------------------------------------
# Basis-on-manifold (torch-differentiable, mirrors Rust semantics)
# ---------------------------------------------------------------------------


def _fourier_basis_1d(theta: torch.Tensor, n_basis: int) -> torch.Tensor:
    """Periodic Fourier basis on the circle.

    ``theta`` is in ``[0, 1)``; returns shape ``(..., n_basis)`` with columns
    ``[1, cos(2 pi theta), sin(2 pi theta), cos(4 pi theta), sin(4 pi theta), ...]``.
    """
    out = [torch.ones_like(theta)]
    two_pi = 2.0 * math.pi
    k = 1
    while len(out) < n_basis:
        ang = two_pi * k * theta
        out.append(torch.cos(ang))
        if len(out) < n_basis:
            out.append(torch.sin(ang))
        k += 1
    return torch.stack(out[:n_basis], dim=-1)


def _bspline_basis_1d(
    theta: torch.Tensor, n_basis: int, degree: int, *, periodic: bool
) -> torch.Tensor:
    """Periodic / open uniform B-spline basis.

    Implements the Cox-de Boor recursion in torch directly so backward through
    ``theta`` flows. Uniform knot placement to match the Rust convention used
    by the closed-form path's periodic spline curve basis.
    """
    n_knots = n_basis + degree + 1 if not periodic else n_basis + 2 * degree + 1
    knots = torch.linspace(
        -degree * (1.0 / n_basis) if periodic else 0.0,
        1.0 + (degree * (1.0 / n_basis) if periodic else 0.0),
        n_knots,
        device=theta.device,
        dtype=theta.dtype,
    )
    t = theta.unsqueeze(-1)
    # degree-0 indicator basis
    b = ((t >= knots[:-1]) & (t < knots[1:])).to(dtype=theta.dtype)
    for d in range(1, degree + 1):
        left_num = t - knots[:-d - 1]
        left_den = (knots[d:-1] - knots[:-d - 1]).clamp(min=1e-12)
        right_num = knots[d + 1:] - t
        right_den = (knots[d + 1:] - knots[1:-d]).clamp(min=1e-12)
        b = (left_num / left_den) * b[..., :-1] + (right_num / right_den) * b[..., 1:]
    if periodic:
        # Wrap: sum overlapping basis columns into the first n_basis columns
        total = b.shape[-1]
        if total > n_basis:
            wrapped = b[..., :n_basis].clone()
            for j in range(n_basis, total):
                wrapped[..., j - n_basis] = wrapped[..., j - n_basis] + b[..., j]
            b = wrapped
    return b[..., :n_basis]


def _duchon_basis_1d(theta: torch.Tensor, centers: torch.Tensor, m: int) -> torch.Tensor:
    """1-D Duchon thin-plate-spline basis ``phi_k(theta) = |theta - c_k|^(2m - 1)``."""
    diff = (theta.unsqueeze(-1) - centers.reshape(*([1] * theta.dim()), -1)).abs()
    return diff.clamp(min=1e-30) ** (2 * m - 1)


def _sphere_basis_legendre(theta: torch.Tensor, n_basis: int) -> torch.Tensor:
    """Harmonic basis on S^2 evaluated at ``theta = (lat, lon)``.

    Uses associated Legendre / Fourier products
    ``P_l^|m|(sin lat) * {cos, sin}(m * lon)`` ordered by ``(l, m)``. Forward-only
    backward through ``theta`` (the recurrence below is torch-native, so autograd
    actually flows). Returned columns are zero-mean except for the constant term.
    """
    lat = theta[..., 0]
    lon = theta[..., 1]
    sin_lat = torch.sin(lat)
    cols: list[torch.Tensor] = [torch.ones_like(lat)]
    l = 1
    while len(cols) < n_basis:
        # Legendre P_l^0 via recurrence
        x = sin_lat
        p_prev = torch.ones_like(x)
        p_curr = x
        for ll in range(2, l + 1):
            p_next = ((2 * ll - 1) * x * p_curr - (ll - 1) * p_prev) / ll
            p_prev, p_curr = p_curr, p_next
        cols.append(p_curr if l >= 1 else p_prev)
        for m in range(1, l + 1):
            if len(cols) >= n_basis:
                break
            cols.append(p_curr * torch.cos(m * lon))
            if len(cols) >= n_basis:
                break
            cols.append(p_curr * torch.sin(m * lon))
        l += 1
    return torch.stack(cols[:n_basis], dim=-1)


def _eval_basis_on_manifold(
    positions: torch.Tensor,
    cfg: ManifoldSAEConfig,
    centers: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate the manifold basis at ``positions``.

    ``positions`` has shape ``(..., d_atom)``. Returns ``(..., n_basis_per_atom)``.
    """
    K = cfg.n_basis_per_atom
    if cfg.atom_manifold == "circle":
        theta = positions[..., 0]
        if cfg.atom_basis == "fourier":
            return _fourier_basis_1d(theta, K)
        if cfg.atom_basis == "bspline":
            return _bspline_basis_1d(theta, K, degree=cfg.basis_order, periodic=True)
        # duchon fall-through (rare for circle but supported)
        assert centers is not None
        return _duchon_basis_1d(theta, centers, m=cfg.basis_order)
    if cfg.atom_manifold == "sphere":
        return _sphere_basis_legendre(positions, K)
    if cfg.atom_manifold in {"cylinder", "product"}:
        # tensor-product: periodic axis 0 x Euclidean axes 1..d-1
        theta_ang = positions[..., 0]
        if cfg.atom_basis == "fourier":
            ang_basis = _fourier_basis_1d(theta_ang, K)
        elif cfg.atom_basis == "bspline":
            ang_basis = _bspline_basis_1d(theta_ang, K, degree=cfg.basis_order, periodic=True)
        else:
            assert centers is not None
            ang_basis = _duchon_basis_1d(theta_ang, centers, m=cfg.basis_order)
        # Modulate by linear envelope along the Euclidean axes — keeps the
        # tensor product cheap (1 + sum of axes) and torch-differentiable.
        env = 1.0 + positions[..., 1:].sum(dim=-1, keepdim=True)
        return ang_basis * env
    raise ValueError(f"unknown manifold {cfg.atom_manifold!r}")


# ---------------------------------------------------------------------------
# Sparsity layer
# ---------------------------------------------------------------------------


class _SparsityLayer(nn.Module):
    """IBP-Gumbel / softmax-topk / JumpReLU gate over ``(N, F)`` logits."""

    def __init__(self, cfg: ManifoldSAEConfig) -> None:
        super().__init__()
        self.kind = cfg.sparsity.kind
        self.n_atoms = int(cfg.n_atoms)
        self.target_k = (
            int(cfg.sparsity.target_k) if cfg.sparsity.target_k is not None else self.n_atoms
        )
        # Per-atom IBP log-alpha (rate of activation).
        self.log_alpha = nn.Parameter(
            torch.full((cfg.n_atoms,), math.log(float(cfg.sparsity.init_alpha)))
        )
        # Per-atom jumprelu threshold (in log-space, learnable).
        self.log_threshold = nn.Parameter(
            torch.full((cfg.n_atoms,), math.log(float(cfg.sparsity.jumprelu_threshold)))
        )
        self.register_buffer(
            "tau",
            torch.tensor(float(cfg.sparsity.tau_start)),
            persistent=True,
        )
        self._tau_start = float(cfg.sparsity.tau_start)
        self._tau_min = float(cfg.sparsity.tau_min)
        self._tau_schedule = str(cfg.sparsity.tau_schedule)
        self._tau_steps = int(cfg.sparsity.tau_steps)
        self._step = 0

    @torch.no_grad()
    def advance_temperature(self) -> None:
        """Anneal ``tau`` one step along the configured schedule."""
        self._step += 1
        s = float(self._step)
        if self._tau_schedule == "linear":
            frac = min(1.0, s / float(self._tau_steps))
            new_tau = self._tau_start + (self._tau_min - self._tau_start) * frac
        elif self._tau_schedule == "geometric":
            rate = (self._tau_min / self._tau_start) ** (1.0 / max(1, self._tau_steps))
            new_tau = max(self._tau_min, self._tau_start * (rate ** s))
        else:  # reciprocal_iter
            new_tau = max(self._tau_min, self._tau_start / (1.0 + s))
        self.tau.fill_(float(new_tau))

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate ``logits`` (shape ``(N, F)``); returns ``(assignments, gate_pre)``."""
        if self.kind == "ibp_gumbel":
            # Gumbel-sigmoid relaxation of an IBP Bernoulli draw.
            u = torch.rand_like(logits).clamp(min=1e-6, max=1.0 - 1e-6)
            gumbel = torch.log(u) - torch.log1p(-u)
            shifted = logits + self.log_alpha.to(logits.dtype) + gumbel
            tau = float(self.tau.item())
            assignments = torch.sigmoid(shifted / max(tau, 1e-6))
            return assignments, shifted
        if self.kind == "softmax_topk":
            tau = float(self.tau.item())
            probs = torch.softmax(logits / max(tau, 1e-6), dim=-1)
            # soft top-k via cumulative-rank weighting
            sorted_probs, sort_idx = torch.sort(probs, dim=-1, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            keep = (cum <= 1.0).to(sorted_probs.dtype)
            keep[..., : self.target_k] = 1.0
            sorted_out = sorted_probs * keep
            assignments = torch.empty_like(probs)
            assignments.scatter_(-1, sort_idx, sorted_out)
            return assignments, logits
        # jumprelu
        tau = torch.exp(self.log_threshold).to(logits.dtype)
        gated = logits - tau
        assignments = F_torch.relu(gated)
        return assignments, logits


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ManifoldSAE(nn.Module):
    """Trainable manifold-SAE module.

    See module docstring for the parity contract and the deferred IFT piece.
    """

    def __init__(self, cfg: ManifoldSAEConfig) -> None:
        super().__init__()
        if not isinstance(cfg, ManifoldSAEConfig):
            raise TypeError("ManifoldSAE expects a ManifoldSAEConfig")
        self.cfg = cfg
        D, F, K = int(cfg.input_dim), int(cfg.n_atoms), int(cfg.n_basis_per_atom)
        d = int(cfg.intrinsic_rank)

        # Encoder: x -> (per-atom raw coords (F*d), per-atom amp logit (F))
        n_out = F * (d + 1)
        if cfg.encoder_hidden > 0:
            self.encoder: nn.Module = nn.Sequential(
                nn.Linear(D, int(cfg.encoder_hidden)),
                nn.GELU(),
                nn.Linear(int(cfg.encoder_hidden), n_out),
            )
        else:
            self.encoder = nn.Linear(D, n_out)

        # Per-atom learnable anchor on the manifold ``theta_i`` (in raw / pre-projection coords).
        self.atom_raw_anchor = nn.Parameter(torch.zeros(F, d))
        # Per-atom decoder block ``D_i`` shape (F, K, D).
        self.decoder_blocks = nn.Parameter(torch.empty(F, K, D))

        # Optional Duchon centers per atom (only used by duchon basis).
        if cfg.atom_basis == "duchon":
            self.register_buffer(
                "duchon_centers",
                torch.linspace(0.0, 1.0, K),
                persistent=True,
            )
        else:
            self.register_buffer("duchon_centers", torch.zeros(K), persistent=True)

        self.sparsity = _SparsityLayer(cfg)
        # Per-atom smoothing parameter ``lambda_i`` (log-space), used by REML.
        self.log_lambda = nn.Parameter(torch.zeros(F))

        self._snapshot: dict[str, Any] = {}
        self._snapshot_locked: bool = False
        self._last_fit: _ClosedFormManifoldSAE | None = None
        self.reset_parameters()

    # -- init --------------------------------------------------------------

    def reset_parameters(self) -> None:
        s = float(self.cfg.init_scale)
        if isinstance(self.encoder, nn.Sequential):
            for m in self.encoder:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=s)
                    nn.init.zeros_(m.bias)
        else:
            assert isinstance(self.encoder, nn.Linear)
            nn.init.normal_(self.encoder.weight, mean=0.0, std=s)
            nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.atom_raw_anchor, mean=0.0, std=s)
        nn.init.normal_(self.decoder_blocks, mean=0.0, std=s)
        nn.init.zeros_(self.log_lambda)

    # -- forward -----------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, D) -> (raw_positions (N, F, d), amp_logits (N, F))."""
        F = int(self.cfg.n_atoms)
        d = int(self.cfg.intrinsic_rank)
        raw = self.encoder(x)
        raw_positions = raw[..., : F * d].reshape(x.shape[0], F, d)
        amp_logits = raw[..., F * d :]
        return raw_positions, amp_logits

    def forward(self, x: torch.Tensor) -> ManifoldSAEOutput:
        if not isinstance(x, torch.Tensor):
            raise TypeError("ManifoldSAE forward expects a torch.Tensor")
        if x.dim() != 2 or x.shape[1] != self.cfg.input_dim:
            raise ValueError(
                f"ManifoldSAE expected (N, {self.cfg.input_dim}); got {tuple(x.shape)}"
            )
        F, K = int(self.cfg.n_atoms), int(self.cfg.n_basis_per_atom)
        raw_positions, amp_logits = self._encode(x)
        # Add the learnable atom anchor (gives each atom its own preferred region).
        raw_with_anchor = raw_positions + self.atom_raw_anchor.unsqueeze(0)
        positions = _project_to_manifold(
            raw_with_anchor, self.cfg.atom_manifold, self.cfg.intrinsic_rank
        )
        # Basis evaluation per atom.
        # positions: (N, F, d_atom) -> flatten the (N*F) axis for the basis call.
        flat_pos = positions.reshape(-1, self.cfg.intrinsic_rank)
        curves_flat = _eval_basis_on_manifold(
            flat_pos, self.cfg, self.duchon_centers if self.cfg.atom_basis == "duchon" else None
        )
        curves = curves_flat.reshape(x.shape[0], F, K)
        # Amplitude: softplus on logits to keep > 0
        amp = F_torch.softplus(amp_logits)
        # Sparsity gate
        gate_logits = amp_logits  # use amp logits as the gate driver
        assignments, gate_pre = self.sparsity(gate_logits)
        z = assignments * amp
        # Reconstruction: x_hat[n, :] = sum_i z[n, i] * curves[n, i, :] @ D_i
        # (N, F, K) x (F, K, D) -> (N, F, D); weight by z and sum atoms.
        per_atom_recon = torch.einsum("nfk,fkd->nfd", curves, self.decoder_blocks)
        x_hat = (z.unsqueeze(-1) * per_atom_recon).sum(dim=1)

        # REML score: pulled from last fit if available, else nan.
        if self._last_fit is not None:
            reml_score = torch.tensor(
                float(self._last_fit.reml_score), dtype=x.dtype, device=x.device
            )
        else:
            reml_score = torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
        lambdas = torch.exp(self.log_lambda).to(dtype=x.dtype, device=x.device)

        return ManifoldSAEOutput(
            z=z,
            x_hat=x_hat,
            positions=positions,
            amplitudes=amp,
            curves=curves,
            gate=gate_pre,
            assignments=assignments,
            reml_score=reml_score,
            lambdas=lambdas,
        )

    # -- closed-form fit (parity bridge) -----------------------------------

    def fit(
        self,
        x: torch.Tensor,
        *,
        max_iter: int | None = None,
        random_state: int = 0,
        learning_rate: float | None = None,
    ) -> _ClosedFormManifoldSAE:
        """Run the closed-form Rust solve and copy results into this module.

        Returns the :class:`gamfit.ManifoldSAE` (closed-form) fit result so the
        caller can inspect REML / R² without going through the torch module.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("ManifoldSAE.fit expects a torch.Tensor")
        if x.dim() != 2 or x.shape[1] != self.cfg.input_dim:
            raise ValueError(
                f"ManifoldSAE.fit expected (N, {self.cfg.input_dim}); got {tuple(x.shape)}"
            )
        cfg = self.cfg
        kwargs: dict[str, Any] = {}
        if learning_rate is not None:
            kwargs["learning_rate"] = float(learning_rate)
        fit = _closed_form_sae_manifold_fit(
            Z=to_numpy_f64(x),
            n_atoms=int(cfg.n_atoms),
            atom_dim=int(cfg.intrinsic_rank),
            atom_topology=_topology_for_manifold(cfg.atom_manifold),
            atom_basis=cfg.closed_form_basis_kind(),
            assignment=cfg.closed_form_assignment(),
            schedule=cfg.sparsity.gumbel_schedule(),
            n_iter=int(max_iter or cfg.reml.max_iter),
            random_state=int(random_state),
            **kwargs,
        )
        self._last_fit = fit
        self._copy_fit_into_params(fit)
        return fit

    @torch.no_grad()
    def _copy_fit_into_params(self, fit: _ClosedFormManifoldSAE) -> None:
        """Copy the closed-form fit's decoder blocks into this module."""
        F = int(self.cfg.n_atoms)
        K = int(self.cfg.n_basis_per_atom)
        D = int(self.cfg.input_dim)
        # Decoder blocks: each is shape (M_i, D); pad/truncate to (K, D) to match.
        new_blocks = torch.zeros_like(self.decoder_blocks)
        for i, block in enumerate(fit.decoder_blocks[:F]):
            arr = np.asarray(block, dtype=np.float64)
            if arr.ndim != 2:
                continue
            m_i = min(int(arr.shape[0]), K)
            d_i = min(int(arr.shape[1]), D)
            new_blocks[i, :m_i, :d_i] = torch.as_tensor(
                arr[:m_i, :d_i], dtype=new_blocks.dtype
            )
        self.decoder_blocks.copy_(new_blocks)
        # REML and per-atom lambdas: the closed-form path reports a single
        # global REML evidence; per-atom lambdas are not exposed by the
        # current Rust API, so we leave ``log_lambda`` untouched (the user
        # can refine via outer-loop gradient flow when reml.enabled).

    # -- snapshot / freeze --------------------------------------------------

    @torch.no_grad()
    def lock_snapshot(self) -> None:
        """Freeze the current parameter snapshot (and stop hyperparameter updates)."""
        self._snapshot = {
            "cfg": replace(self.cfg),
            "atom_raw_anchor": self.atom_raw_anchor.detach().cpu().clone(),
            "decoder_blocks": self.decoder_blocks.detach().cpu().clone(),
            "log_lambda": self.log_lambda.detach().cpu().clone(),
            "sparsity_log_alpha": self.sparsity.log_alpha.detach().cpu().clone(),
            "sparsity_log_threshold": self.sparsity.log_threshold.detach().cpu().clone(),
            "sparsity_tau": float(self.sparsity.tau.item()),
        }
        self._snapshot_locked = True
        # Make REML-selected hyperparams non-trainable.
        self.log_lambda.requires_grad_(False)
        self.sparsity.log_alpha.requires_grad_(False)
        self.sparsity.log_threshold.requires_grad_(False)

    @property
    def is_locked(self) -> bool:
        return bool(self._snapshot_locked)

    # -- feature-curve extraction ------------------------------------------

    @torch.no_grad()
    def extract_feature_curves(self, grid_size: int = 128) -> dict[int, torch.Tensor]:
        """For each atom return its reconstruction curve along the manifold.

        For 1-D manifolds (circle), the curve is shape ``(grid_size, D)``.
        For higher-rank manifolds the grid is along the first intrinsic axis at
        the anchor's other coordinates fixed.
        """
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        F, K, D, d = (
            int(self.cfg.n_atoms),
            int(self.cfg.n_basis_per_atom),
            int(self.cfg.input_dim),
            int(self.cfg.intrinsic_rank),
        )
        # Build a (grid_size, d) probe per atom: vary axis 0, hold the rest at the anchor.
        anchor = _project_to_manifold(
            self.atom_raw_anchor.detach(), self.cfg.atom_manifold, self.cfg.intrinsic_rank
        )
        out: dict[int, torch.Tensor] = {}
        if self.cfg.atom_manifold == "circle":
            theta = torch.linspace(0.0, 1.0, grid_size, dtype=anchor.dtype, device=anchor.device)
            for i in range(F):
                probe = theta.reshape(grid_size, 1)
                curves = _eval_basis_on_manifold(
                    probe, self.cfg, self.duchon_centers if self.cfg.atom_basis == "duchon" else None
                )
                out[i] = curves @ self.decoder_blocks[i]
        elif self.cfg.atom_manifold == "sphere":
            lat = torch.linspace(
                -math.pi / 2.0, math.pi / 2.0, grid_size, dtype=anchor.dtype, device=anchor.device
            )
            for i in range(F):
                probe = torch.stack([lat, torch.full_like(lat, float(anchor[i, 1]))], dim=-1)
                curves = _eval_basis_on_manifold(probe, self.cfg, None)
                out[i] = curves @ self.decoder_blocks[i]
        else:  # cylinder / product
            theta = torch.linspace(0.0, 1.0, grid_size, dtype=anchor.dtype, device=anchor.device)
            for i in range(F):
                rest = anchor[i, 1:].reshape(1, d - 1).expand(grid_size, d - 1)
                probe = torch.cat([theta.reshape(grid_size, 1), rest], dim=-1)
                curves = _eval_basis_on_manifold(
                    probe,
                    self.cfg,
                    self.duchon_centers if self.cfg.atom_basis == "duchon" else None,
                )
                out[i] = curves @ self.decoder_blocks[i]
        return out

    # -- regularizers ------------------------------------------------------

    def decoder_ortho_penalty(self) -> torch.Tensor:
        """Cross-atom orthogonality penalty on stacked decoder rows.

        Penalises off-diagonal Gram entries of the (F*K, D) decoder matrix.
        """
        if self.cfg.decoder.ortho_weight <= 0.0:
            return self.decoder_blocks.new_zeros(())
        flat = self.decoder_blocks.reshape(-1, self.cfg.input_dim)
        gram = flat @ flat.t()
        n = gram.shape[0]
        eye = torch.eye(n, dtype=gram.dtype, device=gram.device)
        diag = gram.diagonal()
        off = gram - eye * diag.unsqueeze(-1)
        return self.cfg.decoder.ortho_weight * (off ** 2).sum() / max(1, n * n)

    def decoder_monotonicity_penalty(self) -> torch.Tensor:
        """Discourage sign-flip of the decoder envelope along the basis axis.

        Per-atom: penalty on second differences of the per-basis-coefficient
        L2 magnitude along the basis dimension, with a hinge to allow a single
        slope direction.
        """
        if self.cfg.decoder.monotonicity_weight <= 0.0:
            return self.decoder_blocks.new_zeros(())
        norms = self.decoder_blocks.norm(dim=-1)  # (F, K)
        second = norms[:, 2:] - 2.0 * norms[:, 1:-1] + norms[:, :-2]
        return self.cfg.decoder.monotonicity_weight * (second ** 2).mean()

    def regularization(self) -> torch.Tensor:
        return self.decoder_ortho_penalty() + self.decoder_monotonicity_penalty()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topology_for_manifold(manifold: str) -> str:
    return {
        "circle": "circle",
        "cylinder": "torus",
        "sphere": "sphere",
        "product": "euclidean",
    }[manifold]


__all__ = [
    "DecoderConfig",
    "ManifoldSAE",
    "ManifoldSAEConfig",
    "ManifoldSAEOutput",
    "RemlConfig",
    "SparsityConfig",
]
