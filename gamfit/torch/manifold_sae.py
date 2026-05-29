"""Manifold SAE as a torch ``nn.Module``.

Atom ``i`` is a 1-D parametric curve in ambient ``R^D`` parameterized by a
point ``theta_i`` on a manifold ``M`` (Circle, Cylinder, Sphere, Product) and
a decoder block ``D_i`` of shape ``(K, D)``. A shared encoder maps each input
``x`` to per-atom on-manifold coordinates ``theta_i(x)`` and a scalar
amplitude ``amp_i(x)``. The atom's contribution to reconstruction is

    amp_i(x) * sum_k phi_k(theta_i(x)) * D_i[k, :]

Architectural rule
------------------
**Python is a thin wrapper over Rust.** Every numerical primitive in this
module — basis evaluation with Jacobian, IBP-Gumbel and JumpReLU sparsity
value/gradient, decoder orthogonality penalty, REML evidence, and the
closed-form SAE fit — calls a PyO3 binding implemented in
``crates/gam-pyffi/src/lib.rs``. Torch enters the picture only because
``loss.backward()`` needs autograd tape continuity: each Rust call is wrapped
in a :class:`torch.autograd.Function` whose backward routes back to the
Rust-computed VJP. The only logic that stays in Python is parameter
registration, shape plumbing, and torch-side glue.

Parity contract
---------------
:meth:`ManifoldSAE.fit` delegates to :func:`gamfit.sae_manifold_fit`, which
shells out to ``sae_manifold_fit_minimal`` in Rust. Identical numerics to the
closed-form path are structural.

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Mapping, cast

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
from .penalties import (
    BlockOrthogonalityPenalty,
    IBPAssignmentPenalty,
    JumpReLUPenalty,
    MonotonicityPenalty,
    ibp_map,
)


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
        """Build the Rust-side :class:`GumbelTemperatureSchedule`."""
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
    """Decoder regularizer config.

    Both ``ortho_weight`` (Rust ``block_orthogonality``) and
    ``monotonicity_weight`` (Rust ``monotonicity``) route through the
    analytic-penalty descriptor surface.
    """

    ortho_weight: float = 0.0
    monotonicity_weight: float = 0.0
    monotonicity_direction: float = 1.0
    monotonicity_smoothing_eps: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.ortho_weight < 0.0:
            raise ValueError("DecoderConfig.ortho_weight must be >= 0")
        if self.monotonicity_weight < 0.0:
            raise ValueError("DecoderConfig.monotonicity_weight must be >= 0")
        if not (math.isfinite(self.monotonicity_direction) and self.monotonicity_direction != 0.0):
            raise ValueError(
                "DecoderConfig.monotonicity_direction must be finite and non-zero"
            )
        if not (
            math.isfinite(self.monotonicity_smoothing_eps)
            and self.monotonicity_smoothing_eps > 0.0
        ):
            raise ValueError("DecoderConfig.monotonicity_smoothing_eps must be > 0")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecoderConfig":
        data = dict(payload)
        # Legacy field kept for backward compatibility with earlier dict payloads.
        data.pop("monotonicity_supported", None)
        return cls(**data)


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
    """Configuration for :class:`ManifoldSAE`."""

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
    dtype: Any = field(default=None)

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
        if self.dtype is None:
            object.__setattr__(self, "dtype", torch.float64)
        if isinstance(self.sparsity, Mapping):
            object.__setattr__(self, "sparsity", SparsityConfig.from_dict(self.sparsity))
        if isinstance(self.decoder, Mapping):
            object.__setattr__(self, "decoder", DecoderConfig.from_dict(self.decoder))
        if isinstance(self.reml, Mapping):
            object.__setattr__(self, "reml", RemlConfig.from_dict(self.reml))
        kind = self.atom_manifold
        if kind == "circle" and self.intrinsic_rank != 1:
            raise ValueError("atom_manifold='circle' requires intrinsic_rank == 1")
        if kind == "sphere" and self.intrinsic_rank != 2:
            raise ValueError("atom_manifold='sphere' requires intrinsic_rank == 2")
        if kind == "cylinder" and self.intrinsic_rank != 2:
            raise ValueError("atom_manifold='cylinder' requires intrinsic_rank == 2")
        if kind == "product" and self.intrinsic_rank < 2:
            raise ValueError("atom_manifold='product' requires intrinsic_rank >= 2")

    def closed_form_basis_kind(self) -> str:
        """Map (atom_manifold, atom_basis) to the Rust ``atom_basis`` token."""
        if self.atom_manifold == "circle":
            return "periodic"
        if self.atom_manifold == "sphere":
            return "sphere"
        if self.atom_manifold == "cylinder":
            return "torus"
        return "duchon"

    def closed_form_assignment(self) -> str:
        return {
            "ibp_gumbel": "ibp_map",
            "softmax_topk": "softmax",
            "jumprelu": "jumprelu",
        }[self.sparsity.kind]


@dataclass(frozen=True, slots=True)
class ManifoldSAEOutput:
    """Bundle returned by :class:`ManifoldSAE.forward`."""

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
# Basis-with-jet autograd bridge to Rust
# ---------------------------------------------------------------------------


class _BasisWithJetFn(torch.autograd.Function):
    """Evaluate a manifold basis through Rust ``basis_with_jet``.

    ``t`` has shape ``(N, d_atom)``; output ``phi`` has shape ``(N, K)`` and
    backward uses the analytic Jacobian ``J`` of shape ``(N, K, d_atom)`` that
    the Rust call returns alongside ``phi``. No basis math is computed in
    Python — this autograd.Function exists purely to keep the torch tape
    continuous around a Rust call.
    """

    @staticmethod
    def forward(
        ctx: Any,
        t: torch.Tensor,
        kind: str,
        params_json: str,
    ) -> torch.Tensor:
        params = _decode_params_json(params_json)
        phi_np, jet_np, _penalty_np = rust_module().basis_with_jet(
            kind, to_numpy_f64(t), params
        )
        ctx.save_for_backward(from_numpy_like(jet_np, t))
        return from_numpy_like(phi_np, t)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Any, ...]:
        (jet,) = ctx.saved_tensors
        (grad_phi,) = grad_outputs
        if grad_phi.shape != jet.shape[:-1]:
            grad_phi = grad_phi.reshape(jet.shape[:-1])
        grad_t = torch.einsum("nk,nkd->nd", grad_phi.to(dtype=jet.dtype), jet)
        return grad_t, None, None


def _decode_params_json(params_json: str) -> dict[str, Any]:
    raw = json.loads(params_json)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "centers":
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            out[key] = arr
        else:
            out[key] = value
    return out


def _basis_rust(
    t: torch.Tensor, cfg: ManifoldSAEConfig, centers: torch.Tensor | None
) -> torch.Tensor:
    """Dispatch ``basis_with_jet`` based on the config's manifold/basis."""
    if t.dim() != 2:
        raise ValueError(f"_basis_rust expects (N, d), got shape {tuple(t.shape)}")
    apply = cast(Callable[..., torch.Tensor], _BasisWithJetFn.apply)
    # Circle / sphere: the manifold dictates the basis; user choice is ignored.
    if cfg.atom_manifold == "circle":
        n_harm = max(1, (cfg.n_basis_per_atom - 1) // 2)
        return apply(t, "periodic", json.dumps({"n_harmonics": int(n_harm)}))
    if cfg.atom_manifold == "sphere":
        return apply(t, "sphere", json.dumps({}))
    # Cylinder / product: the angular column drives the basis; remaining
    # intrinsic coordinates enter through the amplitude path.
    if cfg.atom_basis == "bspline":
        params = {
            "n_basis": int(cfg.n_basis_per_atom),
            "degree": 3,
            "order": int(cfg.basis_order),
            "periodic": False,
        }
        return apply(t[:, :1], "bspline", json.dumps(params))
    if centers is None:
        raise ValueError("Duchon-style manifold requires centers")
    centers_list = to_numpy_f64(centers.reshape(-1, 1)).tolist()
    params = {"centers": centers_list, "m": int(cfg.basis_order)}
    return apply(t[:, :1], "duchon", json.dumps(params))


def _eval_basis_on_manifold(
    positions: torch.Tensor,
    cfg: ManifoldSAEConfig,
    centers: torch.Tensor | None,
) -> torch.Tensor:
    """``positions: (*, d)`` → ``(*, K)`` via Rust ``basis_with_jet``.

    The Rust kernel returns an actual width that may differ from
    ``cfg.n_basis_per_atom``; trim or zero-pad here so callers see a fixed K.
    """
    flat = positions.reshape(-1, positions.shape[-1])
    out = _basis_rust(flat, cfg, centers)
    K = int(cfg.n_basis_per_atom)
    if out.shape[-1] > K:
        out = out[..., :K]
    elif out.shape[-1] < K:
        pad = torch.zeros(
            *out.shape[:-1], K - out.shape[-1], dtype=out.dtype, device=out.device
        )
        out = torch.cat([out, pad], dim=-1)
    return out.reshape(*positions.shape[:-1], K)


# ---------------------------------------------------------------------------
# Manifold parameterization (pointwise autograd, no math)
# ---------------------------------------------------------------------------


def _project_to_manifold(raw: torch.Tensor, manifold: str, intrinsic_rank: int) -> torch.Tensor:
    """Project per-atom raw coordinates onto the manifold domain.

    Pure pointwise sigmoid/tanh — no domain math; the autograd-tape projection
    that maps unconstrained encoder outputs to the basis-with-jet input domain.
    """
    if manifold == "circle":
        return torch.sigmoid(raw)
    if manifold == "cylinder":
        ang = torch.sigmoid(raw[..., :1])
        return torch.cat([ang, raw[..., 1:2]], dim=-1)
    if manifold == "sphere":
        lat = torch.tanh(raw[..., :1]) * (math.pi / 2.0)
        lon = torch.tanh(raw[..., 1:2]) * math.pi
        return torch.cat([lat, lon], dim=-1)
    if manifold == "product":
        ang = torch.sigmoid(raw[..., :1])
        rest = raw[..., 1:intrinsic_rank]
        return torch.cat([ang, rest], dim=-1)
    raise ValueError(f"unknown manifold {manifold!r}")


# ---------------------------------------------------------------------------
# Sparsity layer — composes Rust-backed penalty modules
# ---------------------------------------------------------------------------


class _SparsityLayer(nn.Module):
    """Activation + Rust-backed penalty composer.

    For IBP-Gumbel and JumpReLU the penalty term used in the loss routes
    through :mod:`gamfit.torch.penalties`, which themselves call
    ``analytic_penalty_value_grad`` in Rust. For softmax-topk the activation
    is a pointwise autograd primitive and the closed-form ``.fit()`` path
    drives the Rust selector; no separate Rust penalty descriptor exists.
    """

    def __init__(self, cfg: ManifoldSAEConfig) -> None:
        super().__init__()
        self.kind = cfg.sparsity.kind
        self.n_atoms = int(cfg.n_atoms)
        self.target_k = (
            int(cfg.sparsity.target_k) if cfg.sparsity.target_k is not None else self.n_atoms
        )
        if self.kind == "ibp_gumbel":
            self._ibp = IBPAssignmentPenalty(
                k_max=cfg.n_atoms,
                alpha=float(cfg.sparsity.init_alpha),
                tau=float(cfg.sparsity.tau_start),
                learnable=True,
            )
        elif self.kind == "jumprelu":
            thresholds = torch.full(
                (cfg.n_atoms,), float(cfg.sparsity.jumprelu_threshold), dtype=torch.float64
            )
            self._jumprelu = JumpReLUPenalty(
                thresholds=thresholds,
                weight=1.0,
                smoothing_eps=1e-3,
                learnable_threshold=True,
            )
        self.register_buffer(
            "tau", torch.tensor(float(cfg.sparsity.tau_start)), persistent=True
        )
        # Single source of truth for annealing: the Rust
        # GumbelTemperatureSchedule. We hold the descriptor and query τ through
        # the FFI accessor rather than re-deriving the decay in Python.
        self._schedule = cfg.sparsity.gumbel_schedule()
        self._init_alpha = float(cfg.sparsity.init_alpha)

    @torch.no_grad()
    def advance_temperature(self) -> None:
        """Anneal ``tau`` by advancing the Rust ``GumbelTemperatureSchedule``.

        The schedule owns the iteration counter and evaluates the decay through
        the Rust ``gumbel_schedule_tau`` FFI, so the annealing arithmetic lives
        in exactly one place (no Python-side step bookkeeping).
        """
        self.tau.fill_(float(self._schedule.step()))

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the sparsity gate. Returns ``(assignments, gate_pre)``."""
        if self.kind == "ibp_gumbel":
            tau = max(float(self.tau.item()), 1e-6)
            # Route through the Rust IBP-MAP value+grad kernel so the torch
            # forward applies the stick-breaking prior π_k and temperature
            # scaling that the closed-form fit uses (single source of truth).
            assignments = ibp_map(logits, tau, self._init_alpha)
            return assignments, logits
        if self.kind == "softmax_topk":
            tau = float(self.tau.item())
            probs = torch.softmax(logits / max(tau, 1e-6), dim=-1)
            _, top_idx = torch.topk(probs, k=self.target_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, top_idx, 1.0)
            hard = probs * mask
            assignments = probs + (hard - probs).detach()
            return assignments, logits
        # JumpReLU activation: hard threshold forward, Rust-STE backward.
        assignments = self._jumprelu.gate(logits)
        return assignments, logits

    def penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """Rust-backed scalar penalty value at ``logits``."""
        if self.kind == "ibp_gumbel":
            return self._ibp(logits)
        if self.kind == "jumprelu":
            return self._jumprelu(logits)
        return logits.new_zeros(())


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ManifoldSAE(nn.Module):
    """Trainable manifold-SAE module — see module docstring."""

    def __init__(self, cfg: ManifoldSAEConfig) -> None:
        super().__init__()
        if not isinstance(cfg, ManifoldSAEConfig):
            raise TypeError("ManifoldSAE expects a ManifoldSAEConfig")
        self.cfg = cfg
        dt: torch.dtype = cfg.dtype
        D, F, K = int(cfg.input_dim), int(cfg.n_atoms), int(cfg.n_basis_per_atom)
        d = int(cfg.intrinsic_rank)

        n_out = F * (d + 1)
        if cfg.encoder_hidden > 0:
            self.encoder: nn.Module = nn.Sequential(
                nn.Linear(D, int(cfg.encoder_hidden), dtype=dt),
                nn.GELU(),
                nn.Linear(int(cfg.encoder_hidden), n_out, dtype=dt),
            )
        else:
            self.encoder = nn.Linear(D, n_out, dtype=dt)

        self.atom_raw_anchor = nn.Parameter(torch.zeros(F, d, dtype=dt))
        self.decoder_blocks = nn.Parameter(torch.empty(F, K, D, dtype=dt))

        if cfg.atom_basis == "duchon":
            self.register_buffer(
                "duchon_centers", torch.linspace(0.0, 1.0, K, dtype=dt), persistent=True
            )
        else:
            self.register_buffer("duchon_centers", torch.zeros(K, dtype=dt), persistent=True)

        self.sparsity = _SparsityLayer(cfg)
        self.log_lambda = nn.Parameter(torch.zeros(F, dtype=dt))

        # Decoder orthogonality penalty: Rust ``block_orthogonality`` descriptor.
        # The penalty operates on a flat target of shape ``(n_eff, latent_dim)``
        # and treats ``groups`` as a partition of the latent (column) axis. We
        # want to penalize cross-atom correlations between the per-atom
        # decoder weight matrices ``R_g = decoder_blocks[g] ∈ R^{K×D}``, i.e.
        # ``½·w·Σ_{g≠h} ‖R_g · R_hᵀ‖²_F``. We obtain that by presenting the
        # decoder transposed: rows index the contracted ``input_dim`` axis and
        # columns index the ``F·K`` atom-basis pairs. Each atom owns a
        # contiguous block of ``K`` columns and the per-atom partition is the
        # latent-axis partition Rust expects.
        if cfg.decoder.ortho_weight > 0.0:
            groups = [list(range(i * K, (i + 1) * K)) for i in range(F)]
            self._ortho_penalty: BlockOrthogonalityPenalty | None = (
                BlockOrthogonalityPenalty(
                    groups=groups,
                    weight=float(cfg.decoder.ortho_weight),
                    n_eff=cfg.input_dim,
                )
            )
        else:
            self._ortho_penalty = None

        # Decoder monotonicity penalty: Rust ``monotonicity`` descriptor over
        # the per-atom basis-coefficient axis (K rows of a (K, D) decoder block).
        if cfg.decoder.monotonicity_weight > 0.0:
            self._monotonicity_penalty: MonotonicityPenalty | None = MonotonicityPenalty(
                weight=float(cfg.decoder.monotonicity_weight),
                n_eff=K,
                direction=float(cfg.decoder.monotonicity_direction),
                smoothing_eps=float(cfg.decoder.monotonicity_smoothing_eps),
            )
        else:
            self._monotonicity_penalty = None

        self._snapshot: dict[str, Any] = {}
        self._snapshot_locked: bool = False
        self._last_fit: _ClosedFormManifoldSAE | None = None
        self.reset_parameters()
        self.to(dtype=cfg.dtype)

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

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        if x.dtype != self.cfg.dtype:
            raise TypeError(
                f"ManifoldSAE configured for dtype {self.cfg.dtype}; "
                f"got input dtype {x.dtype}. Cast the input or build the config "
                f"with the matching `dtype=` to avoid silent autograd promotion."
            )
        if self._last_fit is not None:
            return self._forward_from_closed_form(x)
        F = int(self.cfg.n_atoms)
        raw_positions, amp_logits = self._encode(x)
        raw_with_anchor = raw_positions + self.atom_raw_anchor.unsqueeze(0)
        positions = _project_to_manifold(
            raw_with_anchor, self.cfg.atom_manifold, self.cfg.intrinsic_rank
        )
        curves = _eval_basis_on_manifold(
            positions,
            self.cfg,
            self.duchon_centers if self.cfg.atom_basis == "duchon" else None,
        )
        amp = F_torch.softplus(amp_logits)
        assignments, gate_pre = self.sparsity(amp_logits)
        z = assignments * amp
        per_atom_recon = torch.einsum("nfk,fkd->nfd", curves, self.decoder_blocks)
        x_hat = (z.unsqueeze(-1) * per_atom_recon).sum(dim=1)

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

    def _forward_from_closed_form(self, x: torch.Tensor) -> ManifoldSAEOutput:
        # When .fit() has been called, the closed-form Rust solve is the source
        # of truth: forward must reproduce fit.fitted exactly for in-sample x.
        # The random encoder/anchors play no role because the closed-form solver
        # does not produce an encoder to copy back into module parameters.
        fit = self._last_fit
        assert fit is not None
        F = int(self.cfg.n_atoms)
        d = int(self.cfg.intrinsic_rank)
        x_np = x.detach().cpu().numpy()
        is_in_sample = (
            fit.training_data.shape == x_np.shape
            and np.allclose(x_np, fit.training_data)
        )
        if not is_in_sample:
            raise NotImplementedError(
                "ManifoldSAE.forward(x) after .fit() requires x to be the same "
                "data .fit() was called on; for out-of-sample reconstruction "
                "call fit.reconstruct(x) / fit.predict(x) on the returned fit."
            )
        fitted_np = np.asarray(fit.fitted, dtype=np.float64)
        assignments_np = np.asarray(fit.assignments, dtype=np.float64)
        coords_per_atom = [np.asarray(c, dtype=np.float64) for c in fit.coords[:F]]
        positions_np = np.stack(
            [c.reshape(c.shape[0], d) for c in coords_per_atom], axis=1
        )
        x_hat = torch.as_tensor(fitted_np, dtype=x.dtype, device=x.device)
        assignments = torch.as_tensor(
            assignments_np, dtype=x.dtype, device=x.device
        )
        positions = torch.as_tensor(positions_np, dtype=x.dtype, device=x.device)
        curves = _eval_basis_on_manifold(
            positions,
            self.cfg,
            self.duchon_centers if self.cfg.atom_basis == "duchon" else None,
        )
        amp = torch.ones_like(assignments)
        z = assignments
        gate = assignments
        reml_score = torch.tensor(
            float(fit.reml_score), dtype=x.dtype, device=x.device
        )
        lambdas = torch.exp(self.log_lambda).to(dtype=x.dtype, device=x.device)
        return ManifoldSAEOutput(
            z=z,
            x_hat=x_hat,
            positions=positions,
            amplitudes=amp,
            curves=curves,
            gate=gate,
            assignments=assignments,
            reml_score=reml_score,
            lambdas=lambdas,
        )

    def fit(
        self,
        x: torch.Tensor,
        *,
        max_iter: int | None = None,
        random_state: int = 0,
        learning_rate: float | None = None,
    ) -> _ClosedFormManifoldSAE:
        """Run the closed-form Rust solve and copy results into this module."""
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
        if cfg.sparsity.target_k is not None:
            kwargs["top_k"] = int(cfg.sparsity.target_k)
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
        F = int(self.cfg.n_atoms)
        K = int(self.cfg.n_basis_per_atom)
        D = int(self.cfg.input_dim)
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

    @torch.no_grad()
    def lock_snapshot(self) -> None:
        """Freeze the current parameter snapshot and stop hyperparameter updates."""
        self._snapshot = {
            "cfg": replace(self.cfg),
            "atom_raw_anchor": self.atom_raw_anchor.detach().cpu().clone(),
            "decoder_blocks": self.decoder_blocks.detach().cpu().clone(),
            "log_lambda": self.log_lambda.detach().cpu().clone(),
            "sparsity_tau": float(self.sparsity.tau.item()),
        }
        self._snapshot_locked = True
        self.log_lambda.requires_grad_(False)
        for child in self.sparsity.children():
            for p in child.parameters(recurse=True):
                p.requires_grad_(False)

    @property
    def is_locked(self) -> bool:
        return bool(self._snapshot_locked)

    @torch.no_grad()
    def extract_feature_curves(self, grid_size: int = 128) -> dict[int, torch.Tensor]:
        """Per-atom reconstruction curve over a manifold grid.

        Basis evaluation routes through the same Rust ``basis_with_jet`` kernel
        as :meth:`forward`.
        """
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        F = int(self.cfg.n_atoms)
        d = int(self.cfg.intrinsic_rank)
        anchor = _project_to_manifold(
            self.atom_raw_anchor.detach(), self.cfg.atom_manifold, self.cfg.intrinsic_rank
        )
        out: dict[int, torch.Tensor] = {}
        if self.cfg.atom_manifold == "circle":
            theta = torch.linspace(0.0, 1.0, grid_size, dtype=anchor.dtype, device=anchor.device)
            probe = theta.reshape(grid_size, 1)
            for i in range(F):
                curves = _eval_basis_on_manifold(
                    probe,
                    self.cfg,
                    self.duchon_centers if self.cfg.atom_basis == "duchon" else None,
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
        else:
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

    def decoder_ortho_penalty(self) -> torch.Tensor:
        """Rust ``block_orthogonality`` descriptor over per-atom decoder groups.

        See the constructor for the transpose contract: we present the decoder
        as ``(input_dim, F·K)`` so each atom owns a contiguous block of ``K``
        latent (column) axes, matching the Rust partition validator.
        """
        if self._ortho_penalty is None:
            return self.decoder_blocks.new_zeros(())
        # decoder_blocks: (F, K, D). Permute → (D, F, K) → reshape → (D, F·K).
        flat = self.decoder_blocks.permute(2, 0, 1).reshape(
            self.cfg.input_dim, -1
        )
        return self._ortho_penalty(flat)

    def decoder_monotonicity_penalty(self) -> torch.Tensor:
        """Rust ``monotonicity`` descriptor along each atom's basis-coefficient axis."""
        if self._monotonicity_penalty is None:
            return self.decoder_blocks.new_zeros(())
        # The Rust penalty walks adjacent rows along the leading axis of a
        # row-major (n_eff, d) block. We apply it per-atom by summing.
        total = self.decoder_blocks.new_zeros(())
        for i in range(int(self.cfg.n_atoms)):
            total = total + self._monotonicity_penalty(self.decoder_blocks[i])
        return total

    def sparsity_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """Rust-backed sparsity penalty value at ``logits``."""
        return self.sparsity.penalty(logits)

    def regularization(self, logits: torch.Tensor | None = None) -> torch.Tensor:
        """Sum of Rust-backed regularizers used by the loss."""
        reg = self.decoder_ortho_penalty() + self.decoder_monotonicity_penalty()
        if logits is not None:
            reg = reg + self.sparsity_penalty(logits)
        return reg


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
