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
shells out to ``sae_manifold_fit_minimal`` in Rust. The module's current
amortized encoder supplies the real ``t_init`` / ``a_init`` warm-start slots,
and the Rust certified inner solve refines those seeds to stationarity. Direct
closed-form parity is therefore obtained by passing the same initializers to
``gamfit.sae_manifold_fit``.

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
            data["tau_start"] = start
            data["tau_min"] = end
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
    encoder_hidden: int = 16
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
        """Map the ``(atom_manifold, atom_basis)`` pair to the Rust ``atom_basis``
        token accepted by ``sae_manifold_fit_minimal``.

        The closed-form SAE builder (``sae_build_atom_plans`` in
        ``crates/gam-pyffi/src/lib.rs``) accepts exactly these atom-basis kinds:
        ``periodic``, ``duchon``, ``sphere``, ``torus``, ``euclidean_patch``.
        There is **no** closed-form ``bspline`` atom basis (B-splines exist only
        as the 1-D ``basis_with_jet`` torch-side basis), so a ``bspline`` config
        cannot be honored by ``.fit()`` and is rejected loudly rather than
        silently coerced to ``duchon``.

        The pairing is explicit because the manifold and the basis are not
        independent: a ``circle`` is intrinsically periodic, a ``sphere`` uses
        the spherical chart, and a ``cylinder`` (one periodic + one Euclidean
        axis) is *not* a ``torus`` (two periodic axes). The closed-form builder
        has no genuine cylinder atom, so a cylinder config is rejected rather
        than misrepresented as a torus.
        """
        manifold, basis = self.atom_manifold, self.atom_basis
        if manifold == "circle":
            # Circle is intrinsically periodic; the user's basis choice (the
            # forward-path B-spline/Duchon knob) does not apply to the
            # closed-form periodic atom.
            return "periodic"
        if manifold == "sphere":
            return "sphere"
        if manifold == "product":
            if basis == "bspline":
                raise NotImplementedError(
                    "closed-form .fit() has no B-spline atom basis; "
                    f"unsupported (manifold, basis)={(manifold, basis)}. "
                    "Use a torch training loop, or atom_basis='duchon'/'fourier'."
                )
            # Euclidean product patch (Duchon m-spline / radial). 'fourier' here
            # selects the periodic angular treatment the builder folds into the
            # Euclidean patch; 'duchon' is the native radial basis.
            return "duchon"
        # cylinder: one periodic + one non-periodic axis. The closed-form
        # builder offers 'torus' (two periodic axes) and 'euclidean_patch', but
        # neither is a faithful cylinder, so refuse rather than coerce.
        raise NotImplementedError(
            f"closed-form basis unsupported for {(manifold, basis)}: the "
            "closed-form SAE builder has no cylinder (one periodic + one "
            "Euclidean axis) atom; 'torus' would silently change the topology. "
            "Use a torch training loop for cylinder atoms."
        )

    def closed_form_assignment(self) -> str:
        """Map the torch sparsity kind to the closed-form ``assignment`` token.

        The closed-form path accepts ``ibp_map``, ``softmax``, and ``jumprelu``.
        Note that ``softmax_topk`` is **not** mapped to ``softmax``: the torch
        ``softmax_topk`` layer is an *independent* non-negative top-k gate
        (softplus magnitude + hard top-k STE) that can turn **all** atoms off,
        whereas row-``softmax`` is a competitive simplex whose mass always sums
        to one and can never deselect every atom. Coercing one into the other
        would make ``.fit()`` optimize a fundamentally different model than
        backprop. The closest closed-form mode with the same semantics —
        independent gates that can zero every atom, plus the existing post-hoc
        hard top-k projection (forwarded via ``top_k`` in :meth:`fit`) — is
        ``jumprelu`` (independent hard-thresholded gates). So ``softmax_topk``
        maps to ``jumprelu``, aligning the closed-form objective with the torch
        independent-topk gate.
        """
        return {
            "ibp_gumbel": "ibp_map",
            "softmax_topk": "jumprelu",
            "jumprelu": "jumprelu",
        }[self.sparsity.kind]


@dataclass(frozen=True, slots=True)
class ManifoldSAEOutput:
    """Bundle returned by :class:`ManifoldSAE.forward`.

    ``amplitudes`` is the **honest** per-atom magnitude the decoder actually
    applied — it equals the reconstruction code ``z`` (so an atom that
    contributes nothing to ``x_hat`` reports a zero amplitude). For the
    magnitude-carrying gates (``jumprelu`` / ``softmax_topk``) the raw,
    *pre-mask* softplus activation — which is strictly positive even for atoms
    the top-k / threshold dropped — is exposed separately as ``raw_magnitudes``
    so interpretability code never mistakes a dropped atom for an active one.
    For IBP-Gumbel the code factorizes as ``z = gate · amp``; there
    ``raw_magnitudes`` is that separate ``amp`` magnitude.
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
    raw_magnitudes: torch.Tensor


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
    # Cylinder / product (intrinsic_rank > 1):
    #
    # The B-spline `basis_with_jet` kernel IS genuinely 1-D — the Rust FFI
    # rejects any `t` with more than one column — so a B-spline product/cylinder
    # has no full-dimensional torch basis and the later intrinsic coordinates
    # would be silently dead. That combination is refused.
    #
    # The Duchon `basis_with_jet` kernel, by contrast, is fully `d`-dimensional:
    # it accepts `(N, d)` points and `(K, d)` centers and returns a `(N, M, d)`
    # jet — an honest per-axis Jacobian for every intrinsic coordinate (see
    # `duchon_basis_with_jet` in `crates/gam-pyffi/src/model_ffi.rs`). So a
    # multi-dimensional flat `product` patch is genuinely supported: we pass the
    # full `t` and a deterministic `(K, d)` center cloud (the 1-D centers lifted
    # to the `d`-cube by the low-discrepancy `_duchon_centers_nd` Kronecker
    # sequence) instead of collapsing to `t[:, :1]`. The `cylinder` case is
    # refused below: its periodic axis has no topology-faithful torch kernel.
    if cfg.atom_basis == "bspline":
        if int(cfg.intrinsic_rank) > 1:
            raise NotImplementedError(
                f"atom_manifold={cfg.atom_manifold!r} with intrinsic_rank="
                f"{cfg.intrinsic_rank} and atom_basis='bspline' has no "
                "full-dimensional torch basis: the 'bspline' basis_with_jet "
                "kernel is intrinsically 1-D, so the second and later intrinsic "
                "coordinates would be silently dead. Use intrinsic_rank=1, "
                "atom_basis='duchon' (its basis_with_jet kernel is full "
                "d-dimensional), or a manifold whose basis is intrinsically "
                "multi-dimensional (sphere)."
            )
        params = {
            "n_basis": int(cfg.n_basis_per_atom),
            "degree": 3,
            "order": int(cfg.basis_order),
            "periodic": False,
        }
        return apply(t[:, :1], "bspline", json.dumps(params))
    if centers is None:
        raise ValueError("Duchon-style manifold requires centers")
    d = int(cfg.intrinsic_rank)
    if d > 1:
        # The multi-d Duchon `basis_with_jet` kernel is a *flat* Euclidean patch
        # (its radial kernel is the plain Euclidean chord; there is no periodic
        # chord embedding in the jet helper `duchon_sae_atom_basis_with_jet`).
        # That is exactly right for a `product` patch (genuinely R^d). A
        # `cylinder` (S¹ × ℝ), however, needs the leading axis wrapped — fitting
        # it on a flat patch would silently drop the S¹ topology (the very
        # "torus would silently change the topology" hazard the closed-form
        # cylinder refusal names). The topology-faithful CylinderHarmonicEvaluator
        # exists in the Rust core but is reachable only through the closed-form /
        # structure-search birth path, NOT through `basis_with_jet`. So a
        # multi-d `cylinder` forward is refused (accurately), while `product`
        # is wired to the genuine flat multi-d Duchon basis.
        if cfg.atom_manifold == "cylinder":
            raise NotImplementedError(
                "atom_manifold='cylinder' has no topology-faithful torch "
                "basis: the multi-d 'duchon' basis_with_jet kernel is a flat "
                "Euclidean patch, so the periodic S¹ axis would be silently "
                "treated as a line (the topology-change hazard). The genuine "
                "CylinderHarmonicEvaluator is reachable only through the "
                "closed-form structure-search birth path, not basis_with_jet. "
                "Use atom_manifold='product' for a flat patch, atom_manifold="
                "'circle' (intrinsic_rank=1) for a pure periodic axis, or let "
                "the closed-form structure search grow a cylinder by evidence."
            )
        centers_nd = _duchon_centers_nd(centers, d)
        centers_list = to_numpy_f64(centers_nd).tolist()
        params = {"centers": centers_list, "m": int(cfg.basis_order)}
        return apply(t[:, :d], "duchon", json.dumps(params))
    centers_list = to_numpy_f64(centers.reshape(-1, 1)).tolist()
    params = {"centers": centers_list, "m": int(cfg.basis_order)}
    return apply(t[:, :1], "duchon", json.dumps(params))


def _duchon_centers_nd(centers_1d: torch.Tensor, d: int) -> torch.Tensor:
    """Lift the ``(K,)`` 1-D Duchon centers to a ``(K, d)`` cloud in ``[0, 1]^d``.

    Axis 0 keeps the caller's 1-D centers (so the periodic/leading coordinate of
    a cylinder or product patch is seeded exactly as in the 1-D case). The
    remaining ``d - 1`` axes are filled by an additive-recurrence (Kronecker /
    generalized-golden-ratio) low-discrepancy sequence keyed only to ``(K, d)``:
    deterministic, buffer-free, and non-degenerate (the centers do not collapse
    onto a diagonal, so the multi-axis Duchon kernel is well-conditioned). The
    centers are a *fixed* design the decoder learns against, so any deterministic
    well-spread placement is admissible; this one is reproducible and stable
    across forward calls without enlarging the serialized state.
    """
    base = centers_1d.reshape(-1, 1)
    k = int(base.shape[0])
    dtype = base.dtype
    device = base.device
    if d <= 1 or k == 0:
        return base
    # Generalized golden ratio phi_d: the real root of x^{d} = x + 1; its
    # reciprocal powers give the canonical R_d low-discrepancy generators.
    phi = 2.0
    for _ in range(32):
        phi = (1.0 + phi) ** (1.0 / float(d))
    alphas = [((1.0 / phi) ** (j + 1)) % 1.0 for j in range(d - 1)]
    idx = torch.arange(1, k + 1, dtype=dtype, device=device).reshape(-1, 1)
    extra = torch.empty((k, d - 1), dtype=dtype, device=device)
    for j, a in enumerate(alphas):
        extra[:, j] = torch.remainder(idx[:, 0] * float(a) + 0.5, 1.0)
    return torch.cat([base, extra], dim=-1)


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
    ``analytic_penalty_value_grad`` in Rust. The ``softmax_topk`` arm is a
    top-k SAE gate (per-atom independent non-negative activation, hard top-k
    forward with a straight-through estimator); its loss penalty is an L1
    sparsity pressure on that gate, the differentiable analogue of the
    closed-form ``.fit()`` path's entropy-toward-one-hot assignment sparsity.
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
            return self._topk_gate(logits), logits
        # JumpReLU activation: hard threshold forward, Rust-STE backward.
        assignments = self._jumprelu.gate(logits)
        return assignments, logits

    def _topk_activation(self, logits: torch.Tensor) -> torch.Tensor:
        tau = max(float(self.tau.item()), 1e-6)
        return tau * F_torch.softplus(logits / tau)

    def _topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.target_k, scores.shape[-1])
        _, top_idx = torch.topk(scores, k=k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, top_idx, 1.0)
        return mask

    def _topk_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Top-k SAE gate: per-atom **independent** non-negative activation.

        The previous implementation applied a row-wise ``softmax`` over the atom
        axis before the top-k mask. A softmax is a *competitive simplex*: its
        per-row mass is normalized to one, so it can only express which atom wins
        the competition for a row, never whether a feature is *present at all*.
        When a planted feature is absent the encoder cannot turn every atom off —
        the simplex always redistributes ~1.0 of assignment mass — so the gate
        is structurally unable to correlate with feature presence and routing
        collapses into distributed/entangled atoms (issue #583).

        The correct construction for one-atom-per-feature routing is the standard
        top-k SAE encoder: a non-negative activation computed *independently per
        atom*, with the top-k largest kept and the rest zeroed. Independence lets
        every atom sit near zero when its feature is absent and rise on its own
        when present, so the activation tracks features; the top-k constraint
        supplies the sparsity that disentangles atoms. A temperature-scaled
        softplus gives a smooth, strictly-non-negative activation whose hardness
        anneals through the shared ``tau`` schedule (``tau → 0`` ⇒ ReLU). The
        hard top-k mask is applied with a straight-through estimator so gradients
        reach the selected atoms' pre-activations.
        """
        act = self._topk_activation(logits)
        hard = act * self._topk_mask(act)
        # Hard top-k value and masked gradient. Sending reconstruction gradients
        # through inactive atoms teaches every atom every selected row, which is
        # exactly the routing collapse this gate is meant to prevent.
        return hard

    def reconstruction_topk_gate(
        self,
        route_logits: torch.Tensor,
        magnitude_logits: torch.Tensor,
        x: torch.Tensor,
        per_atom_recon: torch.Tensor,
    ) -> torch.Tensor:
        """Residual-energy top-k gate for gradient-trained ``softmax_topk``.

        A logits-only top-k mask can pick an arbitrary atom for a row even when
        another atom's current curve reconstructs the row better. On symmetric
        dictionaries that lets every atom learn every manifold while the hard
        top-1 labels remain chance-level. The closed-form SAE avoids that
        collapse by assigning rows from residual energy, so the gradient path
        does the same local decision here.

        For each row/atom pair, solve the best non-negative scalar code against
        the atom's current reconstruction curve, route by the resulting
        residual, and use that scalar as the forward code. Gradients still flow
        through the selected encoder activation, so the amplitude head remains
        trainable while unselected atoms get no reconstruction-gradient credit
        for the row.
        """
        amp = self._topk_activation(magnitude_logits)
        denom = per_atom_recon.square().sum(dim=-1).clamp_min(1e-12)
        code = (per_atom_recon * x.unsqueeze(1)).sum(dim=-1) / denom
        code = code.clamp_min(0.0)
        residual = ((code.unsqueeze(-1) * per_atom_recon - x.unsqueeze(1)) ** 2).sum(
            dim=-1
        )
        mask = self._topk_mask(route_logits - residual)
        return amp * mask

    def compose_code(self, assignments: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        """Per-atom latent code ``z`` from gate ``assignments`` and ``amp``.

        The factorization depends on the gate's range. The IBP-Gumbel gate is a
        dimensionless activation probability in ``[0, 1]``, so the code is the
        gate scaled by the separate non-negative magnitude ``amp`` (``z = gate ·
        amp``). The JumpReLU and top-k gates are *magnitude-carrying*
        activations — the gate already equals the SAE code — so multiplying by
        ``amp`` again would square the magnitude and distort the
        reconstruction gradient; the gate is returned as the code directly.
        """
        if self.kind == "ibp_gumbel":
            return assignments * amp
        return assignments

    def penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """Rust-backed scalar penalty value at ``logits``."""
        if self.kind == "ibp_gumbel":
            return self._ibp(logits)
        if self.kind == "jumprelu":
            return self._jumprelu(logits)
        # Top-k SAE: an L1 pressure on the (non-negative, top-k-masked)
        # activations. Top-k alone caps the number of active atoms but does not
        # push an *unneeded* selected atom's activation toward zero, so without
        # this term a row can keep ``target_k`` atoms weakly co-active and stay
        # entangled. The L1 on the gate is the convex sparsity prior that drives
        # absent-feature activations to zero, the differentiable analogue of the
        # closed-form softmax path's entropy-toward-one-hot sparsity.
        gate = self._topk_gate(logits)
        return gate.abs().mean()


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

        # The forward path evaluates a Duchon kernel both for an explicit
        # `atom_basis='duchon'` AND for every `product`/`cylinder` patch (whose
        # multi-axis torch basis is the full d-dimensional Duchon kernel — there
        # is no separate tensor-product `basis_with_jet` kind). Seed real
        # linspace centers for both so the multi-d center lift has a
        # non-degenerate axis-0 seed; only the genuinely center-free manifolds
        # (circle/sphere with a built-in chart) keep the zero placeholder.
        _duchon_backed = cfg.atom_basis == "duchon" or cfg.atom_manifold in (
            "product",
            "cylinder",
        )
        if _duchon_backed:
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
        # Serialized closed-form fit. After ``.fit()`` the full solved state
        # (decoder blocks, basis centers, anchors/coords, training data, fitted
        # reconstruction, and every scalar the frozen-decoder OOS solve needs)
        # is JSON-encoded via ``fit.to_dict()`` and stored here as a uint8 byte
        # buffer so ``state_dict()`` carries it. An empty buffer means "no
        # closed-form solve" (the gradient-trained path). The variable-length
        # blob is reloaded by the overridden ``_load_from_state_dict`` below,
        # which rebuilds ``self._last_fit`` so a reloaded module reproduces both
        # in-sample and out-of-sample predictions.
        self.register_buffer(
            "_fit_blob", torch.zeros(0, dtype=torch.uint8), persistent=True
        )
        self.register_buffer(
            "_top1_route_centroids",
            torch.zeros(F, D, dtype=dt),
            persistent=True,
        )
        self.register_buffer(
            "_top1_route_initialized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )
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

    @property
    def _forward_centers(self) -> torch.Tensor | None:
        """Duchon centers the forward path feeds to ``basis_with_jet``.

        The forward Duchon kernel is used for an explicit ``atom_basis='duchon'``
        AND for every ``product``/``cylinder`` patch (whose multi-axis torch
        basis is the full d-dimensional Duchon kernel). Both seed real centers
        in the constructor, so return them whenever the forward is Duchon-backed;
        circle/sphere carry a built-in chart and need no centers.
        """
        if self.cfg.atom_basis == "duchon" or self.cfg.atom_manifold in (
            "product",
            "cylinder",
        ):
            return self.duchon_centers
        return None

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        F = int(self.cfg.n_atoms)
        d = int(self.cfg.intrinsic_rank)
        raw = self.encoder(x)
        raw_positions = raw[..., : F * d].reshape(x.shape[0], F, d)
        amp_logits = raw[..., F * d :]
        return raw_positions, amp_logits

    def _uses_top1_energy_router(self) -> bool:
        return (
            self.cfg.sparsity.kind == "softmax_topk"
            and self.cfg.sparsity.target_k == 1
            and int(self.cfg.n_atoms) > 1
        )

    @staticmethod
    def _row_energy_features(x: torch.Tensor) -> torch.Tensor:
        feat = x.detach().square()
        denom = feat.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        return feat / denom

    @torch.no_grad()
    def _maybe_initialize_top1_energy_router(self, x: torch.Tensor) -> None:
        if not self._uses_top1_energy_router():
            return
        if bool(self._top1_route_initialized.item()):
            return
        feat = self._row_energy_features(x)
        n, d = int(feat.shape[0]), int(feat.shape[1])
        f = int(self.cfg.n_atoms)
        if n == 0:
            raise ValueError("ManifoldSAE cannot initialize top-1 router from an empty batch")

        centroids = torch.empty((f, d), dtype=feat.dtype, device=feat.device)
        first = int(torch.argmax(feat.norm(dim=1)).item())
        centroids[0] = feat[first]
        chosen = torch.zeros(n, dtype=torch.bool, device=feat.device)
        chosen[first] = True
        for atom in range(1, f):
            dist = torch.cdist(feat, centroids[:atom]).amin(dim=1)
            dist = dist.masked_fill(chosen, -1.0)
            idx = int(torch.argmax(dist).item())
            centroids[atom] = feat[idx]
            chosen[idx] = True

        for _ in range(12):
            dist = torch.cdist(feat, centroids)
            labels = torch.argmin(dist, dim=1)
            for atom in range(f):
                rows = labels == atom
                if bool(rows.any().item()):
                    centroids[atom] = feat[rows].mean(dim=0)

        order = sorted(
            range(f),
            key=lambda i: (
                int(torch.argmax(centroids[i]).item()),
                [float(v) for v in centroids[i].detach().cpu()],
            ),
        )
        centroids = centroids[order]
        centroids = centroids / centroids.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        self._top1_route_centroids.copy_(
            centroids.to(
                dtype=self._top1_route_centroids.dtype,
                device=self._top1_route_centroids.device,
            )
        )
        self._top1_route_initialized.fill_(True)

    def _top1_energy_route_logits(
        self, x: torch.Tensor, amp_logits: torch.Tensor
    ) -> torch.Tensor:
        if not self._uses_top1_energy_router():
            return amp_logits
        self._maybe_initialize_top1_energy_router(x)
        feat = self._row_energy_features(x).to(dtype=amp_logits.dtype, device=amp_logits.device)
        centroids = self._top1_route_centroids.to(
            dtype=amp_logits.dtype, device=amp_logits.device
        )
        similarity = feat @ centroids.transpose(0, 1)
        return amp_logits + 8.0 * similarity

    @torch.no_grad()
    def _closed_form_initializers(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """Amortized encoder seeds for the Rust closed-form joint solve.

        ``gamfit.sae_manifold_fit`` already owns the production warm-start
        contract: ``t_init`` is ``(K, N, D_max)`` per-atom coordinates and
        ``a_init`` is ``(N, K)`` raw assignment logits. This helper is the torch
        cotrain bridge for #1154 Design A: predict those seeds with the current
        encoder, then let the certified Rust inner solve refine to stationarity.
        """
        raw_positions, amp_logits = self._encode(x)
        raw_with_anchor = raw_positions + self.atom_raw_anchor.unsqueeze(0)
        positions = _project_to_manifold(
            raw_with_anchor, self.cfg.atom_manifold, self.cfg.intrinsic_rank
        )
        t_init = positions.detach().cpu().numpy().transpose(1, 0, 2)
        a_init = amp_logits.detach().cpu().numpy()
        return {
            "t_init": np.ascontiguousarray(t_init, dtype=np.float64),
            "a_init": np.ascontiguousarray(a_init, dtype=np.float64),
        }

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
        raw_positions, amp_logits = self._encode(x)
        raw_with_anchor = raw_positions + self.atom_raw_anchor.unsqueeze(0)
        positions = _project_to_manifold(
            raw_with_anchor, self.cfg.atom_manifold, self.cfg.intrinsic_rank
        )
        curves = _eval_basis_on_manifold(
            positions,
            self.cfg,
            self._forward_centers,
        )
        route_logits = self._top1_energy_route_logits(x, amp_logits)
        per_atom_recon = torch.einsum("nfk,fkd->nfd", curves, self.decoder_blocks)
        amp = F_torch.softplus(amp_logits)
        if self.cfg.sparsity.kind == "softmax_topk":
            gate_pre = route_logits
            assignments = self.sparsity.reconstruction_topk_gate(
                route_logits, amp_logits, x, per_atom_recon
            )
            z = assignments
        else:
            assignments, gate_pre = self.sparsity(route_logits)
            z = self.sparsity.compose_code(assignments, amp)
        x_hat = (z.unsqueeze(-1) * per_atom_recon).sum(dim=1)

        reml_score = torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
        lambdas = torch.exp(self.log_lambda).to(dtype=x.dtype, device=x.device)

        # `amplitudes` is the magnitude the decoder actually used (== z): for the
        # magnitude-carrying gates z is the top-k/threshold-masked code, so a
        # dropped atom reports zero, not its raw softplus value. The raw softplus
        # magnitude is preserved separately as `raw_magnitudes`.
        return ManifoldSAEOutput(
            z=z,
            x_hat=x_hat,
            positions=positions,
            amplitudes=z,
            curves=curves,
            gate=gate_pre,
            assignments=assignments,
            reml_score=reml_score,
            lambdas=lambdas,
            raw_magnitudes=amp,
        )

    def _forward_from_closed_form(self, x: torch.Tensor) -> ManifoldSAEOutput:
        # When .fit() has been called, the closed-form Rust solve is the source
        # of truth: forward must reproduce the fit exactly for in-sample x AND
        # reconstruct genuinely-unseen rows out of sample.
        #
        # Both cases go through ``fit.converged_latents(x)``: it returns the
        # stored training latents bit-exactly when x matches the training batch,
        # and otherwise runs the frozen-decoder out-of-sample Newton solve
        # (``sae_manifold_predict_oos``) — the SAME per-row latent inner problem
        # the joint fit solved — holding the fitted decoder blocks / basis /
        # anchors fixed, then applies the decoder to get x_hat. The encoder did
        # participate in .fit() by providing the warm-start seeds; after the
        # certified solve, the fitted Rust state is authoritative.
        fit = self._last_fit
        assert fit is not None
        F = int(self.cfg.n_atoms)
        d = int(self.cfg.intrinsic_rank)
        x_np = x.detach().cpu().numpy()
        latents = fit.converged_latents(x_np)
        fitted_np = np.asarray(latents["fitted"], dtype=np.float64)
        assignments_np = np.asarray(latents["assignments"], dtype=np.float64)
        coords_per_atom = [np.asarray(c, dtype=np.float64) for c in latents["coords"][:F]]
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
            self._forward_centers,
        )
        z = assignments
        gate = assignments
        reml_score = torch.tensor(
            float(fit.reml_score), dtype=x.dtype, device=x.device
        )
        lambdas = torch.exp(self.log_lambda).to(dtype=x.dtype, device=x.device)
        # The closed-form code *is* the assignment (the solver carries no
        # separate softplus magnitude), so the honest per-atom amplitude — what
        # the decoder applied — equals z, and there is no distinct raw magnitude.
        return ManifoldSAEOutput(
            z=z,
            x_hat=x_hat,
            positions=positions,
            amplitudes=z,
            curves=curves,
            gate=gate,
            assignments=assignments,
            reml_score=reml_score,
            lambdas=lambdas,
            raw_magnitudes=z,
        )

    def fit(
        self,
        x: torch.Tensor,
        *,
        max_iter: int | None = None,
        random_state: int = 0,
        learning_rate: float | None = None,
    ) -> _ClosedFormManifoldSAE:
        """Run the closed-form Rust solve and put this module into the solved state.

        After this returns the module is in a "closed-form solved" state: the
        returned fit (also cached as ``self._last_fit``) is the source of truth.
        :meth:`forward` is rerouted through :meth:`_forward_from_closed_form`,
        which reconstructs ``x_hat`` / ``positions`` / ``assignments`` /
        ``reml_score`` from the fit. Before the solve, the module's encoder
        supplies ``t_init`` / ``a_init`` to warm-start the Rust certified inner
        solve; after stationarity, the closed-form solver carries its own
        decoder and per-atom anchors. The solved ``decoder_blocks`` and (when
        the solve used a common Duchon center set) the ``duchon_centers`` buffer
        are folded into the module, and the full solved state is captured in the
        ``_fit_blob`` buffer for serialization.

        A solved module is genuinely usable:

        * **Out-of-sample forward.** ``module(x_new)`` for unseen rows works:
          :meth:`_forward_from_closed_form` re-encodes new rows by solving the
          per-row latent inner problem against the *fitted* decoder / basis /
          anchors (the frozen-decoder Newton solve, ``sae_manifold_predict_oos``)
          and applies the decoder — the transductive encode step. In-sample ``x``
          still returns the exact closed-form reconstruction bit-for-bit.
        * **Serialization.** ``state_dict()`` captures the full solved state in
          the ``_fit_blob`` byte buffer (decoder blocks, per-atom basis centers /
          anchors, fitted scalars, training data, and fitted reconstruction), so
          ``load_state_dict(module.state_dict())`` into a fresh module rebuilds
          ``self._last_fit`` and reproduces both in-sample and out-of-sample
          predictions. (``fit.to_dict()`` / ``fit.save(...)`` remains available
          to persist the fit object directly.)

        A solved module must still NOT be used for gradient training /
        fine-tuning: the encoder and anchor were warm-start predictors, not the
        post-refinement fitted state, so their gradients are stale relative to
        the closed-form decoder. Build a fresh, unfitted module for gradient
        training.
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
        if cfg.sparsity.target_k is not None:
            kwargs["top_k"] = int(cfg.sparsity.target_k)
        # Regularizer parity: forward every weight the closed-form FFI honors,
        # and refuse — loudly — any configured-nonzero regularizer it cannot,
        # rather than silently dropping it (which would make two differently
        # regularized configs produce identical fits).
        #
        # `decoder.ortho_weight` penalizes cross-correlation between the per-atom
        # *decoder blocks* (R_g·R_hᵀ); the closed-form `block_orthogonality`
        # knob instead orthogonalizes the latent "t" coordinate block — a
        # different objective. Coercing one into the other would silently change
        # the model, so a nonzero decoder ortho weight is unsupported here.
        if cfg.decoder.ortho_weight > 0.0:
            raise NotImplementedError(
                "closed-form .fit() cannot honor DecoderConfig.ortho_weight: the "
                "closed-form block-orthogonality penalty acts on the latent 't' "
                "block, not on cross-atom decoder-block correlations, so it is a "
                "different objective. Train with a torch loop (uses "
                "decoder_ortho_penalty()), or set decoder.ortho_weight=0."
            )
        # The closed-form FFI exposes no decoder-monotonicity penalty at all.
        if cfg.decoder.monotonicity_weight > 0.0:
            raise NotImplementedError(
                "closed-form .fit() has no decoder-monotonicity penalty "
                "(DecoderConfig.monotonicity_weight > 0 is unsupported); train "
                "with a torch loop (uses decoder_monotonicity_penalty()), or set "
                "decoder.monotonicity_weight=0."
            )
        # The JumpReLU hard-gate threshold is part of the assignment objective:
        # forward it on the jumprelu-assignment path (kind 'jumprelu' or, via
        # closed_form_assignment(), 'softmax_topk'). For other assignments the
        # threshold is meaningless and is not forwarded.
        if cfg.closed_form_assignment() == "jumprelu":
            kwargs["jumprelu_threshold"] = float(cfg.sparsity.jumprelu_threshold)
        kwargs.update(self._closed_form_initializers(x))
        fit = _closed_form_sae_manifold_fit(
            X=to_numpy_f64(x),
            K=int(cfg.n_atoms),
            d_atom=int(cfg.intrinsic_rank),
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
        """Fold the full solved state into params/buffers for serialization.

        The solve's *reconstruction* pieces (decoder blocks + a common Duchon
        center set, when the solve used one) are copied into the matching
        ``decoder_blocks`` parameter and ``duchon_centers`` buffer so a reloaded
        module's eager-path basis is consistent with the solve. The encoder and
        ``atom_raw_anchor`` are intentionally **not** overwritten: the
        closed-form solver carries its own decoder and anchors (per-atom
        coordinates), so there is no encoder to fold in — out-of-sample forward
        re-encodes by the frozen-decoder inner solve instead of an amortized
        encoder pass.

        The authoritative, fully self-contained state — decoder blocks AND
        per-atom basis centers/anchors AND every fitted scalar (alpha, tau,
        sparsity_strength, smoothness, learning_rate, max_iter, random_state)
        AND the training data + fitted reconstruction — is captured by
        JSON-encoding ``fit.to_dict()`` into the ``_fit_blob`` byte buffer. That
        buffer round-trips through ``state_dict()`` and is decoded back into a
        live ``self._last_fit`` by :meth:`_load_from_state_dict`, so a reloaded
        module reproduces both in-sample and out-of-sample predictions without
        needing the original Python fit object.
        """
        blob = json.dumps(fit.to_dict()).encode("utf-8")
        self._fit_blob = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
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
        # Fold the solved Duchon centers into the shared eager-path buffer when
        # the solve used a single common center set of width K, so a reloaded
        # module's eager basis matches the solve. Heterogeneous per-atom centers
        # cannot be represented by the shared (K,) buffer, but they are still
        # serialized in full (per atom) inside ``_fit_blob`` and used by the
        # closed-form forward path, so nothing is lost.
        if self.cfg.atom_basis == "duchon":
            centers = [c for c in fit._duchon_centers[:F] if c is not None]
            if len(centers) == F:
                stacked = [np.asarray(c, dtype=np.float64).reshape(-1) for c in centers]
                if all(s.shape == (K,) for s in stacked) and all(
                    np.allclose(s, stacked[0]) for s in stacked
                ):
                    self.duchon_centers.copy_(
                        torch.as_tensor(stacked[0], dtype=self.duchon_centers.dtype)
                    )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reload a serialized closed-form fit from the ``_fit_blob`` buffer.

        ``_fit_blob`` is a variable-length byte tensor, so the default
        ``load_state_dict`` (which ``copy_``-s into a fixed-shape buffer) would
        raise on a size mismatch. We intercept it here: pop the blob, resize our
        own buffer to match, copy the bytes, and decode it back into a live
        :class:`gamfit._sae_manifold.ManifoldSAE` so the reloaded module's
        :meth:`forward` again routes through :meth:`_forward_from_closed_form`
        and reproduces in-sample and out-of-sample predictions. An empty blob
        means the source module had no closed-form solve, so ``self._last_fit``
        is cleared and the eager (gradient-trained) path is used.
        """
        blob_key = prefix + "_fit_blob"
        blob_tensor: torch.Tensor | None = None
        if blob_key in state_dict:
            incoming = state_dict[blob_key]
            blob_tensor = torch.as_tensor(incoming, dtype=torch.uint8).reshape(-1)
            # The blob is variable length, so resize our registered buffer to
            # match before the default loader runs. Substitute the cloned tensor
            # back into the dict so the default machinery's shape-checked copy_
            # is a same-shape no-op (and the key is not flagged missing).
            self._fit_blob = blob_tensor.clone()
            state_dict = dict(state_dict)
            state_dict[blob_key] = self._fit_blob
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if blob_tensor is not None:
            self._rebuild_fit_from_blob(blob_tensor)

    def _rebuild_fit_from_blob(self, blob: torch.Tensor) -> None:
        """Decode ``blob`` into ``self._last_fit`` (or clear it when empty)."""
        if blob.numel() == 0:
            self._last_fit = None
            return
        raw = blob.detach().cpu().numpy().tobytes()
        payload = json.loads(raw.decode("utf-8"))
        self._last_fit = _ClosedFormManifoldSAE.from_dict(payload)

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

    @property
    def cotrain(self) -> dict[str, Any] | None:
        """Co-trained REML + amortized-encoder diagnostics from the last fit."""
        if self._last_fit is None:
            return None
        return None if self._last_fit.cotrain is None else dict(self._last_fit.cotrain)

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
                    self._forward_centers,
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
                    self._forward_centers,
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
