"""Manifold SAE as a torch ``nn.Module``.

Atom ``i`` is a 1-D parametric curve in ambient ``R^D`` parameterized by a
point ``theta_i`` on a manifold ``M`` (Circle, Cylinder, Sphere, Product) and
a decoder block ``D_i`` of shape ``(K, D)``. A shared encoder maps each input
``x`` to per-atom on-manifold coordinates ``theta_i(x)`` and a gate logit
``l_i(x)``. The atom's contribution to reconstruction is

    a_i(x) * sum_k phi_k(theta_i(x)) * D_i[k, :]

where ``a_i(x)`` is the Rust-computed sparsity gate. For the closed-form-
mappable gate families (IBP-Gumbel / JumpReLU) the gate is bounded in
``[0, 1)`` and all reconstruction magnitude lives in the decoder curves —
exactly the family the Rust closed-form solve optimizes; there is no separate
per-token amplitude. The torch-only ``softmax_topk`` lane keeps its
magnitude-carrying activation (it has no closed-form counterpart and is
refused by ``closed_form_assignment``).

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
    jumprelu_bounded_gate,
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
            # A colon-form spec ("linear:4.0->1.0") carries an embedded
            # start/end and must be parsed; a bare kind ("linear" / "geometric"
            # / "reciprocal_iter") is already a valid field value and is taken
            # as-is (parsing it would wrongly demand a ':').
            if ":" in sched:
                kind, start, end = cls._parse_tau_schedule(sched)
                data["tau_schedule"] = kind
                data["tau_start"] = start
                data["tau_min"] = end
            else:
                kind = sched.strip().lower()
                if kind not in {"linear", "geometric", "reciprocal_iter"}:
                    raise ValueError(
                        "SparsityConfig.tau_schedule must be one of "
                        "'linear'/'geometric'/'reciprocal_iter' (or a "
                        f"colon-form spec like 'linear:4.0->1.0'); got {sched!r}"
                    )
                data["tau_schedule"] = kind
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
    n_atoms: int | None = None
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
    # ``K`` is constructor sugar for ``n_atoms`` (the spelling the docs and the
    # closed-form ``sae_manifold_fit`` API use). It is normalized into
    # ``n_atoms`` in ``__post_init__`` and stored as ``None`` afterwards, so
    # ``dataclasses.replace`` round-trips and ``n_atoms`` stays the single
    # source of truth.
    K: int | None = None

    def __post_init__(self) -> None:
        if self.K is not None:
            if self.n_atoms is not None and int(self.n_atoms) != int(self.K):
                raise ValueError(
                    f"ManifoldSAEConfig: n_atoms={self.n_atoms} and K={self.K} "
                    "are aliases and must agree when both are given"
                )
            object.__setattr__(self, "n_atoms", int(self.K))
            object.__setattr__(self, "K", None)
        if self.n_atoms is None:
            raise ValueError("ManifoldSAEConfig requires n_atoms (alias: K)")
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

        ``softmax_topk`` has **no** faithful closed-form assignment and is
        rejected rather than coerced. It is neither ``softmax`` (a competitive
        simplex whose mass always sums to one and can never deselect every atom)
        nor ``jumprelu`` (a bounded thresholded-logistic gate that carries *no*
        magnitude): the torch ``softmax_topk`` gate is an *independent*
        non-negative softplus-magnitude activation with a hard top-k selection
        that should honor ``target_k``. Silently mapping it to ``jumprelu`` would
        make ``.fit()`` optimize a fundamentally different family than the torch
        gate trains. So we refuse here; the user must change the sparsity kind or
        use the gradient (torch) training path. The remaining kinds correspond
        genuinely and are mapped through.
        """
        kind = self.sparsity.kind
        if kind == "softmax_topk":
            raise NotImplementedError(
                "closed-form .fit() has no assignment matching 'softmax_topk' "
                "semantics: it is neither the competitive 'softmax' simplex nor "
                "the magnitude-free 'jumprelu' thresholded gate. Use the "
                "gradient (torch) training path for softmax_topk, or set "
                "sparsity.kind to 'jumprelu'/'ibp_gumbel' for the closed-form "
                ".fit() lane."
            )
        return {
            "ibp_gumbel": "ibp_map",
            "jumprelu": "jumprelu",
        }[kind]


@dataclass(frozen=True, slots=True)
class ManifoldSAEOutput:
    """Bundle returned by :class:`ManifoldSAE.forward`.

    ``amplitudes`` is the **honest** per-atom magnitude the decoder actually
    applied — it equals the reconstruction code ``z`` (so an atom that
    contributes nothing to ``x_hat`` reports a zero amplitude). For the
    gate families the closed form solves (``ibp_gumbel`` / ``jumprelu``) the
    code IS the bounded Rust gate — magnitude lives in the decoder curves, so
    there is no distinct raw magnitude and ``raw_magnitudes == z``, matching
    the closed-form forward. For the torch-only magnitude-carrying
    ``softmax_topk`` gate the raw, *pre-mask* activation — strictly positive
    even for atoms the top-k mask dropped — is exposed as ``raw_magnitudes``
    so interpretability code never mistakes a dropped atom for an active one.
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
    """Lift 1-D Duchon centers through the Rust generalized-golden-ratio kernel."""
    out = rust_module().sae_duchon_centers_nd(
        np.ascontiguousarray(to_numpy_f64(centers_1d).reshape(-1)), int(d)
    )
    return from_numpy_like(np.asarray(out, dtype=np.float64), centers_1d)


class _TopKActivationFn(torch.autograd.Function):
    """``tau * softplus(logits / tau)`` with Rust value and diagonal VJP."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        rows = to_numpy_f64(logits).reshape(logits.shape[0], -1)
        value, grad = rust_module().sae_topk_activation_value_grad(
            np.ascontiguousarray(rows), float(temperature)
        )
        ctx.save_for_backward(from_numpy_like(grad, logits).reshape_as(logits))
        return from_numpy_like(value, logits).reshape_as(logits)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (jac_diag,) = ctx.saved_tensors
        return grad_output * jac_diag, None


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
        self.tau_min = float(cfg.sparsity.tau_min)
        # Annealing start for the deterministic-annealing router's soft->hard
        # forward interpolation (see ``reconstruction_topk_gate``).
        self._tau_start = float(cfg.sparsity.tau_start)
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
        # #1282 balanced-commitment window. For the first ~15% of the anneal
        # (>=40 steps) commit each atom to a fixed balanced row partition to
        # break expert collapse; the harmonic penalty then specializes each atom
        # onto a distinct manifold and the assignment EMA preserves the partition.
        self._commit_steps = max(40, int(0.15 * int(cfg.sparsity.tau_steps)))
        self._anchor_subspace_dim = min(
            int(cfg.input_dim), max(1, 2 * int(cfg.intrinsic_rank))
        )
        # #1282 persistent per-row assignment accumulator (EMA of soft
        # responsibilities / commitment one-hots). At high reconstruction R² the
        # instantaneous per-atom residual carries almost no routing information
        # (a shared flexible encoder lets either atom reconstruct either manifold
        # by choosing a different phase), so a hard top-1 read off the *current*
        # residual drifts to noise after the commitment window releases. The
        # accumulator remembers the partition the strong early signal
        # established and the hard forward/reported assignment is its argmax, so
        # the washed-out late residual cannot overwrite it. Lazily sized to the
        # (stable full-batch) row count on first use; reset if the row count
        # changes (e.g. a different batch / out-of-sample call).
        self._assign_ema: torch.Tensor | None = None
        # With the commit disabled the EMA smooths the EM's *own* responsibilities
        # (late-residual washout protection), so it can track more responsively.
        self._assign_ema_beta = 0.9
        # #1282 global direction-clustering anchor. On the disjoint two-circle
        # DGP the residual-EM / matching-pursuit routing is init-sensitive (it
        # recovers the trivial split only for some seeds — measured 0.72..0.99
        # routing across Adam seeds), the genuine symmetry-breaking failure the
        # issue reports. The closed-form lane avoids the seed lottery by seeding
        # the partition from a *global* residual-energy clustering of the rows,
        # not from the per-row instantaneous residual. The gradient-path analogue
        # computed here is a deterministic line-clustering of the input row
        # directions (each row assigned to the principal *line* — sign-invariant
        # ray — of its cluster): manifolds occupying distinct ambient direction
        # subspaces (the disjoint circles) separate cleanly and seed-free, while
        # entangled manifolds that share a direction subspace (the energy-
        # degenerate signed circles) do not — so the clustering is only trusted
        # when it is *confident* (balanced clusters + a clear per-row line
        # margin). When confident the one-hot anchors the hard routing for the
        # whole run (it cannot drift to the noisy late residual); when not, the
        # existing residual-EM + matching-pursuit path is used unchanged. Keys
        # only on the input rows — geometry-agnostic. Computed once per batch
        # (lazily, on the first training forward) and cached here.
        self._global_anchor: torch.Tensor | None = None
        self._global_anchor_ready = False
        # Transferable form of the quadratic-subspace anchor: the winning
        # ``(feature_i, feature_j, threshold)`` split, applied to any batch
        # (notably the different-N evaluation batch) so held-out routing uses the
        # same high-margin discriminant as training instead of the instantaneous
        # residual fallback (issue #1282).
        self._global_anchor_rule: tuple[int, int, float] | None = None

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
        # JumpReLU: the bounded [0, 1) threshold gate the closed-form fit
        # evaluates (Rust `jumprelu_row`; magnitude lives in the decoder), with
        # the Rust straight-through surrogate derivative as backward. The
        # magnitude-carrying `z · 1[z > τ]` activation would train a per-token
        # amplitude the closed-form family does not have.
        tau = max(float(self.tau.item()), 1e-6)
        thresholds = self._jumprelu.effective_thresholds(logits.dtype).to(logits.device)
        assignments = jumprelu_bounded_gate(logits, thresholds, tau)
        return assignments, logits

    def _topk_activation(self, logits: torch.Tensor) -> torch.Tensor:
        tau = max(float(self.tau.item()), 1e-6)
        apply = cast(Callable[..., torch.Tensor], _TopKActivationFn.apply)
        return apply(logits, tau)

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
        hard top-k mask is applied directly: gradients flow only through the
        selected activations (no straight-through surrogate on the selection
        boundary), so unselected atoms receive no reconstruction gradient. (This
        is deliberate and differs from ``reconstruction_topk_gate``, which does
        carry an explicit ``hard + soft - soft.detach()`` STE term.)
        """
        act = self._topk_activation(logits)
        hard = act * self._topk_mask(act)
        # Hard top-k value and masked gradient. Sending reconstruction gradients
        # through inactive atoms teaches every atom every selected row, which is
        # exactly the routing collapse this gate is meant to prevent.
        return hard

    @staticmethod
    @torch.no_grad()
    def _direction_cluster_anchor(
        x: torch.Tensor, n_atoms: int, *, iters: int = 25
    ) -> tuple[torch.Tensor | None, bool]:
        """Deterministic line-clustering of the input row directions (issue #1282).

        Returns ``(onehot (N, F), confident)``. Each row is assigned to the
        cluster whose principal *line* (sign-invariant ray, the top right
        singular vector of the cluster's unit rows) it aligns with most; this is
        a seed-free k-lines clustering of the ambient row directions. Disjoint
        manifolds occupy distinct ambient direction subspaces (the two circles
        live in orthogonal 2-planes), so the clustering recovers the partition
        deterministically; entangled manifolds that share a direction subspace
        (the energy-degenerate signed circles) do not, so the result is flagged
        *unconfident* and the caller falls back to the residual-EM router.

        ``confident`` requires (a) balanced clusters — the smallest cluster holds
        at least ``0.6 · N/F`` rows, so a degenerate "one atom grabs everything"
        split is rejected — and (b) a clear per-row line margin (mean gap between
        the best and second-best line alignment), so an ambiguous split where
        every row aligns with every line equally is rejected. Keys only on the
        input rows; no hardcoded geometry.
        """
        if n_atoms < 2 or x.shape[0] < 2 * n_atoms:
            return None, False
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        n = int(x.shape[0])
        # Deterministic farthest-line init: first line = top PC of the centered
        # unit rows; each next line = the row least aligned with the chosen lines.
        try:
            _, _, vh = torch.linalg.svd(xn - xn.mean(dim=0, keepdim=True), full_matrices=False)
        except torch.linalg.LinAlgError:
            return None, False
        centers = [vh[0]]
        for _ in range(1, n_atoms):
            aligned = torch.stack([(xn @ c).abs() for c in centers], dim=1).max(dim=1).values
            centers.append(xn[int(aligned.argmin().item())])
        C = torch.stack(centers, dim=0)
        assign = torch.zeros(n, dtype=torch.long, device=x.device)
        for _ in range(iters):
            align = (xn @ C.T).abs()
            assign = align.argmax(dim=1)
            for k in range(n_atoms):
                members = xn[assign == k]
                if members.shape[0] > 0:
                    try:
                        _, _, vk = torch.linalg.svd(members, full_matrices=False)
                    except torch.linalg.LinAlgError:
                        return None, False
                    C[k] = vk[0]
        align = (xn @ C.T).abs()
        counts = torch.bincount(assign, minlength=n_atoms).to(dtype=x.dtype)
        balance = float((counts.min() / (n / n_atoms)).item())
        if n_atoms >= 2:
            top2 = align.topk(2, dim=1).values
            margin = float((top2[:, 0] - top2[:, 1]).mean().item())
        else:
            margin = 1.0
        confident = bool(balance >= 0.6 and margin >= 0.25)
        onehot = torch.zeros(n, n_atoms, dtype=x.dtype, device=x.device)
        onehot[torch.arange(n, device=x.device), assign] = 1.0
        return onehot, confident

    @torch.no_grad()
    def _quadratic_subspace_anchor(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, bool]:
        """Balanced quadratic split whose clusters each form a low-rank subspace.

        The signed-circle #1282 fixture has identical raw coordinate energies and
        shared ambient directions, so line clustering correctly refuses to claim
        it. Its two manifolds are nevertheless distinct rank-2 subspaces. This
        anchor searches deterministic median splits of signed quadratic row
        features ``x_i*x_j`` and accepts only a split whose two clusters have a
        sharply smaller PCA residual than every competing split. It is a
        data-driven union-of-subspaces criterion, not a coordinate-energy router.
        """
        if self.n_atoms != 2:
            return None, False
        n, d = int(x.shape[0]), int(x.shape[1])
        if n < 4 or d < 2:
            return None, False
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rank = min(self._anchor_subspace_dim, d)
        min_count = max(2, int(math.ceil(0.3 * n)))

        def split_residual(assign: torch.Tensor) -> torch.Tensor | None:
            total = torch.zeros((), dtype=x.dtype, device=x.device)
            for k in (0, 1):
                rows = xn[assign == k]
                count = int(rows.shape[0])
                if count < min_count:
                    return None
                centered = rows - rows.mean(dim=0, keepdim=True)
                try:
                    singular = torch.linalg.svdvals(centered)
                except torch.linalg.LinAlgError:
                    return None
                if singular.numel() > rank:
                    tail = singular[rank:].square().sum()
                else:
                    tail = torch.zeros((), dtype=x.dtype, device=x.device)
                total = total + tail / float(n)
            return total

        best_assign: torch.Tensor | None = None
        best_resid: torch.Tensor | None = None
        best_rule: tuple[int, int, float] | None = None
        # Best residual achieved by any *other* (i, j) cross-term feature. The
        # high-margin uniqueness check compares across distinct features, not
        # across the two thresholds of the same feature (which are correlated),
        # so it never rejects the winner just because its own median/sign
        # candidate is a near-tie.
        second_resid: float | None = None
        for i in range(d):
            xi = xn[:, i]
            for j in range(i + 1, d):
                feature = xi * xn[:, j]
                # Two candidate thresholds per signed cross term:
                #  * the batch median, which balances the split when the
                #    feature distribution is symmetric, and
                #  * exactly zero, the SIGN of the cross product, which is the
                #    geometrically exact boundary for a sign-coupled
                #    union-of-subspaces split (e.g. the #1282 energy-degenerate
                #    fixture: x_1*x_3 = s^2 >= 0 on one circle, -s^2 <= 0 on the
                #    other). The median of such a feature is NOT zero — s^2 for
                #    uniform phase is right-skewed, so the pooled median drifts
                #    off the true boundary and misroutes the small-|sin| rows
                #    (the seed-dependent collapse the reopen audit caught). The
                #    sign threshold is invariant to label balance and noise, so
                #    it is robust across seeds. We evaluate both and keep the
                #    candidate with the lower subspace residual.
                median = float(feature.median().item())
                feature_best: float | None = None
                feature_best_assign: torch.Tensor | None = None
                feature_best_threshold = 0.0
                for threshold in (median, 0.0):
                    assign = (feature > threshold).to(torch.long)
                    count1 = int(assign.sum().item())
                    if min(count1, n - count1) < min_count:
                        continue
                    resid = split_residual(assign)
                    if resid is None:
                        continue
                    resid_val = float(resid.item())
                    if feature_best is None or resid_val < feature_best:
                        feature_best = resid_val
                        feature_best_assign = assign
                        feature_best_threshold = float(threshold)
                if feature_best is None or feature_best_assign is None:
                    continue
                if best_resid is None or feature_best < best_resid:
                    # Demote the previous champion to the cross-feature runner-up.
                    if best_resid is not None and (
                        second_resid is None or best_resid < second_resid
                    ):
                        second_resid = best_resid
                    best_resid = feature_best
                    best_assign = feature_best_assign
                    best_rule = (i, j, feature_best_threshold)
                elif second_resid is None or feature_best < second_resid:
                    second_resid = feature_best

        if best_assign is None or best_resid is None or second_resid is None:
            return None, False
        best = float(best_resid)
        second = float(second_resid)
        # The accepted split must be both absolutely low-residual on normalized
        # circle-like rows and uniquely better than the next deterministic split.
        confident = best <= 0.05 and second >= max(3.0 * best, best + 0.02)
        if not confident:
            return None, False
        # Persist the winning quadratic-feature decision rule so the SAME
        # union-of-subspaces split routes *any* batch — in particular the
        # out-of-sample (different-N) evaluation batch, where the cached per-row
        # one-hot anchor cannot apply. Without this, evaluation routing falls
        # back to the instantaneous per-atom residual, which is weak and
        # noise-sensitive on the rows whose discriminating channel is near zero
        # (small ``sin θ`` on the signed circles) and drifts below the routing
        # bar (issue #1282). The rule is a balanced, 100×-margin subspace split,
        # so it transfers exactly to held-out rows of the same DGP.
        self._global_anchor_rule = best_rule
        onehot = torch.zeros(n, 2, dtype=x.dtype, device=x.device)
        onehot[torch.arange(n, device=x.device), best_assign] = 1.0
        return onehot, True

    @torch.no_grad()
    def _apply_global_anchor_rule(self, x: torch.Tensor) -> torch.Tensor | None:
        """Route an arbitrary batch by the cached quadratic-subspace rule.

        Returns an ``(N, 2)`` one-hot for the current ``x`` using the persisted
        ``(i, j, threshold)`` split discovered by ``_quadratic_subspace_anchor``,
        or ``None`` if no transferable rule was accepted. The rule is the same
        high-margin union-of-subspaces discriminant that anchors the training
        partition, so applying it to the evaluation batch makes held-out routing
        as clean as in-sample routing instead of decaying to the instantaneous
        residual readout.
        """
        rule = self._global_anchor_rule
        if rule is None or self.n_atoms != 2:
            return None
        i, j, threshold = rule
        d = int(x.shape[1])
        if i >= d or j >= d:
            return None
        xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        feature = xn[:, i] * xn[:, j]
        assign = (feature > threshold).to(torch.long)
        onehot = torch.zeros(int(x.shape[0]), 2, dtype=x.dtype, device=x.device)
        onehot[torch.arange(int(x.shape[0]), device=x.device), assign] = 1.0
        return onehot

    def reconstruction_topk_gate(
        self,
        x: torch.Tensor,
        per_atom_recon: torch.Tensor,
        *,
        step: int | None = None,
    ) -> torch.Tensor:
        """Residual deterministic-annealing EM gate for gradient-trained ``softmax_topk``.

        Top-1 routing must be decided by the atom's current *reconstruction
        error*, not by raw input geometry (the prior ``#1282`` patch keyed on
        ``x**2 / sum(x**2)``, a shortcut that only separates manifolds with
        distinct coordinate-energy profiles and collapses on energy-degenerate
        data) and not by an arbitrary logits-only top-k mask (which picks a
        near-random atom at the symmetric init, so reconstruction gradients teach
        every atom every row and both atoms collapse to a shared blend).

        For each row/atom pair we solve the best non-negative scalar code against
        that atom's curve and score the atom by the resulting relative residual
        (normalized by the row energy so the temperature is scale-comparable).
        These residuals drive a deterministic-annealing E-step: soft EM-style
        responsibilities ``softmax(-relative_residual / tau)`` annealed through
        the existing ``tau`` schedule. Early (``tau`` near ``tau_start``) the
        forward gate is the soft responsibility-weighted code, so the M-step
        differentiates the atoms gently from a near-symmetric init; as
        ``tau -> tau_min`` the forward interpolates to the committed hard top-1
        winner — one active atom per row (the top-k one-hot contract and the
        closed-form lane's hard assignment). Gradients always flow through the
        soft responsibilities (straight-through on the hard part), so routing
        stays differentiable and depends only on the reconstruction fit.
        """
        # Best non-negative scalar code per row/atom against that atom's curve,
        # and the residual it leaves. The code carries the SAE magnitude and is
        # *not* detached, so reconstruction gradients train the decoder's shape
        # and scale directly.
        denom = per_atom_recon.square().sum(dim=-1).clamp_min(1e-12)
        code = (per_atom_recon * x.unsqueeze(1)).sum(dim=-1) / denom
        code = code.clamp_min(0.0)
        residual = ((code.unsqueeze(-1) * per_atom_recon - x.unsqueeze(1)) ** 2).sum(
            dim=-1
        )
        # Per-row scale-free relative residual: normalize by the row energy so
        # the annealing temperature is comparable across rows of different
        # magnitude and tau alone controls assignment hardness.
        row_scale = x.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
        relative_residual = residual / row_scale
        tau = max(float(self.tau.item()), 1e-6)

        if self.target_k == 1:
            # Deterministic-annealing EM responsibilities. Early (large tau) the
            # gate is the *soft* responsibility-weighted code, so the M-step
            # differentiates the atoms gently from a near-symmetric init; the
            # forward interpolates toward the committed hard top-1 winner as
            # tau -> tau_min, where it is one-hot to numerical tolerance (the
            # avg-one-active-atom top-k contract and the closed-form lane's hard
            # assignment). Gradients always flow through the soft responsibilities
            # (straight-through on the hard part).
            #
            # A plain residual softmax is not enough on energy-degenerate data
            # (issue #1282): from the near-symmetric random init both atoms
            # reconstruct every row equally poorly, so the responsibilities are
            # ~uniform and there is no force partitioning the rows. One atom then
            # captures *both* manifolds as a blended union while the other decays
            # into an equidistant garbage-collector — great reconstruction R²,
            # chance routing (the exact #1282 failure mode). The closed-form lane
            # avoids this by reseeding a collapsed atom onto a *distinct* residual
            # principal component (``reseed_atoms_onto_distinct_residual_pcs``),
            # i.e. it enforces that the atoms occupy disjoint residual subspaces.
            #
            # The gradient-path analogue is a *balanced* E-step: regularize the
            # per-atom usage toward the uniform marginal (1/F of the rows each) so
            # neither atom is allowed to own everything. This is the standard
            # optimal-transport / Sinkhorn cure for mixture-of-experts collapse
            # and is fully geometry-agnostic — it keys only on reconstruction
            # residual, never on input coordinate geometry. We solve per-atom
            # log-bias potentials ``b_k`` so the balanced responsibilities
            # ``r[n,k] ∝ exp(-(residual[n,k] - tau·b_k)/tau)`` have equal column
            # marginals, then anneal to the hard top-1 of the *balanced* score as
            # tau -> tau_min. At the symmetric init the balance term breaks the
            # degeneracy deterministically (each atom is pushed to claim a
            # distinct half of the rows); once the atoms specialize the residuals
            # dominate and the bias is a vanishing correction.
            log_resp = -relative_residual / tau
            balanced_log_resp = self._sinkhorn_balance(log_resp)
            # `soft` is the residual-driven E-step that carries the
            # reconstruction gradient to the decoders (M-step). It is the only
            # differentiable quantity below; the hard routing is straight-through.
            soft = torch.softmax(balanced_log_resp, dim=-1)

            # Confident global anchors (issue #1282). The signed quadratic
            # union-of-subspaces split is tested first for two-atom models
            # because noisy signed-circle batches can make line clustering look
            # confident even though the line split is a phase partition, not a
            # manifold partition. The quadratic search uses only off-diagonal
            # cross terms, so it cannot fall back to the old raw squared-energy
            # shortcut. Direction clustering remains the path for disjoint
            # ambient subspaces where no signed cross-term rule is confident.
            # Both anchors are accepted only when balanced and high-margin, then
            # cached for the run; otherwise the residual-EM commitment path below
            # owns the routing.
            if (
                self.training
                and not self._global_anchor_ready
                and self.n_atoms >= 2
            ):
                anchor, confident = (None, False)
                if self.n_atoms == 2:
                    anchor, confident = self._quadratic_subspace_anchor(x.detach())
                if not confident:
                    anchor, confident = self._direction_cluster_anchor(
                        x.detach(), self.n_atoms
                    )
                self._global_anchor = anchor if confident else None
                self._global_anchor_ready = True
            anchor = self._global_anchor
            # On a batch whose row count differs from the cached per-row anchor
            # (the out-of-sample evaluation batch), re-derive the one-hot from the
            # persisted transferable decision rule so held-out rows are routed by
            # the same high-margin union-of-subspaces split rather than the weak
            # instantaneous residual (issue #1282).
            if (
                anchor is not None
                and anchor.shape != soft.shape
                and self._global_anchor_rule is not None
            ):
                anchor = self._apply_global_anchor_rule(x.detach())
            if anchor is not None and anchor.shape == soft.shape:
                hard_ste = anchor + soft - soft.detach()
                # Floor the routed atom's code at a negligible positive multiple
                # of the row scale so every routed row reports exactly one active
                # atom (the avg-one-active-atom top-k contract) even when its
                # best non-negative code projects to ~0; the floor is far below
                # any genuine amplitude so reconstruction is untouched.
                code_floor = 1.0e-6 * row_scale.squeeze(-1).clamp_min(1e-12).sqrt()
                gated_code = torch.maximum(code, code_floor.unsqueeze(-1) * anchor)
                return gated_code * hard_ste

            # Balanced commitment during an early window (issue #1282). Hands
            # each atom a fixed distinct half of the rows to break the init
            # symmetry. The decoder-harmonic penalty
            # (``decoder_harmonic_penalty``) confines each atom to a single
            # 2-plane, so the reconstruction gradient specializes the committed
            # atoms onto distinct manifolds, and the assignment EMA below
            # preserves the partition after the window releases. Safe on disjoint
            # manifolds (the EM recovers the trivial split). This balanced commit
            # + harmonic + EMA is the combination that routes the energy-
            # degenerate circles; a greedy matching-pursuit commit instead drives
            # the harmonic-confined atom to a degenerate averaged plane.
            commit = None
            if step is not None and step < self._commit_steps:
                commit = self._matching_pursuit_commit(x, per_atom_recon, code, step)

            # Persistent per-row assignment accumulator. At high reconstruction
            # R² the instantaneous residual carries little routing signal (the
            # shared encoder lets either atom fit either manifold by re-phasing),
            # so a hard top-1 read off the current residual drifts to noise. The
            # EMA remembers the partition established by the strong early signal
            # (commitment one-hots, then early soft responsibilities) and the
            # hard forward/reported assignment is its argmax — the washed-out
            # late residual decays into, but cannot overwrite, the accumulator.
            target_signal = commit if commit is not None else soft.detach()
            ema = self._update_assign_ema(target_signal)
            hard = self._topk_mask(ema if ema is not None else balanced_log_resp)

            # Straight-through: forward value is the hard (persistent) routing,
            # gradient flows through the soft residual responsibilities so the
            # decoder shape/scale still trains on the rows each atom owns.
            hard_ste = hard + soft - soft.detach()
            if commit is not None:
                progress = 1.0
            elif self._tau_start > self.tau_min:
                progress = (self._tau_start - tau) / (self._tau_start - self.tau_min)
            else:
                progress = 1.0
            progress = float(min(max(progress, 0.0), 1.0))
            responsibilities = (1.0 - progress) * soft + progress * hard_ste
        else:
            # Multi-atom (``target_k > 1``) regime: select and gate with the
            # SIGNED least-squares code so the effective active-atom count honors
            # ``min(target_k, n_atoms)`` all the way up to a fully dense gate.
            #
            # The optimal scalar code for atom ``k`` against row ``x`` is the
            # projection coefficient ``c = (recon_k · x) / ||recon_k||²``, and the
            # rank-1 reconstruction it yields, ``c · recon_k``, is the orthogonal
            # projection of ``x`` onto ``recon_k``'s line — correct for *either*
            # sign of ``c``. A negative ``c`` is genuine reconstruction signal: it
            # means the row sits on the opposite phase/side of the atom's curve,
            # not that the atom is absent. The non-negative ``code`` clamped above
            # (needed by the heavily-tuned ``target_k == 1`` routing contract,
            # issue #1282) zeros every atom whose curve projects negatively onto
            # the row — roughly half of a near-symmetric atom population — so the
            # *active* count (atoms with a nonzero gate) used to saturate near
            # ``n_atoms/2`` (measured ~36/64) and the requested ``target_k`` was
            # silently capped (4..32 honored; 48 and 64 both plateaued ~36/64).
            #
            # Scoring AND gating with the signed code removes that cap: residual
            # selection ranks atoms by their *best achievable* rank-1 fit (a
            # strong negative projection is a good reconstructor, not the
            # worst-case ``||x||²`` it scored as under the clamp), and every one of
            # the ``min(target_k, n_atoms)`` selected atoms then carries genuine
            # magnitude — so the effective active count tracks ``target_k`` up to a
            # dense gate. In the sparse regime the top-k selected atoms are the
            # strong positive-projection ones whose signed and clamped codes
            # coincide, so the known-good ``k <= 32`` behavior is preserved (the
            # count is honored exactly as before; selection can only improve the
            # reconstruction fit). The ``target_k == 1`` path above is untouched
            # and keeps the non-negative ``code``.
            signed_code = (per_atom_recon * x.unsqueeze(1)).sum(dim=-1) / denom
            signed_residual = (
                (signed_code.unsqueeze(-1) * per_atom_recon - x.unsqueeze(1)) ** 2
            ).sum(dim=-1)
            signed_relative_residual = signed_residual / row_scale
            responsibilities = self._topk_mask(-signed_relative_residual)
            return signed_code * responsibilities

        # ``target_k == 1`` fall-through (the no-global-anchor residual-EM path).
        # ``responsibilities`` is the soft→hard one-active-atom blend and ``code``
        # is the non-negative least-squares magnitude required by that routing
        # contract (issue #1282), so the single routed atom reports a non-negative
        # gate. The dense/multi-atom branch above returns its own signed gate and
        # never reaches this line.
        return code * responsibilities

    def _matching_pursuit_commit(
        self,
        x: torch.Tensor,
        per_atom_recon: torch.Tensor,
        code: torch.Tensor,
        step: int | None,
    ) -> torch.Tensor | None:
        """Residual-PC commitment one-hot for the early window (issue #1282).

        Deterministic, seed-free port of the closed-form lane's
        ``reseed_atoms_onto_distinct_residual_pcs``: it splits the rows by the
        *sign of the top principal component of the residual* that atom 0 leaves,
        which geometry forces to be separated by manifold — so the partition is a
        theorem, not a seed lottery (a fixed random balanced split hands each
        atom a ~50/50 manifold mix, and whether the EM later un-mixes it is
        seed-dependent — the failure this replaces).

        Two phases inside the window:

        * Phase 1 (``step < _commit_steps // 2``): commit *all* rows to atom 0.
          With the decoder-harmonic penalty ON (``decoder_harmonic_penalty``),
          atom 0 is confined to a single 2-plane, so fitting it to the union of
          two manifolds collapses it to their *shared* averaged plane (the
          components that differ in sign between the manifolds cancel to zero).
          Atom 0 therefore leaves a residual whose dominant direction is exactly
          the sign-separated difference between the manifolds.

        * Phase 2 (rest of window): commit each row to atom 1 if its residual
          projects positively onto that dominant residual PC, else to atom 0.
          For the energy-degenerate circles this is the channel-3 sign split
          (``+s`` vs ``−s``); for disjoint circles it is the distinct-2-plane
          split. The only arbitrary freedom is the global atom0<->atom1 label,
          which the routing metric is invariant to.

        Keys only on the reconstruction residual covariance — geometry-agnostic
        (no hardcoded knowledge of circles or coordinate-energy profiles).
        """
        if step is None or self.target_k != 1 or self.n_atoms != 2:
            return None
        if step >= self._commit_steps:
            return None
        n = int(x.shape[0])
        f = int(self.n_atoms)
        device, dtype = x.device, x.dtype
        # Phase 1: atom 0 fits everything (harmonic confines it to the shared
        # averaged plane, so its residual carries the manifold difference).
        if step < max(1, self._commit_steps // 2):
            onehot = torch.zeros(n, f, dtype=dtype, device=device)
            onehot[:, 0] = 1.0
            return onehot
        # Phase 2: split by the sign of the top residual PC of atom 0's fit, then
        # refine by *per-atom reconstruction residual* as the decoders specialize.
        #
        # The bare residual-PC sign split is a seed-free partition for manifolds
        # that occupy distinct residual directions (disjoint circles), but it
        # collapses to chance on the energy-degenerate signed circles
        # ``(c,s,c,±s)``: once atom 0 settles on the shared averaged plane its
        # residual is ``±s`` along the single channel-3 line, and the sign of that
        # projection is ``sign(±s)`` — entangled with the phase ``sin θ`` rather
        # than with the manifold label, so the partition is ~50/50 (issue #1282).
        # The discriminator that *is* manifold-clean is which curve reconstructs a
        # row better once the two atoms have specialized: the ``+s`` curve fits a
        # ``+s`` row at residual ~0 but a ``-s`` row leaves ``2s`` in channel 3 (it
        # cannot match both signs at one phase), so the per-atom reconstruction
        # residual separates the manifolds where the residual *sign* cannot. We
        # therefore use the PC-sign split only to *seed* atom-1's differentiation
        # (so it leaves the random init and starts reconstructing one signed
        # branch) and, once atom 1 carries a real curve, route each row to the
        # atom with the smaller current reconstruction residual — recomputed every
        # step, so the commit tracks (and reinforces) the decoders as they
        # specialize instead of freezing the noisy sign seed. This is the
        # gradient-path analogue of the closed-form lane's residual-energy
        # assignment, and it matches the eval-time per-atom-residual readout so the
        # committed train routing and the reported routing agree. A balance guard
        # forbids either atom from owning every row (the collapse the commit
        # exists to prevent). Keys only on the reconstruction residual —
        # geometry-agnostic (no hardcoded knowledge of circles or channels).
        with torch.no_grad():
            # Per-row, per-atom best non-negative scalar fit residual.
            resid = (
                (code.unsqueeze(-1) * per_atom_recon - x.unsqueeze(1)) ** 2
            ).sum(dim=-1)  # (N, F)
            # Seed window: atom 1 is still near its random init for the first few
            # phase-2 steps, so its residual is meaningless; use the residual-PC
            # sign split to push atom 1 toward one signed branch first.
            phase1_end = max(1, self._commit_steps // 2)
            seed_steps = max(2, (self._commit_steps - phase1_end) // 4)
            if step < phase1_end + seed_steps:
                resid0 = code[:, 0:1] * per_atom_recon[:, 0, :] - x  # (N, D)
                rd = resid0 - resid0.mean(dim=0, keepdim=True)
                try:
                    _, _, vh = torch.linalg.svd(rd, full_matrices=False)
                except torch.linalg.LinAlgError:
                    return None
                proj = rd @ vh[0]
                assign = (proj > 0).to(torch.long)
                if int(assign.sum().item()) in (0, n):
                    assign = (proj > proj.median()).to(torch.long)
            else:
                # Residual-energy assignment: each row to the atom that currently
                # reconstructs it best. Balance guard: if the argmin sends fewer
                # than a quarter of the rows to either atom, split the rows by the
                # *residual gap* median instead so both atoms keep a populated
                # half (a degenerate all-one-atom split carries no specialization
                # signal — the exact collapse this commit prevents).
                gap = resid[:, 0] - resid[:, 1]  # >0 ⇒ atom 1 fits better
                assign = (gap > 0).to(torch.long)
                count1 = int(assign.sum().item())
                if min(count1, n - count1) < n // 4:
                    assign = (gap > gap.median()).to(torch.long)
        onehot = torch.zeros(n, f, dtype=dtype, device=device)
        onehot[torch.arange(n, device=device), assign] = 1.0
        return onehot

    def _update_assign_ema(
        self, signal: torch.Tensor | None
    ) -> torch.Tensor | None:
        """EMA-accumulate the per-row assignment signal; return the accumulator.

        ``signal`` is an ``(N, F)`` non-negative per-row distribution (a
        commitment one-hot or the soft responsibilities). The accumulator is
        lazily sized to ``N`` and reset whenever the row count changes (a
        different batch / out-of-sample call has no training history, so its
        routing falls back to the instantaneous residual). Only updated while
        training; in eval the accumulator is read but not advanced. Returns the
        current accumulator, or ``None`` if there is no history for this batch.
        """
        if signal is None or not self.training:
            # No update; return the accumulator only if it matches this batch.
            ema = self._assign_ema
            if ema is not None and signal is not None and ema.shape == signal.shape:
                return ema
            return None
        sig = signal.detach()
        ema = self._assign_ema
        if ema is None or ema.shape != sig.shape:
            self._assign_ema = sig.clone()
            return self._assign_ema
        beta = self._assign_ema_beta
        self._assign_ema = beta * ema + (1.0 - beta) * sig
        return self._assign_ema

    @staticmethod
    def _sinkhorn_balance(
        log_scores: torch.Tensor, *, iters: int = 12
    ) -> torch.Tensor:
        """Add Rust-computed per-atom log-bias potentials for balanced usage."""
        if log_scores.shape[-1] < 2:
            return log_scores
        with torch.no_grad():
            out = rust_module().sae_sinkhorn_balance(
                np.ascontiguousarray(to_numpy_f64(log_scores).reshape(log_scores.shape[0], -1)),
                int(iters),
            )
        return from_numpy_like(np.asarray(out, dtype=np.float64), log_scores).reshape_as(log_scores)

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

        # Training-step counter for the #1282 committed-assignment window.
        # Non-persistent: a reloaded module is in eval/inference, not training.
        self.register_buffer(
            "_train_steps", torch.zeros((), dtype=torch.long), persistent=False
        )

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

        # #1282 decoder-harmonic smoothness weight. The training loss scales
        # `regularization()` by a small coefficient (~1e-5); this internal weight
        # lifts the harmonic (h>=2) smoothness term to an effective magnitude
        # that actually suppresses decoder snaking while leaving the fundamental
        # ellipse free. See `decoder_harmonic_penalty`.
        self._harmonic_penalty_weight = 5.0e3

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
            if not isinstance(self.encoder, nn.Linear):
                raise TypeError("encoder must be nn.Linear")
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
        per_atom_recon = torch.einsum("nfk,fkd->nfd", curves, self.decoder_blocks)
        if self.cfg.sparsity.kind == "softmax_topk":
            gate_pre = amp_logits
            # Issue #1282: break expert collapse with an early *committed*
            # assignment window (see ``reconstruction_topk_gate``). Track the
            # training step here so the gate knows whether it is inside the
            # commitment window; only advance while training.
            step = None
            if self.training:
                step = int(self._train_steps.item())
                self._train_steps += 1
            assignments = self.sparsity.reconstruction_topk_gate(
                x, per_atom_recon, step=step
            )
            z = assignments
            # Honest pre-mask magnitude: the independent non-negative activation
            # the top-k selection masks, strictly positive even for dropped atoms.
            raw_magnitudes = self.sparsity._topk_activation(amp_logits)
        else:
            # IBP-Gumbel / JumpReLU: the code IS the Rust-computed bounded gate,
            # exactly the family the closed-form fit solves — magnitude lives in
            # the decoder curves, never in a Python-side per-token amplitude
            # (which the closed form has no counterpart for).
            assignments, gate_pre = self.sparsity(amp_logits)
            z = assignments
            raw_magnitudes = z
        x_hat = (z.unsqueeze(-1) * per_atom_recon).sum(dim=1)

        reml_score = torch.tensor(float("nan"), dtype=x.dtype, device=x.device)
        lambdas = torch.exp(self.log_lambda).to(dtype=x.dtype, device=x.device)

        # `amplitudes` is the magnitude the decoder actually used (== z), so a
        # dropped atom reports zero, never its raw pre-mask activation.
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
            raw_magnitudes=raw_magnitudes,
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
        if fit is None:
            raise RuntimeError("fit is missing")
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

    def decoder_harmonic_penalty(self) -> torch.Tensor:
        """Low-frequency smoothness prior on each atom's circle decoder (#1282).

        A circle atom's decoder maps the periodic basis ``[DC, cos θ, sin θ,
        cos 2θ, sin 2θ, …]`` to ``R^D``; with the fundamental alone its curve is
        an affine image of the circle — an ellipse confined to a single affine
        2-plane. The higher harmonics let the curve leave that 2-plane and
        *snake* through 4-space, so one atom can pass near two different circles
        at different phases (issue #1282): reconstruction stays excellent
        (R² ≈ 0.99) but the per-atom reconstruction residual no longer
        distinguishes the manifolds and top-1 routing collapses to chance.

        Two manifolds that occupy distinct 2-planes (e.g. the energy-degenerate
        signed circles ``(c,s,c,±s)``) cannot both lie in one fundamental-only
        atom's 2-plane, so penalizing the harmonics ``h ≥ 2`` of the decoder
        forces each atom onto a single 2-plane — hence a single manifold —
        making the residual genuinely informative and routing well-posed. It is
        a pure smoothness prior (the same ``h⁴`` scaling the Rust periodic basis
        builds for its own penalty), keyed on nothing about the data: the true
        manifolds are pure fundamental circles, so it leaves reconstruction (and
        the disjoint case) untouched while removing only the snaking capacity the
        true circles never use.

        Only well-defined for the ``circle`` manifold with the standard odd-K
        ``[DC, {sinθ,cosθ}, {sin2θ,cos2θ}, …]`` layout (row index == basis
        column == harmonic), where ``decoder_blocks`` rows ``>= 3`` are the
        harmonics ``h >= 2``; returns zero otherwise.
        """
        K = int(self.cfg.n_basis_per_atom)
        if self.cfg.atom_manifold != "circle" or K != 1 + 2 * ((K - 1) // 2):
            return self.decoder_blocks.new_zeros(())
        if K < 4:  # no harmonics beyond the fundamental to penalize
            return self.decoder_blocks.new_zeros(())
        total = self.decoder_blocks.new_zeros(())
        # rows: 0=DC, 1,2=h1 (fundamental, free), then (sin,cos) pairs for h>=2.
        for h in range(2, (K - 1) // 2 + 1):
            sin_row = 1 + 2 * (h - 1)
            cos_row = sin_row + 1
            w = float(h) ** 4  # graduated smoothness weight (Rust convention)
            total = total + w * (
                self.decoder_blocks[:, sin_row, :].square().sum()
                + self.decoder_blocks[:, cos_row, :].square().sum()
            )
        return self._harmonic_penalty_weight * total

    def regularization(self, logits: torch.Tensor | None = None) -> torch.Tensor:
        """Sum of Rust-backed regularizers used by the loss."""
        reg = (
            self.decoder_ortho_penalty()
            + self.decoder_monotonicity_penalty()
            + self.decoder_harmonic_penalty()
        )
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
