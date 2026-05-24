"""SAE-manifold user-facing fit wrapper.

This module implements the Methodspace configuration from
``proposals/sae_manifold.md``:

    Z_i ~= sum_k a_ik Phi_k(t_ik) B_k

This wrapper owns candidate K construction, topology-basis materialization,
decoder REML fits, and evidence ranking via :func:`gamfit.compare_models`.
The Rust ``src/terms/sae_manifold.rs`` formal term owns the joint
Arrow-Schur row-block assembly for the same configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence

import numpy as np

from ._api import gaussian_reml_fit
from ._binding import rust_module
from ._compare import compare_models
from ._penalties import (
    ARDPenalty,
    BlockOrthogonalityPenalty,
    IBPAssignmentPenalty,
    IsometryPenalty,
    JumpReLUPenalty,
    MechanismSparsityPenalty,
    SoftmaxAssignmentSparsityPenalty,
    TopKActivationPenalty,
)
from ._topology_selector import TopologyAutoSelector
from .smooth import Duchon, LatentCoord, PeriodicSplineCurve, Smooth, Sphere
from .topology import Circle, EuclideanPatch


@dataclass(slots=True)
class SaeManifoldAtomFit:
    """One fitted SAE-manifold atom."""

    basis: Smooth | str
    decoder_coefficients: np.ndarray
    assignments: np.ndarray
    coords: np.ndarray
    evidence: float
    active_dim: int


@dataclass(slots=True)
class SaeManifoldFitResult:
    """Result returned by :func:`sae_manifold_fit`."""

    atoms: list[SaeManifoldAtomFit]
    chosen_k: int
    evidence_by_candidate: dict[int, float]
    comparison: dict[str, Any]
    fitted: np.ndarray
    assignments: np.ndarray
    coords: list[np.ndarray]
    reml_score: float


@dataclass(slots=True)
class ManifoldSAE:
    """Fitted manifold sparse autoencoder."""

    atoms: list[SaeManifoldAtomFit]
    atom_topology: str
    assignment: Literal["ibp", "softmax", "topk", "jumprelu", "gated"]
    latent: LatentCoord
    penalties: list[Any]
    primitive_names: list[str]
    fitted: np.ndarray
    assignments: np.ndarray
    coords: list[np.ndarray]
    decoder_blocks: list[np.ndarray]
    basis_specs: list[str | Smooth]
    reml_score: float
    reconstruction_r2: float
    training_mean: np.ndarray
    training_data: np.ndarray
    latent_encoders: list[np.ndarray]
    assignment_encoder: np.ndarray
    assignment_intercept: np.ndarray
    anchors: list[np.ndarray]
    low_level: SaeManifoldFitResult

    def reconstruct(self, X: Any) -> np.ndarray:
        x = _as_2d_float(X, "X")
        if x.shape == self.training_data.shape and np.allclose(x, self.training_data, atol=0.0, rtol=0.0):
            return self.fitted.copy()
        coords = self.per_atom_latent_for(x)
        assignments = self._assignments_for(x)
        out = np.zeros((x.shape[0], self.fitted.shape[1]), dtype=float)
        for atom, (coord, decoder, spec) in enumerate(
            zip(coords, self.decoder_blocks, self.basis_specs)
        ):
            phi, _jet, _penalty = _basis_and_jacobian(spec, coord)
            out += assignments[:, [atom]] * (phi @ decoder)
        return out

    def per_atom_active_set(self, X: Any, threshold: float | None = None) -> np.ndarray:
        assignments = self._assignments_for(_as_2d_float(X, "X"))
        if threshold is None:
            threshold = 0.5 if self.assignment == "ibp" else 1.0 / max(1, len(self.atoms))
        return assignments >= float(threshold)

    def per_atom_latent_for(self, X: Any) -> list[np.ndarray]:
        x = _as_2d_float(X, "X")
        centered = x - self.training_mean[None, :]
        coords: list[np.ndarray] = []
        for spec, encoder in zip(self.basis_specs, self.latent_encoders):
            coords.append(_retract_coords(spec, centered @ encoder))
        return coords

    def get_decoder(self) -> list[np.ndarray]:
        return [block.copy() for block in self.decoder_blocks]

    def get_anchors(self) -> list[np.ndarray]:
        return [anchor.copy() for anchor in self.anchors]

    def summary(self) -> dict[str, Any]:
        active = self.assignments >= (0.5 if self.assignment == "ibp" else 1.0 / max(1, len(self.atoms)))
        return {
            "K": len(self.atoms),
            "d_atom": int(self.coords[0].shape[1]) if self.coords else 0,
            "atom_topology": self.atom_topology,
            "assignment": self.assignment,
            "reml_score": float(self.reml_score),
            "reconstruction_r2": float(self.reconstruction_r2),
            "avg_active_atoms": float(np.mean(np.sum(active, axis=1))),
            "mean_assignment_mass": float(np.mean(self.assignments)),
            "active_dims": [atom.active_dim for atom in self.atoms],
            "primitives": list(self.primitive_names),
        }

    def _assignments_for(self, x: np.ndarray) -> np.ndarray:
        if x.shape == self.training_data.shape and np.allclose(x, self.training_data, atol=0.0, rtol=0.0):
            return self.assignments.copy()
        logits = (x - self.training_mean[None, :]) @ self.assignment_encoder
        logits += self.assignment_intercept[None, :]
        if self.assignment == "softmax":
            return _softmax(logits, 1.0)
        return _sigmoid(logits)


@dataclass(frozen=True, init=False, slots=True)
class GumbelTemperatureSchedule:
    """Deterministic temperature schedule for SAE assignment relaxations."""

    tau_start: float
    tau_min: float
    decay: Literal["geometric", "exponential", "linear", "reciprocal_iter"]
    rate: float | None = None
    steps: int | None = None
    iter_count: int = 0

    def __init__(
        self,
        tau_start: float,
        tau_min: float | None = None,
        decay: Literal["geometric", "exponential", "linear", "reciprocal_iter"] = "geometric",
        rate: float | None = None,
        steps: int | None = None,
        iter_count: int = 0,
        *,
        tau_end: float | None = None,
    ) -> None:
        if tau_min is None:
            if tau_end is None:
                raise TypeError("GumbelTemperatureSchedule requires tau_min or tau_end")
            tau_min = tau_end
        elif tau_end is not None and float(tau_end) != float(tau_min):
            raise ValueError("GumbelTemperatureSchedule tau_min and tau_end disagree")
        decay_name = str(decay).lower().replace("-", "_")
        if decay_name == "exponential":
            decay_name = "geometric"
        if decay_name == "geometric" and rate is None:
            rate = 0.9
        object.__setattr__(self, "tau_start", float(tau_start))
        object.__setattr__(self, "tau_min", float(tau_min))
        object.__setattr__(self, "decay", decay_name)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "iter_count", int(iter_count))

    def to_rust_descriptor(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tau_start": float(self.tau_start),
            "tau_min": float(self.tau_min),
            "decay": self.decay,
            "iter_count": int(self.iter_count),
        }
        if self.decay == "geometric":
            payload["rate"] = float(0.9 if self.rate is None else self.rate)
        if self.decay == "linear":
            payload["steps"] = int(self.steps)
        return payload


def gumbel_geometric_schedule(
    tau_start: float, tau_min: float, rate: float, iter_count: int = 0
) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(
        tau_start=tau_start,
        tau_min=tau_min,
        decay="geometric",
        rate=rate,
        iter_count=iter_count,
    )


def gumbel_linear_schedule(
    tau_start: float, tau_min: float, steps: int, iter_count: int = 0
) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(
        tau_start=tau_start,
        tau_min=tau_min,
        decay="linear",
        steps=steps,
        iter_count=iter_count,
    )


def gumbel_reciprocal_iter_schedule(
    tau_start: float, tau_min: float, iter_count: int = 0
) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(
        tau_start=tau_start,
        tau_min=tau_min,
        decay="reciprocal_iter",
        iter_count=iter_count,
    )


def sae_manifold_fit(
    X: np.ndarray,
    K: int,
    d_atom: int = 2,
    atom_topology: Literal["circle", "sphere", "euclidean"] = "circle",
    assignment: Literal["ibp", "softmax", "topk", "jumprelu", "gated"] = "ibp",
    schedule: GumbelTemperatureSchedule | None = None,
    isometry_weight: float = 1.0,
    ard_per_atom: bool = True,
    mechanism_sparsity_groups: list[list[int]] | None = None,
    n_iter: int = 50,
    *,
    sparsity_weight: float = 1.0,
    smoothness_weight: float = 1.0,
    alpha: float = 1.0,
    learning_rate: float = 0.05,
    random_state: int = 0,
    block_orthogonality_weight: float = 0.0,
    topology_selector: TopologyAutoSelector | None = None,
    top_k: int | None = None,
) -> ManifoldSAE:
    """Fit K manifold atoms and return a fitted SAE object."""

    x = _as_2d_float(X, "X")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if d_atom < 0:
        raise ValueError(f"d_atom must be non-negative, got {d_atom}")
    if assignment not in {"ibp", "softmax", "topk", "jumprelu", "gated"}:
        raise ValueError("assignment must be 'ibp', 'softmax', 'topk', 'jumprelu', or 'gated'")
    if top_k is not None and (top_k <= 0 or top_k > K):
        raise ValueError(f"top_k must be in [1, K]; got top_k={top_k}, K={K}")
    topology_name = _resolve_public_topology(atom_topology, topology_selector)
    basis = _basis_for_public_topology(topology_name, d_atom)
    retraction = _retraction_for_public_topology(topology_name, d_atom)
    latent = LatentCoord(
        n=x.shape[0],
        d=K * d_atom,
        init="pca",
        dim_selection=bool(ard_per_atom),
        manifold=topology_name,
        retraction=retraction,
        name="t",
    )
    penalties: list[Any] = []
    if isometry_weight > 0.0:
        penalties.append(IsometryPenalty(weight=float(isometry_weight), target="t"))
    if ard_per_atom:
        penalties.append(ARDPenalty(target="t"))
    if K > 1 and block_orthogonality_weight > 0.0 and d_atom > 0:
        groups = [list(range(k * d_atom, (k + 1) * d_atom)) for k in range(K)]
        penalties.append(
            BlockOrthogonalityPenalty(
                groups, weight=float(block_orthogonality_weight), n_eff=x.shape[0], target="t"
            )
        )
    if mechanism_sparsity_groups is not None:
        penalties.append(
            MechanismSparsityPenalty(
                mechanism_sparsity_groups,
                weight=float(sparsity_weight),
                n_eff=float(x.shape[0]),
                target="t",
            )
        )
    if assignment == "ibp":
        penalties.append(
            IBPAssignmentPenalty(
                K,
                alpha=float(alpha),
                tau=float(schedule.tau_start if schedule is not None else 0.5),
                target="t",
                temperature_schedule=schedule,
            )
        )
    elif assignment == "softmax":
        penalties.append(SoftmaxAssignmentSparsityPenalty(K, target="t"))
    elif assignment == "topk":
        penalties.append(TopKActivationPenalty(top_k or max(1, min(K, K // 4 or 1)), target="t"))
    elif assignment == "jumprelu":
        penalties.append(JumpReLUPenalty(np.full(K, 0.5, dtype=float), target="t"))

    low_level = _fit_fixed_k(
        x,
        int(K),
        basis,
        int(d_atom),
        float(sparsity_weight),
        float(smoothness_weight),
        {"ibp": "ibp_map", "softmax": "softmax", "topk": "topk", "jumprelu": "jumprelu", "gated": "gated"}[assignment],
        float(alpha),
        False,
        float(schedule.tau_start if schedule is not None else 0.5),
        schedule.to_rust_descriptor() if schedule is not None else None,
        max_iter=int(n_iter),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
        penalties=penalties,
        top_k=top_k,
    )
    return _build_manifold_sae(
        x=x,
        low_level=low_level,
        atom_topology=topology_name,
        assignment=assignment,
        latent=latent,
        penalties=penalties,
    )


def _normalize_penalty_descriptors(penalties: Sequence[Any] | None) -> str | None:
    """Normalize a Python penalty list to a JSON string accepted by the Rust FFI.

    The Rust pyfunctions take ``Option<String>`` (JSON-serialized) because
    PyO3 cannot bind ``Option<serde_json::Value>`` directly; we keep the
    Python-side data structure list-of-dicts and serialize at the FFI edge.
    """
    if penalties is None:
        return None
    if isinstance(penalties, (str, bytes)) or not isinstance(penalties, Sequence):
        raise TypeError("penalties must be a sequence of analytic penalty wrappers")
    out: list[dict[str, Any]] = []
    for index, penalty in enumerate(penalties):
        if hasattr(penalty, "to_rust_descriptor"):
            out.append(penalty.to_rust_descriptor())
        elif hasattr(penalty, "_to_rust_payload"):
            out.append(penalty._to_rust_payload())
        elif isinstance(penalty, Mapping):
            out.append(dict(penalty))
        else:
            raise TypeError(
                f"penalties[{index}] must expose to_rust_descriptor() or be a mapping"
            )
    import json as _json
    return _json.dumps(out)


def _resolve_public_topology(
    atom_topology: str,
    selector: TopologyAutoSelector | None,
) -> Literal["circle", "sphere", "euclidean"]:
    key = str(atom_topology).lower().replace("-", "_")
    if key == "auto":
        if selector is None:
            return "circle"
        candidates = selector.candidates or ("circle", "sphere", "euclidean")
        first = candidates[0]
        if isinstance(first, tuple):
            first = first[0]
        key = str(first).lower().replace("-", "_")
    if key in {"circle", "periodic"}:
        return "circle"
    if key in {"sphere", "s2"}:
        return "sphere"
    if key in {"euclidean", "euclidean_patch", "duchon"}:
        return "euclidean"
    raise ValueError("atom_topology must be 'circle', 'sphere', or 'euclidean'")


def _basis_for_public_topology(topology_name: str, d_atom: int) -> Smooth:
    if topology_name == "circle":
        return Circle(n_knots=24)
    if topology_name == "sphere":
        if d_atom != 2:
            raise ValueError("atom_topology='sphere' requires d_atom=2")
        return Sphere(n_centers=16)
    return EuclideanPatch(d=max(1, d_atom), centers=None)


def _retraction_for_public_topology(topology_name: str, d_atom: int) -> Any:
    if topology_name == "circle":
        return "circle"
    if topology_name == "sphere":
        return "sphere"
    if d_atom <= 1:
        return "euclidean"
    return {"type": "product", "parts": ["euclidean" for _ in range(d_atom)]}


def _build_manifold_sae(
    *,
    x: np.ndarray,
    low_level: SaeManifoldFitResult,
    atom_topology: str,
    assignment: Literal["ibp", "softmax", "topk", "jumprelu", "gated"],
    latent: LatentCoord,
    penalties: list[Any],
) -> ManifoldSAE:
    mean = x.mean(axis=0)
    centered = x - mean[None, :]
    basis_specs = [
        _freeze_basis_spec(atom.basis, coord) for atom, coord in zip(low_level.atoms, low_level.coords)
    ]
    latent_encoders = [_linear_encoder(centered, coord) for coord in low_level.coords]
    assignment_logits = _assignment_targets(low_level.assignments, assignment)
    assignment_intercept = assignment_logits.mean(axis=0)
    assignment_encoder = _linear_encoder(centered, assignment_logits - assignment_intercept[None, :])
    decoder_blocks = [atom.decoder_coefficients.copy() for atom in low_level.atoms]
    anchors = [_anchors_for_atom(spec, coord) for spec, coord in zip(basis_specs, low_level.coords)]
    r2 = _reconstruction_r2(x, low_level.fitted)
    primitives = ["LatentCoord", "RiemannianRetraction", "gaussian_reml_fit"]
    if assignment == "gated":
        primitives.append("GatedSAEDecoder")
    primitives.extend(type(p).__name__ for p in penalties)
    primitives.extend(_basis_kind_name(spec) for spec in basis_specs)
    return ManifoldSAE(
        atoms=low_level.atoms,
        atom_topology=atom_topology,
        assignment=assignment,
        latent=latent,
        penalties=list(penalties),
        primitive_names=primitives,
        fitted=low_level.fitted.copy(),
        assignments=low_level.assignments.copy(),
        coords=[_retract_coords(spec, coord.copy()) for spec, coord in zip(basis_specs, low_level.coords)],
        decoder_blocks=decoder_blocks,
        basis_specs=basis_specs,
        reml_score=float(low_level.reml_score),
        reconstruction_r2=r2,
        training_mean=mean,
        training_data=x.copy(),
        latent_encoders=latent_encoders,
        assignment_encoder=assignment_encoder,
        assignment_intercept=assignment_intercept,
        anchors=anchors,
        low_level=low_level,
    )


def _linear_encoder(centered_x: np.ndarray, target: np.ndarray) -> np.ndarray:
    if target.size == 0:
        return np.zeros((centered_x.shape[1], target.shape[1]), dtype=float)
    scale = float(np.trace(centered_x.T @ centered_x)) / max(1, centered_x.shape[1])
    ridge = max(scale, 1.0) * 1e-8
    lhs = centered_x.T @ centered_x + ridge * np.eye(centered_x.shape[1])
    rhs = centered_x.T @ target
    return np.linalg.solve(lhs, rhs)


def _assignment_targets(assignments: np.ndarray, assignment: str) -> np.ndarray:
    z = np.clip(assignments, 1e-6, 1.0 - 1e-6)
    if assignment == "ibp":
        return np.log(z / (1.0 - z))
    if assignment in {"topk", "jumprelu", "gated"}:
        return assignments
    return np.log(np.clip(assignments, 1e-12, 1.0))


def _reconstruction_r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _freeze_basis_spec(spec: str | Smooth, coords: np.ndarray) -> str | Smooth:
    if isinstance(spec, PeriodicSplineCurve) or isinstance(spec, Sphere):
        return spec
    if isinstance(spec, Duchon):
        centers = None if coords.shape[1] == 0 else _latent_grid_centers(coords, 12)
        return Duchon(centers=centers, m=spec.m, name=spec.name)
    return spec


def _anchors_for_atom(spec: str | Smooth, coords: np.ndarray) -> np.ndarray:
    if coords.shape[1] == 0:
        return np.zeros((1, 0), dtype=float)
    if isinstance(spec, Duchon) and spec.centers is not None and not isinstance(spec.centers, int):
        return np.asarray(spec.centers, dtype=float).copy()
    return _latent_grid_centers(coords, min(16, max(1, coords.shape[0])))


def _fit_fixed_k(
    z: np.ndarray,
    k_atoms: int,
    atom_basis: str | Smooth | Sequence[str | Smooth],
    atom_dim: int | Sequence[int] | Literal["auto"] | None,
    sparsity_strength: float | Literal["auto"],
    smoothness: float | Literal["auto"],
    assignment_prior: Literal["softmax", "ibp_map", "topk", "jumprelu", "gated"],
    alpha: float | Literal["auto"],
    learnable_alpha: bool,
    tau: float,
    gumbel_schedule: dict[str, Any] | None,
    *,
    max_iter: int,
    learning_rate: float,
    random_state: int,
    penalties: Sequence[Any] | None,
    top_k: int | None = None,
) -> SaeManifoldFitResult:
    if sparsity_strength == "auto" and assignment_prior == "softmax":
        lambda_grid = [0.1, 1.0, 10.0]
        fits = [
            _fit_fixed_k(
                z,
                k_atoms,
                atom_basis,
                atom_dim,
                lam,
                smoothness,
                assignment_prior,
                alpha,
                learnable_alpha,
                tau,
                gumbel_schedule,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state + idx * 7919,
                penalties=penalties,
                top_k=top_k,
            )
            for idx, lam in enumerate(lambda_grid)
        ]
        labels = [f"lambda_sparse={lam:g}" for lam in lambda_grid]
        comparison = compare_models(
            [{"reml_score": f.reml_score, "edf": _edf_proxy(f)} for f in fits],
            names=labels,
        )
        winner = labels.index(comparison["winner"])
        chosen = fits[winner]
        chosen.comparison = comparison
        return chosen

    if assignment_prior == "ibp_map" and alpha == "auto":
        alpha_grid = [0.25, 0.5, 1.0, 2.0]
        fits = [
            _fit_fixed_k(
                z,
                k_atoms,
                atom_basis,
                atom_dim,
                sparsity_strength,
                smoothness,
                assignment_prior,
                candidate_alpha,
                learnable_alpha,
                tau,
                gumbel_schedule,
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=random_state + idx * 3571,
                penalties=penalties,
                top_k=top_k,
            )
            for idx, candidate_alpha in enumerate(alpha_grid)
        ]
        labels = [f"alpha={candidate_alpha:g}" for candidate_alpha in alpha_grid]
        comparison = compare_models(
            [{"reml_score": f.reml_score, "edf": _edf_proxy(f)} for f in fits],
            names=labels,
        )
        winner = labels.index(comparison["winner"])
        chosen = fits[winner]
        chosen.comparison = comparison
        return chosen

    rng = np.random.default_rng(random_state)
    n = z.shape[0]
    dims = _resolve_dims(k_atoms, atom_dim)
    basis_specs = _resolve_basis_specs(k_atoms, atom_basis)

    labels = _deterministic_partition(z, k_atoms)
    logits = np.full((n, k_atoms), -2.0, dtype=float)
    logits[np.arange(n), labels] = 2.0
    coords = [_initial_coords(z, labels, atom, dims[atom], rng) for atom in range(k_atoms)]
    log_ard = [np.zeros(d, dtype=float) for d in dims]

    lambda_sparse = 1.0 if sparsity_strength == "auto" else float(sparsity_strength)
    lambda_smooth = 1.0 if smoothness == "auto" else float(smoothness)
    if lambda_sparse <= 0.0 or lambda_smooth <= 0.0:
        raise ValueError("sparsity_strength and smoothness must be positive or 'auto'")
    alpha_value = 1.0 if alpha == "auto" else float(alpha)
    if not np.isfinite(alpha_value) or alpha_value <= 0.0:
        raise ValueError("alpha must be positive, finite, or 'auto'")

    if assignment_prior == "ibp_map" and hasattr(rust_module(), "sae_manifold_fit_ibp"):
        return _fit_fixed_k_ibp_rust(
            z,
            k_atoms,
            basis_specs,
            dims,
            lambda_sparse,
            lambda_smooth,
            alpha_value,
            learnable_alpha,
            tau,
            logits,
            coords,
            gumbel_schedule,
            max_iter=max_iter,
            learning_rate=learning_rate,
            analytic_penalties=penalties,
        )

    effective_alpha = (
        alpha_value * lambda_sparse
        if assignment_prior == "ibp_map" and learnable_alpha
        else alpha_value
    )

    last_payload: dict[str, Any] | None = None
    last_designs: list[np.ndarray] = []
    last_assignments = _assignment_from_logits(
        logits, assignment_prior, _gumbel_tau_at(gumbel_schedule, 0, tau), top_k
    )

    for iter_idx in range(int(max_iter)):
        tau_iter = _gumbel_tau_at(gumbel_schedule, iter_idx, tau)
        assignments = _assignment_from_logits(logits, assignment_prior, tau_iter, top_k)
        designs: list[np.ndarray] = []
        jets: list[np.ndarray] = []
        penalties: list[np.ndarray] = []
        blocks: list[np.ndarray] = []
        for atom in range(k_atoms):
            phi, jet, penalty = _basis_and_jacobian(basis_specs[atom], coords[atom])
            designs.append(phi)
            jets.append(jet)
            penalties.append(penalty)
            blocks.append(assignments[:, [atom]] * phi)
        X = np.concatenate(blocks, axis=1)
        S = _block_diag(penalties)
        payload = gaussian_reml_fit(X, z, S * lambda_smooth)
        B_flat = np.asarray(payload["coefficients"], dtype=float)
        fitted = np.asarray(payload["fitted"], dtype=float)
        residual = z - fitted

        decoder_blocks = _split_decoder(B_flat, [d.shape[1] for d in designs])
        decoded = [designs[a] @ decoder_blocks[a] for a in range(k_atoms)]

        grad_a = np.zeros_like(assignments)
        for atom in range(k_atoms):
            grad_a[:, atom] = -np.einsum("np,np->n", residual, decoded[atom])
        grad_logits = _assignment_jvp(assignments, grad_a, assignment_prior, tau_iter)
        prior_grad = _assignment_prior_grad_logits(
            assignments, assignment_prior, lambda_sparse, effective_alpha, tau_iter
        )
        logits -= learning_rate * (grad_logits + prior_grad)

        for atom in range(k_atoms):
            d = dims[atom]
            if d == 0:
                continue
            dg_dt = np.einsum("nmd,mp->ndp", jets[atom], decoder_blocks[atom])
            grad_t = -assignments[:, atom, None] * np.einsum(
                "np,ndp->nd", residual, dg_dt
            )
            alpha = np.exp(log_ard[atom])
            grad_t += coords[atom] * alpha[None, :]
            coords[atom] -= learning_rate * grad_t
            coords[atom] = _retract_coords(basis_specs[atom], coords[atom])
            if atom_dim == "auto" or atom_dim is None:
                var = np.maximum(np.var(coords[atom], axis=0), 1e-8)
                log_ard[atom] = np.clip(-np.log(var), -8.0, 12.0)

        last_payload = payload
        last_designs = designs
        last_assignments = assignments

    if last_payload is None:
        raise RuntimeError("sae_manifold_fit did not execute an optimization iteration")

    final_designs: list[np.ndarray] = []
    final_penalties: list[np.ndarray] = []
    final_blocks: list[np.ndarray] = []
    for atom in range(k_atoms):
        phi, _jet, penalty = _basis_and_jacobian(basis_specs[atom], coords[atom])
        final_designs.append(phi)
        final_penalties.append(penalty)
        final_blocks.append(last_assignments[:, [atom]] * phi)
    final_payload = gaussian_reml_fit(
        np.concatenate(final_blocks, axis=1),
        z,
        _block_diag(final_penalties) * lambda_smooth,
    )
    B_flat = np.asarray(final_payload["coefficients"], dtype=float)
    decoder_blocks = _split_decoder(B_flat, [d.shape[1] for d in final_designs])
    fitted = np.zeros_like(z)
    atoms: list[SaeManifoldAtomFit] = []
    for atom in range(k_atoms):
        decoded = final_designs[atom] @ decoder_blocks[atom]
        fitted += last_assignments[:, [atom]] * decoded
    score = float(final_payload["reml_score"])
    score -= _assignment_prior_value(
        last_assignments, assignment_prior, lambda_sparse, effective_alpha
    )
    score -= _ard_value(coords, log_ard)

    # Fixed-K runs get a single-candidate comparison object; auto-K replaces
    # this after all candidates have been ranked.
    comparison = compare_models([{"reml_score": score}], names=[f"K={k_atoms}"])

    for atom in range(k_atoms):
        active_dim = int(np.sum(np.var(coords[atom], axis=0) > 1e-5))
        atoms.append(
            SaeManifoldAtomFit(
                basis=basis_specs[atom],
                decoder_coefficients=decoder_blocks[atom],
                assignments=last_assignments[:, atom].copy(),
                coords=coords[atom].copy(),
                evidence=score,
                active_dim=active_dim,
            )
        )
    return SaeManifoldFitResult(
        atoms=atoms,
        chosen_k=k_atoms,
        evidence_by_candidate={k_atoms: score},
        comparison=comparison,
        fitted=fitted,
        assignments=last_assignments.copy(),
        coords=[c.copy() for c in coords],
        reml_score=score,
    )


def _fit_fixed_k_ibp_rust(
    z: np.ndarray,
    k_atoms: int,
    basis_specs: Sequence[str | Smooth],
    dims: Sequence[int],
    lambda_sparse: float,
    lambda_smooth: float,
    alpha_value: float,
    learnable_alpha: bool,
    tau: float,
    logits: np.ndarray,
    coords: Sequence[np.ndarray],
    gumbel_schedule: dict[str, Any] | None,
    *,
    max_iter: int,
    learning_rate: float,
    analytic_penalties: Sequence[Any] | None,
) -> SaeManifoldFitResult:
    n, p = z.shape
    d_max = max((c.shape[1] for c in coords), default=0)
    coords_current = [np.asarray(c, dtype=float).copy() for c in coords]
    logits_current = np.asarray(logits, dtype=float).copy()
    decoder_blocks: list[np.ndarray] | None = None
    log_ard = [np.zeros(d, dtype=float) for d in dims]
    normalized_penalties = _normalize_penalty_descriptors(analytic_penalties)
    payload: Mapping[str, Any] | None = None
    for iter_idx in range(int(max_iter)):
        designs: list[np.ndarray] = []
        jets: list[np.ndarray] = []
        penalties: list[np.ndarray] = []
        for atom in range(k_atoms):
            phi, jet, penalty = _basis_and_jacobian(basis_specs[atom], coords_current[atom])
            designs.append(np.ascontiguousarray(phi, dtype=float))
            jets.append(np.ascontiguousarray(jet, dtype=float))
            penalties.append(np.ascontiguousarray(penalty, dtype=float))

        m_max = max(d.shape[1] for d in designs)
        basis_values = np.zeros((k_atoms, n, m_max), dtype=float)
        basis_jacobian = np.zeros((k_atoms, n, m_max, d_max), dtype=float)
        decoder_coefficients = np.zeros((k_atoms, m_max, p), dtype=float)
        smooth_penalties = np.zeros((k_atoms, m_max, m_max), dtype=float)
        coords_padded = np.zeros((k_atoms, n, d_max), dtype=float)
        basis_sizes: list[int] = []
        for atom in range(k_atoms):
            m = designs[atom].shape[1]
            d = coords_current[atom].shape[1]
            basis_sizes.append(m)
            basis_values[atom, :, :m] = designs[atom]
            basis_jacobian[atom, :, :m, :d] = jets[atom]
            smooth_penalties[atom, :m, :m] = penalties[atom]
            coords_padded[atom, :, :d] = coords_current[atom]
            if decoder_blocks is not None:
                decoder_coefficients[atom, :m, :] = decoder_blocks[atom]

        schedule_iter = None
        if gumbel_schedule is not None:
            schedule_iter = dict(gumbel_schedule)
            schedule_iter["iter_count"] = int(schedule_iter.get("iter_count", 0)) + iter_idx
        payload = rust_module().sae_manifold_fit_ibp(
            np.ascontiguousarray(z, dtype=float),
            [_basis_kind_name(spec) for spec in basis_specs],
            [int(d) for d in dims],
            np.ascontiguousarray(basis_values),
            np.ascontiguousarray(basis_jacobian),
            basis_sizes,
            np.ascontiguousarray(decoder_coefficients),
            np.ascontiguousarray(smooth_penalties),
            np.ascontiguousarray(logits_current, dtype=float),
            np.ascontiguousarray(coords_padded),
            float(alpha_value),
            float(tau),
            bool(learnable_alpha),
            float(lambda_sparse),
            float(lambda_smooth),
            1,
            float(learning_rate),
            gumbel_schedule=schedule_iter,
            analytic_penalties=normalized_penalties,
        )
        logits_current = np.asarray(payload["logits"], dtype=float)
        coords_current = [
            _retract_coords(basis_specs[atom_idx], np.asarray(atom_payload["on_atom_coords_t"], dtype=float))
            for atom_idx, atom_payload in enumerate(payload["atoms"])
        ]
        decoder_blocks = [
            np.asarray(atom_payload["decoder_B"], dtype=float).copy()
            for atom_payload in payload["atoms"]
        ]
        if "log_ard" in payload:
            log_ard = [np.asarray(v, dtype=float).copy() for v in payload["log_ard"]]

    if payload is None or decoder_blocks is None:
        raise RuntimeError("sae_manifold_fit did not execute an optimization iteration")

    assignments = np.asarray(payload["assignments_z"], dtype=float)
    final_designs = []
    final_penalties = []
    fitted = np.zeros_like(z)
    for atom in range(k_atoms):
        phi, _jet, penalty = _basis_and_jacobian(basis_specs[atom], coords_current[atom])
        final_designs.append(np.ascontiguousarray(phi, dtype=float))
        final_penalties.append(np.ascontiguousarray(penalty, dtype=float))
        fitted += assignments[:, [atom]] * (final_designs[atom] @ decoder_blocks[atom])
    data_fit = 0.5 * float(np.sum((z - fitted) ** 2))
    smooth = 0.0
    for decoder, penalty in zip(decoder_blocks, final_penalties):
        smooth += 0.5 * lambda_smooth * float(np.sum(decoder * (penalty @ decoder)))
    effective_alpha = alpha_value * lambda_sparse if learnable_alpha else alpha_value
    score = -(
        data_fit
        + _assignment_prior_value(assignments, "ibp_map", lambda_sparse, effective_alpha)
        + smooth
        + _ard_value(coords_current, log_ard)
    )
    atoms: list[SaeManifoldAtomFit] = []
    out_coords: list[np.ndarray] = []
    for atom, atom_payload in enumerate(payload["atoms"]):
        atom_coords = coords_current[atom]
        decoder = decoder_blocks[atom]
        out_coords.append(atom_coords.copy())
        atoms.append(
            SaeManifoldAtomFit(
                basis=basis_specs[atom],
                decoder_coefficients=decoder,
                assignments=assignments[:, atom].copy(),
                coords=atom_coords.copy(),
                evidence=score,
                active_dim=int(np.sum(np.var(atom_coords, axis=0) > 1e-5)),
            )
        )
    comparison = compare_models([{"reml_score": score}], names=[f"K={k_atoms}"])
    return SaeManifoldFitResult(
        atoms=atoms,
        chosen_k=k_atoms,
        evidence_by_candidate={k_atoms: score},
        comparison=comparison,
        fitted=fitted,
        assignments=assignments.copy(),
        coords=out_coords,
        reml_score=score,
    )


def _resolve_dims(k_atoms: int, atom_dim: int | Sequence[int] | Literal["auto"] | None) -> list[int]:
    if atom_dim == "auto" or atom_dim is None:
        return [2 for _ in range(k_atoms)]
    if isinstance(atom_dim, int):
        if atom_dim < 0:
            raise ValueError("atom_dim must be non-negative")
        return [atom_dim for _ in range(k_atoms)]
    dims = [int(d) for d in atom_dim]
    if len(dims) != k_atoms:
        raise ValueError(f"atom_dim list length {len(dims)} must equal n_atoms={k_atoms}")
    if any(d < 0 for d in dims):
        raise ValueError("atom_dim entries must be non-negative")
    return dims


def _resolve_basis_specs(k_atoms: int, atom_basis: str | Smooth | Sequence[str | Smooth]) -> list[str | Smooth]:
    if isinstance(atom_basis, (str, Smooth)):
        raw = [atom_basis for _ in range(k_atoms)]
    else:
        raw = list(atom_basis)
        if len(raw) != k_atoms:
            raise ValueError(f"atom_basis list length {len(raw)} must equal n_atoms={k_atoms}")
    return [_basis_from_name(spec) if isinstance(spec, str) else spec for spec in raw]


def _basis_from_name(name: str) -> Smooth:
    key = name.lower().replace("-", "_")
    if key in {"duchon", "euclidean", "euclidean_patch"}:
        return EuclideanPatch(d=2, centers=None)
    if key in {"periodic", "periodic_spline", "circle"}:
        return Circle(n_knots=24)
    if key == "sphere":
        return Sphere(n_centers=16)
    raise ValueError(f"unsupported atom_basis {name!r}")


def _basis_kind_name(spec: str | Smooth) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, PeriodicSplineCurve):
        return "periodic"
    if isinstance(spec, Sphere):
        return "sphere"
    if isinstance(spec, Duchon):
        return "duchon"
    return spec.__class__.__name__


def _basis_and_jacobian(spec: str | Smooth, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(spec, PeriodicSplineCurve):
        return _periodic_fourier_basis(_retract_coords(spec, t)[:, 0], max(3, int(spec.n_knots // 2)))
    if isinstance(spec, Sphere):
        return _sphere_chart_basis(_retract_coords(spec, t))
    if isinstance(spec, Duchon):
        return _duchon_basis_local(t, spec)
    return _duchon_basis_local(t, Duchon(centers=None, m=2))


def _retract_coords(spec: str | Smooth, coords: np.ndarray) -> np.ndarray:
    out = np.asarray(coords, dtype=float).copy()
    if out.size == 0:
        return out
    if isinstance(spec, PeriodicSplineCurve):
        out[:, 0] = np.mod(out[:, 0], 1.0)
        return out
    if isinstance(spec, Sphere):
        out[:, 0] = np.clip(out[:, 0], -np.pi / 2.0, np.pi / 2.0)
        if out.shape[1] > 1:
            out[:, 1] = (out[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
        return out
    return out


def _duchon_basis_local(t: np.ndarray, spec: Duchon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = t.shape
    n_centers = 12 if spec.centers is None else spec.centers
    if isinstance(n_centers, int):
        centers = _latent_grid_centers(t, int(n_centers))
    else:
        centers = _as_2d_float(n_centers, "Duchon.centers")
    m = int(spec.m)
    power = max(1, 2 * m - d)
    diff = t[:, None, :] - centers[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    if power % 2 == 0:
        phi = (r ** power) * np.log(np.maximum(r, 1e-12))
        dphi_dr = power * (r ** (power - 1)) * np.log(np.maximum(r, 1e-12)) + r ** (power - 1)
    else:
        phi = r ** power
        dphi_dr = power * r ** (power - 1)
    jet = np.zeros((n, centers.shape[0], d), dtype=float)
    scale = np.divide(dphi_dr, np.maximum(r, 1e-12), out=np.zeros_like(r), where=r > 0)
    jet[:, :, :] = scale[:, :, None] * diff
    poly = np.concatenate([np.ones((n, 1)), t], axis=1)
    poly_jet = np.zeros((n, d + 1, d), dtype=float)
    for axis in range(d):
        poly_jet[:, 1 + axis, axis] = 1.0
    Phi = np.concatenate([phi, poly], axis=1)
    Jet = np.concatenate([jet, poly_jet], axis=1)
    penalty = np.eye(Phi.shape[1]) * 1e-6
    penalty[: centers.shape[0], : centers.shape[0]] = _rbf_gram(centers, power)
    return Phi, Jet, penalty


def _periodic_fourier_basis(t: np.ndarray, n_harmonics: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.mod(t, 1.0)
    cols = [np.ones_like(x)]
    dcols = [np.zeros_like(x)]
    penalty_diag = [1e-8]
    for h in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * h * x
        cols.extend([np.sin(angle), np.cos(angle)])
        dcols.extend([2.0 * np.pi * h * np.cos(angle), -2.0 * np.pi * h * np.sin(angle)])
        penalty_diag.extend([float(h**4), float(h**4)])
    Phi = np.stack(cols, axis=1)
    Jet = np.stack(dcols, axis=1)[:, :, None]
    return Phi, Jet, np.diag(penalty_diag)


def _sphere_chart_basis(t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.clip(t[:, 0], -np.pi / 2.0, np.pi / 2.0)
    lon = t[:, 1] if t.shape[1] > 1 else np.zeros_like(lat)
    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)
    x = clat * clon
    y = clat * slon
    z = slat
    Phi = np.stack([np.ones_like(lat), x, y, z, x * y, y * z, x * z], axis=1)
    Jet = np.zeros((t.shape[0], Phi.shape[1], t.shape[1]), dtype=float)
    dx_dlat = -slat * clon
    dx_dlon = -clat * slon
    dy_dlat = -slat * slon
    dy_dlon = clat * clon
    dz_dlat = clat
    Jet[:, 1, 0] = dx_dlat
    Jet[:, 2, 0] = dy_dlat
    Jet[:, 3, 0] = dz_dlat
    if t.shape[1] > 1:
        Jet[:, 1, 1] = dx_dlon
        Jet[:, 2, 1] = dy_dlon
        Jet[:, 4, 0] = dx_dlat * y + x * dy_dlat
        Jet[:, 4, 1] = dx_dlon * y + x * dy_dlon
        Jet[:, 5, 0] = dy_dlat * z + y * dz_dlat
        Jet[:, 5, 1] = dy_dlon * z
        Jet[:, 6, 0] = dx_dlat * z + x * dz_dlat
        Jet[:, 6, 1] = dx_dlon * z
    return Phi, Jet, np.diag([1e-8, 1, 1, 1, 4, 4, 4])


def _initial_coords(z: np.ndarray, labels: np.ndarray, atom: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    if dim == 0:
        return np.zeros((z.shape[0], 0), dtype=float)
    centered = z - z.mean(axis=0, keepdims=True)
    mask = labels == atom
    local = z[mask] if np.any(mask) else z
    local = local - local.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(local, full_matrices=False)
    basis = np.zeros((dim, z.shape[1]), dtype=float)
    take = min(dim, vt.shape[0])
    basis[:take] = vt[:take]
    if take < dim:
        basis[take:] = rng.normal(scale=0.01, size=(dim - take, z.shape[1]))
    coords = centered @ basis.T
    scale = np.std(coords, axis=0, keepdims=True)
    return coords / np.maximum(scale, 1e-6)


def _deterministic_partition(z: np.ndarray, k_atoms: int) -> np.ndarray:
    if k_atoms == 1:
        return np.zeros(z.shape[0], dtype=int)
    score = z @ np.linspace(1.0, 2.0, z.shape[1])
    ranks = np.argsort(np.argsort(score))
    return np.minimum(k_atoms - 1, (ranks * k_atoms) // max(1, z.shape[0]))


def _latent_grid_centers(t: np.ndarray, n_centers: int) -> np.ndarray:
    if t.shape[1] == 1:
        qs = np.linspace(0.02, 0.98, n_centers)
        return np.quantile(t[:, 0], qs).reshape(-1, 1)
    rng = np.random.default_rng(17 + n_centers + t.shape[1])
    idx = rng.choice(t.shape[0], size=min(n_centers, t.shape[0]), replace=False)
    centers = t[idx].copy()
    if centers.shape[0] < n_centers:
        extra = rng.normal(scale=0.1, size=(n_centers - centers.shape[0], t.shape[1]))
        centers = np.concatenate([centers, t.mean(axis=0, keepdims=True) + extra], axis=0)
    return centers


def _rbf_gram(centers: np.ndarray, power: int) -> np.ndarray:
    diff = centers[:, None, :] - centers[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    if power % 2 == 0:
        gram = (r ** power) * np.log(np.maximum(r, 1e-12))
    else:
        gram = r ** power
    return gram @ gram.T + np.eye(gram.shape[0]) * 1e-6


def _normalize_gumbel_schedule(
    schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if hasattr(schedule, "to_rust_descriptor"):
        schedule = schedule.to_rust_descriptor()
    if isinstance(schedule, GumbelTemperatureSchedule):
        schedule = asdict(schedule)
    if not isinstance(schedule, Mapping):
        raise ValueError("gumbel_schedule must be GumbelTemperatureSchedule, a mapping, or None")
    decay = str(_require_schedule_key(schedule, "decay")).lower().replace("-", "_")
    if decay == "exponential":
        decay = "geometric"
    tau_start = float(_require_schedule_key(schedule, "tau_start"))
    raw_tau_min = schedule.get("tau_min", schedule.get("tau_end"))
    if raw_tau_min is None:
        raise ValueError("gumbel_schedule is missing required key 'tau_min'")
    tau_min = float(raw_tau_min)
    if not np.isfinite(tau_start) or tau_start <= 0.0:
        raise ValueError("gumbel_schedule['tau_start'] must be finite and positive")
    if not np.isfinite(tau_min) or tau_min <= 0.0:
        raise ValueError("gumbel_schedule['tau_min'] must be finite and positive")
    if tau_min > tau_start:
        raise ValueError("gumbel_schedule['tau_min'] cannot exceed tau_start")
    iter_count = int(schedule.get("iter_count", 0))
    if iter_count < 0:
        raise ValueError("gumbel_schedule['iter_count'] must be non-negative")
    out: dict[str, Any] = {
        "decay": decay,
        "tau_start": tau_start,
        "tau_min": tau_min,
        "iter_count": iter_count,
    }
    if decay == "geometric":
        rate = float(schedule.get("rate", 0.9))
        if not np.isfinite(rate) or rate <= 0.0 or rate >= 1.0:
            raise ValueError("gumbel_schedule['rate'] must be in (0, 1)")
        out["rate"] = rate
    elif decay == "linear":
        steps = int(_require_schedule_key(schedule, "steps"))
        if steps <= 0:
            raise ValueError("gumbel_schedule['steps'] must be positive")
        out["steps"] = steps
    elif decay == "reciprocal_iter":
        out["decay"] = "reciprocal_iter"
    else:
        raise ValueError(
            "gumbel_schedule['decay'] must be 'geometric', 'linear', or 'reciprocal_iter'"
        )
    return out


def _require_schedule_key(schedule: Mapping[str, Any], key: str) -> Any:
    if key not in schedule:
        raise ValueError(f"gumbel_schedule is missing required key {key!r}")
    return schedule[key]


def _gumbel_tau_at(schedule: dict[str, Any] | None, iter_idx: int, tau: float) -> float:
    if schedule is None:
        return tau
    decay = schedule["decay"]
    tau_start = float(schedule["tau_start"])
    tau_min = float(schedule["tau_min"])
    iter_at = int(schedule.get("iter_count", 0)) + iter_idx
    if decay == "geometric":
        raw = tau_start * (float(schedule["rate"]) ** iter_at)
    elif decay == "linear":
        steps = int(schedule["steps"])
        if iter_at >= steps:
            raw = tau_min
        else:
            raw = tau_start + (iter_at / steps) * (tau_min - tau_start)
    else:
        raw = tau_start / (1.0 + iter_at)
    return max(tau_min, raw)


def _assignment_from_logits(
    logits: np.ndarray,
    assignment_prior: Literal["softmax", "ibp_map", "topk", "jumprelu", "gated"],
    tau: float,
    top_k: int | None = None,
) -> np.ndarray:
    if assignment_prior == "softmax":
        return _softmax(logits, tau)
    if assignment_prior == "ibp_map":
        return _sigmoid(logits / tau)
    if assignment_prior == "topk":
        k = top_k if top_k is not None else max(1, min(logits.shape[1], int(round(tau))))
        return _topk_mask(logits, max(1, min(logits.shape[1], int(k))))
    if assignment_prior == "jumprelu":
        gate = _sigmoid((logits - tau) / max(tau, 1e-12))
        return logits * gate
    gate = (_sigmoid(logits / tau) > 0.5).astype(float)
    return gate * logits


def _topk_mask(values: np.ndarray, k: int) -> np.ndarray:
    order = np.argpartition(-np.abs(values), kth=k - 1, axis=1)[:, :k]
    out = np.zeros_like(values, dtype=float)
    rows = np.arange(values.shape[0])[:, None]
    out[rows, order] = values[rows, order]
    return out


def _softmax(logits: np.ndarray, tau: float) -> np.ndarray:
    scaled = logits / tau
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / ex.sum(axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _assignment_jvp(
    assignments: np.ndarray,
    grad_a: np.ndarray,
    assignment_prior: Literal["softmax", "ibp_map", "topk", "jumprelu", "gated"],
    tau: float,
) -> np.ndarray:
    if assignment_prior == "softmax":
        return _softmax_jvp(assignments, grad_a, tau)
    if assignment_prior == "ibp_map":
        return grad_a * assignments * (1.0 - assignments) / tau
    return grad_a * (assignments != 0.0)


def _softmax_jvp(assignments: np.ndarray, grad_a: np.ndarray, tau: float) -> np.ndarray:
    mean = np.sum(assignments * grad_a, axis=1, keepdims=True)
    return assignments * (grad_a - mean) / tau


def _assignment_prior_value(
    assignments: np.ndarray,
    assignment_prior: Literal["softmax", "ibp_map", "topk", "jumprelu", "gated"],
    lambda_sparse: float,
    alpha: float,
) -> float:
    if assignment_prior == "softmax":
        a = np.clip(assignments, 1e-300, 1.0)
        return float(lambda_sparse * np.sum(-a * np.log(a)))
    if assignment_prior in {"topk", "jumprelu", "gated"}:
        return float(lambda_sparse * np.sum(np.abs(assignments)))
    pi = _ibp_pi_map(assignments, alpha)
    z = np.clip(assignments, 1e-12, 1.0 - 1e-12)
    p = np.clip(pi, 1e-12, 1.0 - 1e-12)
    nll = -np.sum(z * np.log(p)[None, :] + (1.0 - z) * np.log(1.0 - p)[None, :])
    nll += np.sum((alpha / assignments.shape[1] - 1.0) * np.log(p))
    return float(nll)


def _assignment_prior_grad_logits(
    assignments: np.ndarray,
    assignment_prior: Literal["softmax", "ibp_map", "topk", "jumprelu", "gated"],
    lambda_sparse: float,
    alpha: float,
    tau: float,
) -> np.ndarray:
    if assignment_prior == "softmax":
        d_h_da = -lambda_sparse * (np.log(np.clip(assignments, 1e-300, 1.0)) + 1.0)
        mean = np.sum(assignments * d_h_da, axis=1, keepdims=True)
        return assignments * (d_h_da - mean) / tau
    if assignment_prior in {"topk", "jumprelu", "gated"}:
        return lambda_sparse * np.sign(assignments) * (assignments != 0.0)
    pi = np.clip(_ibp_pi_map(assignments, alpha), 1e-12, 1.0 - 1e-12)
    d_p_d_z = np.log((1.0 - pi) / pi)[None, :]
    return d_p_d_z * assignments * (1.0 - assignments) / tau


def _ibp_pi_map(assignments: np.ndarray, alpha: float) -> np.ndarray:
    n, k = assignments.shape
    a = alpha / k
    denom = max(float(n) + a - 1.0, 1e-9)
    raw = (assignments.sum(axis=0) + a - 1.0) / denom
    return np.clip(raw, 1e-9, 1.0 - 1e-9)


def _ard_value(coords: list[np.ndarray], log_ard: list[np.ndarray]) -> float:
    total = 0.0
    for t, lp in zip(coords, log_ard):
        for axis in range(t.shape[1]):
            alpha = float(np.exp(lp[axis]))
            total += 0.5 * alpha * float(np.sum(t[:, axis] ** 2))
            total -= 0.5 * t.shape[0] * float(lp[axis])
    return total


def _split_decoder(B: np.ndarray, widths: Sequence[int]) -> list[np.ndarray]:
    out = []
    cursor = 0
    for width in widths:
        out.append(B[cursor : cursor + width, :])
        cursor += width
    return out


def _block_diag(blocks: Sequence[np.ndarray]) -> np.ndarray:
    total = sum(b.shape[0] for b in blocks)
    out = np.zeros((total, total), dtype=float)
    cursor = 0
    for block in blocks:
        n = block.shape[0]
        out[cursor : cursor + n, cursor : cursor + n] = block
        cursor += n
    return out


def _edf_proxy(fit: SaeManifoldFitResult) -> float:
    return float(sum(atom.decoder_coefficients.size + atom.coords.shape[1] for atom in fit.atoms))


def _as_2d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D numeric array; got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


__all__ = [
    "GumbelTemperatureSchedule",
    "SaeManifoldAtomFit",
    "SaeManifoldFitResult",
    "gumbel_geometric_schedule",
    "gumbel_linear_schedule",
    "gumbel_reciprocal_iter_schedule",
    "sae_manifold_fit",
]
