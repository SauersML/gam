"""Thin public facade for Rust-backed SAE manifold fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from ._binding import rust_module


@dataclass(slots=True)
class SaeManifoldAtomFit:
    basis: str
    decoder_coefficients: np.ndarray
    assignments: np.ndarray
    coords: np.ndarray
    evidence: float
    active_dim: int


@dataclass(slots=True)
class SaeManifoldFitResult:
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
    atoms: list[SaeManifoldAtomFit]
    atom_topology: str
    assignment: str
    primitive_names: list[str]
    fitted: np.ndarray
    assignments: np.ndarray
    coords: list[np.ndarray]
    decoder_blocks: list[np.ndarray]
    basis_specs: list[str]
    reml_score: float
    reconstruction_r2: float
    training_mean: np.ndarray
    training_data: np.ndarray
    low_level: SaeManifoldFitResult

    def reconstruct(self, X: Any) -> np.ndarray:
        x = _as_2d_float(X, "X")
        if x.shape != self.training_data.shape or not np.allclose(x, self.training_data):
            raise RuntimeError("SAE manifold prediction for new rows is Rust-owned and not exposed here")
        return self.fitted.copy()

    def per_atom_active_set(self, X: Any, threshold: float | None = None) -> np.ndarray:
        _as_2d_float(X, "X")
        cutoff = 0.5 if threshold is None else float(threshold)
        return self.assignments >= cutoff

    def per_atom_latent_for(self, X: Any) -> list[np.ndarray]:
        _as_2d_float(X, "X")
        return [coord.copy() for coord in self.coords]

    def get_decoder(self) -> list[np.ndarray]:
        return [block.copy() for block in self.decoder_blocks]

    def get_anchors(self) -> list[np.ndarray]:
        return [coord.copy() for coord in self.coords]

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


@dataclass(frozen=True, init=False, slots=True)
class GumbelTemperatureSchedule:
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
        if tau_end is not None and float(tau_end) != float(tau_min):
            raise ValueError("GumbelTemperatureSchedule tau_min and tau_end disagree")
        decay_name = str(decay).lower().replace("-", "_")
        if decay_name == "exponential":
            decay_name = "geometric"
        object.__setattr__(self, "tau_start", float(tau_start))
        object.__setattr__(self, "tau_min", float(tau_min))
        object.__setattr__(self, "decay", decay_name)
        object.__setattr__(self, "rate", 0.9 if rate is None and decay_name == "geometric" else rate)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "iter_count", int(iter_count))

    def to_rust_descriptor(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tau_start": self.tau_start,
            "tau_min": self.tau_min,
            "decay": self.decay,
            "iter_count": self.iter_count,
        }
        if self.rate is not None:
            out["rate"] = float(self.rate)
        if self.steps is not None:
            out["steps"] = int(self.steps)
        return out


def gumbel_geometric_schedule(tau_start: float, tau_min: float, rate: float, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "geometric", rate=rate, iter_count=iter_count)


def gumbel_linear_schedule(tau_start: float, tau_min: float, steps: int, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "linear", steps=steps, iter_count=iter_count)


def gumbel_reciprocal_iter_schedule(tau_start: float, tau_min: float, iter_count: int = 0) -> GumbelTemperatureSchedule:
    return GumbelTemperatureSchedule(tau_start, tau_min, "reciprocal_iter", iter_count=iter_count)


def sae_manifold_fit(
    X: Any,
    K: int | None = None,
    d_atom: int = 2,
    atom_topology: str = "circle",
    assignment: str = "ibp",
    schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None = None,
    isometry_weight: float = 1.0,
    ard_per_atom: bool = True,
    mechanism_sparsity_groups: list[list[int]] | None = None,
    n_iter: int = 50,
    *,
    sparsity_weight: float = 1.0,
    smoothness_weight: float = 1.0,
    alpha: float | str = 1.0,
    learning_rate: float = 0.05,
    random_state: int = 0,
    block_orthogonality_weight: float = 0.0,
    topology_selector: Any | None = None,
    top_k: int | None = None,
    **kwargs: Any,
) -> ManifoldSAE:
    x = _as_2d_float(X, "X")
    k_atoms = int(kwargs.pop("n_atoms", K if K is not None else 0))
    atom_basis = kwargs.pop("atom_basis", None)
    atom_dim = kwargs.pop("atom_dim", d_atom)
    assignment_prior = kwargs.pop("assignment_prior", None)
    gumbel_schedule = kwargs.pop("gumbel_schedule", schedule)
    max_iter = int(kwargs.pop("max_iter", n_iter))
    smoothness = kwargs.pop("smoothness", smoothness_weight)
    sparsity = kwargs.pop("sparsity_strength", sparsity_weight)
    tau = float(kwargs.pop("tau", _schedule_tau_start(gumbel_schedule, 0.5)))
    if kwargs:
        raise TypeError(f"unexpected sae_manifold_fit keyword(s): {', '.join(sorted(kwargs))}")
    if k_atoms <= 0:
        raise ValueError(f"K/n_atoms must be positive, got {k_atoms}")

    dims = _dims(k_atoms, atom_dim)
    bases = _bases(k_atoms, atom_basis, atom_topology)
    assignment_kind = str(assignment_prior or {"ibp": "ibp_map"}.get(assignment, assignment))
    if assignment_kind == "gated":
        assignment_kind = "jumprelu"
    alpha_value = 1.0 if alpha == "auto" else float(alpha)
    schedule_payload = _schedule_payload(gumbel_schedule)
    penalties = [name for name, enabled in {
        "IsometryPenalty": isometry_weight > 0.0,
        "ARDPenalty": ard_per_atom,
        "MechanismSparsityPenalty": mechanism_sparsity_groups is not None,
        "BlockOrthogonalityPenalty": block_orthogonality_weight > 0.0,
    }.items() if enabled]
    payload = rust_module().sae_manifold_fit_minimal(
        x,
        k_atoms,
        bases,
        dims,
        assignment_kind,
        float(alpha_value),
        tau,
        bool(alpha == "auto"),
        float(sparsity),
        float(smoothness),
        max_iter,
        float(learning_rate),
        int(random_state),
        top_k,
        gumbel_schedule=schedule_payload,
    )
    return _wrap_payload(x, payload, atom_topology, assignment, penalties)


def _wrap_payload(x: np.ndarray, payload: Mapping[str, Any], topology: str, assignment: str, penalties: list[str]) -> ManifoldSAE:
    atoms = [
        SaeManifoldAtomFit(
            basis=str(atom.get("basis_kind", "")),
            decoder_coefficients=np.asarray(atom["decoder_B"], dtype=float),
            assignments=np.asarray(atom["assignments_z"], dtype=float),
            coords=np.asarray(atom["on_atom_coords_t"], dtype=float),
            evidence=float(payload["reml_score"]),
            active_dim=int(atom.get("active_dim", 0)),
        )
        for atom in payload["atoms"]
    ]
    fitted = np.asarray(payload["fitted"], dtype=float)
    assignments = np.asarray(payload["assignments_z"], dtype=float)
    coords = [atom.coords.copy() for atom in atoms]
    score = float(payload["reml_score"])
    low = SaeManifoldFitResult(atoms, len(atoms), {len(atoms): score}, {"winner": f"K={len(atoms)}"}, fitted, assignments, coords, score)
    return ManifoldSAE(
        atoms=atoms,
        atom_topology=str(topology),
        assignment=str(assignment),
        primitive_names=["rust_module.sae_manifold_fit_minimal", *penalties],
        fitted=fitted,
        assignments=assignments,
        coords=coords,
        decoder_blocks=[atom.decoder_coefficients.copy() for atom in atoms],
        basis_specs=[atom.basis for atom in atoms],
        reml_score=score,
        reconstruction_r2=_r2(x, fitted),
        training_mean=x.mean(axis=0),
        training_data=x.copy(),
        low_level=low,
    )


def _as_2d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D or 2D numeric array")
    return np.ascontiguousarray(arr)


def _dims(k_atoms: int, atom_dim: int | list[int] | tuple[int, ...] | str | None) -> list[int]:
    if atom_dim in (None, "auto"):
        return [2] * k_atoms
    if isinstance(atom_dim, int):
        return [int(atom_dim)] * k_atoms
    out = [int(dim) for dim in atom_dim]
    if len(out) != k_atoms or min(out, default=0) < 0:
        raise ValueError("atom_dim must provide one non-negative dimension per atom")
    return out


def _bases(k_atoms: int, atom_basis: Any, atom_topology: str) -> list[str]:
    if atom_basis is None:
        atom_basis = {"circle": "circle", "sphere": "sphere", "euclidean": "duchon"}.get(str(atom_topology), atom_topology)
    raw = [atom_basis] * k_atoms if isinstance(atom_basis, str) else list(atom_basis)
    if len(raw) != k_atoms:
        raise ValueError("atom_basis must provide one basis per atom")
    return [str(value) for value in raw]


def _schedule_payload(schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if schedule is None else schedule.to_rust_descriptor() if hasattr(schedule, "to_rust_descriptor") else dict(schedule)


def _schedule_tau_start(schedule: Any, default: float) -> float:
    return default if (payload := _schedule_payload(schedule)) is None else float(payload["tau_start"])


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    return 1.0 - float(np.sum((x - fitted) ** 2)) / max(float(np.sum((x - x.mean(axis=0)) ** 2)), 1e-12)


__all__ = ["GumbelTemperatureSchedule", "ManifoldSAE", "SaeManifoldAtomFit", "SaeManifoldFitResult", "gumbel_geometric_schedule", "gumbel_linear_schedule", "gumbel_reciprocal_iter_schedule", "sae_manifold_fit"]
