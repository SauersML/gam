"""Thin public facade for Rust-backed SAE manifold fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from ._binding import rust_module


# Canonical assignment-kind aliases. Both `assignment=` and `assignment_prior=`
# normalize through this map so the two kwargs are strict synonyms.
_ASSIGNMENT_ALIASES: dict[str, str] = {
    "ibp": "ibp_map",
    "ibp_map": "ibp_map",
    "softmax": "softmax",
    "jumprelu": "jumprelu",
    "gated": "jumprelu",
}


def _canonical_assignment(value: str, label: str) -> str:
    name = str(value).strip().lower()
    canon = _ASSIGNMENT_ALIASES.get(name)
    if canon is None:
        raise ValueError(
            f"{label}={value!r} is not a recognized assignment kind; "
            f"expected one of {sorted(set(_ASSIGNMENT_ALIASES))}"
        )
    return canon


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
    _basis_kinds: list[str]
    _atom_dims: list[int]
    _basis_sizes: list[int]
    _n_harmonics: list[int]
    _duchon_centers: list[np.ndarray | None]
    alpha: float = 1.0
    learnable_alpha: bool = False

    def __repr__(self) -> str:
        d_atom = int(self.coords[0].shape[1]) if self.coords else 0
        n, p = (self.fitted.shape if self.fitted.ndim == 2 else (self.fitted.shape[0], 1))
        return (
            f"ManifoldSAE(K={len(self.atoms)}, d_atom={d_atom}, "
            f"atom_topology={self.atom_topology!r}, assignment={self.assignment!r}, "
            f"alpha={self.alpha!r}, learnable_alpha={self.learnable_alpha}, "
            f"n={n}, p={p}, r2={self.reconstruction_r2:.3f})"
        )

    @classmethod
    def from_payload(cls, x: np.ndarray, payload: Mapping[str, Any], topology: str, assignment: str, penalties: list[str], alpha: float = 1.0, learnable_alpha: bool = False) -> "ManifoldSAE":
        plans = list(payload.get("atom_plans", []))
        atoms = [SaeManifoldAtomFit(
            basis=str(atom.get("basis_kind", "")),
            decoder_coefficients=np.asarray(atom["decoder_B"], dtype=float),
            assignments=np.asarray(atom["assignments_z"], dtype=float),
            coords=np.asarray(atom["on_atom_coords_t"], dtype=float),
            evidence=float(payload["reml_score"]),
            active_dim=int(atom.get("active_dim", 0)),
        ) for atom in payload["atoms"]]
        fitted = np.asarray(payload["fitted"], dtype=float)
        assigns = np.asarray(payload["assignments_z"], dtype=float)
        coords = [atom.coords.copy() for atom in atoms]
        score = float(payload["reml_score"])
        chosen_k = int(payload["chosen_k"]) if "chosen_k" in payload else len(atoms)
        low = SaeManifoldFitResult(atoms, chosen_k, {chosen_k: score}, {"winner": f"K={chosen_k}"}, fitted, assigns, coords, score)
        kinds = [str(p.get("kind", atoms[i].basis)) for i, p in enumerate(plans)] if plans else [a.basis for a in atoms]
        dims = [int(p.get("latent_dim", 0)) for p in plans] if plans else [a.coords.shape[1] if a.coords.ndim == 2 else 0 for a in atoms]
        sizes = [int(p.get("basis_size", 0)) for p in plans] if plans else [int(a.decoder_coefficients.shape[0]) for a in atoms]
        nharm = [int(p.get("n_harmonics", 0)) for p in plans] if plans else [0 for _ in atoms]
        centers: list[np.ndarray | None] = [(None if p.get("duchon_centers") is None else np.asarray(p["duchon_centers"], dtype=float)) for p in plans] if plans else [None for _ in atoms]
        return cls(
            atoms=atoms, atom_topology=str(topology), assignment=str(assignment),
            primitive_names=["rust_module.sae_manifold_fit_minimal", *penalties],
            fitted=fitted, assignments=assigns, coords=coords,
            decoder_blocks=[a.decoder_coefficients.copy() for a in atoms],
            basis_specs=kinds, reml_score=score,
            reconstruction_r2=float(rust_module().sae_manifold_reconstruction_r2(x, fitted)),
            training_mean=x.mean(axis=0), training_data=x.copy(), low_level=low,
            _basis_kinds=kinds, _atom_dims=dims, _basis_sizes=sizes,
            _n_harmonics=nharm, _duchon_centers=centers,
            alpha=float(alpha), learnable_alpha=bool(learnable_alpha),
        )

    def reconstruct(self, X: Any) -> np.ndarray:
        x = _as_2d_float(X, "X")
        if x.shape == self.training_data.shape and np.allclose(x, self.training_data):
            return self.fitted.copy()
        kind = "ibp_map" if self.assignment in {"ibp", "ibp_map"} else ("jumprelu" if self.assignment == "gated" else self.assignment)
        return np.asarray(rust_module().sae_manifold_predict_oos(
            np.ascontiguousarray(x), list(self._basis_kinds), list(self._atom_dims),
            [np.ascontiguousarray(b) for b in self.decoder_blocks],
            [None if c is None else np.ascontiguousarray(c) for c in self._duchon_centers],
            [(int(h) if k in {"periodic", "torus"} else None) for k, h in zip(self._basis_kinds, self._n_harmonics)],
            alpha=1.0, tau=0.5, assignment_kind=str(kind),
            sparsity_strength=1.0, smoothness=1.0, max_iter=50, learning_rate=0.04, random_state=0,
        ), dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        return self.reconstruct(X)

    def per_atom_active_set(self, X: Any, threshold: float | None = None) -> np.ndarray:
        _as_2d_float(X, "X")
        return self.assignments >= (0.5 if threshold is None else float(threshold))

    def per_atom_latent_for(self, X: Any) -> list[np.ndarray]:
        _as_2d_float(X, "X")
        return [c.copy() for c in self.coords]

    def get_decoder(self) -> list[np.ndarray]:
        return [b.copy() for b in self.decoder_blocks]

    def get_anchors(self) -> list[np.ndarray]:
        return [c.copy() for c in self.coords]

    def summary(self) -> dict[str, Any]:
        threshold = 0.5 if self.assignment == "ibp" else 1.0 / max(1, len(self.atoms))
        avg_active, mean_mass = rust_module().sae_manifold_assignment_summary(self.assignments, threshold)
        return {
            "K": len(self.atoms),
            "d_atom": int(self.coords[0].shape[1]) if self.coords else 0,
            "atom_topology": self.atom_topology, "assignment": self.assignment,
            "alpha": float(self.alpha), "learnable_alpha": bool(self.learnable_alpha),
            "reml_score": float(self.reml_score), "reconstruction_r2": float(self.reconstruction_r2),
            "avg_active_atoms": float(avg_active), "mean_assignment_mass": float(mean_mass),
            "active_dims": [a.active_dim for a in self.atoms],
            "primitives": list(self.primitive_names),
        }

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable JSON-compatible serialization of this fit.

        The dict can be passed to :meth:`ManifoldSAE.from_dict` (or written to
        disk via :meth:`save` / :func:`gamfit.save`) to recover an object that
        reproduces :meth:`predict` outputs bit-exactly on training data.
        """
        return {
            "schema": "gamfit.ManifoldSAE/v1",
            "atom_topology": self.atom_topology,
            "assignment": self.assignment,
            "alpha": float(self.alpha),
            "learnable_alpha": bool(self.learnable_alpha),
            "primitive_names": list(self.primitive_names),
            "basis_specs": list(self.basis_specs),
            "reml_score": float(self.reml_score),
            "reconstruction_r2": float(self.reconstruction_r2),
            "training_mean": self.training_mean.tolist(),
            "training_data": self.training_data.tolist(),
            "fitted": self.fitted.tolist(),
            "assignments": self.assignments.tolist(),
            "coords": [c.tolist() for c in self.coords],
            "decoder_blocks": [b.tolist() for b in self.decoder_blocks],
            "atoms": [
                {
                    "basis": a.basis,
                    "decoder_coefficients": a.decoder_coefficients.tolist(),
                    "assignments": a.assignments.tolist(),
                    "coords": a.coords.tolist(),
                    "evidence": float(a.evidence),
                    "active_dim": int(a.active_dim),
                }
                for a in self.atoms
            ],
            "basis_kinds": list(self._basis_kinds),
            "atom_dims": list(self._atom_dims),
            "basis_sizes": list(self._basis_sizes),
            "n_harmonics": list(self._n_harmonics),
            "duchon_centers": [None if c is None else c.tolist() for c in self._duchon_centers],
        }

    def save(self, path: str | Path) -> None:
        """Write this fit to ``path`` as JSON. Round-trips via :func:`gamfit.load`."""
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifoldSAE":
        schema = str(payload.get("schema", ""))
        if schema and schema != "gamfit.ManifoldSAE/v1":
            raise ValueError(f"ManifoldSAE.from_dict: unsupported schema {schema!r}")
        atoms = [
            SaeManifoldAtomFit(
                basis=str(a["basis"]),
                decoder_coefficients=np.asarray(a["decoder_coefficients"], dtype=float),
                assignments=np.asarray(a["assignments"], dtype=float),
                coords=np.asarray(a["coords"], dtype=float),
                evidence=float(a["evidence"]),
                active_dim=int(a["active_dim"]),
            )
            for a in payload["atoms"]
        ]
        fitted = np.asarray(payload["fitted"], dtype=float)
        assigns = np.asarray(payload["assignments"], dtype=float)
        coords = [np.asarray(c, dtype=float) for c in payload["coords"]]
        decoder_blocks = [np.asarray(b, dtype=float) for b in payload["decoder_blocks"]]
        score = float(payload["reml_score"])
        chosen_k = len(atoms)
        low = SaeManifoldFitResult(
            atoms, chosen_k, {chosen_k: score}, {"winner": f"K={chosen_k}"}, fitted, assigns, coords, score,
        )
        centers: list[np.ndarray | None] = [
            None if c is None else np.asarray(c, dtype=float) for c in payload["duchon_centers"]
        ]
        return cls(
            atoms=atoms,
            atom_topology=str(payload["atom_topology"]),
            assignment=str(payload["assignment"]),
            primitive_names=list(payload["primitive_names"]),
            fitted=fitted,
            assignments=assigns,
            coords=coords,
            decoder_blocks=decoder_blocks,
            basis_specs=list(payload["basis_specs"]),
            reml_score=score,
            reconstruction_r2=float(payload["reconstruction_r2"]),
            training_mean=np.asarray(payload["training_mean"], dtype=float),
            training_data=np.asarray(payload["training_data"], dtype=float),
            low_level=low,
            _basis_kinds=list(payload["basis_kinds"]),
            _atom_dims=[int(d) for d in payload["atom_dims"]],
            _basis_sizes=[int(s) for s in payload["basis_sizes"]],
            _n_harmonics=[int(h) for h in payload["n_harmonics"]],
            _duchon_centers=centers,
            alpha=float(payload.get("alpha", 1.0)),
            learnable_alpha=bool(payload.get("learnable_alpha", False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ManifoldSAE":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True, init=False, slots=True)
class GumbelTemperatureSchedule:
    tau_start: float
    tau_min: float
    decay: Literal["geometric", "exponential", "linear", "reciprocal_iter"]
    rate: float | None = None
    steps: int | None = None
    iter_count: int = 0

    def __init__(self, tau_start: float, tau_min: float | None = None, decay: str = "geometric",
                 rate: float | None = None, steps: int | None = None, iter_count: int = 0,
                 *, tau_end: float | None = None) -> None:
        if tau_min is None:
            if tau_end is None:
                raise TypeError("GumbelTemperatureSchedule requires tau_min or tau_end")
            tau_min = tau_end
        if tau_end is not None and float(tau_end) != float(tau_min):
            raise ValueError("GumbelTemperatureSchedule tau_min and tau_end disagree")
        name = str(decay).lower().replace("-", "_")
        if name == "exponential":
            name = "geometric"
        _validate_gumbel_schedule_fields(
            tau_start=float(tau_start), tau_min=float(tau_min), decay=name,
            rate=rate, steps=steps, iter_count=int(iter_count),
        )
        object.__setattr__(self, "tau_start", float(tau_start))
        object.__setattr__(self, "tau_min", float(tau_min))
        object.__setattr__(self, "decay", name)
        object.__setattr__(self, "rate", 0.9 if rate is None and name == "geometric" else rate)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "iter_count", int(iter_count))

    def to_rust_descriptor(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tau_start": self.tau_start, "tau_min": self.tau_min, "decay": self.decay, "iter_count": self.iter_count}
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


def sae_manifold_fit(X: Any = None, K: int | None = None, d_atom: int = 2, atom_topology: str = "circle",
                     assignment: str = "ibp", schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None = None,
                     isometry_weight: float = 1.0, ard_per_atom: bool = True,
                     mechanism_sparsity_groups: list[list[int]] | None = None, n_iter: int = 50, *,
                     Z: Any = None, sparsity_weight: float = 1.0, smoothness_weight: float = 1.0,
                     alpha: float | str = 1.0, learning_rate: float | None = None, random_state: int = 0,
                     block_orthogonality_weight: float = 0.0, topology_selector: Any | None = None,
                     top_k: int | None = None, **kwargs: Any) -> ManifoldSAE:
    src = Z if Z is not None else X
    if src is None:
        raise TypeError("sae_manifold_fit requires Z= (or X=) input array")
    x = _as_2d_float(src, "Z")
    n_atoms_kw = kwargs.pop("n_atoms", None)
    if n_atoms_kw is not None and K is not None and int(n_atoms_kw) != int(K):
        raise ValueError(
            f"sae_manifold_fit: K and n_atoms both supplied with different values "
            f"({int(K)} vs {int(n_atoms_kw)}); pass only one (they are aliases)."
        )
    k_atoms = int(n_atoms_kw if n_atoms_kw is not None else (K if K is not None else 0))
    atom_basis = kwargs.pop("atom_basis", None)
    atom_dim = kwargs.pop("atom_dim", d_atom)
    assignment_prior = kwargs.pop("assignment_prior", None)
    gumbel_schedule = kwargs.pop("gumbel_schedule", schedule)
    max_iter_total = int(kwargs.pop("max_iter", n_iter))
    smoothness = float(kwargs.pop("smoothness", smoothness_weight))
    sparsity = float(kwargs.pop("sparsity_strength", sparsity_weight))
    tau = float(kwargs.pop("tau", _schedule_tau_start(gumbel_schedule, 0.5)))
    if kwargs:
        raise TypeError(f"unexpected sae_manifold_fit keyword(s): {', '.join(sorted(kwargs))}")
    if k_atoms <= 0:
        raise ValueError(f"K/n_atoms must be positive, got {k_atoms}")
    if max_iter_total < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter_total}")
    dims = _dims(k_atoms, atom_dim)
    bases = _bases(k_atoms, atom_basis, atom_topology)
    # Normalize `assignment` and `assignment_prior` through a single alias map.
    # If both are supplied and resolve to different canonical kinds, raise an
    # eager argument-conflict error rather than letting Rust crash in the
    # Schur path.
    canonical_assignment = _canonical_assignment(assignment, "assignment")
    if assignment_prior is not None:
        canonical_prior = _canonical_assignment(assignment_prior, "assignment_prior")
        if canonical_prior != canonical_assignment:
            raise ValueError(
                f"sae_manifold_fit: assignment={assignment!r} and assignment_prior={assignment_prior!r} "
                f"resolve to different kinds ({canonical_assignment!r} vs {canonical_prior!r}); "
                f"pass only one (they are aliases)."
            )
        kind = canonical_prior
    else:
        kind = canonical_assignment
    alpha_value = 1.0 if alpha == "auto" else float(alpha)
    # Magic-by-default learning rate: the SAE Newton kernel is a damped
    # Gauss-Newton step against a quadratic local model with Armijo
    # backtracking. For softmax / IBP-MAP assignments the natural full step
    # is `lr=1.0` (matches the Rust reference test
    # `sae_manifold_fit_10_steps_one_harmonic_reaches_high_r2`, which reaches
    # R² ≥ 0.95 in 10 steps from a phase-shifted init). A small literal
    # `lr=0.05` starves the assignment posterior of gradient mass and lets
    # the IBP sigmoid drift into the saturated tail (the issue #165
    # collapse: assignment mass ~1e-146). JumpReLU keeps the historical
    # smaller step because its hard-gate STE is more sensitive to
    # overshooting the threshold. Callers can still override explicitly.
    if learning_rate is None:
        effective_lr = 0.05 if kind == "jumprelu" else 1.0
    else:
        effective_lr = float(learning_rate)
    penalties = [n for n, ok in (("IsometryPenalty", isometry_weight > 0.0), ("ARDPenalty", ard_per_atom),
        ("MechanismSparsityPenalty", mechanism_sparsity_groups is not None),
        ("BlockOrthogonalityPenalty", block_orthogonality_weight > 0.0)) if ok]
    payload = rust_module().sae_manifold_fit_minimal(
        np.ascontiguousarray(x),
        int(k_atoms),
        [str(b) for b in bases],
        [int(d) for d in dims],
        str(kind),
        float(alpha_value),
        float(tau),
        bool(alpha == "auto"),
        float(sparsity),
        float(smoothness),
        int(max_iter_total),
        float(effective_lr),
        int(random_state),
        int(top_k) if top_k is not None else 0,
        gumbel_schedule=_schedule_payload(gumbel_schedule),
    )
    return ManifoldSAE.from_payload(
        x, dict(payload), atom_topology, assignment, penalties,
        alpha=float(alpha_value), learnable_alpha=bool(alpha == "auto"),
    )


def _as_2d_float(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1D or 2D numeric array")
    return np.ascontiguousarray(arr)


def _dims(k_atoms: int, atom_dim: Any) -> list[int]:
    if atom_dim in (None, "auto"):
        return [2] * k_atoms
    if isinstance(atom_dim, int):
        return [int(atom_dim)] * k_atoms
    out = [int(d) for d in atom_dim]
    if len(out) != k_atoms or min(out, default=0) < 0:
        raise ValueError("atom_dim must provide one non-negative dimension per atom")
    return out


def _bases(k_atoms: int, atom_basis: Any, atom_topology: str) -> list[str]:
    if atom_basis is None:
        atom_basis = {"circle": "periodic", "periodic": "periodic", "sphere": "sphere", "torus": "torus", "euclidean": "duchon"}.get(str(atom_topology), atom_topology)
    raw = [atom_basis] * k_atoms if isinstance(atom_basis, str) else list(atom_basis)
    if len(raw) != k_atoms:
        raise ValueError("atom_basis must provide one basis per atom")
    return [str(v) for v in raw]


def _validate_gumbel_schedule_fields(
    *, tau_start: float, tau_min: float, decay: str,
    rate: float | None, steps: int | None, iter_count: int,
) -> None:
    if not (np.isfinite(tau_start) and tau_start > 0.0):
        raise ValueError(f"GumbelTemperatureSchedule: tau_start must be finite and positive; got {tau_start}")
    if not (np.isfinite(tau_min) and tau_min > 0.0):
        raise ValueError(f"GumbelTemperatureSchedule: tau_min must be finite and positive; got {tau_min}")
    if tau_min > tau_start:
        raise ValueError(
            f"GumbelTemperatureSchedule: tau_min ({tau_min}) cannot exceed tau_start ({tau_start})"
        )
    if decay not in {"geometric", "linear", "reciprocal_iter"}:
        raise ValueError(f"GumbelTemperatureSchedule: unknown decay {decay!r}")
    if rate is not None and (not np.isfinite(rate) or rate <= 0.0 or rate >= 1.0):
        raise ValueError(f"GumbelTemperatureSchedule: rate must be in (0, 1); got {rate}")
    if steps is not None and int(steps) < 1:
        raise ValueError(f"GumbelTemperatureSchedule: steps must be >= 1; got {steps}")
    if int(iter_count) < 0:
        raise ValueError(f"GumbelTemperatureSchedule: iter_count must be >= 0; got {iter_count}")


def _schedule_payload(schedule: Any) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, GumbelTemperatureSchedule):
        return schedule.to_rust_descriptor()
    descriptor = dict(schedule)
    decay = str(descriptor.get("decay", "geometric")).lower().replace("-", "_")
    if decay == "exponential":
        decay = "geometric"
    if "tau_start" not in descriptor:
        raise ValueError("GumbelTemperatureSchedule (dict form): missing 'tau_start'")
    tau_start = float(descriptor["tau_start"])
    if "tau_min" in descriptor:
        tau_min = float(descriptor["tau_min"])
    elif "tau_end" in descriptor:
        tau_min = float(descriptor["tau_end"])
    else:
        raise ValueError("GumbelTemperatureSchedule (dict form): missing 'tau_min' (or 'tau_end')")
    rate = descriptor.get("rate")
    steps = descriptor.get("steps")
    iter_count = int(descriptor.get("iter_count", 0))
    _validate_gumbel_schedule_fields(
        tau_start=tau_start, tau_min=tau_min, decay=decay,
        rate=None if rate is None else float(rate),
        steps=None if steps is None else int(steps),
        iter_count=iter_count,
    )
    descriptor["decay"] = decay
    descriptor["tau_min"] = tau_min
    descriptor["tau_start"] = tau_start
    descriptor["iter_count"] = iter_count
    return descriptor


def _schedule_tau_start(schedule: Any, default: float) -> float:
    payload = _schedule_payload(schedule)
    return default if payload is None else float(payload["tau_start"])


__all__ = ["GumbelTemperatureSchedule", "ManifoldSAE", "SaeManifoldAtomFit", "SaeManifoldFitResult",
           "gumbel_geometric_schedule", "gumbel_linear_schedule", "gumbel_reciprocal_iter_schedule",
           "sae_manifold_fit"]
