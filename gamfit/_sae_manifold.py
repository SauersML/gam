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
    _basis_kinds: list[str]
    _atom_dims: list[int]
    _basis_sizes: list[int]
    _n_harmonics: list[int]
    _duchon_centers: list[np.ndarray | None]

    def reconstruct(self, X: Any) -> np.ndarray:
        x = _as_2d_float(X, "X")
        if x.shape == self.training_data.shape and np.allclose(x, self.training_data):
            return self.fitted.copy()
        return _oos_reconstruct(self, x)

    def predict(self, X: Any) -> np.ndarray:
        return self.reconstruct(X)

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
        threshold = 0.5 if self.assignment == "ibp" else 1.0 / max(1, len(self.atoms))
        avg_active, mean_mass = rust_module().sae_manifold_assignment_summary(
            self.assignments, threshold
        )
        return {
            "K": len(self.atoms),
            "d_atom": int(self.coords[0].shape[1]) if self.coords else 0,
            "atom_topology": self.atom_topology,
            "assignment": self.assignment,
            "reml_score": float(self.reml_score),
            "reconstruction_r2": float(self.reconstruction_r2),
            "avg_active_atoms": float(avg_active),
            "mean_assignment_mass": float(mean_mass),
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


_PERIODIC_BASES = frozenset({"periodic", "circle", "circular", "fourier"})
_DUCHON_BASES = frozenset({"duchon", "euclidean", "thin_plate", "tps"})


def sae_manifold_fit(
    X: Any = None,
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
    Z: Any = None,
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
    data_source = Z if Z is not None else X
    if data_source is None:
        raise TypeError("sae_manifold_fit requires Z= (or X=) input array")
    x = _as_2d_float(data_source, "Z")

    k_atoms = int(kwargs.pop("n_atoms", K if K is not None else 0))
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
    assignment_kind = str(assignment_prior or {"ibp": "ibp_map"}.get(assignment, assignment))
    if assignment_kind == "gated":
        assignment_kind = "jumprelu"
    alpha_value = 1.0 if alpha == "auto" else float(alpha)
    learnable_alpha = bool(alpha == "auto")
    schedule_payload = _schedule_payload(gumbel_schedule)
    penalties = [name for name, enabled in {
        "IsometryPenalty": isometry_weight > 0.0,
        "ARDPenalty": ard_per_atom,
        "MechanismSparsityPenalty": mechanism_sparsity_groups is not None,
        "BlockOrthogonalityPenalty": block_orthogonality_weight > 0.0,
    }.items() if enabled]

    rng = np.random.default_rng(int(random_state))
    n_obs, p_out = x.shape

    # Per-atom basis specs (kind + n_harmonics / centers).
    atom_specs: list[dict[str, Any]] = []
    for atom_idx in range(k_atoms):
        kind = bases[atom_idx].lower()
        if kind in _PERIODIC_BASES:
            n_harm = max(1, int(dims[atom_idx]))
            atom_specs.append({"kind": "periodic", "n_harmonics": n_harm, "m": 1 + 2 * n_harm})
        elif kind in _DUCHON_BASES:
            d = max(1, int(dims[atom_idx]))
            n_centers = max(8, min(n_obs, 32))
            idx = rng.choice(n_obs, size=n_centers, replace=False) if n_obs >= n_centers else np.arange(n_obs)
            seed_coords = _pca_seed_coords(x, d, rng)
            centers = seed_coords[idx, :].copy()
            atom_specs.append({"kind": "duchon", "centers": centers, "m": n_centers + d + 1})
        else:
            raise ValueError(f"unsupported atom basis kind: {bases[atom_idx]!r}")

    m_max = max(spec["m"] for spec in atom_specs)
    d_max = max(dims) if dims else 1

    # PCA-style coord seeding (in-Python; mirrors what an in-Rust seeder would do).
    initial_coords = np.zeros((k_atoms, n_obs, d_max), dtype=float)
    for atom_idx in range(k_atoms):
        d = dims[atom_idx]
        if d <= 0:
            continue
        spec = atom_specs[atom_idx]
        if spec["kind"] == "periodic":
            theta = _pca_periodic_seed(x, rng, atom_idx)
            initial_coords[atom_idx, :, 0] = theta
            for axis in range(1, d):
                initial_coords[atom_idx, :, axis] = _pca_axis(x, axis, rng)
        else:
            seed = _pca_seed_coords(x, d, rng)
            initial_coords[atom_idx, :, :d] = seed

    # Pre-allocate stacked basis arrays.
    decoder_coefficients = np.zeros((k_atoms, m_max, p_out), dtype=float)
    smooth_penalties_stack = np.zeros((k_atoms, m_max, m_max), dtype=float)
    initial_logits = np.zeros((n_obs, k_atoms), dtype=float)
    basis_sizes = [int(spec["m"]) for spec in atom_specs]

    last_payload: dict[str, Any] | None = None
    for _ in range(max_iter_total):
        basis_values, basis_jacobian = _build_basis_stack(
            atom_specs, initial_coords, dims, m_max, d_max
        )
        smooth_penalties_stack = _build_penalty_stack(atom_specs, m_max)

        rust = rust_module()
        result = rust.sae_manifold_fit(
            np.ascontiguousarray(x),
            [spec["kind"] for spec in atom_specs],
            list(dims),
            np.ascontiguousarray(basis_values),
            np.ascontiguousarray(basis_jacobian),
            list(basis_sizes),
            np.ascontiguousarray(decoder_coefficients),
            np.ascontiguousarray(smooth_penalties_stack),
            np.ascontiguousarray(initial_logits),
            np.ascontiguousarray(initial_coords),
            float(alpha_value),
            float(tau),
            bool(learnable_alpha),
            assignment_kind=str(assignment_kind),
            sparsity_strength=float(sparsity),
            smoothness=float(smoothness),
            max_iter=1,
            learning_rate=float(learning_rate),
            gumbel_schedule=schedule_payload,
        )
        last_payload = dict(result)
        # Pull updated state for the next iteration.
        for atom_idx, atom in enumerate(last_payload["atoms"]):
            new_coords = np.asarray(atom["on_atom_coords_t"], dtype=float)
            d = dims[atom_idx]
            if d > 0 and new_coords.size:
                initial_coords[atom_idx, :, :d] = new_coords[:, :d]
            decoder = np.asarray(atom["decoder_B"], dtype=float)
            m = basis_sizes[atom_idx]
            if decoder.size:
                decoder_coefficients[atom_idx, :m, :] = decoder[:m, :]
        if "logits" in last_payload:
            new_logits = np.asarray(last_payload["logits"], dtype=float)
            if new_logits.shape == initial_logits.shape:
                initial_logits = np.ascontiguousarray(new_logits)

    if last_payload is None:
        raise RuntimeError("sae_manifold_fit produced no iterations")

    return _wrap_payload(
        x,
        last_payload,
        atom_topology,
        assignment,
        penalties,
        atom_specs,
        dims,
        basis_sizes,
    )


def _build_basis_stack(
    atom_specs: list[dict[str, Any]],
    coords: np.ndarray,
    dims: list[int],
    m_max: int,
    d_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    k_atoms = len(atom_specs)
    n_obs = coords.shape[1]
    phi_stack = np.zeros((k_atoms, n_obs, m_max), dtype=float)
    jet_stack = np.zeros((k_atoms, n_obs, m_max, d_max), dtype=float)
    rust = rust_module()
    for atom_idx, spec in enumerate(atom_specs):
        d = dims[atom_idx]
        if spec["kind"] == "periodic":
            t = np.ascontiguousarray(coords[atom_idx, :, 0])
            phi, jet, _penalty = rust.periodic_basis_with_jet(t, int(spec["n_harmonics"]))
            phi_arr = np.asarray(phi, dtype=float)
            jet_arr = np.asarray(jet, dtype=float)
            m = phi_arr.shape[1]
            phi_stack[atom_idx, :, :m] = phi_arr
            jet_stack[atom_idx, :, :m, :1] = jet_arr[:, :, :1]
        else:  # duchon
            centers = np.ascontiguousarray(spec["centers"])
            pts = np.ascontiguousarray(coords[atom_idx, :, :d])
            phi, jet, _penalty = rust.duchon_basis_with_jet(pts, centers, 2)
            phi_arr = np.asarray(phi, dtype=float)
            jet_arr = np.asarray(jet, dtype=float)
            m = phi_arr.shape[1]
            phi_stack[atom_idx, :, :m] = phi_arr
            jet_stack[atom_idx, :, :m, :d] = jet_arr[:, :, :d]
    return phi_stack, jet_stack


def _build_penalty_stack(atom_specs: list[dict[str, Any]], m_max: int) -> np.ndarray:
    k_atoms = len(atom_specs)
    stack = np.zeros((k_atoms, m_max, m_max), dtype=float)
    rust = rust_module()
    for atom_idx, spec in enumerate(atom_specs):
        if spec["kind"] == "periodic":
            t_zero = np.zeros(1, dtype=float)
            _phi, _jet, penalty = rust.periodic_basis_with_jet(t_zero, int(spec["n_harmonics"]))
            pen = np.asarray(penalty, dtype=float)
            m = pen.shape[0]
            stack[atom_idx, :m, :m] = pen
        else:
            centers = np.ascontiguousarray(spec["centers"])
            _phi, _jet, penalty = rust.duchon_basis_with_jet(centers, centers, 2)
            pen = np.asarray(penalty, dtype=float)
            m = pen.shape[0]
            stack[atom_idx, :m, :m] = pen
    return stack


def _pca_periodic_seed(x: np.ndarray, rng: np.random.Generator, atom_idx: int) -> np.ndarray:
    # PCA seeding: theta = atan2(z @ pc2, z @ pc1) / (2*pi), in [-0.5, 0.5].
    centered = x - x.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return rng.uniform(-0.5, 0.5, size=x.shape[0])
    if vt.shape[0] < 2:
        return rng.uniform(-0.5, 0.5, size=x.shape[0])
    pc1 = vt[0]
    pc2 = vt[1 + (atom_idx % max(1, vt.shape[0] - 1))] if atom_idx > 0 else vt[1]
    a = centered @ pc1
    b = centered @ pc2
    theta = np.arctan2(b, a) / (2.0 * np.pi)
    return theta


def _pca_axis(x: np.ndarray, axis: int, rng: np.random.Generator) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return rng.normal(scale=0.01, size=x.shape[0])
    pc = vt[axis % vt.shape[0]]
    proj = centered @ pc
    span = float(proj.max() - proj.min())
    if span <= 0.0:
        return np.zeros(x.shape[0])
    return (proj - proj.min()) / span - 0.5


def _pca_seed_coords(x: np.ndarray, d: int, rng: np.random.Generator) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    try:
        u, s, _vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return rng.normal(scale=0.01, size=(x.shape[0], d))
    k = min(d, u.shape[1], s.shape[0])
    coords = np.zeros((x.shape[0], d), dtype=float)
    coords[:, :k] = u[:, :k] * s[:k]
    # Normalize each column to roughly unit range.
    for col in range(d):
        span = float(coords[:, col].max() - coords[:, col].min())
        if span > 0.0:
            coords[:, col] = (coords[:, col] - coords[:, col].min()) / span - 0.5
    return coords


def _oos_reconstruct(model: ManifoldSAE, x_new: np.ndarray) -> np.ndarray:
    """Out-of-sample reconstruction reusing the trained Rust Newton kernel.

    Routes the held-out rows through the same ``sae_manifold_fit`` per-row
    Newton update used at training time, with the trained decoder
    coefficients ``B_k`` clamped (re-overwritten after every Rust step) so
    only the latent coords ``t_ik`` and the assignment logits move. The
    inferred ``phi_ik @ B_k`` reconstruction is the fitted matrix returned
    by the final Newton step — i.e. exactly the same math as training, just
    with the decoder held fixed.
    """
    rust = rust_module()
    n_new, p_out = x_new.shape
    k_atoms = len(model.atoms)
    dims = list(model._atom_dims)
    basis_sizes = list(model._basis_sizes)
    basis_kinds = list(model._basis_kinds)
    n_harmonics = list(model._n_harmonics)
    duchon_centers = list(model._duchon_centers)

    m_max = max(basis_sizes) if basis_sizes else 1
    d_max = max(dims) if dims else 1

    # Trained decoder coefficients, padded to (K, m_max, p_out).
    decoder_coefficients = np.zeros((k_atoms, m_max, p_out), dtype=float)
    for atom_idx, block in enumerate(model.decoder_blocks):
        m = basis_sizes[atom_idx]
        decoder_coefficients[atom_idx, :m, :] = block[:m, :]
    frozen_decoder = decoder_coefficients.copy()

    # Per-atom basis spec for the existing _build_basis_stack helper.
    atom_specs: list[dict[str, Any]] = []
    for atom_idx in range(k_atoms):
        kind = basis_kinds[atom_idx]
        if kind == "periodic":
            n_harm = max(1, n_harmonics[atom_idx])
            atom_specs.append(
                {"kind": "periodic", "n_harmonics": n_harm, "m": 1 + 2 * n_harm}
            )
        else:  # duchon
            centers = duchon_centers[atom_idx]
            if centers is None:
                raise RuntimeError(
                    f"OOS reconstruct: duchon atom {atom_idx} missing centers"
                )
            atom_specs.append(
                {"kind": "duchon", "centers": centers, "m": basis_sizes[atom_idx]}
            )

    # Smoothness penalty stack from trained atom specs.
    smooth_penalties_stack = _build_penalty_stack(atom_specs, m_max)

    # Seed coords for new rows: PCA in the new data's own frame.
    rng = np.random.default_rng(0)
    initial_coords = np.zeros((k_atoms, n_new, d_max), dtype=float)
    for atom_idx in range(k_atoms):
        d = dims[atom_idx]
        if d <= 0:
            continue
        spec = atom_specs[atom_idx]
        if spec["kind"] == "periodic":
            theta = _pca_periodic_seed(x_new, rng, atom_idx)
            initial_coords[atom_idx, :, 0] = theta
            for axis in range(1, d):
                initial_coords[atom_idx, :, axis] = _pca_axis(x_new, axis, rng)
        else:
            seed = _pca_seed_coords(x_new, d, rng)
            initial_coords[atom_idx, :, :d] = seed

    initial_logits = np.zeros((n_new, k_atoms), dtype=float)
    if k_atoms == 1 and model.assignment in {"ibp", "ibp_map"}:
        initial_logits[:, 0] = 4.0  # strong "atom on" prior; trained ibp behaves the same

    max_iter_total = max(1, int(np.asarray(model.coords[0]).shape[0] >= 0) * 50)
    assignment_kind = str(
        {"ibp": "ibp_map"}.get(model.assignment, model.assignment)
    )
    if assignment_kind == "gated":
        assignment_kind = "jumprelu"
    alpha_value = 1.0
    tau = 0.5

    last_fitted: np.ndarray | None = None
    for _ in range(max_iter_total):
        basis_values, basis_jacobian = _build_basis_stack(
            atom_specs, initial_coords, dims, m_max, d_max
        )
        # Freeze the decoder going in.
        decoder_coefficients[...] = frozen_decoder
        result = rust.sae_manifold_fit(
            np.ascontiguousarray(x_new),
            [spec["kind"] for spec in atom_specs],
            list(dims),
            np.ascontiguousarray(basis_values),
            np.ascontiguousarray(basis_jacobian),
            list(basis_sizes),
            np.ascontiguousarray(decoder_coefficients),
            np.ascontiguousarray(smooth_penalties_stack),
            np.ascontiguousarray(initial_logits),
            np.ascontiguousarray(initial_coords),
            float(alpha_value),
            float(tau),
            False,
            assignment_kind=assignment_kind,
            sparsity_strength=1.0,
            smoothness=1.0,
            max_iter=1,
            learning_rate=0.04,
            gumbel_schedule=None,
        )
        payload = dict(result)
        for atom_idx, atom in enumerate(payload["atoms"]):
            new_coords = np.asarray(atom["on_atom_coords_t"], dtype=float)
            d = dims[atom_idx]
            if d > 0 and new_coords.size:
                initial_coords[atom_idx, :, :d] = new_coords[:, :d]
        if "logits" in payload:
            new_logits = np.asarray(payload["logits"], dtype=float)
            if new_logits.shape == initial_logits.shape:
                initial_logits = np.ascontiguousarray(new_logits)
        last_fitted = np.asarray(payload["fitted"], dtype=float)

    if last_fitted is None:
        raise RuntimeError("OOS reconstruct produced no Newton step")

    # Final pass: reconstruct using frozen decoder + inferred coords to
    # eliminate any decoder drift accumulated inside the Rust step (the
    # Newton solver couples beta with t; we re-overwrote decoder above but
    # the *returned* fitted uses the post-step decoder).
    basis_values, _basis_jacobian = _build_basis_stack(
        atom_specs, initial_coords, dims, m_max, d_max
    )
    if model.assignment in {"ibp", "ibp_map"}:
        assignments = 1.0 / (1.0 + np.exp(-initial_logits))
    elif assignment_kind == "softmax":
        max_logits = initial_logits.max(axis=1, keepdims=True)
        weights = np.exp((initial_logits - max_logits) / max(tau, 1.0e-12))
        assignments = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)
    else:
        assignments = np.maximum(initial_logits, 0.0)
    if k_atoms == 1:
        assignments = np.ones((n_new, 1), dtype=float)
    fitted = np.zeros((n_new, p_out), dtype=float)
    for atom_idx in range(k_atoms):
        m = basis_sizes[atom_idx]
        phi = basis_values[atom_idx, :, :m]
        b = frozen_decoder[atom_idx, :m, :]
        fitted += assignments[:, atom_idx : atom_idx + 1] * (phi @ b)
    return fitted


def _wrap_payload(
    x: np.ndarray,
    payload: Mapping[str, Any],
    topology: str,
    assignment: str,
    penalties: list[str],
    atom_specs: list[dict[str, Any]],
    dims: list[int],
    basis_sizes: list[int],
) -> ManifoldSAE:
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
    low = SaeManifoldFitResult(
        atoms, len(atoms), {len(atoms): score}, {"winner": f"K={len(atoms)}"},
        fitted, assignments, coords, score,
    )
    n_harmonics = [int(spec.get("n_harmonics", 0)) for spec in atom_specs]
    duchon_centers: list[np.ndarray | None] = [
        spec.get("centers").copy() if spec["kind"] == "duchon" else None
        for spec in atom_specs
    ]
    basis_kinds = [spec["kind"] for spec in atom_specs]
    return ManifoldSAE(
        atoms=atoms,
        atom_topology=str(topology),
        assignment=str(assignment),
        primitive_names=["rust_module.sae_manifold_fit", *penalties],
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
        _basis_kinds=basis_kinds,
        _atom_dims=list(dims),
        _basis_sizes=list(basis_sizes),
        _n_harmonics=n_harmonics,
        _duchon_centers=duchon_centers,
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
        atom_basis = {
            "circle": "periodic",
            "periodic": "periodic",
            "sphere": "sphere",
            "euclidean": "duchon",
        }.get(str(atom_topology), atom_topology)
    raw = [atom_basis] * k_atoms if isinstance(atom_basis, str) else list(atom_basis)
    if len(raw) != k_atoms:
        raise ValueError("atom_basis must provide one basis per atom")
    return [str(value) for value in raw]


def _schedule_payload(schedule: GumbelTemperatureSchedule | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if schedule is None:
        return None
    if isinstance(schedule, GumbelTemperatureSchedule):
        return schedule.to_rust_descriptor()
    return dict(schedule)


def _schedule_tau_start(schedule: Any, default: float) -> float:
    return default if (payload := _schedule_payload(schedule)) is None else float(payload["tau_start"])


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    return float(rust_module().sae_manifold_reconstruction_r2(x, fitted))


__all__ = [
    "GumbelTemperatureSchedule",
    "ManifoldSAE",
    "SaeManifoldAtomFit",
    "SaeManifoldFitResult",
    "gumbel_geometric_schedule",
    "gumbel_linear_schedule",
    "gumbel_reciprocal_iter_schedule",
    "sae_manifold_fit",
]
