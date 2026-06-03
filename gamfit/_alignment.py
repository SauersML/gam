"""Alignment diagnostics for independently fitted manifold SAE dictionaries.

The public entry points are :func:`align` and :func:`align_fits`. They compare two fitted
ManifoldSAE-like objects in the quotient induced by atom permutation,
orthogonal coordinate gauge, and decoder subspace basis choice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np


_TOPOLOGY_ALIASES: dict[str, str] = {
    "periodic": "circle",
    "fourier": "circle",
    "circle": "circle",
    "duchon": "euclidean",
    "euclidean_patch": "euclidean",
    "euclidean": "euclidean",
    "line": "euclidean",
    "sphere": "sphere",
    "torus": "torus",
    "cylinder": "cylinder",
}


@dataclass(frozen=True, slots=True)
class AtomAlignment:
    """Diagnostics for one Hungarian-matched atom pair."""

    atom_a: int
    atom_b: int
    topology_a: str
    topology_b: str
    topology_flipped: bool
    evidence_margin: float
    decoder_rank_a: int
    decoder_rank_b: int
    principal_angles: np.ndarray
    grassmann_distance: float
    normalized_grassmann_distance: float
    procrustes_distance: float
    procrustes_rmsd: float
    procrustes_scale: float
    procrustes_rotation: np.ndarray
    assignment_cosine: float
    match_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_a": self.atom_a,
            "atom_b": self.atom_b,
            "topology_a": self.topology_a,
            "topology_b": self.topology_b,
            "topology_flipped": self.topology_flipped,
            "evidence_margin": self.evidence_margin,
            "decoder_rank_a": self.decoder_rank_a,
            "decoder_rank_b": self.decoder_rank_b,
            "principal_angles": self.principal_angles.tolist(),
            "grassmann_distance": self.grassmann_distance,
            "normalized_grassmann_distance": self.normalized_grassmann_distance,
            "procrustes_distance": self.procrustes_distance,
            "procrustes_rmsd": self.procrustes_rmsd,
            "procrustes_scale": self.procrustes_scale,
            "procrustes_rotation": self.procrustes_rotation.tolist(),
            "assignment_cosine": self.assignment_cosine,
            "match_cost": self.match_cost,
        }


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """Permutation/gauge-invariant reproducibility comparison for two fits."""

    pairs: list[AtomAlignment]
    cost_matrix: np.ndarray
    assignment: list[tuple[int, int]]
    unmatched_a: tuple[int, ...]
    unmatched_b: tuple[int, ...]
    summary: dict[str, float]
    flip_rate_by_evidence_margin: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [pair.to_dict() for pair in self.pairs],
            "cost_matrix": self.cost_matrix.tolist(),
            "assignment": [list(pair) for pair in self.assignment],
            "unmatched_a": list(self.unmatched_a),
            "unmatched_b": list(self.unmatched_b),
            "summary": dict(self.summary),
            "flip_rate_by_evidence_margin": dict(self.flip_rate_by_evidence_margin),
        }


@dataclass(frozen=True, slots=True)
class _AtomView:
    decoder: np.ndarray
    coords: np.ndarray
    assignments: np.ndarray
    topology: str
    evidence_margin: float


def align_fits(fit_a: Any, fit_b: Any) -> AlignmentResult:
    """Align two ManifoldSAE-like fits and report reproducibility metrics.

    Matching uses ``scipy.optimize.linear_sum_assignment`` on a cost equal to
    normalized decoder-subspace Grassmann distance plus a 0/1 topology mismatch
    penalty. Matched atom diagnostics then include raw principal angles,
    Grassmann distance, Procrustes-aligned per-token coordinate error,
    assignment-vector cosine similarity, topology-flip flags, and aggregate
    reproducibility summaries.

    Parameters
    ----------
    fit_a, fit_b
        ``gamfit.ManifoldSAE`` instances, their ``to_dict()`` payloads, or
        compatible objects carrying per-atom decoder blocks, coordinates,
        assignments, and topology labels.
    """

    atoms_a = _extract_atoms(fit_a, "fit_a")
    atoms_b = _extract_atoms(fit_b, "fit_b")
    if not atoms_a:
        raise ValueError("fit_a contains no atoms")
    if not atoms_b:
        raise ValueError("fit_b contains no atoms")

    cost_matrix = np.empty((len(atoms_a), len(atoms_b)), dtype=float)
    for i, atom_a in enumerate(atoms_a):
        for j, atom_b in enumerate(atoms_b):
            subspace = _decoder_subspace_metrics(atom_a.decoder, atom_b.decoder)
            topology_mismatch = float(atom_a.topology != atom_b.topology)
            decoder_tiebreak = _decoder_matrix_tiebreak(atom_a.decoder, atom_b.decoder)
            cost_matrix[i, j] = (
                subspace["normalized_distance"]
                + topology_mismatch
                + 1.0e-9 * decoder_tiebreak
            )

    row_ind, col_ind = _linear_sum_assignment(cost_matrix)
    assignment = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
    pairs = [
        _align_pair(i, j, atoms_a[i], atoms_b[j], float(cost_matrix[i, j]))
        for i, j in assignment
    ]
    matched_a = {i for i, _ in assignment}
    matched_b = {j for _, j in assignment}
    unmatched_a = tuple(i for i in range(len(atoms_a)) if i not in matched_a)
    unmatched_b = tuple(j for j in range(len(atoms_b)) if j not in matched_b)
    flip_by_margin = _flip_rate_by_margin(pairs)

    return AlignmentResult(
        pairs=pairs,
        cost_matrix=cost_matrix,
        assignment=assignment,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        summary=_summary(pairs, len(atoms_a), len(atoms_b), len(unmatched_a), len(unmatched_b)),
        flip_rate_by_evidence_margin=flip_by_margin,
    )


def align(fit_a: Any, fit_b: Any) -> AlignmentResult:
    """Public alias for :func:`align_fits`.

    ``gamfit.align(a, b)`` returns an :class:`AlignmentResult`; call
    ``result.to_dict()`` when a JSON-friendly payload is needed.
    """
    return align_fits(fit_a, fit_b)


def _align_pair(
    atom_a_index: int,
    atom_b_index: int,
    atom_a: _AtomView,
    atom_b: _AtomView,
    match_cost: float,
) -> AtomAlignment:
    subspace = _decoder_subspace_metrics(atom_a.decoder, atom_b.decoder)
    procrustes = _procrustes_metrics(atom_a.coords, atom_b.coords)
    margin = _pair_evidence_margin(atom_a.evidence_margin, atom_b.evidence_margin)
    return AtomAlignment(
        atom_a=atom_a_index,
        atom_b=atom_b_index,
        topology_a=atom_a.topology,
        topology_b=atom_b.topology,
        topology_flipped=atom_a.topology != atom_b.topology,
        evidence_margin=margin,
        decoder_rank_a=int(subspace["rank_a"]),
        decoder_rank_b=int(subspace["rank_b"]),
        principal_angles=np.asarray(subspace["principal_angles"], dtype=float),
        grassmann_distance=float(subspace["distance"]),
        normalized_grassmann_distance=float(subspace["normalized_distance"]),
        procrustes_distance=float(procrustes["distance"]),
        procrustes_rmsd=float(procrustes["rmsd"]),
        procrustes_scale=float(procrustes["scale"]),
        procrustes_rotation=np.asarray(procrustes["rotation"], dtype=float),
        assignment_cosine=_assignment_cosine(atom_a.assignments, atom_b.assignments),
        match_cost=float(match_cost),
    )


def _extract_atoms(fit: Any, label: str) -> list[_AtomView]:
    atoms_obj = _get(fit, "atoms", None)
    if atoms_obj is None:
        raise TypeError(f"{label} must expose an 'atoms' sequence")
    atoms = list(atoms_obj)

    fit_decoders = _optional_sequence(_get(fit, "decoder_blocks", None))
    fit_coords = _optional_sequence(_get(fit, "coords", None))
    fit_assignments = _optional_array(_get(fit, "assignments", None))
    fit_topologies = _topologies_for_fit(fit, len(atoms))
    fit_margins = _optional_array(
        _first_present(
            fit,
            ("evidence_margins", "topology_evidence_margins", "topology_margins"),
        )
    )

    views: list[_AtomView] = []
    for index, atom in enumerate(atoms):
        decoder = _extract_decoder(atom, fit_decoders, index, label)
        coords = _extract_coords(atom, fit_coords, index, label)
        assignments = _extract_assignments(atom, fit_assignments, index, label)
        topology = _extract_topology(atom, fit_topologies[index])
        margin = _extract_margin(atom, fit_margins, index)
        views.append(
            _AtomView(
                decoder=decoder,
                coords=coords,
                assignments=assignments,
                topology=topology,
                evidence_margin=margin,
            )
        )
    _validate_token_lengths(views, label)
    return views


def _extract_decoder(
    atom: Any,
    fit_decoders: Sequence[Any] | None,
    index: int,
    label: str,
) -> np.ndarray:
    value = _first_present(
        atom,
        ("decoder_coefficients", "decoder_B", "decoder", "decoder_block"),
    )
    if value is None and fit_decoders is not None and index < len(fit_decoders):
        value = fit_decoders[index]
    if value is None:
        raise TypeError(f"{label}.atoms[{index}] has no decoder coefficients")
    decoder = np.asarray(value, dtype=float)
    if decoder.ndim != 2:
        raise ValueError(
            f"{label}.atoms[{index}] decoder must be 2D; got shape {decoder.shape}"
        )
    return np.ascontiguousarray(decoder)


def _extract_coords(
    atom: Any,
    fit_coords: Sequence[Any] | None,
    index: int,
    label: str,
) -> np.ndarray:
    value = _get(atom, "coords", None)
    if value is None and fit_coords is not None and index < len(fit_coords):
        value = fit_coords[index]
    if value is None:
        raise TypeError(f"{label}.atoms[{index}] has no per-token coordinates")
    coords = np.asarray(value, dtype=float)
    if coords.ndim != 2:
        raise ValueError(
            f"{label}.atoms[{index}] coords must be 2D; got shape {coords.shape}"
        )
    return np.ascontiguousarray(coords)


def _extract_assignments(
    atom: Any,
    fit_assignments: np.ndarray | None,
    index: int,
    label: str,
) -> np.ndarray:
    value = _first_present(atom, ("assignments", "assignments_z", "gates"))
    if value is None and fit_assignments is not None:
        if fit_assignments.ndim != 2 or index >= fit_assignments.shape[1]:
            raise ValueError(
                f"{label}.assignments must have atom columns; got shape "
                f"{fit_assignments.shape}"
            )
        value = fit_assignments[:, index]
    if value is None:
        raise TypeError(f"{label}.atoms[{index}] has no per-token assignments")
    assignments = np.asarray(value, dtype=float)
    if assignments.ndim != 1:
        raise ValueError(
            f"{label}.atoms[{index}] assignments must be 1D; got shape "
            f"{assignments.shape}"
        )
    return np.ascontiguousarray(assignments)


def _topologies_for_fit(fit: Any, n_atoms: int) -> list[str]:
    values = _optional_sequence(_get(fit, "atom_topologies", None))
    if values is not None:
        if len(values) != n_atoms:
            raise ValueError(
                f"fit atom_topologies length {len(values)} does not match "
                f"{n_atoms} atoms"
            )
        return [_canonical_topology(value) for value in values]

    scalar = _get(fit, "atom_topology", None)
    if scalar is not None and str(scalar) != "mixed":
        return [_canonical_topology(scalar) for _ in range(n_atoms)]
    return ["unknown" for _ in range(n_atoms)]


def _extract_topology(atom: Any, fit_topology: str) -> str:
    value = _first_present(atom, ("topology", "atom_topology", "basis", "basis_kind"))
    if value is None:
        return fit_topology
    return _canonical_topology(value)


def _extract_margin(atom: Any, fit_margins: np.ndarray | None, index: int) -> float:
    value = _first_present(
        atom,
        ("evidence_margin", "topology_evidence_margin", "topology_margin"),
    )
    if value is None and fit_margins is not None and index < fit_margins.shape[0]:
        value = fit_margins[index]
    if value is None:
        return float("nan")
    return float(value)


def _validate_token_lengths(atoms: Sequence[_AtomView], label: str) -> None:
    n = atoms[0].coords.shape[0]
    for index, atom in enumerate(atoms):
        if atom.coords.shape[0] != n:
            raise ValueError(
                f"{label}.atoms[{index}] coords have {atom.coords.shape[0]} rows; "
                f"expected {n}"
            )
        if atom.assignments.shape[0] != n:
            raise ValueError(
                f"{label}.atoms[{index}] assignments have "
                f"{atom.assignments.shape[0]} rows; expected {n}"
            )


def _decoder_subspace_metrics(decoder_a: np.ndarray, decoder_b: np.ndarray) -> dict[str, Any]:
    basis_a = _decoder_ambient_basis(decoder_a)
    basis_b = _decoder_ambient_basis(decoder_b)
    if basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError(
            "decoder ambient dimensions differ: "
            f"{basis_a.shape[0]} vs {basis_b.shape[0]}"
        )
    angles = _principal_angles(basis_a, basis_b)
    distance = float(np.linalg.norm(angles))
    rank = max(basis_a.shape[1], basis_b.shape[1])
    normalizer = np.sqrt(rank) * (np.pi / 2.0) if rank else 1.0
    return {
        "principal_angles": angles,
        "distance": distance,
        "normalized_distance": float(distance / normalizer),
        "rank_a": basis_a.shape[1],
        "rank_b": basis_b.shape[1],
    }


def _decoder_ambient_basis(decoder: np.ndarray) -> np.ndarray:
    ambient_vectors = np.asarray(decoder, dtype=float).T
    if ambient_vectors.ndim != 2:
        raise ValueError(f"decoder must be 2D; got shape {ambient_vectors.shape}")
    if ambient_vectors.size == 0:
        return np.empty((ambient_vectors.shape[0], 0), dtype=float)
    u, singular_values, _ = np.linalg.svd(ambient_vectors, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((ambient_vectors.shape[0], 0), dtype=float)
    tol = np.finfo(float).eps * max(ambient_vectors.shape) * float(singular_values[0])
    rank = int(np.sum(singular_values > tol))
    return np.ascontiguousarray(u[:, :rank])


def _principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    rank_a = basis_a.shape[1]
    rank_b = basis_b.shape[1]
    common_rank = min(rank_a, rank_b)
    if common_rank:
        singular_values = np.linalg.svd(
            basis_a.T @ basis_b,
            full_matrices=False,
            compute_uv=False,
        )
        angles = np.arccos(np.clip(singular_values[:common_rank], -1.0, 1.0))
    else:
        angles = np.empty(0, dtype=float)
    if rank_a != rank_b:
        missing = np.full(abs(rank_a - rank_b), np.pi / 2.0, dtype=float)
        angles = np.concatenate([angles, missing])
    return angles


def _procrustes_metrics(coords_a: np.ndarray, coords_b: np.ndarray) -> dict[str, Any]:
    if coords_a.shape[0] != coords_b.shape[0]:
        raise ValueError(
            "per-token coordinate rows differ: "
            f"{coords_a.shape[0]} vs {coords_b.shape[0]}"
        )
    a = _pad_coordinate_dim(np.asarray(coords_a, dtype=float), coords_b.shape[1])
    b = _pad_coordinate_dim(np.asarray(coords_b, dtype=float), coords_a.shape[1])
    a_centered = a - np.mean(a, axis=0, keepdims=True)
    b_centered = b - np.mean(b, axis=0, keepdims=True)
    a_norm = float(np.linalg.norm(a_centered))
    b_norm = float(np.linalg.norm(b_centered))
    dim = a_centered.shape[1]
    rotation = np.eye(dim, dtype=float)

    if a_norm == 0.0 and b_norm == 0.0:
        return {"distance": 0.0, "rmsd": 0.0, "scale": 1.0, "rotation": rotation}
    if a_norm == 0.0 or b_norm == 0.0:
        distance = 1.0
        return {
            "distance": distance,
            "rmsd": float(distance / np.sqrt(max(1, a_centered.size))),
            "scale": 0.0,
            "rotation": rotation,
        }

    a_unit = a_centered / a_norm
    b_unit = b_centered / b_norm
    rotation, scale = _orthogonal_procrustes(a_unit, b_unit)
    residual = a_unit @ rotation - b_unit
    distance = float(np.linalg.norm(residual))
    rmsd = float(np.sqrt(np.mean(residual * residual)))
    return {
        "distance": distance,
        "rmsd": rmsd,
        "scale": float(scale),
        "rotation": rotation,
    }


def _pad_coordinate_dim(coords: np.ndarray, other_dim: int) -> np.ndarray:
    dim = max(coords.shape[1], int(other_dim))
    if coords.shape[1] == dim:
        return coords
    out = np.zeros((coords.shape[0], dim), dtype=float)
    out[:, : coords.shape[1]] = coords
    return out


def _decoder_matrix_tiebreak(decoder_a: np.ndarray, decoder_b: np.ndarray) -> float:
    a = np.asarray(decoder_a, dtype=float)
    b = np.asarray(decoder_b, dtype=float)
    rows = max(a.shape[0], b.shape[0])
    cols = max(a.shape[1], b.shape[1])
    ap = np.zeros((rows, cols), dtype=float)
    bp = np.zeros((rows, cols), dtype=float)
    ap[: a.shape[0], : a.shape[1]] = a
    bp[: b.shape[0], : b.shape[1]] = b
    denom = max(float(np.linalg.norm(ap)), float(np.linalg.norm(bp)), 1.0e-12)
    return float(np.linalg.norm(ap - bp) / denom)


def _linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError:
        return _linear_sum_assignment_small(cost_matrix)
    return linear_sum_assignment(cost_matrix)


def _linear_sum_assignment_small(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D; got shape {cost.shape}")
    n_rows, n_cols = cost.shape
    if min(n_rows, n_cols) > 8:
        raise ModuleNotFoundError(
            "gamfit.align requires scipy for assignments larger than 8 atoms"
        )
    if n_rows <= n_cols:
        rows = tuple(range(n_rows))
        best_cols: tuple[int, ...] | None = None
        best_cost = float("inf")
        for cols in permutations(range(n_cols), n_rows):
            total = float(sum(cost[row, col] for row, col in zip(rows, cols)))
            if total < best_cost:
                best_cost = total
                best_cols = tuple(int(col) for col in cols)
        return np.asarray(rows, dtype=int), np.asarray(best_cols or (), dtype=int)

    best_rows: tuple[int, ...] | None = None
    best_cols: tuple[int, ...] | None = None
    best_cost = float("inf")
    cols = tuple(range(n_cols))
    for rows_subset in combinations(range(n_rows), n_cols):
        for rows in permutations(rows_subset):
            total = float(sum(cost[row, col] for row, col in zip(rows, cols)))
            if total < best_cost:
                best_cost = total
                best_rows = tuple(int(row) for row in rows)
                best_cols = cols
    return np.asarray(best_rows or (), dtype=int), np.asarray(best_cols or (), dtype=int)


def _orthogonal_procrustes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        from scipy.linalg import orthogonal_procrustes
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gamfit.align requires scipy. Install the test/sklearn/scipy stack "
            "before using SAE alignment diagnostics."
        ) from exc
    return orthogonal_procrustes(a, b)


def _assignment_cosine(assignments_a: np.ndarray, assignments_b: np.ndarray) -> float:
    if assignments_a.shape[0] != assignments_b.shape[0]:
        raise ValueError(
            "assignment vector lengths differ: "
            f"{assignments_a.shape[0]} vs {assignments_b.shape[0]}"
        )
    norm = float(np.linalg.norm(assignments_a) * np.linalg.norm(assignments_b))
    if norm == 0.0:
        return 1.0
    return float(np.dot(assignments_a, assignments_b) / norm)


def _summary(
    pairs: Sequence[AtomAlignment],
    n_atoms_a: int,
    n_atoms_b: int,
    n_unmatched_a: int,
    n_unmatched_b: int,
) -> dict[str, float]:
    return {
        "n_atoms_a": float(n_atoms_a),
        "n_atoms_b": float(n_atoms_b),
        "n_matched": float(len(pairs)),
        "n_unmatched_a": float(n_unmatched_a),
        "n_unmatched_b": float(n_unmatched_b),
        "mean_match_cost": _mean(pair.match_cost for pair in pairs),
        "mean_grassmann_distance": _mean(pair.grassmann_distance for pair in pairs),
        "mean_normalized_grassmann_distance": _mean(
            pair.normalized_grassmann_distance for pair in pairs
        ),
        "mean_procrustes_distance": _mean(pair.procrustes_distance for pair in pairs),
        "mean_procrustes_rmsd": _mean(pair.procrustes_rmsd for pair in pairs),
        "mean_assignment_cosine": _mean(pair.assignment_cosine for pair in pairs),
        "topology_flip_rate": _mean(float(pair.topology_flipped) for pair in pairs),
        "perfect_topology_match": float(
            all(not pair.topology_flipped for pair in pairs)
        ),
    }


def _flip_rate_by_margin(pairs: Sequence[AtomAlignment]) -> dict[str, float]:
    thresholds = (0.0, 1.0, 2.0, 5.0)
    out: dict[str, float] = {}
    margins = np.asarray([pair.evidence_margin for pair in pairs], dtype=float)
    flips = np.asarray([pair.topology_flipped for pair in pairs], dtype=bool)
    for threshold in thresholds:
        mask = np.isfinite(margins) & (margins >= threshold)
        key = f">={threshold:g}"
        out[key] = float(np.mean(flips[mask])) if np.any(mask) else float("nan")
        out[f"{key}:n"] = float(np.sum(mask))
    return out


def _pair_evidence_margin(margin_a: float, margin_b: float) -> float:
    finite = [margin for margin in (margin_a, margin_b) if np.isfinite(margin)]
    if not finite:
        return float("nan")
    return float(min(finite))


def _mean(values: Any) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _canonical_topology(value: Any) -> str:
    text = str(value).strip().lower()
    return _TOPOLOGY_ALIASES.get(text, text)


def _get(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first_present(obj: Any, keys: Sequence[str]) -> Any:
    for key in keys:
        value = _get(obj, key, None)
        if value is not None:
            return value
    return None


def _optional_sequence(value: Any) -> Sequence[Any] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence):
        raise TypeError(f"expected a sequence, got {type(value).__name__}")
    return value


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


__all__ = ["AlignmentResult", "AtomAlignment", "align", "align_fits"]
