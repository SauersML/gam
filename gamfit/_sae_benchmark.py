"""Split-sensitive synthetic benchmark for manifold SAE recovery.

The benchmark plants multiple curved atoms in ambient random subspaces and
scores the part of recovery that reconstruction R^2 and routing MCC miss: can
the active atoms be split into their own coordinates when their decoder
subspaces are coherent and many tokens contain multiple active atoms?

``coverage`` is the per-token probability that each atom is active. Empty rows
are repaired by activating one random atom so every token has signal.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import Any

import numpy as np

N_TOKENS = 384
_SWEEP_COHERENCES = (0.35, 0.55, 0.75, 0.90)
_SWEEP_COVERAGES = (0.50, 0.70, 0.85, 0.95)
_SWEEP_TOPOLOGIES = ("circle", "sphere", "torus")


def run_benchmark(
    coherence: float,
    coverage: float,
    K: int,
    topology: str,
    seed: int,
) -> dict[str, Any]:
    """Run one coherence x coverage benchmark instance.

    Returns a JSON-friendly dictionary with the required split-sensitive
    metrics:

    * coordinate recovery R^2 after orthogonal Procrustes alignment,
    * ``sigma_min`` of the active planted tangent matrix,
    * cross-atom decoder cross-Gram energy.
    """

    topology = _normalize_topology(topology)
    _validate_inputs(coherence=coherence, coverage=coverage, K=K)
    rng = np.random.default_rng(seed)
    local_dim = _local_dim(topology)
    ambient_dim = local_dim * (K + 1) + 3

    decoders = _decoder_frames(
        rng=rng,
        ambient_dim=ambient_dim,
        local_dim=local_dim,
        K=K,
        coherence=coherence,
    )
    points, tangents = _sample_topology(topology, rng=rng, K=K, n=N_TOKENS)
    active = _active_mask(rng=rng, n=N_TOKENS, K=K, coverage=coverage)
    amplitudes = active * rng.uniform(0.75, 1.25, size=(N_TOKENS, K))
    signal = _superpose(points=points, decoders=decoders, amplitudes=amplitudes)
    projected = _oracle_project(signal=signal, decoders=decoders, topology=topology)

    coordinate = _coordinate_recovery(points=points, projected=projected, active=active)
    tangent = _active_tangent_scores(decoders=decoders, tangents=tangents, active=active)
    cross_gram = _decoder_cross_gram(decoders)

    per_atom_r2 = [item["r2"] for item in coordinate["per_atom"]]
    per_atom_rmse = [item["rmse"] for item in coordinate["per_atom"]]
    return {
        "config": {
            "coherence": float(coherence),
            "coverage": float(coverage),
            "K": int(K),
            "topology": topology,
            "seed": int(seed),
            "n_tokens": N_TOKENS,
            "ambient_dim": int(ambient_dim),
            "local_dim": int(local_dim),
        },
        "coverage": {
            "requested": float(coverage),
            "actual_active_fraction": float(active.mean()),
            "coactive_token_fraction": float(np.mean(active.sum(axis=1) >= 2)),
        },
        "coordinate_recovery": {
            "r2_mean": _finite_mean(per_atom_r2),
            "r2_worst": _finite_min(per_atom_r2),
            "rmse_mean": _finite_mean(per_atom_rmse),
            "per_atom": coordinate["per_atom"],
        },
        "active_tangent": tangent,
        "decoder_cross_gram": cross_gram,
    }


def sweep() -> list[dict[str, Any]]:
    """Tabulate the hard regime over topology, coherence, and coverage."""

    rows: list[dict[str, Any]] = []
    for topology in _SWEEP_TOPOLOGIES:
        for coherence in _SWEEP_COHERENCES:
            for coverage in _SWEEP_COVERAGES:
                result = run_benchmark(
                    coherence=coherence,
                    coverage=coverage,
                    K=3,
                    topology=topology,
                    seed=17,
                )
                rows.append(_summary_row(result))
    rows.sort(
        key=lambda row: (
            row["coord_r2_worst"],
            row["tangent_sigma_min_p10"],
            -row["decoder_cross_gram_max"],
        )
    )
    return rows


def format_markdown(rows: list[dict[str, Any]]) -> str:
    """Format ``sweep`` output as a compact markdown report."""

    header = (
        "# Split-Sensitive SAE Benchmark\n\n"
        "Synthetic superpositions use random decoder subspaces with controlled "
        "cross-atom coherence and per-token atom coverage. Scores are computed "
        "from oracle subspace projections, so failures indicate split ambiguity "
        "in the data geometry rather than optimizer noise.\n\n"
        "| topology | coh | cov | active | coactive | coord R2 worst | "
        "sigma_min p10 | crossGram max |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            f"| {row['topology']} | {row['coherence']:.2f} | "
            f"{row['coverage']:.2f} | {row['actual_active_fraction']:.2f} | "
            f"{row['coactive_token_fraction']:.2f} | "
            f"{row['coord_r2_worst']:.3f} | "
            f"{row['tangent_sigma_min_p10']:.4f} | "
            f"{row['decoder_cross_gram_max']:.3f} |"
        )
    return header + "\n".join(lines) + "\n"


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    cfg = result["config"]
    cov = result["coverage"]
    coord = result["coordinate_recovery"]
    tangent = result["active_tangent"]
    cross = result["decoder_cross_gram"]
    return {
        "topology": cfg["topology"],
        "coherence": cfg["coherence"],
        "coverage": cfg["coverage"],
        "actual_active_fraction": cov["actual_active_fraction"],
        "coactive_token_fraction": cov["coactive_token_fraction"],
        "coord_r2_mean": coord["r2_mean"],
        "coord_r2_worst": coord["r2_worst"],
        "coord_rmse_mean": coord["rmse_mean"],
        "tangent_sigma_min_min": tangent["sigma_min_min"],
        "tangent_sigma_min_p10": tangent["sigma_min_p10"],
        "tangent_sigma_min_median": tangent["sigma_min_median"],
        "decoder_cross_gram_mean": cross["mean_fro_normalized"],
        "decoder_cross_gram_max": cross["max_fro_normalized"],
    }


def _normalize_topology(topology: str) -> str:
    key = topology.strip().lower()
    aliases = {"s1": "circle", "s2": "sphere", "t2": "torus"}
    key = aliases.get(key, key)
    if key not in {"circle", "sphere", "torus"}:
        raise ValueError("topology must be one of: circle, sphere, torus")
    return key


def _validate_inputs(*, coherence: float, coverage: float, K: int) -> None:
    if not 0.0 <= coherence <= 0.99:
        raise ValueError("coherence must be in [0.0, 0.99]")
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0.0, 1.0]")
    if K < 1:
        raise ValueError("K must be positive")


def _local_dim(topology: str) -> int:
    if topology == "circle":
        return 2
    if topology == "sphere":
        return 3
    return 4


def _decoder_frames(
    *,
    rng: np.random.Generator,
    ambient_dim: int,
    local_dim: int,
    K: int,
    coherence: float,
) -> list[np.ndarray]:
    raw = rng.standard_normal((ambient_dim, local_dim * (K + 1)))
    basis, _ = np.linalg.qr(raw)
    shared = basis[:, :local_dim]
    frames = []
    shared_weight = sqrt(coherence)
    unique_weight = sqrt(max(1.0 - coherence, 0.0))
    for atom in range(K):
        start = local_dim * (atom + 1)
        unique = basis[:, start : start + local_dim]
        frames.append(shared_weight * shared + unique_weight * unique)
    return frames


def _sample_topology(
    topology: str,
    *,
    rng: np.random.Generator,
    K: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    if topology == "circle":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=(K, n))
        points = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        tangents = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)[..., None]
        return points, tangents

    if topology == "sphere":
        raw = rng.standard_normal((K, n, 3))
        points = raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
        tangents = np.empty((K, n, 3, 2), dtype=float)
        for atom in range(K):
            for row in range(n):
                tangents[atom, row] = _sphere_tangent_basis(points[atom, row])
        return points, tangents

    first = rng.uniform(0.0, 2.0 * np.pi, size=(K, n))
    second = rng.uniform(0.0, 2.0 * np.pi, size=(K, n))
    points = np.stack(
        [
            np.cos(first),
            np.sin(first),
            np.cos(second),
            np.sin(second),
        ],
        axis=-1,
    ) / sqrt(2.0)
    tangents = np.zeros((K, n, 4, 2), dtype=float)
    tangents[..., 0, 0] = -np.sin(first)
    tangents[..., 1, 0] = np.cos(first)
    tangents[..., 2, 1] = -np.sin(second)
    tangents[..., 3, 1] = np.cos(second)
    return points, tangents


def _sphere_tangent_basis(point: np.ndarray) -> np.ndarray:
    pivot = np.array([1.0, 0.0, 0.0])
    if abs(float(point @ pivot)) > 0.85:
        pivot = np.array([0.0, 1.0, 0.0])
    first = pivot - float(pivot @ point) * point
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(point, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return np.column_stack([first, second])


def _active_mask(
    *,
    rng: np.random.Generator,
    n: int,
    K: int,
    coverage: float,
) -> np.ndarray:
    active = rng.random((n, K)) < coverage
    empty = np.flatnonzero(active.sum(axis=1) == 0)
    if empty.size:
        active[empty, rng.integers(0, K, size=empty.size)] = True
    return active


def _superpose(
    *,
    points: np.ndarray,
    decoders: list[np.ndarray],
    amplitudes: np.ndarray,
) -> np.ndarray:
    n = amplitudes.shape[0]
    ambient_dim = decoders[0].shape[0]
    signal = np.zeros((n, ambient_dim), dtype=float)
    for atom, decoder in enumerate(decoders):
        signal += amplitudes[:, atom, None] * (points[atom] @ decoder.T)
    return signal


def _oracle_project(
    *,
    signal: np.ndarray,
    decoders: list[np.ndarray],
    topology: str,
) -> list[np.ndarray]:
    return [_project_to_topology(signal @ decoder, topology) for decoder in decoders]


def _project_to_topology(values: np.ndarray, topology: str) -> np.ndarray:
    if topology in {"circle", "sphere"}:
        return _normalize_rows(values)
    left = _normalize_rows(values[:, :2])
    right = _normalize_rows(values[:, 2:])
    return np.column_stack([left, right]) / sqrt(2.0)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    out = values / np.maximum(norms, 1e-12)
    degenerate = np.flatnonzero(norms[:, 0] <= 1e-12)
    if degenerate.size:
        out[degenerate] = 0.0
        out[degenerate, 0] = 1.0
    return out


def _coordinate_recovery(
    *,
    points: np.ndarray,
    projected: list[np.ndarray],
    active: np.ndarray,
) -> dict[str, Any]:
    per_atom = []
    for atom, prediction in enumerate(projected):
        mask = active[:, atom]
        r2, rmse = _procrustes_r2(prediction[mask], points[atom, mask])
        per_atom.append(
            {
                "atom": int(atom),
                "n_active": int(mask.sum()),
                "r2": float(r2),
                "rmse": float(rmse),
            }
        )
    return {"per_atom": per_atom}


def _procrustes_r2(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    if predicted.shape[0] < 2:
        return float("nan"), float("nan")
    u, _s, vt = np.linalg.svd(predicted.T @ truth, full_matrices=False)
    rotation = u @ vt
    aligned = predicted @ rotation
    residual = aligned - truth
    ss_res = float(np.sum(residual * residual))
    centered = truth - truth.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum(centered * centered))
    rmse = sqrt(ss_res / max(float(truth.shape[0]), 1.0))
    return 1.0 - ss_res / max(ss_tot, 1e-12), rmse


def _active_tangent_scores(
    *,
    decoders: list[np.ndarray],
    tangents: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    sigma_min = []
    for row in range(active.shape[0]):
        atoms = np.flatnonzero(active[row])
        if atoms.size < 2:
            continue
        columns = [decoders[atom] @ tangents[atom, row] for atom in atoms]
        tangent = np.concatenate(columns, axis=1)
        singular = np.linalg.svd(tangent, compute_uv=False)
        sigma_min.append(float(singular[-1]))
    values = np.asarray(sigma_min, dtype=float)
    if values.size == 0:
        return {
            "n_tokens": 0,
            "sigma_min_min": float("nan"),
            "sigma_min_p10": float("nan"),
            "sigma_min_median": float("nan"),
        }
    return {
        "n_tokens": int(values.size),
        "sigma_min_min": float(np.min(values)),
        "sigma_min_p10": float(np.quantile(values, 0.10)),
        "sigma_min_median": float(np.median(values)),
    }


def _decoder_cross_gram(decoders: list[np.ndarray]) -> dict[str, Any]:
    if len(decoders) < 2:
        return {
            "mean_fro_normalized": float("nan"),
            "max_fro_normalized": float("nan"),
            "pairs": [],
        }
    local_dim = decoders[0].shape[1]
    pairs = []
    values = []
    for left, right in combinations(range(len(decoders)), 2):
        cross = decoders[left].T @ decoders[right]
        fro = float(np.linalg.norm(cross, ord="fro") / sqrt(local_dim))
        values.append(fro)
        pairs.append({"left": int(left), "right": int(right), "fro_normalized": fro})
    return {
        "mean_fro_normalized": float(np.mean(values)),
        "max_fro_normalized": float(np.max(values)),
        "pairs": pairs,
    }


def _finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _finite_min(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


__all__ = ["format_markdown", "run_benchmark", "sweep"]
