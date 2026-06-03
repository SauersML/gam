"""Python normalization for SAE trust diagnostics.

Rust fit payloads expose trust diagnostics as plain dictionaries and arrays.
This module validates that payload boundary once and normalizes numeric values
so downstream APIs can rely on stable field names and types.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


_ATOM_DIAGNOSTIC_KEYS = (
    "trust_score",
    "sigma_min_tangent",
    "sigma_max_tangent",
    "tangent_condition_score",
    "mean_neighbor_coherence",
    "coherence_score",
    "topology_evidence_margin",
    "topology_margin_score",
    "coverage",
    "activation_frequency",
    "coverage_score",
    "typed_reconstruction_mse",
    "level0_reference_mse",
    "level0_residual_ratio",
    "level0_score",
    "untyped",
    "active_token_count",
)


def coerce_sae_trust_diagnostics(
    payload: Mapping[str, Any],
    *,
    n_atoms: int | None = None,
    assignments: Any | None = None,
    logits: Any | None = None,
) -> dict[str, Any]:
    """Validate and normalize the ``payload["diagnostics"]`` trust block.

    Parameters
    ----------
    payload
        Fit payload or serialized ``ManifoldSAE`` dictionary containing a
        ``diagnostics`` mapping with ``atom_trust``, ``atoms``, and
        ``level0_test`` fields.

    Returns
    -------
    dict
        ``{"atom_trust", "atoms", "level0_test"}`` where ``atom_trust`` is a
        1D ``float`` array and each atom record contains the full normalized
        trust diagnostic schema.
    """
    if "diagnostics" not in payload:
        return _default_sae_trust_diagnostics(
            payload,
            n_atoms=n_atoms,
            assignments=assignments,
            logits=logits,
        )

    diagnostics = dict(payload["diagnostics"])
    atoms = [dict(atom) for atom in diagnostics["atoms"]]
    trust = np.asarray(diagnostics["atom_trust"], dtype=float)
    if trust.ndim != 1:
        raise ValueError(
            f"SAE trust diagnostics require a 1D atom_trust array; got shape {trust.shape}"
        )
    if len(atoms) != trust.shape[0]:
        raise ValueError(
            "SAE trust diagnostics length mismatch: "
            f"{len(atoms)} atom records vs atom_trust length {trust.shape[0]}"
        )
    normalized_atoms: list[dict[str, Any]] = []
    for atom_idx, atom in enumerate(atoms):
        missing = [key for key in _ATOM_DIAGNOSTIC_KEYS if key not in atom]
        if missing:
            raise ValueError(
                f"SAE trust diagnostics atom {atom_idx} is missing keys: {missing}"
            )
        normalized = {key: atom[key] for key in _ATOM_DIAGNOSTIC_KEYS}
        normalized["trust_score"] = float(normalized["trust_score"])
        normalized["sigma_min_tangent"] = float(normalized["sigma_min_tangent"])
        normalized["sigma_max_tangent"] = float(normalized["sigma_max_tangent"])
        normalized["tangent_condition_score"] = float(
            normalized["tangent_condition_score"]
        )
        normalized["mean_neighbor_coherence"] = float(
            normalized["mean_neighbor_coherence"]
        )
        normalized["coherence_score"] = float(normalized["coherence_score"])
        normalized["topology_evidence_margin"] = float(
            normalized["topology_evidence_margin"]
        )
        normalized["topology_margin_score"] = float(
            normalized["topology_margin_score"]
        )
        normalized["coverage"] = float(normalized["coverage"])
        normalized["activation_frequency"] = float(
            normalized["activation_frequency"]
        )
        normalized["coverage_score"] = float(normalized["coverage_score"])
        normalized["typed_reconstruction_mse"] = float(
            normalized["typed_reconstruction_mse"]
        )
        normalized["level0_reference_mse"] = float(
            normalized["level0_reference_mse"]
        )
        normalized["level0_residual_ratio"] = float(
            normalized["level0_residual_ratio"]
        )
        normalized["level0_score"] = float(normalized["level0_score"])
        normalized["untyped"] = bool(normalized["untyped"])
        normalized["active_token_count"] = int(normalized["active_token_count"])
        normalized_atoms.append(normalized)
    return {
        "atom_trust": trust.copy(),
        "atoms": normalized_atoms,
        "level0_test": str(diagnostics["level0_test"]),
    }


def sae_trust_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Public spelling for :func:`coerce_sae_trust_diagnostics`."""
    return coerce_sae_trust_diagnostics(payload)


def atom_trust_scores(diagnostics: Mapping[str, Any]) -> np.ndarray:
    """Return the validated per-atom trust score vector from diagnostics."""
    trust = np.asarray(diagnostics["atom_trust"], dtype=float)
    if trust.ndim != 1:
        raise ValueError(f"atom_trust must be 1D; got shape {trust.shape}")
    return trust.copy()


def _default_sae_trust_diagnostics(
    payload: Mapping[str, Any],
    *,
    n_atoms: int | None,
    assignments: Any | None,
    logits: Any | None,
) -> dict[str, Any]:
    """Construct conservative diagnostics for older Rust payloads.

    Missing diagnostics mean the fit did not run the trust-score pipeline.
    We still return the public schema so callers can inspect the fit, but all
    unknown accuracy terms are marked untrusted rather than inferred.
    """

    inferred_atoms = _infer_atom_count(payload, n_atoms, assignments, logits)
    assigns = _assignment_matrix(payload, assignments, inferred_atoms)
    if assigns is None:
        coverage = np.zeros(inferred_atoms, dtype=float)
        active_counts = np.zeros(inferred_atoms, dtype=int)
    else:
        active = np.asarray(assigns, dtype=float) > 1.0e-9
        coverage = active.mean(axis=0) if active.size else np.zeros(inferred_atoms)
        active_counts = active.sum(axis=0).astype(int) if active.size else np.zeros(inferred_atoms, dtype=int)

    atoms: list[dict[str, Any]] = []
    for atom_idx in range(inferred_atoms):
        cov = float(coverage[atom_idx]) if atom_idx < coverage.shape[0] else 0.0
        count = int(active_counts[atom_idx]) if atom_idx < active_counts.shape[0] else 0
        atoms.append({
            "trust_score": 0.0,
            "sigma_min_tangent": 0.0,
            "sigma_max_tangent": 0.0,
            "tangent_condition_score": 0.0,
            "mean_neighbor_coherence": 1.0,
            "coherence_score": 0.0,
            "topology_evidence_margin": 0.0,
            "topology_margin_score": 0.0,
            "coverage": cov,
            "activation_frequency": cov,
            "coverage_score": cov,
            "typed_reconstruction_mse": 0.0,
            "level0_reference_mse": 0.0,
            "level0_residual_ratio": float("inf"),
            "level0_score": 0.0,
            "untyped": True,
            "active_token_count": count,
        })
    return {
        "atom_trust": np.zeros(inferred_atoms, dtype=float),
        "atoms": atoms,
        "level0_test": "not_run",
    }


def _infer_atom_count(
    payload: Mapping[str, Any],
    n_atoms: int | None,
    assignments: Any | None,
    logits: Any | None,
) -> int:
    if n_atoms is not None:
        return int(n_atoms)
    atoms = payload.get("atoms")
    if atoms is not None:
        return len(atoms)
    for value in (assignments, logits, payload.get("assignments_z"), payload.get("assignments")):
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim == 2:
            return int(arr.shape[1])
    return 0


def _assignment_matrix(
    payload: Mapping[str, Any],
    assignments: Any | None,
    n_atoms: int,
) -> np.ndarray | None:
    value = assignments
    if value is None:
        value = payload.get("assignments_z", payload.get("assignments"))
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != int(n_atoms):
        return None
    return arr


__all__ = [
    "atom_trust_scores",
    "coerce_sae_trust_diagnostics",
    "sae_trust_diagnostics",
]
