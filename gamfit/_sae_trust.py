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
        # The Rust runtime does not (yet) emit the trust-diagnostic block —
        # no released pyffi ever has. A missing optional diagnostic must not
        # brick the primary fit API; return the explicit empty block, and the
        # per-atom accessors raise with a precise message on use instead.
        return {"atom_trust": np.zeros(0, dtype=float), "atoms": [], "level0_test": None}

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


__all__ = [
    "atom_trust_scores",
    "coerce_sae_trust_diagnostics",
    "sae_trust_diagnostics",
]
