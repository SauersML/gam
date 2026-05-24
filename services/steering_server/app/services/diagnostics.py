from __future__ import annotations

from app.models import ConceptTarget


def build_diagnostics(concept_targets: list[ConceptTarget], layer: int) -> dict[str, object]:
    anchor_curvature: list[dict[str, float | str]] = []
    locality: list[dict[str, float | str]] = []
    instability: list[dict[str, float | str]] = []

    for index, concept in enumerate(concept_targets):
        anchor_count = max(len(concept.anchors), 1)
        curvature = round(abs(concept.target) * (layer + 1) / anchor_count, 6)
        variance = round((1.0 + abs(concept.target)) / max(concept.locality_weight, 0.001), 6)
        risk = round(curvature * 0.65 + variance * 0.35 + index * 0.01, 6)
        anchor_curvature.append({"concept": concept.name, "curvature": curvature})
        locality.append({"concept": concept.name, "variance": variance, "locality_weight": concept.locality_weight})
        instability.append({"concept": concept.name, "instability": risk})

    instability.sort(key=lambda row: float(row["instability"]), reverse=True)
    return {
        "anchor_curvature": anchor_curvature,
        "variance_vs_concept_locality": locality,
        "predicted_instability_ranking": instability,
        "notes": [
            "diagnostics are deterministic estimates from registered anchors",
            "stub: replace with manifold curvature and variance probes from the fitted LLM backend",
        ],
    }
