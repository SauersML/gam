"""Regression tests for discovery-only variance-normalized shape projection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).parent / "sae" / "qwen_real_sae_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("qwen_real_sae_pipeline", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_PIPELINE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PIPELINE)


def _masked_ring_codes(rows: int, phase_offset: float) -> np.ndarray:
    index = np.arange(rows, dtype=np.float64)
    angle = 2.0 * np.pi * (index / rows + phase_offset)
    # A high-variance positive linear feature dominates PC1. Four nonnegative
    # half-wave features decode to a much lower-variance signed circle.
    line = 5.0 + 4.0 * np.sin(index * 1.618033988749895)
    cosine = 0.25 * np.cos(angle)
    sine = 0.25 * np.sin(angle)
    return np.column_stack(
        [
            line,
            np.maximum(cosine, 0.0),
            np.maximum(-cosine, 0.0),
            np.maximum(sine, 0.0),
            np.maximum(-sine, 0.0),
        ]
    )


def test_subspace_search_recovers_ring_below_dominant_linear_pc() -> None:
    discovery = _masked_ring_codes(240, 0.0)
    evaluation = _masked_ring_codes(180, 0.013)
    decoder = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    projected = _PIPELINE.group_coords_2d(
        discovery,
        evaluation,
        np.arange(5),
        decoder,
        max_rows=1000,
        max_search_pcs=3,
        seed=2262,
    )
    assert projected is not None
    coords, metadata = projected
    assert metadata["selected_pc_axes"] == [1, 2]
    assert metadata["searched_pcs"] == 3
    radii = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    assert radii.std() / radii.mean() < 0.02


def test_subspace_selection_cannot_see_evaluation_rows() -> None:
    discovery = _masked_ring_codes(240, 0.0)
    evaluation = _masked_ring_codes(180, 0.013)
    changed_evaluation = evaluation.copy()
    changed_evaluation[:, 0] *= 100.0
    decoder = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    baseline = _PIPELINE.group_coords_2d(
        discovery,
        evaluation,
        np.arange(5),
        decoder,
        max_rows=1000,
        max_search_pcs=3,
        seed=2262,
    )
    changed = _PIPELINE.group_coords_2d(
        discovery,
        changed_evaluation,
        np.arange(5),
        decoder,
        max_rows=1000,
        max_search_pcs=3,
        seed=2262,
    )
    assert baseline is not None and changed is not None
    _, baseline_metadata = baseline
    _, changed_metadata = changed
    assert baseline_metadata == changed_metadata
