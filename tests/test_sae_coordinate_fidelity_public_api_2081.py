"""#2081 public coordinate-fidelity API and serialization plumbing."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gamfit")
from gamfit._sae_manifold import ManifoldSAE  # noqa: E402


_GOLDEN_FULL = (
    Path(__file__).resolve().parent / "fixtures" / "manifold_sae" / "golden_full.json"
)
_GOLDEN_PAYLOAD = json.loads(_GOLDEN_FULL.read_text())


def _coordinate_fidelity() -> dict:
    return {
        "note": "test coordinate-fidelity payload",
        "atoms": [
            {
                "atom": 0,
                "topology": "circle",
                "uniformity_statistic": 0.08,
                "uniformity_p_value": 0.41,
                "arclength_defect": 0.02,
                "n_coords": 4,
                "verdict": "recoverable_via_arclength",
                "certified": True,
                "coords_u_arc": [0.0, 0.25, 0.5, 0.75],
                "raw_arclength_defect_rms": 0.12,
                "raw_arclength_defect_max": 0.2,
                "min_speed_over_mean": 0.8,
                "max_speed_over_mean": 1.2,
                "log_speed_rms": 0.05,
            },
            {
                "atom": 1,
                "topology": "circle",
                # The Rust-owned persisted surface is JSON. Undefined metrics
                # for a degenerate chart are represented as null, not NaN.
                "uniformity_statistic": None,
                "uniformity_p_value": None,
                "arclength_defect": None,
                "n_coords": 4,
                "verdict": "degenerate",
                "certified": False,
                "coords_u_arc": None,
                "raw_arclength_defect_rms": None,
                "raw_arclength_defect_max": None,
                "min_speed_over_mean": 0.0,
                "max_speed_over_mean": 2.0,
                "log_speed_rms": None,
            },
        ],
    }


def _certificates() -> dict:
    return {
        "overall": "insufficient",
        "overall_certified": False,
        "claims": {
            "coordinate-fidelity": {
                "claim": "eligible d=1 SAE charts report faithful coordinates",
                "verdict": "insufficient",
                "certified": False,
                "evidence": {
                    "atom_count": 2,
                    "eligible_d1_atoms": 2,
                    "certified_d1_atoms": 1,
                    "degenerate_d1_atoms": 1,
                    "worst_coordinate_verdict": "degenerate",
                },
            }
        },
    }


def _model() -> ManifoldSAE:
    payload = deepcopy(_GOLDEN_PAYLOAD)
    coordinate_fidelity = _coordinate_fidelity()
    payload["coordinate_fidelity"] = coordinate_fidelity
    payload["certificates"] = _certificates()
    payload["atoms"][0]["coords_u_arc"] = deepcopy(
        coordinate_fidelity["atoms"][0]["coords_u_arc"]
    )
    payload["atoms"][1]["coords_u_arc"] = None
    return ManifoldSAE.from_dict(payload)


def test_coordinate_fidelity_report_and_gated_angle_reader() -> None:
    model = _model()

    rows = model.coordinate_fidelity_report()
    assert len(rows) == 2
    assert rows[0]["verdict"] == "recoverable_via_arclength"
    np.testing.assert_allclose(model.atom_angle_coordinate(0), [0.0, 0.25, 0.5, 0.75])

    with pytest.raises(ValueError, match="degenerate"):
        model.atom_angle_coordinate(1)


def test_coordinate_fidelity_round_trips_through_dict() -> None:
    restored = ManifoldSAE.from_dict(_model().to_dict())

    assert restored.certificates is not None
    assert "coordinate-fidelity" in restored.certificates["claims"]
    assert restored.certificates["claims"]["coordinate-fidelity"]["verdict"] == "insufficient"
    assert restored.atoms[0].coords_u_arc is not None
    np.testing.assert_allclose(restored.atoms[0].coords_u_arc, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(restored.atom_angle_coordinate(0), [0.0, 0.25, 0.5, 0.75])
