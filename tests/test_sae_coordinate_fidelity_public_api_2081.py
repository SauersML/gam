"""#2081 public coordinate-fidelity API and serialization plumbing."""

from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import ManifoldSAE, SaeManifoldAtomFit, SaeManifoldFitResult  # noqa: E402


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
                "coords_u_arc": np.array([0.0, 0.25, 0.5, 0.75]),
                "raw_arclength_defect_rms": 0.12,
                "raw_arclength_defect_max": 0.2,
                "min_speed_over_mean": 0.8,
                "max_speed_over_mean": 1.2,
                "log_speed_rms": 0.05,
            },
            {
                "atom": 1,
                "topology": "circle",
                "uniformity_statistic": float("nan"),
                "uniformity_p_value": float("nan"),
                "arclength_defect": float("nan"),
                "n_coords": 4,
                "verdict": "degenerate",
                "certified": False,
                "coords_u_arc": None,
                "raw_arclength_defect_rms": float("nan"),
                "raw_arclength_defect_max": float("nan"),
                "min_speed_over_mean": 0.0,
                "max_speed_over_mean": 2.0,
                "log_speed_rms": float("nan"),
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


def _trust_atom() -> dict:
    return {
        "trust_score": 1.0,
        "sigma_min_tangent": 1.0,
        "sigma_max_tangent": 1.0,
        "tangent_condition_score": 1.0,
        "coverage": 1.0,
        "activation_frequency": 1.0,
        "untyped": False,
        "active_token_count": 4,
    }


def _model() -> ManifoldSAE:
    n, p = 4, 2
    fitted = np.zeros((n, p))
    assignments = np.ones((n, 2))
    logits = np.zeros((n, 2))
    coords = [
        np.array([[0.0], [0.2], [0.7], [0.9]]),
        np.array([[0.0], [0.0], [0.0], [0.0]]),
    ]
    atoms = [
        SaeManifoldAtomFit(
            basis="periodic",
            decoder_coefficients=np.zeros((3, p)),
            assignments=assignments[:, 0],
            coords=coords[0],
            evidence=0.0,
            active_dim=1,
            coords_u_arc=np.array([0.0, 0.25, 0.5, 0.75]),
        ),
        SaeManifoldAtomFit(
            basis="periodic",
            decoder_coefficients=np.zeros((3, p)),
            assignments=assignments[:, 1],
            coords=coords[1],
            evidence=0.0,
            active_dim=1,
            coords_u_arc=None,
        ),
    ]
    low = SaeManifoldFitResult(
        atoms=atoms,
        chosen_k=2,
        evidence_by_candidate={2: 0.0},
        comparison={"winner": "K=2"},
        fitted=fitted,
        assignments=assignments,
        coords=coords,
        reml_score=0.0,
    )
    return ManifoldSAE(
        atoms=atoms,
        atom_topology="circle",
        atom_topologies=["circle", "circle"],
        assignment="ibp_map",
        assignment_label="ibp_map",
        primitive_names=["rust_module.sae_manifold_fit_minimal"],
        fitted=fitted,
        assignments=assignments,
        coords=coords,
        decoder_blocks=[atom.decoder_coefficients.copy() for atom in atoms],
        basis_specs=["periodic", "periodic"],
        penalized_loss_score=0.0,
        reconstruction_r2=0.0,
        training_mean=np.zeros(p),
        training_data=np.zeros((n, p)),
        low_level=low,
        low_level_logits=logits,
        diagnostics={"atom_trust": [1.0, 1.0], "atoms": [_trust_atom(), _trust_atom()]},
        _basis_kinds=["periodic", "periodic"],
        _atom_dims=[1, 1],
        _basis_sizes=[3, 3],
        _n_harmonics=[1, 1],
        _duchon_centers=[None, None],
        coordinate_fidelity=_coordinate_fidelity(),
        certificates=_certificates(),
    )


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
