"""#1204 — the public ``ManifoldSAE`` must surface the ``hybrid_split`` report.

The FFI emits a structured ``hybrid_split`` payload (per-atom curved-vs-linear
verdicts: ``fitted_turning`` Θ, ``train_loao_delta_ev``, ``curved_evidence_margin``,
plus dictionary-level aggregates), but the public Python ``ManifoldSAE`` used to
DROP it — ``from_payload`` never read ``payload["hybrid_split"]``, the dataclass
had no field for it, and ``to_dict`` omitted it. Callers had to monkey-patch
``from_payload`` (see ``examples/structural_truth_ledger.py``) to read the raw
block, which is exactly the "frontier not queryable off the normal object" bug.

These tests pin the plumbing WITHOUT needing the built Rust wheel: they build a
``ManifoldSAE`` dataclass directly with a representative ``hybrid_split`` dict and
assert it survives ``to_dict`` and the ``to_dict → from_dict`` round-trip, and
that the dataclass exposes the field (defaulting to ``None`` for older payloads).
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import (  # noqa: E402
    ManifoldSAE,
    SaeManifoldFitResult,
    _penalized_loss_score,
)


def _representative_hybrid_split() -> dict:
    """A hybrid_split block shaped exactly as the FFI emits it (one curved + one
    linear verdict, plus the dictionary-level aggregates)."""
    return {
        "curved_atom_count": 1,
        "linear_atom_count": 1,
        "total_negative_log_evidence": 12.5,
        "total_parameters": 7,
        "is_pure_linear": False,
        "is_pure_curved": False,
        "atoms": [
            {
                "atom": "atom_0",
                "kept_curved": True,
                "parameterization": "curved(d=1)",
                "negative_log_evidence": 8.0,
                "num_parameters": 5,
                "curved_evidence_margin": 1.5,
                "fitted_turning": 3.14159,
                "train_loao_delta_ev": 0.12,
                "curved_ev": 0.82,
                "topm_linear_ev": 0.91,
                "curved_vs_envelope_ratio": 0.9010989010989011,
            },
            {
                "atom": "atom_1",
                "kept_curved": False,
                "parameterization": "linear",
                "negative_log_evidence": 4.5,
                "num_parameters": 2,
                "curved_evidence_margin": -0.3,
                "fitted_turning": 0.0,
                "train_loao_delta_ev": 0.03,
                "curved_ev": 0.18,
                "topm_linear_ev": 0.42,
                "curved_vs_envelope_ratio": 0.42857142857142855,
            },
        ],
    }


def _minimal_model(hybrid_split: dict | None) -> ManifoldSAE:
    """A zero-atom ManifoldSAE — enough to exercise to_dict / from_dict plumbing
    without the Rust fit path (which to_dict / from_dict do not touch)."""
    n, p = 4, 3
    fitted = np.zeros((n, p))
    assigns = np.zeros((n, 0))
    logits = np.zeros((n, 0))
    diagnostics = {"atom_trust": [], "atoms": []}
    low = SaeManifoldFitResult(
        [], 0, {0: 0.0}, {"winner": "K=0"}, fitted, assigns, [], 0.0,
    )
    return ManifoldSAE(
        atoms=[],
        atom_topology="euclidean",
        atom_topologies=[],
        assignment="ibp_map",
        assignment_label="ibp_map",
        primitive_names=["rust_module.sae_manifold_fit_minimal"],
        fitted=fitted,
        assignments=assigns,
        coords=[],
        decoder_blocks=[],
        basis_specs=[],
        reml_score=0.0,
        reconstruction_r2=0.0,
        training_mean=np.zeros(p),
        training_data=np.zeros((n, p)),
        low_level=low,
        low_level_logits=logits,
        diagnostics=diagnostics,
        _basis_kinds=[],
        _atom_dims=[],
        _basis_sizes=[],
        _n_harmonics=[],
        _duchon_centers=[],
        hybrid_split=hybrid_split,
    )


def test_manifoldsae_has_hybrid_split_field_defaulting_none():
    # Field exists and defaults to None when the Rust payload has no report.
    m = _minimal_model(None)
    assert hasattr(m, "hybrid_split"), "ManifoldSAE must expose a hybrid_split field"
    assert m.hybrid_split is None


def test_to_dict_emits_hybrid_split():
    hs = _representative_hybrid_split()
    d = _minimal_model(hs).to_dict()
    assert "hybrid_split" in d, "to_dict() must include the hybrid_split key"
    assert d["hybrid_split"]["curved_atom_count"] == 1
    assert len(d["hybrid_split"]["atoms"]) == 2
    assert d["hybrid_split"]["atoms"][0]["fitted_turning"] == pytest.approx(3.14159)
    assert d["hybrid_split"]["atoms"][1]["train_loao_delta_ev"] == pytest.approx(0.03)
    assert d["hybrid_split"]["atoms"][0]["topm_linear_ev"] == pytest.approx(0.91)
    assert d["hybrid_split"]["atoms"][1]["curved_vs_envelope_ratio"] == pytest.approx(
        0.42857142857142855
    )


def test_to_dict_emits_none_when_absent():
    d = _minimal_model(None).to_dict()
    assert d["hybrid_split"] is None


def test_hybrid_split_round_trips_through_from_dict():
    hs = _representative_hybrid_split()
    restored = ManifoldSAE.from_dict(_minimal_model(hs).to_dict())
    assert restored.hybrid_split is not None, (
        "from_dict must read the hybrid_split block back, not drop it"
    )
    assert restored.hybrid_split["total_negative_log_evidence"] == pytest.approx(12.5)
    assert restored.hybrid_split["atoms"][0]["curved_evidence_margin"] == pytest.approx(1.5)
    assert restored.hybrid_split["atoms"][0]["kept_curved"] is True
    assert restored.hybrid_split["atoms"][0]["curved_ev"] == pytest.approx(0.82)
    assert restored.hybrid_split["atoms"][0]["topm_linear_ev"] == pytest.approx(0.91)
    assert restored.hybrid_split["atoms"][0]["curved_vs_envelope_ratio"] == pytest.approx(
        0.9010989010989011
    )
    assert restored.hybrid_split["atoms"][1]["parameterization"] == "linear"


def test_penalized_loss_score_rejects_legacy_reml_alias():
    with pytest.raises(KeyError, match="penalized-loss score"):
        _penalized_loss_score({"reml_score": 1.25})
