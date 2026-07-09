"""Generate the golden ``ManifoldSAE`` serialization fixtures (issue #2091).

These fixtures are the **contract** for the Rust-owned payload port. They are the
byte-for-byte output of the *current* Python ``ManifoldSAE.to_dict()`` on a
representative fitted model that exercises every optional field: multiple atom
topologies (periodic / euclidean / duchon), a TopK routing block, a jumprelu
threshold, an installed output-Fisher steering shard, the terminal REML-selected
``ρ*`` (``selected_log_*``), and every diagnostic / certificate report block.

The Rust ``ManifoldSaePayload`` (serde) must round-trip these JSON files
value-for-value; the Python cutover must keep emitting the identical schema.

Run ``python tests/fixtures/manifold_sae/generate_golden.py`` to (re)write the
fixtures. The generator constructs the dataclass directly (no fit / no wheel is
needed to *serialize*, because ``to_dict`` only touches Rust for a non-None
decoder covariance, which this fixture deliberately leaves ``None`` — see NOTE).
When the built wheel *is* importable it additionally verifies the pure-Python
``to_dict -> from_dict -> to_dict`` round-trip is a fixed point.

NOTE (covariance arm): ``ManifoldSAE.to_dict`` marshals a non-None
``decoder_covariance`` through the Rust ``decoder_channel_cov_factors`` FFI
(slice ``36bec0e29``). Capturing a covariance-bearing golden fixture therefore
requires the built wheel; this generator emits that second fixture only when the
wheel is present, so the wheel-free path still produces the primary contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gamfit._sae_manifold import (
    ManifoldSAE,
    SaeManifoldAtomFit,
    SaeManifoldFitResult,
)

FIXTURE_DIR = Path(__file__).resolve().parent
PRIMARY = FIXTURE_DIR / "golden_full.json"
COVARIANCE = FIXTURE_DIR / "golden_with_covariance.json"


def _arange(*shape: int, start: float = 0.0, step: float = 1.0) -> np.ndarray:
    """Deterministic dense array so fixtures are reproducible byte-for-byte."""
    n = int(np.prod(shape))
    return (start + step * np.arange(n, dtype=float)).reshape(shape)


def _trust_atom(seed: float) -> dict:
    """A fully-populated per-atom trust diagnostic (exact key set required)."""
    return {
        "trust_score": 0.5 + 0.1 * seed,
        "sigma_min_tangent": 0.2 + 0.01 * seed,
        "sigma_max_tangent": 1.5 + 0.01 * seed,
        "tangent_condition_score": 0.9 - 0.01 * seed,
        "coverage": 0.8,
        "activation_frequency": 0.3 + 0.05 * seed,
        "untyped": bool(seed == 1),
        "active_token_count": int(10 + seed),
    }


def build_model() -> ManifoldSAE:
    """A representative fitted model exercising every optional field.

    Shapes: N=5 rows, p=4 channels, K=3 atoms.
      atom 0: periodic, M0=5 (H=2 -> 2H+1), d0=1  (carries coords_u_arc + band)
      atom 1: euclidean, M1=3, d1=2
      atom 2: duchon,    M2=4, d2=2               (carries duchon centers)
    """
    N, p = 5, 4
    m = [5, 3, 4]
    d = [1, 2, 2]
    kinds = ["periodic", "euclidean", "duchon"]

    decoder_blocks = [_arange(m[k], p, start=1.0 + k) for k in range(3)]
    coords = [_arange(N, d[k], start=0.1 * (k + 1), step=0.3) for k in range(3)]
    per_atom_assign = [_arange(N, start=0.05 * (k + 1), step=0.11) for k in range(3)]

    atoms = []
    for k in range(3):
        # Shape band + arc coordinate only for the d=1 periodic atom.
        if d[k] == 1:
            g = 6
            band_coords = _arange(g, d[k], step=0.2)
            band_mean = _arange(g, p, start=0.3, step=0.05)
            band_sd = _arange(g, p, start=0.01, step=0.002)
            u_arc = _arange(N, start=0.0, step=0.19)
            func_ev = {
                "average_value": 1.25,
                "average_derivative": -0.4,
                "peak_contrast": 2.1,
            }
        else:
            band_coords = band_mean = band_sd = u_arc = None
            func_ev = None
        atoms.append(
            SaeManifoldAtomFit(
                basis=kinds[k],
                decoder_coefficients=decoder_blocks[k].copy(),
                assignments=per_atom_assign[k].copy(),
                coords=coords[k].copy(),
                evidence=-12.5 - k,
                active_dim=d[k],
                decoder_covariance=None,  # see module NOTE (covariance arm)
                shape_band_coords=band_coords,
                shape_band_mean=band_mean,
                shape_band_sd=band_sd,
                functional_evidence=func_ev,
                coords_u_arc=u_arc,
            )
        )

    fitted = _arange(N, p, start=0.5, step=0.07)
    assignments = _arange(N, 3, start=0.02, step=0.09)
    logits = _arange(N, 3, start=-1.0, step=0.13)
    training_mean = _arange(p, start=0.4, step=0.25)

    low = SaeManifoldFitResult(
        atoms=atoms,
        chosen_k=3,
        evidence_by_candidate={3: -37.5},
        comparison={"winner": "K=3"},
        fitted=fitted.copy(),
        assignments=assignments.copy(),
        coords=[c.copy() for c in coords],
        reml_score=-37.5,
    )

    diagnostics = {
        "atom_trust": [0.6, 0.7, 0.55],
        "atoms": [_trust_atom(float(k)) for k in range(3)],
    }

    duchon_centers = [None, None, _arange(4, 2, step=0.5)]

    # Fisher steering shard (r=2) — the field acceptance bullet 2 wants owned.
    r = 2
    fisher_factors = _arange(N, p, r, start=0.01, step=0.003)
    fisher_mass_residual = _arange(N, start=0.001, step=0.0005)
    selected_log_lambda_smooth = np.array([-1.2, 0.3, 1.1])
    selected_log_ard = [np.array([0.1]), np.array([0.2, -0.3]), np.array([0.4, 0.5])]

    return ManifoldSAE(
        atoms=atoms,
        atom_topology="mixed",
        atom_topologies=list(kinds),
        assignment="topk",  # canonical (from_dict canonicalizes; keep it stable)
        assignment_label="TopK",
        primitive_names=["circle_0", "line_1", "duchon_2"],
        fitted=fitted.copy(),
        assignments=assignments.copy(),
        coords=[c.copy() for c in coords],
        decoder_blocks=[b.copy() for b in decoder_blocks],
        basis_specs=["periodic:H2", "euclidean:3", "duchon:4"],
        penalized_loss_score=-37.5,
        reconstruction_r2=0.8123,
        training_mean=training_mean.copy(),
        training_data=fitted.copy(),  # replaced by a metadata handle in __post_init__
        low_level=low,
        low_level_logits=logits.copy(),
        diagnostics=diagnostics,
        _basis_kinds=list(kinds),
        _atom_dims=list(d),
        _basis_sizes=list(m),
        _n_harmonics=[2, 0, 0],
        _duchon_centers=duchon_centers,
        _oos_projection_top1=True,
        alpha=0.75,
        learnable_alpha=True,
        tau=0.4,
        sparsity_strength=1.3,
        smoothness=0.9,
        learning_rate=0.03,
        max_iter=42,
        random_state=7,
        top_k=2,
        top_k_projection={"kind": "reconstruction", "k": 2, "residual_ev": 0.12},
        pre_topk={"active_fraction": 0.66, "note": "pre-projection gate"},
        jumprelu_threshold=0.15,
        solver_plan={"stages": ["seed", "refine"], "max_outer": 3},
        dispersion=1.07,
        metric_provenance="OutputFisher",
        fisher_mass_residual=fisher_mass_residual,
        structured_residual_diagnostics=[{"pass": 0, "lambda_hat": 0.5}],
        atom_two_lens={"presence": [0.1, 0.2, 0.3], "coupling": [0.05, 0.15, 0.25]},
        residual_gauge={"group_signature": "SO(2)", "generators": []},
        incoherence_report={"mu_hat": 0.31, "activity_floor": [0.1, 0.1, 0.1]},
        curvature_report={"atoms": [{"atom": 0, "kappa_hat": 0.9}]},
        coordinate_fidelity={"atoms": [{"atom": 0, "verdict": "arclength_honest"}]},
        topology_persistence={"atoms": [{"atom": 0, "betti": [1, 1], "contested": False}]},
        atom_inference_reports=[
            {
                "atom_index": 0,
                "atom_name": "circle_0",
                "functionals": {"slope": 0.2},
                "smooth_significance": {"log_e_nonconstant": 3.4},
            },
            {
                "atom_index": 1,
                "atom_name": "line_1",
                "functionals": None,
                "smooth_significance": {"log_e_nonconstant": None},
            },
        ],
        certificates={
            "overall": "certified",
            "overall_certified": True,
            "claims": {"c0": {"claim": "nonconstant", "verdict": "certified", "certified": True, "evidence": {}}},
        },
        structure_certificate_json=json.dumps(
            {
                "alpha": 0.1,
                "entries": [
                    {"kind": "NonConstant", "log_e": 3.4, "steps": 2, "confirmed": True}
                ],
            }
        ),
        cotrain={
            "recon_consistency": 0.02,
            "uncertified_fraction": 0.1,
            "n_uncertified": 1,
            "n_encodes": 10,
        },
        hybrid_split={
            "curved_atom_count": 1,
            "linear_atom_count": 2,
            "total_negative_log_evidence": 12.0,
            "total_parameters": 12,
            "is_pure_linear": False,
            "is_pure_curved": False,
            "atoms": [
                {
                    "atom": "circle_0",
                    "kept_curved": True,
                    "parameterization": "periodic",
                    "negative_log_evidence": 4.0,
                    "num_parameters": 5,
                    "curved_evidence_margin": 0.3,
                    "fitted_turning": 1.2,
                    "train_loao_delta_ev": 0.05,
                    "curved_ev": 0.9,
                    "topm_linear_ev": 0.8,
                    "curved_vs_envelope_ratio": 1.125,
                    "chart_efficiency_eta": 1.125,
                }
            ],
        },
        fisher_factors=fisher_factors,
        fisher_provenance="output_fisher_downstream",
        selected_log_lambda_sparse=-0.6,
        selected_log_lambda_smooth=selected_log_lambda_smooth,
        selected_log_ard=selected_log_ard,
    )


def main() -> None:
    model = build_model()
    payload = model.to_dict()
    PRIMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {PRIMARY} ({PRIMARY.stat().st_size} bytes)")

    # Assert the write-drop / read-tolerate asymmetry is captured faithfully:
    # to_dict never emits structured_residual_diagnostics even though the model
    # carries one (from_dict reads it with a [] default). This is a real schema
    # fact the Rust serde port must mirror (Serialize skips it; Deserialize
    # tolerates + defaults it).
    assert "structured_residual_diagnostics" not in payload, (
        "to_dict unexpectedly emitted structured_residual_diagnostics"
    )
    # reml_score is written as a duplicate of penalized_loss_score.
    assert payload["reml_score"] == payload["penalized_loss_score"]

    # When the wheel is importable, verify the Python round-trip is a fixed point
    # (to_dict -> from_dict -> to_dict). This is the same contract the Rust port
    # must satisfy; capturing it here catches Python-side drift too.
    try:
        reloaded = ManifoldSAE.from_dict(payload)
    except Exception as exc:  # noqa: BLE001 - wheel-free path skips this arm
        print(f"[skip] from_dict round-trip needs the built wheel: {exc!r}")
    else:
        again = reloaded.to_dict()
        if again != payload:
            diff = sorted(set(payload) ^ set(again))
            mismatched = [k for k in payload if k in again and payload[k] != again[k]]
            raise AssertionError(
                f"round-trip not a fixed point; key-set diff={diff} "
                f"value-mismatch keys={mismatched}"
            )
        print("round-trip fixed-point verified (wheel present)")
        # Covariance arm: attach a real covariance to atom 1 and capture the
        # compact-channel-factor on-disk form (needs the wheel).
        cov_model = build_model()
        M1, pp = cov_model.atoms[1].decoder_coefficients.shape
        dim = M1 * pp
        base = _arange(dim, dim, step=0.001) + np.eye(dim)
        cov_model.atoms[1].decoder_covariance = base @ base.T
        cov_payload = cov_model.to_dict()
        COVARIANCE.write_text(json.dumps(cov_payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {COVARIANCE} ({COVARIANCE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
