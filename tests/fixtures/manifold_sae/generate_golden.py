"""Generate the canonical v3 ``ManifoldSAE`` serialization fixtures (#2091).

The fixtures pin the JSON contract owned by Rust's ``ManifoldSaePayload``. The
representative payload exercises every optional surface: mixed atom topologies,
TopK routing, a smooth-gate threshold, output-Fisher steering state, the selected
``ρ*`` fields, and every diagnostic/certificate report block.

The fitted model is the Rust-owned ``ManifoldSAE`` PyO3 class. Accordingly this
generator builds the JSON-compatible v3 payload
directly, so fixture generation needs neither a fit nor a compiled extension.

The covariance fixture persists the compact per-channel factors consumed by the
schema. They are sliced directly from a deterministic dense covariance using the
documented ``(basis, channel)`` flat layout; no fitted core is mutated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
PRIMARY = FIXTURE_DIR / "golden_full.json"
COVARIANCE = FIXTURE_DIR / "golden_with_covariance.json"


def _arange(*shape: int, start: float = 0.0, step: float = 1.0) -> np.ndarray:
    """Deterministic dense array so fixtures are reproducible byte-for-byte."""
    n = int(np.prod(shape))
    return (start + step * np.arange(n, dtype=float)).reshape(shape)


def _trust_atom(seed: float) -> dict[str, Any]:
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


def build_payload() -> dict[str, Any]:
    """Build a representative canonical v3 payload.

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

    atoms: list[dict[str, Any]] = []
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
            {
                "basis": kinds[k],
                "decoder_coefficients": decoder_blocks[k].tolist(),
                "assignments": per_atom_assign[k].tolist(),
                "coords": coords[k].tolist(),
                "coords_u_arc": None if u_arc is None else u_arc.tolist(),
                "evidence": -12.5 - k,
                "active_dim": d[k],
                "decoder_covariance_channel_factors": None,
                "shape_band_coords": (
                    None if band_coords is None else band_coords.tolist()
                ),
                "shape_band_mean": None if band_mean is None else band_mean.tolist(),
                "shape_band_sd": None if band_sd is None else band_sd.tolist(),
                "functional_evidence": func_ev,
            }
        )

    fitted = _arange(N, p, start=0.5, step=0.07)
    assignments = _arange(N, 3, start=0.02, step=0.09)
    logits = _arange(N, 3, start=-1.0, step=0.13)
    training_mean = _arange(p, start=0.4, step=0.25)

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

    return {
        "schema": "gamfit.ManifoldSAE/v5",
        "atoms": atoms,
        "atom_topology": "mixed",
        "atom_topologies": ["circle", "euclidean", "euclidean"],
        "assignment": "topk",
        "assignment_label": "topk",
        "primitive_names": ["circle_0", "line_1", "duchon_2"],
        "fitted": fitted.tolist(),
        "assignments": assignments.tolist(),
        "coords": [c.tolist() for c in coords],
        "decoder_blocks": [b.tolist() for b in decoder_blocks],
        "basis_specs": ["periodic:H2", "euclidean:3", "duchon:4"],
        "penalized_loss_score": -37.5,
        "penalized_quasi_laplace_criterion": 41.25,
        "reconstruction_r2": 0.8123,
        "training_mean": training_mean.tolist(),
        "tier0_scale": _arange(p, start=1.0, step=0.2).tolist(),
        "logits": logits.tolist(),
        "diagnostics": diagnostics,
        "basis_kinds": list(kinds),
        "atom_dims": list(d),
        "basis_sizes": list(m),
        "n_harmonics": [2, 0, 0],
        "duchon_centers": [
            None if center is None else center.tolist() for center in duchon_centers
        ],
        "crosscoder": None,
        "oos_projection_top1": True,
        "alpha": 0.75,
        "learnable_alpha": True,
        "tau": 0.4,
        "sparsity_strength": 1.3,
        "smoothness": 0.9,
        "learning_rate": 0.03,
        "max_iter": 42,
        "random_state": 7,
        "top_k": 2,
        "threshold_gate_threshold": 0.15,
        "solver_plan": {"stages": ["seed", "refine"], "max_outer": 3},
        "dispersion": 1.07,
        "metric_provenance": "OutputFisherDownstream",
        "fisher_mass_residual": fisher_mass_residual.tolist(),
        "atom_two_lens": {
            "presence": [0.1, 0.2, 0.3],
            "coupling": [0.05, 0.15, 0.25],
        },
        "residual_gauge": {"group_signature": "SO(2)", "generators": []},
        "incoherence_report": {
            "mu_hat": 0.31,
            "activity_floor": [0.1, 0.1, 0.1],
        },
        "curvature_report": {"atoms": [{"atom": 0, "kappa_hat": 0.9}]},
        "coordinate_fidelity": {
            "atoms": [{"atom": 0, "verdict": "arclength_honest"}]
        },
        "topology_persistence": {
            "atoms": [{"atom": 0, "betti": [1, 1], "contested": False}]
        },
        "atom_inference": [
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
        "certificates": {
            "overall": "certified",
            "overall_certified": True,
            "claims": {
                "c0": {
                    "claim": "nonconstant",
                    "verdict": "certified",
                    "certified": True,
                    "evidence": {},
                }
            },
        },
        "structure_certificate": json.dumps(
            {
                "alpha": 0.1,
                "entries": [
                    {"kind": "NonConstant", "log_e": 3.4, "steps": 2, "confirmed": True}
                ],
            }
        ),
        "cotrain": {
            "recon_consistency": 0.02,
            "unconverged_fraction": 0.1,
            "n_unconverged": 1,
            "n_encodes": 10,
        },
        "hybrid_split": {
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
        "fisher_factors": fisher_factors.tolist(),
        "fisher_provenance": "output_fisher_downstream",
        "fisher_factor_kind": "uncertified_approximation",
        "selected_log_lambda_sparse": -0.6,
        "selected_log_lambda_smooth": selected_log_lambda_smooth.tolist(),
        "selected_log_ard": [values.tolist() for values in selected_log_ard],
        "structured_residual_diagnostics": [],
        "termination": None,
    }


def build_covariance_payload() -> dict[str, Any]:
    """Add atom 1's compact channel factors without constructing a fit core."""
    payload = build_payload()
    atom = payload["atoms"][1]
    decoder = np.asarray(atom["decoder_coefficients"], dtype=float)
    m_basis, p_out = decoder.shape
    side = m_basis * p_out

    # Preserve the old fixture's deterministic dense covariance exactly, then
    # retain only the same-channel M×M blocks used by posterior shape bands.
    base = _arange(side, side, step=0.001) + np.eye(side)
    covariance = base @ base.T
    factors = np.empty((p_out, m_basis, m_basis), dtype=float)
    for channel in range(p_out):
        indices = np.arange(channel, side, p_out)
        factors[channel] = covariance[np.ix_(indices, indices)]
    atom["decoder_covariance_channel_factors"] = factors.tolist()
    return payload


def main() -> None:
    payload = build_payload()
    PRIMARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {PRIMARY} ({PRIMARY.stat().st_size} bytes)")

    covariance_payload = build_covariance_payload()
    COVARIANCE.write_text(
        json.dumps(covariance_payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {COVARIANCE} ({COVARIANCE.stat().st_size} bytes)")

    assert covariance_payload["atoms"][1][
        "decoder_covariance_channel_factors"
    ] is not None


if __name__ == "__main__":
    main()
