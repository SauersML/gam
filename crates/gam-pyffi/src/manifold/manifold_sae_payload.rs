//! Rust-owned serde schema for the fitted `ManifoldSAE` model artifact (#2091).
//!
//! The fitted SAE-manifold model is serialized to a JSON payload tagged
//! `"gamfit.ManifoldSAE/v6"` schema. Version 6 is deliberately breaking: it
//! persists each atom's constructor-validated geometry plan as the sole source
//! of topology, chart dimension, analytic resolution, basis width, center set,
//! and reference metric.
//! Historically the schema lived only in the Python dataclass
//! `gamfit/_sae_manifold.py::ManifoldSAE`
//! (`to_dict` / `from_dict`), so a field-name / default / None-handling change
//! there silently corrupts saved models. This module mirrors that schema in Rust
//! (serde), so the round-trip is owned where the math lives and the contract is
//! machine-checked against the golden fixtures in
//! `tests/fixtures/manifold_sae/`.
//!
//! # Design
//!
//! * **Numeric arrays are nested `Vec`s, not `ndarray`.** The Python `to_dict`
//!   serializes arrays via numpy `.tolist()` — nested JSON lists. Mirroring that
//!   with `Vec<Vec<f64>>` / `Vec<f64>` / `Vec<Vec<Vec<f64>>>` reproduces the exact
//!   JSON shape with the derived `Serialize` (no custom array serializer needed),
//!   and value-equality of the reparsed JSON is order-independent.
//! * **Report / certificate blocks are opaque `serde_json::Value`.** They are all
//!   produced by the Rust fit and only read/filtered downstream; holding them
//!   losslessly avoids mirroring ~14 sub-schemas that no consumer computes over.
//! * **Strict schema.** Every field is required, unknown fields are rejected,
//!   and runtime diagnostics serialize with the rest of the model.  There are
//!   no deprecated aliases, missing-field defaults, or load-time repairs.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use gam::terms::sae::manifold::SaeAtomGeometryPlan;

/// The on-disk schema tag. `from_json` rejects any other value, matching the
/// Python `from_dict` guard.
pub(crate) const SCHEMA_TAG: &str = "gamfit.ManifoldSAE/v6";

/// Per-atom payload (`atoms[k]`), one per `SaeManifoldAtomFit`.
///
/// Optional-valued fields serialize explicitly as `null`; omission is invalid.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct AtomPayload {
    /// Physical decoder `B_k`, including the atom's fitted radial scale. There
    /// is no separate hidden amplitude coordinate; persisted predictions read
    /// this decoder directly.
    pub(crate) decoder_coefficients: Vec<Vec<f64>>,
    pub(crate) assignments: Vec<f64>,
    pub(crate) coords: Vec<Vec<f64>>,
    pub(crate) coords_u_arc: Option<Vec<f64>>,
    pub(crate) evidence: Option<f64>,
    pub(crate) active_dim: i64,
    /// Compact per-channel factor `(p, M_k, M_k)`; the dense `(M_k·p)²` joint is
    /// never stored on disk (see `decoder_channel_cov_factors`, #2091 slice 1).
    pub(crate) decoder_covariance_channel_factors: Option<Vec<Vec<Vec<f64>>>>,
    pub(crate) shape_band_coords: Option<Vec<Vec<f64>>>,
    pub(crate) shape_band_mean: Option<Vec<Vec<f64>>>,
    pub(crate) shape_band_sd: Option<Vec<Vec<f64>>>,
    pub(crate) functional_evidence: Option<Value>,
}

/// Stacked-column metadata required to interpret a manifold crosscoder's
/// decoder blocks in honest per-layer units. `None` on an ordinary SAE.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CrosscoderPayload {
    pub(crate) anchor_label: String,
    pub(crate) anchor_dim: i64,
    pub(crate) block_dims: Vec<i64>,
    pub(crate) labels: Vec<String>,
    pub(crate) log_lambda_block: Vec<f64>,
    /// Rust-produced cross-layer drift report, or an explicit undefined status
    /// when consecutive layers do not share an ambient width.
    pub(crate) drift: Value,
    /// Per-atom consecutive-layer transport-law reports.
    pub(crate) transport: Vec<Value>,
}

/// The full fitted-model payload. Field names match the JSON keys `to_dict`
/// emits exactly (the two `#[serde(rename)]`s bridge the Python attribute name
/// to the on-disk key).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManifoldSaePayload {
    pub(crate) schema: String,

    // --- A. scalar config -------------------------------------------------
    pub(crate) assignment: String,
    pub(crate) assignment_label: String,
    pub(crate) alpha: f64,
    pub(crate) learnable_alpha: bool,
    pub(crate) tau: f64,
    pub(crate) sparsity_strength: f64,
    pub(crate) smoothness: f64,
    pub(crate) learning_rate: f64,
    pub(crate) max_iter: i64,
    pub(crate) random_state: i64,
    /// Required key, but nullable (`to_dict` always writes `top_k`, `None`→`null`).
    pub(crate) top_k: Option<i64>,
    pub(crate) threshold_gate_threshold: f64,
    pub(crate) oos_projection_top1: bool,
    pub(crate) dispersion: f64,

    // --- scores -----------------------------------------------------------
    pub(crate) penalized_loss_score: Option<f64>,
    pub(crate) penalized_quasi_laplace_criterion: f64,
    pub(crate) reconstruction_r2: f64,

    // --- B. immutable atom geometry ---------------------------------------
    pub(crate) primitive_names: Vec<String>,
    pub(crate) geometry_plans: Vec<SaeAtomGeometryPlan>,

    // --- C. dense numeric -------------------------------------------------
    pub(crate) training_mean: Vec<f64>,
    /// Required key, nullable only for model families that did not standardize
    /// the output columns before fitting.
    pub(crate) tier0_scale: Option<Vec<f64>>,
    pub(crate) fitted: Vec<Vec<f64>>,
    pub(crate) assignments: Vec<Vec<f64>>,
    #[serde(rename = "logits")]
    pub(crate) low_level_logits: Vec<Vec<f64>>,
    pub(crate) coords: Vec<Vec<Vec<f64>>>,
    pub(crate) decoder_blocks: Vec<Vec<Vec<f64>>>,

    // --- manifold crosscoder layout (#2231 Inc D) ------------------------
    /// Required key in v5; null for a plain SAE.
    pub(crate) crosscoder: Option<CrosscoderPayload>,

    // --- E. per-atom payload ---------------------------------------------
    pub(crate) atoms: Vec<AtomPayload>,

    // --- diagnostics / report blocks (opaque, always emitted) -------------
    pub(crate) diagnostics: Value,
    pub(crate) solver_plan: Option<Value>,
    pub(crate) atom_two_lens: Option<Value>,
    pub(crate) residual_gauge: Option<Value>,
    pub(crate) incoherence_report: Option<Value>,
    pub(crate) curvature_report: Option<Value>,
    pub(crate) coordinate_fidelity: Option<Value>,
    pub(crate) topology_persistence: Option<Value>,
    pub(crate) atom_inference: Option<Value>,
    pub(crate) certificates: Option<Value>,
    /// A JSON *string* (the Rust core's serialized `StructureCertificate`), or null.
    pub(crate) structure_certificate: Option<String>,
    pub(crate) cotrain: Option<Value>,
    pub(crate) hybrid_split: Option<Value>,

    // --- D. Fisher steering ----------------------------------------------
    pub(crate) fisher_factors: Option<Vec<Vec<Vec<f64>>>>,
    pub(crate) fisher_provenance: Option<String>,
    /// Required whenever factors are retained; never inferred from rank or a
    /// scalar tail estimate (#2249).
    pub(crate) fisher_factor_kind: Option<String>,
    pub(crate) metric_provenance: String,
    pub(crate) fisher_mass_residual: Option<Vec<f64>>,
    pub(crate) selected_log_lambda_sparse: Option<f64>,
    pub(crate) selected_log_lambda_smooth: Option<Vec<f64>>,
    pub(crate) selected_log_ard: Option<Vec<Vec<f64>>>,

    // --- runtime diagnostics persisted by v6 -----------------------------
    pub(crate) structured_residual_diagnostics: Vec<Value>,
    /// #2235 — the outer-ρ termination verdict/ledger the fit emitted
    /// (`{"verdict", "evals", "evals_since_improvement", "wall_seconds"}`). Like
    /// Persisted so save/load never drops a live convergence verdict.
    pub(crate) termination: Option<Value>,
}

impl ManifoldSaePayload {
    const REQUIRED_FIELDS: &'static [&'static str] = &[
        "schema",
        "assignment",
        "assignment_label",
        "alpha",
        "learnable_alpha",
        "tau",
        "sparsity_strength",
        "smoothness",
        "learning_rate",
        "max_iter",
        "random_state",
        "top_k",
        "threshold_gate_threshold",
        "oos_projection_top1",
        "dispersion",
        "penalized_loss_score",
        "penalized_quasi_laplace_criterion",
        "reconstruction_r2",
        "primitive_names",
        "geometry_plans",
        "training_mean",
        "tier0_scale",
        "fitted",
        "assignments",
        "logits",
        "coords",
        "decoder_blocks",
        "crosscoder",
        "atoms",
        "diagnostics",
        "solver_plan",
        "atom_two_lens",
        "residual_gauge",
        "incoherence_report",
        "curvature_report",
        "coordinate_fidelity",
        "topology_persistence",
        "atom_inference",
        "certificates",
        "structure_certificate",
        "cotrain",
        "hybrid_split",
        "fisher_factors",
        "fisher_provenance",
        "fisher_factor_kind",
        "metric_provenance",
        "fisher_mass_residual",
        "selected_log_lambda_sparse",
        "selected_log_lambda_smooth",
        "selected_log_ard",
        "structured_residual_diagnostics",
        "termination",
    ];

    const REQUIRED_ATOM_FIELDS: &'static [&'static str] = &[
        "decoder_coefficients",
        "assignments",
        "coords",
        "coords_u_arc",
        "evidence",
        "active_dim",
        "decoder_covariance_channel_factors",
        "shape_band_coords",
        "shape_band_mean",
        "shape_band_sd",
        "functional_evidence",
    ];

    /// Parse the exact v6 artifact schema. Missing optional-valued fields are
    /// still errors: `null` is the explicit absence representation.
    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| format!("ManifoldSAE.from_json: {e}"))?;
        let object = value.as_object().ok_or_else(|| {
            "ManifoldSAE.from_json: model artifact must be a JSON object".to_string()
        })?;
        for field in Self::REQUIRED_FIELDS {
            if !object.contains_key(*field) {
                return Err(format!("ManifoldSAE.from_json: missing field {field:?}"));
            }
        }
        let schema = object.get("schema").and_then(Value::as_str);
        if schema != Some(SCHEMA_TAG) {
            return Err(format!(
                "ManifoldSAE.from_json: unsupported schema {:?}",
                schema
            ));
        }
        if let Some(atoms) = object.get("atoms").and_then(Value::as_array) {
            for (index, atom) in atoms.iter().enumerate() {
                let atom = atom.as_object().ok_or_else(|| {
                    format!("ManifoldSAE.from_json: atoms[{index}] must be an object")
                })?;
                for field in Self::REQUIRED_ATOM_FIELDS {
                    if !atom.contains_key(*field) {
                        return Err(format!(
                            "ManifoldSAE.from_json: atoms[{index}] missing field {field:?}"
                        ));
                    }
                }
            }
        }
        let payload: Self =
            serde_json::from_value(value).map_err(|e| format!("ManifoldSAE.from_json: {e}"))?;
        crate::manifold::manifold_sae_coercion::canonical_assignment_kind(&payload.assignment)
            .map_err(|error| format!("ManifoldSAE.from_json: {error}"))?;
        if payload.assignment_label != payload.assignment {
            return Err(format!(
                "ManifoldSAE.from_json: assignment_label {:?} must equal canonical assignment {:?}",
                payload.assignment_label, payload.assignment
            ));
        }
        if let Some(scale) = payload.tier0_scale.as_ref() {
            if scale.len() != payload.training_mean.len()
                || !scale.iter().all(|value| value.is_finite() && *value > 0.0)
            {
                return Err(format!(
                    "ManifoldSAE.from_json: tier0_scale must be a finite positive length-{} vector",
                    payload.training_mean.len()
                ));
            }
        }
        if (payload.assignment == "topk") != payload.top_k.is_some() {
            return Err(
                "ManifoldSAE.from_json: top_k is required exactly for assignment='topk'"
                    .to_string(),
            );
        }
        let k = payload.geometry_plans.len();
        if k == 0 {
            return Err("ManifoldSAE.from_json: geometry_plans must be non-empty".to_string());
        }
        if payload.max_iter < 1 {
            return Err("ManifoldSAE.from_json: max_iter must be at least 1".to_string());
        }
        if let Some(top_k) = payload.top_k {
            if top_k < 1 || usize::try_from(top_k).ok().is_none_or(|value| value > k) {
                return Err(format!(
                    "ManifoldSAE.from_json: top_k must satisfy 1 <= top_k <= K={k}; got {top_k}"
                ));
            }
        }
        if payload.atoms.len() != k
            || payload.coords.len() != k
            || payload.decoder_blocks.len() != k
        {
            return Err(format!(
                "ManifoldSAE.from_json: atoms, coords, and decoder_blocks must all have geometry-plan count K={k}"
            ));
        }
        let n_obs = payload.assignments.len();
        let p_out = payload.training_mean.len();
        if n_obs == 0 || p_out == 0 {
            return Err(
                "ManifoldSAE.from_json: persisted training frame must be non-empty".to_string(),
            );
        }
        if !payload.training_mean.iter().all(|value| value.is_finite()) {
            return Err("ManifoldSAE.from_json: training_mean must be finite".to_string());
        }
        for (field, rows, width) in [
            ("assignments", &payload.assignments, k),
            ("logits", &payload.low_level_logits, k),
            ("fitted", &payload.fitted, p_out),
        ] {
            if rows.len() != n_obs
                || rows
                    .iter()
                    .any(|row| row.len() != width || !row.iter().all(|value| value.is_finite()))
            {
                return Err(format!(
                    "ManifoldSAE.from_json: {field} must be a finite ({n_obs}, {width}) matrix"
                ));
            }
        }
        for (index, plan) in payload.geometry_plans.iter().enumerate() {
            let latent_dim = plan.latent_dim();
            let basis_size = plan
                .basis_size()
                .map_err(|error| format!("ManifoldSAE.from_json: geometry_plans[{index}]: {error}"))?;
            let coords = &payload.coords[index];
            if coords.len() != n_obs
                || coords.iter().any(|row| {
                    row.len() != latent_dim || !row.iter().all(|value| value.is_finite())
                })
            {
                return Err(format!(
                    "ManifoldSAE.from_json: coords[{index}] must be a finite ({n_obs}, {latent_dim}) matrix"
                ));
            }
            let decoder = &payload.decoder_blocks[index];
            if decoder.len() != basis_size
                || decoder
                    .iter()
                    .any(|row| row.len() != p_out || !row.iter().all(|value| value.is_finite()))
            {
                return Err(format!(
                    "ManifoldSAE.from_json: decoder_blocks[{index}] must be a finite ({basis_size}, {p_out}) matrix derived from geometry_plans[{index}]"
                ));
            }
            let atom = &payload.atoms[index];
            if atom.assignments.len() != n_obs
                || !atom.assignments.iter().all(|value| value.is_finite())
                || atom.coords.len() != n_obs
                || atom.coords.iter().any(|row| {
                    row.len() != latent_dim || !row.iter().all(|value| value.is_finite())
                })
                || atom.decoder_coefficients.len() != basis_size
                || atom.decoder_coefficients.iter().any(|row| {
                    row.len() != p_out || !row.iter().all(|value| value.is_finite())
                })
            {
                return Err(format!(
                    "ManifoldSAE.from_json: atoms[{index}] numeric shapes must agree with its ({n_obs}, {latent_dim}, {basis_size}, {p_out}) geometry plan"
                ));
            }
        }
        Ok(payload)
    }

    pub(crate) fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("ManifoldSAE.to_json: {e}"))
    }
}

/// Parse a `ManifoldSAE.to_dict()` payload through the Rust-owned schema and
/// re-serialize it — the load-bearing round-trip the Python `from_dict`/`to_dict`
/// pair will delegate to at cutover. Errors surface as the same
/// unsupported-schema / malformed-payload messages the Python guard raised.
pub(crate) fn roundtrip_json(json: &str) -> Result<String, String> {
    ManifoldSaePayload::from_json(json)?.to_json()
}

#[cfg(test)]
mod manifold_sae_payload_serde_tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/manifold_sae")
            .join(name)
    }

    fn load_value(name: &str) -> Value {
        let raw = std::fs::read_to_string(fixture(name))
            .unwrap_or_else(|e| panic!("read fixture {name}: {e}"));
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
    }

    /// The typed serde round-trip is a fixed point on the golden payload: parse →
    /// typed struct → re-serialize reproduces the same JSON *value* (key order is
    /// irrelevant; `serde_json::Value` object equality is order-independent).
    #[test]
    fn golden_full_is_a_serde_round_trip_fixed_point() {
        let golden = load_value("golden_full.json");
        let raw = serde_json::to_string(&golden).unwrap();
        let reser = roundtrip_json(&raw).expect("round-trip golden payload");
        let back: Value = serde_json::from_str(&reser).expect("reparse round-tripped payload");
        assert_eq!(
            back, golden,
            "typed serde round-trip must reproduce the golden payload value-for-value"
        );
    }

    #[test]
    fn missing_optional_valued_field_is_rejected() {
        let mut golden = load_value("golden_full.json");
        golden.as_object_mut().unwrap().remove("fisher_factors");
        let error = roundtrip_json(&serde_json::to_string(&golden).unwrap()).unwrap_err();
        assert!(
            error.contains("missing field \"fisher_factors\""),
            "{error}"
        );
    }

    #[test]
    fn runtime_diagnostics_round_trip_in_schema() {
        let mut golden = load_value("golden_full.json");
        golden.as_object_mut().unwrap().insert(
            "structured_residual_diagnostics".to_string(),
            serde_json::json!([{"pass": 0, "lambda_hat": 0.5}]),
        );
        let raw = serde_json::to_string(&golden).unwrap();
        let out: Value = serde_json::from_str(&roundtrip_json(&raw).unwrap()).unwrap();
        assert_eq!(
            out["structured_residual_diagnostics"],
            golden["structured_residual_diagnostics"]
        );
    }

    #[test]
    fn crosscoder_layout_and_reports_round_trip_in_v6() {
        let mut payload = load_value("golden_full.json");
        payload.as_object_mut().unwrap().insert(
            "crosscoder".to_string(),
            serde_json::json!({
                "anchor_label": "L12",
                "anchor_dim": 64,
                "block_dims": [64, 64],
                "labels": ["L13", "L14"],
                "log_lambda_block": [-0.25, 0.5],
                "drift": {"status": "measured", "mean_drift": 0.2, "steps": []},
                "transport": [{"atom": 0, "law_gap": 0.01}]
            }),
        );
        let raw = serde_json::to_string(&payload).unwrap();
        let round_tripped: Value =
            serde_json::from_str(&roundtrip_json(&raw).expect("crosscoder round trip")).unwrap();
        assert_eq!(round_tripped["crosscoder"], payload["crosscoder"]);
    }

    #[test]
    fn wrong_schema_tag_is_rejected() {
        let mut golden = load_value("golden_full.json");
        golden.as_object_mut().unwrap().insert(
            "schema".to_string(),
            Value::String("gamfit.ManifoldSAE/v2".to_string()),
        );
        let raw = serde_json::to_string(&golden).unwrap();
        let err = roundtrip_json(&raw).unwrap_err();
        assert!(
            err.contains("unsupported schema"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_field_is_rejected() {
        let mut golden = load_value("golden_full.json");
        golden
            .as_object_mut()
            .unwrap()
            .insert("legacy_alias".to_string(), Value::Null);
        let error = roundtrip_json(&serde_json::to_string(&golden).unwrap()).unwrap_err();
        assert!(error.contains("unknown field"), "{error}");
    }

    #[test]
    fn invalid_geometry_tuple_and_assignment_alias_are_rejected() {
        let mut invalid_geometry = load_value("golden_full.json");
        invalid_geometry["geometry_plans"][0]["latent_dim"] = Value::from(2);
        let error = roundtrip_json(&serde_json::to_string(&invalid_geometry).unwrap()).unwrap_err();
        assert!(
            error.contains("invalid atom geometry tuple"),
            "{error}"
        );

        let mut assignment_alias = load_value("golden_full.json");
        assignment_alias["assignment_label"] = Value::String("TopK".to_string());
        let error = roundtrip_json(&serde_json::to_string(&assignment_alias).unwrap()).unwrap_err();
        assert!(
            error.contains("assignment_label") && error.contains("canonical"),
            "{error}"
        );
    }

    #[test]
    fn v1_payload_is_rejected_instead_of_guessing_crosscoder_layout() {
        let mut legacy = load_value("golden_full.json");
        legacy.as_object_mut().unwrap().insert(
            "schema".to_string(),
            Value::String("gamfit.ManifoldSAE/v1".to_string()),
        );
        let error = roundtrip_json(&serde_json::to_string(&legacy).unwrap()).unwrap_err();
        assert!(error.contains("unsupported schema"), "unexpected: {error}");
    }
}
