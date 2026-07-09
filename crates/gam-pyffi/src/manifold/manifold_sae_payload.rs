//! Rust-owned serde schema for the fitted `ManifoldSAE` model artifact (#2091).
//!
//! The fitted SAE-manifold model is serialized to a JSON payload tagged
//! `"gamfit.ManifoldSAE/v1"` that ~45 consumers read off disk. Historically the
//! schema lived only in the Python dataclass `gamfit/_sae_manifold.py::ManifoldSAE`
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
//! * **Two contract asymmetries the Python facade encodes are reproduced here:**
//!   1. `reml_score` is a duplicate write-alias of `penalized_loss_score`
//!      (`to_dict` writes both from one value; `from_dict` reads
//!      `penalized_loss_score` and falls back to `reml_score`). See
//!      [`ManifoldSaePayload::normalize_after_load`] / [`ManifoldSaePayload::to_json`].
//!   2. `structured_residual_diagnostics` is **write-dropped, read-tolerated**:
//!      `to_dict` never emits it, `from_dict` reads it with a `[]` default. The
//!      field here is `skip_serializing` + `default`, retained (for the eventual
//!      `#[pyclass]`) but never re-emitted.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The on-disk schema tag. `from_json` rejects any other value, matching the
/// Python `from_dict` guard.
pub(crate) const SCHEMA_TAG: &str = "gamfit.ManifoldSAE/v1";

fn default_metric_provenance() -> String {
    "Euclidean".to_string()
}

/// Per-atom payload (`atoms[k]`), one per `SaeManifoldAtomFit`.
///
/// The always-emitted optional fields (`coords_u_arc`, `evidence`,
/// `decoder_covariance_channel_factors`, `shape_band_*`, `functional_evidence`)
/// serialize as `null` when absent — `to_dict` writes those keys unconditionally,
/// and a plain `Option<T>` field emits `null` for `None`, so the shape matches.
/// The `#[serde(default)]` on the read-tolerant ones lets a legacy dict that
/// omits a newer key still deserialize.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct AtomPayload {
    pub(crate) basis: String,
    pub(crate) decoder_coefficients: Vec<Vec<f64>>,
    pub(crate) assignments: Vec<f64>,
    pub(crate) coords: Vec<Vec<f64>>,
    #[serde(default)]
    pub(crate) coords_u_arc: Option<Vec<f64>>,
    pub(crate) evidence: Option<f64>,
    pub(crate) active_dim: i64,
    /// Compact per-channel factor `(p, M_k, M_k)`; the dense `(M_k·p)²` joint is
    /// never stored on disk (see `decoder_channel_cov_factors`, #2091 slice 1).
    #[serde(default)]
    pub(crate) decoder_covariance_channel_factors: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(default)]
    pub(crate) shape_band_coords: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub(crate) shape_band_mean: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub(crate) shape_band_sd: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    pub(crate) functional_evidence: Option<Value>,
}

/// The full fitted-model payload. Field names match the JSON keys `to_dict`
/// emits exactly (the two `#[serde(rename)]`s bridge the Python attribute name
/// to the on-disk key).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub(crate) struct ManifoldSaePayload {
    pub(crate) schema: String,

    // --- A. scalar config -------------------------------------------------
    pub(crate) atom_topology: String,
    pub(crate) atom_topologies: Vec<String>,
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
    pub(crate) jumprelu_threshold: f64,
    pub(crate) oos_projection_top1: bool,
    pub(crate) dispersion: f64,

    // --- scores (with the reml_score write-alias, handled in to_json) -----
    #[serde(default)]
    pub(crate) penalized_loss_score: Option<f64>,
    #[serde(default)]
    pub(crate) reml_score: Option<f64>,
    pub(crate) reconstruction_r2: f64,

    // --- B. string / int lists --------------------------------------------
    pub(crate) primitive_names: Vec<String>,
    pub(crate) basis_specs: Vec<String>,
    pub(crate) basis_kinds: Vec<String>,
    pub(crate) atom_dims: Vec<i64>,
    pub(crate) basis_sizes: Vec<i64>,
    pub(crate) n_harmonics: Vec<i64>,

    // --- C. dense numeric -------------------------------------------------
    pub(crate) training_mean: Vec<f64>,
    /// Always written as `null` by `to_dict` (new fits do not retain X).
    #[serde(default)]
    pub(crate) training_data: Option<Value>,
    #[serde(default)]
    pub(crate) training_data_retained: bool,
    pub(crate) fitted: Vec<Vec<f64>>,
    pub(crate) assignments: Vec<Vec<f64>>,
    #[serde(rename = "logits")]
    pub(crate) low_level_logits: Vec<Vec<f64>>,
    pub(crate) coords: Vec<Vec<Vec<f64>>>,
    pub(crate) decoder_blocks: Vec<Vec<Vec<f64>>>,
    pub(crate) duchon_centers: Vec<Option<Vec<Vec<f64>>>>,

    // --- E. per-atom payload ---------------------------------------------
    pub(crate) atoms: Vec<AtomPayload>,

    // --- diagnostics / report blocks (opaque, always emitted) -------------
    pub(crate) diagnostics: Value,
    #[serde(default)]
    pub(crate) top_k_projection: Option<Value>,
    #[serde(default)]
    pub(crate) pre_topk: Option<Value>,
    #[serde(default)]
    pub(crate) solver_plan: Option<Value>,
    #[serde(default)]
    pub(crate) atom_two_lens: Option<Value>,
    #[serde(default)]
    pub(crate) residual_gauge: Option<Value>,
    #[serde(default)]
    pub(crate) incoherence_report: Option<Value>,
    #[serde(default)]
    pub(crate) curvature_report: Option<Value>,
    #[serde(default)]
    pub(crate) coordinate_fidelity: Option<Value>,
    #[serde(default)]
    pub(crate) topology_persistence: Option<Value>,
    #[serde(default)]
    pub(crate) atom_inference: Option<Value>,
    #[serde(default)]
    pub(crate) certificates: Option<Value>,
    /// A JSON *string* (the Rust core's serialized `StructureCertificate`), or null.
    #[serde(default)]
    pub(crate) structure_certificate: Option<String>,
    #[serde(default)]
    pub(crate) cotrain: Option<Value>,
    #[serde(default)]
    pub(crate) hybrid_split: Option<Value>,

    // --- D. Fisher steering ----------------------------------------------
    #[serde(default)]
    pub(crate) fisher_factors: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(default)]
    pub(crate) fisher_provenance: Option<String>,
    #[serde(default = "default_metric_provenance")]
    pub(crate) metric_provenance: String,
    #[serde(default)]
    pub(crate) fisher_mass_residual: Option<Vec<f64>>,
    #[serde(default)]
    pub(crate) selected_log_lambda_sparse: Option<f64>,
    #[serde(default)]
    pub(crate) selected_log_lambda_smooth: Option<Vec<f64>>,
    #[serde(default)]
    pub(crate) selected_log_ard: Option<Vec<Vec<f64>>>,

    // --- write-dropped, read-tolerated (schema asymmetry) -----------------
    /// `to_dict` never emits this; `from_dict` reads it with a `[]` default. We
    /// accept it (so a dict that carries it still loads) and retain it for the
    /// eventual `#[pyclass]`, but never re-serialize it.
    #[serde(default, skip_serializing)]
    pub(crate) structured_residual_diagnostics: Vec<Value>,
}

impl ManifoldSaePayload {
    /// Parse a `to_dict` payload, enforcing the schema tag and the
    /// `penalized_loss_score` / `reml_score` fallback, mirroring `from_dict`.
    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        let mut payload: ManifoldSaePayload =
            serde_json::from_str(json).map_err(|e| format!("ManifoldSAE.from_json: {e}"))?;
        if payload.schema != SCHEMA_TAG {
            return Err(format!(
                "ManifoldSAE.from_json: unsupported schema {:?}",
                payload.schema
            ));
        }
        payload.normalize_after_load();
        Ok(payload)
    }

    /// `penalized_loss_score = payload.get("penalized_loss_score",
    /// payload.get("reml_score"))` — the primary key wins, the deprecated alias
    /// is the fallback for dicts written by older versions.
    fn normalize_after_load(&mut self) {
        if self.penalized_loss_score.is_none() {
            self.penalized_loss_score = self.reml_score;
        }
    }

    /// Serialize to the `to_dict` schema, writing `reml_score` as the duplicate
    /// alias of `penalized_loss_score`.
    pub(crate) fn to_json(&self) -> Result<String, String> {
        let mut out = self.clone();
        out.reml_score = out.penalized_loss_score;
        serde_json::to_string(&out).map_err(|e| format!("ManifoldSAE.to_json: {e}"))
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

    /// `reml_score` is a duplicate alias; a legacy dict carrying only `reml_score`
    /// recovers `penalized_loss_score` from it, and the re-serialized payload
    /// emits both keys with the same value.
    #[test]
    fn reml_score_alias_falls_back_and_is_reduplicated() {
        let mut golden = load_value("golden_full.json");
        let obj = golden.as_object_mut().unwrap();
        let score = obj.get("penalized_loss_score").cloned().unwrap();
        assert!(!score.is_null(), "fixture must carry a non-null score");
        obj.remove("penalized_loss_score");
        let raw = serde_json::to_string(&golden).unwrap();
        let out: Value = serde_json::from_str(&roundtrip_json(&raw).unwrap()).unwrap();
        assert_eq!(out["penalized_loss_score"], score);
        assert_eq!(out["reml_score"], score);
    }

    /// `structured_residual_diagnostics` is write-dropped: a payload that carries
    /// it still loads, but the re-serialized payload never emits it.
    #[test]
    fn structured_residual_diagnostics_is_write_dropped_read_tolerated() {
        let mut golden = load_value("golden_full.json");
        assert!(
            golden.get("structured_residual_diagnostics").is_none(),
            "golden fixture must not carry the write-dropped key"
        );
        golden.as_object_mut().unwrap().insert(
            "structured_residual_diagnostics".to_string(),
            serde_json::json!([{"pass": 0, "lambda_hat": 0.5}]),
        );
        let raw = serde_json::to_string(&golden).unwrap();
        let out: Value = serde_json::from_str(&roundtrip_json(&raw).unwrap()).unwrap();
        assert!(
            out.get("structured_residual_diagnostics").is_none(),
            "round-trip must not re-emit the write-dropped key"
        );
    }

    /// A non-`v1` schema tag is rejected with the same message shape as the
    /// Python `from_dict` guard.
    #[test]
    fn wrong_schema_tag_is_rejected() {
        let mut golden = load_value("golden_full.json");
        golden.as_object_mut().unwrap().insert(
            "schema".to_string(),
            Value::String("gamfit.ManifoldSAE/v2".to_string()),
        );
        let raw = serde_json::to_string(&golden).unwrap();
        let err = roundtrip_json(&raw).unwrap_err();
        assert!(err.contains("unsupported schema"), "unexpected error: {err}");
    }
}
