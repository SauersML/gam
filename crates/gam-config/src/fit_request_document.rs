use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;

/// Stable identity of the serialized fit-request document.
pub const FIT_REQUEST_SCHEMA: &str = "gam.fit-request";

/// Current fit-request schema version.
pub const FIT_REQUEST_SCHEMA_VERSION: u32 = 1;

/// A complete, frontend-neutral formula fit request.
///
/// Training data is intentionally not embedded: Rust callers supply a
/// dataset, Python supplies an in-memory table/array, and the CLI
/// supplies a dataset path. Everything that changes the fitted model belongs in
/// this document.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitRequestDocument {
    pub schema: String,
    pub schema_version: u32,
    pub formula: String,
    #[serde(default)]
    pub config: FitRequestConfigDocument,
}

impl FitRequestDocument {
    pub fn new(
        formula: impl Into<String>,
        config: FitRequestConfigDocument,
    ) -> Result<Self, String> {
        let document = Self {
            schema: FIT_REQUEST_SCHEMA.to_string(),
            schema_version: FIT_REQUEST_SCHEMA_VERSION,
            formula: formula.into(),
            config,
        };
        document.validate()?;
        Ok(document)
    }

    pub fn from_json(raw: &str) -> Result<Self, String> {
        let document = serde_json::from_str::<Self>(raw)
            .map_err(|error| format!("invalid fit request document: {error}"))?;
        document.validate()?;
        Ok(document)
    }

    pub fn to_canonical_json(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| format!("failed to serialize fit request document: {error}"))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != FIT_REQUEST_SCHEMA {
            return Err(format!(
                "fit request schema must be '{FIT_REQUEST_SCHEMA}', got {:?}",
                self.schema
            ));
        }
        if self.schema_version != FIT_REQUEST_SCHEMA_VERSION {
            return Err(format!(
                "unsupported fit request schema_version {}; expected {}",
                self.schema_version, FIT_REQUEST_SCHEMA_VERSION
            ));
        }
        if self.formula.trim().is_empty() {
            return Err("fit request formula must be non-empty".to_string());
        }
        Ok(())
    }
}

/// Serializable model configuration shared by Rust, Python, and the CLI.
///
/// Optional fields mean "use the core [`gam_models::fit_orchestration::FitConfig`]
/// default". The document deliberately has one spelling for each concept; the
/// parser does not carry aliases or legacy wire formats.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitRequestConfigDocument {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adaptive_regularization: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_makeham: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_shape: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctn_stage1: Option<CtnStage1Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expectile_tau: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub firth: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flexible_link: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frailty_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frailty_sd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub group_metadata: Option<BTreeMap<String, JsonValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hazard_loading: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_coordinates: Option<LatentCoordinatesDocument>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logslope_formula: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_binomial_theta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_formula: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_offset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outer_max_iter: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analytic_penalties: Option<AnalyticPenaltiesDocument>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pilot_subsample_threshold: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision_hyperpriors: Option<BTreeMap<String, PrecisionHyperpriorDocument>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ridge_lambda: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_dimensions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma_time_degree: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma_time_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub smooth_descriptors: Option<SmoothDescriptorsDocument>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub survival_distribution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub survival_likelihood: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold_time_degree: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold_time_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_basis: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_degree: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_num_internal_knots: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_smooth_lambda: Option<f64>,
    /// Container type of the caller's training table (`"pandas"`, `"polars"`,
    /// `"pyarrow"`, `"numpy"`, ...), passed through opaquely into the saved
    /// model payload for the predict-time output-container fallback (#394).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_table_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformation_normal: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub z_column: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrecisionHyperpriorDocument {
    pub shape: f64,
    pub rate: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct LatentCoordinatesDocument(pub BTreeMap<String, LatentCoordinateDocument>);

impl LatentCoordinatesDocument {
    pub fn to_json_value(&self) -> Result<JsonValue, String> {
        for (symbol, coordinate) in &self.0 {
            if symbol.trim().is_empty() {
                return Err("latent_coordinates keys must be non-empty symbols".to_string());
            }
            if coordinate.n == 0 || coordinate.d == 0 {
                return Err(format!(
                    "latent_coordinates['{symbol}'] requires positive n and d"
                ));
            }
            if coordinate
                .name
                .as_deref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(format!(
                    "latent_coordinates['{symbol}'].name must be non-empty"
                ));
            }
        }
        serde_json::to_value(self)
            .map_err(|error| format!("failed to serialize latent coordinates: {error}"))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LatentCoordinateDocument {
    pub n: usize,
    pub d: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifold: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retraction: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux_prior: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim_selection: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aux_outcome: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct AnalyticPenaltiesDocument(pub Vec<JsonValue>);

impl AnalyticPenaltiesDocument {
    pub fn to_json_value(&self) -> Result<JsonValue, String> {
        for (index, descriptor) in self.0.iter().enumerate() {
            let descriptor = descriptor
                .as_object()
                .ok_or_else(|| format!("analytic_penalties[{index}] must be an object"))?;
            if !descriptor.get("target").is_some_and(JsonValue::is_string) {
                return Err(format!(
                    "analytic_penalties[{index}].target must be a latent-coordinate name"
                ));
            }
        }
        serde_json::to_value(self)
            .map_err(|error| format!("failed to serialize analytic penalties: {error}"))
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SmoothDescriptorsDocument(pub BTreeMap<String, JsonValue>);

impl SmoothDescriptorsDocument {
    pub fn to_json_value(&self) -> Result<JsonValue, String> {
        for (symbol, descriptor) in &self.0 {
            if symbol.trim().is_empty() {
                return Err("smooth_descriptors keys must be non-empty symbols".to_string());
            }
            if !descriptor.is_object() {
                return Err(format!("smooth_descriptors['{symbol}'] must be an object"));
            }
        }
        serde_json::to_value(self)
            .map_err(|error| format!("failed to serialize smooth descriptors: {error}"))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CtnStage1Document {
    pub response_column: String,
    pub covariate_formula_rhs: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<CtnStage1ConfigDocument>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_column: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CtnStage1ConfigDocument {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_degree: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_num_internal_knots: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_penalty_order: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_extra_penalty_orders: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub double_penalty: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn canonical_document_round_trips_identically() {
        let document = FitRequestDocument::new(
            "y ~ duchon(x)",
            FitRequestConfigDocument {
                ctn_stage1: Some(CtnStage1Document {
                    response_column: "dose".to_string(),
                    covariate_formula_rhs: "s(age)".to_string(),
                    config: Some(CtnStage1ConfigDocument {
                        response_degree: Some(4),
                        response_penalty_order: Some(2),
                        ..CtnStage1ConfigDocument::default()
                    }),
                    weight_column: Some("case_weight".to_string()),
                    offset_column: None,
                }),
                latent_coordinates: Some(
                    serde_json::from_value(json!({
                        "x": {"d": 2, "init": "pca", "n": 12, "name": "x"}
                    }))
                    .unwrap(),
                ),
                analytic_penalties: Some(AnalyticPenaltiesDocument(vec![json!(
                    {"kind": "orthogonality", "target": "x", "weight": 1.0}
                )])),
                precision_hyperpriors: Some(BTreeMap::from([(
                    "s(x):roughness".to_string(),
                    PrecisionHyperpriorDocument {
                        shape: 2.0,
                        rate: 0.5,
                    },
                )])),
                smooth_descriptors: Some(
                    serde_json::from_value(json!({
                        "x": {"centers": 8, "kind": "duchon", "vars": ["x"]}
                    }))
                    .unwrap(),
                ),
                ..FitRequestConfigDocument::default()
            },
        )
        .unwrap();

        let encoded = document.to_canonical_json().unwrap();
        let decoded = FitRequestDocument::from_json(&encoded).unwrap();
        assert_eq!(decoded, document);
        assert_eq!(decoded.to_canonical_json().unwrap(), encoded);
    }

    #[test]
    fn parser_rejects_another_schema_or_version() {
        let wrong_schema = r#"{"schema":"other","schema_version":1,"formula":"y ~ x","config":{}}"#;
        assert!(
            FitRequestDocument::from_json(wrong_schema)
                .unwrap_err()
                .contains("schema must be")
        );

        let wrong_version =
            r#"{"schema":"gam.fit-request","schema_version":2,"formula":"y ~ x","config":{}}"#;
        assert!(
            FitRequestDocument::from_json(wrong_version)
                .unwrap_err()
                .contains("unsupported fit request schema_version")
        );
    }

    #[test]
    fn parser_rejects_removed_topology_selector_descriptor() {
        let legacy = r#"{
            "schema": "gam.fit-request",
            "schema_version": 1,
            "formula": "y ~ x",
            "config": {"topology_auto_selector": {"candidates": ["circle"]}}
        }"#;
        let error = FitRequestDocument::from_json(legacy)
            .expect_err("removed no-op selector field must not be silently ignored");
        assert!(
            error.contains("unknown field `topology_auto_selector`"),
            "{error}"
        );
    }
}
