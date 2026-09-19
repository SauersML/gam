use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Stable identity of the serialized fit-request document.
pub(crate) const FIT_REQUEST_SCHEMA: &str = "gam.fit-request";

/// Current fit-request schema version.
pub(crate) const FIT_REQUEST_SCHEMA_VERSION: u32 = 1;

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

    pub(crate) fn from_json(raw: &str) -> Result<Self, String> {
        let document = serde_json::from_str::<Self>(raw)
            .map_err(|error| format!("invalid fit request document: {error}"))?;
        document.validate()?;
        Ok(document)
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frozen_ctn: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transformation_normal_config: Option<CtnStage1ConfigDocument>,
    /// Expectile level(s): one level, or a strictly increasing list fitted
    /// jointly without crossing. A bare number is the one-level spelling.
    #[serde(
        default,
        deserialize_with = "deserialize_expectile_levels",
        skip_serializing_if = "Option::is_none"
    )]
    pub expectile_tau: Option<Vec<f64>>,
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
    pub slope_formula: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_binomial_theta: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_formula: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noise_offset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analytic_penalties: Option<AnalyticPenaltiesDocument>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision_hyperpriors: Option<BTreeMap<String, PrecisionHyperpriorDocument>>,
    /// Explicit root for cross-process warm starts. Omit to disable on-disk
    /// persistence. The path is used exactly as supplied; no temp/cache
    /// discovery or environment fallback is performed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persistent_warm_start_root: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_dimensions: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slope_time_degree: Option<usize>,
    /// Number of B-spline basis functions on the `log t` margin of the
    /// survival marginal-slope slope block (gam#2765, gam#2767). Omitted =
    /// a slope that does not move along follow-up.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slope_time_k: Option<usize>,
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
    /// Explicit centering anchor for the survival baseline time basis, in the
    /// data's own time units. Omit to let the fit pick it from the likelihood
    /// mode and the truncation shape of the data — the robust interior median
    /// exit for marginal-slope and for any genuinely left-truncated dataset
    /// (#751/#1790), the earliest entry age otherwise.
    ///
    /// Until #2631 this document had no field for the anchor, so a complete
    /// request could not carry it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub survival_time_anchor: Option<f64>,
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
    /// Residual genetic repair columns for the Bernoulli marginal-slope family
    /// (gam#2924): conditionally centred genetic residual features entering the
    /// genetic drive beside the score with ridge-shrunk constant coefficients.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_columns: Option<Vec<String>>,
    /// The supplied z column is already transformed by a frozen external model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frozen_score: Option<bool>,
    /// The latent law a marginal-slope fit anchors on (gam#2926): `"auto"` (the
    /// default: the law of the score estimated on the marginal-index span, global
    /// or local by context), `"gaussian"` (the closed form, refused when the
    /// score fails the adequacy check), `"global-empirical"`, or
    /// `"conditional-location-scale"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latent_measure: Option<String>,
    /// A declared finite law of the latent score for a marginal-slope fit
    /// (gam#2923, gam#2926): `{"nodes": [...], "weights": [...]}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declared_latent_law: Option<DeclaredLatentLawDocument>,
}

/// Ascending nodes and positive weights summing to one.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredLatentLawDocument {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
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
    pub(crate) fn to_json_value(&self) -> Result<JsonValue, String> {
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
    pub(crate) fn to_json_value(&self) -> Result<JsonValue, String> {
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
    pub(crate) fn to_json_value(&self) -> Result<JsonValue, String> {
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
    pub fold_column: Option<String>,
    pub group_column: Option<String>,
    /// Generated group folds; absent takes the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub folds: Option<usize>,
    /// Seed of the group-fold assignment; absent takes the recipe's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
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

/// `expectile_tau` accepts one level (`0.9`) or a list of levels (`[0.1, 0.9]`).
fn deserialize_expectile_levels<'de, D>(deserializer: D) -> Result<Option<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Levels {
        One(f64),
        Many(Vec<f64>),
    }
    Ok(
        Option::<Levels>::deserialize(deserializer)?.map(|levels| match levels {
            Levels::One(tau) => vec![tau],
            Levels::Many(levels) => levels,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expectile_tau_accepts_one_level_or_a_list() {
        let parse = |value: &str| -> Option<Vec<f64>> {
            let json = format!(
                r#"{{"schema":"gam.fit-request","schema_version":1,"formula":"y ~ x","config":{{"expectile_tau":{value}}}}}"#
            );
            FitRequestDocument::from_json(&json)
                .unwrap()
                .config
                .expectile_tau
        };
        assert_eq!(parse("0.9"), Some(vec![0.9]));
        assert_eq!(parse("[0.1, 0.5, 0.9]"), Some(vec![0.1, 0.5, 0.9]));
        assert_eq!(parse("null"), None);
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
