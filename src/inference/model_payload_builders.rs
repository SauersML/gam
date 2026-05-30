//! Shared, source-agnostic builders for saved-model payloads.
//!
//! The CLI (`src/main.rs`) and the Python FFI (`crates/gam-pyffi/src/lib.rs`)
//! both persist fitted models, and both used to assemble the serialized
//! [`FittedModelPayload`] independently. That meant the on-disk contract for a
//! given model kind could silently drift depending on whether the model was
//! created through the CLI or through Python — exactly the failure mode that
//! repeatedly bit the marginal-slope save→load path.
//!
//! This module assembles the *semantic* payload exactly once. Each caller is
//! responsible only for the source-specific work of producing the resolved
//! semantic inputs (the CLI threads them through from its argument parsing and
//! fit pipeline; the FFI freezes term collections from designs and re-derives
//! metadata from the [`FitConfig`]). Once both sides hand the same semantic
//! content to the same assembler, payload drift becomes impossible by
//! construction.

use crate::estimate::UnifiedFitResult;
use crate::families::bms::deviation_runtime::AnchorComponentTag;
use crate::families::bms::{DeviationRuntime, LatentMeasureKind, LatentZRankIntCalibration};
use crate::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL;
use crate::families::transformation_normal::TransformationNormalFamily;
use crate::inference::model::{
    DataSchema, FittedFamily, FittedModelPayload, LikelihoodSpec, MODEL_PAYLOAD_VERSION, ModelKind,
    SavedAnchorComponent, SavedAnchorKind, SavedCompiledFlexBlock, SavedLatentZNormalization,
    TransformationScoreCalibration,
};
use crate::smooth::TermCollectionSpec;
use crate::types::{InverseLink, ResponseFamily, StandardLink, inverse_link_to_binomial_spec};

/// Family tag persisted for Bernoulli marginal-slope saved models.
const FAMILY_BERNOULLI_MARGINAL_SLOPE: &str = "bernoulli-marginal-slope";

/// Family tag persisted for transformation-normal saved models.
const FAMILY_TRANSFORMATION_NORMAL: &str = "transformation-normal";

/// Serialize an anchored-deviation [`DeviationRuntime`] (score-warp or
/// link-deviation block) into its persistable [`SavedCompiledFlexBlock`] form.
///
/// This is the single source of truth for that conversion; the CLI and FFI
/// payload builders both route through it so the serialized flex contract
/// cannot diverge between the two save paths.
pub fn serialize_anchored_deviation_runtime(runtime: &DeviationRuntime) -> SavedCompiledFlexBlock {
    let mut anchor_correction: Option<Vec<Vec<f64>>> = None;
    let mut anchor_components: Vec<SavedAnchorComponent> = Vec::new();
    if let Some(installed) = runtime.installed_flex_block() {
        anchor_correction = Some(
            installed
                .anchor_correction
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<Vec<f64>>>(),
        );
        for component in &installed.anchor_components {
            anchor_components.push(SavedAnchorComponent {
                kind: match component {
                    AnchorComponentTag::Parametric { block, ncols } => {
                        SavedAnchorKind::Parametric {
                            block: *block,
                            ncols: *ncols,
                        }
                    }
                    AnchorComponentTag::FlexEvaluation { ncols } => {
                        SavedAnchorKind::FlexEvaluation { ncols: *ncols }
                    }
                },
            });
        }
    }
    SavedCompiledFlexBlock {
        kernel: ANCHORED_DEVIATION_KERNEL.to_string(),
        breakpoints: runtime.breakpoints().to_vec(),
        basis_dim: runtime.basis_dim(),
        span_c0: runtime
            .span_c0()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c1: runtime
            .span_c1()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c2: runtime
            .span_c2()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c3: runtime
            .span_c3()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        anchor_correction,
        anchor_components,
    }
}

/// Source-specific metadata that the CLI and FFI populate differently but that
/// every saved payload carries.
///
/// `training_feature_ranges` is the only field the FFI path cannot currently
/// supply (it persists headers without per-feature ranges); modeling it as
/// `Option` keeps that distinction explicit instead of silently encoding an
/// empty vector as if ranges were known.
pub struct SavedModelSourceMetadata {
    pub training_headers: Vec<String>,
    pub training_feature_ranges: Option<Vec<(f64, f64)>>,
    pub offset_column: Option<String>,
    pub noise_offset_column: Option<String>,
}

impl SavedModelSourceMetadata {
    fn apply_to(self, payload: &mut FittedModelPayload) {
        match self.training_feature_ranges {
            Some(ranges) => payload.set_training_feature_metadata(self.training_headers, ranges),
            None => payload.training_headers = Some(self.training_headers),
        }
        payload.offset_column = self.offset_column;
        payload.noise_offset_column = self.noise_offset_column;
    }
}

/// The resolved, source-agnostic semantic content of a Bernoulli
/// marginal-slope saved model.
///
/// The CLI threads these in directly from its fit pipeline; the FFI produces
/// them by freezing its term collections and reading the [`FitConfig`]. Either
/// way, the assembler below turns them into the canonical payload.
pub struct BernoulliMarginalSlopeInputs<'a> {
    pub formula: String,
    pub data_schema: DataSchema,
    pub logslope_formula: String,
    pub z_column: String,
    pub resolved_marginalspec: TermCollectionSpec,
    pub resolved_logslopespec: TermCollectionSpec,
    pub fit_result: UnifiedFitResult,
    pub baseline_marginal: f64,
    pub baseline_logslope: f64,
    pub latent_z_normalization: SavedLatentZNormalization,
    pub latent_measure: LatentMeasureKind,
    pub latent_z_rank_int_calibration: Option<LatentZRankIntCalibration>,
    pub score_warp_runtime: Option<&'a DeviationRuntime>,
    pub link_dev_runtime: Option<&'a DeviationRuntime>,
    pub base_link: InverseLink,
    pub frailty: crate::families::lognormal_kernel::FrailtySpec,
}

/// Assemble the canonical Bernoulli marginal-slope payload.
///
/// This is the single place that decides which payload fields a marginal-slope
/// model carries and how the singular/vector mirror fields
/// (`formula_logslope(s)`, `z_column(s)`, `logslope_baseline(s)`,
/// `resolved_termspec_logslope(s)`) are kept consistent — so the CLI and FFI
/// saved models are byte-equivalent for identical semantic content.
pub fn assemble_bernoulli_marginal_slope_payload(
    inputs: BernoulliMarginalSlopeInputs<'_>,
    source: SavedModelSourceMetadata,
) -> Result<FittedModelPayload, String> {
    let BernoulliMarginalSlopeInputs {
        formula,
        data_schema,
        logslope_formula,
        z_column,
        resolved_marginalspec,
        resolved_logslopespec,
        fit_result,
        baseline_marginal,
        baseline_logslope,
        latent_z_normalization,
        latent_measure,
        latent_z_rank_int_calibration,
        score_warp_runtime,
        link_dev_runtime,
        base_link,
        frailty,
    } = inputs;

    let marginal_likelihood_spec =
        inverse_link_to_binomial_spec(&base_link).map_err(|e| e.to_string())?;

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        formula,
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood: marginal_likelihood_spec,
            base_link: Some(base_link.clone()),
            frailty,
        },
        FAMILY_BERNOULLI_MARGINAL_SLOPE.to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.formula_logslope = Some(logslope_formula.clone());
    payload.z_column = Some(z_column.clone());
    payload.formula_logslopes = Some(vec![logslope_formula]);
    payload.z_columns = Some(vec![z_column]);
    payload.latent_z_normalization = Some(latent_z_normalization);
    payload.latent_measure = Some(latent_measure);
    payload.latent_z_rank_int_calibration = latent_z_rank_int_calibration;
    payload.marginal_baseline = Some(baseline_marginal);
    payload.logslope_baseline = Some(baseline_logslope);
    payload.logslope_baselines = Some(vec![baseline_logslope]);
    payload.link = Some(base_link);
    payload.resolved_termspec = Some(resolved_marginalspec);
    payload.resolved_termspec_logslopes = Some(vec![resolved_logslopespec.clone()]);
    payload.resolved_termspec_logslope = Some(resolved_logslopespec);
    payload.score_warp_runtime =
        score_warp_runtime.map(serialize_anchored_deviation_runtime);
    payload.link_deviation_runtime =
        link_dev_runtime.map(serialize_anchored_deviation_runtime);
    source.apply_to(&mut payload);
    Ok(payload)
}

/// The resolved, source-agnostic semantic content of a transformation-normal
/// saved model.
///
/// As with the marginal-slope inputs, the CLI threads the family and resolved
/// covariate spec straight from its fit pipeline while the FFI reads them off
/// its fit-result struct (freezing the covariate spec from its design first).
pub struct TransformationNormalInputs<'a> {
    pub formula: String,
    pub data_schema: DataSchema,
    pub resolved_covariate_spec: TermCollectionSpec,
    pub fit_result: UnifiedFitResult,
    pub family: &'a TransformationNormalFamily,
    pub score_calibration: TransformationScoreCalibration,
}

/// Assemble the canonical transformation-normal payload.
///
/// Centralizing the response-transform snapshot (`knots`, `transform`,
/// `degree`, `median`) and the fixed Gaussian-identity likelihood means the CLI
/// and FFI cannot encode a transformation-normal model two different ways.
pub fn assemble_transformation_normal_payload(
    inputs: TransformationNormalInputs<'_>,
    source: SavedModelSourceMetadata,
) -> FittedModelPayload {
    let TransformationNormalInputs {
        formula,
        data_schema,
        resolved_covariate_spec,
        fit_result,
        family,
        score_calibration,
    } = inputs;

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        formula,
        ModelKind::TransformationNormal,
        FittedFamily::TransformationNormal {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
        },
        FAMILY_TRANSFORMATION_NORMAL.to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.resolved_termspec = Some(resolved_covariate_spec);
    payload.transformation_response_knots = Some(family.response_knots().to_vec());
    payload.transformation_response_transform = Some(
        family
            .response_transform()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    );
    payload.transformation_response_degree = Some(family.response_degree());
    payload.transformation_response_median = Some(family.response_median());
    payload.transformation_score_calibration = Some(score_calibration);
    source.apply_to(&mut payload);
    payload
}
