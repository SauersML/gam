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

use crate::bms::deviation_runtime::AnchorComponentTag;
use crate::bms::{
    BernoulliMarginalSlopeFitResult, DeviationRuntime, LatentLawConsumed, LatentMeasureKind, LatentZConditionalCalibration,
};
use crate::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL;
use crate::fit_orchestration::drivers::freeze_term_collection_from_design;
use crate::fit_orchestration::{
    DispersionLocationScaleFitResult, ExpectileLocationScaleFitResult, FitConfig, FitNoteSink,
    FitNotes, FitRequest, FitResult, StandardFitResult, WorkflowError,
    expectile_levels_for_config, fit_formula_through_adaptive_resolution,
    fit_materialized_standard_with_notes, fit_model, materialize,
};
use crate::gamlss::{
    BinomialLocationScaleFitResult, DispersionFamilyKind, GaussianLocationScaleFitResult,
};
use crate::inference::model::{
    FittedEstimator, FittedFamily, FittedModelPayload, JOINT_EXPECTILE_FAMILY_TAG,
    ModelKind, SavedAnchorComponent, SavedAnchorKind, SavedCompiledFlexBlock, SavedLatentZNormalization,
    SavedResidualCascade, SavedSplineScan, SavedSurvivalLocationScaleStructure,
    SavedTransformationNormalGeometry, TransformationNormalParameterization,
    TransformationScoreCalibration,
};
use crate::survival::construction::{
    SavedSurvivalTimeBasis, SurvivalBaselineConfig, survival_baseline_targetname,
};
use crate::survival::marginal_slope::SurvivalMarginalSlopeFitResult;
use crate::survival::predict::apply_inverse_link_state_to_fit_result;
use crate::survival::location_scale::{
    ResidualDistribution, SurvivalCovariateTimeBasis, SurvivalLocationScaleTimeParameterization,
    residual_distribution_from_inverse_link,
};
use crate::transformation_normal::{TransformationNormalFamily, TransformationNormalFitResult};
use crate::wiggle::{WigglePenaltyMetadata, canonical_wiggle_function_penalties};
use gam_data::{DataSchema, EncodedDataset};
use gam_linalg::matrix::LinearOperator;
use gam_problem::types::{
    InverseLink, LikelihoodSpec, ResponseFamily, StandardLink, inverse_link_to_binomial_spec,
};
use gam_solve::estimate::{
    FittedLinkState, UnifiedFitResult, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit,
};
use gam_terms::inference::formula_dsl::{
    parse_formula, parse_surv_interval_response, parse_surv_response,
};
use gam_terms::smooth::{BlockwisePenalty, TermCollectionDesign, TermCollectionSpec};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

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
pub(crate) fn serialize_anchored_deviation_runtime(runtime: &DeviationRuntime) -> SavedCompiledFlexBlock {
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

/// Complete semantic input for persisting a standard formula fit.
///
/// The workflow result is consumed as one value so callers cannot accidentally
/// mix a design, resolved term specification, fitted link, or wiggle state from
/// different fits.  Formula front ends should fit through `fit_from_formula`
/// and hand its `Standard` result directly to this assembler.
pub struct StandardPayloadInputs<'a> {
    pub formula: String,
    pub dataset: &'a EncodedDataset,
    pub fit_config: &'a FitConfig,
    pub result: StandardFitResult,
}

fn fitted_inverse_link(state: &FittedLinkState) -> Option<InverseLink> {
    match state {
        FittedLinkState::Standard(Some(link)) => Some(InverseLink::Standard(*link)),
        FittedLinkState::Standard(None) => None,
        FittedLinkState::LatentCLogLog { state } => Some(InverseLink::LatentCLogLog(*state)),
        FittedLinkState::Sas { state, .. } => Some(InverseLink::Sas(*state)),
        FittedLinkState::BetaLogistic { state, .. } => Some(InverseLink::BetaLogistic(*state)),
        FittedLinkState::Mixture { state, .. } => Some(InverseLink::Mixture(state.clone())),
    }
}

/// The complete penalty topology in the fit geometry's raw coefficient frame.
///
/// `TermCollectionDesign` describes the formula's mean block only. A standard
/// learnable-link fit appends a `LinkWiggle` block, so treating that base design
/// as the fitted model's complete raw topology compares different coordinate
/// systems: the former has `p_mean` columns while the latter's gauge and
/// Hessian have `p_mean + p_wiggle`. Keep the join here, at the one persistence
/// boundary that owns both the realized term design and the canonical saved
/// wiggle semantics, and hand null-space analysis one indivisible topology.
struct RealizedRawPenaltyTopology {
    coefficient_dim: usize,
    penalties: Vec<BlockwisePenalty>,
}

impl RealizedRawPenaltyTopology {
    fn from_standard_fit(
        design: &TermCollectionDesign,
        wiggle_knots: Option<&Array1<f64>>,
        wiggle_degree: Option<usize>,
        wiggle_penalty_metadata: Option<&WigglePenaltyMetadata>,
    ) -> Result<Self, String> {
        let mean_dim = design.design.ncols();
        let mut topology = Self {
            coefficient_dim: mean_dim,
            penalties: design.penalties.clone(),
        };

        let (knots, degree, metadata) = match (
            wiggle_knots,
            wiggle_degree,
            wiggle_penalty_metadata,
        ) {
            (None, None, None) => return Ok(topology),
            (Some(knots), Some(degree), Some(metadata)) => (knots, degree, metadata),
            _ => {
                return Err(
                    "standard fit has partial link-wiggle penalty topology; knots, degree, and canonical penalty metadata must be present together"
                        .to_string(),
                );
            }
        };

        // Rebuild through the same canonical function-penalty factory used by
        // fitting and saved-model replay. This is not a guessed coefficient
        // ridge: these are the exact final-function penalties named by the
        // fitted topology, in the fitted LinkWiggle raw coefficient frame.
        let canonical = canonical_wiggle_function_penalties(
            knots,
            degree,
            &metadata.derivative_orders,
            metadata.double_penalty,
        )
        .map_err(|reason| {
            format!("failed to realize standard link-wiggle penalty topology: {reason}")
        })?;
        if canonical.metadata != *metadata {
            return Err(format!(
                "standard link-wiggle penalty topology {:?} disagrees with the canonical topology {:?} rebuilt from the fitted knots and derivative orders",
                metadata.blocks, canonical.metadata.blocks,
            ));
        }
        let wiggle_dim = canonical
            .matrices
            .first()
            .map(|matrix| matrix.nrows())
            .ok_or_else(|| "standard link-wiggle topology has no penalty blocks".to_string())?;
        if wiggle_dim == 0 {
            return Err("standard link-wiggle topology has zero raw coefficients".to_string());
        }
        let wiggle_range = mean_dim..mean_dim + wiggle_dim;
        for (index, matrix) in canonical.matrices.into_iter().enumerate() {
            if matrix.dim() != (wiggle_dim, wiggle_dim) {
                return Err(format!(
                    "standard link-wiggle penalty {index} is {}x{} but the realized raw block has width {wiggle_dim}",
                    matrix.nrows(),
                    matrix.ncols(),
                ));
            }
            topology
                .penalties
                .push(BlockwisePenalty::new(wiggle_range.clone(), matrix));
        }
        topology.coefficient_dim = wiggle_range.end;
        Ok(topology)
    }
}

/// The frozen penalty the full-conformal set of an eligible standard fit needs.
/// For Gaussian identity it is recovered as `Sλ = M₀ − XᵀX` from the unit-weight
/// training Gram; for the GLM families of
/// [`crate::inference::full_conformal_glm`] as `φ·(H − XᵀW_H X)` from the
/// observed-information weights `W_H` the converged P-IRLS Hessian was built
/// from, which puts it on the unit-dispersion likelihood the conformal refits
/// minimize. Offsets are allowed: they enter only the linear predictor. Only the
/// p × p penalty is persisted: the labeled rows the set is built on are
/// supplied again at prediction time, so the saved model never grows with `n`.
fn standard_conformal_penalty(
    fit_config: &FitConfig,
    family: &LikelihoodSpec,
    fit: &UnifiedFitResult,
    design: &TermCollectionDesign,
) -> Option<crate::inference::full_conformal::ExactFullConformalPenalty> {
    let expectile = fit_config.family.as_deref().is_some_and(|family| {
        let family = family.trim().to_ascii_lowercase();
        family == "expectile" || family.starts_with("expectile(")
    });
    // The conformal refits minimize the plain penalized likelihood: an
    // inequality-constrained coefficient space is a different fitting map.
    let constrained = design.linear_constraints.is_some()
        || design
            .coefficient_lower_bounds
            .as_ref()
            .is_some_and(|bounds| bounds.iter().any(|bound| bound.is_finite()));
    if expectile
        || fit_config.weight_column.is_some()
        || fit_config.flexible_link
        || constrained
    {
        return None;
    }
    let normal_matrix = fit.penalized_hessian()?;
    // The penalty may legitimately be unavailable (the Gram cannot be formed
    // for this design). `None` is the contract, but the reason is what explains
    // a fit that ships without exact full-conformal intervals.
    let penalty = if family.is_gaussian_identity() {
        let unit_weights = Array1::<f64>::ones(design.design.nrows());
        design.design.diag_xtw_x(&unit_weights).and_then(|gram| {
            crate::inference::full_conformal::ExactFullConformalPenalty::from_gram_and_normal_matrix(
                &gram,
                normal_matrix,
                fit.lambdas.len(),
            )
        })
    } else if let Some(glm) =
        crate::inference::full_conformal_glm::ConformalGlmFamily::from_likelihood(family)
    {
        glm_conformal_penalty(glm, fit, design, normal_matrix)
    } else {
        return None;
    };
    match penalty {
        Ok(penalty) => Some(penalty),
        Err(reason) => {
            log::trace!("exact full-conformal penalty unavailable: {reason}");
            None
        }
    }
}

fn glm_conformal_penalty(
    glm: crate::inference::full_conformal_glm::ConformalGlmFamily,
    fit: &UnifiedFitResult,
    design: &TermCollectionDesign,
    normal_matrix: &Array2<f64>,
) -> Result<crate::inference::full_conformal::ExactFullConformalPenalty, String> {
    let pirls = fit
        .artifacts
        .pirls
        .as_ref()
        .ok_or_else(|| "the fit retains no P-IRLS observed-information weights".to_string())?;
    let n = design.design.nrows();
    if pirls.finalweights.len() != n {
        return Err(format!(
            "P-IRLS observed-information weights have {} rows but the design has {n}",
            pirls.finalweights.len()
        ));
    }
    if normal_matrix.nrows() != design.design.ncols() {
        return Err(format!(
            "penalized Hessian is {0}×{0} but the design has {1} columns",
            normal_matrix.nrows(),
            design.design.ncols()
        ));
    }
    let weights = pirls.finalweights.to_owned();
    let gram = design.design.diag_xtw_x(&weights)?;
    // The P-IRLS weights carry the Gamma dispersion as `w/φ`; the other
    // supported families have unit dispersion.
    let scale = match glm {
        crate::inference::full_conformal_glm::ConformalGlmFamily::GammaLog => pirls
            .likelihood
            .resolved_scale()
            .and_then(|scale| scale.gamma_phi())
            .map_err(|err| err.to_string())?,
        _ => 1.0,
    };
    let penalized: Vec<std::ops::Range<usize>> = design
        .penalties
        .iter()
        .map(|penalty| penalty.col_range.clone())
        .collect();
    let s_lambda = crate::inference::full_conformal_glm::penalty_from_normal_and_gram(
        normal_matrix,
        &gram,
        &penalized,
        scale,
    )?;
    // v40 (#4103): carry the COMPONENTS beside the sum. `s_lambda` above is
    // `Σ_k λ_k S_k` recovered by cancellation from the normal matrix, and the
    // honest map cannot re-select against a sum -- its criterion carries
    // `log|Σ_k e^{ρ_k}S_k|₊`, which a sum has already lost (#2644).
    //
    // The blocks carried are the DESIGN's own penalties, unscaled, paired with
    // the `λ_k` the criterion selected. The Gamma dispersion `scale` above
    // belongs to the P-IRLS weights the Gram was formed with, not to the
    // criterion's penalties, so applying it here would pair a scaled block with
    // an unscaled strength. The pair carried is the one the fit actually
    // optimised.
    let p = design.design.ncols();
    let mut components = Vec::with_capacity(design.penalties.len());
    for (index, penalty) in design.penalties.iter().enumerate() {
        let range = penalty.col_range.clone();
        if range.end > p || penalty.local.nrows() != range.len() {
            return Err(format!(
                "full conformal penalty: component {index} is {}x{} on columns {range:?} of a \
                 {p}-column design",
                penalty.local.nrows(),
                penalty.local.ncols()
            ));
        }
        let mut block = Array2::<f64>::zeros((p, p));
        block
            .slice_mut(ndarray::s![range.clone(), range])
            .assign(&penalty.local);
        components.push(block);
    }
    if fit.lambdas.len() != components.len() {
        return Err(format!(
            "full conformal penalty: {} fitted smoothing parameter(s) against {} penalty \
             component(s)",
            fit.lambdas.len(),
            components.len()
        ));
    }
    let mut log_strengths = Vec::with_capacity(components.len());
    for (index, &lambda) in fit.lambdas.iter().enumerate() {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(format!(
                "full conformal penalty: fitted smoothing parameter {index} is {lambda}, which \
                 has no log-strength"
            ));
        }
        log_strengths.push(lambda.ln());
    }
    crate::inference::full_conformal::ExactFullConformalPenalty::from_s_lambda(
        s_lambda,
        fit.lambdas.len(),
    )?
    .with_components(components, log_strengths)
}

/// The comparable REML/LAML criterion of a standard fit: its raw criterion
/// plus the Tierney-Kadane normalizer over its realized penalty null space,
/// formed exactly as the saved payload forms it, so two fits of the same data
/// at different basis resolutions are ranked on one scale (lower is better).
/// `Ok(None)` when the fit has no finite criterion.
pub(crate) fn standard_fit_comparable_reml_score(
    result: &StandardFitResult,
) -> Result<Option<f64>, String> {
    let Some(raw_reml_score) = result.fit.reml_score() else {
        return Ok(None);
    };
    let topology = RealizedRawPenaltyTopology::from_standard_fit(
        &result.design,
        result.wiggle_knots.as_ref(),
        result.wiggle_degree,
        result.wiggle_penalty_metadata.as_ref(),
    )?;
    let (null_space_dim, null_space_logdet) = gam_solve::estimate::null_space_normalizer_metadata(
        topology.coefficient_dim,
        &topology.penalties,
        &result.fit,
    )?;
    gam_solve::topology_selector::comparable_reml_score(
        raw_reml_score,
        Some(null_space_dim as f64),
        Some(null_space_logdet),
    )
}

/// Assemble the one canonical saved payload for a standard formula fit.
pub fn assemble_standard_payload(
    inputs: StandardPayloadInputs<'_>,
) -> Result<FittedModelPayload, String> {
    let StandardPayloadInputs {
        formula,
        dataset,
        fit_config,
        result,
    } = inputs;
    let StandardFitResult {
        mut fit,
        design,
        resolvedspec,
        basis_adequacy,
        saved_link_state,
        wiggle_knots,
        wiggle_degree,
        wiggle_penalty_metadata,
        wiggle_saved_index_shift,
        ..
    } = result;
    fit.fitted_link = saved_link_state;
    let resolved_termspec = freeze_term_collection_from_design(&resolvedspec, &design)
        .map_err(|err| format!("failed to freeze standard term specification: {err}"))?;
    let raw_penalty_topology = RealizedRawPenaltyTopology::from_standard_fit(
        &design,
        wiggle_knots.as_ref(),
        wiggle_degree,
        wiggle_penalty_metadata.as_ref(),
    )?;
    let (null_space_dim, null_space_logdet) =
        gam_solve::estimate::null_space_normalizer_metadata(
            raw_penalty_topology.coefficient_dim,
            &raw_penalty_topology.penalties,
            &fit,
        )?;
    fit.artifacts.null_space_dim = Some(null_space_dim);
    fit.artifacts.null_space_logdet = Some(null_space_logdet);
    let family = fit
        .likelihood_family
        .clone()
        .ok_or_else(|| {
            "standard fit reached payload assembly without its resolved likelihood family"
                .to_string()
        })?;
    let estimator = match expectile_levels_for_config(fit_config)
        .map_err(|error| format!("failed to persist estimator metadata: {error}"))?
        .as_deref()
    {
        None => FittedEstimator::Likelihood,
        Some([tau]) => FittedEstimator::Expectile { tau: *tau },
        Some(levels) => {
            return Err(format!(
                "a standard fit cannot persist the joint expectile levels {levels:?}; they are \
                 fitted as one location-scale model"
            ));
        }
    };
    let family_label = match &estimator {
        FittedEstimator::Expectile { tau } => format!("expectile({tau})"),
        _ => family.name().to_string(),
    };
    let full_conformal = standard_conformal_penalty(fit_config, &family, &fit, &design);
    let latent_cloglog_state = if family.is_latent_cloglog() {
        Some(saved_latent_cloglog_state_from_fit(&fit).ok_or_else(|| {
            "latent-cloglog-binomial fit did not produce a fitted latent-cloglog state".to_string()
        })?)
    } else {
        saved_latent_cloglog_state_from_fit(&fit)
    };
    let mut payload = FittedModelPayload::new(
        formula,
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: family.clone(),
            link: StandardLink::try_from(family.link_function()).ok(),
            latent_cloglog_state,
            mixture_state: saved_mixture_state_from_fit(&fit),
            sas_state: saved_sas_state_from_fit(&fit),
        },
        family_label,
    );
    payload.estimator = estimator;
    payload.fit_result = Some(fit.clone());
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = fitted_inverse_link(&fit.fitted_link).or_else(|| Some(family.link.clone()));
    payload.linkwiggle_knots = wiggle_knots.map(|knots| knots.to_vec());
    payload.linkwiggle_degree = wiggle_degree;
    payload.linkwiggle_penalty_metadata = wiggle_penalty_metadata;
    payload.link_wiggle_index_shift = wiggle_saved_index_shift;
    payload.set_training_feature_metadata(dataset.headers.clone(), dataset.feature_ranges());
    payload.resolved_termspec = Some(resolved_termspec);
    payload.basis_adequacy = basis_adequacy;
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    payload.weight_column = fit_config.weight_column.clone();
    payload.full_conformal = full_conformal;
    Ok(payload)
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
    pub slope_formula: String,
    pub z_column: String,
    pub resolved_marginalspec: TermCollectionSpec,
    pub resolved_slopespec: TermCollectionSpec,
    pub fit_result: UnifiedFitResult,
    /// Number of *raw* marginal design columns `p_m` (= the term-collection
    /// marginal design's `ncols()` BEFORE any #461 influence-absorber widening).
    ///
    /// When the Stage-1 influence absorber is active (A2), the fitted marginal
    /// block carries the widened coefficient `[β_m; γ]` (length `p_m + p₁`) and
    /// the joint covariance is dimensioned over the widened block. The absorbed
    /// influence columns `Z̃_infl` are a TRAINING-only leakage absorber that does
    /// not exist at predict rows, so the persisted model must drop `γ` and the
    /// marginalized-out covariance sub-block to stay self-consistent against the
    /// raw `p_m` marginal design at predict. The assembler uses this to truncate
    /// the fit result once (shared CLI + FFI). With no absorber it equals the
    /// fitted block width and the truncation is a no-op.
    pub p_marginal: usize,
    pub baseline_marginal: f64,
    pub baseline_slope: f64,
    pub latent_z_normalization: SavedLatentZNormalization,
    pub latent_measure: LatentMeasureKind,
    pub latent_law_consumed: LatentLawConsumed,
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
    pub score_warp_runtime: Option<&'a DeviationRuntime>,
    pub link_dev_runtime: Option<&'a DeviationRuntime>,
    pub base_link: InverseLink,
    pub frailty: crate::survival::lognormal_kernel::FrailtySpec,
    /// The residual genetic repair geometry (gam#2924) when the fit carried a
    /// residual block; its coefficients are block 2 of `fit_result`.
    pub residual_repair: Option<crate::bms::ResidualRepairGeometry>,
}

/// Drop the #461 training-only influence-absorber coefficients `γ` from a fitted
/// Bernoulli marginal-slope result so the persisted model is self-consistent
/// against the raw `p_m`-column marginal design at predict.
///
/// When the A2 influence absorber is active the marginal block (block 0) is the
/// widened `[β_m; γ]` (length `p_m + p₁`, with `γ` the contiguous trailing `p₁`
/// columns — see bms `widen_marginal_dense_with_influence`) and the joint
/// conditional covariance is dimensioned over the widened joint coefficient
/// vector. The absorbed columns `Z̃_infl` exist only at training rows; predict
/// reconstructs the marginal index from the raw `p_m` design and the
/// orthogonalized `β̂_m` is a property of the training fit. So this:
///
///  * slices `blocks[0].beta` and `block_states[0].beta` to their first `p_m`
///    entries (the flat `beta` is recomputed from the blocks by
///    `try_from_parts`),
///  * **marginalizes** `γ` out of the joint Gaussian by dropping the `γ`
///    rows/cols from the conditional covariance — taking the corresponding
///    SUB-BLOCK of `Σ` is the exact marginal of a joint Gaussian (no
///    re-inversion), so the kept `[β_m | β_slope | …]` covariance is the
///    correct predictive uncertainty accounting for the fitted absorber,
///  * drops the persisted joint penalized-Hessian geometry: it is a precision
///    over the *widened* joint coefficient vector, so a sub-block would be the
///    wrong marginalization, and the only predict path that consumes it is the
///    covariance-fallback that re-inverts `H` — which post-truncation would have
///    the wrong dimension anyway. With the dense (already-marginalized) `Σ`
///    matching the predict dimension, that fallback is never taken, so dropping
///    the geometry removes a stale, wrong-dimension path rather than a used one.
///
/// Block-level `edf` / `lambdas` are left untouched: they are fitted scalars
/// that legitimately reflect the full model (the absorber consumed real dof at
/// fit time) and are persisted as-is. With no absorber (`block0.len() == p_m`)
/// this is a no-op clone.
fn truncate_marginal_slope_influence_absorber(
    fit_result: UnifiedFitResult,
    p_marginal: usize,
) -> Result<UnifiedFitResult, String> {
    let Some(block0) = fit_result.blocks.first() else {
        return Err("marginal-slope fit result has no coefficient blocks".to_string());
    };
    let widened_len = block0.beta.len();
    if widened_len <= p_marginal {
        // No influence absorber installed (or already raw width): nothing to drop.
        return Ok(fit_result);
    }
    let p_influence = widened_len - p_marginal;

    // The input fit's existence is its convergence proof (sealed
    // `FitConvergenceEvidence`); carry the certified inner status into the
    // narrowed reassembly, which revalidates the preserved artifacts.
    let pirls_status = fit_result.convergence_evidence().inner_status();
    let training_sample_size = fit_result.training_sample_size();
    let training_response_fingerprint = fit_result.training_response_fingerprint();
    // Read through the accessors before destructuring: the criterion pair is
    // private so that no consumer can substitute a number for an absent one,
    // and a narrowing reassembly must carry the absence forward unchanged.
    let reml_score = fit_result.reml_score();
    let penalized_objective = fit_result.penalized_objective();
    let UnifiedFitResult {
        mut blocks,
        log_lambdas,
        lambdas,
        likelihood_family,
        likelihood_scale,
        log_likelihood_normalization,
        log_likelihood,
        deviance,
        stable_penalty_term,
        used_device,
        outer_iterations,
        outer_gradient_norm,
        standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference,
        fitted_link,
        geometry: _,
        mut block_states,
        beta: _,
        max_abs_eta,
        constraint_kkt,
        artifacts,
        inner_cycles,
        outer_cost_evals,
        inner_pirls_solves,
        ..
    } = fit_result;

    // Slice block 0's coefficients (and matching block-state) to the raw p_m,
    // dropping the trailing γ absorber columns.
    blocks[0].beta = blocks[0].beta.slice(ndarray::s![..p_marginal]).to_owned();
    if let Some(state0) = block_states.first_mut() {
        state0.beta = state0.beta.slice(ndarray::s![..p_marginal]).to_owned();
    }

    // Marginalize γ out of the joint conditional covariance: keep every index
    // except the contiguous γ block [p_marginal, p_marginal + p_influence).
    let drop_gamma_block = |cov: Option<Array2<f64>>| -> Option<Array2<f64>> {
        cov.map(|cov| {
            let total = cov.nrows();
            let kept: Vec<usize> = (0..p_marginal)
                .chain((p_marginal + p_influence)..total)
                .collect();
            let mut out = Array2::<f64>::zeros((kept.len(), kept.len()));
            for (ri, &r) in kept.iter().enumerate() {
                for (ci, &c) in kept.iter().enumerate() {
                    out[[ri, ci]] = cov[[r, c]];
                }
            }
            out
        })
    };
    let covariance_conditional = drop_gamma_block(covariance_conditional);
    let covariance_corrected = drop_gamma_block(covariance_corrected);

    UnifiedFitResult::try_from_parts(gam_solve::estimate::UnifiedFitResultParts {
        blocks,
        training_sample_size,
        training_response_fingerprint,
        log_lambdas,
        lambdas,
        likelihood_family,
        likelihood_scale,
        log_likelihood_normalization,
        log_likelihood,
        deviance,
        reml_score,
        stable_penalty_term,
        penalized_objective,
        // Preserve the GPU-execution flag across the absorber-column
        // truncation: dropping the trailing γ columns does not change which
        // device ran the solve.
        used_device,
        outer_iterations,
        outer_converged: true,
        outer_gradient_norm,
        standard_deviation,
        covariance_conditional,
        covariance_corrected,
        inference,
        fitted_link,
        // Drop the widened-joint penalized Hessian: see the doc comment.
        geometry: None,
        block_states,
        pirls_status,
        max_abs_eta,
        constraint_kkt,
        artifacts,
        inner_cycles,
    })
    // The truncation does not change how much outer work the fit did, so
    // carry the whole-fit counters over from the widened solve.
    .map(|mut narrowed| {
        narrowed.outer_cost_evals = outer_cost_evals;
        narrowed.inner_pirls_solves = inner_pirls_solves;
        narrowed
    })
    .map_err(|e| {
        format!("marginal-slope influence-absorber truncation produced an invalid fit result: {e}")
    })
}

/// Assemble the canonical spline-scan payload (#1030/#1034): a standard
/// Gaussian-identity model whose fit representation is the exact O(n)
/// smoothing-spline smoother state instead of a dense `fit_result`. The CLI
/// and FFI save paths both route through here so the scan on-disk contract
/// cannot diverge between sources.
pub fn assemble_spline_scan_payload(
    formula: String,
    feature_column: String,
    fit: &gam_solve::spline_scan::SplineScanFit,
    data_schema: DataSchema,
    training_headers: Vec<String>,
    training_feature_ranges: Vec<(f64, f64)>,
) -> FittedModelPayload {
    let mut payload = FittedModelPayload::new(
        formula,
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::gaussian_identity(),
            link: None,
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    );
    payload.spline_scan = Some(SavedSplineScan {
        feature_column,
        state: fit.to_state(),
    });
    payload.data_schema = Some(data_schema);
    payload.set_training_feature_metadata(training_headers, training_feature_ranges);
    payload
}

/// Assemble the canonical residual-cascade payload (#1032).
///
/// The CLI and FFI save paths both route through here so the cascade on-disk
/// contract cannot diverge between sources.  Mirrors `assemble_spline_scan_payload`
/// but for d ∈ {2,3} scattered coordinates (the Wendland multilevel-frame state).
pub fn assemble_residual_cascade_payload(
    formula: String,
    feature_columns: Vec<String>,
    fit: &gam_solve::residual_cascade::ResidualCascadeFit,
    data_schema: DataSchema,
    training_headers: Vec<String>,
    training_feature_ranges: Vec<(f64, f64)>,
) -> Result<FittedModelPayload, String> {
    let mut payload = FittedModelPayload::new(
        formula,
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: gam_problem::types::LikelihoodSpec::gaussian_identity(),
            link: None,
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    );
    payload.residual_cascade = Some(SavedResidualCascade {
        feature_columns,
        state: fit.to_state().map_err(|e| {
            format!("residual-cascade to_state failed during payload assembly: {e}")
        })?,
    });
    payload.data_schema = Some(data_schema);
    payload.set_training_feature_metadata(training_headers, training_feature_ranges);
    Ok(payload)
}

/// Assemble the canonical Bernoulli marginal-slope payload.
///
/// This is the single place that decides which payload fields a marginal-slope
/// model carries and how the singular/vector mirror fields
/// (`z_column(s)`, `resolved_slopespec(s)`) are kept consistent — so the CLI and FFI
/// saved models are byte-equivalent for identical semantic content.
pub fn assemble_bernoulli_marginal_slope_payload(
    inputs: BernoulliMarginalSlopeInputs<'_>,
    source: SavedModelSourceMetadata,
) -> Result<FittedModelPayload, String> {
    let BernoulliMarginalSlopeInputs {
        formula,
        data_schema,
        slope_formula,
        z_column,
        resolved_marginalspec,
        resolved_slopespec,
        fit_result,
        p_marginal,
        baseline_marginal,
        baseline_slope,
        latent_z_normalization,
        latent_measure,
        latent_law_consumed,
        latent_z_conditional_calibration,
        score_warp_runtime,
        link_dev_runtime,
        base_link,
        frailty,
        residual_repair,
    } = inputs;

    // #461 predict seam: drop the training-only influence-absorber γ (and
    // marginalize it out of the covariance) so the persisted model matches the
    // raw p_m marginal design at predict. No-op when the absorber is inactive.
    let fit_result = truncate_marginal_slope_influence_absorber(fit_result, p_marginal)?;

    let marginal_likelihood_spec =
        inverse_link_to_binomial_spec(&base_link).map_err(|e| e.to_string())?;

    let mut payload = FittedModelPayload::new(
        formula,
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood: marginal_likelihood_spec,
            base_link: base_link.clone(),
            frailty,
        },
        FAMILY_BERNOULLI_MARGINAL_SLOPE.to_string(),
    );
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.slope_formula = Some(slope_formula);
    payload.z_column = Some(z_column.clone());
    payload.z_columns = Some(vec![z_column]);
    payload.latent_z_normalization = Some(latent_z_normalization);
    payload.latent_measure = Some(latent_measure);
    latent_law_consumed.require_recorded("bernoulli marginal-slope payload")?;
    payload.latent_law_consumed = Some(latent_law_consumed);
    payload.latent_z_conditional_calibration = latent_z_conditional_calibration;
    payload.marginal_baseline = Some(baseline_marginal);
    payload.baseline_slope = Some(baseline_slope);
    payload.link = Some(base_link);
    payload.resolved_termspec = Some(resolved_marginalspec);
    payload.resolved_slopespecs = Some(vec![resolved_slopespec.clone()]);
    payload.resolved_slopespec = Some(resolved_slopespec);
    payload.score_warp_runtime = score_warp_runtime.map(serialize_anchored_deviation_runtime);
    payload.link_deviation_runtime = link_dev_runtime.map(serialize_anchored_deviation_runtime);
    payload.residual_repair = residual_repair;
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
) -> Result<FittedModelPayload, String> {
    let TransformationNormalInputs {
        formula,
        data_schema,
        resolved_covariate_spec,
        fit_result,
        family,
        score_calibration,
    } = inputs;

    fit_result.require_posterior_mean("transformation-normal saved-model assembly")
        .map_err(|error| error.to_string())?;

    let mut payload = FittedModelPayload::new(
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
    payload.transformation_geometry = Some(transformation_normal_geometry(family));
    payload.transformation_score_calibration = Some(score_calibration);
    source.apply_to(&mut payload);
    Ok(payload)
}

/// Snapshot the direct-α CTN geometry (gam#2306) a saved model needs to replay
/// the transform and the certified-domain prediction refusal.
///
/// The response value basis is `[1, I_1, …, I_K]` (`p_resp` columns), so the
/// shape-coordinate count is `p_resp − 1` (column 0 is the unconstrained
/// location field). The Khatri-Rao positivity-cone carrier is the `n × p_cov`
/// covariate design, and the certified response support is the clamped-knot
/// span `[knots.first, knots.last]` the endpoint bases were evaluated at.
fn transformation_normal_geometry(
    family: &TransformationNormalFamily,
) -> SavedTransformationNormalGeometry {
    let knots = family.response_knots();
    let lo = knots.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = knots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    SavedTransformationNormalGeometry {
        parameterization: TransformationNormalParameterization::DirectAlpha,
        response_degree: family.response_degree(),
        response_knot_count: knots.len(),
        shape_coordinate_count: family.p_resp().saturating_sub(1),
        cone_carrier_covariate_width: family.p_cov(),
        cone_carrier_row_count: family.n_obs(),
        certified_response_support: (lo, hi),
        response_median: family.response_median(),
    }
}

/// Which likelihood a (non-survival) location-scale model carries: Gaussian
/// (residual response scale) or binomial (likelihood resolved from the inverse
/// link). The assembler resolves the `FittedFamily` from this once, rather than
/// each save path stamping a (potentially wrong) likelihood and patching it
/// afterwards. No location-scale fit residualizes its noise design, so none
/// persists a scale-deviation transform (#3015).
pub enum LocationScaleResponse {
    /// Gaussian identity; `base_link` is the optional resolved base link the CLI
    /// may pass through from `link(...)` (the FFI leaves it `None`).
    Gaussian {
        response_scale: f64,
        /// σ floor in standardized response units
        /// (`GaussianLocationScaleFitResult::sigma_floor`).
        sigma_floor: f64,
        base_link: Option<InverseLink>,
    },
    /// Binomial under `link`.
    Binomial { link: InverseLink },
    /// A genuine-dispersion mean family (NegativeBinomial / Gamma / Beta /
    /// Tweedie) whose log-precision channel carries `noise_formula` (#913). The
    /// `likelihood` is the family's own [`LikelihoodSpec`]; `base_link` is the
    /// mean inverse link (log, or logit for Beta). The log-precision block
    /// coefficients are the fit's `BlockRole::Scale` block.
    Dispersion {
        likelihood: LikelihoodSpec,
        base_link: InverseLink,
        family_tag: &'static str,
    },
}

/// Optional link-wiggle basis metadata persisted alongside a location-scale
/// model. The knots are already in raw response units — the Gaussian
/// standardization and its inverse remap live inside
/// `fit_gaussian_location_scale_model`, so the save path persists them verbatim.
/// The wiggle coefficients are the fit's `LinkWiggle` block; they are not
/// stored a second time.
pub struct LocationScaleWiggle {
    pub knots: Vec<f64>,
    pub degree: usize,
}

/// Source-agnostic semantic content of a (non-survival) location-scale saved
/// model — the shared core behind the CLI's Gaussian/binomial save paths and
/// the FFI's two location-scale builders.
pub struct LocationScaleInputs {
    pub formula: String,
    pub data_schema: DataSchema,
    pub noise_formula: String,
    pub resolved_termspec: TermCollectionSpec,
    pub resolved_termspec_noise: TermCollectionSpec,
    pub fit_result: UnifiedFitResult,
    pub wiggle: Option<LocationScaleWiggle>,
}

/// Assemble the canonical (non-survival) location-scale payload — single source
/// of truth for that on-disk contract. The family/likelihood is resolved from
/// the [`LocationScaleResponse`] so the binomial branch never persists a wrong
/// probit likelihood that a caller must patch afterwards.
pub fn assemble_location_scale_payload(
    inputs: LocationScaleInputs,
    response: LocationScaleResponse,
    source: SavedModelSourceMetadata,
) -> Result<FittedModelPayload, String> {
    inputs
        .fit_result
        .require_posterior_mean("location-scale saved-model assembly")
        .map_err(|error| error.to_string())?;
    let (family_tag, likelihood, base_link, link, gaussian_scales) = match response {
        LocationScaleResponse::Gaussian {
            response_scale,
            sigma_floor,
            base_link,
        } => (
            "gaussian-location-scale".to_string(),
            LikelihoodSpec::gaussian_identity(),
            // Gaussian location-scale does not carry a base link in its family
            // state; the resolved link is persisted in `payload.link` below so
            // prediction can recover it.
            None,
            Some(base_link.unwrap_or(InverseLink::Standard(StandardLink::Identity))),
            Some((response_scale, sigma_floor)),
        ),
        LocationScaleResponse::Binomial { link } => {
            let likelihood = inverse_link_to_binomial_spec(&link).map_err(|e| {
                format!("failed to resolve LikelihoodSpec for binomial location-scale link {link:?}: {e}")
            })?;
            (
                "binomial-location-scale".to_string(),
                likelihood,
                Some(link.clone()),
                Some(link),
                None,
            )
        }
        LocationScaleResponse::Dispersion {
            likelihood,
            base_link,
            family_tag,
        } => (
            family_tag.to_string(),
            likelihood,
            Some(base_link.clone()),
            Some(base_link),
            None,
        ),
    };

    let mut payload = FittedModelPayload::new(
        inputs.formula,
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood,
            base_link,
        },
        family_tag,
    );
    payload.fit_result = Some(inputs.fit_result);
    payload.data_schema = Some(inputs.data_schema);
    payload.link = link;
    payload.formula_noise = Some(inputs.noise_formula);
    payload.gaussian_response_scale = gaussian_scales.map(|(response_scale, _)| response_scale);
    payload.gaussian_sigma_floor = gaussian_scales.map(|(_, sigma_floor)| sigma_floor);
    payload.resolved_termspec = Some(inputs.resolved_termspec);
    payload.resolved_termspec_noise = Some(inputs.resolved_termspec_noise);
    if let Some(wiggle) = inputs.wiggle {
        payload.linkwiggle_knots = Some(wiggle.knots);
        payload.linkwiggle_degree = Some(wiggle.degree);
    }
    source.apply_to(&mut payload);
    Ok(payload)
}

/// Source-agnostic semantic content of a survival marginal-slope
/// (Royston-Parmar net) saved model. Centralizing assembly also fixes the
/// FFI's prior omission of the `*_columns` vector mirrors the CLI wrote.
pub struct SurvivalMarginalSlopeInputs<'a> {
    pub formula: String,
    pub data_schema: DataSchema,
    pub fit_result: UnifiedFitResult,
    pub frailty: crate::survival::lognormal_kernel::FrailtySpec,
    pub survival_entry: Option<String>,
    pub survival_exit: String,
    pub survival_event: String,
    pub survivalspec: String,
    pub baseline_cfg: SurvivalBaselineConfig,
    pub time_basis: SavedSurvivalTimeBasis,
    pub survival_likelihood_label: String,
    pub resolved_marginalspec: TermCollectionSpec,
    pub resolved_slopespec: TermCollectionSpec,
    /// The fit's resolved slope follow-up time margin (gam#2765, gam#2767),
    /// or `None` for a slope that is constant within a person.
    ///
    /// `resolved_slopespec` names the covariate factor only; with a margin
    /// present the fitted coefficients live against `X_cov ⊗ᵣ B(log t)`, so this
    /// is the half of the block's authority the term spec cannot carry.
    pub slope_time_basis: Option<SurvivalCovariateTimeBasis>,
    pub slope_formula: String,
    pub z_column: String,
    pub latent_z_normalization: SavedLatentZNormalization,
    /// The automatic latent-measure gate's decision for the persisted score
    /// surface (gam#2768), split by
    /// [`SurvivalMarginalSlopeFitResult::persisted_latent_z_calibrations`].
    /// `None` unless the declared conditional location-scale law calibrated the
    /// score.
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
    /// Which latent law the fit consumed (gam#2926).
    pub latent_law_consumed: LatentLawConsumed,
    /// The latent measure the fit's row program integrated against
    /// (gam#2923): the standard-normal law of the closed form, or the declared
    /// finite law the index was anchored on. Replayed by the shared
    /// marginal-slope predictor through the same anchoring equation.
    pub latent_measure: LatentMeasureKind,
    /// The declared atoms a compressed fit was certified against, and the
    /// compression's ledger (gam#2928); both `None` for a law anchored as
    /// declared.
    pub declared_latent_law: Option<crate::bms::EmpiricalZGrid>,
    pub declared_latent_law_compression: Option<crate::inference::model::SavedDeclaredLawCompression>,
    pub baseline_slope: f64,
    /// Frozen nonlinear time-wiggle authority, including the raw fitted tail.
    pub timewiggle: Option<SurvivalTimewiggle>,
    pub score_warp_runtime: Option<&'a DeviationRuntime>,
    pub link_dev_runtime: Option<&'a DeviationRuntime>,
    /// Width `p₁` of the absorbed Stage-1 influence block (#461) when the fit
    /// hosted a dedicated additive absorber. Predict drops the absorber's `γ`;
    /// this is persisted only so the predictor accounts for the extra trailing
    /// block in the saved block count.
    pub influence_absorber_width: Option<usize>,
    pub influence_absorber_design: Option<&'a Array2<f64>>,
    pub score_covariance: &'a Array2<f64>,
}

/// Construct a Royston-Parmar survival [`FittedModelPayload`] through the
/// canonical `Survival` family scaffold shared by every RP on-disk contract
/// (marginal-slope, transformation, location-scale): the identity-link
/// `RoystonParmar` likelihood, the persisted likelihood label, and the
/// `fit_result` / `data_schema` install. Callers supply the two variants that
/// differ — `survival_distribution` and `frailty` — and then set their own
/// family-specific fields on the returned payload.
///
/// A fit whose constrained posterior declined its moments keeps its optimizer
/// mode under that typed decline, which records why the moments are unavailable
/// at the boundary and, when a boundary-mode approximation was measured, its
/// certificate and overturn tail mass. The model is saved with the mode: plug-in
/// predictions read it, and every consumer of posterior moments refuses by the
/// decline's summary (#979, gnomon#2336).
fn new_royston_parmar_survival_payload(
    formula: String,
    fit_result: UnifiedFitResult,
    data_schema: DataSchema,
    survival_likelihood_label: &str,
    survival_distribution: Option<ResidualDistribution>,
    frailty: crate::survival::lognormal_kernel::FrailtySpec,
) -> Result<FittedModelPayload, String> {
    if let Some(decline) = fit_result.posterior_moment_decline() {
        log::debug!(
            "[survival saved-model assembly] saving the converged constrained mode; posterior \
             moments are unavailable at the boundary: {}",
            decline.summary()
        );
    }
    let mut payload = FittedModelPayload::new(
        formula,
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some(survival_likelihood_label.to_string()),
            survival_distribution,
            frailty,
        },
        ResponseFamily::RoystonParmar.name().to_string(),
    );
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    Ok(payload)
}

/// Assemble the canonical survival marginal-slope payload — single source of
/// truth for that Royston-Parmar / Gaussian-residual on-disk contract.
pub fn assemble_survival_marginal_slope_payload(
    inputs: SurvivalMarginalSlopeInputs<'_>,
    source: SavedModelSourceMetadata,
) -> Result<FittedModelPayload, String> {
    let mut payload = new_royston_parmar_survival_payload(
        inputs.formula,
        inputs.fit_result,
        inputs.data_schema,
        &inputs.survival_likelihood_label,
        Some(ResidualDistribution::Gaussian),
        inputs.frailty,
    )?;
    payload.survival_entry = inputs.survival_entry;
    payload.survival_exit = Some(inputs.survival_exit);
    payload.survival_event = Some(inputs.survival_event);
    payload.survivalspec = Some(inputs.survivalspec);
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(inputs.baseline_cfg.target).to_string());
    payload.survival_baseline_scale = inputs.baseline_cfg.scale;
    payload.survival_baseline_shape = inputs.baseline_cfg.shape;
    payload.survival_baseline_rate = inputs.baseline_cfg.rate;
    payload.survival_baseline_makeham = inputs.baseline_cfg.makeham;
    payload.apply_survival_time_basis(&inputs.time_basis);
    payload.survival_likelihood = Some(inputs.survival_likelihood_label);
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    payload.resolved_termspec = Some(inputs.resolved_marginalspec);
    payload.resolved_slopespecs = Some(vec![inputs.resolved_slopespec.clone()]);
    payload.resolved_slopespec = Some(inputs.resolved_slopespec);
    payload.slope_time_basis = inputs.slope_time_basis;
    payload.slope_formula = Some(inputs.slope_formula);
    payload.z_column = Some(inputs.z_column.clone());
    payload.z_columns = Some(vec![inputs.z_column]);
    payload.latent_z_normalization = Some(inputs.latent_z_normalization);
    // The measure the fit integrated against (gam#2923): the closed form's
    // standard-normal law, or the declared finite law the anchored frame solved
    // the marginal identity on. The pair below is the pre-transform applied to z
    // before either kernel.
    payload.latent_measure = Some(inputs.latent_measure);
    payload.declared_latent_law = inputs.declared_latent_law;
    payload.declared_latent_law_compression = inputs.declared_latent_law_compression;
    inputs
        .latent_law_consumed
        .require_recorded("survival marginal-slope payload")?;
    payload.latent_law_consumed = Some(inputs.latent_law_consumed);
    payload.latent_z_conditional_calibration = inputs.latent_z_conditional_calibration;
    payload.baseline_slope = Some(inputs.baseline_slope);
    if let Some(timewiggle) = inputs.timewiggle {
        payload.baseline_timewiggle_degree = Some(timewiggle.degree);
        payload.baseline_timewiggle_knots = Some(timewiggle.knots);
        payload.baseline_timewiggle_penalty_orders = timewiggle.penalty_orders;
        payload.baseline_timewiggle_double_penalty = timewiggle.double_penalty;
        apply_timewiggle_beta(&mut payload, timewiggle.beta);
    }
    payload.score_warp_runtime = inputs
        .score_warp_runtime
        .map(serialize_anchored_deviation_runtime);
    payload.link_deviation_runtime = inputs
        .link_dev_runtime
        .map(serialize_anchored_deviation_runtime);
    payload.influence_absorber_width = inputs.influence_absorber_width;
    payload.influence_absorber_design = inputs
        .influence_absorber_design
        .map(|design| design.rows().into_iter().map(|row| row.to_vec()).collect());
    payload.survival_marginal_slope_score_covariance = Some(
        inputs
            .score_covariance
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    );
    source.apply_to(&mut payload);
    Ok(payload)
}

/// Fitted baseline-timewiggle coefficients: a single block (net) or one per
/// cause (joint cause-specific). Callers pass already-sliced coefficients.
pub enum SurvivalTimewiggleBeta {
    Single(Vec<f64>),
    ByCause(Vec<Vec<f64>>),
}

/// Route the fitted baseline-timewiggle coefficients into the matching payload
/// slot. Both survival payload assemblers funnel through this ONE exhaustive
/// `match` so a new [`SurvivalTimewiggleBeta`] variant is a compile error rather
/// than a silent drop (the location-scale assembler previously `if let`-matched
/// only `Single` and silently discarded `ByCause`).
fn apply_timewiggle_beta(payload: &mut FittedModelPayload, beta: SurvivalTimewiggleBeta) {
    match beta {
        SurvivalTimewiggleBeta::Single(beta) => {
            payload.beta_baseline_timewiggle = Some(beta);
        }
        SurvivalTimewiggleBeta::ByCause(by_cause) => {
            payload.beta_baseline_timewiggle_by_cause = Some(by_cause);
        }
    }
}

/// Snapshot of the baseline-timewiggle block persisted with a survival model.
pub struct SurvivalTimewiggle {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub penalty_orders: Option<Vec<usize>>,
    pub double_penalty: Option<bool>,
    pub beta: SurvivalTimewiggleBeta,
}

/// Source-agnostic semantic content of a survival transformation
/// (Royston-Parmar) saved model — net single-cause or joint cause-specific.
pub struct SurvivalTransformationInputs {
    pub formula: String,
    pub data_schema: DataSchema,
    pub fit_result: UnifiedFitResult,
    pub survival_entry: Option<String>,
    pub survival_exit: String,
    pub survival_event: String,
    pub survivalspec: String,
    /// `None` = net single-cause; `Some(n)` persists `survival_cause_count` and
    /// `cause_1..cause_n` endpoint names.
    pub cause_count: Option<usize>,
    pub baseline_cfg: SurvivalBaselineConfig,
    pub time_basis: SavedSurvivalTimeBasis,
    pub survival_likelihood_label: String,
    pub resolved_termspec: TermCollectionSpec,
    pub timewiggle: Option<SurvivalTimewiggle>,
}

/// Assemble the canonical survival transformation payload — single source of
/// truth for the Royston-Parmar transformation on-disk contract.
pub fn assemble_survival_transformation_payload(
    inputs: SurvivalTransformationInputs,
    source: SavedModelSourceMetadata,
) -> Result<FittedModelPayload, String> {
    let mut payload = new_royston_parmar_survival_payload(
        inputs.formula,
        inputs.fit_result,
        inputs.data_schema,
        &inputs.survival_likelihood_label,
        None,
        crate::survival::lognormal_kernel::FrailtySpec::None,
    )?;
    payload.survival_entry = inputs.survival_entry;
    payload.survival_exit = Some(inputs.survival_exit);
    payload.survival_event = Some(inputs.survival_event);
    payload.survivalspec = Some(inputs.survivalspec);
    if let Some(cause_count) = inputs.cause_count {
        payload.survival_cause_count = Some(cause_count);
        payload.survival_endpoint_names = Some(
            (1..=cause_count)
                .map(|idx| format!("cause_{idx}"))
                .collect(),
        );
    }
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(inputs.baseline_cfg.target).to_string());
    payload.survival_baseline_scale = inputs.baseline_cfg.scale;
    payload.survival_baseline_shape = inputs.baseline_cfg.shape;
    payload.survival_baseline_rate = inputs.baseline_cfg.rate;
    payload.survival_baseline_makeham = inputs.baseline_cfg.makeham;
    payload.apply_survival_time_basis(&inputs.time_basis);
    if let Some(timewiggle) = inputs.timewiggle {
        payload.baseline_timewiggle_degree = Some(timewiggle.degree);
        payload.baseline_timewiggle_knots = Some(timewiggle.knots);
        payload.baseline_timewiggle_penalty_orders = timewiggle.penalty_orders;
        payload.baseline_timewiggle_double_penalty = timewiggle.double_penalty;
        apply_timewiggle_beta(&mut payload, timewiggle.beta);
    }
    payload.survival_likelihood = Some(inputs.survival_likelihood_label);
    payload.resolved_termspec = Some(inputs.resolved_termspec);
    source.apply_to(&mut payload);
    Ok(payload)
}

/// Source-agnostic semantic content of a survival location-scale
/// (Royston-Parmar with a learned residual link) saved model. Centralizing
/// fixes the drift where CLI and FFI disagreed on `formula_noise`,
/// `baseline_timewiggle_*`, and exact location-scale replay topology.
pub struct SurvivalLocationScaleInputs {
    pub formula: String,
    pub data_schema: DataSchema,
    /// Fit result with the fitted inverse-link state and link-wiggle artifacts
    /// already applied by the caller.
    pub fit_result: UnifiedFitResult,
    pub fitted_inverse_link: InverseLink,
    // Independent `Option`s (not an all-or-nothing group) so the assembler
    // reproduces exactly what the CLI and FFI each persist independently.
    pub linkwiggle_degree: Option<usize>,
    pub linkwiggle_knots: Option<Vec<f64>>,
    pub baseline_timewiggle: Option<SurvivalTimewiggle>,
    pub survival_entry: Option<String>,
    pub survival_exit: String,
    pub survival_event: String,
    pub survivalspec: String,
    pub baseline_cfg: SurvivalBaselineConfig,
    pub time_basis: SavedSurvivalTimeBasis,
    pub survival_likelihood_label: String,
    pub time_parameterization: SurvivalLocationScaleTimeParameterization,
    pub threshold_time_basis: Option<SurvivalCovariateTimeBasis>,
    pub log_sigma_time_basis: Option<SurvivalCovariateTimeBasis>,
    pub formula_noise: Option<String>,
    pub resolved_thresholdspec: TermCollectionSpec,
    pub resolved_log_sigmaspec: TermCollectionSpec,
}

/// Assemble the canonical survival location-scale payload (the single source of
/// truth for that on-disk contract).
pub fn assemble_survival_location_scale_payload(
    inputs: SurvivalLocationScaleInputs,
    source: SavedModelSourceMetadata,
) -> Result<FittedModelPayload, String> {
    let survival_distribution =
        residual_distribution_from_inverse_link(&inputs.fitted_inverse_link);
    let mut payload = new_royston_parmar_survival_payload(
        inputs.formula,
        inputs.fit_result,
        inputs.data_schema,
        &inputs.survival_likelihood_label,
        survival_distribution,
        crate::survival::lognormal_kernel::FrailtySpec::None,
    )?;
    payload.link = Some(inputs.fitted_inverse_link);
    payload.linkwiggle_degree = inputs.linkwiggle_degree;
    payload.linkwiggle_knots = inputs.linkwiggle_knots;
    if let Some(timewiggle) = inputs.baseline_timewiggle {
        payload.baseline_timewiggle_degree = Some(timewiggle.degree);
        payload.baseline_timewiggle_knots = Some(timewiggle.knots);
        payload.baseline_timewiggle_penalty_orders = timewiggle.penalty_orders;
        payload.baseline_timewiggle_double_penalty = timewiggle.double_penalty;
        apply_timewiggle_beta(&mut payload, timewiggle.beta);
    }
    payload.survival_entry = inputs.survival_entry;
    payload.survival_exit = Some(inputs.survival_exit);
    payload.survival_event = Some(inputs.survival_event);
    payload.survivalspec = Some(inputs.survivalspec);
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(inputs.baseline_cfg.target).to_string());
    payload.survival_baseline_scale = inputs.baseline_cfg.scale;
    payload.survival_baseline_shape = inputs.baseline_cfg.shape;
    payload.survival_baseline_rate = inputs.baseline_cfg.rate;
    payload.survival_baseline_makeham = inputs.baseline_cfg.makeham;
    payload.apply_survival_time_basis(&inputs.time_basis);
    payload.survival_likelihood = Some(inputs.survival_likelihood_label);
    payload.survival_location_scale_structure = Some(SavedSurvivalLocationScaleStructure {
        time_parameterization: inputs.time_parameterization,
        threshold_time_basis: inputs.threshold_time_basis,
        log_sigma_time_basis: inputs.log_sigma_time_basis,
    });
    payload.formula_noise = inputs.formula_noise;
    payload.survival_distribution = survival_distribution;
    payload.resolved_termspec = Some(inputs.resolved_thresholdspec);
    payload.resolved_termspec_noise = Some(inputs.resolved_log_sigmaspec);
    source.apply_to(&mut payload);
    Ok(payload)
}

/// Source-agnostic semantic content of a latent survival / latent binary saved
/// model. The caller resolves the family (splicing the learned latent SD into
/// the persisted frailty for survival) and the model-class / likelihood labels.
pub struct LatentWindowInputs {
    pub formula: String,
    pub data_schema: DataSchema,
    pub fit_result: UnifiedFitResult,
    pub family: FittedFamily,
    pub model_class_label: String,
    pub likelihood_label: String,
    pub survival_entry: Option<String>,
    pub survival_exit: String,
    pub survival_event: String,
    pub baseline_cfg: SurvivalBaselineConfig,
    pub time_basis: SavedSurvivalTimeBasis,
    pub resolved_termspec: TermCollectionSpec,
}

/// Assemble the canonical latent survival / latent binary payload.
pub fn assemble_latent_window_payload(
    inputs: LatentWindowInputs,
    source: SavedModelSourceMetadata,
) -> FittedModelPayload {
    let mut payload = FittedModelPayload::new(
        inputs.formula,
        ModelKind::Survival,
        inputs.family,
        inputs.model_class_label,
    );
    payload.fit_result = Some(inputs.fit_result);
    payload.data_schema = Some(inputs.data_schema);
    payload.survival_entry = inputs.survival_entry;
    payload.survival_exit = Some(inputs.survival_exit);
    payload.survival_event = Some(inputs.survival_event);
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(inputs.baseline_cfg.target).to_string());
    payload.survival_baseline_scale = inputs.baseline_cfg.scale;
    payload.survival_baseline_shape = inputs.baseline_cfg.shape;
    payload.survival_baseline_rate = inputs.baseline_cfg.rate;
    payload.survival_baseline_makeham = inputs.baseline_cfg.makeham;
    payload.apply_survival_time_basis(&inputs.time_basis);
    payload.survival_likelihood = Some(inputs.likelihood_label);
    payload.resolved_termspec = Some(inputs.resolved_termspec);
    source.apply_to(&mut payload);
    payload
}

/// Copy the frontend-neutral request metadata onto a freshly assembled payload.
///
/// These three fields are *request* metadata, not fit output: nothing in the
/// fitted result can reconstruct them, so every save route has to copy them
/// across by hand, and a route that copies two of the three silently persists a
/// different model than its sibling front end does for the same canonical
/// `gam.fit-request` document. `training_table_kind` was exactly that hole: the
/// shared `fit_formula_to_payload` service (Python FFI) copied it, while every
/// `gam fit --out` save route in the CLI copied only `group_metadata` and
/// `inference_notes`, so a request document carrying `"polars"` persisted as
/// `"polars"` from Python and as the `"unknown"` default from the CLI. This
/// function is the single owner of that copy so the two cannot drift again;
/// `frontend_request_metadata_parity_2470` is the executable statement of it.
/// (#2470)
pub fn apply_request_metadata(
    payload: &mut FittedModelPayload,
    fit_config: &FitConfig,
    notes: FitNotes,
) {
    payload.group_metadata = fit_config.group_metadata.clone();
    payload.training_table_kind = fit_config.training_table_kind.clone();
    payload.inference_notes = notes.advisories;
    payload.informational_notes = notes.informational;
}

/// Record, on every certified outer point the payload carries, the fingerprint of
/// the inputs it is certified for, so a later warm start can tell a resume from a
/// new fit (gam#3002). `None` leaves the point able only to join a later search.
fn record_input_fingerprint(payload: &mut FittedModelPayload, input_fingerprint: Option<String>) {
    if let Some(record) = payload
        .fit_result
        .as_mut()
        .and_then(|fit| fit.artifacts.outer_warm_start.as_mut())
    {
        record.input_fingerprint = input_fingerprint;
    }
}

/// One authoritative "formula fit → saved payload" service: materialize once,
/// dispatch on the request variant, fit, and assemble the persistence payload.
/// Both front ends (CLI, Python FFI) must route through this function so a fit
/// requested through any surface produces an identical saved model. (#2470)
///
/// An automatic `.` term is expanded against `dataset` first, so the payload
/// stores (and `model.formula` shows) the formula that was actually fitted, and
/// the expansion's notes lead the payload's inference notes.
///
/// The fit runs on a worker of the process pool
/// ([`gam_runtime::parallel::install`]), so its parallel loops start where they
/// run: on a one-thread pool they run in place and never hand work across
/// threads.
pub fn fit_formula_to_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
) -> Result<FittedModelPayload, WorkflowError> {
    gam_runtime::parallel::install(|| fit_formula_to_payload_here(formula, dataset, fit_config))
}

fn fit_formula_to_payload_here(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
) -> Result<FittedModelPayload, WorkflowError> {
    let dataset = &*crate::fit_orchestration::drop_zero_weight_rows(dataset, fit_config)?;
    let automatic = crate::fit_orchestration::expand_automatic_fit_formula(
        &formula, dataset, fit_config,
    )?;
    let mut payload = fit_expanded_formula_to_payload(automatic.formula, dataset, fit_config)?;
    if !automatic.notes.is_empty() {
        let mut notes = automatic.notes;
        notes.append(&mut payload.inference_notes);
        payload.inference_notes = notes;
    }
    Ok(payload)
}

fn fit_expanded_formula_to_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
) -> Result<FittedModelPayload, WorkflowError> {
    let warm_start_route_refused = |route: &'static str| WorkflowError::WarmStartRefused {
        refusal: crate::fit_orchestration::WarmStartRefusal::NoSearchTakesIt { route },
    };
    if fit_config.warm_start.is_some()
        && crate::fit_orchestration::expectile_levels_for_config(fit_config)?.is_some()
    {
        return Err(warm_start_route_refused("an expectile fit"));
    }
    // A CTN chain's certified point is its outcome fit's, and that fit is a
    // function of the chain's own inputs (the stage-1 transform is fitted or
    // frozen from them), so the point is recorded against the chain's inputs.
    // The outcome fit takes the warm start through its configuration and says
    // what its searches did with it; the stage-1 fits never receive it.
    if fit_config.ctn_stage1.is_some() || fit_config.frozen_ctn.is_some() {
        let mut payload = crate::inference::ctn::fit_chain(formula.clone(), dataset, fit_config)?;
        record_input_fingerprint(
            &mut payload,
            crate::fit_orchestration::fit_input_fingerprint(&formula, dataset, fit_config),
        );
        return Ok(payload);
    }
    // Expectile (Newey–Powell LAWS) family (#1777): the expectile estimator is an
    // OUTER driver that wraps the standard Gaussian-identity GAM with iterative
    // asymmetric reweighting, so it is selected *before* `materialize` (which has
    // no expectile arm). It runs through the same resolve → structural start →
    // resolution-loop owner the library call uses, so a one-level fit reaches the
    // same basis on every front end (#4062), and the inner materialization's notes
    // are carried like any other fit's (#4027). One level returns an ordinary
    // `StandardFitResult`, assembled by the shared `assemble_standard_payload`.
    if expectile_levels_for_config(fit_config)?.is_some() {
        let outcome = fit_formula_through_adaptive_resolution(&formula, dataset, fit_config)?;
        let mut payload = match outcome.result {
            FitResult::Standard(result) => assemble_standard_payload(StandardPayloadInputs {
                formula,
                dataset,
                fit_config,
                result,
            })?,
            FitResult::ExpectileLocationScale(joint) => {
                payload_for_joint_expectile(formula, dataset, fit_config, joint)?
            }
            _ => {
                return Err(WorkflowError::SchemaMismatch {
                    reason: "an expectile request returned a non-expectile fit result".to_string(),
                });
            }
        };
        apply_request_metadata(&mut payload, fit_config, outcome.inference_notes);
        payload.unidentified_scalar_terms = outcome.unidentified_scalar_terms;
        return Ok(payload);
    }
    // Standard-fit dispatch must materialize at the adaptive structural start:
    // this request becomes the first fitted design below. Other estimator
    // materializers do not consume this standard-only orchestration field.
    let mut dispatch_config = fit_config.clone();
    dispatch_config.adaptive_resolution = Some(Vec::new());
    let formula_for_fingerprint = formula.clone();
    let materialized = materialize(&formula, dataset, &dispatch_config)?;
    let request = materialized.request;
    // The time basis THIS materialization built, carried to the save path so a
    // survival payload records the basis its own fit used instead of a second,
    // independently re-derived one (#2470).
    let survival_time_basis = materialized.survival_time_basis;
    // Advisories produced while materializing (e.g. the mgcv-style "k reduced to
    // the data support" / basis-degradation notes from the cr/cs/sz cap, #1541
    // #1542). The CLI prints these via `print_inference_summary`; the Python
    // path used to drop them on the floor, so a gamfit user whose basis was
    // silently capped got no signal at all (#1543). Carry them into the
    // serialized payload so gamfit can surface them as `GamInferenceWarning`s
    // and via `model.notes`.
    let mut inference_notes = materialized.inference_notes;
    // The typed record of scalar terms materialization removed as unidentified,
    // published beside the notes (#2627).
    let unidentified_scalar_terms = materialized.unidentified_scalar_terms;

    let mut payload = match request {
        FitRequest::Standard(standard_request) => {
            // Fit the request that selected this arm, then hand its converged
            // result to the same loop owner the CLI uses. Re-entering the
            // formula entry point here used to materialize the spatial design a
            // second time; before the adaptive loop landed, the first discarded
            // design was also the old fully provisioned rank (#1689).
            let standard_spec = standard_request.spec.clone();
            let initial_notes = std::mem::take(&mut inference_notes);
            let outcome = fit_materialized_standard_with_notes(
                &formula,
                dataset,
                fit_config,
                standard_request,
                initial_notes,
            )?;
            inference_notes = outcome.inference_notes;
            match outcome.result {
                FitResult::Standard(standard_result) => {
                    assemble_standard_payload(StandardPayloadInputs {
                        formula,
                        dataset,
                        fit_config,
                        result: standard_result,
                    })?
                }
                FitResult::SplineScan(scan) => {
                    // The scan detection is structural on the materialized
                    // shape, so the dispatch request's single smooth is the
                    // same 1-D B-spline the entry point scan-routed.
                    let feature_col = match &standard_spec.smooth_terms[0].basis {
                        gam_terms::smooth::SmoothBasisSpec::BSpline1D { feature_col, .. } => {
                            *feature_col
                        }
                        _ => {
                            return Err(WorkflowError::SchemaMismatch {
                                reason: "spline-scan detection accepted a non-1D basis".to_string(),
                            });
                        }
                    };
                    let feature_column =
                        dataset.headers.get(feature_col).cloned().ok_or_else(|| {
                            WorkflowError::SchemaMismatch {
                                reason: format!(
                                    "spline-scan feature column {feature_col} has no header"
                                ),
                            }
                        })?;
                    let mut scan_payload = assemble_spline_scan_payload(
                        formula,
                        feature_column,
                        &scan,
                        dataset.schema.clone(),
                        dataset.headers.clone(),
                        dataset.feature_ranges(),
                    );
                    scan_payload.weight_column = fit_config.weight_column.clone();
                    apply_request_metadata(&mut scan_payload, fit_config, inference_notes);
                    return Ok(scan_payload);
                }
                FitResult::ResidualCascade(cascade) => {
                    // The cascade fires only for a single scattered radial
                    // smooth; recover its feature columns from the dispatch
                    // request the same way the CLI does from its parsed
                    // formula.
                    let feature_cols = standard_spec
                        .smooth_terms
                        .iter()
                        .find_map(|term| match &term.basis {
                            gam_terms::smooth::SmoothBasisSpec::ThinPlate {
                                feature_cols, ..
                            }
                            | gam_terms::smooth::SmoothBasisSpec::Duchon {
                                feature_cols, ..
                            }
                            | gam_terms::smooth::SmoothBasisSpec::Matern {
                                feature_cols, ..
                            } => Some(feature_cols.clone()),
                            _ => None,
                        })
                        .ok_or_else(|| WorkflowError::SchemaMismatch {
                            reason: "residual-cascade result has no radial smooth in the \
                                     materialized request"
                                .to_string(),
                        })?;
                    let feature_columns = feature_cols
                        .into_iter()
                        .map(|col| {
                            dataset.headers.get(col).cloned().ok_or_else(|| {
                                WorkflowError::SchemaMismatch {
                                    reason: format!(
                                        "residual-cascade feature column {col} has no header"
                                    ),
                                }
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut cascade_payload = assemble_residual_cascade_payload(
                        formula,
                        feature_columns,
                        &cascade,
                        dataset.schema.clone(),
                        dataset.headers.clone(),
                        dataset.feature_ranges(),
                    )
                    .map_err(|reason| {
                        WorkflowError::Fit(crate::fit_orchestration::FitFailure::raised(
                            gam_problem::FailureCategory::Invariant,
                            reason,
                        ))
                    })?;
                    apply_request_metadata(&mut cascade_payload, fit_config, inference_notes);
                    return Ok(cascade_payload);
                }
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the standard workflow to return a standard fit result"
                            .to_string(),
                    });
                }
            }
        }
        FitRequest::TransformationNormal(tn_request) => {
            let fit_result = fit_model(FitRequest::TransformationNormal(tn_request))?;
            let tn_result = match fit_result {
                FitResult::TransformationNormal(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the transformation-normal workflow to return a transformation-normal fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_transformation_normal(formula, dataset, fit_config, tn_result)?
        }
        FitRequest::BernoulliMarginalSlope(ms_request) => {
            let base_link = ms_request.spec.base_link.clone();
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::BernoulliMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::BernoulliMarginalSlope(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the bernoulli marginal-slope workflow to return a marginal-slope fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_bernoulli_marginal_slope(
                formula,
                dataset,
                fit_config,
                base_link,
                frailty,
                ms_result,
            )?
        }
        FitRequest::SurvivalMarginalSlope(ms_request) => {
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::SurvivalMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::SurvivalMarginalSlope(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival marginal-slope workflow to return a survival marginal-slope fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_survival_marginal_slope(formula, dataset, fit_config, frailty, ms_result)?
        }
        FitRequest::GaussianLocationScale(ls_request) => {
            let fit_result = fit_model(FitRequest::GaussianLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::GaussianLocationScale(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the gaussian location-scale workflow to return a gaussian location-scale fit result"
                            .to_string(),
                    });
                }
            };
            // Persist the response standardization factor and the σ floor the
            // fit applied so prediction reconstructs the raw floor at
            // `response_scale·sigma_floor`, keeping predictive σ
            // response-scale-equivariant (#884). The fit already mapped the
            // log-σ `exp(η)` term to raw units via the `+ln(response_scale)`
            // intercept shift; only the additive floor still needs the factor at
            // reconstruction time.
            payload_for_gaussian_location_scale(formula, dataset, fit_config, ls_result)?
        }
        FitRequest::BinomialLocationScale(ls_request) => {
            let link_kind = ls_request.spec.link_kind.clone();
            let fit_result = fit_model(FitRequest::BinomialLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::BinomialLocationScale(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the binomial location-scale workflow to return a binomial location-scale fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_binomial_location_scale(
                formula,
                dataset,
                fit_config,
                link_kind,
                ls_result,
            )?
        }
        FitRequest::SurvivalLocationScale(ls_request) => {
            let fit_result = fit_model(FitRequest::SurvivalLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::SurvivalLocationScale(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival location-scale workflow to return a survival location-scale fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_survival_location_scale(
                formula,
                dataset,
                fit_config,
                ls_result,
                survival_time_basis,
            )?
        }
        FitRequest::SurvivalTransformation(rp_request) => {
            let fit_result = fit_model(FitRequest::SurvivalTransformation(rp_request))?;
            let rp_result = match fit_result {
                FitResult::SurvivalTransformation(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the survival transformation workflow to return a survival transformation fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_survival_transformation(formula, dataset, fit_config, rp_result)?
        }
        FitRequest::LatentSurvival(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentSurvival(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentSurvival(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the latent survival workflow to return a latent survival fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_latent_survival(
                formula,
                dataset,
                fit_config,
                frailty,
                lat_result,
                survival_time_basis,
            )?
        }
        FitRequest::LatentBinary(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentBinary(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentBinary(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the latent binary workflow to return a latent binary fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_latent_binary(
                formula,
                dataset,
                fit_config,
                frailty,
                lat_result,
                survival_time_basis,
            )?
        }
        FitRequest::DispersionLocationScale(ls_request) => {
            // Genuine-dispersion location-scale family (#913): NB / Gamma / Beta
            // / Tweedie mean families whose `noise_formula` models the
            // overdispersion channel. Magic-detected upstream from a
            // `noise_formula` on one of those families; the FFI freezes the mean
            // and log-precision specs and persists them via the same shared
            // location-scale assembler the CLI uses.
            let kind = ls_request.spec.kind;
            let fit_result = fit_model(FitRequest::DispersionLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::DispersionLocationScale(result) => result,
                _ => {
                    return Err(WorkflowError::SchemaMismatch {
                        reason: "python binding expected the dispersion location-scale workflow to return a dispersion location-scale fit result"
                            .to_string(),
                    });
                }
            };
            payload_for_dispersion_location_scale(formula, dataset, fit_config, kind, ls_result)?
        }
    };
    // A route whose outer driver never received the model's point fitted cold;
    // that is not the warm start the caller asked for. One that received it says
    // what it did in the model's notes, so a point it did not use is never silent.
    if let Some(warm_start) = fit_config.warm_start.as_ref() {
        match warm_start.recorded() {
            None => return Err(warm_start_route_refused("this fit's route")),
            // A point that was used is what the caller asked for; one that was
            // not is a fit that differs from the request.
            Some(gam_model_api::WarmStartOutcome::Resumed) => inference_notes.inform(
                "warm_start_from: resumed from the model's certified point (same inputs)"
                    .to_string(),
            ),
            Some(gam_model_api::WarmStartOutcome::JoinedMultistart) => inference_notes.inform(
                "warm_start_from: the model's certified point joined the multistart as one \
                 more seed (other inputs)"
                    .to_string(),
            ),
            Some(gam_model_api::WarmStartOutcome::NotUsed(reason)) => inference_notes.advise(
                format!("warm_start_from: the model's certified point was not used: {reason}"),
            ),
        }
    }
    record_input_fingerprint(
        &mut payload,
        crate::fit_orchestration::fit_input_fingerprint(
            &formula_for_fingerprint,
            dataset,
            fit_config,
        ),
    );
    payload.unidentified_scalar_terms = unidentified_scalar_terms;
    apply_request_metadata(&mut payload, fit_config, inference_notes);
    Ok(payload)
}

fn payload_for_transformation_normal(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    tn_result: TransformationNormalFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_covariate = freeze_term_collection_from_design(
        &tn_result.covariate_spec_resolved,
        &tn_result.covariate_design,
    )
    .map_err(|err| format!("failed to freeze transformation-normal covariate spec: {err}"))?;

    // Thin adapter over the shared core assembler; the FFI freezes the
    // covariate spec from its design and reads the offset column from the
    // FitConfig. See `assemble_transformation_normal_payload`.
    assemble_transformation_normal_payload(
        TransformationNormalInputs {
            formula,
            data_schema: dataset.schema.clone(),
            resolved_covariate_spec: frozen_covariate,
            fit_result: tn_result.fit.clone(),
            family: &tn_result.family,
            score_calibration: tn_result.score_calibration.clone(),
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: None,
        },
    )
}

fn payload_for_bernoulli_marginal_slope(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    base_link: InverseLink,
    frailty: crate::survival::lognormal_kernel::FrailtySpec,
    ms_result: BernoulliMarginalSlopeFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_marginal = freeze_term_collection_from_design(
        &ms_result.marginalspec_resolved,
        &ms_result.marginal_design,
    )
    .map_err(|err| format!("failed to freeze marginal spec: {err}"))?;
    let frozen_slope = freeze_term_collection_from_design(
        &ms_result.slopespec_resolved,
        &ms_result.slope_design,
    )
    .map_err(|err| format!("failed to freeze slope spec: {err}"))?;

    let slope_formula = fit_config
        .slope_formula
        .clone()
        .ok_or_else(|| "bernoulli marginal-slope requires slope_formula".to_string())?;
    let z_column = fit_config
        .z_column
        .clone()
        .ok_or_else(|| "bernoulli marginal-slope requires z_column".to_string())?;

    // Thin adapter over the shared core assembler. The FFI's source-specific
    // work is freezing term collections from their designs, reading the
    // slope formula / z column / offset columns from the FitConfig, and
    // persisting headers without per-feature ranges; the semantic payload is
    // assembled by the same core path the CLI uses, so the two save routes
    // produce identical contracts.
    assemble_bernoulli_marginal_slope_payload(
        BernoulliMarginalSlopeInputs {
            formula,
            data_schema: dataset.schema.clone(),
            slope_formula,
            z_column,
            resolved_marginalspec: frozen_marginal,
            resolved_slopespec: frozen_slope,
            fit_result: ms_result.fit.clone(),
            p_marginal: ms_result.marginal_design.design.ncols(),
            baseline_marginal: ms_result.baseline_marginal,
            baseline_slope: ms_result.baseline_slope,
            latent_z_normalization: SavedLatentZNormalization {
                mean: ms_result.z_normalization.mean,
                sd: ms_result.z_normalization.sd,
            },
            latent_measure: ms_result.latent_measure.clone(),
            latent_law_consumed: ms_result.latent_law_consumed.clone(),
            latent_z_conditional_calibration: ms_result.latent_z_conditional_calibration.clone(),
            score_warp_runtime: ms_result.score_warp_runtime.as_ref(),
            link_dev_runtime: ms_result.link_dev_runtime.as_ref(),
            base_link,
            frailty,
            residual_repair: ms_result.residual_repair.clone(),
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            // Every other adapter persists per-feature ranges; this arm alone
            // passed `None`, so Python-saved Bernoulli marginal-slope models
            // were the only ones that could not clip out-of-hull predict rows
            // (#2470).
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

fn payload_for_survival_marginal_slope(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    frailty: crate::survival::lognormal_kernel::FrailtySpec,
    ms_result: SurvivalMarginalSlopeFitResult,
) -> Result<FittedModelPayload, String> {
    use crate::survival::construction::{
        build_survival_time_basis, parse_survival_likelihood_mode,
        parse_survival_time_basis_config, resolve_survival_time_anchor_for_mode,
        survival_likelihood_modename,
    };
    use ndarray::s;

    let frozen_marginal = freeze_term_collection_from_design(
        &ms_result.marginalspec_resolved,
        &ms_result.marginal_design,
    )
    .map_err(|err| format!("failed to freeze survival marginal spec: {err}"))?;
    let frozen_slope = freeze_term_collection_from_design(
        &ms_result.slopespec_resolved,
        &ms_result.slope_design,
    )
    .map_err(|err| format!("failed to freeze survival slope spec: {err}"))?;

    let slope_formula = fit_config
        .slope_formula
        .clone()
        .unwrap_or_else(|| "same-as-main".to_string());
    let z_column = fit_config
        .z_column
        .clone()
        .ok_or_else(|| "survival marginal-slope requires z_column".to_string())?;
    let parsed = parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival marginal formula: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival marginal-slope FFI requires Surv(...) response".to_string())?;
    let col_map: HashMap<String, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    // `entryname == None` is the right-censored shorthand `Surv(time, event)`:
    // entry times are synthesized as zero, no column lookup required.
    let entry_idx: Option<usize> = entryname
        .as_deref()
        .map(|name| {
            col_map
                .get(name)
                .copied()
                .ok_or_else(|| format!("entry column '{name}' not found"))
        })
        .transpose()?;
    let exit_idx = *col_map
        .get(&exitname)
        .ok_or_else(|| format!("exit column '{exitname}' not found"))?;
    let n = dataset.values.nrows();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let entry_val = entry_idx.map_or(0.0, |idx| dataset.values[[i, idx]]);
        let (t0, t1) = crate::survival::construction::normalize_survival_time_pair(
            entry_val,
            dataset.values[[i, exit_idx]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    // The request may leave baseline parameters to estimation. Replaying its
    // initial values here would save a different time chart from the one the
    // returned coefficients use; persist the certified fitted chart instead.
    let baseline_cfg = ms_result.baseline_config.clone();
    let likelihood_mode = parse_survival_likelihood_mode(fit_config.resolved_survival_likelihood())?;
    let time_cfg = if parsed.timewiggle.is_some() {
        crate::survival::construction::SurvivalTimeBasisConfig::None
    } else {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
        )?
    };
    // Re-derivation, so it must ask the same question the fit asked — including
    // the caller's explicit anchor, which this site used to ignore, persisting
    // the median exit onto a model whose fit centered somewhere else (#2631).
    let time_anchor = resolve_survival_time_anchor_for_mode(
        likelihood_mode,
        &age_entry,
        &age_exit,
        fit_config.survival_time_anchor,
    )?;
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some(fit_config.time_num_internal_knots),
    )?;
    let timewiggle = match (
        ms_result.time_wiggle_knots.as_ref(),
        ms_result.time_wiggle_degree,
        ms_result.time_wiggle_ncols,
    ) {
        (None, None, 0) => None,
        (Some(knots), Some(degree), ncols) if ncols > 0 => {
            let beta_time = &ms_result
                .fit
                .blocks
                .first()
                .ok_or_else(|| {
                    "survival marginal-slope FFI fit is missing its time block".to_string()
                })?
                .beta;
            let p_base = time_build.x_exit_time.ncols();
            if beta_time.len() != p_base + ncols {
                return Err(format!(
                    "survival marginal-slope FFI timewiggle width mismatch: time beta={}, base={p_base}, wiggle={ncols}",
                    beta_time.len(),
                ));
            }
            Some(SurvivalTimewiggle {
                degree,
                knots: knots.to_vec(),
                penalty_orders: parsed
                    .timewiggle
                    .as_ref()
                    .map(|config| config.penalty_orders.clone()),
                double_penalty: parsed
                    .timewiggle
                    .as_ref()
                    .map(|config| config.double_penalty),
                beta: SurvivalTimewiggleBeta::Single(beta_time.slice(s![p_base..]).to_vec()),
            })
        }
        _ => {
            return Err(
                "survival marginal-slope FFI fit has incomplete timewiggle authority".to_string(),
            );
        }
    };
    let persisted_conditional = ms_result.persisted_latent_z_calibrations()?;
    // gam#2929: a K ≥ 2 per-score fit anchored on the joint law of its score
    // vector persists that law, one score column and one slope surface per
    // coordinate; the law carries each coordinate's unit map (gam#4331). What
    // the single-score contract cannot carry is refused here.
    let joint_state = match ms_result.joint_latent_law.as_ref() {
        None => None,
        Some(law) => {
            let k = law.score_dim;
            let surface_specs = ms_result
                .slope_surface_specs
                .as_ref()
                .filter(|specs| specs.len() == k)
                .ok_or_else(|| {
                    format!(
                        "survival marginal-slope joint latent law is K={k} but the fit carries no \
                         per-score slope surface specs to rebuild its surfaces"
                    )
                })?;
            if let Some(reason) = joint_latent_law_calibration_save_refusal(
                persisted_conditional.as_ref(),
                &ms_result.latent_z_calibrations,
                law,
            ) {
                return Err(reason);
            }
            if law.conditional.is_some() && !ms_result.latent_conditioning_reproducible {
                return Err(
                    "survival marginal-slope joint latent law transports its law by a conditional \
                     covariance fitted on the marginal design frozen before the spatial \
                     length-scale search, and that search then moved the design: prediction \
                     would rebuild a different span. Pin the marginal formula's spatial \
                     length_scale= if this model must be saved"
                        .to_string(),
                );
            }
            let (_, parsed_slope) = gam_terms::inference::formula_dsl::parse_matching_auxiliary_formula(
                &slope_formula,
                &parsed.response,
                "slope_formula",
            )
            .map_err(|err| format!("failed to re-parse survival slope formula: {err}"))?;
            let surfaces =
                gam_terms::inference::formula_dsl::marginal_slope_surfaces(&parsed_slope, &z_column)?;
            if surfaces.len() != k {
                return Err(format!(
                    "survival marginal-slope joint latent law is K={k} but the slope formula \
                     declares {} surfaces",
                    surfaces.len()
                ));
            }
            Some((
                law.clone(),
                surfaces
                    .into_iter()
                    .map(|surface| surface.z_column)
                    .collect::<Vec<_>>(),
                surface_specs.clone(),
            ))
        }
    };

    // Thin adapter over the shared core assembler. The FFI's source-specific
    // work is re-deriving the survival response columns, baseline config, and
    // time basis from the formula + FitConfig and freezing its term collections
    // from their designs; the semantic payload is assembled by the same core
    // path the CLI uses, so the two save routes produce identical contracts.
    let mut payload = assemble_survival_marginal_slope_payload(
        SurvivalMarginalSlopeInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result: ms_result.fit.clone(),
            frailty,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: "net".to_string(),
            baseline_cfg,
            time_basis: SavedSurvivalTimeBasis::from_build(&time_build, time_anchor),
            survival_likelihood_label: survival_likelihood_modename(likelihood_mode).to_string(),
            resolved_marginalspec: frozen_marginal,
            resolved_slopespec: frozen_slope,
            slope_time_basis: ms_result.slope_time_basis.clone(),
            slope_formula,
            z_column,
            latent_z_normalization: SavedLatentZNormalization {
                mean: ms_result.z_normalization.mean,
                sd: ms_result.z_normalization.sd,
            },
            latent_law_consumed: ms_result.latent_law_consumed.clone(),
            latent_z_conditional_calibration: persisted_conditional,
            latent_measure: ms_result.latent_measure.clone(),
            declared_latent_law: ms_result.declared_latent_law.clone(),
            declared_latent_law_compression: ms_result
                .latent_law_compression
                .as_ref()
                .map(crate::inference::model::SavedDeclaredLawCompression::from),
            baseline_slope: ms_result.baseline_slope,
            timewiggle,
            score_warp_runtime: ms_result.score_warp_runtime.as_ref(),
            link_dev_runtime: ms_result.link_dev_runtime.as_ref(),
            influence_absorber_width: ms_result.influence_absorber_width,
            influence_absorber_design: ms_result.influence_absorber_design.as_ref(),
            score_covariance: ms_result.persistable_score_covariance()?,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )?;
    if let Some((law, z_columns, surface_specs)) = joint_state {
        payload.z_columns = Some(z_columns);
        payload.resolved_slopespecs = Some(surface_specs);
        payload.survival_marginal_slope_joint_latent_law = Some(law);
    }
    Ok(payload)
}

/// Why a fit anchored on the joint latent law of `K ≥ 2` scores cannot be saved
/// with the latent-score calibrations it persisted, or `None` when it can
/// (gam#2929). The saved joint-law contract replays the anchor on the raw score
/// columns, so a model whose scores were calibrated before the fit would predict
/// on scores other than the ones it was fitted on.
/// The one place a `K ≥ 2` model's score maps can be lost, checked where it
/// would happen (gam#2929, gam#2949).
///
/// The fit reads each coordinate on the axis its own latent-law gate chose, so
/// a coordinate the gate calibrated reaches the row program as
/// `ζ = (z − m(a))/√v(a)`. Prediction must read a new score on that same axis,
/// and the object it reads every coordinate against is the joint latent law —
/// so the law carries one map per coordinate and the single-surface payload
/// field stays empty. Two ways that can go wrong, both refused here rather than
/// written:
///
/// * the fit calibrated a coordinate and the law does not carry the maps, which
///   would give prediction an uncalibrated axis for it;
/// * the payload ALSO carries a scalar map beside the law's, which would map a
///   coordinate twice.
fn joint_latent_law_calibration_save_refusal(
    conditional: Option<&crate::bms::LatentZConditionalCalibration>,
    fitted: &[crate::bms::LatentMeasureCalibration],
    law: &crate::survival::marginal_slope::SurvivalJointLatentLaw,
) -> Option<String> {
    let calibrated: Vec<usize> = fitted
        .iter()
        .enumerate()
        .filter(|(_, calibration)| {
            !matches!(calibration, crate::bms::LatentMeasureCalibration::None)
        })
        .map(|(column, _)| column)
        .collect();
    if conditional.is_some() {
        return Some(
            "survival marginal-slope K ≥ 2 model carries a scalar latent-z conditional \
             calibration beside its joint latent law's per-coordinate maps: prediction reads \
             every coordinate against the law, so a second copy of one coordinate's map would \
             apply it twice. Saving is refused rather than writing a model whose prediction \
             evaluates different scores"
                .to_string(),
        );
    }
    if !calibrated.is_empty() && law.score_calibrations.is_empty() {
        return Some(format!(
            "survival marginal-slope K ≥ 2 fit calibrated latent-score column(s) {calibrated:?} \
             before the row program read them, and its joint latent law carries no per-coordinate \
             map: prediction would read those columns on an uncalibrated axis. Saving is refused \
             rather than writing a model whose prediction evaluates different scores"
        ));
    }
    None
}

fn payload_for_survival_transformation(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    rp_result: crate::fit_orchestration::SurvivalTransformationFitResult,
) -> Result<FittedModelPayload, String> {
    use crate::survival::construction::survival_likelihood_modename;
    use ndarray::s;

    let parsed = parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival transformation formula: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival transformation FFI requires Surv(...) response".to_string())?;
    let likelihood_label = survival_likelihood_modename(rp_result.likelihood_mode).to_string();

    let cause_count = rp_result.fit.blocks.len().max(1);
    let is_joint_cause_specific = cause_count > 1;

    // Source-specific work: extract the baseline-timewiggle coefficients from
    // the differently-shaped fit struct (one block for net, one per cause for
    // joint cause-specific). The canonical payload is then assembled by the same
    // shared core the CLI uses.
    let timewiggle = rp_result
        .baseline_timewiggle
        .as_ref()
        .map(|timewiggle| -> Result<SurvivalTimewiggle, String> {
            let start = rp_result.time_base_ncols;
            let end = start + timewiggle.ncols;
            let beta = if is_joint_cause_specific {
                let mut by_cause = Vec::with_capacity(cause_count);
                for (cause_idx, block) in rp_result.fit.blocks.iter().enumerate() {
                    if block.beta.len() < end {
                        return Err(format!(
                            "joint cause-specific survival timewiggle beta mismatch for cause {}: beta has {}, needs {end}",
                            cause_idx + 1,
                            block.beta.len()
                        ));
                    }
                    by_cause.push(block.beta.slice(s![start..end]).to_vec());
                }
                SurvivalTimewiggleBeta::ByCause(by_cause)
            } else {
                let beta = &rp_result.fit.beta;
                if beta.len() < end {
                    return Err(format!(
                        "survival transformation timewiggle beta mismatch: beta has {}, needs {end}",
                        beta.len()
                    ));
                }
                SurvivalTimewiggleBeta::Single(beta.slice(s![start..end]).to_vec())
            };
            Ok(SurvivalTimewiggle {
                degree: timewiggle.degree,
                knots: timewiggle.knots.to_vec(),
                penalty_orders: parsed.timewiggle.as_ref().map(|cfg| cfg.penalty_orders.clone()),
                double_penalty: parsed.timewiggle.as_ref().map(|cfg| cfg.double_penalty),
                beta,
            })
        })
        .transpose()?;

    let payload = assemble_survival_transformation_payload(
        SurvivalTransformationInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result: rp_result.fit.clone(),
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: if is_joint_cause_specific {
                "cause-specific".to_string()
            } else {
                "net".to_string()
            },
            cause_count: is_joint_cause_specific.then_some(cause_count),
            baseline_cfg: rp_result.baseline_cfg.clone(),
            time_basis: rp_result.time_basis.clone(),
            survival_likelihood_label: likelihood_label,
            resolved_termspec: rp_result.resolvedspec,
            timewiggle,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: None,
        },
    )?;
    Ok(payload)
}

/// The saved model of a Gaussian location-scale fit: the builder
/// `fit_formula_to_payload` uses. It is public so a caller that keeps the fit
/// result, and with it the fitted block states the payload does not persist,
/// saves that fit through the same route (#3001).
pub fn payload_for_gaussian_location_scale(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    ls_result: GaussianLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    let response_scale = ls_result.response_scale;
    let sigma_floor = ls_result.sigma_floor;
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze gaussian location-scale mean spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze gaussian location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "gaussian location-scale requires noise_formula".to_string())?;

    let fit = ls_result.fit.fit;
    let wiggle = location_scale_wiggle_from_parts(ls_result.wiggle_knots, ls_result.wiggle_degree);

    // Thin adapter over the shared core assembler; the FFI freezes the mean and
    // noise specs from their designs and reads offset columns from the
    // FitConfig. See `assemble_location_scale_payload`.
    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            wiggle,
        },
        LocationScaleResponse::Gaussian {
            response_scale,
            sigma_floor,
            base_link: None,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

/// Saved payload of a joint multi-level expectile fit: the Gaussian
/// location-scale payload of its `μ`/`σ` surfaces, tagged with the joint
/// estimator that turns them into one non-crossing curve per level.
fn payload_for_joint_expectile(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    joint: ExpectileLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    let noise_formula = crate::fit_orchestration::expectile_noise_formula(&formula, fit_config)
        .map_err(|error| error.to_string())?;
    let location_scale_config = FitConfig {
        noise_formula: Some(noise_formula),
        ..fit_config.clone()
    };
    let mut payload = payload_for_gaussian_location_scale(
        formula,
        dataset,
        &location_scale_config,
        joint.location_scale,
    )?;
    payload.family = JOINT_EXPECTILE_FAMILY_TAG.to_string();
    payload.estimator = FittedEstimator::ExpectileLocationScale {
        levels: joint.levels,
        standardized_expectiles: joint.standardized_expectiles,
    };
    Ok(payload)
}

/// Map the optional `(knots, degree)` link-wiggle basis a location-scale fit
/// may produce into the shared [`LocationScaleWiggle`] form. Both are present
/// together or not at all; the coefficients stay in the fit's `LinkWiggle`
/// block.
fn location_scale_wiggle_from_parts(
    knots: Option<Array1<f64>>,
    degree: Option<usize>,
) -> Option<LocationScaleWiggle> {
    match (knots, degree) {
        (Some(knots), Some(degree)) => Some(LocationScaleWiggle {
            knots: knots.to_vec(),
            degree,
        }),
        _ => None,
    }
}

fn payload_for_binomial_location_scale(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    link_kind: InverseLink,
    ls_result: BinomialLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze binomial location-scale threshold spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze binomial location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "binomial location-scale requires noise_formula".to_string())?;

    let fit = ls_result.fit.fit;
    let wiggle = location_scale_wiggle_from_parts(ls_result.wiggle_knots, ls_result.wiggle_degree);

    // Thin adapter over the shared core assembler; the FFI freezes the threshold
    // and noise specs from their designs and reads offset columns from the
    // FitConfig. See `assemble_location_scale_payload`.
    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            wiggle,
        },
        LocationScaleResponse::Binomial { link: link_kind },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

/// Assemble the saved-model payload for a genuine-dispersion location-scale fit
/// (#913): NegativeBinomial / Gamma / Beta / Tweedie with a `noise_formula` on
/// the overdispersion channel. Mirrors the CLI dispersion save path
/// (`assemble_location_scale_payload` + `LocationScaleResponse::Dispersion`),
/// deriving the persisted likelihood and mean base-link from the single
/// source of truth on [`DispersionFamilyKind`]. The log-precision block
/// coefficients are the fit's `BlockRole::Scale` block; there is no
/// link-wiggle and no response standardization for these families.
fn payload_for_dispersion_location_scale(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    kind: DispersionFamilyKind,
    ls_result: DispersionLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze dispersion location-scale mean spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze dispersion location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "dispersion location-scale requires noise_formula".to_string())?;

    let fit = ls_result.fit.fit;

    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            wiggle: None,
        },
        LocationScaleResponse::Dispersion {
            likelihood: kind.likelihood_spec(),
            base_link: kind.base_link(),
            family_tag: kind.family_tag(),
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

fn payload_for_survival_location_scale(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    ls_result: crate::fit_orchestration::SurvivalLocationScaleFitResult,
    time_basis: Option<SavedSurvivalTimeBasis>,
) -> Result<FittedModelPayload, String> {
    use crate::survival::construction::{
        SurvivalLikelihoodMode, parse_survival_baseline_config, survival_likelihood_modename,
    };
    // The time basis is CARRIED from the materialization that produced this fit
    // (#2470). It is not re-derived here: `materialize_survival` switches the
    // time anchor to the robust interior exit time whenever the data is left
    // truncated, and the re-derivation this replaced always took the
    // earliest-entry anchor — so a left-truncated model persisted an anchor its
    // own fit never centered at, and predict then re-centered the design in a
    // different affine frame than the coefficients were fitted in.
    let time_basis = time_basis.ok_or_else(|| {
        "survival location-scale payload requires the materialized survival time basis".to_string()
    })?;
    let parsed = parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival formula for FFI payload: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival location-scale FFI requires Surv(...) response".to_string())?;
    // A nonlinear baseline's shape is selected by the fit together with ρ
    // (#3413), so the saved baseline is the fitted one; a linear baseline has
    // no coordinates and is the configured one.
    let baseline_cfg = match ls_result.fit.baseline_config.clone() {
        Some(fitted) => fitted,
        None => parse_survival_baseline_config(
            &fit_config.baseline_target,
            fit_config.baseline_scale,
            fit_config.baseline_shape,
            fit_config.baseline_rate,
            fit_config.baseline_makeham,
        )?,
    };

    let fitted_inverse_link = ls_result.inverse_link.clone();
    // Compact the inner UnifiedFitResult and apply the fitted link state so
    // downstream prediction can recover the inverse-link parameters from the
    // saved fit_result. Mirrors the CLI's
    // compact_saved_survival_location_scale_fit_result helper.
    let mut fit_result = ls_result.fit.fit.clone();
    apply_inverse_link_state_to_fit_result(&mut fit_result, &fitted_inverse_link);
    fit_result.artifacts.survival_link_wiggle_knots = ls_result.wiggle_knots.clone();
    fit_result.artifacts.survival_link_wiggle_degree = ls_result.wiggle_degree;

    let resolved_thresholdspec = freeze_term_collection_from_design(
        &ls_result.fit.resolved_thresholdspec,
        &ls_result.fit.threshold_design,
    )
    .map_err(|err| err.to_string())?;
    let resolved_log_sigmaspec = freeze_term_collection_from_design(
        &ls_result.fit.resolved_log_sigmaspec,
        &ls_result.fit.log_sigma_design,
    )
    .map_err(|err| err.to_string())?;

    // Thin adapter over the shared core assembler. The FFI's source-specific
    // work above re-derives the survival metadata and compacts the fit result
    // with the fitted link state; the canonical payload is assembled by the
    // same path the CLI uses.
    assemble_survival_location_scale_payload(
        SurvivalLocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result,
            fitted_inverse_link: fitted_inverse_link.clone(),
            linkwiggle_degree: ls_result.wiggle_degree,
            linkwiggle_knots: ls_result.wiggle_knots.as_ref().map(|k| k.to_vec()),
            baseline_timewiggle: None,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: "net".to_string(),
            baseline_cfg,
            time_basis,
            // A location-scale fit result is one whatever the configuration named:
            // a noise formula or `linkwiggle(...)` selects this model under the
            // default `transformation` likelihood.
            survival_likelihood_label: survival_likelihood_modename(
                SurvivalLikelihoodMode::LocationScale,
            )
            .to_string(),
            time_parameterization: ls_result.fit.time_parameterization,
            threshold_time_basis: ls_result.fit.threshold_time_basis.clone(),
            log_sigma_time_basis: ls_result.fit.log_sigma_time_basis.clone(),
            formula_noise: fit_config.noise_formula.clone(),
            resolved_thresholdspec,
            resolved_log_sigmaspec,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    )
}

fn payload_for_latent_survival(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: crate::survival::lognormal_kernel::FrailtySpec,
    lat_result: crate::survival::latent::LatentSurvivalTermFitResult,
    time_basis: Option<SavedSurvivalTimeBasis>,
) -> Result<FittedModelPayload, String> {
    payload_for_latent_window(
        formula,
        dataset,
        fit_config,
        request_frailty,
        lat_result.fit,
        lat_result.resolvedspec,
        lat_result.design,
        Some(lat_result.latent_sd),
        lat_result.baseline_config,
        true,
        time_basis,
    )
}

fn payload_for_latent_binary(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: crate::survival::lognormal_kernel::FrailtySpec,
    lat_result: crate::survival::latent::LatentBinaryTermFitResult,
    time_basis: Option<SavedSurvivalTimeBasis>,
) -> Result<FittedModelPayload, String> {
    payload_for_latent_window(
        formula,
        dataset,
        fit_config,
        request_frailty,
        lat_result.fit,
        lat_result.resolvedspec,
        lat_result.design,
        None,
        lat_result.baseline_config,
        false,
        time_basis,
    )
}

fn payload_for_latent_window(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: crate::survival::lognormal_kernel::FrailtySpec,
    fit: UnifiedFitResult,
    resolvedspec: TermCollectionSpec,
    cov_design: TermCollectionDesign,
    learned_latent_sd: Option<f64>,
    // The baseline the fit's time offsets were realized from, carried on the
    // fit result. Re-parsing `FitConfig` here saved the user's seed instead of
    // the fitted baseline, and failed outright when scale/shape were unset
    // (#2714).
    baseline_cfg: crate::survival::construction::SurvivalBaselineConfig,
    is_survival: bool,
    time_basis: Option<SavedSurvivalTimeBasis>,
) -> Result<FittedModelPayload, String> {
    // Carried from the materialization that produced this fit, not re-derived
    // (#2470) — see `payload_for_survival_location_scale` for the anchor
    // divergence this closes.
    let time_basis = time_basis.ok_or_else(|| {
        "latent survival/binary payload requires the materialized survival time basis".to_string()
    })?;
    let parsed = parse_formula(&formula).map_err(|err| {
        format!("failed to re-parse latent survival formula for FFI payload: {err}")
    })?;
    // An interval-censored `SurvInterval(L, R, event)` fit materializes L as its
    // exit column with no entry column (`materialize/survival.rs`), and every saved
    // model reader already resolves that response with `parse_surv_interval_response`
    // (`FittedModel::prediction_required_columns`). The save path must accept it too.
    let (entryname, exitname, eventname) = match parse_surv_response(&parsed.response)? {
        Some(names) => names,
        None => parse_surv_interval_response(&parsed.response)?
            .map(|names| (None, names.0, names.2))
            .ok_or_else(|| {
                "latent survival/binary FFI requires a Surv(...) or SurvInterval(...) response"
                    .to_string()
            })?,
    };

    // For latent survival, splice the fitted latent_sd into the persisted
    // HazardMultiplier frailty (mirrors CLI behaviour at main.rs:5541).
    let saved_family = if is_survival {
        let frailty = match (&request_frailty, learned_latent_sd) {
            (
                crate::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
                    scale: crate::survival::lognormal_kernel::FrailtyScale::Learned { .. },
                    loading,
                },
                Some(sigma),
            ) => crate::survival::lognormal_kernel::FrailtySpec::HazardMultiplier {
                scale: crate::survival::lognormal_kernel::FrailtyScale::Fixed { sigma },
                loading: *loading,
            },
            _ => request_frailty.clone(),
        };
        FittedFamily::LatentSurvival { frailty }
    } else {
        FittedFamily::LatentBinary {
            frailty: request_frailty.clone(),
        }
    };
    let model_class_label = if is_survival {
        "latent-survival".to_string()
    } else {
        "latent-binary".to_string()
    };
    let likelihood_label = if is_survival {
        "latent".to_string()
    } else {
        "latent-binary".to_string()
    };

    let resolved_termspec = freeze_term_collection_from_design(&resolvedspec, &cov_design)
        .map_err(|err| err.to_string())?;

    Ok(assemble_latent_window_payload(
        LatentWindowInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result: fit,
            family: saved_family,
            model_class_label,
            likelihood_label,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            baseline_cfg,
            time_basis,
            resolved_termspec,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    ))
}

#[cfg(test)]
mod joint_latent_law_save_tests {
    use super::*;

    /// The two ways a `K ≥ 2` model could lose or double a score's map, refused
    /// by name, and the two states that save (gam#2929, gam#2949).
    #[test]
    fn joint_law_model_refuses_to_save_a_map_it_would_lose_or_double_2949() {
        use crate::bms::{LatentMeasureCalibration, LatentZConditionalCalibration};

        let conditional = LatentZConditionalCalibration {
            mean_coeffs: vec![0.1, 0.4],
            log_var_coeffs: Vec::new(),
            basis_ncols: 1,
            homoskedastic_var: 1.0,
            post_mean: 0.0,
            post_sd: 1.0,
            theta1_cov: ndarray::Array2::zeros((0, 0)),
        };
        let law = |maps: Vec<Option<LatentZConditionalCalibration>>| {
            crate::survival::marginal_slope::SurvivalJointLatentLaw {
                score_dim: 2,
                residual_nodes: vec![vec![-1.0, 0.0], vec![0.0, 1.0], vec![1.0, -1.0]],
                weights: vec![0.25, 0.5, 0.25],
                score_mean: vec![0.0, 0.0],
                pooled_factor: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                conditional: None,
                score_location: vec![0.0, 0.0],
                score_scale: vec![1.0, 1.0],
                score_calibrations: maps,
            }
        };
        let uncalibrated = vec![LatentMeasureCalibration::None; 2];
        let calibrated = vec![
            LatentMeasureCalibration::None,
            LatentMeasureCalibration::ConditionalLocationScale(conditional.clone()),
        ];

        // Nothing calibrated, nothing carried: the model saves.
        assert!(
            joint_latent_law_calibration_save_refusal(None, &uncalibrated, &law(Vec::new()))
                .is_none(),
            "an uncalibrated joint-law model must save"
        );
        // Calibrated and carried by the law: the model saves, and this is the
        // state gam#2949 adds.
        assert!(
            joint_latent_law_calibration_save_refusal(
                None,
                &calibrated,
                &law(vec![None, Some(conditional.clone())])
            )
            .is_none(),
            "a joint-law model whose law carries the calibrated column's map must save"
        );
        // Calibrated and NOT carried: prediction would read that column on an
        // uncalibrated axis.
        let lost = joint_latent_law_calibration_save_refusal(None, &calibrated, &law(Vec::new()))
            .expect("a map the law does not carry must refuse the save");
        assert!(
            lost.contains("column(s) [1]") && lost.contains("uncalibrated axis"),
            "unexpected refusal {lost}"
        );
        // Carried by the law AND by the payload: prediction would map twice.
        let doubled = joint_latent_law_calibration_save_refusal(
            Some(&conditional),
            &calibrated,
            &law(vec![None, Some(conditional.clone())]),
        )
        .expect("a second copy of a map must refuse the save");
        assert!(
            doubled.contains("apply it twice") && doubled.contains("refused"),
            "unexpected refusal {doubled}"
        );
    }
}

#[cfg(test)]
mod apply_timewiggle_beta_tests {
    use super::*;

    /// Minimal payload with both baseline-timewiggle slots unset. Uses the
    /// fixture-free `LatentBinary` family so the test needs no `LikelihoodSpec`.
    fn empty_payload() -> FittedModelPayload {
        FittedModelPayload::new(
            "y ~ 1".to_string(),
            ModelKind::Survival,
            FittedFamily::LatentBinary {
                frailty: crate::survival::lognormal_kernel::FrailtySpec::None,
            },
            "test".to_string(),
        )
    }

    #[test]
    fn by_cause_beta_populates_only_the_by_cause_slot() {
        let mut payload = empty_payload();
        apply_timewiggle_beta(
            &mut payload,
            SurvivalTimewiggleBeta::ByCause(vec![vec![1.0, 2.0], vec![3.0]]),
        );
        assert_eq!(
            payload.beta_baseline_timewiggle_by_cause,
            Some(vec![vec![1.0, 2.0], vec![3.0]]),
            "ByCause coefficients must land in the by-cause slot (regression: the \
             location-scale assembler used to silently drop them)"
        );
        assert!(
            payload.beta_baseline_timewiggle.is_none(),
            "ByCause must not populate the single-block slot"
        );
    }

    #[test]
    fn single_beta_populates_only_the_flat_slot() {
        let mut payload = empty_payload();
        apply_timewiggle_beta(&mut payload, SurvivalTimewiggleBeta::Single(vec![4.0, 5.0]));
        assert_eq!(payload.beta_baseline_timewiggle, Some(vec![4.0, 5.0]));
        assert!(payload.beta_baseline_timewiggle_by_cause.is_none());
    }
}

#[cfg(test)]
mod standard_payload_penalty_topology_tests {
    use super::*;
    use crate::fit_orchestration::fit_from_formula;
    use csv::StringRecord;
    use gam_data::encode_recordswith_inferred_schema;

    /// A deterministic, well-conditioned binomial fixture with five linearly
    /// independent covariates. Including the intercept makes the formula's raw
    /// mean width exactly six; the canonical eight-knot cubic LinkWiggle block
    /// has eleven raw columns, reproducing #2748's `6 -> 17` payload mismatch
    /// without a benchmark dataset or a spatial outer search.
    fn six_column_flexible_binomial_fixture() -> EncodedDataset {
        const N: usize = 256;
        let headers = ["y", "x0", "x1", "x2", "x3", "x4"]
            .into_iter()
            .map(String::from)
            .collect();
        let rows = (0..N)
            .map(|row| {
                let t = -2.75 + 5.5 * row as f64 / (N - 1) as f64;
                let x0 = t;
                let x1 = (1.3 * t).sin();
                let x2 = (0.7 * t).cos();
                let x3 = t * t - 2.5;
                let x4 = (2.1 * t + 0.2).sin();
                let eta = 0.15 + 0.65 * x0 - 0.45 * x1 + 0.30 * x2 - 0.08 * x3 + 0.22 * x4;
                // A monotone non-logit response map gives the learned warp a
                // genuine, numerically mild signal. The irrational rotation is
                // a deterministic low-discrepancy Bernoulli draw, avoiding both
                // RNG state and accidental separation.
                let warped_eta = eta + 0.35 * eta.tanh();
                let probability = 1.0 / (1.0 + (-warped_eta).exp());
                let uniform = ((row + 1) as f64 * 0.618_033_988_749_894_9).fract();
                let y = usize::from(uniform < probability);
                StringRecord::from(vec![
                    y.to_string(),
                    x0.to_string(),
                    x1.to_string(),
                    x2.to_string(),
                    x3.to_string(),
                    x4.to_string(),
                ])
            })
            .collect();
        encode_recordswith_inferred_schema(headers, rows).expect("encode #2748 fixture")
    }

    fn check_flexible_binomial_payload(link: StandardLink, link_name: &str) {
        let dataset = six_column_flexible_binomial_fixture();
        let formula = format!("y ~ x0 + x1 + x2 + x3 + x4 + link(type=flexible({link_name}))");
        let config = FitConfig {
            family: Some("binomial".to_string()),
            ..FitConfig::default()
        };
        let FitResult::Standard(result) = fit_from_formula(&formula, &dataset, &config)
            .expect("the small identifiable flexible-link fixture must fit")
        else {
            panic!("flexible-link formula did not produce a standard fit");
        };
        let mean_dim = result.design.design.ncols();
        let raw_dim = result
            .fit
            .geometry
            .as_ref()
            .expect("joint flexible fit must retain coefficient geometry")
            .coefficient_gauge
            .raw_total();
        assert_eq!(mean_dim, 6, "fixture must reproduce the base width");
        assert_eq!(raw_dim, 17, "fixture must reproduce #2748's 6 -> 17 join");
        let expected_family =
            LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Standard(link));
        assert_eq!(
            result.fit.likelihood_family.as_ref(),
            Some(&expected_family),
            "the custom link-wiggle solve must retain the response likelihood",
        );

        let payload = assemble_standard_payload(StandardPayloadInputs {
            formula: formula.to_string(),
            dataset: &dataset,
            fit_config: &config,
            result,
        })
        .expect("payload assembly must use the full realized raw penalty topology");
        let encoded = serde_json::to_vec(&payload).expect("serialize flexible binomial payload");
        let payload: FittedModelPayload =
            serde_json::from_slice(&encoded).expect("reload flexible binomial payload");
        assert_eq!(payload.family, expected_family.name());
        match &payload.family_state {
            FittedFamily::Standard { likelihood, link: saved_link, .. } => {
                assert_eq!(likelihood, &expected_family);
                assert_eq!(*saved_link, Some(link));
            }
            other => panic!("flexible binomial payload has wrong family state: {other:?}"),
        }
        let fit = payload
            .fit_result
            .expect("standard payload must retain its canonical fit result");
        assert_eq!(fit.likelihood_family.as_ref(), Some(&expected_family));
        // x0..x4 each carry the default null-recovery ridge (b7b874a2a), so beside the
        // full-rank LinkWiggle penalty only the intercept stays unpenalized.
        assert_eq!(
            fit.artifacts.null_space_dim,
            Some(1),
            "the five linear ridges and the full-rank LinkWiggle penalty leave only the intercept unpenalized",
        );
        assert!(
            fit.artifacts
                .null_space_logdet
                .is_some_and(f64::is_finite),
            "the null-space Hessian log-determinant must be finite",
        );
    }

    #[test]
    fn flexible_fit_payload_uses_the_full_six_plus_eleven_penalty_topology_2748() {
        check_flexible_binomial_payload(StandardLink::Logit, "logit");
    }

    #[test]
    fn flexible_probit_payload_preserves_its_binomial_response_scale_2748() {
        check_flexible_binomial_payload(StandardLink::Probit, "probit");
    }

    /// A deterministic Gaussian fixture: `n = 300`, `y = sin(1.5 x)` plus an
    /// irrational-rotation perturbation, so no RNG state is involved.
    fn gaussian_sine_fixture() -> EncodedDataset {
        const N: usize = 300;
        let headers = ["y", "x"].into_iter().map(String::from).collect();
        let rows = (0..N)
            .map(|row| {
                let x = -3.0 + 6.0 * row as f64 / (N - 1) as f64;
                let perturbation = ((row + 1) as f64 * 0.618_033_988_749_894_9).fract() - 0.5;
                let y = (1.5 * x).sin() + perturbation;
                StringRecord::from(vec![y.to_string(), x.to_string()])
            })
            .collect();
        encode_recordswith_inferred_schema(headers, rows).expect("encode gaussian sine fixture")
    }

    /// `y ~ 1` has no penalty block, so its one coefficient is unpenalized and the
    /// null-space normalizer must run over it exactly as it runs over the
    /// intercept direction of `y ~ x`, whose slope carries the default ridge. Both
    /// report `q = 1` and, with unit weights, `log|H_null| = log n`. The empty-
    /// penalty branch used to report `q = 0` and switch the normalizer off.
    #[test]
    fn empty_penalty_fit_normalizes_over_its_unpenalized_intercept_2627() {
        let dataset = gaussian_sine_fixture();
        let config = FitConfig::default();
        let n = dataset.values.nrows() as f64;
        for formula in ["y ~ 1", "y ~ x"] {
            let FitResult::Standard(result) = fit_from_formula(formula, &dataset, &config)
                .unwrap_or_else(|error| panic!("{formula} must fit: {error}"))
            else {
                panic!("{formula} did not produce a standard fit");
            };
            let raw_dim = result.design.design.ncols();
            let payload = assemble_standard_payload(StandardPayloadInputs {
                formula: formula.to_string(),
                dataset: &dataset,
                fit_config: &config,
                result,
            })
            .unwrap_or_else(|error| panic!("{formula} payload must assemble: {error}"));
            let fit = payload
                .fit_result
                .expect("standard payload must retain its canonical fit result");
            assert_eq!(
                fit.artifacts.null_space_dim,
                Some(1),
                "{formula}: only the intercept is unpenalized"
            );
            let logdet = fit
                .artifacts
                .null_space_logdet
                .expect("null-space Hessian log-determinant must be published");
            // The restriction is accepted up to the pullback's own backward-error
            // limit, sqrt(eps) per raw column, so that is the resolution of the value.
            let resolution = f64::EPSILON.sqrt() * raw_dim as f64 * n.ln();
            assert!(
                (logdet - n.ln()).abs() <= resolution,
                "{formula}: null-space log-determinant {logdet} is not log n = {} \
                 (resolution {resolution:e})",
                n.ln()
            );
        }
    }
}

#[cfg(test)]
mod latent_saved_baseline_tests {
    use super::*;
    use crate::inference::model::FittedModel;
    use crate::survival::lognormal_kernel::{
        FrailtyScale, FrailtySpec, HazardLoading, LatentSurvivalRow, LatentSurvivalRowJet,
    };
    use crate::survival::predict::{
        SurvivalPredictEstimand, SurvivalPredictRequest, predict_latent_window_survival,
    };
    use csv::StringRecord;
    use ndarray::Array1;

    /// Deterministic Weibull survival rows with a covariate effect and
    /// administrative censoring, encoded through the ordinary schema inference.
    fn weibull_survival_dataset(n: usize) -> EncodedDataset {
        let headers = vec!["time".to_string(), "status".to_string(), "x".to_string()];
        let records = (0..n)
            .map(|i| {
                let x = ((i * 37) % n) as f64 / n as f64;
                let u = (((i * 53 + 11) % 97) as f64 + 0.5) / 97.0;
                let event_time = 60.0 * (-u.ln() / (0.8 * x).exp()).powf(1.0 / 1.4);
                let censor_time = 20.0 + 100.0 * ((((i * 29 + 7) % 83) as f64 + 0.5) / 83.0);
                let (time, status) = if event_time <= censor_time {
                    (event_time, 1.0)
                } else {
                    (censor_time, 0.0)
                };
                StringRecord::from(vec![time.to_string(), status.to_string(), x.to_string()])
            })
            .collect();
        gam_data::encode_recordswith_inferred_schema(headers, records)
            .expect("encode synthetic latent survival rows")
    }

    fn saved_window_log_survival(model: &FittedModel, data: &EncodedDataset) -> Array1<f64> {
        let col_map = data.column_map();
        let zeros = Array1::<f64>::zeros(data.values.nrows());
        predict_latent_window_survival(SurvivalPredictRequest {
            model,
            data: data.values.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        })
        .expect("saved latent survival model must predict at the training rows")
        .window_survival
        .mapv(f64::ln)
    }

    fn max_abs_gap(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .fold(0.0_f64, |acc, (l, r)| acc.max((l - r).abs()))
    }

    /// #2714: a saved latent survival model predicts with the baseline its time
    /// coefficients were fitted against. The payload used to re-parse the user's
    /// `FitConfig`, which persisted the seed baseline instead of the fitted one,
    /// and failed outright when scale and shape were unset.
    ///
    /// The in-memory side is evaluated from the fit's own coefficients and the
    /// request's realized time designs and offsets; nothing there is rebuilt from
    /// saved fields. The positive control persists a different shape and must
    /// move the predictions far past the agreement bar, so a payload that saved
    /// the wrong baseline cannot pass.
    #[test]
    fn saved_latent_survival_model_predicts_at_its_fitted_baseline_2714() {
        let n = 80;
        let data = weibull_survival_dataset(n);
        let formula = "Surv(time, status) ~ x";
        let sigma = 0.5;
        let config = FitConfig {
            survival_likelihood: Some("latent".to_string()),
            baseline_target: "weibull".to_string(),
            time_basis: "ispline".to_string(),
            frailty: FrailtySpec::HazardMultiplier {
                scale: FrailtyScale::Fixed { sigma },
                loading: HazardLoading::Full,
            },
            ..FitConfig::default()
        };
        assert!(
            config.baseline_scale.is_none() && config.baseline_shape.is_none(),
            "precondition: the pin covers the unset scale/shape configuration"
        );

        let materialized =
            materialize(formula, &data, &config).expect("latent survival formula must materialize");
        let survival_time_basis = materialized.survival_time_basis.clone();
        let FitRequest::LatentSurvival(request) = materialized.request else {
            panic!("survival_likelihood=latent must materialize a latent survival request");
        };
        let time_design_entry = request.spec.time_block.design_entry.clone();
        let time_design_exit = request.spec.time_block.design_exit.clone();
        // The fit selects its baseline together with ρ (#2714), so the request's
        // prepared offsets belong to the seed baseline. The in-memory side realizes
        // them at the fitted chart point through the chart the fit itself uses;
        // nothing there is rebuilt from saved fields.
        let chart = crate::survival::construction::LatentSurvivalFrozenOffsetChart::new(
            &request.spec.age_entry,
            &request.spec.age_exit,
            None,
            &request.spec.baseline_config,
            HazardLoading::Full,
            &request.spec.time_block.offset_entry,
            &request.spec.time_block.offset_exit,
            &request.spec.time_block.derivative_offset_exit,
            &Array1::zeros(n),
        )
        .expect("latent survival baseline chart")
        .expect("a Weibull baseline has chart coordinates");
        let unloaded_entry = request.spec.unloaded_mass_entry.clone();
        let unloaded_exit = request.spec.unloaded_mass_exit.clone();
        let mean_offset = request.spec.mean_offset.clone();
        let frailty = request.frailty.clone();

        let result = match fit_model(FitRequest::LatentSurvival(request)) {
            Ok(FitResult::LatentSurvival(result)) => result,
            Ok(_) => panic!("latent survival request returned another result variant"),
            Err(error) => panic!("latent survival fit failed: {error}"),
        };
        let fitted_baseline = result.baseline_config.clone();
        assert!(
            fitted_baseline.shape.is_some_and(|shape| shape != 1.0),
            "precondition: the fitted baseline shape must differ from the unset-shape seed 1.0, \
             otherwise persisting the seed and persisting the fit are indistinguishable here \
             (got {:?})",
            fitted_baseline.shape
        );
        let fitted_theta =
            crate::survival::construction::survival_baseline_theta_from_config(&fitted_baseline)
                .expect("fitted baseline chart coordinates")
                .expect("a Weibull baseline has chart coordinates");
        let fitted_offsets = chart
            .evaluate(&fitted_theta)
            .expect("the chart realizes the fitted baseline");

        let mean_beta = result
            .fit
            .block_by_role(gam_problem::BlockRole::Mean)
            .expect("latent survival fit carries a mean block")
            .beta
            .clone();
        let time_beta = result
            .fit
            .block_by_role(gam_problem::BlockRole::Time)
            .expect("latent survival fit carries a time block")
            .beta
            .clone();
        let eta = result.design.design.dot(&mean_beta) + &mean_offset;
        let q_entry = time_design_entry.dot(&time_beta) + &fitted_offsets.offset_entry;
        let q_exit = time_design_exit.dot(&time_beta) + &fitted_offsets.offset_exit;
        let quadrature = gam_solve::quadrature::QuadratureContext::new();
        let in_memory_log_survival = Array1::from_shape_fn(n, |row| {
            let latent_row = LatentSurvivalRow::right_censored(
                q_entry[row].exp(),
                q_exit[row].exp(),
                unloaded_entry[row],
                unloaded_exit[row],
            );
            LatentSurvivalRowJet::evaluate(&quadrature, &latent_row, eta[row], sigma)
                .expect("in-memory latent survival row evaluation")
                .log_lik
        });

        let payload = payload_for_latent_survival(
            formula.to_string(),
            &data,
            &config,
            frailty,
            result,
            survival_time_basis,
        )
        .expect("a latent survival fit with baseline scale/shape unset must build its payload");
        assert_eq!(payload.survival_baseline_scale, fitted_baseline.scale);
        assert_eq!(payload.survival_baseline_shape, fitted_baseline.shape);

        let saved_log_survival =
            saved_window_log_survival(&FittedModel::from_payload(payload.clone()), &data);
        let agreement_gap = max_abs_gap(&saved_log_survival, &in_memory_log_survival);

        // The same saved law over a time grid: a grid time equal to a row's own exit
        // closes that row's window exactly where its own window closes, and no row's
        // fitted survival rises along the sorted grid.
        let saved_model = FittedModel::from_payload(payload.clone());
        let columns = data.column_map();
        let rows = data.values.nrows();
        let zeros = Array1::<f64>::zeros(rows);
        let mut grid: Vec<f64> = (0..4).map(|row| data.values[[row, columns["time"]]]).collect();
        grid.sort_by(f64::total_cmp);
        let grid_prediction = predict_latent_window_survival(SurvivalPredictRequest {
            model: &saved_model,
            data: data.values.view(),
            col_map: &columns,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: Some(&grid),
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        })
        .expect("a saved latent survival model must predict over a time grid");
        let grid_survival = grid_prediction
            .grid_survival
            .expect("a time-grid request must return the grid survival surface");
        assert_eq!(grid_survival.dim(), (rows, grid.len()));
        let mut matched_windows = 0usize;
        for (column, &time) in grid.iter().enumerate() {
            for row in 0..rows {
                if data.values[[row, columns["time"]]].to_bits() == time.to_bits() {
                    matched_windows += 1;
                    let own = grid_prediction.window_survival[row];
                    assert!(
                        (grid_survival[[row, column]] - own).abs() <= 1e-12,
                        "row {row}: the grid survival at its own exit {time} is {} but its window survival is {own}",
                        grid_survival[[row, column]]
                    );
                }
            }
        }
        assert!(
            matched_windows >= grid.len(),
            "precondition: every grid time is some row's own exit, matched {matched_windows} windows for {} times",
            grid.len()
        );
        for row in 0..rows {
            for column in 1..grid.len() {
                assert!(
                    grid_survival[[row, column]] <= grid_survival[[row, column - 1]],
                    "row {row}: fitted survival rises from {} at t={} to {} at t={}",
                    grid_survival[[row, column - 1]],
                    grid[column - 1],
                    grid_survival[[row, column]],
                    grid[column]
                );
            }
        }

        let mut seed_payload = payload;
        seed_payload.survival_baseline_shape = Some(1.0);
        let seed_log_survival =
            saved_window_log_survival(&FittedModel::from_payload(seed_payload), &data);
        let seed_gap = max_abs_gap(&seed_log_survival, &in_memory_log_survival);

        eprintln!(
            "[2714] saved latent baseline pin: fitted shape={:?} scale={:?} \
             agreement_gap={agreement_gap:.3e} seed_gap={seed_gap:.3e}",
            fitted_baseline.shape, fitted_baseline.scale
        );
        let bar = 1e-10;
        assert!(
            agreement_gap <= bar,
            "saved latent survival predictions disagree with the fit at its baseline: \
             max |log S_saved - log S_fit| = {agreement_gap:.3e} > {bar:.1e}"
        );
        assert!(
            seed_gap > 1e3 * bar,
            "positive control: persisting the seed shape must move the predictions past the \
             agreement bar, got max gap {seed_gap:.3e}"
        );
    }
}

#[cfg(test)]
mod survival_payload_decline_tests {
    use super::*;
    use crate::survival::lognormal_kernel::FrailtySpec;
    use gam_problem::BlockRole;
    use gam_problem::LinearInequalityConstraints;
    use gam_problem::types::{LikelihoodScaleMetadata, LogLikelihoodNormalization};
    use gam_solve::constrained_posterior::{
        ConePosteriorMomentDecline, ConePropernessEvidence, ConstrainedPosteriorGeometry,
    };
    use gam_solve::estimate::{FitArtifacts, FitGeometry, FittedBlock};
    use gam_solve::pirls::PirlsStatus;
    use ndarray::{Array1, Array2, array};

    /// A two-coefficient Royston-Parmar fit. Declined, it stores the mode `(0, 0.5)`
    /// on the bound `β₀ ≥ 0` under a typed moment decline and has no covariance;
    /// otherwise it carries a covariance and no constrained geometry.
    fn survival_fit(declined: bool) -> UnifiedFitResult {
        let geometry = declined.then(|| FitGeometry {
            coefficient_gauge: gam_problem::gauge::Gauge::identity(&[2]),
            penalized_hessian: array![[1.0, 0.0], [0.0, -2.0]].into(),
            constrained_posterior: Some(ConstrainedPosteriorGeometry::with_decline(
                LinearInequalityConstraints::new(array![[1.0, 0.0]], array![0.0])
                    .expect("a 1x2 inequality system with a matching bound is well formed"),
                array![0.0, 0.5],
                ConePosteriorMomentDecline {
                    ambient_precision_failure: "fixture: the ambient precision is indefinite"
                        .to_string(),
                    properness: ConePropernessEvidence::CertificationFailed {
                        reason: "fixture: properness was not certified".to_string(),
                    },
                    active_rows: vec![0],
                    boundary_approximation_refusal: None,
                },
            )),
            working: None,
        });
        UnifiedFitResult::try_from_parts(gam_solve::estimate::UnifiedFitResultParts {
            blocks: vec![FittedBlock {
                beta: array![0.0, 0.5],
                role: BlockRole::Threshold,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            }],
            training_sample_size: 16,
            training_response_fingerprint: None,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            likelihood_family: Some(LikelihoodSpec::royston_parmar()),
            likelihood_scale: LikelihoodScaleMetadata::Unspecified,
            log_likelihood_normalization: LogLikelihoodNormalization::Full,
            log_likelihood: 0.0,
            deviance: 0.0,
            reml_score: Some(0.0),
            stable_penalty_term: 0.0,
            penalized_objective: Some(0.0),
            used_device: false,
            outer_iterations: 0,
            outer_converged: true,
            outer_gradient_norm: None,
            standard_deviation: 1.0,
            covariance_conditional: (!declined).then(|| Array2::eye(2)),
            covariance_corrected: None,
            inference: None,
            fitted_link: FittedLinkState::Standard(None),
            geometry,
            block_states: Vec::new(),
            pirls_status: PirlsStatus::Converged,
            max_abs_eta: 0.0,
            constraint_kkt: None,
            artifacts: FitArtifacts {
                pirls: None,
                ..Default::default()
            },
            inner_cycles: 0,
        })
        .expect("the survival fixture fit must assemble")
    }

    /// #979 ruling (c), gnomon#2336: a fit that keeps its optimizer mode under a
    /// moment decline is saved with that mode, and the saved fit keeps the typed
    /// decline, so every consumer of posterior moments still refuses it, naming the
    /// operation, the missing estimand and the decline's own reason. The control is
    /// the same fit with reportable moments, which must still assemble.
    #[test]
    fn a_declined_survival_fit_is_saved_with_its_mode_and_its_decline_979() {
        let schema = DataSchema { columns: Vec::new() };
        let payload = new_royston_parmar_survival_payload(
            "Surv(time, status) ~ x".to_string(),
            survival_fit(true),
            schema.clone(),
            "royston-parmar",
            None,
            FrailtySpec::None,
        )
        .expect("a declined fit saves its converged mode");
        let saved = payload
            .fit_result
            .as_ref()
            .expect("the declined payload must carry its fit");
        assert!(
            saved.posterior_moment_decline().is_some(),
            "the saved fit must keep its typed moment decline"
        );
        let refusal = saved
            .require_posterior_mean("saved-model covariance summary")
            .expect_err("a saved declined fit has no posterior mean")
            .to_string();
        for needle in [
            "saved-model covariance summary",
            "posterior-mean",
            "the ambient precision is indefinite",
        ] {
            assert!(
                refusal.contains(needle),
                "the refusal must name {needle:?}, got: {refusal}"
            );
        }
        let payload = new_royston_parmar_survival_payload(
            "Surv(time, status) ~ x".to_string(),
            survival_fit(false),
            schema,
            "royston-parmar",
            None,
            FrailtySpec::None,
        )
        .expect("a fit with reportable posterior moments must assemble");
        assert!(payload.fit_result.is_some(), "the control payload must carry its fit");
    }
}
