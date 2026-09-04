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
    BernoulliMarginalSlopeFitResult, DeviationRuntime, LatentMeasureKind, LatentZConditionalCalibration, LatentZRankIntCalibration,
};
use crate::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL;
use crate::fit_orchestration::drivers::freeze_term_collection_from_design;
use crate::fit_orchestration::{
    DispersionLocationScaleFitResult, FitConfig, FitRequest, FitResult, StandardFitResult,
    WorkflowError, expectile_tau_for_config, fit_expectile_if_requested,
    fit_materialized_standard_with_notes, fit_model, materialize,
};
use crate::gamlss::{
    BinomialLocationScaleFitResult, DispersionFamilyKind, GaussianLocationScaleFitResult,
};
use crate::inference::model::{
    FittedEstimator, FittedFamily, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
    SavedAnchorComponent, SavedAnchorKind, SavedCompiledFlexBlock, SavedLatentZNormalization,
    SavedResidualCascade, SavedSplineScan, SavedSurvivalLocationScaleStructure,
    SavedTransformationNormalGeometry, TransformationNormalParameterization,
    TransformationScoreCalibration,
};
use crate::scale_design::{ScaleDeviationTransform, build_scale_deviation_transform};
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
use faer::Side;
use gam_data::{DataSchema, EncodedDataset};
use gam_linalg::faer_ndarray::{FaerCholesky, array2_to_nested_vec};
use gam_problem::BlockRole;
use gam_problem::types::{
    InverseLink, LikelihoodSpec, ResponseFamily, StandardLink, inverse_link_to_binomial_spec,
};
use gam_solve::estimate::{
    FittedLinkState, UnifiedFitResult, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit,
};
use gam_terms::inference::formula_dsl::{parse_formula, parse_surv_response};
use gam_terms::smooth::{BlockwisePenalty, TermCollectionDesign, TermCollectionSpec};
use ndarray::{Array1, Array2, s};
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

fn standard_null_space_metadata(
    topology: &RealizedRawPenaltyTopology,
    fit: &UnifiedFitResult,
) -> Result<(usize, f64), String> {
    let hessian = fit
        .penalized_hessian()
        .ok_or_else(|| "null-space Hessian logdet requires fitted penalized Hessian".to_string())?;
    let hessian_dim = hessian.nrows();
    if hessian.ncols() != hessian_dim {
        return Err(format!(
            "null-space Hessian logdet requires a square Hessian, got {}x{}",
            hessian.nrows(),
            hessian.ncols()
        ));
    }
    let p = topology.coefficient_dim;
    if topology.penalties.is_empty() {
        return Ok((0, 0.0));
    }
    let mut penalty = Array2::<f64>::zeros((p, p));
    for (idx, block) in topology.penalties.iter().enumerate() {
        let range = block.col_range.clone();
        if range.start > range.end
            || range.end > p
            || block.local.nrows() != range.len()
            || block.local.ncols() != range.len()
        {
            return Err(format!(
                "null-space Hessian logdet penalty {idx} shape mismatch: range {}..{}, local {}x{}, p={p}",
                range.start,
                range.end,
                block.local.nrows(),
                block.local.ncols()
            ));
        }
        penalty
            .slice_mut(s![range.clone(), range])
            .scaled_add(1.0, &block.local);
    }
    let (null_basis, _) = gam_linalg::faer_ndarray::rrqr_nullspace_basis(
        &penalty,
        gam_linalg::faer_ndarray::default_rrqr_rank_alpha(),
    )
    .map_err(|err| format!("failed to compute penalty null-space basis: {err}"))?;
    let q = null_basis.ncols();
    if q == 0 {
        return Ok((0, 0.0));
    }

    // The saved Hessian lives in the active coordinates declared by the fit's
    // gauge, while `null_basis` is expressed in the design's raw coordinates.
    // Pull every raw null-space direction N back through the injective lift
    // `T`: solve `T C = N`, then restrict as `C' H_active C`. Treating a
    // rectangular active Hessian as if it were raw curvature was the hidden
    // identity-gauge assumption exposed by exact smoothing boundaries (#2623).
    let active_null_basis = if let Some(geometry) = fit.geometry.as_ref() {
        let gauge = &geometry.coefficient_gauge;
        if gauge.raw_total() != p || gauge.reduced_total() != hessian_dim {
            return Err(format!(
                "null-space Hessian logdet gauge mismatch: realized penalty topology has {p} raw columns, gauge \
                 maps {} raw from {} active coordinates, Hessian is {hessian_dim}x{hessian_dim}",
                gauge.raw_total(),
                gauge.reduced_total(),
            ));
        }
        let t = &gauge.t_full;
        let raw_gram = t.t().dot(t);
        let gram = (&raw_gram + &raw_gram.t().to_owned()) * 0.5;
        let chol = gram.cholesky(Side::Lower).map_err(|error| {
            format!(
                "null-space Hessian logdet coefficient gauge is not injective: {error}"
            )
        })?;
        let coordinates = chol.solve_mat(&t.t().dot(&null_basis));
        let residual = t.dot(&coordinates) - &null_basis;
        let residual_max = residual
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let basis_max = null_basis
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let backward_error = residual_max / basis_max;
        let roundoff_limit = f64::EPSILON.sqrt() * p.max(hessian_dim).max(1) as f64;
        if !backward_error.is_finite() || backward_error > roundoff_limit {
            return Err(format!(
                "null-space Hessian logdet raw penalty null space is not contained in the \
                 fitted active gauge: relative residual {backward_error:.6e}, numerical limit \
                 {roundoff_limit:.6e}"
            ));
        }
        coordinates
    } else {
        if hessian_dim != p {
            return Err(format!(
                "null-space Hessian logdet design/Hessian mismatch without a coefficient \
                 gauge: design has {p} columns but Hessian is {hessian_dim}x{hessian_dim}"
            ));
        }
        null_basis
    };
    let projected = hessian.dot(&active_null_basis);
    let mut restricted = active_null_basis.t().dot(&projected);
    restricted = (&restricted + &restricted.t()) * 0.5;
    let chol = restricted
        .cholesky(Side::Lower)
        .map_err(|err| format!("null-space Hessian is not positive definite: {err}"))?;
    let logdet = 2.0 * chol.diag().iter().map(|value| value.ln()).sum::<f64>();
    if logdet.is_finite() {
        Ok((q, logdet))
    } else {
        Err(format!("null-space Hessian logdet is not finite: {logdet}"))
    }
}

fn response_for_standard_payload(formula: &str, dataset: &EncodedDataset) -> Option<Array1<f64>> {
    let response = gam_terms::inference::formula_dsl::parse_formula(formula)
        .ok()?
        .response;
    let column = *dataset.column_map().get(&response)?;
    Some(dataset.values.column(column).to_owned())
}

fn standard_conformal_substrates(
    formula: &str,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    family: &LikelihoodSpec,
    fit: &UnifiedFitResult,
    design: &TermCollectionDesign,
) -> (
    Option<crate::inference::full_conformal::GaussianJackknifePlusStats>,
    Option<crate::inference::full_conformal::ExactFullConformalSubstrate>,
) {
    // #2633: these two substrates are ~94% of a saved Gaussian model at
    // n=20,000 and grow with the training rows, to save ~5.6 ms of rebuild. A
    // caller that keeps its training data, or never asks for a conformal
    // interval, can decline them; see `FitConfig::precompute_conformal` for the
    // measured trade-off and why the default is to keep them.
    if fit_config.precompute_conformal == Some(false) {
        return (None, None);
    }
    let expectile = fit_config.family.as_deref().is_some_and(|family| {
        let family = family.trim().to_ascii_lowercase();
        family == "expectile" || family.starts_with("expectile(")
    });
    if expectile
        || !family.is_gaussian_identity()
        || fit_config.weight_column.is_some()
        || fit_config.offset_column.is_some()
        || fit_config.flexible_link
        || design.affine_offset.iter().any(|value| *value != 0.0)
    {
        return (None, None);
    }
    let Some(y) = response_for_standard_payload(formula, dataset) else {
        return (None, None);
    };
    let Ok(x) = design.design.try_to_dense_arc("standard conformal design") else {
        return (None, None);
    };
    let Some(normal_matrix) = fit.penalized_hessian() else {
        return (None, None);
    };
    if x.nrows() != y.len()
        || normal_matrix.nrows() != x.ncols()
        || normal_matrix.ncols() != x.ncols()
    {
        return (None, None);
    }
    let weights = Array1::<f64>::ones(y.len());
    // Either substrate may legitimately decline this design (rank, shape, or a
    // non-invertible normal matrix). `None` is the contract, but the reason is
    // what explains a fit that silently ships without conformal intervals.
    let jackknife = match crate::inference::full_conformal::GaussianJackknifePlusStats::from_design_unit_weight_normal_matrix(
        x.as_ref(),
        &y,
        &weights,
        normal_matrix,
    ) {
        Ok(stats) => Some(stats),
        Err(reason) => {
            log::debug!("jackknife+ conformal substrate unavailable: {reason}");
            None
        }
    };
    let full = match crate::inference::full_conformal::ExactFullConformalSubstrate::from_design_unit_weight_normal_matrix(
        x.as_ref(),
        &y,
        &weights,
        normal_matrix,
    ) {
        Ok(substrate) => Some(substrate),
        Err(reason) => {
            log::debug!("exact full-conformal substrate unavailable: {reason}");
            None
        }
    };
    (jackknife, full)
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
        wiggle_saved_warp_beta,
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
        standard_null_space_metadata(&raw_penalty_topology, &fit)?;
    fit.artifacts.null_space_dim = Some(null_space_dim);
    fit.artifacts.null_space_logdet = Some(null_space_logdet);
    let family = fit
        .likelihood_family
        .clone()
        .unwrap_or_else(LikelihoodSpec::gaussian_identity);
    let estimator = expectile_tau_for_config(fit_config)
        .map_err(|error| format!("failed to persist estimator metadata: {error}"))?
        .map_or(FittedEstimator::Likelihood, |tau| {
            FittedEstimator::Expectile { tau }
        });
    let family_label = match estimator {
        FittedEstimator::Likelihood => family.name().to_string(),
        FittedEstimator::Expectile { tau } => format!("expectile({tau})"),
    };
    let (gaussian_jackknife_plus, full_conformal) =
        standard_conformal_substrates(&formula, dataset, fit_config, &family, &fit, &design);
    let latent_cloglog_state = if family.is_latent_cloglog() {
        Some(saved_latent_cloglog_state_from_fit(&fit).ok_or_else(|| {
            "latent-cloglog-binomial fit did not produce a fitted latent-cloglog state".to_string()
        })?)
    } else {
        saved_latent_cloglog_state_from_fit(&fit)
    };
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
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
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit.clone());
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = fitted_inverse_link(&fit.fitted_link).or_else(|| Some(family.link.clone()));
    payload.linkwiggle_knots = wiggle_knots.map(|knots| knots.to_vec());
    payload.linkwiggle_degree = wiggle_degree;
    payload.linkwiggle_penalty_metadata = wiggle_penalty_metadata;
    payload.beta_link_wiggle = wiggle_saved_warp_beta;
    payload.link_wiggle_index_shift = wiggle_saved_index_shift;
    match &fit.fitted_link {
        FittedLinkState::Mixture { covariance, .. } => {
            payload.mixture_link_param_covariance = covariance.as_ref().map(array2_to_nested_vec);
        }
        FittedLinkState::Sas { covariance, .. }
        | FittedLinkState::BetaLogistic { covariance, .. } => {
            payload.sas_param_covariance = covariance.as_ref().map(array2_to_nested_vec);
        }
        FittedLinkState::Standard(_) | FittedLinkState::LatentCLogLog { .. } => {}
    }
    payload.set_training_feature_metadata(dataset.headers.clone(), dataset.feature_ranges());
    payload.resolved_termspec = Some(resolved_termspec);
    payload.basis_adequacy = basis_adequacy;
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    payload.weight_column = fit_config.weight_column.clone();
    payload.gaussian_jackknife_plus = gaussian_jackknife_plus;
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
    pub latent_z_rank_int_calibration: Option<LatentZRankIntCalibration>,
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
    pub score_warp_runtime: Option<&'a DeviationRuntime>,
    pub link_dev_runtime: Option<&'a DeviationRuntime>,
    pub base_link: InverseLink,
    pub frailty: crate::survival::lognormal_kernel::FrailtySpec,
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
        outer_cost_evals: _,
        inner_pirls_solves: _,
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
        MODEL_PAYLOAD_VERSION,
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
        MODEL_PAYLOAD_VERSION,
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
/// (`slope_formula(s)`, `z_column(s)`, `baseline_slope(s)`,
/// `resolved_slopespec(s)`) are kept consistent — so the CLI and FFI
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
        latent_z_rank_int_calibration,
        latent_z_conditional_calibration,
        score_warp_runtime,
        link_dev_runtime,
        base_link,
        frailty,
    } = inputs;

    // #461 predict seam: drop the training-only influence-absorber γ (and
    // marginalize it out of the covariance) so the persisted model matches the
    // raw p_m marginal design at predict. No-op when the absorber is inactive.
    let fit_result = truncate_marginal_slope_influence_absorber(fit_result, p_marginal)?;

    let marginal_likelihood_spec =
        inverse_link_to_binomial_spec(&base_link).map_err(|e| e.to_string())?;

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        formula,
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood: marginal_likelihood_spec,
            base_link: base_link.clone(),
            frailty,
        },
        FAMILY_BERNOULLI_MARGINAL_SLOPE.to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload.slope_formula = Some(slope_formula.clone());
    payload.z_column = Some(z_column.clone());
    payload.slope_formulas = Some(vec![slope_formula]);
    payload.z_columns = Some(vec![z_column]);
    payload.latent_z_normalization = Some(latent_z_normalization);
    payload.latent_measure = Some(latent_measure);
    payload.latent_z_rank_int_calibration = latent_z_rank_int_calibration;
    payload.latent_z_conditional_calibration = latent_z_conditional_calibration;
    payload.marginal_baseline = Some(baseline_marginal);
    payload.baseline_slope = Some(baseline_slope);
    payload.baseline_slopes = Some(vec![baseline_slope]);
    payload.link = Some(base_link);
    payload.resolved_termspec = Some(resolved_marginalspec);
    payload.resolved_slopespecs = Some(vec![resolved_slopespec.clone()]);
    payload.resolved_slopespec = Some(resolved_slopespec);
    payload.score_warp_runtime = score_warp_runtime.map(serialize_anchored_deviation_runtime);
    payload.link_deviation_runtime = link_dev_runtime.map(serialize_anchored_deviation_runtime);
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
    payload.transformation_geometry = Some(transformation_normal_geometry(family));
    // Persist the monotonicity-cone carrier Ψ (the fitted covariate design at
    // κ̂), row-major n × p_cov, so constrained posterior sampling can certify
    // draws against the positivity cone without replaying the (non-bitwise)
    // spatial warp. The covariate design is materialized during fitting, so this
    // is a cache hit; a post-fit materialization failure is an internal invariant
    // break, not a recoverable condition.
    let cone_carrier = family
        .covariate_dense_arc()
        .expect("CTN covariate design must materialize for the persisted cone carrier");
    payload.transformation_cone_carrier = Some(cone_carrier.iter().copied().collect());
    payload.transformation_score_calibration = Some(score_calibration);
    source.apply_to(&mut payload);
    payload
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
/// (residual response scale) or binomial (noise scale-deviation transform whose
/// likelihood is resolved from the inverse link). The assembler resolves the
/// `FittedFamily` from this once, rather than each save path stamping a
/// (potentially wrong) likelihood and patching it afterwards.
pub enum LocationScaleResponse<'a> {
    /// Gaussian identity; `base_link` is the optional resolved base link the CLI
    /// may pass through from `link(...)` (the FFI leaves it `None`).
    Gaussian {
        response_scale: f64,
        base_link: Option<InverseLink>,
    },
    /// Binomial under `link`, with the encoded noise scale-deviation transform.
    Binomial {
        link: InverseLink,
        noise_transform: &'a ScaleDeviationTransform,
    },
    /// A genuine-dispersion mean family (NegativeBinomial / Gamma / Beta /
    /// Tweedie) whose log-precision channel carries `noise_formula` (#913). The
    /// `likelihood` is the family's own [`LikelihoodSpec`]; `base_link` is the
    /// mean inverse link (log, or logit for Beta). The log-precision block
    /// coefficients ride in [`LocationScaleInputs::beta_noise`].
    Dispersion {
        likelihood: LikelihoodSpec,
        base_link: InverseLink,
        family_tag: &'static str,
    },
}

/// Optional link-wiggle metadata persisted alongside a location-scale model.
/// Knots/coefficients are already in raw response units — the Gaussian
/// standardization and its inverse remap live inside
/// `fit_gaussian_location_scale_model`, so the save path persists them verbatim.
pub struct LocationScaleWiggle {
    pub knots: Vec<f64>,
    pub degree: usize,
    pub beta_link_wiggle: Vec<f64>,
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
    pub beta_noise: Option<Vec<f64>>,
    pub wiggle: Option<LocationScaleWiggle>,
}

/// Assemble the canonical (non-survival) location-scale payload — single source
/// of truth for that on-disk contract. The family/likelihood is resolved from
/// the [`LocationScaleResponse`] so the binomial branch never persists a wrong
/// probit likelihood that a caller must patch afterwards.
pub fn assemble_location_scale_payload(
    inputs: LocationScaleInputs,
    response: LocationScaleResponse<'_>,
    source: SavedModelSourceMetadata,
) -> Result<FittedModelPayload, String> {
    inputs
        .fit_result
        .require_posterior_mean("location-scale saved-model assembly")
        .map_err(|error| error.to_string())?;
    let (family_tag, likelihood, base_link, link, response_scale, noise_transform) = match response
    {
        LocationScaleResponse::Gaussian {
            response_scale,
            base_link,
        } => (
            "gaussian-location-scale".to_string(),
            LikelihoodSpec::gaussian_identity(),
            // Gaussian location-scale does not carry a base link in its family
            // state; the resolved link is persisted in `payload.link` below so
            // prediction can recover it.
            None,
            Some(base_link.unwrap_or(InverseLink::Standard(StandardLink::Identity))),
            Some(response_scale),
            None,
        ),
        LocationScaleResponse::Binomial {
            link,
            noise_transform,
        } => {
            let likelihood = inverse_link_to_binomial_spec(&link).map_err(|e| {
                format!("failed to resolve LikelihoodSpec for binomial location-scale link {link:?}: {e}")
            })?;
            (
                "binomial-location-scale".to_string(),
                likelihood,
                Some(link.clone()),
                Some(link),
                None,
                Some(noise_transform),
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
            None,
        ),
    };

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        inputs.formula,
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood,
            base_link,
        },
        family_tag,
    );
    payload.unified = Some(inputs.fit_result.clone());
    payload.fit_result = Some(inputs.fit_result);
    payload.data_schema = Some(inputs.data_schema);
    payload.link = link;
    payload.formula_noise = Some(inputs.noise_formula);
    payload.beta_noise = inputs.beta_noise;
    payload.gaussian_response_scale = response_scale;
    if let Some(transform) = noise_transform {
        payload.noise_projection = Some(
            transform
                .projection_coef
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
        );
        payload.noise_center = Some(transform.weighted_column_mean.to_vec());
        payload.noise_scale = Some(transform.rescale.to_vec());
        payload.noise_non_intercept_start = Some(transform.non_intercept_start);
        payload.noise_projection_ridge_alpha = Some(transform.projection_ridge_alpha);
    }
    payload.resolved_termspec = Some(inputs.resolved_termspec);
    payload.resolved_termspec_noise = Some(inputs.resolved_termspec_noise);
    if let Some(wiggle) = inputs.wiggle {
        payload.linkwiggle_knots = Some(wiggle.knots);
        payload.linkwiggle_degree = Some(wiggle.degree);
        payload.beta_link_wiggle = Some(wiggle.beta_link_wiggle);
    }
    source.apply_to(&mut payload);
    Ok(payload)
}

/// Source-agnostic semantic content of a survival marginal-slope
/// (Royston-Parmar net) saved model. Centralizing assembly also fixes the
/// FFI's prior omission of the `*_slopes`/`*_columns`/`slope_formulas`
/// vector mirrors the CLI wrote.
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
    /// Mutually exclusive; both `None` when the gate did not fire.
    pub latent_z_rank_int_calibration: Option<LatentZRankIntCalibration>,
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
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
fn new_royston_parmar_survival_payload(
    formula: String,
    fit_result: UnifiedFitResult,
    data_schema: DataSchema,
    survival_likelihood_label: &str,
    survival_distribution: Option<ResidualDistribution>,
    frailty: crate::survival::lognormal_kernel::FrailtySpec,
) -> FittedModelPayload {
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
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
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(data_schema);
    payload
}

/// Assemble the canonical survival marginal-slope payload — single source of
/// truth for that Royston-Parmar / Gaussian-residual on-disk contract.
pub fn assemble_survival_marginal_slope_payload(
    inputs: SurvivalMarginalSlopeInputs<'_>,
    source: SavedModelSourceMetadata,
) -> FittedModelPayload {
    let mut payload = new_royston_parmar_survival_payload(
        inputs.formula,
        inputs.fit_result,
        inputs.data_schema,
        &inputs.survival_likelihood_label,
        Some(ResidualDistribution::Gaussian),
        inputs.frailty,
    );
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
    payload.slope_formula = Some(inputs.slope_formula.clone());
    payload.slope_formulas = Some(vec![inputs.slope_formula]);
    payload.z_column = Some(inputs.z_column.clone());
    payload.z_columns = Some(vec![inputs.z_column]);
    payload.latent_z_normalization = Some(inputs.latent_z_normalization);
    // Not an assumption: the survival marginal-slope row program is the
    // closed-form standard-normal probit lowering and owns no empirical-grid
    // branch, so its latent-measure gate is asked for
    // `EmpiricalLatentMeasureSupport::StandardNormalOnly` and the invariant is
    // enforced at the gate's call site in
    // `survival/marginal_slope/latent_measure.rs`. What the gate CAN vary is the
    // pre-transform applied to z before that kernel, and that is the pair below.
    payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
    payload.latent_z_rank_int_calibration = inputs.latent_z_rank_int_calibration;
    payload.latent_z_conditional_calibration = inputs.latent_z_conditional_calibration;
    payload.baseline_slope = Some(inputs.baseline_slope);
    payload.baseline_slopes = Some(vec![inputs.baseline_slope]);
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
    payload
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
    /// Rigid time-block beta, persisted only by the cause-specific CLI path.
    pub survival_beta_time: Option<Vec<f64>>,
    pub timewiggle: Option<SurvivalTimewiggle>,
}

/// Assemble the canonical survival transformation payload — single source of
/// truth for the Royston-Parmar transformation on-disk contract.
pub fn assemble_survival_transformation_payload(
    inputs: SurvivalTransformationInputs,
    source: SavedModelSourceMetadata,
) -> FittedModelPayload {
    let mut payload = new_royston_parmar_survival_payload(
        inputs.formula,
        inputs.fit_result,
        inputs.data_schema,
        &inputs.survival_likelihood_label,
        None,
        crate::survival::lognormal_kernel::FrailtySpec::None,
    );
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
    payload.survival_beta_time = inputs.survival_beta_time;
    payload.resolved_termspec = Some(inputs.resolved_termspec);
    source.apply_to(&mut payload);
    payload
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
    pub beta_link_wiggle: Option<Vec<f64>>,
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
    pub survival_beta_time: Vec<f64>,
    pub survival_beta_threshold: Vec<f64>,
    pub survival_beta_log_sigma: Vec<f64>,
    pub resolved_thresholdspec: TermCollectionSpec,
    pub resolved_log_sigmaspec: TermCollectionSpec,
}

/// Assemble the canonical survival location-scale payload (the single source of
/// truth for that on-disk contract).
pub fn assemble_survival_location_scale_payload(
    inputs: SurvivalLocationScaleInputs,
    source: SavedModelSourceMetadata,
) -> FittedModelPayload {
    let survival_distribution =
        residual_distribution_from_inverse_link(&inputs.fitted_inverse_link);
    let mut payload = new_royston_parmar_survival_payload(
        inputs.formula,
        inputs.fit_result,
        inputs.data_schema,
        &inputs.survival_likelihood_label,
        survival_distribution,
        crate::survival::lognormal_kernel::FrailtySpec::None,
    );
    payload.link = Some(inputs.fitted_inverse_link);
    payload.linkwiggle_degree = inputs.linkwiggle_degree;
    payload.linkwiggle_knots = inputs.linkwiggle_knots;
    payload.beta_link_wiggle = inputs.beta_link_wiggle;
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
    payload.survival_beta_time = Some(inputs.survival_beta_time);
    payload.survival_beta_threshold = Some(inputs.survival_beta_threshold);
    payload.survival_beta_log_sigma = Some(inputs.survival_beta_log_sigma);
    payload.survival_distribution = survival_distribution;
    payload.resolved_termspec = Some(inputs.resolved_thresholdspec);
    payload.resolved_termspec_noise = Some(inputs.resolved_log_sigmaspec);
    source.apply_to(&mut payload);
    payload
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
    pub beta_time: Vec<f64>,
    pub resolved_termspec: TermCollectionSpec,
}

/// Assemble the canonical latent survival / latent binary payload.
pub fn assemble_latent_window_payload(
    inputs: LatentWindowInputs,
    source: SavedModelSourceMetadata,
) -> FittedModelPayload {
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        inputs.formula,
        ModelKind::Survival,
        inputs.family,
        inputs.model_class_label,
    );
    payload.unified = Some(inputs.fit_result.clone());
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
    payload.survival_beta_time = Some(inputs.beta_time);
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
    inference_notes: Vec<String>,
) {
    payload.group_metadata = fit_config.group_metadata.clone();
    payload.training_table_kind = fit_config.training_table_kind.clone();
    payload.inference_notes = inference_notes;
}

/// One authoritative "formula fit → saved payload" service: materialize once,
/// dispatch on the request variant, fit, and assemble the persistence payload.
/// Both front ends (CLI, Python FFI) must route through this function so a fit
/// requested through any surface produces an identical saved model. (#2470)
pub fn fit_formula_to_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
) -> Result<FittedModelPayload, WorkflowError> {
    // Expectile (Newey–Powell LAWS) family (#1777): the expectile estimator is an
    // OUTER driver that wraps the standard Gaussian-identity GAM with iterative
    // asymmetric reweighting, so it is selected *before* `materialize` (which has
    // no expectile arm) — exactly as the in-process `fit_from_formula` does. We
    // route it through the single shared dispatch seam so the Python API reaches
    // the same estimator the library call does instead of failing with
    // `unknown family 'expectile(τ)'`. The driver returns an ordinary
    // `StandardFitResult`, so the persistence payload is built by the same
    // `assemble_standard_payload` used for every other standard fit.
    if let Some(expectile_result) = fit_expectile_if_requested(&formula, dataset, fit_config)? {
        let mut payload = assemble_standard_payload(StandardPayloadInputs {
            formula,
            dataset,
            fit_config,
            result: expectile_result,
        })?;
        // The LAWS driver materializes its inner Gaussian design itself; there are
        // no outer materialize advisories to carry (matches `fit_from_formula`).
        apply_request_metadata(&mut payload, fit_config, Vec::new());
        return Ok(payload);
    }
    // Calibrated marginal-slope chain (#461): when a CTN Stage-1 recipe is present
    // (config.ctn_stage1), the marginal-slope materializer cross-fits the CTN and
    // produces the calibrated `z` out-of-fold — no z_column is needed and no
    // Stage-1 pre-fit / synthetic column round-trip is performed here. The recipe
    // rides on fit_config straight into materialize.
    // Standard-fit dispatch must materialize at the adaptive structural start:
    // this request becomes the first fitted design below. Other estimator
    // materializers do not consume this standard-only orchestration field.
    let mut dispatch_config = fit_config.clone();
    dispatch_config.spatial_center_counts = Some(Vec::new());
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
                    .map_err(|reason| WorkflowError::IntegrationFailed { reason })?;
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
            // Persist the response standardization factor the fit applied so
            // prediction reconstructs the σ floor at `response_scale·0.01`,
            // keeping predictive σ response-scale-equivariant (#884). The fit
            // already mapped the log-σ `exp(η)` term to raw units via the
            // `+ln(response_scale)` intercept shift; only the additive floor
            // still needs the factor at reconstruction time.
            let response_scale = ls_result.response_scale;
            payload_for_gaussian_location_scale(
                formula,
                dataset,
                fit_config,
                ls_result,
                response_scale,
            )?
        }
        FitRequest::BinomialLocationScale(ls_request) => {
            let weights = ls_request.spec.weights.clone();
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
                &weights,
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
    Ok(assemble_transformation_normal_payload(
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
    ))
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
            latent_z_rank_int_calibration: ms_result.latent_z_rank_int_calibration.clone(),
            latent_z_conditional_calibration: ms_result.latent_z_conditional_calibration.clone(),
            score_warp_runtime: ms_result.score_warp_runtime.as_ref(),
            link_dev_runtime: ms_result.link_dev_runtime.as_ref(),
            base_link,
            frailty,
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
        survival_likelihood_modename, survival_marginal_slope_offset_baseline_config,
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
    // The saved baseline chart is the one the fit CERTIFIED, not a re-parse of
    // the request. A request names only the target (`--baseline-target
    // weibull`) and leaves scale and shape to the fit's own ψ coordinates;
    // re-parsing it here demanded `--baseline-scale > 0` and refused to save
    // exactly the fits that estimated their chart (gam#2765: a follow-up-varying
    // slope on a Weibull chart converged and then could not leave the process).
    // `baseline_config` on the fit result is the chart at the certified θ.
    let baseline_cfg = ms_result.baseline_config.clone();
    let likelihood_mode = parse_survival_likelihood_mode(fit_config.resolved_survival_likelihood())?;
    let time_cfg = if parsed.timewiggle.is_some() {
        crate::survival::construction::SurvivalTimeBasisConfig::None
    } else {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
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
        Some((
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )),
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
    let saved_offset_baseline =
        survival_marginal_slope_offset_baseline_config(&age_exit, &baseline_cfg);
    let (persisted_rank_int, persisted_conditional) =
        ms_result.persisted_latent_z_calibrations()?;

    // Thin adapter over the shared core assembler. The FFI's source-specific
    // work is re-deriving the survival response columns, baseline config, and
    // time basis from the formula + FitConfig and freezing its term collections
    // from their designs; the semantic payload is assembled by the same core
    // path the CLI uses, so the two save routes produce identical contracts.
    Ok(assemble_survival_marginal_slope_payload(
        SurvivalMarginalSlopeInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result: ms_result.fit.clone(),
            frailty,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: "net".to_string(),
            baseline_cfg: saved_offset_baseline,
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
            latent_z_rank_int_calibration: persisted_rank_int,
            latent_z_conditional_calibration: persisted_conditional,
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
    ))
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
            survival_beta_time: None,
            timewiggle,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: None,
        },
    );
    Ok(payload)
}

fn payload_for_gaussian_location_scale(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    ls_result: GaussianLocationScaleFitResult,
    response_scale: f64,
) -> Result<FittedModelPayload, String> {
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
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());
    let wiggle = location_scale_wiggle_from_parts(
        ls_result.wiggle_knots,
        ls_result.wiggle_degree,
        ls_result.beta_link_wiggle,
    );

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
            beta_noise: scale_beta,
            wiggle,
        },
        LocationScaleResponse::Gaussian {
            response_scale,
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

/// Map the optional `(knots, degree, beta)` link-wiggle parts a location-scale
/// fit may produce into the shared [`LocationScaleWiggle`] form. All three are
/// present together or not at all.
fn location_scale_wiggle_from_parts(
    knots: Option<Array1<f64>>,
    degree: Option<usize>,
    beta_link_wiggle: Option<Vec<f64>>,
) -> Option<LocationScaleWiggle> {
    match (knots, degree, beta_link_wiggle) {
        (Some(knots), Some(degree), Some(beta_link_wiggle)) => Some(LocationScaleWiggle {
            knots: knots.to_vec(),
            degree,
            beta_link_wiggle,
        }),
        _ => None,
    }
}

fn payload_for_binomial_location_scale(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    link_kind: InverseLink,
    weights: &Array1<f64>,
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

    let dense_mean = ls_result
        .fit
        .mean_design
        .design
        .try_to_dense_by_chunks("binomial location-scale mean design")?;
    let dense_noise = ls_result
        .fit
        .noise_design
        .design
        .try_to_dense_by_chunks("binomial location-scale noise design")?;
    let non_intercept_start = ls_result
        .fit
        .noise_design
        .intercept_range
        .end
        .min(ls_result.fit.noise_design.design.ncols());
    let binomial_noise_transform =
        build_scale_deviation_transform(&dense_mean, &dense_noise, weights, non_intercept_start)
            .map_err(|err| format!("failed to encode binomial noise transform: {err}"))?;

    let fit = ls_result.fit.fit;
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());
    let wiggle = location_scale_wiggle_from_parts(
        ls_result.wiggle_knots,
        ls_result.wiggle_degree,
        ls_result.beta_link_wiggle,
    );

    // Thin adapter over the shared core assembler; the FFI freezes the threshold
    // and noise specs from their designs, encodes the binomial noise
    // scale-deviation transform, and reads offset columns from the FitConfig.
    // See `assemble_location_scale_payload`.
    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            beta_noise: scale_beta,
            wiggle,
        },
        LocationScaleResponse::Binomial {
            link: link_kind,
            noise_transform: &binomial_noise_transform,
        },
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
/// coefficients ride in `beta_noise`; there is no link-wiggle and no response
/// standardization for these families.
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
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());

    assemble_location_scale_payload(
        LocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            noise_formula,
            resolved_termspec: frozen_meanspec,
            resolved_termspec_noise: frozen_noisespec,
            fit_result: fit,
            beta_noise: scale_beta,
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
        parse_survival_baseline_config, parse_survival_likelihood_mode,
        survival_likelihood_modename,
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
    let baseline_cfg = parse_survival_baseline_config(
        &fit_config.baseline_target,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    let likelihood_mode = parse_survival_likelihood_mode(fit_config.resolved_survival_likelihood())?;

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
    Ok(assemble_survival_location_scale_payload(
        SurvivalLocationScaleInputs {
            formula,
            data_schema: dataset.schema.clone(),
            fit_result,
            fitted_inverse_link: fitted_inverse_link.clone(),
            linkwiggle_degree: ls_result.wiggle_degree,
            linkwiggle_knots: ls_result.wiggle_knots.as_ref().map(|k| k.to_vec()),
            beta_link_wiggle: ls_result
                .fit
                .fit
                .beta_link_wiggle()
                .as_ref()
                .map(|b| b.to_vec()),
            baseline_timewiggle: None,
            survival_entry: entryname,
            survival_exit: exitname,
            survival_event: eventname,
            survivalspec: "net".to_string(),
            baseline_cfg,
            time_basis,
            survival_likelihood_label: survival_likelihood_modename(likelihood_mode).to_string(),
            time_parameterization: ls_result.fit.time_parameterization,
            threshold_time_basis: ls_result.fit.threshold_time_basis.clone(),
            log_sigma_time_basis: ls_result.fit.log_sigma_time_basis.clone(),
            formula_noise: None,
            survival_beta_time: ls_result.fit.fit.beta_time().to_vec(),
            survival_beta_threshold: ls_result.fit.fit.beta_threshold().to_vec(),
            survival_beta_log_sigma: ls_result.fit.fit.beta_log_sigma().to_vec(),
            resolved_thresholdspec,
            resolved_log_sigmaspec,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: fit_config.offset_column.clone(),
            noise_offset_column: fit_config.noise_offset_column.clone(),
        },
    ))
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
    is_survival: bool,
    time_basis: Option<SavedSurvivalTimeBasis>,
) -> Result<FittedModelPayload, String> {
    use crate::survival::construction::parse_survival_baseline_config;

    // Carried from the materialization that produced this fit, not re-derived
    // (#2470) — see `payload_for_survival_location_scale` for the anchor
    // divergence this closes.
    let time_basis = time_basis.ok_or_else(|| {
        "latent survival/binary payload requires the materialized survival time basis".to_string()
    })?;
    let parsed = parse_formula(&formula).map_err(|err| {
        format!("failed to re-parse latent survival formula for FFI payload: {err}")
    })?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "latent survival/binary FFI requires Surv(...) response".to_string())?;
    let baseline_cfg = parse_survival_baseline_config(
        &fit_config.baseline_target,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;

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

    let beta_time = fit.beta_time().to_vec();
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
            beta_time,
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
mod apply_timewiggle_beta_tests {
    use super::*;

    /// Minimal payload with both baseline-timewiggle slots unset. Uses the
    /// fixture-free `LatentBinary` family so the test needs no `LikelihoodSpec`.
    fn empty_payload() -> FittedModelPayload {
        FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
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

    #[test]
    fn flexible_fit_payload_uses_the_full_six_plus_eleven_penalty_topology_2748() {
        let dataset = six_column_flexible_binomial_fixture();
        let formula =
            "y ~ x0 + x1 + x2 + x3 + x4 + link(type=flexible(logit))";
        let config = FitConfig {
            family: Some("binomial".to_string()),
            ..FitConfig::default()
        };
        let FitResult::Standard(result) = fit_from_formula(formula, &dataset, &config)
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

        let payload = assemble_standard_payload(StandardPayloadInputs {
            formula: formula.to_string(),
            dataset: &dataset,
            fit_config: &config,
            result,
        })
        .expect("payload assembly must use the full realized raw penalty topology");
        let fit = payload
            .fit_result
            .expect("standard payload must retain its canonical fit result");
        assert_eq!(
            fit.artifacts.null_space_dim,
            Some(mean_dim),
            "the full-rank LinkWiggle penalty leaves exactly the six unpenalized mean coordinates",
        );
        assert!(
            fit.artifacts
                .null_space_logdet
                .is_some_and(f64::is_finite),
            "the null-space Hessian log-determinant must be finite",
        );
    }
}
