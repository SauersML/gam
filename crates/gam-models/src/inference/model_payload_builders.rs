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
    DeviationRuntime, LatentMeasureKind, LatentZConditionalCalibration, LatentZRankIntCalibration,
};
use crate::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL;
use crate::fit_orchestration::drivers::freeze_term_collection_from_design;
use crate::fit_orchestration::{FitConfig, StandardFitResult};
use crate::inference::model::{
    FittedFamily, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind, SavedAnchorComponent,
    SavedAnchorKind, SavedCompiledFlexBlock, SavedLatentZNormalization, SavedResidualCascade,
    SavedSplineScan, TransformationScoreCalibration,
};
use crate::scale_design::ScaleDeviationTransform;
use crate::survival::construction::{
    SavedSurvivalTimeBasis, SurvivalBaselineConfig, survival_baseline_targetname,
};
use crate::survival::location_scale::{
    ResidualDistribution, residual_distribution_from_inverse_link,
};
use crate::transformation_normal::TransformationNormalFamily;
use faer::Side;
use gam_data::{DataSchema, EncodedDataset};
use gam_linalg::faer_ndarray::{FaerCholesky, array2_to_nested_vec};
use gam_problem::types::{
    InverseLink, LikelihoodSpec, ResponseFamily, StandardLink, inverse_link_to_binomial_spec,
};
use gam_solve::estimate::{
    FittedLinkState, UnifiedFitResult, saved_latent_cloglog_state_from_fit,
    saved_mixture_state_from_fit, saved_sas_state_from_fit,
};
use gam_terms::smooth::{TermCollectionDesign, TermCollectionSpec};
use ndarray::{Array1, Array2, s};

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

fn standard_null_space_metadata(
    design: &TermCollectionDesign,
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
    let p = design.design.ncols();
    if hessian_dim < p {
        return Err(format!(
            "null-space Hessian logdet design/Hessian mismatch: design has {p} columns but \
             Hessian is only {hessian_dim}x{hessian_dim}"
        ));
    }
    if design.penalties.is_empty() {
        return Ok((0, 0.0));
    }
    let hessian = if hessian_dim > p {
        hessian.slice(s![0..p, 0..p]).to_owned()
    } else {
        hessian.clone()
    };
    let mut penalty = Array2::<f64>::zeros((p, p));
    for (idx, block) in design.penalties.iter().enumerate() {
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
    let projected = hessian.dot(&null_basis);
    let mut restricted = null_basis.t().dot(&projected);
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
    let expectile = fit_config.family.as_deref().is_some_and(|family| {
        let family = family.trim().to_ascii_lowercase();
        family == "expectile" || family.starts_with("expectile(")
    });
    if expectile
        || !family.is_gaussian_identity()
        || fit_config.weight_column.is_some()
        || fit_config.offset_column.is_some()
        || fit_config.flexible_link
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
    let jackknife = crate::inference::full_conformal::GaussianJackknifePlusStats::from_design_unit_weight_normal_matrix(
        x.as_ref(),
        &y,
        &weights,
        normal_matrix,
    )
    .ok();
    let full = crate::inference::full_conformal::ExactFullConformalSubstrate::from_design_unit_weight_normal_matrix(
        x.as_ref(),
        &y,
        &weights,
        normal_matrix,
    )
    .ok();
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
        adaptive_diagnostics,
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
    let (null_space_dim, null_space_logdet) = standard_null_space_metadata(&design, &fit)?;
    fit.artifacts.null_space_dim = Some(null_space_dim);
    fit.artifacts.null_space_logdet = Some(null_space_logdet);
    let family = fit
        .likelihood_family
        .clone()
        .unwrap_or_else(LikelihoodSpec::gaussian_identity);
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
        family.name().to_string(),
    );
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
    payload.adaptive_regularization_diagnostics = adaptive_diagnostics;
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
    pub logslope_formula: String,
    pub z_column: String,
    pub resolved_marginalspec: TermCollectionSpec,
    pub resolved_logslopespec: TermCollectionSpec,
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
    pub baseline_logslope: f64,
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
///    re-inversion), so the kept `[β_m | β_logslope | …]` covariance is the
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
    let UnifiedFitResult {
        mut blocks,
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
        p_marginal,
        baseline_marginal,
        baseline_logslope,
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
    payload.formula_logslope = Some(logslope_formula.clone());
    payload.z_column = Some(z_column.clone());
    payload.formula_logslopes = Some(vec![logslope_formula]);
    payload.z_columns = Some(vec![z_column]);
    payload.latent_z_normalization = Some(latent_z_normalization);
    payload.latent_measure = Some(latent_measure);
    payload.latent_z_rank_int_calibration = latent_z_rank_int_calibration;
    payload.latent_z_conditional_calibration = latent_z_conditional_calibration;
    payload.marginal_baseline = Some(baseline_marginal);
    payload.logslope_baseline = Some(baseline_logslope);
    payload.logslope_baselines = Some(vec![baseline_logslope]);
    payload.link = Some(base_link);
    payload.resolved_termspec = Some(resolved_marginalspec);
    payload.resolved_termspec_logslopes = Some(vec![resolved_logslopespec.clone()]);
    payload.resolved_termspec_logslope = Some(resolved_logslopespec);
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
    payload.transformation_score_calibration = Some(score_calibration);
    source.apply_to(&mut payload);
    payload
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
/// FFI's prior omission of the `*_logslopes`/`*_columns`/`formula_logslopes`
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
    pub ridge_lambda: f64,
    pub survival_likelihood_label: String,
    pub resolved_marginalspec: TermCollectionSpec,
    pub resolved_logslopespec: TermCollectionSpec,
    pub logslope_formula: String,
    pub z_column: String,
    pub latent_z_normalization: SavedLatentZNormalization,
    pub baseline_logslope: f64,
    pub score_warp_runtime: Option<&'a DeviationRuntime>,
    pub link_dev_runtime: Option<&'a DeviationRuntime>,
    /// Width `p₁` of the absorbed Stage-1 influence block (#461) when the fit
    /// hosted a dedicated additive absorber. Predict drops the absorber's `γ`;
    /// this is persisted only so the predictor accounts for the extra trailing
    /// block in the saved block count.
    pub influence_absorber_width: Option<usize>,
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
    payload.survivalridge_lambda = Some(inputs.ridge_lambda);
    payload.survival_likelihood = Some(inputs.survival_likelihood_label);
    payload.survival_distribution = Some(ResidualDistribution::Gaussian);
    payload.link = Some(InverseLink::Standard(StandardLink::Probit));
    payload.resolved_termspec = Some(inputs.resolved_marginalspec);
    payload.resolved_termspec_logslopes = Some(vec![inputs.resolved_logslopespec.clone()]);
    payload.resolved_termspec_logslope = Some(inputs.resolved_logslopespec);
    payload.formula_logslope = Some(inputs.logslope_formula.clone());
    payload.formula_logslopes = Some(vec![inputs.logslope_formula]);
    payload.z_column = Some(inputs.z_column.clone());
    payload.z_columns = Some(vec![inputs.z_column]);
    payload.latent_z_normalization = Some(inputs.latent_z_normalization);
    payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
    payload.logslope_baseline = Some(inputs.baseline_logslope);
    payload.logslope_baselines = Some(vec![inputs.baseline_logslope]);
    payload.score_warp_runtime = inputs
        .score_warp_runtime
        .map(serialize_anchored_deviation_runtime);
    payload.link_deviation_runtime = inputs
        .link_dev_runtime
        .map(serialize_anchored_deviation_runtime);
    payload.influence_absorber_width = inputs.influence_absorber_width;
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
    pub ridge_lambda: f64,
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
    payload.survivalridge_lambda = Some(inputs.ridge_lambda);
    payload.survival_likelihood = Some(inputs.survival_likelihood_label);
    payload.survival_beta_time = inputs.survival_beta_time;
    payload.resolved_termspec = Some(inputs.resolved_termspec);
    source.apply_to(&mut payload);
    payload
}

/// Source-agnostic semantic content of a survival location-scale
/// (Royston-Parmar with a learned residual link) saved model. Centralizing
/// fixes the drift where CLI and FFI disagreed on `formula_noise`,
/// `baseline_timewiggle_*`, and `survival_noise_projection_ridge_alpha`.
pub struct SurvivalLocationScaleInputs<'a> {
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
    pub ridge_lambda: f64,
    pub survival_likelihood_label: String,
    pub formula_noise: Option<String>,
    pub survival_beta_time: Vec<f64>,
    pub survival_beta_threshold: Vec<f64>,
    pub survival_beta_log_sigma: Vec<f64>,
    pub noise_transform: &'a ScaleDeviationTransform,
    pub resolved_thresholdspec: TermCollectionSpec,
    pub resolved_log_sigmaspec: TermCollectionSpec,
}

/// Assemble the canonical survival location-scale payload (the single source of
/// truth for that on-disk contract).
pub fn assemble_survival_location_scale_payload(
    inputs: SurvivalLocationScaleInputs<'_>,
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
    payload.survivalridge_lambda = Some(inputs.ridge_lambda);
    payload.survival_likelihood = Some(inputs.survival_likelihood_label);
    payload.formula_noise = inputs.formula_noise;
    payload.survival_beta_time = Some(inputs.survival_beta_time);
    payload.survival_beta_threshold = Some(inputs.survival_beta_threshold);
    payload.survival_beta_log_sigma = Some(inputs.survival_beta_log_sigma);
    payload.survival_noise_projection = Some(
        inputs
            .noise_transform
            .projection_coef
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    );
    payload.survival_noise_center = Some(inputs.noise_transform.weighted_column_mean.to_vec());
    payload.survival_noise_scale = Some(inputs.noise_transform.rescale.to_vec());
    payload.survival_noise_non_intercept_start = Some(inputs.noise_transform.non_intercept_start);
    payload.survival_noise_projection_ridge_alpha =
        Some(inputs.noise_transform.projection_ridge_alpha);
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
    pub ridge_lambda: f64,
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
    payload.survivalridge_lambda = Some(inputs.ridge_lambda);
    payload.resolved_termspec = Some(inputs.resolved_termspec);
    source.apply_to(&mut payload);
    payload
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
