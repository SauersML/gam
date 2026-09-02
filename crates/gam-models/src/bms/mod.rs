use crate::cubic_cell_kernel as exact_kernel;
use crate::custom_family::{
    BatchedOuterGradientTerms, BlockEffectiveJacobian, BlockWorkingSet, BlockwiseFitOptions,
    CustomFamily, CustomFamilyJointHyperModeSelection, CustomFamilyWarmStart, EvalMode,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace, FamilyEvaluation,
    FamilyLinearizationState, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    custom_family_outer_derivatives, evaluate_custom_family_joint_hyper_best_mode_shared,
    fit_custom_family, fit_custom_family_fixed_log_lambdas_from_mode_selection,
    joint_hyper_options_for_outer_tolerance,
};
use crate::exact_mode_branch::ExactCoefficientModeBranch;
use crate::fit_orchestration::drivers::{
    ExactJointEfsEvaluation, ExactJointEvaluation, ExactJointHyperSetup, SpatialFitProvenance,
    apply_spatial_anisotropy_pilot_initializer, build_term_collection_designs_and_freeze_joint,
    optimize_spatial_length_scale_exact_joint, spatial_length_scale_term_indices,
};
use crate::marginal_slope_shared::{
    CoeffSupport, ObservedDenestedCellPartials, SparsePrimaryCoeffJetView, add_optional_matrix,
    add_optional_vector, add_two_surface_psi_outer,
    build_denested_partition_cells as shared_denested_partition_cells, chunked_row_reduction,
    eval_coeff4_at, first_parameter_directional_order2_terms, first_parameter_order2_terms,
    observed_denested_cell_partials as shared_observed_denested_cell_partials, outer_row_indices,
    outer_weighted_rows, parameter_block_specs_match_rows, probit_frailty_scale,
    psi_derivative_location, scale_coeff4, second_parameter_order2_terms,
};
use crate::model_types::UnifiedFitResult;
use crate::outer_subsample::WeightedOuterRow;
use crate::parameter_block::ParameterBlockInput;
use crate::probability::{
    chi_square_sf, normal_cdf, normal_logcdf, normal_pdf, normal_two_sided_probability,
    signed_probit_logcdf_and_mills_ratio, standard_normal_quantile,
};
use crate::row_kernel::{
    RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache, row_kernel_gradient,
    row_kernel_hessian_dense, row_kernel_log_likelihood,
};
use crate::spatial_psi_bridge::{
    CoefficientSpatialPsiBlockTransform, build_block_spatial_psi_derivatives,
    build_block_spatial_psi_derivatives_with_transform,
};
use crate::survival::lognormal_kernel::{FrailtyScale, FrailtySpec};
use gam_linalg::matrix::{DesignMatrix, SymmetricMatrix};
use gam_problem::{
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    HyperOperator, InverseLink, StandardLink, WigglePenaltyConfig,
};
use gam_solve::estimate::reml::reml_outer_engine::{DenseSpectralOperator, HessianFactorization};
use gam_solve::pirls::LinearInequalityConstraints;
use gam_terms::smooth::{
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, s};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod alo_replay;
pub mod deviation_runtime;
pub mod gpu;
pub(crate) use alo_replay::exact_runtime_from_saved;
pub use alo_replay::{BernoulliMarginalSlopeAloRowGeometry, BernoulliMarginalSlopeAloRowInput, BernoulliMarginalSlopeSavedAloReplay, BernoulliMarginalSlopeSavedAloRowGeometry};
pub(crate) use alo_replay::{
    BernoulliMarginalSlopeSavedAloReplayInput, replay_saved_bernoulli_marginal_slope_alo,
};
pub use deviation_runtime::DeviationRuntime;
pub use deviation_runtime::ParametricAnchorBlock;

/// Above this size, FLEX spatial length-scale optimization uses the pilot
/// geometry initializer and skips the iterative joint κ/ψ outer loop. This is
/// a spatial-optimizer policy only; it must not gate exact outer Hessian
/// capability or row-cell moment materialization.
pub(crate) const BMS_FLEX_SPATIAL_OUTER_PILOT_ROW_THRESHOLD: usize = 50_000;

#[derive(Clone, Debug)]
pub struct DeviationBlockConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub penalty_orders: Vec<usize>,
    pub double_penalty: bool,
    pub monotonicity_eps: f64,
}

impl Default for DeviationBlockConfig {
    fn default() -> Self {
        WigglePenaltyConfig::cubic_triple_operator_default().into()
    }
}

impl From<WigglePenaltyConfig> for DeviationBlockConfig {
    fn from(cfg: WigglePenaltyConfig) -> Self {
        let penalty_order = *cfg.penalty_orders.iter().max().unwrap_or(&2);
        Self {
            degree: cfg.degree,
            num_internal_knots: cfg.num_internal_knots,
            penalty_order,
            penalty_orders: cfg.penalty_orders,
            double_penalty: cfg.double_penalty,
            monotonicity_eps: cfg.monotonicity_eps,
        }
    }
}

#[derive(Clone)]
pub(crate) struct DeviationPrepared {
    pub(crate) block: ParameterBlockInput,
    pub(crate) runtime: DeviationRuntime,
}

impl std::fmt::Debug for DeviationPrepared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviationPrepared").finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct BernoulliMarginalSlopeTermSpec {
    pub y: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array1<f64>,
    pub base_link: InverseLink,
    pub marginalspec: TermCollectionSpec,
    pub slopespec: TermCollectionSpec,
    pub marginal_offset: Array1<f64>,
    pub slope_offset: Array1<f64>,
    /// GaussianShift frailty on the final probit index: U ~ N(0, σ²) added
    /// to the scalar argument of Φ.  This is exact because the sextic
    /// microcell kernel is preserved — the Gaussian-decoupling identity
    /// E[Φ(η + U)] = Φ(η / √(1+σ²)) rescales the index by 1/τ where
    /// τ = √(1+σ²), and every derivative chain rule factor is polynomial
    /// in τ, so all six kernel derivatives remain closed-form.
    ///
    /// **HazardMultiplier frailty is NOT supported in this family.**
    /// HazardMultiplier frailty + score_warp/linkwiggle cubic marginal-slope
    /// is not finite-state exact.  For hazard-multiplier frailty, use the
    /// standalone LatentCloglogBinomial / LatentSurvival families instead.
    pub frailty: FrailtySpec,
    pub score_warp: Option<DeviationBlockConfig>,
    pub link_dev: Option<DeviationBlockConfig>,
    pub latent_z_policy: LatentZPolicy,
    /// Out-of-fold Stage-1 score-influence Jacobian `J = ∂z/∂θ₁` (n × p₁)
    /// from cross-fitting a CTN transformation-normal Stage-1 model (#461).
    /// When `Some`, the realized leakage directions `Z_infl = diag(s_f·β̂₀)·J`
    /// are absorbed as a null-penalized block so the joint solve makes the
    /// β estimating equation orthogonal to `span(Z_infl)` — the x-dependent
    /// realization of `ψ − Π_η[ψ]`. `None` ⇒ raw `--z-column` with no CTN
    /// Stage-1, in which case the free 1-D `score_warp` spline is the
    /// fallback basis (it spans only the x-free leakage column).
    pub score_influence_jacobian: Option<Array2<f64>>,
}

pub struct BernoulliMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub slopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    pub slope_design: TermCollectionDesign,
    pub baseline_marginal: f64,
    pub baseline_slope: f64,
    pub z_normalization: LatentZNormalization,
    pub latent_measure: LatentMeasureKind,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
    /// Learned or fixed Gaussian-shift frailty SD.  `None` = no frailty.
    pub gaussian_frailty_sd: Option<f64>,
    /// Structured warnings emitted during fit-time setup when a flex
    /// block was fully aliased by its anchor union and got dropped. The
    /// fit proceeds without the dropped block (its contribution to the
    /// joint design was numerically reproducible by the anchor span, so
    /// keeping it would leave the joint Hessian rank-deficient). Empty
    /// for fits where every flex block carried independent directions.
    pub cross_block_warnings: Vec<CrossBlockIdentifiabilityWarning>,
    /// Optional weighted rank inverse-normal (Blom rankit) calibration
    /// installed at fit time when the auto latent-z normality check
    /// failed. `Some(_)` ⇒ the training z was transformed in place via
    /// [`LatentZRankIntCalibration::apply_to_training`] before any
    /// downstream consumer (pooled probit baseline, term-collection
    /// designs, family PIRLS loops) saw it, and the rigid kernel
    /// routes through the standard-normal closed-form path on the
    /// calibrated scale. `None` ⇒ no calibration was applied (training
    /// z already passed the standard-normal diagnostics, or the caller
    /// explicitly selected a non-Auto `LatentMeasureSpec`).
    ///
    /// Persisted to disk so prediction applies the same monotone map
    /// via [`LatentZRankIntCalibration::apply_at_predict`] to incoming
    /// z before the standard-normal kernel runs. The public field name
    /// is `latent_z_rank_int_calibration` — Agent D's persistence
    /// pipeline reads it under that exact identifier.
    pub latent_z_rank_int_calibration: Option<LatentZRankIntCalibration>,
    /// Optional conditional location-scale calibration of the latent score
    /// (#905). `Some(_)` ⇒ the Auto path's conditional `E[z|C]`/`Var(z|C)` Rao
    /// gate detected PC/grouping-dependence that the pooled-marginal gate
    /// cannot see, so the training z was replaced in place by
    /// `ζ = (z − m(C))/√v(C)` (via [`LatentZConditionalCalibration::apply`])
    /// before any downstream consumer saw it. Mutually exclusive with
    /// `latent_z_rank_int_calibration`: rank-INT fixes a pooled-marginal
    /// defect, the conditional correction fixes a conditional-shift defect that
    /// rank-INT provably cannot. Persisted so prediction rebuilds `a(C)` from
    /// the (reproducible) marginal design and applies the identical map.
    pub latent_z_conditional_calibration: Option<LatentZConditionalCalibration>,
}

#[derive(Clone, Debug)]
pub enum LatentZCheckMode {
    Strict,
    WarnOnly,
    Off,
}

#[derive(Clone, Debug)]
pub enum LatentZNormalizationMode {
    None,
    FitWeighted,
    Frozen { mean: f64, sd: f64 },
}

pub const DEFAULT_EMPIRICAL_LATENT_GRID_SIZE: usize = 65;
pub(crate) const AUTO_Z_NORMAL_SKEW_TOL: f64 = 0.10;
pub(crate) const AUTO_Z_NORMAL_KURT_TOL: f64 = 0.25;
pub(crate) const AUTO_Z_NORMAL_KS_TOL: f64 = 0.025;
pub(crate) const AUTO_Z_NORMAL_MAX_ABS: f64 = 8.0;
/// Inner σ level at which the empirical tail mass of latent z is compared
/// against the standard normal's theoretical two-sided tail in the auto
/// normality gate. Chosen well inside `AUTO_Z_NORMAL_MAX_ABS` so a fat inner
/// tail is caught before any single observation trips the hard `max |z|` bound.
pub(crate) const AUTO_Z_NORMAL_TAIL_SIGMA_INNER: f64 = 4.0;
/// Outer σ level for the same tail-mass comparison; catches heavier far-tail
/// excess that the inner level can miss.
pub(crate) const AUTO_Z_NORMAL_TAIL_SIGMA_OUTER: f64 = 6.0;
/// Multiplier applied to the normal's theoretical tail mass before comparison:
/// the empirical tail may be up to this many times the Gaussian tail at the
/// same σ before the gate fails, allowing for finite-sample sampling noise.
pub(crate) const AUTO_Z_NORMAL_TAIL_MASS_SLACK: f64 = 2.0;
/// Absolute additive floor on the inner-σ tail comparison, so the gate does
/// not fail on round-off when the Gaussian tail itself is already tiny.
pub(crate) const AUTO_Z_NORMAL_TAIL_FLOOR_INNER: f64 = 1e-5;
/// Absolute additive floor on the outer-σ tail comparison; smaller than the
/// inner floor because the 6σ Gaussian tail is many orders smaller than 4σ.
pub(crate) const AUTO_Z_NORMAL_TAIL_FLOOR_OUTER: f64 = 1e-8;
/// Significance level for the conditional `E[z|C]` / `Var(z|C)` Rao gate in the
/// core Auto path (#905). When the latent score's conditional mean or variance
/// on the marginal-index span `a(C)` is significant at this level, the Auto
/// path escalates from the pooled-marginal rank-INT to a conditional
/// location-scale correction. Chosen small (0.1%) so the escalation fires only
/// on clear conditional structure, not finite-sample noise — the gate runs once
/// over the whole training sample, so a per-test α this tight still has ample
/// power against the grouping mean-shift the issue names.
pub(crate) const AUTO_Z_CONDITIONAL_RAO_ALPHA: f64 = 1.0e-3;
/// Relative ridge added to the weighted normal equations when regressing the
/// latent score on the marginal-index span for the conditional correction.
/// Stabilizes the solve when `a(C)` is rank-deficient or collinear (penalized
/// spline marginal indices routinely are) without materially biasing the
/// conditional mean/variance fit.
pub(crate) const AUTO_Z_CONDITIONAL_RIDGE_REL: f64 = 1.0e-8;
/// Floor on the fitted conditional variance `v(C)`, as a fraction of the global
/// weighted variance of the latent score. Keeps `ζ = (z−m)/√v` finite and
/// well-scaled where the linear variance model would otherwise fit a
/// non-positive or vanishing conditional variance.
pub(crate) const AUTO_Z_CONDITIONAL_VAR_FLOOR_FRAC: f64 = 1.0e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentMeasureSpec {
    Auto { grid_size: usize },
    StandardNormal,
    GlobalEmpirical { grid_size: usize },
}

impl LatentMeasureSpec {
    pub fn auto_default() -> Self {
        Self::Auto {
            grid_size: DEFAULT_EMPIRICAL_LATENT_GRID_SIZE,
        }
    }
}

impl Default for LatentMeasureSpec {
    fn default() -> Self {
        Self::auto_default()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmpiricalZGrid {
    pub nodes: Vec<f64>,
    pub weights: Vec<f64>,
}

impl EmpiricalZGrid {
    /// Construct a grid whose node/weight invariants (equal length ≥ 2, finite
    /// ascending nodes, finite positive weights, weights summing to 1 within
    /// 1e-8) are enforced up-front. Sorted order is part of the input contract
    /// so hot denested-cell kernels can consume contiguous buckets without a
    /// constructor-side reorder or allocation. Prefer this over building the
    /// struct literally; every code path that goes through `new` satisfies the
    /// same contract that `validate_empirical_z_grid` checks on read.
    pub fn new(nodes: Vec<f64>, weights: Vec<f64>, context: &str) -> Result<Self, String> {
        validate_empirical_z_grid(&nodes, &weights, context)?;
        Ok(Self { nodes, weights })
    }

    /// Iterate over co-indexed `(node, weight)` pairs. Use this instead of
    /// reading `.nodes`/`.weights` separately whenever a loop wants both
    /// arrays in lockstep — eliminates the chance of mismatched indexing.
    #[inline]
    pub fn pairs(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.nodes.iter().copied().zip(self.weights.iter().copied())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
#[derive(Default)]
pub enum LatentMeasureKind {
    #[default]
    StandardNormal,
    GlobalEmpirical {
        grid: EmpiricalZGrid,
    },
    LocalEmpirical {
        feature_cols: Vec<usize>,
        #[serde(default)]
        input_scales: Option<Vec<f64>>,
        centers: Vec<Vec<f64>>,
        grids: Vec<EmpiricalZGrid>,
        top_k: usize,
        bandwidth: f64,
        #[serde(skip)]
        train_row_mixtures: Arc<Vec<Vec<(usize, f64)>>>,
    },
}

impl LatentMeasureKind {
    pub fn validate(&self, context: &str) -> Result<(), String> {
        match self {
            Self::StandardNormal => Ok(()),
            Self::GlobalEmpirical { grid } => {
                validate_empirical_z_grid(&grid.nodes, &grid.weights, context)
            }
            Self::LocalEmpirical {
                feature_cols,
                input_scales,
                centers,
                grids,
                top_k,
                bandwidth,
                ..
            } => {
                if feature_cols.is_empty() {
                    return Err(format!(
                        "{context} local empirical latent measure needs feature columns"
                    ));
                }
                if centers.is_empty() {
                    return Err(format!(
                        "{context} local empirical latent measure needs centers"
                    ));
                }
                if centers.len() != grids.len() {
                    return Err(format!(
                        "{context} local empirical latent measure center/grid length mismatch: centers={}, grids={}",
                        centers.len(),
                        grids.len()
                    ));
                }
                if *top_k == 0 || *top_k > centers.len() {
                    return Err(format!(
                        "{context} local empirical latent measure top_k must be in 1..={}, got {top_k}",
                        centers.len()
                    ));
                }
                if !(*bandwidth).is_finite() || *bandwidth <= 0.0 {
                    return Err(format!(
                        "{context} local empirical latent measure bandwidth must be finite and positive, got {bandwidth}"
                    ));
                }
                if let Some(scales) = input_scales.as_ref() {
                    if scales.len() != feature_cols.len() {
                        return Err(format!(
                            "{context} local empirical latent measure input scale dimension mismatch: scales={}, features={}",
                            scales.len(),
                            feature_cols.len()
                        ));
                    }
                    for (scale_idx, scale) in scales.iter().enumerate() {
                        if !(scale.is_finite() && *scale > 0.0) {
                            return Err(format!(
                                "{context} local empirical latent measure input scale {scale_idx} must be finite and positive, got {scale}"
                            ));
                        }
                    }
                }
                for (center_idx, center) in centers.iter().enumerate() {
                    if center.len() != feature_cols.len() {
                        return Err(format!(
                            "{context} local empirical latent center {center_idx} dimension mismatch: got {}, expected {}",
                            center.len(),
                            feature_cols.len()
                        ));
                    }
                    if center.iter().any(|value| !value.is_finite()) {
                        return Err(format!(
                            "{context} local empirical latent center {center_idx} has non-finite coordinates"
                        ));
                    }
                }
                for (grid_idx, grid) in grids.iter().enumerate() {
                    validate_empirical_z_grid(
                        &grid.nodes,
                        &grid.weights,
                        &format!("{context} local empirical grid {grid_idx}"),
                    )?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn is_empirical(&self) -> bool {
        matches!(
            self,
            Self::GlobalEmpirical { .. } | Self::LocalEmpirical { .. }
        )
    }

    /// Per-row empirical latent grid, borrowed where possible. This sits in
    /// the innermost per-row loops of every criterion/gradient/Hessian
    /// evaluation, so the global grid MUST come back as a borrow — the old
    /// `grid.clone()` here allocated two `grid_size`-length vectors per row
    /// per evaluation across the whole fit. Only the local-mixture path,
    /// which genuinely synthesizes a new grid per row, returns an owned
    /// value.
    pub(crate) fn empirical_grid_for_training_row(
        &self,
        row: usize,
    ) -> Result<Option<std::borrow::Cow<'_, EmpiricalZGrid>>, String> {
        match self {
            Self::StandardNormal => Ok(None),
            Self::GlobalEmpirical { grid } => Ok(Some(std::borrow::Cow::Borrowed(grid))),
            Self::LocalEmpirical {
                grids,
                train_row_mixtures,
                ..
            } => {
                let mixture = train_row_mixtures.get(row).ok_or_else(|| {
                    format!(
                        "local empirical latent measure is missing training mixture for row {row}"
                    )
                })?;
                Ok(Some(std::borrow::Cow::Owned(combine_empirical_grids(
                    grids, mixture,
                )?)))
            }
        }
    }
}

/// Allocation-free heapsort of parallel empirical node/weight storage.
/// Used by the local-mixture constructor, whose concatenated sorted component
/// grids are not globally ordered. Moving pairs in place avoids the third
/// temporary allocation that a `Vec<(node, weight)>` canonicalization would
/// add to that per-row path.
fn sort_empirical_node_weight_pairs(nodes: &mut [f64], weights: &mut [f64]) {
    assert_eq!(
        nodes.len(),
        weights.len(),
        "empirical grid nodes and weights must remain parallel"
    );
    fn sift_down(nodes: &mut [f64], weights: &mut [f64], mut root: usize, end: usize) {
        loop {
            let mut child = 2 * root + 1;
            if child >= end {
                return;
            }
            if child + 1 < end && nodes[child].total_cmp(&nodes[child + 1]).is_lt() {
                child += 1;
            }
            if !nodes[root].total_cmp(&nodes[child]).is_lt() {
                return;
            }
            nodes.swap(root, child);
            weights.swap(root, child);
            root = child;
        }
    }

    let len = nodes.len();
    for root in (0..len / 2).rev() {
        sift_down(nodes, weights, root, len);
    }
    for end in (1..len).rev() {
        nodes.swap(0, end);
        weights.swap(0, end);
        sift_down(nodes, weights, 0, end);
    }
}

pub(crate) fn validate_empirical_z_grid(
    nodes: &[f64],
    weights: &[f64],
    context: &str,
) -> Result<(), String> {
    if nodes.len() != weights.len() {
        return Err(format!(
            "{context} empirical latent measure node/weight length mismatch: nodes={}, weights={}",
            nodes.len(),
            weights.len()
        ));
    }
    if nodes.len() < 2 {
        return Err(format!(
            "{context} empirical latent measure requires at least two nodes"
        ));
    }
    let mut total = 0.0;
    let mut previous_node = f64::NEG_INFINITY;
    for (idx, (&node, &weight)) in nodes.iter().zip(weights.iter()).enumerate() {
        if !node.is_finite() {
            return Err(format!(
                "{context} empirical latent measure node {idx} is non-finite ({node})"
            ));
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(format!(
                "{context} empirical latent measure weight {idx} must be finite and positive, got {weight}"
            ));
        }
        if node < previous_node {
            return Err(format!(
                "{context} empirical latent measure nodes must be sorted ascending, but node {idx} ({node}) is below node {} ({previous_node})",
                idx - 1
            ));
        }
        previous_node = node;
        total += weight;
    }
    if !(total.is_finite() && (total - 1.0).abs() <= 1e-8) {
        return Err(format!(
            "{context} empirical latent measure weights must sum to 1, got {total}"
        ));
    }
    Ok(())
}

pub(crate) fn combine_empirical_grids(
    grids: &[EmpiricalZGrid],
    mixture: &[(usize, f64)],
) -> Result<EmpiricalZGrid, String> {
    if mixture.is_empty() {
        return Err("local empirical latent measure row mixture is empty".to_string());
    }
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for &(grid_idx, grid_weight) in mixture {
        if !grid_weight.is_finite() || grid_weight <= 0.0 {
            return Err(format!(
                "local empirical latent mixture weight must be finite and positive, got {grid_weight}"
            ));
        }
        let grid = grids.get(grid_idx).ok_or_else(|| {
            format!("local empirical latent mixture references missing grid {grid_idx}")
        })?;
        for (node, weight) in grid.pairs() {
            nodes.push(node);
            weights.push(grid_weight * weight);
        }
    }
    let total = weights.iter().copied().sum::<f64>();
    if !(total.is_finite() && total > 0.0) {
        return Err(
            "local empirical latent combined grid has non-positive total weight".to_string(),
        );
    }
    for weight in &mut weights {
        *weight /= total;
    }
    sort_empirical_node_weight_pairs(&mut nodes, &mut weights);
    validate_empirical_z_grid(&nodes, &weights, "local empirical latent combined grid")?;
    Ok(EmpiricalZGrid { nodes, weights })
}

#[derive(Clone, Debug)]
pub struct LatentZPolicy {
    pub check_mode: LatentZCheckMode,
    pub normalization: LatentZNormalizationMode,
    pub latent_measure: LatentMeasureSpec,
    pub mean_tol_multiplier: f64,
    pub sd_tol_multiplier: f64,
    pub max_abs_skew: f64,
    pub max_abs_excess_kurtosis: f64,
}

impl LatentZPolicy {
    pub fn frozen_transformation_normal() -> Self {
        // Defaults relaxed to `WarnOnly` with the same thresholds the
        // exploratory-weighted preset uses (skew ≤ 4.0, |excess kurt| ≤ 20.0).
        // Rationale: the upstream conditional transformation-normal
        // preprocessor may be fit isotropically (no per-axis κ). At large-scale
        // dimensionality (16 PCs, 15 ancestries) an isotropic fit can leave
        // the global latent-z distribution mildly heavy-tailed (skew ≈ 4,
        // excess kurt ≈ 30–40 in synthetic studies) without violating per-
        // grouping mean/variance calibration. The downstream marginal-slope
        // model still uses the latent-Gaussian probit/score-warp link; the
        // emitted warning makes the deviation visible without aborting the
        // fit. Callers that need strict enforcement can construct a custom
        // `LatentZPolicy` with `check_mode: LatentZCheckMode::Strict`.
        Self {
            check_mode: LatentZCheckMode::WarnOnly,
            normalization: LatentZNormalizationMode::Frozen { mean: 0.0, sd: 1.0 },
            latent_measure: LatentMeasureSpec::auto_default(),
            mean_tol_multiplier: 4.0,
            sd_tol_multiplier: 4.0,
            max_abs_skew: 4.0,
            max_abs_excess_kurtosis: 20.0,
        }
    }

    pub fn exploratory_fit_weighted() -> Self {
        Self {
            check_mode: LatentZCheckMode::WarnOnly,
            normalization: LatentZNormalizationMode::FitWeighted,
            latent_measure: LatentMeasureSpec::auto_default(),
            mean_tol_multiplier: 8.0,
            sd_tol_multiplier: 8.0,
            max_abs_skew: 4.0,
            max_abs_excess_kurtosis: 20.0,
        }
    }
}

impl Default for LatentZPolicy {
    fn default() -> Self {
        Self::frozen_transformation_normal()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentZNormalization {
    pub mean: f64,
    pub sd: f64,
}

impl LatentZNormalization {
    pub fn apply(&self, z: &Array1<f64>, context: &str) -> Result<Array1<f64>, String> {
        if !(self.mean.is_finite() && self.sd.is_finite() && self.sd > BMS_VARIANCE_FLOOR) {
            return Err(format!(
                "{context} requires finite latent z normalization with sd > {BMS_VARIANCE_FLOOR:e}; got mean={} sd={}",
                self.mean, self.sd
            ));
        }
        if z.iter().any(|value| !value.is_finite()) {
            return Err(format!("{context} requires finite z values"));
        }
        Ok(z.mapv(|zi| (zi - self.mean) / self.sd))
    }
}

/// Weighted mid-distribution rank inverse-normal transform for the
/// latent score.
///
/// When the latent z fails the standard-normal auto-detection
/// (`latent_z_normal_adequacy`), the BMS family applied to
/// pretend the score is N(0,1) anyway would distort the closed-form
/// probit log-CDF kernel. The historical fallback (local- or
/// global-empirical latent measure) is *mathematically correct* but
/// triggers the per-row intercept Newton solve in the empirical-grid
/// closed-form kernels (`empirical_rigid_primary_grad_hess_closed_form`
/// and its higher-order siblings); at large scale that is the dominant
/// cost.
///
/// **Rank-INT is a modeling choice, not a reparameterisation.** The
/// rigid BMS predictor is *affine* in the latent score
/// (`η = q·√(1+b²) + b·z`), so a nonlinear monotone map of `z` changes
/// the model class and its likelihood — it does not leave them
/// invariant. Applying the calibration redefines the latent axis: the
/// affine model is *specified on the calibrated score* `T(z)`. Nor is
/// the calibrated training sample exactly N(0,1): a finite set of
/// normal scores is discrete, and with heavy ties it can stay far from
/// Gaussian. The closed-form standard-normal kernel is therefore
/// adequate only when the calibrated sample itself passes the same
/// standard-normal adequacy gate applied to raw z
/// (`latent_z_normal_adequacy`);
/// `build_latent_measure_with_geometry` re-checks the calibrated
/// sample and falls back to the mathematically exact global-empirical
/// latent measure when that re-check fails. On the passing path the
/// kept work is the same closed-form
/// `signed_probit_logcdf_and_mills_ratio` evaluation as the
/// no-calibration path; the dropped work is the empirical-grid jet
/// machinery. Persisted to disk so prediction applies the same
/// monotone map to incoming z and re-routes through the closed-form
/// kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentZRankIntCalibration {
    /// Sorted unique positive-mass z values seen during training, ascending.
    /// Knot table for `apply_to_training` / `apply_at_predict`. Zero-weight
    /// knots carry no probability mass and are not stored.
    pub sorted_z: Vec<f64>,
    /// Weighted mid-distribution rank `(W_before + w_knot/2) / W_total` at
    /// each `sorted_z` knot. Strictly increasing, strictly inside `(0, 1)`
    /// (each knot is bounded away from the endpoints by half its own mass),
    /// and invariant to a common rescaling of the weights.
    pub weighted_cdf: Vec<f64>,
    /// Weighted mean of the calibrated training sample. Used as a
    /// sanity-check value on `fit`; should be very close to zero.
    pub post_mean: f64,
    /// Weighted SD of the calibrated training sample. Used as a
    /// sanity-check value on `fit`; should be very close to one.
    pub post_sd: f64,
}

impl LatentZRankIntCalibration {
    /// Fit the weighted rank-INT calibration from training z and weights.
    ///
    /// Algorithm:
    /// 1. Sort rows by ascending z and merge ties into one knot per unique
    ///    z with the tie group's total weight `w_g`; discard zero-mass
    ///    knots.
    /// 2. Weighted mid-distribution rank at each knot:
    ///    `p_g = (W_before + w_g/2) / W_total`,
    ///    with `W_before` the cumulative weight strictly below the knot.
    /// 3. Store `(sorted_z, weighted_cdf = p_g)`.
    ///
    /// The mid-rank depends only on *relative* weights (rescaling every
    /// weight by a common factor leaves every `p_g` unchanged), is strictly
    /// increasing across knots, and lies strictly inside `(0, 1)` — each
    /// knot is separated from the endpoints by half its own mass, so no
    /// ad-hoc clamp is needed and `Φ⁻¹(p_g)` is always finite.
    ///
    /// Returns the calibration plus the post-transform sample's weighted
    /// mean / SD for sanity-check logging.
    pub fn fit(z: &Array1<f64>, weights: &Array1<f64>) -> Result<Self, String> {
        if z.len() != weights.len() {
            return Err(format!(
                "rank-INT calibration: z length {} != weights length {}",
                z.len(),
                weights.len()
            ));
        }
        if z.is_empty() {
            return Err("rank-INT calibration requires at least one observation".to_string());
        }
        let w_total = weights.iter().copied().sum::<f64>();
        if !(w_total.is_finite() && w_total > 0.0) {
            return Err(format!(
                "rank-INT calibration requires positive finite total weight, got {w_total}"
            ));
        }
        for (idx, value) in z.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "rank-INT calibration: z[{idx}] = {value} not finite"
                ));
            }
        }
        for (idx, weight) in weights.iter().enumerate() {
            if !(weight.is_finite() && *weight >= 0.0) {
                return Err(format!(
                    "rank-INT calibration: weight[{idx}] = {weight} not finite/non-negative"
                ));
            }
        }
        let mut order: Vec<usize> = (0..z.len()).collect();
        order.sort_by(|&a, &b| z[a].partial_cmp(&z[b]).unwrap_or(std::cmp::Ordering::Equal));

        let mut sorted_z: Vec<f64> = Vec::with_capacity(z.len());
        let mut weighted_cdf: Vec<f64> = Vec::with_capacity(z.len());
        // Merge ties into one knot per unique z, then assign the weighted
        // mid-distribution rank p_g = (W_before + w_g/2) / W_total. This is
        // the mid-point of the tie group's probability mass, so it depends
        // only on relative weights, is strictly increasing, and sits
        // strictly inside (0, 1) without any clamp. Zero-mass tie groups
        // are not knots of the weighted empirical distribution and are
        // dropped.
        let mut cum_before = 0.0_f64;
        let mut pos = 0usize;
        while pos < order.len() {
            let zi = z[order[pos]];
            let mut w_group = 0.0_f64;
            let mut end = pos;
            while end < order.len() && z[order[end]] == zi {
                w_group += weights[order[end]];
                end += 1;
            }
            if w_group > 0.0 {
                sorted_z.push(zi);
                weighted_cdf.push((cum_before + 0.5 * w_group) / w_total);
                cum_before += w_group;
            }
            pos = end;
        }
        if sorted_z.is_empty() {
            return Err(
                "rank-INT calibration requires at least one positive-weight observation"
                    .to_string(),
            );
        }

        // Compute sanity-check post-mean and post-sd on the transformed
        // sample, weighted by the original weights.
        let mut sum_wz = 0.0_f64;
        let mut sum_w = 0.0_f64;
        for &idx in &order {
            let zi = z[idx];
            let calibrated = Self::apply_with_knots(zi, &sorted_z, &weighted_cdf);
            sum_wz += weights[idx] * calibrated;
            sum_w += weights[idx];
        }
        let post_mean = if sum_w > 0.0 { sum_wz / sum_w } else { 0.0 };
        let mut sum_w_dev = 0.0_f64;
        for &idx in &order {
            let zi = z[idx];
            let calibrated = Self::apply_with_knots(zi, &sorted_z, &weighted_cdf);
            let d = calibrated - post_mean;
            sum_w_dev += weights[idx] * d * d;
        }
        let post_sd = if sum_w > 0.0 {
            (sum_w_dev / sum_w).sqrt()
        } else {
            1.0
        };

        Ok(Self {
            sorted_z,
            weighted_cdf,
            post_mean,
            post_sd,
        })
    }

    /// Apply the calibration to the full training z vector, returning the
    /// calibrated sample. Equivalent to mapping each row's z through
    /// [`Self::apply_at_predict`], but vectorised.
    pub fn apply_to_training(&self, z: &Array1<f64>) -> Result<Array1<f64>, String> {
        if self.sorted_z.is_empty() {
            return Err("rank-INT calibration has no knots".to_string());
        }
        let mut out = Array1::<f64>::zeros(z.len());
        for (idx, &zi) in z.iter().enumerate() {
            if !zi.is_finite() {
                return Err(format!(
                    "rank-INT calibration apply: z[{idx}] = {zi} not finite"
                ));
            }
            out[idx] = self.apply_at_predict(zi);
        }
        Ok(out)
    }

    /// Apply the calibration to a single z at predict time.
    ///
    /// Linear interpolation on `(sorted_z, weighted_cdf)` to obtain
    /// `p ∈ [eps, 1 − eps]`, then `Φ⁻¹(p)` via
    /// [`standard_normal_quantile`]. Out-of-range z's clip to the
    /// boundary CDF before the quantile, so the calibration extrapolates
    /// monotonically beyond the training support.
    pub fn apply_at_predict(&self, z: f64) -> f64 {
        Self::apply_with_knots(z, &self.sorted_z, &self.weighted_cdf)
    }

    pub(crate) fn apply_with_knots(z: f64, sorted_z: &[f64], weighted_cdf: &[f64]) -> f64 {
        assert_eq!(sorted_z.len(), weighted_cdf.len());
        assert!(!sorted_z.is_empty());
        let n = sorted_z.len();
        let p = if z <= sorted_z[0] {
            weighted_cdf[0]
        } else if z >= sorted_z[n - 1] {
            weighted_cdf[n - 1]
        } else {
            // Binary search for the right knot.
            let mut lo = 0usize;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if sorted_z[mid] <= z {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let z_lo = sorted_z[lo];
            let z_hi = sorted_z[hi];
            let p_lo = weighted_cdf[lo];
            let p_hi = weighted_cdf[hi];
            if z_hi == z_lo {
                p_hi
            } else {
                let t = (z - z_lo) / (z_hi - z_lo);
                p_lo + t * (p_hi - p_lo)
            }
        };
        // Φ⁻¹(p); clip away from {0, 1} to keep the quantile finite.
        standard_normal_quantile(p).unwrap_or_else(|err| {
            let clipped = if p < 0.5 { -8.0 } else { 8.0 };
            log::debug!(
                "standard_normal_quantile({p}) failed ({err}); clipping the latent score to {clipped}"
            );
            clipped
        })
    }
}

/// Optional calibration applied to the latent score before the BMS
/// kernel runs. When `RankInverseNormal`, both the training and predict
/// paths route the input z through `LatentZRankIntCalibration::apply_*`
/// before the standard-normal closed-form kernel is invoked.
#[derive(Clone, Debug)]
pub enum LatentMeasureCalibration {
    None,
    RankInverseNormal(LatentZRankIntCalibration),
    ConditionalLocationScale(LatentZConditionalCalibration),
}

/// Conditional location-scale calibration of the latent score (#905).
///
/// The marginal-slope Auto trigger's pooled-z gate (KS / skewness / kurtosis +
/// the rank inverse-normal transform) only inspects the **marginal** law of
/// `z`. A conditional shift `E[z | C] = m(C) ≠ 0` — the allele-frequency-driven
/// grouping mean shift — passes the marginal gate while leaving `z | C`
/// off-center, so the slope contribution `b(C)·m(C)` leaks into the influence
/// channel `q`. Rank-INT provably cannot fix this: no transform `T` depending
/// only on the marginal `F_Z` can enforce `E[T(Z) | C] ≡ const` for all joint
/// laws.
///
/// The unique Fisher-orthogonal location-scale correction (for the Gaussian
/// working metric the closed-form probit kernel assumes) is
/// `ζ = (z − m(C)) / √v(C)`, where `m(C) = E[z|C]` and `v(C) = Var(z|C)` are
/// estimated by weighted ridge regression of `z` (and its squared residual) on
/// the marginal-index span `a(C) = [1 | X_marginal]`. The corrected `ζ` is
/// conditionally centered (and homoskedastic when the variance block is
/// active) by construction, so the `b(C)·m(C)` leakage vanishes. Matching the
/// first two conditional moments does **not** by itself make `ζ` standard
/// normal (a two-point residual law survives location-scale correction
/// unchanged in shape), so `build_latent_measure_with_geometry` re-checks
/// the calibrated sample against the standard-normal adequacy gate and
/// retains an empirical latent measure for the residual distribution when
/// that re-check fails; only a passing `ζ` uses the closed-form
/// standard-normal kernel. Persisted so prediction rebuilds `a(C)` from the
/// (reproducible) marginal design and applies the identical map to incoming
/// z.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatentZConditionalCalibration {
    /// Coefficients for the conditional mean `m(C) = β_m·[1 | a(C)]` over the
    /// basis `[1 | marginal-design row]`. Length `1 + basis_ncols` (leading
    /// entry is the intercept).
    pub mean_coeffs: Vec<f64>,
    /// Coefficients for the conditional variance
    /// `v(C) = max(β_v·[1 | a(C)], var_floor)`. Length `1 + basis_ncols`, or
    /// empty when the conditional-variance block of the Rao gate was not
    /// significant (mean-only correction); then `v(C) ≡ homoskedastic_var`.
    pub var_coeffs: Vec<f64>,
    /// Number of marginal-design columns in the basis (excludes the leading
    /// intercept). The predict-time marginal design must present exactly this
    /// many columns.
    pub basis_ncols: usize,
    /// Floor on the fitted conditional variance, in the (normalized)
    /// latent-score scale (= `AUTO_Z_CONDITIONAL_VAR_FLOOR_FRAC ·` the global
    /// weighted variance of the training score).
    pub var_floor: f64,
    /// The homoskedastic conditional variance `v(C) ≡ Var(z | C)`, used when the
    /// Breusch-Pagan stage did not fire and `v` is therefore constant in `C`.
    ///
    /// This is the *residual* variance of the conditional-mean regression,
    /// `Σ w (z − m̂(C))² / Σ w`, and NOT the global (marginal) variance of z —
    /// which is what it used to hold, and what its old on-disk name still says
    /// (gam#2768).
    ///
    /// The distinction is the whole correction. `ζ = (z − m(C))/√v` is supposed
    /// to be conditionally standard normal, because the marginal-slope identity
    /// that makes `q` the MARGINAL index,
    /// `E_ζ[Φ(q√(1+b²) + bζ)] = Φ(q)`, holds only at `Var(ζ|C) = 1`; at
    /// `Var(ζ|C) = v` it becomes `Φ(q√(1+b²)/√(1+b²v))`, a multiplicative
    /// distortion of every marginal coefficient. Dividing by the marginal
    /// variance guaranteed `v ≠ 1`: with z standardised,
    /// `1 = Var(m(C)) + E[Var(z|C)]`, so the residual variance is `1 − R²` and
    /// is strictly below the marginal variance *whenever the gate fires at all*.
    /// The bug was therefore not a corner case — it was on every fired gate, and
    /// it grew with exactly the conditional structure the correction exists to
    /// remove. At `R² = 0.25` it left `sd(ζ) = 0.87` against the `post_sd ≈ 1`
    /// this struct documents, distorted the marginal coefficients by ~4%, and
    /// (worse) made the calibrated residual fail the standard-normal adequacy
    /// re-check on the SD clause alone at any appreciable n — sending BMS to the
    /// empirical measure and, per gam#2718, withholding the covariance.
    ///
    /// Kept under the on-disk name `global_var` so a model saved before the fix
    /// still deserializes AND still applies the map it was fitted with: the
    /// stored number is whatever that fit divided by, and predict must reproduce
    /// the fit, not the current formula.
    #[serde(rename = "global_var")]
    pub homoskedastic_var: f64,
    /// Weighted mean of the calibrated training sample (sanity-check, ≈ 0).
    pub post_mean: f64,
    /// Weighted SD of the calibrated training sample (sanity-check, ≈ 1).
    pub post_sd: f64,
    /// Joint first-stage (generated-regressor) sandwich covariance of
    /// `θ₁ = (mean_coeffs, var_coeffs)`, shape `dim θ₁ × dim θ₁`.
    ///
    /// Replaced a stored PAIR of per-stage sandwiches that
    /// [`Self::theta1_covariance`] assembled block-diagonally -- i.e. that
    /// asserted `Cov(mean_coeffs, var_coeffs) = 0`. The stages are not
    /// independent: stage B regresses the squared MEAN residual on the same
    /// basis, so the stacked bread is block lower-triangular and the meat has a
    /// cross-block proportional to the residual's third moment. Both vanish
    /// under a Gaussian residual, and neither vanishes on the branch this
    /// covariance serves (gam#2484). Built by
    /// `stacked_first_stage_sandwich_cov`; the two retired fields were its
    /// diagonal blocks.
    ///
    /// Fit-time only: predict applies the map from `mean_coeffs`/`var_coeffs`
    /// and never reads their uncertainty. Verified rather than assumed -- the
    /// only production consumer of this matrix is the Murphy-Topel assembly in
    /// `block_specs.rs`, which runs at fit.
    ///
    /// `#[serde(default)]` so a model saved before gam#2484 -- carrying the two
    /// retired per-stage blocks and no joint -- still deserializes on the
    /// predict path. An empty default is NOT a silent zero: it is refused at the
    /// single point of consumption (see
    /// [`Self::generated_regressor_correction`]), because a covariance that
    /// quietly vanishes is exactly the failure this issue exists to prevent.
    #[serde(default)]
    pub theta1_cov: Array2<f64>,
}

impl LatentZConditionalCalibration {
    #[inline]
    pub(crate) fn affine(coeffs: &[f64], a_row: ArrayView1<'_, f64>) -> f64 {
        let mut acc = coeffs[0];
        for (c, &x) in coeffs[1..].iter().zip(a_row.iter()) {
            acc += c * x;
        }
        acc
    }

    pub(crate) fn conditional_mean(&self, a_row: ArrayView1<'_, f64>) -> f64 {
        Self::affine(&self.mean_coeffs, a_row)
    }

    pub(crate) fn conditional_var(&self, a_row: ArrayView1<'_, f64>) -> f64 {
        if self.var_coeffs.is_empty() {
            self.homoskedastic_var.max(self.var_floor)
        } else {
            Self::affine(&self.var_coeffs, a_row).max(self.var_floor)
        }
    }

    /// Apply `ζ = (z − m(C))/√v(C)` to a batch. `a_block` is the marginal
    /// design (`n × basis_ncols`); `z` is the (normalized) latent score. Used
    /// at both training and predict time, so the map is identical.
    pub fn apply(
        &self,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Result<Array1<f64>, String> {
        if a_block.ncols() != self.basis_ncols {
            return Err(format!(
                "conditional latent calibration expects {} basis columns, got {}",
                self.basis_ncols,
                a_block.ncols()
            ));
        }
        if a_block.nrows() != z.len() {
            return Err(format!(
                "conditional latent calibration row mismatch: z={}, basis rows={}",
                z.len(),
                a_block.nrows()
            ));
        }
        if self.mean_coeffs.len() != self.basis_ncols + 1 {
            return Err(format!(
                "conditional latent calibration mean coefficient length {} != basis_ncols+1 ({})",
                self.mean_coeffs.len(),
                self.basis_ncols + 1
            ));
        }
        let mut out = Array1::<f64>::zeros(z.len());
        for i in 0..z.len() {
            let a_row = a_block.row(i);
            if !z[i].is_finite() {
                return Err(format!(
                    "conditional latent calibration: z[{i}] = {} not finite",
                    z[i]
                ));
            }
            let m = self.conditional_mean(a_row);
            let v = self.conditional_var(a_row);
            if !(v.is_finite() && v > 0.0) {
                return Err(format!(
                    "conditional latent calibration produced non-positive variance {v} at row {i}"
                ));
            }
            let zeta = (z[i] - m) / v.sqrt();
            if !zeta.is_finite() {
                return Err(format!(
                    "conditional latent calibration produced non-finite zeta at row {i}"
                ));
            }
            out[i] = zeta;
        }
        Ok(out)
    }

    /// Dimension of the first-stage parameter vector `θ₁ = (mean_coeffs,
    /// var_coeffs)` whose estimation uncertainty the generated-regressor
    /// correction propagates. Equals `len(mean_coeffs)` when the variance block
    /// is inactive, otherwise `len(mean_coeffs) + len(var_coeffs)`.
    pub fn theta1_dim(&self) -> usize {
        self.mean_coeffs.len() + self.var_coeffs.len()
    }

    /// Per-row sensitivity `∂ζ_i/∂θ₁` of the calibrated score to the first-stage
    /// calibration coefficients, stacked as `[∂ζ/∂mean_coeffs | ∂ζ/∂var_coeffs]`
    /// (length [`Self::theta1_dim`]). With `ζ = (z − m(C))/√v(C)`,
    /// `A_i = [1 | a(C_i)]`, `m = A_iᵀ·mean_coeffs`, `v = A_iᵀ·var_coeffs`:
    ///
    ///   `∂ζ/∂m = −1/√v`,  `∂ζ/∂v = −(z − m)/(2 v^{3/2}) = −ζ/(2v)`,
    ///
    /// and by the chain rule through the affine basis
    /// `∂ζ/∂mean_coeffs = (∂ζ/∂m)·A_i`, `∂ζ/∂var_coeffs = (∂ζ/∂v)·A_i`. The
    /// variance block contributes only when `var_coeffs` is active AND the
    /// fitted `v(C_i)` is above the floor (a floored row has `∂v/∂var_coeffs = 0`
    /// in the applied map). `z` is the (normalized) raw latent score at this row.
    pub fn zeta_theta1_jacobian_row(&self, z: f64, a_row: ArrayView1<'_, f64>) -> Vec<f64> {
        let m = self.conditional_mean(a_row);
        let v = self.conditional_var(a_row);
        let inv_sqrt_v = 1.0 / v.sqrt();
        // Intercept-augmented basis row A_i = [1 | a(C_i)].
        let mut out = Vec::with_capacity(self.theta1_dim());
        let dzeta_dm = -inv_sqrt_v;
        out.push(dzeta_dm); // intercept column of A
        for &x in a_row.iter() {
            out.push(dzeta_dm * x);
        }
        if !self.var_coeffs.is_empty() {
            // ∂ζ/∂v active only off the floor; on the floor the applied v(C) is
            // constant in var_coeffs, so the variance sensitivity is exactly 0.
            let raw_v = Self::affine(&self.var_coeffs, a_row);
            let dzeta_dv = if raw_v > self.var_floor {
                let zeta = (z - m) * inv_sqrt_v;
                -zeta / (2.0 * v)
            } else {
                0.0
            };
            out.push(dzeta_dv);
            for &x in a_row.iter() {
                out.push(dzeta_dv * x);
            }
        }
        out
    }

    /// Joint first-stage covariance `V₁` of `θ₁ = (mean_coeffs, var_coeffs)`
    /// of `θ₁`, ordered to match [`Self::zeta_theta1_jacobian_row`]. The two
    /// stages are fit on (asymptotically) uncorrelated estimating equations
    /// (the mean score `Σ w û A` and the Breusch–Pagan variance score
    /// `Σ w (û² − v) A` are orthogonal under the Gaussian working model), so the
    /// joint first-stage covariance is block-diagonal to first order — the same
    /// approximation the Rao gate above uses.
    pub fn theta1_covariance(&self) -> Array2<f64> {
        self.theta1_cov.clone()
    }

    /// Murphy–Topel generated-regressor correction term for the second-stage
    /// slope covariance. Given the second-stage information `H_β` (the penalized
    /// joint Hessian of the slope fit, whose inverse is the naive `V_β`) and the
    /// cross-derivative `G = ∂(score_β)/∂θ₁` (`p_β × dim θ₁`), the corrected
    /// covariance is
    ///
    ///   `V_β = V_β^naive + (H_β⁻¹ G) V₁ (H_β⁻¹ G)ᵀ`.
    ///
    /// This returns the additive rank-`dim θ₁` term `(H_β⁻¹ G) V₁ (H_β⁻¹ G)ᵀ`
    /// given the already-formed `hbeta_inv_g = H_β⁻¹ G` (`p_β × dim θ₁`). The
    /// caller forms `G` by accumulating the per-row slope-score sensitivity to
    /// `ζ_i` times [`Self::zeta_theta1_jacobian_row`] (chain rule
    /// `∂score_β/∂θ₁ = Σ_i (∂score_β/∂ζ_i) (∂ζ_i/∂θ₁)`).
    pub fn generated_regressor_term(&self, hbeta_inv_g: ArrayView2<'_, f64>) -> Array2<f64> {
        let v1 = self.theta1_covariance();
        hbeta_inv_g.dot(&v1).dot(&hbeta_inv_g.t())
    }

    /// Assemble the full Murphy–Topel generated-regressor correction
    /// `(Vb·G)·V₁·(Vb·G)ᵀ` for the second-stage slope covariance, given the ONE
    /// engine-side quantity it cannot reconstruct post-fit: the per-row
    /// reduced-frame slope-score sensitivity to the calibrated score,
    /// `s_i = ∂score_β,i/∂ζ_i` (a `p_β`-vector in the joint flat-β reduced frame
    /// `solved_fit.beta_covariance()` lives in). With `score_β,i = ∂ℓ_i/∂β`,
    /// `s_i = ∂²ℓ_i/∂β∂ζ_i = J_iᵀ·(∂²ℓ_i/∂η_i∂ζ_i)` is the mixed `(β, ζ)`
    /// second derivative of the warped row kernel contracted through the slope
    /// design Jacobian `J_i` — exactly the #932 `RowProgram` z-jet
    /// channel (`z` is already a row-program input; one extra mixed `(β, z)` jet
    /// channel reads off `∂²ℓ/∂β∂z`). It must be evaluated at the converged `β̂`
    /// in the SAME reduced frame as `vb`.
    ///
    /// Everything else is built here from the stored first-stage quantities and
    /// the second-stage fit, dissolving the post-fit-reconstruction blocker:
    ///   - `G = Σ_i s_i · (∂ζ_i/∂θ₁)ᵀ` (`p_β × dim θ₁`), the chain-rule outer
    ///     product accumulated row-by-row with `∂ζ_i/∂θ₁ =
    ///     `[`Self::zeta_theta1_jacobian_row`]`(z_i, a_row_i)` (exact-zero on
    ///     floored rows, so floored rows contribute nothing — `G`'s support is
    ///     the gate-fired rows);
    ///   - `Vb·G = vb·G` since the naive second-stage covariance `vb` IS
    ///     `H_β⁻¹` (the coordinator's `H_β⁻¹ G = Vb.dot(G)`);
    ///   - the term `(Vb·G)·V₁·(Vb·G)ᵀ` via [`Self::generated_regressor_term`].
    ///
    /// `score_zeta_sensitivity` is `n × p_β` (row `i` = `s_i`); `z` is the
    /// per-row normalized latent score (`n`); `a_block` is the marginal design
    /// `n × basis_ncols` whose rows feed `zeta_theta1_jacobian_row`; `vb` is the
    /// naive reduced-frame slope covariance `n_β × n_β`. The returned term is
    /// PSD (a congruence of the PSD `V₁`), so adding it to `vb` makes the
    /// corrected slope SE strictly ≥ the naive SE whenever the gate fires
    /// (`G ≠ 0`) and exactly equal when every row is floored (`G = 0`).
    pub fn generated_regressor_correction(
        &self,
        score_zeta_sensitivity: ArrayView2<'_, f64>,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
        vb: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = score_zeta_sensitivity.nrows();
        let p_beta = score_zeta_sensitivity.ncols();
        if z.len() != n || a_block.nrows() != n {
            return Err(format!(
                "generated_regressor_correction row mismatch: score_zeta_sensitivity rows={n}, \
                 z={}, a_block rows={}",
                z.len(),
                a_block.nrows()
            ));
        }
        if a_block.ncols() != self.basis_ncols {
            return Err(format!(
                "generated_regressor_correction expects {} basis columns, got {}",
                self.basis_ncols,
                a_block.ncols()
            ));
        }
        if vb.nrows() != p_beta || vb.ncols() != p_beta {
            return Err(format!(
                "generated_regressor_correction: vb must be {p_beta}×{p_beta}, got {}×{}",
                vb.nrows(),
                vb.ncols()
            ));
        }
        // G = Σ_i s_i ⊗ (∂ζ_i/∂θ₁)  (p_β × dim θ₁). Each row contributes the
        // rank-1 outer product `s_i ⊗ J_zeta_i`, so summed over the n rows this
        // is exactly the cross product `G = Sᵀ·J` of the score-sensitivity
        // matrix `S` (`n × p_β`, supplied) and the per-row ζ-Jacobian matrix
        // `J` (`n × dim θ₁`). Forming `J` row-by-row is O(n·dim θ₁); the cross
        // product is then a single BLAS-3 GEMM rather than the O(n·p_β·dim θ₁)
        // scalar triple loop (≈1.5e9 FMA at biobank scale, n≈194k, the dominant
        // ~13s/disease cost of the SE correction). Floored rows yield an exact
        // all-zero `J` row, so they contribute zero to the GEMM — bit-identical
        // to skipping them, no approximation.
        // gam#2484: refuse an absent or ill-shaped first-stage covariance rather
        // than multiplying by it. This fires for a payload written before the
        // joint covariance existed (`theta1_cov` defaults to `0×0` there); such
        // a model cannot supply the term, and silently contributing nothing
        // would understate the interval by exactly the amount the correction
        // exists to add.
        let dim_theta1 = self.mean_coeffs.len() + self.var_coeffs.len();
        if self.theta1_cov.nrows() != dim_theta1 || self.theta1_cov.ncols() != dim_theta1 {
            return Err(format!(
                "generated_regressor_correction: the first-stage covariance is {}×{} but \
                 theta1 has dimension {dim_theta1}. A calibration deserialized from a payload \
                 written before gam#2484 carries no joint first-stage covariance; refit rather \
                 than publishing an uncorrected interval.",
                self.theta1_cov.nrows(),
                self.theta1_cov.ncols()
            ));
        }
        let j_mat = self.build_zeta_theta1_jacobian(z, a_block);
        let vb_g = self.beta_theta1_sensitivity(score_zeta_sensitivity, j_mat.view(), vb)?;
        Ok(self.generated_regressor_term(vb_g.view()))
    }

    /// Per-row ζ-Jacobian matrix `J` (`n × dim θ₁`, row `i` = `∂ζ_i/∂θ₁`) built
    /// row-by-row from [`Self::zeta_theta1_jacobian_row`]. Floored rows yield an
    /// exact all-zero row, so they contribute nothing to the `G = Sᵀ·J` cross
    /// product (bit-identical to skipping them).
    fn build_zeta_theta1_jacobian(
        &self,
        z: ArrayView1<'_, f64>,
        a_block: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let n = a_block.nrows();
        let dim_theta1 = self.theta1_dim();
        let mut j_mat = Array2::<f64>::zeros((n, dim_theta1));
        for i in 0..n {
            let j_zeta_row = self.zeta_theta1_jacobian_row(z[i], a_block.row(i));
            assert_eq!(
                j_zeta_row.len(),
                dim_theta1,
                "J_zeta row width must match the first-stage hyperparameter dimension"
            );
            let mut dst = j_mat.row_mut(i);
            for (slot, jz) in dst.iter_mut().zip(j_zeta_row.into_iter()) {
                *slot = jz;
            }
        }
        j_mat
    }

    /// Signed first-order sensitivity `∂β̂/∂θ₁ = Vb·G` (`p_β × dim θ₁`) of the
    /// converged second-stage slope to the first-stage calibration parameters,
    /// the SIGNED quantity the Murphy–Topel correction is built from.
    ///
    /// `G = Sᵀ·J = Σ_i s_i ⊗ (∂ζ_i/∂θ₁)` with `s_i = ∂score_β,i/∂ζ_i` the
    /// LOG-LIKELIHOOD-score sensitivity (the sign convention #1131 fixes at the
    /// source in [`gradient_paths::rigid_standard_normal_mixed_z_sensitivity`]),
    /// and `Vb = H_β⁻¹` the NLL-Hessian inverse. Under this convention the
    /// implicit-function theorem on `∂(log L)/∂β = 0` gives
    /// `∂β̂/∂θ₁ = +H_β⁻¹·G = +Vb·G`, so the returned matrix matches the finite
    /// difference of the refit slope in θ₁ in BOTH sign and magnitude — unlike
    /// the PSD correction term [`Self::generated_regressor_correction`], which is
    /// invariant to this sign. `j_zeta` is the per-row ζ-Jacobian matrix
    /// (`n × dim θ₁`, row `i` = `∂ζ_i/∂θ₁`).
    fn beta_theta1_sensitivity(
        &self,
        score_zeta_sensitivity: ArrayView2<'_, f64>,
        j_zeta: ArrayView2<'_, f64>,
        vb: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // G = Sᵀ·J (p_β × dim θ₁) via the SIMD/GPU-routed cross product.
        let g = gam_linalg::faer_ndarray::fast_atb(&score_zeta_sensitivity, &j_zeta);
        // Vb·G = H_β⁻¹·G (vb is the naive reduced-frame covariance the fit
        // already produced — reused, never recomputed).
        Ok(vb.dot(&g))
    }
}

/// First-stage robust (HC0) sandwich covariance of a weighted-ridge coefficient
/// vector: `V₁ = M⁺ (Σ_i w_i² û_i² A_i A_iᵀ) M⁺` with `M = AᵀWA + λR` the
/// ridge normal matrix that produced the coefficients, `W = diag(weights)`,
/// `û_i` the per-row residual, and `A` the regression basis (here `[1 | a(C)]`).
/// `M⁺` is the Moore–Penrose pseudo-inverse via eigendecomposition with a
/// relative tolerance: identifiable directions get the usual `(λ_eff)⁻¹` weight,
/// and rank-deficient directions (where some θ₁ components are not identified
/// by `A`) are zeroed — they carry no asymptotic distribution, so V₁ in those
/// directions is zero, and the Murphy–Topel propagation through identifiable
/// functionals of β remains finite and consistent. Using the ordinary inverse
/// here let the unregularized direction's `1/ε` blow `M⁻¹·meat·M⁻¹` through
/// the f64 range whenever the wide marginal-index span had a near-null
/// direction (the bug behind "conditional latent calibration sandwich
/// covariance is non-finite" on wide rank-deficient duchon/spline conditioning).
/// The meat is formed as `BᵀB` with `B_i = w_i û_i A_iᵀ` (signed) so the
/// fused-multiply GEMM is the same SIMD path used everywhere else in the
/// codebase, instead of a hand-rolled triple loop whose partial sums could
/// overflow on a single pathological row of the basis.
/// `M⁺` for a weighted-ridge normal matrix, in RAW coordinates, computed on the
/// Jacobi-preconditioned matrix for the conditioning reasons spelled out in
/// [`weighted_ridge_sandwich_cov`].
///
/// `M̃ = S·M·S` with `S = diag(1/√M_jj)`, so `M⁻¹ = S·M̃⁻¹·S`. Returned rather
/// than folded into a sandwich because the stacked system needs the SAME `M⁺`
/// in three different products.
///
/// `weighted_ridge_sandwich_cov` deliberately keeps its own inlined copy of this
/// arithmetic: extracting it there would regroup its floating-point operations
/// and move numbers on a currently-passing path for no benefit.
pub(crate) fn preconditioned_normal_pseudoinverse(
    normal_matrix: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let p = normal_matrix.nrows();
    if normal_matrix.ncols() != p {
        return Err(format!(
            "stacked first-stage sandwich needs a square normal matrix, got {}x{}",
            normal_matrix.nrows(),
            normal_matrix.ncols()
        ));
    }
    let mut m_sym = normal_matrix.clone();
    gam_linalg::matrix::symmetrize_in_place(&mut m_sym);
    let scale: Vec<f64> = (0..p)
        .map(|j| 1.0 / m_sym[[j, j]].max(f64::MIN_POSITIVE).sqrt())
        .collect();
    let mut m_scaled = m_sym;
    for i in 0..p {
        for j in 0..p {
            m_scaled[[i, j]] *= scale[i] * scale[j];
        }
    }
    let mut pinv = gam_linalg::utils::rank_certified_psd_pseudoinverse(&m_scaled, 1.0e-10)
        .map_err(|e| format!("stacked first-stage sandwich pseudo-inverse failed: {e}"))?
        .into_pseudoinverse();
    for i in 0..p {
        for j in 0..p {
            pinv[[i, j]] *= scale[i] * scale[j];
        }
    }
    Ok(pinv)
}

/// Weighted Gram `Σ_i s_i · A_i A_iᵀ` for SIGNED per-row scalars `s`.
///
/// Not expressible as `BᵀB` (that forces a non-negative weight), so this is the
/// explicit `Aᵀ diag(s) A`.
fn signed_weighted_gram(basis: ArrayView2<'_, f64>, s: &[f64]) -> Array2<f64> {
    let mut scaled = basis.to_owned();
    for (mut row, &value) in scaled.rows_mut().into_iter().zip(s.iter()) {
        row.iter_mut().for_each(|entry| *entry *= value);
    }
    basis.t().dot(&scaled)
}

/// The joint first-stage covariance `V₁` of `θ₁ = (mean_coeffs, var_coeffs)`,
/// as the sandwich of the STACKED estimating system (gam#2484).
///
/// With `A = [1 | a(C)]`, `W = diag(w)`, `M = AᵀWA + λR`,
/// `û_i = z_i − A_iᵀβ_m` and `r_i = û_i² − A_iᵀβ_v`:
///
/// ```text
/// ψ^m_i = w_i A_i û_i          ψ^v_i = w_i A_i r_i
///
/// B = [ M      0 ]   with  M_vm = ∂ψ^v/∂β_m = −2·Σ_i w_i û_i A_i A_iᵀ
///     [ M_vm   M ]
///
/// Ω = Σ_i w_i² [A_i û_i ; A_i r_i][·]ᵀ      V₁ = B⁻¹ Ω B⁻ᵀ
/// ```
///
/// The previous form kept only `blkdiag(M⁻¹Ω_mm M⁻¹, M⁻¹Ω_vv M⁻¹)`, which is
/// this expression with `M_vm` and `Ω_mv` set to zero. Both are zero in
/// expectation for a Gaussian residual (`E[û|A] = 0` and `E[û³] = 0`) and
/// neither is zero on the branch this covariance serves.
///
/// `B⁻¹ = [[M⁻¹, 0], [K, M⁻¹]]` with `K = −M⁻¹·M_vm·M⁻¹`, so no `2p × 2p`
/// factorization is formed: one `M⁺` is reused. `V₁` stays PSD -- it is a
/// congruence of the PSD Gram `Ω` -- so the caller's PSD guarantee on the
/// generated-regressor term is untouched.
pub(crate) fn stacked_first_stage_sandwich_cov(
    basis: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    mean_residuals: &[f64],
    var_residuals: &[f64],
    normal_matrix: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let n = basis.nrows();
    let p = basis.ncols();
    if mean_residuals.len() != n || var_residuals.len() != n || weights.len() != n {
        return Err(format!(
            "stacked first-stage sandwich length mismatch: rows={n}, mean_residuals={}, \
             var_residuals={}, weights={}",
            mean_residuals.len(),
            var_residuals.len(),
            weights.len()
        ));
    }
    let m_pinv = preconditioned_normal_pseudoinverse(normal_matrix)?;
    if m_pinv.nrows() != p {
        return Err(format!(
            "stacked first-stage sandwich normal-matrix shape mismatch: basis cols={p}, normal {}",
            m_pinv.nrows()
        ));
    }

    // Meat blocks as Grams of the score-scaled basis.
    let scaled = |values: &[f64]| -> Array2<f64> {
        let mut b = basis.to_owned();
        for (i, mut row) in b.rows_mut().into_iter().enumerate() {
            let scale = weights[i] * values[i];
            if scale == 0.0 {
                row.fill(0.0);
            } else {
                row.iter_mut().for_each(|entry| *entry *= scale);
            }
        }
        b
    };
    let bm = scaled(mean_residuals);
    let bv = scaled(var_residuals);
    let omega_mm = gam_linalg::faer_ndarray::fast_ata(&bm);
    let omega_vv = gam_linalg::faer_ndarray::fast_ata(&bv);
    let omega_mv = bm.t().dot(&bv);

    // Bread cross-block and the resulting inverse-bread row.
    let wu: Vec<f64> = (0..n).map(|i| weights[i] * mean_residuals[i]).collect();
    let m_vm = signed_weighted_gram(basis, &wu).mapv(|value| -2.0 * value);
    let k = m_pinv.dot(&m_vm).dot(&m_pinv).mapv(|value| -value);

    let v11 = m_pinv.dot(&omega_mm).dot(&m_pinv);
    let v12 = m_pinv.dot(&omega_mm).dot(&k.t()) + m_pinv.dot(&omega_mv).dot(&m_pinv);
    let v22 = k.dot(&omega_mm).dot(&k.t())
        + k.dot(&omega_mv).dot(&m_pinv)
        + m_pinv.dot(&omega_mv.t()).dot(&k.t())
        + m_pinv.dot(&omega_vv).dot(&m_pinv);

    let mut v1 = Array2::<f64>::zeros((2 * p, 2 * p));
    v1.slice_mut(s![..p, ..p]).assign(&v11);
    v1.slice_mut(s![..p, p..]).assign(&v12);
    v1.slice_mut(s![p.., ..p]).assign(&v12.t());
    v1.slice_mut(s![p.., p..]).assign(&v22);
    gam_linalg::matrix::symmetrize_in_place(&mut v1);
    if v1.iter().any(|value| !value.is_finite()) {
        return Err("stacked first-stage sandwich covariance is non-finite".to_string());
    }
    Ok(v1)
}

pub(crate) fn weighted_ridge_sandwich_cov(
    basis: ArrayView2<'_, f64>,
    residuals: &[f64],
    weights: ArrayView1<'_, f64>,
    normal_matrix: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let n = basis.nrows();
    let p = basis.ncols();
    if residuals.len() != n || weights.len() != n {
        return Err(format!(
            "weighted ridge sandwich length mismatch: rows={n}, residuals={}, weights={}",
            residuals.len(),
            weights.len()
        ));
    }
    if normal_matrix.nrows() != p || normal_matrix.ncols() != p {
        return Err(format!(
            "weighted ridge sandwich normal-matrix shape mismatch: basis cols={p}, normal {}x{}",
            normal_matrix.nrows(),
            normal_matrix.ncols()
        ));
    }
    // Robust HC0 meat as a Gram: build `B` with `B_i = (w_i û_i) A_iᵀ` (rows of
    // basis scaled by `w_i û_i`, sign carried), so `meat = BᵀB = Σ_i w_i² û_i²
    // A_i A_iᵀ` from one BLAS Gramian. Identical math to the per-row outer-
    // product accumulation, but the GEMM path keeps partial sums vectorized
    // and is less sensitive to a single pathological row producing an
    // intermediate that overflows f64 before the column-wise reduction cancels.
    let mut b = basis.to_owned();
    for i in 0..n {
        let wi = weights[i];
        let ri = residuals[i];
        let scale = wi * ri;
        if scale == 0.0 {
            b.row_mut(i).fill(0.0);
            continue;
        }
        b.row_mut(i).iter_mut().for_each(|value| *value *= scale);
    }
    let meat = gam_linalg::faer_ndarray::fast_ata(&b);
    // SPD pseudo-inverse of `M = AᵀWA + λR` via eigendecomposition with a
    // relative tolerance; symmetrize first to absorb floating-point asymmetry
    // accumulated in the AᵀWA assembly.
    let mut m_sym = normal_matrix.clone();
    gam_linalg::matrix::symmetrize_in_place(&mut m_sym);
    // Jacobi (symmetric diagonal) preconditioning. When the conditioning basis
    // spans many orders of magnitude — a power-9 Duchon RBF over 16 standardized
    // PCs produces columns differing by ~30 decades — `M` and `meat` live on
    // wildly different per-column scales, and the eigendecomposition behind
    // `M⁺ meat M⁺` loses all accuracy: the relative truncation tolerance is set
    // by `λ_max(M)` (dominated by the largest-scale column), so a genuinely
    // identified small-scale direction can be dropped while a near-null one is
    // kept, and the surviving `1/λ` then multiplies the huge `meat` straight
    // through the f64 range. Precondition by `D = diag(√M_jj)`. Because the ridge
    // penalty diagonal is built as the weighted Gram diagonal itself
    // (`penalty_jj = Σ_i w_i a_ij² = (AᵀWA)_jj`), `M_jj = (1+ρ)(AᵀWA)_jj`, so
    // `M̃ = D⁻¹ M D⁻¹` has EXACT unit diagonal and `M̃ = C + (ρ/(1+ρ))·I` with
    // `C` the basis correlation matrix (PSD). Hence `λ_min(M̃) ≥ ρ/(1+ρ) ≈ 1e-8`
    // even for a fully collinear basis, which clears the pseudo-inverse's
    // relative tolerance `≈ 1e-10·λ_max(M̃)` for the conditioning widths that
    // occur here: no direction is spuriously dropped, so `M̃⁺ = M̃⁻¹ = D M⁻¹ D`
    // and `cov = D⁻¹ (M̃⁻¹ meat̃ M̃⁻¹) D⁻¹ = M⁻¹ meat M⁻¹` EXACTLY — the scaling
    // cancels, this is the same sandwich, only computed on a well-conditioned
    // matrix. (Should a pure-ridge direction ever fall under tolerance at very
    // large width, dropping it is the correct scale-invariant identifiability
    // call.) `meat̃ = D⁻¹ meat D⁻¹`; `M_jj > 0` (Gram diagonal floored positive)
    // so `D` is always finite and invertible.
    let scale: Vec<f64> = (0..p)
        .map(|j| 1.0 / m_sym[[j, j]].max(f64::MIN_POSITIVE).sqrt())
        .collect();
    let mut m_scaled = m_sym;
    let mut meat_scaled = meat;
    for i in 0..p {
        for j in 0..p {
            let s = scale[i] * scale[j];
            m_scaled[[i, j]] *= s;
            meat_scaled[[i, j]] *= s;
        }
    }
    let m_pinv = gam_linalg::utils::rank_certified_psd_pseudoinverse(&m_scaled, 1.0e-10)
        .map_err(|e| format!("conditional latent calibration sandwich pseudo-inverse failed: {e}"))?
        .into_pseudoinverse();
    let mut cov = m_pinv.dot(&meat_scaled).dot(&m_pinv);
    // Undo the symmetric scaling: cov_raw = D⁻¹ cov_scaled D⁻¹.
    for i in 0..p {
        for j in 0..p {
            cov[[i, j]] *= scale[i] * scale[j];
        }
    }
    if cov.iter().any(|v| !v.is_finite()) {
        return Err("conditional latent calibration sandwich covariance is non-finite".to_string());
    }
    Ok(cov)
}

/// Weighted mean of a slice of values.
pub(crate) fn weighted_mean(
    values: &[f64],
    weights: ArrayView1<'_, f64>,
    total_weight: f64,
) -> f64 {
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * v)
        .sum::<f64>()
        / total_weight
}

/// Robust (heteroskedasticity-consistent) Rao/LM score-test p-value for the
/// null that the centered basis columns `ã(C)` carry no information about the
/// centered response `u`. This is the LAN locally-optimal statistic the issue
/// names: `s = Σ_i w_i u_i ã(C_i)`, `Ω̂ = Σ_i w_i² u_i² ã(C_i)ã(C_i)ᵀ`,
/// `D = sᵀ Ω̂⁺ s ⟶ χ²_{rank Ω̂}`. Both the conditional-mean test
/// (`u_i = z_i − z̄`) and the conditional-variance / Breusch-Pagan test
/// (`u_i = (z_i − z̄)² − σ̂²`) are this statistic with the same centered basis.
///
/// Returns `None` when the test is degenerate (no usable basis directions),
/// otherwise the asymptotic p-value.
pub(crate) fn robust_conditional_score_pvalue(
    a_centered: ArrayView2<'_, f64>,
    u: &[f64],
    weights: ArrayView1<'_, f64>,
) -> Result<Option<f64>, String> {
    let n = a_centered.nrows();
    let r = a_centered.ncols();
    if r == 0 || n == 0 {
        return Ok(None);
    }
    if u.len() != n || weights.len() != n {
        return Err(format!(
            "conditional score test length mismatch: rows={n}, u={}, weights={}",
            u.len(),
            weights.len()
        ));
    }
    // Build the per-row scaled basis `B` with `B_i = (w_i u_i) ã_i` once, then
    // recover both the score and the HC0 robust meat from it with two BLAS-3
    // GEMMs over chunked row-blocks instead of an `O(n · r²)` per-row scatter:
    //   • score  `s   = ãᵀ (w ∘ u) = Bᵀ 1`     (column sums of `B`),
    //   • meat   `Ω̂  = Σ_i w_i² u_i² ã_i ã_iᵀ = BᵀB` since `(w_i u_i)² = w_i² u_i²`.
    // A non-positive weight zeroes that row of `B` (its score and meat
    // contributions both vanish), reproducing the `wi <= 0.0` skip EXACTLY.
    // `fast_ata` is the same parallel Gramian the second-stage sandwich uses, so
    // the statistic is numerically identical to the row-accumulated form up to
    // the deterministic GEMM reduction order.
    let mut b = a_centered.to_owned();
    for i in 0..n {
        let wi = weights[i];
        let scale = if wi > 0.0 { wi * u[i] } else { 0.0 };
        if scale == 0.0 {
            b.row_mut(i).fill(0.0);
            continue;
        }
        b.row_mut(i).iter_mut().for_each(|value| *value *= scale);
    }
    let s = b.sum_axis(ndarray::Axis(0));
    let omega = gam_linalg::faer_ndarray::fast_ata(&b);
    if !s.iter().all(|v| v.is_finite()) || !omega.iter().all(|v| v.is_finite()) {
        return Ok(None);
    }
    let omega_geometry = gam_linalg::utils::rank_certified_psd_pseudoinverse(&omega, 1.0e-10)
        .map_err(|e| format!("conditional score test pseudo-inverse failed: {e}"))?;
    let rank = omega_geometry.rank();
    let omega_pinv = omega_geometry.into_pseudoinverse();
    if rank == 0 {
        return Ok(None);
    }
    let d_stat = s.dot(&omega_pinv.dot(&s));
    if !(d_stat.is_finite() && d_stat >= 0.0) {
        return Ok(None);
    }
    // The shared survival primitive owns both the direct upper-gamma identity
    // and its exact `Q(a, 0) = 1` boundary.
    let p_value = chi_square_sf(d_stat, rank as f64);
    Ok(p_value.is_finite().then_some(p_value))
}

/// Fit the conditional location-scale calibration (#905) if the conditional
/// `E[z|C]`/`Var(z|C)` Rao gate fires on the marginal-index basis `a_block`.
///
/// Returns `None` when there is no conditional structure to correct (the gate
/// does not fire, or the basis is degenerate) — in that case the caller falls
/// back to the existing pooled-marginal gate (rank-INT or no calibration).
pub(crate) fn fit_conditional_latent_calibration_if_needed(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    a_block: ArrayView2<'_, f64>,
) -> Result<Option<LatentZConditionalCalibration>, String> {
    let n = z.len();
    let p = a_block.ncols();
    if n != weights.len() {
        return Err(format!(
            "conditional latent gate length mismatch: z={n}, weights={}",
            weights.len()
        ));
    }
    if a_block.nrows() != n {
        return Err(format!(
            "conditional latent gate row mismatch: z={n}, basis rows={}",
            a_block.nrows()
        ));
    }
    if p == 0 {
        return Ok(None);
    }
    let total_weight = weights.iter().copied().sum::<f64>();
    if !(total_weight.is_finite() && total_weight > 0.0) {
        return Ok(None);
    }
    if z.iter().any(|v| !v.is_finite()) || a_block.iter().any(|v| !v.is_finite()) {
        return Ok(None);
    }

    let z_mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / total_weight;
    let global_var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - z_mean) * (zi - z_mean))
        .sum::<f64>()
        / total_weight;
    if !(global_var.is_finite() && global_var > 0.0) {
        return Ok(None);
    }

    // Center each basis column by its weighted mean so the score test is about
    // conditional structure *beyond* the global level (the intercept nuisance).
    // A constant marginal-design column collapses to ~0 and is dropped by the
    // pseudo-inverse rank, so an intercept already present in a(C) is harmless.
    let mut a_centered = a_block.to_owned();
    for j in 0..p {
        let col = a_block.column(j);
        let col_mean = col
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| w * v)
            .sum::<f64>()
            / total_weight;
        a_centered.column_mut(j).mapv_inplace(|v| v - col_mean);
    }

    // Conditional-mean Rao test: u = z − z̄.
    let u_mean: Vec<f64> = z.iter().map(|&zi| zi - z_mean).collect();
    let p_mean = robust_conditional_score_pvalue(a_centered.view(), &u_mean, weights.view())?;
    // Conditional-variance (Breusch-Pagan) Rao test: u = (z − z̄)² − σ̂².
    let u_var: Vec<f64> = u_mean.iter().map(|&e| e * e - global_var).collect();
    let p_var = robust_conditional_score_pvalue(a_centered.view(), &u_var, weights.view())?;

    let mean_fires = p_mean.is_some_and(|p| p < AUTO_Z_CONDITIONAL_RAO_ALPHA);
    let var_fires = p_var.is_some_and(|p| p < AUTO_Z_CONDITIONAL_RAO_ALPHA);
    if !mean_fires && !var_fires {
        return Ok(None);
    }

    // Escalation fires. Fit the conditional mean over the full basis
    // [1 | a(C)] via a weighted ridge (the ridge stabilizes a rank-deficient
    // marginal-index span; it does not meaningfully shrink the few directions
    // that triggered the gate). The conditional-mean correction is applied
    // whenever the gate fires (a pure-variance trigger leaves the C-slopes of
    // m(C) ≈ 0, so it reduces to harmless global centering).
    let basis = build_intercept_basis(a_block);
    // Per-column Tikhonov penalty scaled by the weighted Gram diagonal, so the
    // ridge is *relative* to each column's scale (a 1e-8 absolute ridge would
    // be negligible against an O(n) Gram and would not stabilize a
    // rank-deficient penalized-spline marginal index). `diag_jj = Σ_i w_i a_ij²`;
    // floored positive so the all-zero (already-dropped) directions still
    // receive a finite ridge and the factorization cannot fail.
    let mut penalty = Array2::<f64>::zeros((basis.ncols(), basis.ncols()));
    for j in 0..basis.ncols() {
        let diag_jj = basis
            .column(j)
            .iter()
            .zip(weights.iter())
            .map(|(&x, &w)| w * x * x)
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        penalty[[j, j]] = diag_jj;
    }
    let z_col = z.view().insert_axis(ndarray::Axis(1));
    let (mean_coeffs_mat, mean_fitted) = gam_linalg::utils::gaussian_weighted_ridge(
        basis.view(),
        z_col,
        penalty.view(),
        weights.view(),
        AUTO_Z_CONDITIONAL_RIDGE_REL,
    )?;
    let mean_coeffs: Vec<f64> = mean_coeffs_mat.column(0).to_vec();

    // First-stage (generated-regressor) normal matrix `M = AᵀWA + λR`, the same
    // weighted-ridge system `gaussian_weighted_ridge` factorizes internally;
    // rebuilt here so its inverse can form the closed-form coefficient sandwich
    // `V₁` that the second-stage Murphy–Topel correction consumes. `p` is the
    // marginal-index width (small), so this is a cheap dense `(p+1)²` form.
    let normal_matrix = {
        let mut wa = basis.to_owned();
        for i in 0..wa.nrows() {
            let wi = weights[i];
            wa.row_mut(i).iter_mut().for_each(|value| *value *= wi);
        }
        let mut m = basis.t().dot(&wa);
        m += &(penalty.to_owned() * AUTO_Z_CONDITIONAL_RIDGE_REL);
        m
    };
    let mean_residuals: Vec<f64> = z
        .iter()
        .zip(mean_fitted.column(0).iter())
        .map(|(&zi, &mi)| zi - mi)
        .collect();
    let mean_cov = weighted_ridge_sandwich_cov(
        basis.view(),
        &mean_residuals,
        weights.view(),
        &normal_matrix,
    )?;

    let var_floor = (AUTO_Z_CONDITIONAL_VAR_FLOOR_FRAC * global_var).max(f64::MIN_POSITIVE);
    // gam#2484: the per-stage variance sandwich is gone -- it was one diagonal
    // block of a matrix that is now built jointly, and computing it here only to
    // discard it would be dead work. Its Breusch-Pagan residual is kept, because
    // the stacked meat's cross-block needs it.
    let mut var_residuals: Option<Vec<f64>> = None;
    let var_coeffs: Vec<f64> = if var_fires {
        // Conditional-variance correction: regress the squared mean-residual on
        // the same basis. Fitted values are floored at `var_floor` when applied.
        let resid_sq: Array1<f64> = mean_residuals.iter().map(|&e| e * e).collect();
        let resid_col = resid_sq.view().insert_axis(ndarray::Axis(1));
        let (var_coeffs_mat, var_fitted) = gam_linalg::utils::gaussian_weighted_ridge(
            basis.view(),
            resid_col,
            penalty.view(),
            weights.view(),
            AUTO_Z_CONDITIONAL_RIDGE_REL,
        )?;
        // `r_i = (z−m̂)²_i − v̂_i`, the Breusch-Pagan residual of stage B.
        var_residuals = Some(
            resid_sq
                .iter()
                .zip(var_fitted.column(0).iter())
                .map(|(&si, &vi)| si - vi)
                .collect(),
        );
        var_coeffs_mat.column(0).to_vec()
    } else {
        Vec::new()
    };

    // gam#2484: the joint sandwich when the variance stage fired, otherwise the
    // mean-only sandwich -- with no variance coefficients there is no second
    // block and no cross-term to get wrong.
    let theta1_cov = match var_residuals.as_ref() {
        Some(residuals) => stacked_first_stage_sandwich_cov(
            basis.view(),
            weights.view(),
            &mean_residuals,
            residuals,
            &normal_matrix,
        )?,
        None => mean_cov,
    };
    // gam#2768: the homoskedastic branch's `v(C)` is the RESIDUAL variance of
    // the conditional-mean regression, not the marginal variance of z. See the
    // field doc for why the difference is the correction rather than a detail:
    // with z standardised, `1 = Var(m(C)) + E[Var(z|C)]`, so the marginal
    // variance overstates `Var(z|C)` by exactly the structure the gate just
    // detected, and dividing by it leaves `ζ` at `sd = √(1−R²)`.
    //
    // Its own estimation uncertainty is not propagated into `theta1_cov`, for
    // the same reason the marginal variance's never was: the second-stage
    // Murphy-Topel correction treats a CONSTANT scale as known, and a plug-in
    // variance's contribution is one order down in `n` from the mean
    // coefficients' (`θ₁` carries the mean block, whose sensitivity
    // `∂ζ/∂m = −1/√v` is O(1)). When the Breusch-Pagan stage fires, `v(C)` is a
    // fitted function of the basis and IS carried in `θ₁`.
    let homoskedastic_var = {
        let residual_sum = mean_residuals
            .iter()
            .zip(weights.iter())
            .map(|(&e, &w)| w * e * e)
            .sum::<f64>();
        (residual_sum / total_weight).max(var_floor)
    };
    let mut calibration = LatentZConditionalCalibration {
        mean_coeffs,
        var_coeffs,
        basis_ncols: p,
        var_floor,
        homoskedastic_var,
        post_mean: 0.0,
        post_sd: 1.0,
        theta1_cov,
    };

    // Sanity-check post-correction moments on the training sample.
    let calibrated = calibration.apply(z.view(), a_block)?;
    let post_mean = weighted_mean(
        calibrated
            .as_slice()
            .expect("calibration.apply returns an owned standard-layout 1-D array"),
        weights.view(),
        total_weight,
    );
    let post_var = calibrated
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - post_mean) * (zi - post_mean))
        .sum::<f64>()
        / total_weight;
    calibration.post_mean = post_mean;
    calibration.post_sd = post_var.max(0.0).sqrt();

    Ok(Some(calibration))
}

/// Prepend a column of ones to `a_block`, producing the `[1 | a(C)]` regression
/// basis used by the conditional location-scale fit.
pub(crate) fn build_intercept_basis(a_block: ArrayView2<'_, f64>) -> Array2<f64> {
    let n = a_block.nrows();
    let p = a_block.ncols();
    let mut basis = Array2::<f64>::ones((n, p + 1));
    basis.slice_mut(s![.., 1..]).assign(&a_block);
    basis
}

/// Which latent measures the *calling family's row kernel* can actually
/// evaluate.
///
/// This is the only thing that differs between the two marginal-slope families'
/// latent-measure decisions, so it is the only argument
/// [`build_latent_measure_decision`] takes to serve both. The Bernoulli kernel
/// owns an empirical-grid branch (`empirical_rigid_primary_grad_hess_closed_form`
/// and its higher-order siblings, driven by a per-row intercept Newton solve);
/// the survival marginal-slope kernel does not — its row program is the
/// closed-form standard-normal probit lowering and nothing else. A decision that
/// handed the survival family a `GlobalEmpirical` measure would be a measure it
/// cannot evaluate, so the two families must reach *different* terminal states
/// from the *same* gate. Making the capability an argument is what keeps the
/// gate itself a single object (gam#2768).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EmpiricalLatentMeasureSupport {
    /// The caller can evaluate `LatentMeasureKind::GlobalEmpirical`.
    Available,
    /// The caller can only evaluate `LatentMeasureKind::StandardNormal`.
    StandardNormalOnly,
}

/// The latent-measure gate's verdict: the measure the kernel will integrate
/// against, the pre-transform applied to z before it reaches that kernel, and —
/// for a [`EmpiricalLatentMeasureSupport::StandardNormalOnly`] caller — the
/// adequacy ledger of the sample the standard-normal kernel is actually being
/// handed when that sample failed the gate.
pub(crate) struct LatentMeasureDecision {
    pub(crate) kind: LatentMeasureKind,
    pub(crate) calibration: LatentMeasureCalibration,
    /// `Some` only for a `StandardNormalOnly` caller, and only when the sample
    /// the kernel sees failed the standard-normal adequacy gate with no
    /// empirical measure available to carry the residual law. Never a silent
    /// state: the caller must route it through its own [`LatentZCheckMode`].
    pub(crate) unmodelled_residual: Option<LatentNormalAdequacy>,
    /// `Some` exactly when `kind` is a freshly built `GlobalEmpirical`: the
    /// record of the equal-mass compression that produced it, which is what
    /// makes the measure differentiable in the sample it was built from.
    ///
    /// Fit-time only and deliberately not part of `kind`: the measure is on the
    /// persistence wire and its identity is its nodes and weights, while this is
    /// provenance about the rows behind them. The gam#2484 Murphy–Topel
    /// correction is its only consumer.
    pub(crate) empirical_build: Option<empirical_measure_sensitivity::EmpiricalZGridBuild>,
}

impl LatentMeasureDecision {
    fn standard_normal(calibration: LatentMeasureCalibration) -> Self {
        Self {
            kind: LatentMeasureKind::StandardNormal,
            calibration,
            unmodelled_residual: None,
            empirical_build: None,
        }
    }
}

pub(crate) fn build_latent_measure_with_geometry(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
    conditioning: Option<ArrayView2<'_, f64>>,
) -> Result<
    (
        LatentMeasureKind,
        LatentMeasureCalibration,
        Option<empirical_measure_sensitivity::EmpiricalZGridBuild>,
    ),
    String,
> {
    let decision = build_latent_measure_decision(
        z,
        weights,
        policy,
        conditioning,
        EmpiricalLatentMeasureSupport::Available,
        "BMS",
    )?;
    if decision.unmodelled_residual.is_some() {
        // Structurally unreachable: an `Available` caller always has an
        // empirical measure to carry a failing residual law, so the gate never
        // hands one back unmodelled. Checked rather than assumed, because the
        // silent alternative is publishing a standard-normal kernel over a
        // sample the gate rejected — the exact failure this decision exists to
        // make impossible.
        return Err(
            "BMS latent-measure gate returned an unmodelled residual law even though the \
             empirical latent measure is available to this kernel"
                .to_string(),
        );
    }
    Ok((
        decision.kind,
        decision.calibration,
        decision.empirical_build,
    ))
}

/// The latent-measure gate, shared by both marginal-slope families.
///
/// The gate order is fixed and family-independent:
///
/// 1. the conditional `E[z|C]` / `Var(z|C)` Rao gate on the marginal-index span
///    `a(C)` (#905) — the `b(C)·m(C)` leakage the pooled gate cannot see and
///    that rank-INT provably cannot fix, so it takes precedence;
/// 2. the pooled standard-normal adequacy gate on raw z;
/// 3. the weighted mid-rank inverse-normal transform, re-gated on its own
///    output.
///
/// What differs between families is only the *terminal* state when the sample a
/// kernel would see fails the adequacy gate, and that is exactly what `support`
/// selects. With [`EmpiricalLatentMeasureSupport::Available`] the decision falls
/// back to the mathematically exact empirical latent measure. With
/// [`EmpiricalLatentMeasureSupport::StandardNormalOnly`] there is no such
/// fallback, so the decision instead keeps the *best available pre-transform* —
/// the one whose sample is closest to the kernel's own assumption — and returns
/// the failing ledger rather than dropping it.
///
/// Keeping the pre-transform in that case is not a smaller version of the
/// empirical branch, it is the correct choice among the two axes a
/// standard-normal-only kernel can be given: the kernel assumes N(0,1) by
/// construction, the mid-rank transform matches every quantile of that law up to
/// the sample's own discreteness, and raw z can be arbitrarily far from it. The
/// residual inadequacy is reported, never absorbed.
pub(crate) fn build_latent_measure_decision(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
    conditioning: Option<ArrayView2<'_, f64>>,
    support: EmpiricalLatentMeasureSupport,
    context: &str,
) -> Result<LatentMeasureDecision, String> {
    match policy.latent_measure {
        LatentMeasureSpec::Auto { grid_size } => {
            // #905: conditional `E[z|C]`/`Var(z|C)` Rao gate. Inspect the latent
            // score's conditional moments on the marginal-index span a(C)
            // BEFORE the pooled-marginal gate. A significant conditional shift
            // is the `b(C)·m(C)` leakage the pooled gate cannot see and that
            // rank-INT provably cannot fix, so it takes precedence: route to the
            // conditional location-scale correction `ζ = (z−m(C))/√v(C)`.
            if let Some(a_block) = conditioning
                && let Some(cal) =
                    fit_conditional_latent_calibration_if_needed(z, weights, a_block)?
            {
                // Matching the first two conditional moments does not
                // establish Gaussianity of the residual ζ (a two-point
                // residual law survives location-scale correction unchanged
                // in shape). The closed-form standard-normal kernel is only
                // admissible when the calibrated sample passes the same
                // pooled adequacy gate raw z faces; otherwise retain an
                // empirical latent measure built from ζ, so the residual
                // distribution stays the one the data show.
                let zeta = cal.apply(z.view(), a_block)?;
                let residual_adequacy = latent_z_normal_adequacy(&zeta, weights, policy)?;
                let residual_is_standard_normal = residual_adequacy.passes();
                let (kind, empirical_build) = match (residual_is_standard_normal, support) {
                    (true, _) | (false, EmpiricalLatentMeasureSupport::StandardNormalOnly) => {
                        (LatentMeasureKind::StandardNormal, None)
                    }
                    (false, EmpiricalLatentMeasureSupport::Available) => {
                        let (kind, build) =
                            build_global_empirical_latent_measure(&zeta, weights, grid_size)?;
                        (kind, Some(build))
                    }
                };
                log::info!(
                    "[{context} latent-z] conditional location-scale calibrated: basis_ncols={} var_active={} post_mean={:.3e} post_sd={:.3e} residual_measure={} (E[z|C]/Var(z|C) Rao gate fired)",
                    cal.basis_ncols,
                    !cal.var_coeffs.is_empty(),
                    cal.post_mean,
                    cal.post_sd,
                    if matches!(kind, LatentMeasureKind::StandardNormal) {
                        "standard-normal"
                    } else {
                        "global-empirical"
                    },
                );
                if !residual_is_standard_normal
                    && support == EmpiricalLatentMeasureSupport::StandardNormalOnly
                {
                    // No empirical measure exists for this kernel, so the
                    // conditional correction is kept (it removes the first-order
                    // `b(C)·m(C)` leakage regardless of the residual's shape)
                    // and the residual's distance from the kernel's assumption
                    // is handed back to the caller's `LatentZCheckMode`.
                    return Ok(LatentMeasureDecision {
                        kind,
                        calibration: LatentMeasureCalibration::ConditionalLocationScale(cal),
                        unmodelled_residual: Some(residual_adequacy),
                        empirical_build,
                    });
                }
                if !residual_is_standard_normal {
                    // gam#2484: this pair is a legitimate POINT-ESTIMATION
                    // state, so it is minted rather than refused here -- but a
                    // later Murphy-Topel generated-regressor covariance request
                    // is already determined to fail, and it used to fail three
                    // stages away with no reference back to the decision that
                    // caused it. Say so at the decision, with the evidence.
                    log::warn!(
                        "[{context} latent-z] the calibrated residual FAILED the standard-normal \
                         adequacy gate, so the second-stage latent measure is global-empirical. \
                         Point estimation is unaffected; a Murphy-Topel generated-regressor \
                         covariance will be REFUSED for this fit, because that correction needs \
                         a per-row mixed derivative and an empirical measure built from the \
                         whole calibrated-residual vector does not have one (gam#2484). \
                         Adequacy ledger (x = statistic / bound, x<=1 passed): {}",
                        residual_adequacy.ledger(),
                    );
                }
                return Ok(LatentMeasureDecision {
                    kind,
                    calibration: LatentMeasureCalibration::ConditionalLocationScale(cal),
                    unmodelled_residual: None,
                    empirical_build,
                });
            }
            let pooled_adequacy = latent_z_normal_adequacy(z, weights, policy)?;
            if pooled_adequacy.passes() {
                Ok(LatentMeasureDecision::standard_normal(
                    LatentMeasureCalibration::None,
                ))
            } else {
                // P4: route bad-normal latent z through a weighted
                // mid-distribution-rank inverse-normal transform. Rank-INT
                // redefines the latent axis (the affine rigid model is
                // specified on the calibrated score); it makes the calibrated
                // sample approximately — not exactly — N(0,1), so the
                // closed-form standard-normal kernel is admitted only when
                // the calibrated sample itself passes the adequacy gate.
                // When it cannot (heavy ties leave the calibrated law
                // discrete), fall back to the mathematically exact
                // global-empirical latent measure on the raw score.
                let calibration = LatentZRankIntCalibration::fit(z, weights)?;
                let calibrated = calibration.apply_to_training(z)?;
                let calibrated_adequacy = latent_z_normal_adequacy(&calibrated, weights, policy)?;
                if calibrated_adequacy.passes() {
                    log::info!(
                        "[{context} latent-z] rank-INT calibrated: post_mean={:.3e} post_sd={:.3e} knots={}",
                        calibration.post_mean,
                        calibration.post_sd,
                        calibration.sorted_z.len(),
                    );
                    Ok(LatentMeasureDecision::standard_normal(
                        LatentMeasureCalibration::RankInverseNormal(calibration),
                    ))
                } else {
                    match support {
                        EmpiricalLatentMeasureSupport::Available => {
                            log::info!(
                                "[{context} latent-z] rank-INT output failed the standard-normal adequacy gate (post_mean={:.3e} post_sd={:.3e} knots={}); using the global-empirical latent measure",
                                calibration.post_mean,
                                calibration.post_sd,
                                calibration.sorted_z.len(),
                            );
                            let (kind, build) =
                                build_global_empirical_latent_measure(z, weights, grid_size)?;
                            Ok(LatentMeasureDecision {
                                kind,
                                calibration: LatentMeasureCalibration::None,
                                unmodelled_residual: None,
                                empirical_build: Some(build),
                            })
                        }
                        EmpiricalLatentMeasureSupport::StandardNormalOnly => {
                            // Both candidate axes are inadequate and there is no
                            // empirical measure to carry either law. Take the
                            // one the gate itself measures as closer to the
                            // kernel's assumption -- by construction the mid-rank
                            // transform matches every quantile of N(0,1) up to
                            // the sample's discreteness, which raw z need not do
                            // at all -- and report the residual gap.
                            Ok(LatentMeasureDecision {
                                kind: LatentMeasureKind::StandardNormal,
                                calibration: LatentMeasureCalibration::RankInverseNormal(
                                    calibration,
                                ),
                                unmodelled_residual: Some(calibrated_adequacy),
                                empirical_build: None,
                            })
                        }
                    }
                }
            }
        }
        LatentMeasureSpec::StandardNormal => Ok(LatentMeasureDecision::standard_normal(
            LatentMeasureCalibration::None,
        )),
        LatentMeasureSpec::GlobalEmpirical { grid_size } => match support {
            EmpiricalLatentMeasureSupport::Available => {
                let (kind, build) = build_global_empirical_latent_measure(z, weights, grid_size)?;
                Ok(LatentMeasureDecision {
                    kind,
                    calibration: LatentMeasureCalibration::None,
                    unmodelled_residual: None,
                    empirical_build: Some(build),
                })
            }
            EmpiricalLatentMeasureSupport::StandardNormalOnly => Err(format!(
                "{context} was asked for a global-empirical latent measure, but its row kernel \
                 exists only in the closed-form standard-normal branch: there is no empirical-grid \
                 lowering to integrate against. Use the Auto latent measure (which will apply the \
                 conditional location-scale or rank inverse-normal pre-transform when the data \
                 need one) or fit this data with the Bernoulli marginal-slope family, whose kernel \
                 owns the empirical branch"
            )),
        },
    }
}

/// The standard-normal adequacy verdict on a latent-z sample: every statistic
/// the gate forms, beside the bound it was judged against.
///
/// The gate used to return a bare `bool`, and all three of its call sites threw
/// the evidence away. That is why "how far is the calibrated residual from
/// standard normal when the gate trips?" could not be answered from a fit --
/// the failing clause and its margin existed only inside the conjunction and
/// were discarded at the `&&`. The conjunction now lives in [`Self::passes`]
/// and the evidence survives it, which is what lets the conditional
/// location-scale branch say what the data did (gam#2484).
///
/// Every field is in the units of its own clause; no field is a ratio, so a
/// consumer can report either the raw statistic or its margin.
#[derive(Clone, Debug)]
pub(crate) struct LatentNormalAdequacy {
    /// Kish effective sample size `(Σw)² / Σw²`, which sets the moment bounds.
    pub(crate) effective_n: f64,
    pub(crate) mean: f64,
    pub(crate) mean_tol: f64,
    pub(crate) sd: f64,
    pub(crate) sd_tol: f64,
    pub(crate) skew: f64,
    pub(crate) skew_tol: f64,
    pub(crate) excess_kurtosis: f64,
    pub(crate) excess_kurtosis_tol: f64,
    pub(crate) ks: f64,
    pub(crate) ks_tol: f64,
    pub(crate) tail_mass_inner: f64,
    pub(crate) tail_bound_inner: f64,
    pub(crate) tail_mass_outer: f64,
    pub(crate) tail_bound_outer: f64,
    pub(crate) max_abs: f64,
    pub(crate) max_abs_tol: f64,
}

impl LatentNormalAdequacy {
    /// The nine-clause conjunction that admits the closed-form standard-normal
    /// kernel. A `NaN` statistic never passes: the `is_finite` clauses and the
    /// comparisons both reject it, which is how a degenerate sample (constant
    /// or non-finite) is refused without being reported as a large deviation it
    /// was never measured to have.
    pub(crate) fn passes(&self) -> bool {
        self.mean.abs() <= self.mean_tol
            && (self.sd - 1.0).abs() <= self.sd_tol
            && self.skew.is_finite()
            && self.skew.abs() <= self.skew_tol
            && self.excess_kurtosis.is_finite()
            && self.excess_kurtosis.abs() <= self.excess_kurtosis_tol
            && self.ks.is_finite()
            && self.ks <= self.ks_tol
            && self.tail_mass_inner <= self.tail_bound_inner
            && self.tail_mass_outer <= self.tail_bound_outer
            && self.max_abs < self.max_abs_tol
    }

    /// One line naming every clause, its statistic, its bound, and the factor by
    /// which it missed -- so a reader can tell a marginal failure from a
    /// structural one without re-running the fit. `x` is the ratio of the
    /// statistic to its bound; a clause with `x <= 1` passed.
    pub(crate) fn ledger(&self) -> String {
        fn clause(name: &str, value: f64, bound: f64) -> String {
            format!(
                "{name}={value:.4e}/{bound:.4e}(x{:.2})",
                (value / bound).abs()
            )
        }
        format!(
            "n_eff={:.1} {} {} {} {} {} {} {} {}",
            self.effective_n,
            clause("|mean|", self.mean, self.mean_tol),
            clause("|sd-1|", self.sd - 1.0, self.sd_tol),
            clause("|skew|", self.skew, self.skew_tol),
            clause(
                "|excess_kurtosis|",
                self.excess_kurtosis,
                self.excess_kurtosis_tol
            ),
            clause("ks", self.ks, self.ks_tol),
            clause(
                "tail_mass_inner",
                self.tail_mass_inner,
                self.tail_bound_inner
            ),
            clause(
                "tail_mass_outer",
                self.tail_mass_outer,
                self.tail_bound_outer
            ),
            clause("max_abs", self.max_abs, self.max_abs_tol),
        )
    }
}

/// Measure a latent-z sample against the standard-normal adequacy gate,
/// returning every statistic and bound rather than only the verdict.
pub(crate) fn latent_z_normal_adequacy(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    policy: &LatentZPolicy,
) -> Result<LatentNormalAdequacy, String> {
    if z.len() != weights.len() {
        return Err(format!(
            "latent-measure auto-detection length mismatch: z={}, weights={}",
            z.len(),
            weights.len()
        ));
    }
    let weight_sum = weights.iter().copied().sum::<f64>();
    let weight_sq_sum = weights.iter().map(|&w| w * w).sum::<f64>();
    if !(weight_sum.is_finite()
        && weight_sum > 0.0
        && weight_sq_sum.is_finite()
        && weight_sq_sum > 0.0)
    {
        return Err("latent-measure auto-detection requires positive finite weights".to_string());
    }
    let effective_n = weight_sum * weight_sum / weight_sq_sum;
    if !(effective_n.is_finite() && effective_n > 1.0) {
        return Err(
            "latent-measure auto-detection requires at least two effective observations"
                .to_string(),
        );
    }
    let mean = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * zi)
        .sum::<f64>()
        / weight_sum;
    let var = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| wi * (zi - mean) * (zi - mean))
        .sum::<f64>()
        / weight_sum;
    let sd = var.sqrt();
    // The two moment bounds depend only on the effective sample size, so they
    // are formed before the degeneracy check and stated once for both exits.
    let mean_tol = policy.mean_tol_multiplier / effective_n.sqrt();
    let sd_tol = policy.sd_tol_multiplier / (2.0 * (effective_n - 1.0).max(1.0)).sqrt();
    if !(mean.is_finite() && sd.is_finite() && sd > 0.0) {
        // A constant or non-finite sample has no shape: every standardized
        // statistic divides by `sd`, so skewness, kurtosis and the tail masses
        // are UNDEFINED here rather than merely large. `NaN` is the honest
        // entry -- `passes` rejects it through the same `is_finite` clauses the
        // conjunction always had, so this exit refuses the standard-normal
        // kernel exactly as the previous `Ok(false)` did, without reporting a
        // deviation that was never measured.
        return Ok(LatentNormalAdequacy {
            effective_n,
            mean,
            mean_tol,
            sd,
            sd_tol,
            skew: f64::NAN,
            skew_tol: policy.max_abs_skew.min(AUTO_Z_NORMAL_SKEW_TOL),
            excess_kurtosis: f64::NAN,
            excess_kurtosis_tol: policy.max_abs_excess_kurtosis.min(AUTO_Z_NORMAL_KURT_TOL),
            ks: f64::NAN,
            ks_tol: AUTO_Z_NORMAL_KS_TOL,
            tail_mass_inner: f64::NAN,
            tail_bound_inner: AUTO_Z_NORMAL_TAIL_MASS_SLACK
                * normal_two_sided_probability(AUTO_Z_NORMAL_TAIL_SIGMA_INNER)
                + AUTO_Z_NORMAL_TAIL_FLOOR_INNER,
            tail_mass_outer: f64::NAN,
            tail_bound_outer: AUTO_Z_NORMAL_TAIL_MASS_SLACK
                * normal_two_sided_probability(AUTO_Z_NORMAL_TAIL_SIGMA_OUTER)
                + AUTO_Z_NORMAL_TAIL_FLOOR_OUTER,
            max_abs: f64::NAN,
            max_abs_tol: AUTO_Z_NORMAL_MAX_ABS,
        });
    }
    let skew = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| {
            let centered = (zi - mean) / sd;
            wi * centered.powi(3)
        })
        .sum::<f64>()
        / weight_sum;
    let excess_kurtosis = z
        .iter()
        .zip(weights.iter())
        .map(|(&zi, &wi)| {
            let centered = (zi - mean) / sd;
            wi * centered.powi(4)
        })
        .sum::<f64>()
        / weight_sum
        - 3.0;
    let ks_to_normal = weighted_ks_to_standard_normal(z, weights, weight_sum)?;
    let tail_mass_4 = weighted_tail_mass(z, weights, weight_sum, AUTO_Z_NORMAL_TAIL_SIGMA_INNER);
    let tail_mass_6 = weighted_tail_mass(z, weights, weight_sum, AUTO_Z_NORMAL_TAIL_SIGMA_OUTER);
    let max_abs_z = z.iter().fold(0.0_f64, |acc, &zi| acc.max(zi.abs()));
    let normal_tail_4 = normal_two_sided_probability(AUTO_Z_NORMAL_TAIL_SIGMA_INNER);
    let normal_tail_6 = normal_two_sided_probability(AUTO_Z_NORMAL_TAIL_SIGMA_OUTER);
    Ok(LatentNormalAdequacy {
        effective_n,
        mean,
        mean_tol,
        sd,
        sd_tol,
        skew,
        skew_tol: policy.max_abs_skew.min(AUTO_Z_NORMAL_SKEW_TOL),
        excess_kurtosis,
        excess_kurtosis_tol: policy.max_abs_excess_kurtosis.min(AUTO_Z_NORMAL_KURT_TOL),
        ks: ks_to_normal,
        ks_tol: AUTO_Z_NORMAL_KS_TOL,
        tail_mass_inner: tail_mass_4,
        tail_bound_inner: AUTO_Z_NORMAL_TAIL_MASS_SLACK * normal_tail_4
            + AUTO_Z_NORMAL_TAIL_FLOOR_INNER,
        tail_mass_outer: tail_mass_6,
        tail_bound_outer: AUTO_Z_NORMAL_TAIL_MASS_SLACK * normal_tail_6
            + AUTO_Z_NORMAL_TAIL_FLOOR_OUTER,
        max_abs: max_abs_z,
        max_abs_tol: AUTO_Z_NORMAL_MAX_ABS,
    })
}

/// The global-empirical latent measure AND the fit-time record of how it was
/// built.
///
/// The two are returned together, never separately: the record is what makes
/// the measure differentiable in the sample it was compressed from (gam#2484),
/// and a record obtained from a second call to the builder would be a record of
/// a second compression.
pub(crate) fn build_global_empirical_latent_measure(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    grid_size: usize,
) -> Result<
    (
        LatentMeasureKind,
        empirical_measure_sensitivity::EmpiricalZGridBuild,
    ),
    String,
> {
    let build = empirical_measure_sensitivity::build_empirical_z_grid_with_alpha(
        z.view(),
        weights.view(),
        grid_size,
        "empirical latent measure",
    )?;
    let measure = LatentMeasureKind::GlobalEmpirical {
        grid: build.grid.clone(),
    };
    measure.validate("empirical latent measure")?;
    Ok((measure, build))
}

pub(crate) fn weighted_ks_to_standard_normal(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    total_weight: f64,
) -> Result<f64, String> {
    let mut pairs = Vec::<(f64, f64)>::with_capacity(z.len());
    for (&zi, &wi) in z.iter().zip(weights.iter()) {
        if !zi.is_finite() || !wi.is_finite() || wi < 0.0 {
            return Err(
                "latent-measure KS diagnostic requires finite z and non-negative finite weights"
                    .to_string(),
            );
        }
        if wi > 0.0 {
            pairs.push((zi, wi));
        }
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .expect("validated latent z values are finite")
    });
    let mut prev = 0.0;
    let mut ks = 0.0_f64;
    for (zi, wi) in pairs {
        let cdf = normal_cdf(zi);
        let next = prev + wi / total_weight;
        ks = ks.max((cdf - prev).abs()).max((cdf - next).abs());
        prev = next;
    }
    Ok(ks)
}

pub(crate) fn weighted_tail_mass(
    z: &Array1<f64>,
    weights: &Array1<f64>,
    total_weight: f64,
    cutoff: f64,
) -> f64 {
    z.iter()
        .zip(weights.iter())
        .filter(|&(&zi, _)| zi.abs() > cutoff)
        .map(|(_, &wi)| wi)
        .sum::<f64>()
        / total_weight
}

// ---------------------------------------------------------------------------
// Cross-module constants — declared here so all submodules can reach them
// via `use super::*` without promoting implementation details to pub(crate).
// ---------------------------------------------------------------------------
pub(super) const BMS_AUTO_SUBSAMPLE_PHASE1_BUDGET: usize = 12;
pub(super) const BERNOULLI_LINK_PROBABILITY_EPS: f64 = 1e-12;
pub(super) const BMS_VARIANCE_FLOOR: f64 = 1e-12;
pub(super) const BMS_DERIV_TOL: f64 = 1e-8;
/// Relative tolerance below which a residual weight is treated as exhausted in
/// the equal-mass empirical-grid compression loop. Used both for the per-bin
/// "need" remaining (relative to the target bin weight) and for the per-pair
/// remainder (relative to that pair's weight), so a pair/bin that is filled to
/// within a few ulps advances the cursor instead of spinning on round-off.
pub(super) const EMPIRICAL_GRID_WEIGHT_EXHAUSTED_REL_TOL: f64 = 1e-14;
/// Upper bound (and large-`n` default) for rows-per-chunk in the parallel
/// row-accumulation phases.
///
/// This is also a hard *ceiling* the [`bms_row_chunk_size`] chunk sizing must
/// respect: several per-chunk fast paths (block-Hessian / block-gradient
/// assembly) allocate fixed `[0.0f64; ROW_CHUNK_SIZE]` stack buffers and index
/// them by the chunk's local row position, so a chunk may never carry more than
/// `ROW_CHUNK_SIZE` rows.
pub(super) const ROW_CHUNK_SIZE: usize = 1024;
/// Floor for rows-per-chunk: below it the per-chunk scratch allocation +
/// scheduler hand-off cost dominates the row arithmetic. Small enough that a
/// moderate `n` on a many-core box still carves several chunks per worker.
pub(super) const ROW_CHUNK_MIN: usize = 64;
/// Target number of row-chunks per rayon worker for the BMS exact-Newton
/// row-fan-out phases (gradient / HVP / diagonal directional-derivative sweeps).
///
/// Several chunks per worker keeps the pool load-balanced across the uneven
/// per-row cost tail (work-stealing moves whole chunks, never partial sums) so
/// the heavy coord-corrections / row-stream phases saturate the cores instead
/// of stranding the tail on one worker.
pub(super) const ROW_CHUNKS_PER_WORKER: usize = 4;

/// Pool-aware rows-per-chunk for the BMS exact-Newton row fan-outs.
///
/// A *fixed* `ROW_CHUNK_SIZE` divisor makes the chunk **count** scale with `n`,
/// so at moderate `n` (e.g. `n = 10·ROW_CHUNK_SIZE` on a 64-core box) the
/// `into_par_iter` over `⌈n/ROW_CHUNK_SIZE⌉` chunks has far fewer tasks than
/// workers and most cores idle — the measured ~30-90% core utilization on the
/// biobank coord-corrections / row-stream phases. This sizes the chunk so the
/// chunk count targets `ROW_CHUNKS_PER_WORKER × worker_count` (the same policy
/// `chunked_row_reduction` uses), clamped to `[ROW_CHUNK_MIN, ROW_CHUNK_SIZE]`:
///
/// * the `ROW_CHUNK_SIZE` ceiling is mandatory — the block-assembly fast paths
///   index fixed `[…; ROW_CHUNK_SIZE]` stack buffers by local row, so a chunk
///   can never exceed it. At large `n` the per-1024-row count already exceeds
///   the worker count, so the clamp costs nothing there;
/// * the `ROW_CHUNK_MIN` floor stops sub-floor fan-out at tiny `n`.
///
/// Reproducibility contract (#1045): the worker count used here is the
/// process-stable machine parallelism (`reproducible_chunk_parallelism`), NOT
/// the live `rayon::current_num_threads()` of the executing (possibly scoped,
/// possibly shrunk) pool. Keying the chunk *count* — and hence the chunk
/// boundaries `chunk_idx·chunk → (chunk_idx+1)·chunk` — to the transient pool
/// size made the per-chunk row sums regroup when the pool was narrowed, so a
/// fit reduced over these chunks and fed into the iterative REML optimizer moved
/// its `(ρ, λ)` selection with the pool size. Anchoring to a process constant
/// makes the boundaries — and therefore the `try_fold`/`try_reduce` reduction
/// tree that round-trips through them — invariant to how many workers run the
/// fit, while rayon still fans the chunks across whatever workers exist. For a
/// given `n` the returned chunk size is stable across calls and pool sizes.
#[inline]
pub(super) fn bms_row_chunk_size(n: usize) -> usize {
    if n == 0 {
        return ROW_CHUNK_SIZE;
    }
    let workers = crate::marginal_slope_shared::reproducible_chunk_parallelism();
    let target_chunks = workers.saturating_mul(ROW_CHUNKS_PER_WORKER).max(1);
    // Rows per chunk that yields ≈ `target_chunks` chunks, clamped into
    // `[ROW_CHUNK_MIN, ROW_CHUNK_SIZE]`.
    n.div_ceil(target_chunks)
        .clamp(ROW_CHUNK_MIN, ROW_CHUNK_SIZE)
}
pub(super) const EXACT_WORK_LOG_MIN_ROWS: usize = 50_000;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_EXPECTED_REUSE_PASSES: usize = 3;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_MIN_REUSE_PASSES: usize = 2;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_TILE_ROWS: usize = 8192;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_NUM: u64 = 1;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_SINGLE_FRACTION_DEN: u64 = 4;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_GLOBAL_FRACTION_NUM: u64 = 1;
pub(super) const BMS_ROW_PRIMARY_HESSIAN_GLOBAL_FRACTION_DEN: u64 = 2;
pub(super) const BERNOULLI_MARGSLOPE_LINE_SEARCH_EARLY_EXIT_CHUNK_ROWS: usize = 10_000;

// ---------------------------------------------------------------------------
// Submodule declarations
// ---------------------------------------------------------------------------
pub(crate) mod block_specs;
pub mod conditional_score_covariance;
pub(crate) mod exact_eval_cache;
pub(crate) mod family;
pub(crate) mod flex_row_program;
pub(crate) mod gradient_paths;
pub(crate) mod hessian_paths;
pub(crate) mod install_flex;
pub(crate) mod row_kernel;
#[cfg(test)]
mod tests {
    include!("../../../../tests/src_modules/misc/families_bms_identifiability_rigid_tests.rs");
    include!(
        "../../../../tests/src_modules/optimization/families_bms_joint_hessian_hvp_correction_tests.rs"
    );

    #[test]
    fn empirical_grid_constructor_preserves_canonical_node_order() {
        let grid = EmpiricalZGrid::new(
            vec![-2.0, 0.5, 1.0],
            vec![0.3, 0.5, 0.2],
            "sorted-grid invariant",
        )
        .expect("canonical sorted grid");
        assert_eq!(grid.nodes, vec![-2.0, 0.5, 1.0]);
        assert_eq!(grid.weights, vec![0.3, 0.5, 0.2]);
    }

    #[test]
    fn empirical_grid_constructor_rejects_noncanonical_node_order() {
        let err = EmpiricalZGrid::new(vec![0.0, -1.0], vec![0.5, 0.5], "sorted-grid invariant")
            .expect_err("constructed grids must already be canonical");
        assert!(err.contains("nodes must be sorted ascending"), "{err}");
    }
}

#[cfg(test)]
mod stacked_first_stage_sandwich_2484_tests {
    use super::{stacked_first_stage_sandwich_cov, weighted_ridge_sandwich_cov};
    use ndarray::{Array1, Array2, array};

    /// `A`, weights, and a normal matrix `M = AᵀWA + λR` built exactly the way
    /// the calibration builds it, so the sandwich is exercised on a realistic
    /// bread rather than on an identity.
    fn system(basis: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
        let mut wa = basis.clone();
        for (mut row, &w) in wa.rows_mut().into_iter().zip(weights.iter()) {
            row.iter_mut().for_each(|value| *value *= w);
        }
        let mut m = basis.t().dot(&wa);
        let diag: Vec<f64> = (0..m.nrows()).map(|j| m[[j, j]]).collect();
        for (j, value) in diag.iter().enumerate() {
            m[[j, j]] += value * 1.0e-8;
        }
        m
    }

    /// PAIRED fixture: every conditioning row appears twice with equal weight
    /// and mean residuals `+c` / `−c`.
    ///
    /// That makes both cross-terms cancel EXACTLY rather than approximately —
    /// the bread's `Σ w·û·A Aᵀ` cancels because `û` flips sign while `A Aᵀ`
    /// does not, and the meat's `Σ w²·û·r·A Aᵀ` cancels because `r` depends on
    /// `û²` and is therefore equal across the pair. So this arm asserts an
    /// identity, not a tolerance: with no third moment, the joint sandwich must
    /// reproduce the block-diagonal form the code used to assume.
    #[test]
    fn symmetric_residuals_reproduce_the_block_diagonal_form_2484() {
        let basis = array![
            [1.0, 0.4],
            [1.0, 0.4],
            [1.0, -0.7],
            [1.0, -0.7],
            [1.0, 1.3],
            [1.0, 1.3],
        ];
        let weights = Array1::from(vec![1.0, 1.0, 0.5, 0.5, 2.0, 2.0]);
        let mean_residuals = vec![0.6, -0.6, 0.9, -0.9, 0.3, -0.3];
        // r depends on û only through û², so it is equal within each pair.
        let var_residuals: Vec<f64> = mean_residuals.iter().map(|&u| u * u - 0.5).collect();
        let m = system(&basis, &weights);

        let joint = stacked_first_stage_sandwich_cov(
            basis.view(),
            weights.view(),
            &mean_residuals,
            &var_residuals,
            &m,
        )
        .expect("joint sandwich");
        let mean_block =
            weighted_ridge_sandwich_cov(basis.view(), &mean_residuals, weights.view(), &m)
                .expect("mean sandwich");
        let var_block =
            weighted_ridge_sandwich_cov(basis.view(), &var_residuals, weights.view(), &m)
                .expect("var sandwich");

        let p = basis.ncols();
        for i in 0..p {
            for j in 0..p {
                let tol = 1.0e-9 * (1.0 + mean_block[[i, j]].abs());
                assert!(
                    (joint[[i, j]] - mean_block[[i, j]]).abs() <= tol,
                    "gam#2484: with no third moment the (mean,mean) block must reproduce the \
                     standalone sandwich; got {} vs {}",
                    joint[[i, j]],
                    mean_block[[i, j]]
                );
                let tol_v = 1.0e-9 * (1.0 + var_block[[i, j]].abs());
                assert!(
                    (joint[[p + i, p + j]] - var_block[[i, j]]).abs() <= tol_v,
                    "gam#2484: with no third moment the (var,var) block must reproduce the \
                     standalone sandwich; got {} vs {}",
                    joint[[p + i, p + j]],
                    var_block[[i, j]]
                );
                assert!(
                    joint[[i, p + j]].abs() <= 1.0e-9,
                    "gam#2484: the cross-block must vanish when the residual third moment does; \
                     got {}",
                    joint[[i, p + j]]
                );
            }
        }
    }

    /// The arm that asserts the change DOES something. Break the pairing so the
    /// residual carries a third moment, and the cross-block must be measurably
    /// non-zero — otherwise an implementation that quietly returns the old
    /// block-diagonal form passes the test above and ships.
    #[test]
    fn skewed_residuals_make_the_cross_block_nonzero_2484() {
        let basis = array![
            [1.0, 0.4],
            [1.0, 0.9],
            [1.0, -0.7],
            [1.0, 0.2],
            [1.0, 1.3],
            [1.0, -1.1],
        ];
        let weights = Array1::from(vec![1.0, 1.0, 0.5, 0.5, 2.0, 2.0]);
        // Strongly right-skewed mean residuals: one large positive, the rest
        // small negative. This is the shape the adequacy gate rejects.
        let mean_residuals = vec![-0.2, -0.3, -0.25, -0.15, 2.4, -0.35];
        let var_residuals: Vec<f64> = mean_residuals.iter().map(|&u| u * u - 0.5).collect();
        let m = system(&basis, &weights);

        let joint = stacked_first_stage_sandwich_cov(
            basis.view(),
            weights.view(),
            &mean_residuals,
            &var_residuals,
            &m,
        )
        .expect("joint sandwich");
        let var_block =
            weighted_ridge_sandwich_cov(basis.view(), &var_residuals, weights.view(), &m)
                .expect("var sandwich");

        let p = basis.ncols();
        let cross = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| joint[[i, p + j]].abs())
            .fold(0.0_f64, f64::max);
        let scale = (0..p)
            .map(|i| joint[[i, i]].abs().max(joint[[p + i, p + i]].abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            cross > 1.0e-6 * scale.max(1.0e-12),
            "gam#2484: a skewed residual must produce a NON-ZERO first-stage cross-block. \
             max|cross|={cross:e} against block scale {scale:e}. If this fails, the \
             implementation is still returning a block-diagonal V1 and the whole change is \
             inert."
        );

        // And the (var,var) block must differ from the standalone sandwich: the
        // triangular bread feeds the mean-stage uncertainty into it.
        let var_shift = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| (joint[[p + i, p + j]] - var_block[[i, j]]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            var_shift > 1.0e-6 * scale.max(1.0e-12),
            "gam#2484: the (var,var) block must absorb the mean-stage uncertainty through the \
             lower-triangular bread; max shift {var_shift:e} is indistinguishable from the old \
             standalone block"
        );
    }
}

pub(crate) mod axis_direction_search;
pub(crate) mod cell_moment_assembly;
#[cfg(test)]
mod empirical_measure_2484_tests;
pub(crate) mod empirical_measure_sensitivity;
// #932 BMS flex single-source jet substrate (runtime-dimension `Jet2` + IFT
// lift + cell base-moment jets). A bare `#[cfg(test)] mod` with an allowed name
// so the build.rs ban-scanner exempts it; shared by its own FD gates and the
// `cell_moment_assembly` flex-fixture oracle gate as a private child of `bms`.
#[cfg(test)]
mod test_support;
// #932 INDEPENDENT adversarial verifier (bms-flex-verify): a high-order
// finite-difference oracle on the production compiled lowering
// `lower_bms_flex_row_order2_from_parts` of the canonical BMS FLEX program,
// plus a moving-edge Leibniz
// cross-check + a planted-corruption tripwire. Bare `#[cfg(test)] mod` with the
// allowed `*_tests` name so the build.rs ban-scanner exempts it; owned solely by
// the verifier (never edits the implementer's row_primary_hessian /
// gradient_paths / cell_moment_assembly).
pub(crate) mod custom_family_impl;
#[cfg(test)]
mod flex_verify_932_tests;
// #932 direct production-path measurement: forced 65-node empirical grid,
// warmed/cold row-op allocation counting + ns/row diagnostics for the MSI
// A/B ledger. The asserted gate is per-row allocation calls (deterministic);
// timing is eprintln-only per the SPEC ban on wall-clock correctness budgets.
#[cfg(test)]
mod flex_measure_932_tests;
// gam#2768 unit gates on the shared latent-measure decision and the conditional
// location-scale calibration it escalates to. Bare `#[cfg(test)] mod` with the
// allowed `*_tests` name so the build.rs ban-scanner exempts it.
#[cfg(test)]
mod latent_measure_2768_tests;
pub(crate) mod row_primary_hessian;

pub use block_specs::fit_bernoulli_marginal_slope_terms;
pub use conditional_score_covariance::{
    ConditionalScoreCoordinate, ConditionalScoreCovariance, ScoreCovarianceField,
};
pub use gradient_paths::{MarginalSlopeCovariance, MarginalSlopeCovarianceShape, marginal_slope_covariance_from_scores, padded_deviation_seed};
pub use install_flex::CrossBlockIdentifiabilityWarning;
pub(crate) use install_flex::FlexCompileOutcome;

// pub(crate) re-exports for internal callers:
pub(crate) use block_specs::push_deviation_aux_blockspecs;
pub use block_specs::{BmsSlopeJacobian, BmsMarginalJacobian};
pub(crate) use family::{
    BernoulliMarginalLinkMap, bernoulli_marginal_link_map,
    build_link_deviation_block_from_knots_design_seed_and_weights,
    build_score_warp_deviation_block_from_seed,
};
pub(crate) use gradient_paths::MarginalSlopeCovarianceRef;
pub(crate) use gradient_paths::signed_probit_neglog_unary_stack;
pub(crate) use gradient_paths::standardize_latent_z_with_policy;
pub(crate) use gradient_paths::{
    empirical_intercept_from_marginal, signed_probit_neglog_derivatives_up_to_fourth,
    unary_derivatives_inverse_sqrt, unary_derivatives_log, unary_derivatives_log_normal_pdf,
    unary_derivatives_neglog_phi, unary_derivatives_sqrt,
};
pub(crate) use install_flex::{
    install_compiled_flex_block_into_runtime, project_monotone_feasible_beta,
};
