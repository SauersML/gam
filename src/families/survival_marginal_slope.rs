use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyWarmStart,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    build_block_spatial_psi_derivatives, custom_family_outer_derivatives,
    evaluate_custom_family_joint_hyper_efs_shared, evaluate_custom_family_joint_hyper_shared,
    fit_custom_family, fit_custom_family_fixed_log_lambda_warm_start,
    joint_hyper_options_for_outer_tolerance,
};
use crate::estimate::UnifiedFitResult;
use crate::faer_ndarray::{fast_ab, fast_av};
use crate::families::bernoulli_marginal_slope::{
    DeviationBlockConfig, DeviationRuntime, LatentZNormalization, LatentZPolicy,
    MarginalSlopeCovariance, build_link_deviation_block_from_knots_design_seed_and_weights,
    build_score_warp_deviation_block_from_seed, marginal_slope_covariance_from_scores,
    marginal_slope_preserving_scale, marginal_slope_probit_eta, padded_deviation_seed,
    project_monotone_feasible_beta, push_deviation_aux_blockspecs,
    signed_probit_neglog_derivatives_up_to_fourth, standardize_latent_z_with_policy,
    unary_derivatives_log, unary_derivatives_log_normal_pdf, unary_derivatives_neglog_phi,
    unary_derivatives_sqrt,
};
use crate::families::cubic_cell_kernel as exact_kernel;
use crate::families::gamlss::{ParameterBlockInput, monotone_wiggle_basis_with_derivative_order};
use crate::families::lognormal_kernel::FrailtySpec;
use crate::families::marginal_slope_shared::{
    CoeffSupport, ObservedDenestedCellPartials, SparsePrimaryCoeffJetView, add_optional_matrix,
    add_optional_vector, add_scaled_coeff4, add_two_surface_psi_outer,
    build_denested_partition_cells as shared_denested_partition_cells, chunked_row_reduction,
    eval_coeff4_at, is_sigma_aux_index as shared_is_sigma_aux_index,
    observed_denested_cell_partials as shared_observed_denested_cell_partials, outer_row_indices,
    outer_row_weights_by_index, outer_weighted_rows, probit_frailty_scale,
    probit_frailty_scale_multi_dir_jet, psi_derivative_location, scale_coeff4,
};
use crate::families::row_kernel::{
    RowKernel, RowKernelHessianWorkspace, build_row_kernel_cache, row_kernel_gradient,
    row_kernel_hessian_dense, row_kernel_log_likelihood,
};
use crate::families::survival::{OffsetChannelCurvatures, OffsetChannelResiduals};
use crate::families::survival_location_scale::{
    TimeBlockInput, TimeWiggleBlockInput, project_onto_linear_constraints,
    structural_time_coefficient_constraints,
};
use crate::matrix::{DesignMatrix, SymmetricMatrix};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::signed_probit_logcdf_and_mills_ratio;
use crate::smooth::{
    BlockwisePenalty, ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions,
    SpatialLogKappaCoords, TermCollectionDesign, TermCollectionSpec,
    build_term_collection_designs_and_freeze_joint, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::solver::estimate::reml::unified::HyperOperator;
use crate::types::{InverseLink, LinkFunction};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::cell::RefCell;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

/// Inline-stored polynomial coefficient vector for survival marginal-slope
/// integrand assembly. `poly_*` helpers in this module routinely build
/// degree ≤ ~28 polynomials (max product of four affine cell coefficient
/// arrays of length 4) inside per-row hot loops; the previous `Vec<f64>`
/// returns drove millions of mallocs per outer iteration on biobank-scale
/// fits. Thirty-two inline slots cover every observed shape.
type PolyVec = SmallVec<[f64; 32]>;

// ── Typed errors ──────────────────────────────────────────────────────
//
// Categorizes the failure modes of the survival marginal-slope family so
// callers can match on the kind without parsing strings. The `reason`
// fields preserve the original `format!(...)` text byte-for-byte; the
// `Display` impl prints just the reason, making `e.to_string()` identical
// to the pre-migration `String` errors that flowed through `?`.
#[derive(Debug, Clone)]
pub enum SurvivalMarginalSlopeError {
    /// Spec, data, or runtime configuration failed input validation
    /// (finite/non-negative weights, derivative_guard > 0, supported
    /// base_link, frailty constraints, missing block state, etc.).
    InvalidInput { reason: String },
    /// Lengths, row/column counts, basis widths, or coefficient block
    /// sizes do not agree (covariance dim vs z, design rows vs n,
    /// basis/beta length mismatch, post-update beta length, time
    /// constraints A vs b, hessian_matvec dim mismatch, ...).
    IncompatibleDimensions { reason: String },
    /// A row's transformed time derivative or structural slack fell
    /// below `derivative_guard` (`qd1 < guard`), violating the
    /// monotonicity contract.
    MonotonicityViolation { reason: String },
    /// A numerical step produced a non-finite, non-positive, or
    /// internally inconsistent quantity that downstream code cannot
    /// consume (e.g. non-positive `D`, non-positive `chi1`, calibration
    /// derivative disagrees with the direct evaluation, transformed
    /// derivative not strictly positive).
    NumericalFailure { reason: String },
    /// An integration / outer-optimization step failed to converge to
    /// the requested tolerance (intercept residual, REML outer loop).
    IntegrationFailed { reason: String },
    /// The requested combination of options is not implemented (non-
    /// probit base link, flexible row calculus with K > 1, spatial psi
    /// for unsupported block roles, ...).
    UnsupportedConfiguration { reason: String },
}

impl std::fmt::Display for SurvivalMarginalSlopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurvivalMarginalSlopeError::InvalidInput { reason }
            | SurvivalMarginalSlopeError::IncompatibleDimensions { reason }
            | SurvivalMarginalSlopeError::MonotonicityViolation { reason }
            | SurvivalMarginalSlopeError::NumericalFailure { reason }
            | SurvivalMarginalSlopeError::IntegrationFailed { reason }
            | SurvivalMarginalSlopeError::UnsupportedConfiguration { reason } => {
                f.write_str(reason)
            }
        }
    }
}

impl std::error::Error for SurvivalMarginalSlopeError {}

impl From<SurvivalMarginalSlopeError> for String {
    fn from(err: SurvivalMarginalSlopeError) -> String {
        err.to_string()
    }
}

impl From<String> for SurvivalMarginalSlopeError {
    /// Inbound conversion from helpers in this module (and adjacent
    /// families) that still surface `Result<_, String>`. The text is
    /// preserved verbatim; `InvalidInput` is the catch-all category for
    /// strings produced outside this module.
    fn from(reason: String) -> SurvivalMarginalSlopeError {
        SurvivalMarginalSlopeError::InvalidInput { reason }
    }
}

// ── Spec and result types ─────────────────────────────────────────────

#[derive(Clone)]
pub struct SurvivalMarginalSlopeTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub z: Array2<f64>,
    pub base_link: InverseLink,
    pub marginalspec: TermCollectionSpec,
    pub marginal_offset: Array1<f64>,
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
    /// Strict lower bound on q'(t) used by both the likelihood domain and
    /// the monotonicity constraints.
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
    pub logslopespec: TermCollectionSpec,
    pub logslopespecs: Option<Vec<TermCollectionSpec>>,
    pub logslope_offset: Array1<f64>,
    pub score_warp: Option<DeviationBlockConfig>,
    pub link_dev: Option<DeviationBlockConfig>,
    pub latent_z_policy: LatentZPolicy,
}

pub const DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD: f64 = 1e-6;
const SURVIVAL_INTERCEPT_ABS_RESIDUAL_TOL: f64 = 1e-12;
const SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL: f64 = 1e-8;
const SURVIVAL_INTERCEPT_LOG_TAIL_THRESHOLD: f64 = 1e-8;

#[inline]
fn survival_derivative_guard_tolerance(qd1: f64, derivative_guard: f64) -> f64 {
    256.0 * f64::EPSILON * (1.0 + qd1.abs().max(derivative_guard.abs()))
}

#[inline]
fn survival_derivative_guard_violated(qd1: f64, derivative_guard: f64) -> bool {
    !qd1.is_finite()
        | !derivative_guard.is_finite()
        | (qd1 + survival_derivative_guard_tolerance(qd1, derivative_guard) < derivative_guard)
}

pub struct SurvivalMarginalSlopeFitResult {
    pub fit: UnifiedFitResult,
    pub marginalspec_resolved: TermCollectionSpec,
    pub logslopespec_resolved: TermCollectionSpec,
    pub marginal_design: TermCollectionDesign,
    /// Learned or fixed Gaussian-shift frailty SD.  `None` = no frailty.
    pub gaussian_frailty_sd: Option<f64>,
    pub logslope_design: TermCollectionDesign,
    pub baseline_slope: f64,
    pub baseline_offset_residuals: OffsetChannelResiduals,
    pub baseline_offset_curvatures: OffsetChannelCurvatures,
    pub z_normalization: LatentZNormalization,
    pub time_block_penalties_len: usize,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
}

// ── Family struct ─────────────────────────────────────────────────────

/// The time block has one beta vector but THREE design matrices (entry, exit,
/// derivative-at-exit). The ParameterBlockSpec uses the exit design as its
/// "official" design, so block_states[0].eta = design_exit @ beta + offset_exit.
/// This eta is NOT used in the likelihood computation — row_neglog_directional
/// recomputes all 3 linear predictors from beta_time directly. The exit-design
/// eta exists only to satisfy the CustomFamily/PIRLS interface; ExactNewton
/// blocks do not use eta for working response/weights.
#[derive(Clone)]
struct SurvivalMarginalSlopeFamily {
    n: usize,
    event: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    z: Arc<Array2<f64>>,
    score_covariance: MarginalSlopeCovariance,
    gaussian_frailty_sd: Option<f64>,
    derivative_guard: f64,
    /// Time block: 3 designs sharing one beta vector.
    /// Stored as DesignMatrix to support sparse local-support bases at
    /// biobank scale (B-spline/I-spline rows have only degree+1 nonzeros).
    design_entry: DesignMatrix,
    design_exit: DesignMatrix,
    design_derivative_exit: DesignMatrix,
    offset_entry: Arc<Array1<f64>>,
    offset_exit: Arc<Array1<f64>>,
    derivative_offset_exit: Arc<Array1<f64>>,
    /// Baseline covariate block: contributes additively to q0 and q1, but not qd1.
    marginal_design: DesignMatrix,
    /// Log-slope block: standard single design.
    logslope_design: DesignMatrix,
    logslope_surface_ranges: Vec<std::ops::Range<usize>>,
    score_warp: Option<DeviationRuntime>,
    link_dev: Option<DeviationRuntime>,
    time_linear_constraints: Option<LinearInequalityConstraints>,
    time_wiggle_knots: Option<Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    /// Per-row cache of the previous PIRLS iter's converged intercepts. Two
    /// slots per row: `[entry_q0, exit_q1]`. Across consecutive PIRLS
    /// iterations β changes only a little, so the previously-converged `a` is
    /// an excellent initial guess for the calibration root and typically lets
    /// the solver finish in ~1–2 iterations versus the rigid closed-form seed
    /// which can be many bracket-expansion steps away. Slots are initialised
    /// to `NaN` (sentinel for "not yet solved") and overwritten with the
    /// converged intercept on every successful call.
    ///
    /// Set to `None` for unit-test fixtures that build a
    /// `SurvivalMarginalSlopeFamily` directly without running the full fit
    /// pipeline; production paths go through `make_family` which initialises
    /// the cache to length-`n`. When `None`, the solver behaves exactly as it
    /// did before the warm-start machinery was added (closed-form rigid seed).
    intercept_warm_starts: Option<Arc<SurvivalInterceptWarmStartCache>>,
    /// Per-fit counter of outer evaluations. Increments on each distinct
    /// outer step (detected via the concatenated-beta proxy stored in
    /// `auto_subsample_last_rho`). Drives the same two-phase
    /// auto-subsample schedule used by `BernoulliMarginalSlopeFamily`:
    /// the first `SURVIVAL_MGS_AUTO_SUBSAMPLE_PHASE1_BUDGET` evaluations
    /// install a stratified Horvitz-Thompson mask (Phase 1, ≈ 1 %
    /// gradient noise); subsequent evaluations revert to full data
    /// (Phase 2). The counter resets per fit because each fit
    /// constructs a fresh family.
    auto_subsample_phase_counter: Arc<AtomicUsize>,
    /// Companion to `auto_subsample_phase_counter`. Stores the
    /// concatenated-beta vector seen at the most recent counter bump.
    /// Survival entry points (`*_workspace_with_options`) do not receive
    /// the outer ρ directly, so we use the joint coefficient vector as
    /// a stable per-outer-eval key. Within a single outer eval all
    /// downstream calls share the same betas, so retries don't bump the
    /// counter; across outer evals the betas change so the counter
    /// increments cleanly.
    auto_subsample_last_rho: Arc<Mutex<Option<Array1<f64>>>>,
}

/// Number of outer evaluations the survival auto-subsample schedule
/// spends in Phase 1 before reverting to full data. Mirrors the BMS
/// budget so the two families share an empirical noise-floor schedule.
const SURVIVAL_MGS_AUTO_SUBSAMPLE_PHASE1_BUDGET: usize = 12;

/// Discriminates the two intercept slots per row: the entry-time intercept
/// (solved against `q0`) and the exit-time intercept (solved against `q1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurvivalInterceptSlotKind {
    Entry = 0,
    Exit = 1,
}

/// Per-row warm-start storage for the survival calibration root solver.
/// Two atomic `f64::to_bits` slots per row — one for the entry intercept
/// (`q = q0`) and one for the exit intercept (`q = q1`). NaN bits flag a
/// slot that hasn't been populated yet.
struct SurvivalInterceptWarmStartCache {
    entry: Vec<std::sync::atomic::AtomicU64>,
    exit: Vec<std::sync::atomic::AtomicU64>,
}

impl SurvivalInterceptWarmStartCache {
    #[inline]
    fn slots_for(&self, kind: SurvivalInterceptSlotKind) -> &[std::sync::atomic::AtomicU64] {
        match kind {
            SurvivalInterceptSlotKind::Entry => &self.entry,
            SurvivalInterceptSlotKind::Exit => &self.exit,
        }
    }

    #[inline]
    fn load(&self, row: usize, kind: SurvivalInterceptSlotKind) -> Option<f64> {
        let slot = self.slots_for(kind).get(row)?;
        let value = f64::from_bits(slot.load(std::sync::atomic::Ordering::Relaxed));
        value.is_finite().then_some(value)
    }

    #[inline]
    fn store(&self, row: usize, kind: SurvivalInterceptSlotKind, a: f64) {
        if let Some(slot) = self.slots_for(kind).get(row) {
            slot.store(a.to_bits(), std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn new_intercept_warm_start_cache(n: usize) -> Arc<SurvivalInterceptWarmStartCache> {
    Arc::new(SurvivalInterceptWarmStartCache {
        entry: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
        exit: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
    })
}

#[derive(Clone, Default)]
struct ThetaHints {
    time_beta: Option<Array1<f64>>,
    marginal_beta: Option<Array1<f64>>,
    logslope_beta: Option<Array1<f64>>,
    score_warp_beta: Option<Array1<f64>>,
    link_dev_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
struct PerZScoreWarpPrepared {
    block: ParameterBlockInput,
    runtime: DeviationRuntime,
    score_dim: usize,
}

impl PerZScoreWarpPrepared {
    #[inline]
    fn basis_dim(&self) -> usize {
        self.runtime.basis_dim()
    }

    #[inline]
    fn total_basis_dim(&self) -> usize {
        self.basis_dim() * self.score_dim
    }
}

fn score_warp_component_range(runtime: &DeviationRuntime, coord: usize) -> std::ops::Range<usize> {
    let p = runtime.basis_dim();
    coord * p..(coord + 1) * p
}

fn score_warp_component_beta<'a>(
    runtime: &DeviationRuntime,
    beta: &'a Array1<f64>,
    coord: usize,
) -> Result<Array1<f64>, String> {
    let range = score_warp_component_range(runtime, coord);
    if range.end > beta.len() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival score-warp coefficient block is too short for z coordinate {coord}: need {}, got {}",
                range.end,
                beta.len()
            ),
        }
        .into());
    }
    Ok(beta.slice(s![range]).to_owned())
}

fn build_per_z_score_warp_deviation_block_from_seed(
    z: &Array2<f64>,
    cfg: &DeviationBlockConfig,
) -> Result<PerZScoreWarpPrepared, String> {
    let score_dim = z.ncols();
    if score_dim == 0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival score-warp requires at least one z coordinate".to_string(),
        }
        .into());
    }
    let z_primary = z.column(0).to_owned();
    let base = build_score_warp_deviation_block_from_seed(&z_primary, cfg)?;
    if score_dim == 1 {
        return Ok(PerZScoreWarpPrepared {
            block: base.block,
            runtime: base.runtime,
            score_dim,
        });
    }

    // Vector-z score warp is the direct sum of K scalar warp spaces:
    //
    //     h_i = sum_{k=1}^K W_k(z_{ik}) beta_k .
    //
    // The coefficient vector is laid out as [beta_1 | ... | beta_K],
    // and the row design is the horizontal concatenation of the K scalar
    // designs.  Penalties are block-local on each coordinate slice, giving
    // each W_k its own smoothing parameter unless a later grouping layer
    // intentionally ties precision labels.  When K=1, the direct sum has
    // one component and the function has already returned the exact legacy
    // scalar block above.
    let p = base.runtime.basis_dim();
    let n = z.nrows();
    let mut design = Array2::<f64>::zeros((n, p * score_dim));
    for coord in 0..score_dim {
        let z_coord = z.column(coord).to_owned();
        let coord_design = base.runtime.design(&z_coord)?;
        if coord_design.nrows() != n || coord_design.ncols() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp design shape mismatch for z coordinate {coord}: got {}x{}, expected {n}x{p}",
                    coord_design.nrows(),
                    coord_design.ncols()
                ),
            }
            .into());
        }
        design
            .slice_mut(s![.., coord * p..(coord + 1) * p])
            .assign(&coord_design);
    }

    let mut block = base.block.clone();
    block.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design));
    block.offset = Array1::zeros(n);
    block.initial_beta = Some(Array1::zeros(p * score_dim));
    block.initial_log_lambdas = None;
    let base_penalties = base.block.penalties.clone();
    let base_nullspaces = base.block.nullspace_dims.clone();
    block.penalties.clear();
    block.nullspace_dims.clear();
    for coord in 0..score_dim {
        let col_range = coord * p..(coord + 1) * p;
        for (penalty_idx, penalty) in base_penalties.iter().enumerate() {
            let local = match penalty {
                crate::solver::estimate::PenaltySpec::Dense(matrix)
                | crate::solver::estimate::PenaltySpec::DenseWithMean { matrix, .. } => {
                    matrix.clone()
                }
                crate::solver::estimate::PenaltySpec::Block { local, .. } => local.clone(),
            };
            block
                .penalties
                .push(crate::solver::estimate::PenaltySpec::Block {
                    local,
                    col_range: col_range.clone(),
                    prior_mean: crate::estimate::CoefficientPriorMean::Zero,
                    structure_hint: None,
                    op: None,
                });
            block.nullspace_dims.push(
                base_nullspaces
                    .get(penalty_idx)
                    .copied()
                    .unwrap_or_default(),
            );
        }
    }

    Ok(PerZScoreWarpPrepared {
        block,
        runtime: base.runtime,
        score_dim,
    })
}

fn build_per_z_score_warp_aux_blockspec(
    prepared: &PerZScoreWarpPrepared,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> Result<ParameterBlockSpec, String> {
    let mut block = prepared.block.clone();
    block.initial_log_lambdas = Some(rho);
    let total_p = prepared.total_basis_dim();
    let candidate = beta_hint.unwrap_or_else(|| Array1::<f64>::zeros(total_p));
    if candidate.len() != total_p {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival score-warp beta hint length mismatch: got {}, expected {total_p}",
                candidate.len()
            ),
        }
        .into());
    }
    let mut projected = Array1::<f64>::zeros(total_p);
    for coord in 0..prepared.score_dim {
        let range = score_warp_component_range(&prepared.runtime, coord);
        let proposed = candidate.slice(s![range.clone()]).to_owned();
        let zero = Array1::<f64>::zeros(prepared.basis_dim());
        let local = project_monotone_feasible_beta(
            &prepared.runtime,
            &zero,
            &proposed,
            &format!("score_warp_dev[z{coord}]"),
        )?;
        projected.slice_mut(s![range]).assign(&local);
    }
    block.initial_beta = Some(projected);
    let mut spec = block.intospec("score_warp_dev")?;
    if prepared.score_dim > 1 {
        // The physical penalty order mirrors the direct-sum coefficient
        // layout: all penalties for beta_1, then beta_2, ..., beta_K.  Giving
        // each coordinate-local penalty a distinct precision label makes the
        // default MAP problem one-lambda-per-W_k; task-04 coefficient-group
        // penalties can still introduce intentionally shared precision
        // factors without accidentally tying these base smoothness penalties.
        if spec.penalties.len() % prepared.score_dim != 0 {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp penalty count {} is not divisible by K={}",
                    spec.penalties.len(),
                    prepared.score_dim
                ),
            }
            .into());
        }
        let penalties_per_coord = spec.penalties.len() / prepared.score_dim;
        for coord in 0..prepared.score_dim {
            for penalty_idx in 0..penalties_per_coord {
                let flat_idx = coord * penalties_per_coord + penalty_idx;
                let label = format!("score_warp_dev[z{coord}].penalty{penalty_idx}");
                spec.penalties[flat_idx] =
                    spec.penalties[flat_idx].clone().with_precision_label(label);
            }
        }
    }
    Ok(spec)
}

// ── Block layout ──────────────────────────────────────────────────────

#[derive(Clone)]
struct BlockSlices {
    time: std::ops::Range<usize>,
    marginal: std::ops::Range<usize>,
    logslope: std::ops::Range<usize>,
    score_warp: Option<std::ops::Range<usize>>,
    link_dev: Option<std::ops::Range<usize>>,
    total: usize,
}

fn block_slices(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
) -> BlockSlices {
    if !block_states.is_empty() {
        let expected_blocks =
            3 + usize::from(family.score_warp.is_some()) + usize::from(family.link_dev.is_some());
        assert_eq!(
            block_states.len(),
            expected_blocks,
            "survival marginal-slope block layout mismatch: expected {expected_blocks} blocks, got {}",
            block_states.len()
        );
    }
    let time = 0..family.design_entry.ncols();
    let marginal = time.end..time.end + family.marginal_design.ncols();
    let logslope = marginal.end..marginal.end + family.logslope_design.ncols();
    let mut cursor = logslope.end;
    let score_warp = family.score_warp.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim() * family.score_dim();
        cursor = range.end;
        range
    });
    let link_dev = family.link_dev.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    let total = cursor;
    BlockSlices {
        time,
        marginal,
        logslope,
        score_warp,
        link_dev,
        total,
    }
}

// ── Primary-space helpers ─────────────────────────────────────────────

// Primary scalar indices: 0=q0, 1=q1, 2=qd1, 3=g
const N_PRIMARY: usize = 4;

#[derive(Clone)]
struct FlexPrimarySlices {
    q0: usize,
    q1: usize,
    qd1: usize,
    g: usize,
    h: Option<std::ops::Range<usize>>,
    w: Option<std::ops::Range<usize>>,
    total: usize,
}

fn flex_primary_slices(family: &SurvivalMarginalSlopeFamily) -> FlexPrimarySlices {
    let q0 = 0usize;
    let q1 = 1usize;
    let qd1 = 2usize;
    let g = 3usize;
    let mut cursor = 4usize;
    let h = family.score_warp.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim() * family.score_dim();
        cursor = range.end;
        range
    });
    let w = family.link_dev.as_ref().map(|runtime| {
        let range = cursor..cursor + runtime.basis_dim();
        cursor = range.end;
        range
    });
    FlexPrimarySlices {
        q0,
        q1,
        qd1,
        g,
        h,
        w,
        total: cursor,
    }
}

fn flex_identity_block_pairs(
    primary: &FlexPrimarySlices,
    slices: &BlockSlices,
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let mut pairs = Vec::with_capacity(2);
    if let (Some(primary_range), Some(block_range)) =
        (primary.h.as_ref(), slices.score_warp.as_ref())
    {
        pairs.push((primary_range.clone(), block_range.clone()));
    }
    if let (Some(primary_range), Some(block_range)) = (primary.w.as_ref(), slices.link_dev.as_ref())
    {
        pairs.push((primary_range.clone(), block_range.clone()));
    }
    pairs
}

#[derive(Clone)]
struct DynamicQBlockwiseAccumulator {
    log_likelihood: f64,
    grad_time: Array1<f64>,
    grad_marginal: Array1<f64>,
    grad_logslope: Array1<f64>,
    hess_time: Array2<f64>,
    hess_marginal: Array2<f64>,
    hess_logslope: Array2<f64>,
    grad_score_warp: Option<Array1<f64>>,
    hess_score_warp: Option<Array2<f64>>,
    grad_link_dev: Option<Array1<f64>>,
    hess_link_dev: Option<Array2<f64>>,
}

#[derive(Clone)]
struct DynamicQCoreHessianBlocks {
    hess_time: Array2<f64>,
    hess_marginal: Array2<f64>,
    hess_logslope: Array2<f64>,
    hess_time_marginal: Array2<f64>,
    hess_time_logslope: Array2<f64>,
    hess_marginal_logslope: Array2<f64>,
}

impl DynamicQBlockwiseAccumulator {
    fn new(slices: &BlockSlices) -> Self {
        Self {
            log_likelihood: 0.0,
            grad_time: Array1::zeros(slices.time.len()),
            grad_marginal: Array1::zeros(slices.marginal.len()),
            grad_logslope: Array1::zeros(slices.logslope.len()),
            hess_time: Array2::zeros((slices.time.len(), slices.time.len())),
            hess_marginal: Array2::zeros((slices.marginal.len(), slices.marginal.len())),
            hess_logslope: Array2::zeros((slices.logslope.len(), slices.logslope.len())),
            grad_score_warp: slices
                .score_warp
                .as_ref()
                .map(|range| Array1::zeros(range.len())),
            hess_score_warp: slices
                .score_warp
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
            grad_link_dev: slices
                .link_dev
                .as_ref()
                .map(|range| Array1::zeros(range.len())),
            hess_link_dev: slices
                .link_dev
                .as_ref()
                .map(|range| Array2::zeros((range.len(), range.len()))),
        }
    }

    fn add_assign(&mut self, other: &Self) {
        self.log_likelihood += other.log_likelihood;
        self.grad_time += &other.grad_time;
        self.grad_marginal += &other.grad_marginal;
        self.grad_logslope += &other.grad_logslope;
        self.hess_time += &other.hess_time;
        self.hess_marginal += &other.hess_marginal;
        self.hess_logslope += &other.hess_logslope;
        add_optional_vector(&mut self.grad_score_warp, &other.grad_score_warp);
        add_optional_vector(&mut self.grad_link_dev, &other.grad_link_dev);
        add_optional_matrix(&mut self.hess_score_warp, &other.hess_score_warp);
        add_optional_matrix(&mut self.hess_link_dev, &other.hess_link_dev);
    }

    fn into_family_evaluation(self) -> FamilyEvaluation {
        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: self.grad_time,
                hessian: SymmetricMatrix::Dense(self.hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: self.grad_marginal,
                hessian: SymmetricMatrix::Dense(self.hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: self.grad_logslope,
                hessian: SymmetricMatrix::Dense(self.hess_logslope),
            },
        ];
        if let (Some(gradient), Some(hessian)) = (self.grad_score_warp, self.hess_score_warp) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        if let (Some(gradient), Some(hessian)) = (self.grad_link_dev, self.hess_link_dev) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        FamilyEvaluation {
            log_likelihood: self.log_likelihood,
            blockworking_sets,
        }
    }
}

struct DenestedCellPrimaryFixedPartials {
    dc_da: [f64; 4],
    dc_daa: [f64; 4],
    dc_daaa: [f64; 4],
    coeff_u: Vec<[f64; 4]>,
    coeff_au: Vec<[f64; 4]>,
    coeff_bu: Vec<[f64; 4]>,
    coeff_aau: Vec<[f64; 4]>,
    coeff_abu: Vec<[f64; 4]>,
    coeff_bbu: Vec<[f64; 4]>,
    coeff_aaau: Vec<[f64; 4]>,
    coeff_aabu: Vec<[f64; 4]>,
    coeff_abbu: Vec<[f64; 4]>,
    coeff_bbbu: Vec<[f64; 4]>,
}

const COEFF_SUPPORT_GHW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: true,
    include_w: true,
};
const COEFF_SUPPORT_GW: CoeffSupport = CoeffSupport {
    include_primary: true,
    include_h: false,
    include_w: true,
};

/// Pre-computed partition cell data for a single timepoint evaluation.
/// Built once per (a, b, β_h, β_w) and reused across the three passes
/// (F, D, D_uv) that previously each rebuilt partition cells independently.
struct CachedPartitionCells {
    cells: Vec<CachedCellEntry>,
}

struct CachedCellEntry {
    partition_cell: exact_kernel::DenestedPartitionCell,
    neg_cell: exact_kernel::DenestedCubicCell,
    state: exact_kernel::CellMomentState,
    fixed: DenestedCellPrimaryFixedPartials,
}

struct SurvivalFlexTimepointExact {
    eta: f64,
    chi: f64,
    d: f64,
    eta_u: Array1<f64>,
    eta_uv: Array2<f64>,
    chi_u: Array1<f64>,
    chi_uv: Array2<f64>,
    d_u: Array1<f64>,
    d_uv: Array2<f64>,
}

struct SurvivalFlexTimepointFirstOrderExact {
    eta: f64,
    chi: f64,
    d: f64,
    eta_u: Array1<f64>,
    chi_u: Array1<f64>,
    d_u: Array1<f64>,
}

/// Directional extensions of a timepoint's exact quantities, contracted with
/// a single direction.  These are the pieces needed to compose the third-order
/// NLL contraction.
struct SurvivalFlexTimepointDirectionalExact {
    eta_uv_dir: Array2<f64>,
    chi_uv_dir: Array2<f64>,
    d_u_dir: Array1<f64>,
    d_uv_dir: Array2<f64>,
}

struct SurvivalFlexTimepointBiDirectionalExact {
    eta_uv_uv: Array2<f64>,
    chi_uv_uv: Array2<f64>,
    d_uv_uv: Array2<f64>,
}

#[derive(Clone)]
struct SurvivalTimeWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    basis_d3: Array2<f64>,
    basis_d4: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
    d4q_dq04: Array1<f64>,
    d5q_dq05: Array1<f64>,
}

#[derive(Clone)]
struct SurvivalTimeWiggleFirstOrderGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
}

#[derive(Clone)]
struct SurvivalMarginalSlopeDynamicRowValues {
    q0: f64,
    q1: f64,
    qd1: f64,
}

#[derive(Clone)]
struct SurvivalMarginalSlopeDynamicRowGradient {
    q0: f64,
    q1: f64,
    qd1: f64,
    dq0_time: Array1<f64>,
    dq1_time: Array1<f64>,
    dqd1_time: Array1<f64>,
    dq0_marginal: Array1<f64>,
    dq1_marginal: Array1<f64>,
    dqd1_marginal: Array1<f64>,
}

#[derive(Clone)]
struct SurvivalMarginalSlopeDynamicRow {
    q0: f64,
    q1: f64,
    qd1: f64,
    dq0_time: Array1<f64>,
    dq1_time: Array1<f64>,
    dqd1_time: Array1<f64>,
    dq0_marginal: Array1<f64>,
    dq1_marginal: Array1<f64>,
    dqd1_marginal: Array1<f64>,
    d2q0_time_time: Array2<f64>,
    d2q1_time_time: Array2<f64>,
    d2qd1_time_time: Array2<f64>,
    d2q0_time_marginal: Array2<f64>,
    d2q1_time_marginal: Array2<f64>,
    d2qd1_time_marginal: Array2<f64>,
    d2q0_marginal_marginal: Array2<f64>,
    d2q1_marginal_marginal: Array2<f64>,
    d2qd1_marginal_marginal: Array2<f64>,
}

impl SurvivalMarginalSlopeDynamicRow {
    /// Construct a zero-sized workspace. Sizes are filled in lazily by
    /// [`reset`] on the first call to [`row_dynamic_q_geometry_into`].
    fn empty_workspace() -> Self {
        Self {
            q0: 0.0,
            q1: 0.0,
            qd1: 0.0,
            dq0_time: Array1::zeros(0),
            dq1_time: Array1::zeros(0),
            dqd1_time: Array1::zeros(0),
            dq0_marginal: Array1::zeros(0),
            dq1_marginal: Array1::zeros(0),
            dqd1_marginal: Array1::zeros(0),
            d2q0_time_time: Array2::zeros((0, 0)),
            d2q1_time_time: Array2::zeros((0, 0)),
            d2qd1_time_time: Array2::zeros((0, 0)),
            d2q0_time_marginal: Array2::zeros((0, 0)),
            d2q1_time_marginal: Array2::zeros((0, 0)),
            d2qd1_time_marginal: Array2::zeros((0, 0)),
            d2q0_marginal_marginal: Array2::zeros((0, 0)),
            d2q1_marginal_marginal: Array2::zeros((0, 0)),
            d2qd1_marginal_marginal: Array2::zeros((0, 0)),
        }
    }

    /// Resize buffers to `(p_time, p_marginal)` and zero them in place.
    /// Reallocates only when the existing buffer shape differs from the
    /// requested shape; otherwise reuses the existing storage with
    /// `fill(0.0)` to keep the per-row allocator pressure flat.
    fn reset(&mut self, p_time: usize, p_marginal: usize) {
        self.q0 = 0.0;
        self.q1 = 0.0;
        self.qd1 = 0.0;
        reset_array1(&mut self.dq0_time, p_time);
        reset_array1(&mut self.dq1_time, p_time);
        reset_array1(&mut self.dqd1_time, p_time);
        reset_array1(&mut self.dq0_marginal, p_marginal);
        reset_array1(&mut self.dq1_marginal, p_marginal);
        reset_array1(&mut self.dqd1_marginal, p_marginal);
        reset_array2(&mut self.d2q0_time_time, p_time, p_time);
        reset_array2(&mut self.d2q1_time_time, p_time, p_time);
        reset_array2(&mut self.d2qd1_time_time, p_time, p_time);
        reset_array2(&mut self.d2q0_time_marginal, p_time, p_marginal);
        reset_array2(&mut self.d2q1_time_marginal, p_time, p_marginal);
        reset_array2(&mut self.d2qd1_time_marginal, p_time, p_marginal);
        reset_array2(&mut self.d2q0_marginal_marginal, p_marginal, p_marginal);
        reset_array2(&mut self.d2q1_marginal_marginal, p_marginal, p_marginal);
        reset_array2(&mut self.d2qd1_marginal_marginal, p_marginal, p_marginal);
    }
}

#[inline]
fn reset_array1(arr: &mut Array1<f64>, len: usize) {
    if arr.len() == len {
        arr.fill(0.0);
    } else {
        *arr = Array1::zeros(len);
    }
}

#[inline]
fn reset_array2(arr: &mut Array2<f64>, rows: usize, cols: usize) {
    if arr.shape() == [rows, cols] {
        arr.fill(0.0);
    } else {
        *arr = Array2::zeros((rows, cols));
    }
}

struct TimewiggleMarginalPsiRowLift {
    dir: Array1<f64>,
    u_q0_time: Array1<f64>,
    u_q1_time: Array1<f64>,
    u_qd1_time: Array1<f64>,
    u_q0_marginal: Array1<f64>,
    u_q1_marginal: Array1<f64>,
    u_qd1_marginal: Array1<f64>,
    x_entry_base: Array1<f64>,
    x_exit_base: Array1<f64>,
    x_deriv_base: Array1<f64>,
    marginal_row: Array1<f64>,
    entry_basis_d1: Array1<f64>,
    entry_basis_d2: Array1<f64>,
    exit_basis_d1: Array1<f64>,
    exit_basis_d2: Array1<f64>,
    exit_basis_d3: Array1<f64>,
    entry_m2: f64,
    entry_m3: f64,
    exit_m2: f64,
    exit_m3: f64,
    exit_m4: f64,
    d_raw: f64,
    mu: f64,
    psi_row: Array1<f64>,
}

fn unit_primary_direction(idx: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    out[idx] = 1.0;
    out
}

/// Returns a reference to the static unit-direction table (one Array1<f64>
/// per primary axis). Reusing these references avoids per-row heap
/// allocations in third/fourth-order contracted assemblies, where the inner
/// loop calls into the row-directional kernel 10 (third-order) or 10
/// (fourth-order) times per row.
fn unit_primary_direction_table() -> &'static [Array1<f64>; N_PRIMARY] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[Array1<f64>; N_PRIMARY]> = OnceLock::new();
    TABLE.get_or_init(|| {
        [
            unit_primary_direction(0),
            unit_primary_direction(1),
            unit_primary_direction(2),
            unit_primary_direction(3),
        ]
    })
}

#[inline]
fn unit_primary_direction_ref(idx: usize) -> &'static Array1<f64> {
    &unit_primary_direction_table()[idx]
}

/// Returns a reference to the static zero direction in primary space
/// (an `Array1::zeros(N_PRIMARY)`). Used by sigma-jet contractions to
/// avoid the per-call `Array1::zeros(primary_dim)` allocation storm in
/// `row_sigma_primary_terms`, which previously allocated 2-4 fresh zero
/// slots per kernel invocation and ~30 zero slots per row.
#[inline]
fn zero_primary_direction_ref() -> &'static Array1<f64> {
    use std::sync::OnceLock;
    static ZERO: OnceLock<Array1<f64>> = OnceLock::new();
    ZERO.get_or_init(|| Array1::<f64>::zeros(N_PRIMARY))
}

#[inline]
fn poly_mul(lhs: &[f64], rhs: &[f64]) -> PolyVec {
    if lhs.is_empty() || rhs.is_empty() {
        return PolyVec::new();
    }
    let n = lhs.len() + rhs.len() - 1;
    let mut out: PolyVec = smallvec![0.0; n];
    // Inner loop: out[i..i+rhs.len()] += lv * rhs. One sub-slicing bounds
    // check per outer iter lets LLVM elide the inner-loop bounds checks and
    // auto-vectorize the FMA. Hot path: ~13% self-time across poly_mul +
    // poly_add + poly_scale per the smoke-fixture profile.
    let out_slice = out.as_mut_slice();
    for (i, &lv) in lhs.iter().enumerate() {
        let dst = &mut out_slice[i..i + rhs.len()];
        for (d, &rv) in dst.iter_mut().zip(rhs.iter()) {
            *d += lv * rv;
        }
    }
    out
}

#[inline]
fn poly_sub(lhs: &[f64], rhs: &[f64]) -> PolyVec {
    let mut out: PolyVec = SmallVec::new();
    out.extend_from_slice(lhs);
    if rhs.len() > lhs.len() {
        out.resize(rhs.len(), 0.0);
    }
    for (d, &v) in out[..rhs.len()].iter_mut().zip(rhs.iter()) {
        *d -= v;
    }
    out
}

#[inline]
fn poly_add(lhs: &[f64], rhs: &[f64]) -> PolyVec {
    // Copy the longer operand verbatim, then add the shorter onto its
    // prefix. Avoids the redundant zero-fill of `smallvec![0.0; n]` plus
    // two additive passes from the legacy implementation.
    let (a, b) = if lhs.len() >= rhs.len() {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let mut out: PolyVec = SmallVec::new();
    out.extend_from_slice(a);
    for (d, &v) in out[..b.len()].iter_mut().zip(b.iter()) {
        *d += v;
    }
    out
}

#[inline]
fn poly_scale(poly: &[f64], scale: f64) -> PolyVec {
    let mut out: PolyVec = SmallVec::with_capacity(poly.len());
    for &v in poly {
        out.push(scale * v);
    }
    out
}

fn spatial_block_primary_loading(block_idx: usize) -> Result<Array1<f64>, String> {
    match block_idx {
        1 => Ok(Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0])),
        2 => Ok(Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0])),
        _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival marginal-slope spatial psi loading requested for unsupported block {block_idx}"
            ),
        }
        .into()),
    }
}

use crate::families::jet_partitions::MultiDirJet;

fn scalar_composite_bilinear(
    base: f64,
    da: f64,
    daa: f64,
    fixed_d1: f64,
    fixed_d2: f64,
    fixed_d12: f64,
    da_d1: f64,
    da_d2: f64,
    ad1: f64,
    ad2: f64,
    ad12: f64,
) -> MultiDirJet {
    MultiDirJet::bilinear(
        base,
        da * ad1 + fixed_d1,
        da * ad2 + fixed_d2,
        da * ad12 + daa * ad1 * ad2 + da_d1 * ad2 + da_d2 * ad1 + fixed_d12,
    )
}

fn coeff4_fixed_bilinear(
    base: &[f64; 4],
    d1: &[f64; 4],
    d2: &[f64; 4],
    d12: &[f64; 4],
) -> Vec<MultiDirJet> {
    (0..4)
        .map(|k| MultiDirJet::bilinear(base[k], d1[k], d2[k], d12[k]))
        .collect()
}

fn coeff4_composite_bilinear(
    base: &[f64; 4],
    da: &[f64; 4],
    daa: &[f64; 4],
    fixed_d1: &[f64; 4],
    fixed_d2: &[f64; 4],
    fixed_d12: &[f64; 4],
    da_d1: &[f64; 4],
    da_d2: &[f64; 4],
    ad1: f64,
    ad2: f64,
    ad12: f64,
) -> Vec<MultiDirJet> {
    (0..4)
        .map(|k| {
            scalar_composite_bilinear(
                base[k],
                da[k],
                daa[k],
                fixed_d1[k],
                fixed_d2[k],
                fixed_d12[k],
                da_d1[k],
                da_d2[k],
                ad1,
                ad2,
                ad12,
            )
        })
        .collect()
}

fn poly_add_jets(lhs: &[MultiDirJet], rhs: &[MultiDirJet]) -> Vec<MultiDirJet> {
    let count = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(count);
    for idx in 0..count {
        let left = lhs
            .get(idx)
            .cloned()
            .unwrap_or_else(|| MultiDirJet::zero(2));
        let right = rhs
            .get(idx)
            .cloned()
            .unwrap_or_else(|| MultiDirJet::zero(2));
        out.push(left.add(&right));
    }
    out
}

fn poly_scale_jets(poly: &[MultiDirJet], scale: &MultiDirJet) -> Vec<MultiDirJet> {
    poly.iter().map(|coeff| coeff.mul(scale)).collect()
}

fn poly_mul_jets(lhs: &[MultiDirJet], rhs: &[MultiDirJet]) -> Vec<MultiDirJet> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    let mut out = vec![MultiDirJet::zero(2); lhs.len() + rhs.len() - 1];
    for (i, left) in lhs.iter().enumerate() {
        for (j, right) in rhs.iter().enumerate() {
            let prod = left.mul(right);
            out[i + j] = out[i + j].add(&prod);
        }
    }
    out
}

fn poly_coeff_mask(poly: &[MultiDirJet], mask: usize) -> Vec<f64> {
    poly.iter().map(|coeff| coeff.coeff(mask)).collect()
}

/// Derive a primary-space direction from a precomputed psi design row and beta,
/// avoiding a redundant psi design row build inside `row_primary_psi_direction`.
fn primary_direction_from_psi_row(
    block_idx: usize,
    psi_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    let value = psi_row.dot(beta_block);
    match block_idx {
        1 => {
            out[0] = value;
            out[1] = value;
        }
        2 => {
            out[3] = value;
        }
        _ => {}
    }
    out
}

fn spatial_block_primary_loading_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
) -> Result<Array1<f64>, String> {
    let mut out = Array1::<f64>::zeros(primary.total);
    match block_idx {
        1 => {
            out[primary.q0] = 1.0;
            out[primary.q1] = 1.0;
            Ok(out)
        }
        2 => {
            out[primary.g] = 1.0;
            Ok(out)
        }
        _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival marginal-slope spatial psi loading requested for unsupported flex block {block_idx}"
            ),
        }
        .into()),
    }
}

fn primary_direction_from_psi_row_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
    psi_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(primary.total);
    let value = psi_row.dot(beta_block);
    match block_idx {
        1 => {
            out[primary.q0] = value;
            out[primary.q1] = value;
        }
        2 => {
            out[primary.g] = value;
        }
        _ => {}
    }
    out
}

/// Derive a primary-space psi action on a direction from a precomputed psi design row.
fn primary_psi_action_from_psi_row(
    block_idx: usize,
    psi_row: &Array1<f64>,
    d_beta_block: ndarray::ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(N_PRIMARY);
    let value = psi_row.dot(&d_beta_block);
    match block_idx {
        1 => {
            out[0] = value;
            out[1] = value;
        }
        2 => {
            out[3] = value;
        }
        _ => {}
    }
    out
}

fn primary_psi_action_from_psi_row_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
    psi_row: &Array1<f64>,
    d_beta_block: ndarray::ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(primary.total);
    let value = psi_row.dot(&d_beta_block);
    match block_idx {
        1 => {
            out[primary.q0] = value;
            out[primary.q1] = value;
        }
        2 => {
            out[primary.g] = value;
        }
        _ => {}
    }
    out
}

/// Derive a primary-space second-order direction from a precomputed second psi design row.
fn primary_second_direction_from_psi_row(
    block_idx: usize,
    psi_second_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    primary_direction_from_psi_row(block_idx, psi_second_row, beta_block)
}

fn primary_second_direction_from_psi_row_flex(
    primary: &FlexPrimarySlices,
    block_idx: usize,
    psi_second_row: &Array1<f64>,
    beta_block: &Array1<f64>,
) -> Array1<f64> {
    primary_direction_from_psi_row_flex(primary, block_idx, psi_second_row, beta_block)
}

// ── Block-local Hessian accumulator ────────────────────────────────────
//
// Avoids O(n p²) per-row allocation of full p×p matrices by accumulating
// the 6 independent block matrices (3 diagonal + 3 off-diagonal) directly.
// Assembly to a dense p×p matrix or an implicit operator is a single O(p²)
// pass at the end, after the n-loop.

struct BlockHessianAccumulator {
    h_tt: Array2<f64>,
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_hh: Array2<f64>,
    h_ww: Array2<f64>,
    h_tm: Array2<f64>,
    h_tg: Array2<f64>,
    h_th: Array2<f64>,
    h_tw: Array2<f64>,
    h_mg: Array2<f64>,
    h_mh: Array2<f64>,
    h_mw: Array2<f64>,
    h_gh: Array2<f64>,
    h_gw: Array2<f64>,
    h_hw: Array2<f64>,
}

const PULLBACK_PARALLEL_MIN_CELLS: usize = 16_384;
const PULLBACK_PARALLEL_TARGET_CELLS: usize = 65_536;

impl BlockHessianAccumulator {
    fn new(p_t: usize, p_m: usize, p_g: usize, p_h: usize, p_w: usize) -> Self {
        Self {
            h_tt: Array2::zeros((p_t, p_t)),
            h_mm: Array2::zeros((p_m, p_m)),
            h_gg: Array2::zeros((p_g, p_g)),
            h_hh: Array2::zeros((p_h, p_h)),
            h_ww: Array2::zeros((p_w, p_w)),
            h_tm: Array2::zeros((p_t, p_m)),
            h_tg: Array2::zeros((p_t, p_g)),
            h_th: Array2::zeros((p_t, p_h)),
            h_tw: Array2::zeros((p_t, p_w)),
            h_mg: Array2::zeros((p_m, p_g)),
            h_mh: Array2::zeros((p_m, p_h)),
            h_mw: Array2::zeros((p_m, p_w)),
            h_gh: Array2::zeros((p_g, p_h)),
            h_gw: Array2::zeros((p_g, p_w)),
            h_hw: Array2::zeros((p_h, p_w)),
        }
    }

    fn block_dims(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.h_tt.nrows(),
            self.h_mm.nrows(),
            self.h_gg.nrows(),
            self.h_hh.nrows(),
            self.h_ww.nrows(),
        )
    }

    fn deterministic_lhs_chunks(lhs_len: usize, rhs_len: usize) -> Vec<std::ops::Range<usize>> {
        let cells = lhs_len.saturating_mul(rhs_len);
        if cells < PULLBACK_PARALLEL_MIN_CELLS || lhs_len <= 1 || rayon::current_num_threads() <= 1
        {
            return std::iter::once(0..lhs_len).collect();
        }
        let chunk_len = (PULLBACK_PARALLEL_TARGET_CELLS / rhs_len.max(1))
            .max(1)
            .min(lhs_len);
        (0..lhs_len)
            .step_by(chunk_len)
            .map(|start| start..(start + chunk_len).min(lhs_len))
            .collect()
    }

    fn add_ordered_lhs_partials(
        target: &mut Array2<f64>,
        partials: Vec<Result<(std::ops::Range<usize>, Array2<f64>), String>>,
    ) -> Result<(), String> {
        for partial in partials {
            let (range, block) = partial?;
            target.slice_mut(s![range, ..]).scaled_add(1.0, &block);
        }
        Ok(())
    }

    /// Accumulate a primary-space Hessian into block-local matrices.
    /// Equivalent to `add_pullback_primary_hessian` but avoids the p×p target.
    fn add_pullback(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        primary_hessian: &Array2<f64>,
    ) -> Result<(), String> {
        // Time×time block: 3×3 design cross-products
        let time_designs = [
            &family.design_entry,
            &family.design_exit,
            &family.design_derivative_exit,
        ];
        let (p_t, _, _, _, _) = self.block_dims();
        let tt_chunks = Self::deterministic_lhs_chunks(p_t, p_t);
        if tt_chunks.len() == 1 {
            for a in 0..3 {
                for b in 0..3 {
                    time_designs[a]
                        .row_outer_into(
                            row,
                            time_designs[b],
                            primary_hessian[[a, b]],
                            &mut self.h_tt,
                        )
                        .map_err(|e| format!("add_pullback time row_outer_into: {e}"))?;
                }
            }
        } else {
            let time_rows: Vec<Array1<f64>> = time_designs
                .iter()
                .map(|des| {
                    let chunk = des
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                    Ok(chunk.row(0).to_owned())
                })
                .collect::<Result<_, String>>()?;
            let time_partials: Vec<Result<(std::ops::Range<usize>, Array2<f64>), String>> =
                tt_chunks
                    .into_par_iter()
                    .map(|chunk| {
                        let mut local = Array2::zeros((chunk.len(), p_t));
                        for (local_a, coeff_a) in chunk.clone().enumerate() {
                            for coeff_b in 0..p_t {
                                let mut value = 0.0;
                                for a in 0..3 {
                                    for b in 0..3 {
                                        value += primary_hessian[[a, b]]
                                            * time_rows[a][coeff_a]
                                            * time_rows[b][coeff_b];
                                    }
                                }
                                local[[local_a, coeff_b]] = value;
                            }
                        }
                        Ok((chunk, local))
                    })
                    .collect();
            Self::add_ordered_lhs_partials(&mut self.h_tt, time_partials)?;
        }

        // Marginal×marginal: single rank-1 with combined weight
        let mm_weight = primary_hessian[[0, 0]]
            + primary_hessian[[0, 1]]
            + primary_hessian[[1, 0]]
            + primary_hessian[[1, 1]];
        family
            .marginal_design
            .syr_row_into(row, mm_weight, &mut self.h_mm)
            .map_err(|e| format!("add_pullback marginal syr_row_into: {e}"))?;

        // Logslope×logslope: single rank-1
        family
            .logslope_design
            .syr_row_into(row, primary_hessian[[3, 3]], &mut self.h_gg)
            .map_err(|e| format!("add_pullback logslope syr_row_into: {e}"))?;

        // Marginal×logslope cross-block
        let mg_weight = primary_hessian[[0, 3]] + primary_hessian[[1, 3]];
        if mg_weight != 0.0 {
            let m_chunk = family
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
            let m_row = m_chunk.row(0);
            let g_chunk = family
                .logslope_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
            let g_row = g_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                mg_weight,
                &m_row.view().insert_axis(Axis(1)),
                &g_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_mg,
            );
        }

        // Time×logslope cross-block
        let tg_weights = [
            primary_hessian[[0, 3]],
            primary_hessian[[1, 3]],
            primary_hessian[[2, 3]],
        ];
        let g_chunk = family
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
        let g_row = g_chunk.row(0);
        for (des, alpha) in time_designs.iter().zip(tg_weights.iter()) {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
            let t_row = t_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                *alpha,
                &t_row.view().insert_axis(Axis(1)),
                &g_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_tg,
            );
        }

        // Time×marginal cross-block
        let tm_weights = [
            primary_hessian[[0, 0]] + primary_hessian[[0, 1]],
            primary_hessian[[1, 0]] + primary_hessian[[1, 1]],
            primary_hessian[[2, 0]] + primary_hessian[[2, 1]],
        ];
        let m_chunk = family
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
        let m_row = m_chunk.row(0);
        for (des, alpha) in time_designs.iter().zip(tm_weights.iter()) {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
            let t_row = t_chunk.row(0);
            ndarray::linalg::general_mat_mul(
                *alpha,
                &t_row.view().insert_axis(Axis(1)),
                &m_row.view().insert_axis(Axis(0)),
                1.0,
                &mut self.h_tm,
            );
        }

        let primary = flex_primary_slices(family);
        if let Some(h_range) = primary.h.as_ref() {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                let th_weights = [
                    primary_hessian[[0, idx]],
                    primary_hessian[[1, idx]],
                    primary_hessian[[2, idx]],
                ];
                for (des, alpha) in time_designs.iter().zip(th_weights.iter()) {
                    if *alpha == 0.0 {
                        continue;
                    }
                    let t_chunk = des
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                    let t_row = t_chunk.row(0);
                    for coeff_idx in 0..t_row.len() {
                        self.h_th[[coeff_idx, local_idx]] += *alpha * t_row[coeff_idx];
                    }
                }

                let mh_weight = primary_hessian[[0, idx]] + primary_hessian[[1, idx]];
                if mh_weight != 0.0 {
                    let m_chunk = family
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
                    let m_row = m_chunk.row(0);
                    for coeff_idx in 0..m_row.len() {
                        self.h_mh[[coeff_idx, local_idx]] += mh_weight * m_row[coeff_idx];
                    }
                }

                let gh_weight = primary_hessian[[3, idx]];
                if gh_weight != 0.0 {
                    let g_chunk = family
                        .logslope_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
                    let g_row = g_chunk.row(0);
                    for coeff_idx in 0..g_row.len() {
                        self.h_gh[[coeff_idx, local_idx]] += gh_weight * g_row[coeff_idx];
                    }
                }
            }

            for left_local in 0..h_range.len() {
                for right_local in 0..h_range.len() {
                    self.h_hh[[left_local, right_local]] +=
                        primary_hessian[[h_range.start + left_local, h_range.start + right_local]];
                }
            }
        }

        if let Some(w_range) = primary.w.as_ref() {
            for local_idx in 0..w_range.len() {
                let idx = w_range.start + local_idx;
                let tw_weights = [
                    primary_hessian[[0, idx]],
                    primary_hessian[[1, idx]],
                    primary_hessian[[2, idx]],
                ];
                for (des, alpha) in time_designs.iter().zip(tw_weights.iter()) {
                    if *alpha == 0.0 {
                        continue;
                    }
                    let t_chunk = des
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                    let t_row = t_chunk.row(0);
                    for coeff_idx in 0..t_row.len() {
                        self.h_tw[[coeff_idx, local_idx]] += *alpha * t_row[coeff_idx];
                    }
                }

                let mw_weight = primary_hessian[[0, idx]] + primary_hessian[[1, idx]];
                if mw_weight != 0.0 {
                    let m_chunk = family
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
                    let m_row = m_chunk.row(0);
                    for coeff_idx in 0..m_row.len() {
                        self.h_mw[[coeff_idx, local_idx]] += mw_weight * m_row[coeff_idx];
                    }
                }

                let gw_weight = primary_hessian[[3, idx]];
                if gw_weight != 0.0 {
                    let g_chunk = family
                        .logslope_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
                    let g_row = g_chunk.row(0);
                    for coeff_idx in 0..g_row.len() {
                        self.h_gw[[coeff_idx, local_idx]] += gw_weight * g_row[coeff_idx];
                    }
                }
            }

            for left_local in 0..w_range.len() {
                for right_local in 0..w_range.len() {
                    self.h_ww[[left_local, right_local]] +=
                        primary_hessian[[w_range.start + left_local, w_range.start + right_local]];
                }
            }
        }

        if let (Some(h_range), Some(w_range)) = (primary.h.as_ref(), primary.w.as_ref()) {
            for h_local in 0..h_range.len() {
                for w_local in 0..w_range.len() {
                    self.h_hw[[h_local, w_local]] +=
                        primary_hessian[[h_range.start + h_local, w_range.start + w_local]];
                }
            }
        }

        Ok(())
    }

    /// Add a rank-1 update from psi_row (in the psi block) crossed with the
    /// pullback of a primary-space vector. Adds both left⊗right and right⊗left.
    fn add_rank1_psi_cross(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        psi_block_idx: usize,
        psi_row: &Array1<f64>,
        right_primary: &Array1<f64>,
    ) -> Result<(), String> {
        // right_primary components mapped to blocks:
        // time:     entry*rp[0] + exit*rp[1] + deriv*rp[2]
        // marginal: marginal*(rp[0] + rp[1])
        // logslope: logslope*rp[3]
        let psi_col = psi_row.view().insert_axis(Axis(1));

        // Block (psi, time): psi_row ⊗ right_time
        // Block (time, psi): right_time ⊗ psi_row  (= transpose of above)
        let time_designs = [
            (&family.design_entry, right_primary[0]),
            (&family.design_exit, right_primary[1]),
            (&family.design_derivative_exit, right_primary[2]),
        ];
        for (des, alpha) in &time_designs {
            if *alpha == 0.0 {
                continue;
            }
            let t_chunk = des
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_rank1_psi_cross time design try_row_chunk: {e}"))?;
            let t_row = t_chunk.row(0);
            let t_col = t_row.view().insert_axis(Axis(1));
            match psi_block_idx {
                1 => {
                    // psi=marginal: (time, marginal) block = h_tm
                    // right⊗left: right_time ⊗ psi_row → h_tm
                    ndarray::linalg::general_mat_mul(
                        *alpha,
                        &t_col,
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_tm,
                    );
                    // left⊗right: psi_row ⊗ right_time → h_tm^T (handled by symmetry)
                }
                2 => {
                    // psi=logslope: (time, logslope) block = h_tg
                    ndarray::linalg::general_mat_mul(
                        *alpha,
                        &t_col,
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_tg,
                    );
                }
                _ => {}
            }
        }

        // Block (psi, marginal) or (marginal, psi)
        let m_alpha = right_primary[0] + right_primary[1];
        if m_alpha != 0.0 {
            let m_chunk = family
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_rank1_psi_cross marginal_design try_row_chunk: {e}"))?;
            let m_row = m_chunk.row(0);
            match psi_block_idx {
                1 => {
                    // psi=marginal: (marginal, marginal) = h_mm, symmetric rank-2
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &psi_col,
                        &m_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mm,
                    );
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &m_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mm,
                    );
                }
                2 => {
                    // psi=logslope: (marginal, logslope) block = h_mg
                    // left⊗right: psi_row(logslope) ⊗ m_row → goes to h_mg^T
                    // right⊗left: m_row ⊗ psi_row(logslope) → goes to h_mg
                    ndarray::linalg::general_mat_mul(
                        m_alpha,
                        &m_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mg,
                    );
                }
                _ => {}
            }
        }

        // Block (psi, logslope) or (logslope, psi)
        if right_primary[3] != 0.0 {
            let g_chunk = family
                .logslope_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("add_rank1_psi_cross logslope_design try_row_chunk: {e}"))?;
            let g_row = g_chunk.row(0);
            match psi_block_idx {
                1 => {
                    // psi=marginal: (marginal, logslope) = h_mg
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &psi_col,
                        &g_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_mg,
                    );
                }
                2 => {
                    // psi=logslope: (logslope, logslope) = h_gg, symmetric rank-2
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &psi_col,
                        &g_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_gg,
                    );
                    ndarray::linalg::general_mat_mul(
                        right_primary[3],
                        &g_row.view().insert_axis(Axis(1)),
                        &psi_row.view().insert_axis(Axis(0)),
                        1.0,
                        &mut self.h_gg,
                    );
                }
                _ => {}
            }
        }

        let primary = flex_primary_slices(family);
        if let Some(h_range) = primary.h.as_ref() {
            for local_idx in 0..h_range.len() {
                let alpha = right_primary[h_range.start + local_idx];
                if alpha == 0.0 {
                    continue;
                }
                match psi_block_idx {
                    1 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_mh[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    2 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_gh[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(w_range) = primary.w.as_ref() {
            for local_idx in 0..w_range.len() {
                let alpha = right_primary[w_range.start + local_idx];
                if alpha == 0.0 {
                    continue;
                }
                match psi_block_idx {
                    1 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_mw[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    2 => {
                        for coeff_idx in 0..psi_row.len() {
                            self.h_gw[[coeff_idx, local_idx]] += alpha * psi_row[coeff_idx];
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Add outer product of two psi block-local rows (possibly in different blocks).
    /// Adds both α·(a ⊗ b) and α·(b ⊗ a) to maintain symmetry.
    ///
    /// The full p×p symmetric Hessian has blocks:
    ///   (block_i, block_j) += α · psi_row_i ⊗ psi_row_j
    ///   (block_j, block_i) += α · psi_row_j ⊗ psi_row_i   (= transpose)
    /// Our off-diagonal storage convention (h_mg = marginal×logslope) handles
    /// the transpose automatically in to_dense/operator assembly.
    fn add_psi_psi_outer(
        &mut self,
        block_i: usize,
        psi_row_i: &Array1<f64>,
        block_j: usize,
        psi_row_j: &Array1<f64>,
        alpha: f64,
    ) {
        add_two_surface_psi_outer(
            block_i,
            psi_row_i,
            block_j,
            psi_row_j,
            alpha,
            1,
            2,
            &mut self.h_mm,
            &mut self.h_gg,
            &mut self.h_mg,
        );
    }

    /// Assemble into a dense p×p matrix.
    fn to_dense(&self, slices: &BlockSlices) -> Array2<f64> {
        let mut out = Array2::zeros((slices.total, slices.total));
        out.slice_mut(s![slices.time.clone(), slices.time.clone()])
            .assign(&self.h_tt);
        out.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()])
            .assign(&self.h_mm);
        out.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()])
            .assign(&self.h_gg);
        if let Some(range) = slices.score_warp.as_ref() {
            out.slice_mut(s![range.clone(), range.clone()])
                .assign(&self.h_hh);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            out.slice_mut(s![range.clone(), range.clone()])
                .assign(&self.h_ww);
        }
        out.slice_mut(s![slices.time.clone(), slices.marginal.clone()])
            .assign(&self.h_tm);
        out.slice_mut(s![slices.marginal.clone(), slices.time.clone()])
            .assign(&self.h_tm.t());
        out.slice_mut(s![slices.time.clone(), slices.logslope.clone()])
            .assign(&self.h_tg);
        out.slice_mut(s![slices.logslope.clone(), slices.time.clone()])
            .assign(&self.h_tg.t());
        out.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()])
            .assign(&self.h_mg);
        out.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()])
            .assign(&self.h_mg.t());
        if let Some(range) = slices.score_warp.as_ref() {
            out.slice_mut(s![slices.time.clone(), range.clone()])
                .assign(&self.h_th);
            out.slice_mut(s![range.clone(), slices.time.clone()])
                .assign(&self.h_th.t());
            out.slice_mut(s![slices.marginal.clone(), range.clone()])
                .assign(&self.h_mh);
            out.slice_mut(s![range.clone(), slices.marginal.clone()])
                .assign(&self.h_mh.t());
            out.slice_mut(s![slices.logslope.clone(), range.clone()])
                .assign(&self.h_gh);
            out.slice_mut(s![range.clone(), slices.logslope.clone()])
                .assign(&self.h_gh.t());
        }
        if let Some(range) = slices.link_dev.as_ref() {
            out.slice_mut(s![slices.time.clone(), range.clone()])
                .assign(&self.h_tw);
            out.slice_mut(s![range.clone(), slices.time.clone()])
                .assign(&self.h_tw.t());
            out.slice_mut(s![slices.marginal.clone(), range.clone()])
                .assign(&self.h_mw);
            out.slice_mut(s![range.clone(), slices.marginal.clone()])
                .assign(&self.h_mw.t());
            out.slice_mut(s![slices.logslope.clone(), range.clone()])
                .assign(&self.h_gw);
            out.slice_mut(s![range.clone(), slices.logslope.clone()])
                .assign(&self.h_gw.t());
        }
        if let (Some(h_range), Some(w_range)) =
            (slices.score_warp.as_ref(), slices.link_dev.as_ref())
        {
            out.slice_mut(s![h_range.clone(), w_range.clone()])
                .assign(&self.h_hw);
            out.slice_mut(s![w_range.clone(), h_range.clone()])
                .assign(&self.h_hw.t());
        }
        out
    }

    fn into_operator(self, slices: BlockSlices) -> BlockHessianOperator {
        BlockHessianOperator {
            h_tt: self.h_tt,
            h_mm: self.h_mm,
            h_gg: self.h_gg,
            h_hh: self.h_hh,
            h_ww: self.h_ww,
            h_tm: self.h_tm,
            h_tg: self.h_tg,
            h_th: self.h_th,
            h_tw: self.h_tw,
            h_mg: self.h_mg,
            h_mh: self.h_mh,
            h_mw: self.h_mw,
            h_gh: self.h_gh,
            h_gw: self.h_gw,
            h_hw: self.h_hw,
            slices,
        }
    }

    fn add(&mut self, other: &BlockHessianAccumulator) {
        self.h_tt += &other.h_tt;
        self.h_mm += &other.h_mm;
        self.h_gg += &other.h_gg;
        self.h_hh += &other.h_hh;
        self.h_ww += &other.h_ww;
        self.h_tm += &other.h_tm;
        self.h_tg += &other.h_tg;
        self.h_th += &other.h_th;
        self.h_tw += &other.h_tw;
        self.h_mg += &other.h_mg;
        self.h_mh += &other.h_mh;
        self.h_mw += &other.h_mw;
        self.h_gh += &other.h_gh;
        self.h_gw += &other.h_gw;
        self.h_hw += &other.h_hw;
    }

    fn diagonal(&self, slices: &BlockSlices) -> Array1<f64> {
        let mut out = Array1::zeros(slices.total);
        out.slice_mut(s![slices.time.clone()])
            .assign(&self.h_tt.diag());
        out.slice_mut(s![slices.marginal.clone()])
            .assign(&self.h_mm.diag());
        out.slice_mut(s![slices.logslope.clone()])
            .assign(&self.h_gg.diag());
        if let Some(range) = slices.score_warp.as_ref() {
            out.slice_mut(s![range.clone()]).assign(&self.h_hh.diag());
        }
        if let Some(range) = slices.link_dev.as_ref() {
            out.slice_mut(s![range.clone()]).assign(&self.h_ww.diag());
        }
        out
    }
}

impl std::ops::AddAssign<&BlockHessianAccumulator> for BlockHessianAccumulator {
    fn add_assign(&mut self, other: &BlockHessianAccumulator) {
        self.add(other);
    }
}

impl BlockHessianAccumulator {
    /// Lifted pullback: J^T H J + Σ_a f_a K_a using actual Jacobians
    fn add_pullback_with_q_geometry(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        fg: &Array1<f64>,
        ph: &Array2<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        let ktt = [&qg.d2q0_time_time, &qg.d2q1_time_time, &qg.d2qd1_time_time];
        let ktm = [
            &qg.d2q0_time_marginal,
            &qg.d2q1_time_marginal,
            &qg.d2qd1_time_marginal,
        ];
        let kmm = [
            &qg.d2q0_marginal_marginal,
            &qg.d2q1_marginal_marginal,
            &qg.d2qd1_marginal_marginal,
        ];
        let pt = jt[0].len();
        let pm = jm[0].len();
        // Serial accumulation directly into self.h_tt / h_mm / h_tm.
        // The outer (over rows) parallelism is what saturates threads at
        // biobank N; nesting inner par_iter here adds work-stealing
        // overhead, per-chunk Array2::zeros allocations, and risks
        // OnceLock + nested rayon deadlock. Row order, accumulation
        // order, and per-row arithmetic remain bit-identical to the
        // previous serialised partial-merge pass.
        for a in 0..pt {
            for b in 0..pt {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * jt[u][a] * jt[w][b];
                    }
                }
                for u in 0..3 {
                    v += fg[u] * ktt[u][[a, b]];
                }
                self.h_tt[[a, b]] += v;
            }
        }

        for a in 0..pm {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * jm[u][a] * jm[w][b];
                    }
                }
                for u in 0..3 {
                    v += fg[u] * kmm[u][[a, b]];
                }
                self.h_mm[[a, b]] += v;
            }
        }
        family
            .logslope_design
            .syr_row_into(row, ph[[3, 3]], &mut self.h_gg)
            .map_err(|e| format!("add_pullback_with_q_geometry gg syr: {e}"))?;
        for a in 0..pt {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * jt[u][a] * jm[w][b];
                    }
                }
                for u in 0..3 {
                    v += fg[u] * ktm[u][[a, b]];
                }
                self.h_tm[[a, b]] += v;
            }
        }
        let gc = family
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_pullback_with_q_geometry logslope try_row_chunk: {e}"))?;
        let gr = gc.row(0);
        for a in 0..pt {
            let mut w = 0.0;
            for u in 0..3 {
                w += ph[[u, 3]] * jt[u][a];
            }
            if w != 0.0 {
                for b in 0..gr.len() {
                    self.h_tg[[a, b]] += w * gr[b];
                }
            }
        }
        for a in 0..pm {
            let mut w = 0.0;
            for u in 0..3 {
                w += ph[[u, 3]] * jm[u][a];
            }
            if w != 0.0 {
                for b in 0..gr.len() {
                    self.h_mg[[a, b]] += w * gr[b];
                }
            }
        }
        let pl = flex_primary_slices(family);
        if let Some(hr) = pl.h.as_ref() {
            for li in 0..hr.len() {
                let ix = hr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jt[u][a];
                    }
                    self.h_th[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jm[u][a];
                    }
                    self.h_mh[[a, li]] += w;
                }
                let gw = ph[[3, ix]];
                if gw != 0.0 {
                    for b in 0..gr.len() {
                        self.h_gh[[b, li]] += gw * gr[b];
                    }
                }
            }
            for l in 0..hr.len() {
                for r in 0..hr.len() {
                    self.h_hh[[l, r]] += ph[[hr.start + l, hr.start + r]];
                }
            }
        }
        if let Some(wr) = pl.w.as_ref() {
            for li in 0..wr.len() {
                let ix = wr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jt[u][a];
                    }
                    self.h_tw[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * jm[u][a];
                    }
                    self.h_mw[[a, li]] += w;
                }
                let gw = ph[[3, ix]];
                if gw != 0.0 {
                    for b in 0..gr.len() {
                        self.h_gw[[b, li]] += gw * gr[b];
                    }
                }
            }
            for l in 0..wr.len() {
                for r in 0..wr.len() {
                    self.h_ww[[l, r]] += ph[[wr.start + l, wr.start + r]];
                }
            }
        }
        if let (Some(hr), Some(wr)) = (pl.h.as_ref(), pl.w.as_ref()) {
            for hl in 0..hr.len() {
                for wl in 0..wr.len() {
                    self.h_hw[[hl, wl]] += ph[[hr.start + hl, wr.start + wl]];
                }
            }
        }
        Ok(())
    }

    /// U^α cross terms (eq 47, terms 1+2)
    fn add_timewiggle_psi_u_cross(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        lift: &TimewiggleMarginalPsiRowLift,
        ph: &Array2<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        let ut = [&lift.u_q0_time, &lift.u_q1_time, &lift.u_qd1_time];
        let um = [
            &lift.u_q0_marginal,
            &lift.u_q1_marginal,
            &lift.u_qd1_marginal,
        ];
        let pt = jt[0].len();
        let pm = jm[0].len();
        for a in 0..pt {
            for b in 0..pt {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * (ut[u][a] * jt[w][b] + jt[u][a] * ut[w][b]);
                    }
                }
                self.h_tt[[a, b]] += v;
            }
        }
        for a in 0..pm {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * (um[u][a] * jm[w][b] + jm[u][a] * um[w][b]);
                    }
                }
                self.h_mm[[a, b]] += v;
            }
        }
        for a in 0..pt {
            for b in 0..pm {
                let mut v = 0.0;
                for u in 0..3 {
                    for w in 0..3 {
                        v += ph[[u, w]] * (ut[u][a] * jm[w][b] + jt[u][a] * um[w][b]);
                    }
                }
                self.h_tm[[a, b]] += v;
            }
        }
        let gc = family
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("add_timewiggle_psi_u_cross logslope try_row_chunk: {e}"))?;
        let gr = gc.row(0);
        for a in 0..pt {
            let mut wt = 0.0;
            for u in 0..3 {
                wt += ph[[u, 3]] * ut[u][a];
            }
            if wt != 0.0 {
                for b in 0..gr.len() {
                    self.h_tg[[a, b]] += wt * gr[b];
                }
            }
        }
        for a in 0..pm {
            let mut wt = 0.0;
            for u in 0..3 {
                wt += ph[[u, 3]] * um[u][a];
            }
            if wt != 0.0 {
                for b in 0..gr.len() {
                    self.h_mg[[a, b]] += wt * gr[b];
                }
            }
        }
        let pl = flex_primary_slices(family);
        if let Some(hr) = pl.h.as_ref() {
            for li in 0..hr.len() {
                let ix = hr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * ut[u][a];
                    }
                    self.h_th[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * um[u][a];
                    }
                    self.h_mh[[a, li]] += w;
                }
            }
        }
        if let Some(wr) = pl.w.as_ref() {
            for li in 0..wr.len() {
                let ix = wr.start + li;
                for a in 0..pt {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * ut[u][a];
                    }
                    self.h_tw[[a, li]] += w;
                }
                for a in 0..pm {
                    let mut w = 0.0;
                    for u in 0..3 {
                        w += ph[[u, ix]] * um[u][a];
                    }
                    self.h_mw[[a, li]] += w;
                }
            }
        }
        Ok(())
    }

    /// K^{BC,α} terms (eq 47, term 5)
    fn add_timewiggle_psi_kappa_alpha(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        lift: &TimewiggleMarginalPsiRowLift,
        fg: &Array1<f64>,
    ) {
        let tw = family.time_wiggle_range();
        let pb = lift.x_entry_base.len();
        let pm = lift.marginal_row.len();
        let mu = lift.mu;
        let fq0 = fg[0];
        if fq0 != 0.0 {
            let (m3, m2) = (lift.entry_m3, lift.entry_m2);
            for i in 0..pb {
                for j in 0..pb {
                    self.h_tt[[i, j]] +=
                        fq0 * m3 * mu * lift.x_entry_base[i] * lift.x_entry_base[j];
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for i in 0..pb {
                    let v = fq0 * lift.entry_basis_d2[loc] * mu * lift.x_entry_base[i];
                    self.h_tt[[i, ti]] += v;
                    self.h_tt[[ti, i]] += v;
                }
            }
            for i in 0..pb {
                for j in 0..pm {
                    self.h_tm[[i, j]] += fq0
                        * (m3 * mu * lift.x_entry_base[i] * lift.marginal_row[j]
                            + m2 * lift.x_entry_base[i] * lift.psi_row[j]);
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for j in 0..pm {
                    self.h_tm[[ti, j]] += fq0
                        * (lift.entry_basis_d2[loc] * mu * lift.marginal_row[j]
                            + lift.entry_basis_d1[loc] * lift.psi_row[j]);
                }
            }
            for i in 0..pm {
                for j in 0..pm {
                    self.h_mm[[i, j]] += fq0
                        * (m3 * mu * lift.marginal_row[i] * lift.marginal_row[j]
                            + m2 * (lift.psi_row[i] * lift.marginal_row[j]
                                + lift.marginal_row[i] * lift.psi_row[j]));
                }
            }
        }
        let fq1 = fg[1];
        if fq1 != 0.0 {
            let (m3, m2) = (lift.exit_m3, lift.exit_m2);
            for i in 0..pb {
                for j in 0..pb {
                    self.h_tt[[i, j]] += fq1 * m3 * mu * lift.x_exit_base[i] * lift.x_exit_base[j];
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for i in 0..pb {
                    let v = fq1 * lift.exit_basis_d2[loc] * mu * lift.x_exit_base[i];
                    self.h_tt[[i, ti]] += v;
                    self.h_tt[[ti, i]] += v;
                }
            }
            for i in 0..pb {
                for j in 0..pm {
                    self.h_tm[[i, j]] += fq1
                        * (m3 * mu * lift.x_exit_base[i] * lift.marginal_row[j]
                            + m2 * lift.x_exit_base[i] * lift.psi_row[j]);
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for j in 0..pm {
                    self.h_tm[[ti, j]] += fq1
                        * (lift.exit_basis_d2[loc] * mu * lift.marginal_row[j]
                            + lift.exit_basis_d1[loc] * lift.psi_row[j]);
                }
            }
            for i in 0..pm {
                for j in 0..pm {
                    self.h_mm[[i, j]] += fq1
                        * (m3 * mu * lift.marginal_row[i] * lift.marginal_row[j]
                            + m2 * (lift.psi_row[i] * lift.marginal_row[j]
                                + lift.marginal_row[i] * lift.psi_row[j]));
                }
            }
        }
        let fqd = fg[2];
        if fqd != 0.0 {
            let (m4, m3, m2, dr) = (lift.exit_m4, lift.exit_m3, lift.exit_m2, lift.d_raw);
            for i in 0..pb {
                for j in 0..pb {
                    self.h_tt[[i, j]] += fqd
                        * (m4 * mu * dr * lift.x_exit_base[i] * lift.x_exit_base[j]
                            + m3 * mu
                                * (lift.x_exit_base[i] * lift.x_deriv_base[j]
                                    + lift.x_deriv_base[i] * lift.x_exit_base[j]));
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for i in 0..pb {
                    let v = fqd
                        * (lift.exit_basis_d3[loc] * mu * dr * lift.x_exit_base[i]
                            + lift.exit_basis_d2[loc] * mu * lift.x_deriv_base[i]);
                    self.h_tt[[i, ti]] += v;
                    self.h_tt[[ti, i]] += v;
                }
            }
            for i in 0..pb {
                for j in 0..pm {
                    self.h_tm[[i, j]] += fqd
                        * (m4 * mu * dr * lift.x_exit_base[i] * lift.marginal_row[j]
                            + m3 * dr * lift.x_exit_base[i] * lift.psi_row[j]
                            + m3 * mu * lift.x_deriv_base[i] * lift.marginal_row[j]
                            + m2 * lift.x_deriv_base[i] * lift.psi_row[j]);
                }
            }
            for loc in 0..tw.len() {
                let ti = tw.start + loc;
                for j in 0..pm {
                    self.h_tm[[ti, j]] += fqd
                        * (lift.exit_basis_d3[loc] * mu * dr * lift.marginal_row[j]
                            + lift.exit_basis_d2[loc] * dr * lift.psi_row[j]);
                }
            }
            for i in 0..pm {
                for j in 0..pm {
                    self.h_mm[[i, j]] += fqd
                        * (m4 * mu * dr * lift.marginal_row[i] * lift.marginal_row[j]
                            + m3 * dr
                                * (lift.psi_row[i] * lift.marginal_row[j]
                                    + lift.marginal_row[i] * lift.psi_row[j]));
                }
            }
        }
    }

    /// Σ_a w_a K_a^{BC}: second-pullback weighted by arbitrary vector (eq 47, term 4)
    fn add_second_pullback_weighted(
        &mut self,
        qg: &SurvivalMarginalSlopeDynamicRow,
        w: &Array1<f64>,
    ) {
        let ktt = [&qg.d2q0_time_time, &qg.d2q1_time_time, &qg.d2qd1_time_time];
        let ktm = [
            &qg.d2q0_time_marginal,
            &qg.d2q1_time_marginal,
            &qg.d2qd1_time_marginal,
        ];
        let kmm = [
            &qg.d2q0_marginal_marginal,
            &qg.d2q1_marginal_marginal,
            &qg.d2qd1_marginal_marginal,
        ];
        let pt = ktt[0].nrows();
        let pm = kmm[0].nrows();
        for q in 0..3 {
            let wq = w[q];
            if wq == 0.0 {
                continue;
            }
            for a in 0..pt {
                for b in 0..pt {
                    self.h_tt[[a, b]] += wq * ktt[q][[a, b]];
                }
            }
            for a in 0..pt {
                for b in 0..pm {
                    self.h_tm[[a, b]] += wq * ktm[q][[a, b]];
                }
            }
            for a in 0..pm {
                for b in 0..pm {
                    self.h_mm[[a, b]] += wq * kmm[q][[a, b]];
                }
            }
        }
    }

    /// Rank-1 psi cross with actual Jacobians from q-geometry
    fn add_rank1_psi_cross_with_q_geometry(
        &mut self,
        family: &SurvivalMarginalSlopeFamily,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        psi_block: usize,
        psi_row: &Array1<f64>,
        rp: &Array1<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        let pt = jt[0].len();
        let pm = jm[0].len();
        for a in 0..pt {
            let mut w = 0.0;
            for q in 0..3 {
                w += rp[q] * jt[q][a];
            }
            if w == 0.0 {
                continue;
            }
            let tgt = match psi_block {
                1 => &mut self.h_tm,
                2 => &mut self.h_tg,
                _ => continue,
            };
            for b in 0..psi_row.len() {
                tgt[[a, b]] += w * psi_row[b];
            }
        }
        for a in 0..pm {
            let mut w = 0.0;
            for q in 0..3 {
                w += rp[q] * jm[q][a];
            }
            if w == 0.0 {
                continue;
            }
            match psi_block {
                1 => {
                    for b in 0..psi_row.len() {
                        self.h_mm[[a, b]] += w * psi_row[b];
                        self.h_mm[[b, a]] += w * psi_row[b];
                    }
                }
                2 => {
                    for b in 0..psi_row.len() {
                        self.h_mg[[a, b]] += w * psi_row[b];
                    }
                }
                _ => {}
            }
        }
        let gc = family
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| {
                format!("add_rank1_psi_cross_with_q_geometry logslope try_row_chunk: {e}")
            })?;
        let gr = gc.row(0);
        let gw = rp[3];
        if gw != 0.0 {
            match psi_block {
                1 => {
                    for a in 0..gr.len() {
                        for b in 0..psi_row.len() {
                            self.h_mg[[b, a]] += gw * gr[a] * psi_row[b];
                        }
                    }
                }
                2 => {
                    for a in 0..gr.len() {
                        for b in 0..psi_row.len() {
                            self.h_gg[[a, b]] += gw * gr[a] * psi_row[b];
                            self.h_gg[[b, a]] += gw * gr[a] * psi_row[b];
                        }
                    }
                }
                _ => {}
            }
        }
        let pl = flex_primary_slices(family);
        if let Some(hr) = pl.h.as_ref() {
            for li in 0..hr.len() {
                let hw = rp[hr.start + li];
                if hw == 0.0 {
                    continue;
                }
                match psi_block {
                    1 => {
                        for b in 0..psi_row.len() {
                            self.h_mh[[b, li]] += hw * psi_row[b];
                        }
                    }
                    2 => {
                        for b in 0..psi_row.len() {
                            self.h_gh[[b, li]] += hw * psi_row[b];
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(wr) = pl.w.as_ref() {
            for li in 0..wr.len() {
                let ww = rp[wr.start + li];
                if ww == 0.0 {
                    continue;
                }
                match psi_block {
                    1 => {
                        for b in 0..psi_row.len() {
                            self.h_mw[[b, li]] += ww * psi_row[b];
                        }
                    }
                    2 => {
                        for b in 0..psi_row.len() {
                            self.h_gw[[b, li]] += ww * psi_row[b];
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

/// Block-structured HyperOperator for survival marginal-slope psi Hessians.
/// Stores the full 5-block exact joint Hessian layout and performs matvec
/// blockwise instead of materializing dense p×p structure in the outer path.
struct BlockHessianOperator {
    h_tt: Array2<f64>,
    h_mm: Array2<f64>,
    h_gg: Array2<f64>,
    h_hh: Array2<f64>,
    h_ww: Array2<f64>,
    h_tm: Array2<f64>,
    h_tg: Array2<f64>,
    h_th: Array2<f64>,
    h_tw: Array2<f64>,
    h_mg: Array2<f64>,
    h_mh: Array2<f64>,
    h_mw: Array2<f64>,
    h_gh: Array2<f64>,
    h_gw: Array2<f64>,
    h_hw: Array2<f64>,
    slices: BlockSlices,
}

impl HyperOperator for BlockHessianOperator {
    fn dim(&self) -> usize {
        self.slices.total
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let v_t = v.slice(s![self.slices.time.clone()]);
        let v_m = v.slice(s![self.slices.marginal.clone()]);
        let v_g = v.slice(s![self.slices.logslope.clone()]);
        let v_h = self
            .slices
            .score_warp
            .as_ref()
            .map(|range| v.slice(s![range.clone()]));
        let v_w = self
            .slices
            .link_dev
            .as_ref()
            .map(|range| v.slice(s![range.clone()]));
        let mut out = Array1::zeros(self.slices.total);
        {
            let mut o_t = out.slice_mut(s![self.slices.time.clone()]);
            o_t += &self.h_tt.dot(&v_t);
            o_t += &self.h_tm.dot(&v_m);
            o_t += &self.h_tg.dot(&v_g);
            if let Some(v_h) = v_h.as_ref() {
                o_t += &self.h_th.dot(v_h);
            }
            if let Some(v_w) = v_w.as_ref() {
                o_t += &self.h_tw.dot(v_w);
            }
        }
        {
            let mut o_m = out.slice_mut(s![self.slices.marginal.clone()]);
            o_m += &self.h_tm.t().dot(&v_t);
            o_m += &self.h_mm.dot(&v_m);
            o_m += &self.h_mg.dot(&v_g);
            if let Some(v_h) = v_h.as_ref() {
                o_m += &self.h_mh.dot(v_h);
            }
            if let Some(v_w) = v_w.as_ref() {
                o_m += &self.h_mw.dot(v_w);
            }
        }
        {
            let mut o_g = out.slice_mut(s![self.slices.logslope.clone()]);
            o_g += &self.h_tg.t().dot(&v_t);
            o_g += &self.h_mg.t().dot(&v_m);
            o_g += &self.h_gg.dot(&v_g);
            if let Some(v_h) = v_h.as_ref() {
                o_g += &self.h_gh.dot(v_h);
            }
            if let Some(v_w) = v_w.as_ref() {
                o_g += &self.h_gw.dot(v_w);
            }
        }
        if let (Some(range), Some(v_h)) = (self.slices.score_warp.as_ref(), v_h.as_ref()) {
            let mut o_h = out.slice_mut(s![range.clone()]);
            o_h += &self.h_th.t().dot(&v_t);
            o_h += &self.h_mh.t().dot(&v_m);
            o_h += &self.h_gh.t().dot(&v_g);
            o_h += &self.h_hh.dot(v_h);
            if let Some(v_w) = v_w.as_ref() {
                o_h += &self.h_hw.dot(v_w);
            }
        }
        if let (Some(range), Some(v_w)) = (self.slices.link_dev.as_ref(), v_w.as_ref()) {
            let mut o_w = out.slice_mut(s![range.clone()]);
            o_w += &self.h_tw.t().dot(&v_t);
            o_w += &self.h_mw.t().dot(&v_m);
            o_w += &self.h_gw.t().dot(&v_g);
            if let Some(v_h) = v_h.as_ref() {
                o_w += &self.h_hw.t().dot(v_h);
            }
            o_w += &self.h_ww.dot(v_w);
        }
        out
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let v_t = v.slice(s![self.slices.time.clone()]);
        let v_m = v.slice(s![self.slices.marginal.clone()]);
        let v_g = v.slice(s![self.slices.logslope.clone()]);
        let v_h = self
            .slices
            .score_warp
            .as_ref()
            .map(|range| v.slice(s![range.clone()]));
        let v_w = self
            .slices
            .link_dev
            .as_ref()
            .map(|range| v.slice(s![range.clone()]));
        let u_t = u.slice(s![self.slices.time.clone()]);
        let u_m = u.slice(s![self.slices.marginal.clone()]);
        let u_g = u.slice(s![self.slices.logslope.clone()]);
        let u_h = self
            .slices
            .score_warp
            .as_ref()
            .map(|range| u.slice(s![range.clone()]));
        let u_w = self
            .slices
            .link_dev
            .as_ref()
            .map(|range| u.slice(s![range.clone()]));
        let mut total = v_t.dot(&self.h_tt.dot(&u_t));
        total += v_m.dot(&self.h_mm.dot(&u_m));
        total += v_g.dot(&self.h_gg.dot(&u_g));
        if let (Some(v_h), Some(u_h)) = (v_h.as_ref(), u_h.as_ref()) {
            total += v_h.dot(&self.h_hh.dot(u_h));
        }
        if let (Some(v_w), Some(u_w)) = (v_w.as_ref(), u_w.as_ref()) {
            total += v_w.dot(&self.h_ww.dot(u_w));
        }
        total += v_t.dot(&self.h_tm.dot(&u_m));
        total += v_m.dot(&self.h_tm.t().dot(&u_t));
        total += v_t.dot(&self.h_tg.dot(&u_g));
        total += v_g.dot(&self.h_tg.t().dot(&u_t));
        total += v_m.dot(&self.h_mg.dot(&u_g));
        total += v_g.dot(&self.h_mg.t().dot(&u_m));
        if let (Some(v_h), Some(u_h)) = (v_h.as_ref(), u_h.as_ref()) {
            total += v_t.dot(&self.h_th.dot(u_h));
            total += v_h.dot(&self.h_th.t().dot(&u_t));
            total += v_m.dot(&self.h_mh.dot(u_h));
            total += v_h.dot(&self.h_mh.t().dot(&u_m));
            total += v_g.dot(&self.h_gh.dot(u_h));
            total += v_h.dot(&self.h_gh.t().dot(&u_g));
        }
        if let (Some(v_w), Some(u_w)) = (v_w.as_ref(), u_w.as_ref()) {
            total += v_t.dot(&self.h_tw.dot(u_w));
            total += v_w.dot(&self.h_tw.t().dot(&u_t));
            total += v_m.dot(&self.h_mw.dot(u_w));
            total += v_w.dot(&self.h_mw.t().dot(&u_m));
            total += v_g.dot(&self.h_gw.dot(u_w));
            total += v_w.dot(&self.h_gw.t().dot(&u_g));
        }
        if let ((Some(v_h), Some(u_w)), (Some(v_w), Some(u_h))) =
            ((v_h.as_ref(), u_w.as_ref()), (v_w.as_ref(), u_h.as_ref()))
        {
            total += v_h.dot(&self.h_hw.dot(u_w));
            total += v_w.dot(&self.h_hw.t().dot(u_h));
        }
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut out = Array2::zeros((self.slices.total, self.slices.total));
        out.slice_mut(s![self.slices.time.clone(), self.slices.time.clone()])
            .assign(&self.h_tt);
        out.slice_mut(s![
            self.slices.marginal.clone(),
            self.slices.marginal.clone()
        ])
        .assign(&self.h_mm);
        out.slice_mut(s![
            self.slices.logslope.clone(),
            self.slices.logslope.clone()
        ])
        .assign(&self.h_gg);
        if let Some(range) = self.slices.score_warp.as_ref() {
            out.slice_mut(s![range.clone(), range.clone()])
                .assign(&self.h_hh);
        }
        if let Some(range) = self.slices.link_dev.as_ref() {
            out.slice_mut(s![range.clone(), range.clone()])
                .assign(&self.h_ww);
        }
        out.slice_mut(s![self.slices.time.clone(), self.slices.marginal.clone()])
            .assign(&self.h_tm);
        out.slice_mut(s![self.slices.marginal.clone(), self.slices.time.clone()])
            .assign(&self.h_tm.t());
        out.slice_mut(s![self.slices.time.clone(), self.slices.logslope.clone()])
            .assign(&self.h_tg);
        out.slice_mut(s![self.slices.logslope.clone(), self.slices.time.clone()])
            .assign(&self.h_tg.t());
        out.slice_mut(s![
            self.slices.marginal.clone(),
            self.slices.logslope.clone()
        ])
        .assign(&self.h_mg);
        out.slice_mut(s![
            self.slices.logslope.clone(),
            self.slices.marginal.clone()
        ])
        .assign(&self.h_mg.t());
        if let Some(range) = self.slices.score_warp.as_ref() {
            out.slice_mut(s![self.slices.time.clone(), range.clone()])
                .assign(&self.h_th);
            out.slice_mut(s![range.clone(), self.slices.time.clone()])
                .assign(&self.h_th.t());
            out.slice_mut(s![self.slices.marginal.clone(), range.clone()])
                .assign(&self.h_mh);
            out.slice_mut(s![range.clone(), self.slices.marginal.clone()])
                .assign(&self.h_mh.t());
            out.slice_mut(s![self.slices.logslope.clone(), range.clone()])
                .assign(&self.h_gh);
            out.slice_mut(s![range.clone(), self.slices.logslope.clone()])
                .assign(&self.h_gh.t());
        }
        if let Some(range) = self.slices.link_dev.as_ref() {
            out.slice_mut(s![self.slices.time.clone(), range.clone()])
                .assign(&self.h_tw);
            out.slice_mut(s![range.clone(), self.slices.time.clone()])
                .assign(&self.h_tw.t());
            out.slice_mut(s![self.slices.marginal.clone(), range.clone()])
                .assign(&self.h_mw);
            out.slice_mut(s![range.clone(), self.slices.marginal.clone()])
                .assign(&self.h_mw.t());
            out.slice_mut(s![self.slices.logslope.clone(), range.clone()])
                .assign(&self.h_gw);
            out.slice_mut(s![range.clone(), self.slices.logslope.clone()])
                .assign(&self.h_gw.t());
        }
        if let (Some(h_range), Some(w_range)) = (
            self.slices.score_warp.as_ref(),
            self.slices.link_dev.as_ref(),
        ) {
            out.slice_mut(s![h_range.clone(), w_range.clone()])
                .assign(&self.h_hw);
            out.slice_mut(s![w_range.clone(), h_range.clone()])
                .assign(&self.h_hw.t());
        }
        out
    }

    fn is_implicit(&self) -> bool {
        false
    }
}

// ── Closed-form row kernel ─────────────────────────────────────────────
//
// The survival marginal-slope NLL for row i is:
//
//   ℓ_i = w_i [ (1-d)·neglogΦ(-η₁) + logΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
//
// with η₀ = q₀c + s_f g z, η₁ = q₁c + s_f g z, a'₁ = qd₁·c,
// c = √(1 + (s_f g)²), s_f = 1/√(1+σ²).
//
// All derivatives w.r.t. the 4 primary scalars (q₀, q₁, qd₁, g) are
// closed-form scalar formulas. No jets, no per-row heap allocation.

#[inline]
fn rigid_observed_logslope(g: f64, probit_scale: f64) -> f64 {
    probit_scale * g
}

#[inline]
fn rigid_observed_scale(g: f64, probit_scale: f64) -> f64 {
    let observed_g = rigid_observed_logslope(g, probit_scale);
    (1.0 + observed_g * observed_g).sqrt()
}

#[inline]
fn rigid_observed_eta(q: f64, g: f64, z: f64, probit_scale: f64) -> f64 {
    q * rigid_observed_scale(g, probit_scale) + rigid_observed_logslope(g, probit_scale) * z
}

pub fn survival_marginal_slope_vector_scale(
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) -> Result<f64, String> {
    marginal_slope_preserving_scale(slopes, covariance, probit_scale)
}

pub fn survival_marginal_slope_vector_eta(
    q: f64,
    z: &[f64],
    slopes: &[f64],
    covariance: &MarginalSlopeCovariance,
    probit_scale: f64,
) -> Result<f64, String> {
    if z.len() != covariance.dim() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector eta: score/covariance dimension mismatch: z={}, covariance={}",
                z.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    marginal_slope_probit_eta(q, z, slopes, covariance, probit_scale)
        .map_err(|err| format!("survival marginal-slope vector eta: {err}"))
}

pub fn survival_marginal_slope_vector_neglog(
    q0: f64,
    q1: f64,
    qd1: f64,
    slopes: &[f64],
    z: &[f64],
    covariance: &MarginalSlopeCovariance,
    weight: f64,
    event: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<f64, String> {
    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }
    let c = survival_marginal_slope_vector_scale(slopes, covariance, probit_scale)?;
    let eta0 = survival_marginal_slope_vector_eta(q0, z, slopes, covariance, probit_scale)?;
    let eta1 = survival_marginal_slope_vector_eta(q1, z, slopes, covariance, probit_scale)?;
    let ad1 = qd1 * c;
    if !(ad1.is_finite() && ad1 > 0.0) {
        return Err(SurvivalMarginalSlopeError::NumericalFailure {
            reason: format!(
                "survival marginal-slope transformed derivative must be positive, got {ad1}"
            ),
        }
        .into());
    }

    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    // Same survival row density as the scalar closed-form path, with only
    // eta and d eta/dt changed by the vector marginal-preserving map:
    //
    //   ell = w[(1-d)(-log Phi(-eta1)) + log Phi(-eta0)
    //           - d log phi(eta1) - d log(qd1 c)].
    //
    // There is no extra baseline -log phi(q1) or -log qd1 factor; adding
    // either would make K=1 diverge from `row_primary_closed_form`.
    Ok(weight
        * ((1.0 - event) * (-logcdf_neg_eta1) + logcdf_neg_eta0
            - event * log_phi_eta1
            - event * ad1.ln()))
}

fn marginal_slope_covariance_matvec(
    covariance: &MarginalSlopeCovariance,
    vector: &[f64],
) -> Result<Vec<f64>, String> {
    covariance.validate("survival marginal-slope covariance matvec")?;
    if vector.len() != covariance.dim() {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope covariance matvec dimension mismatch: vector={}, covariance={}",
                vector.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    Ok(match covariance {
        MarginalSlopeCovariance::Diagonal(diag) => vector
            .iter()
            .zip(diag.iter())
            .map(|(&v, &sigma)| sigma * v)
            .collect(),
        MarginalSlopeCovariance::Full(cov) => {
            let mut out = vec![0.0; cov.nrows()];
            for i in 0..cov.nrows() {
                for j in 0..cov.ncols() {
                    out[i] += cov[[i, j]] * vector[j];
                }
            }
            out
        }
        MarginalSlopeCovariance::LowRank(factor) => {
            let mut projected = vec![0.0; factor.ncols()];
            for r in 0..factor.ncols() {
                for k in 0..factor.nrows() {
                    projected[r] += factor[[k, r]] * vector[k];
                }
            }
            let mut out = vec![0.0; factor.nrows()];
            for k in 0..factor.nrows() {
                for r in 0..factor.ncols() {
                    out[k] += factor[[k, r]] * projected[r];
                }
            }
            out
        }
    })
}

fn row_primary_closed_form_vector(
    q0: f64,
    q1: f64,
    qd1: f64,
    slopes: &[f64],
    z: &[f64],
    covariance: &MarginalSlopeCovariance,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    let k = slopes.len();
    if z.len() != k || covariance.dim() != k {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope vector row dimension mismatch: slopes={}, z={}, covariance={}",
                k,
                z.len(),
                covariance.dim()
            ),
        }
        .into());
    }
    let c = survival_marginal_slope_vector_scale(slopes, covariance, probit_scale)?;
    let sigma_g = marginal_slope_covariance_matvec(covariance, slopes)?;
    let s2 = probit_scale * probit_scale;
    let mut c1 = vec![0.0; k];
    for a in 0..k {
        c1[a] = s2 * sigma_g[a] / c;
    }
    let mut c2 = Array2::<f64>::zeros((k, k));
    for a in 0..k {
        for b in 0..k {
            let sigma_ab = match covariance {
                MarginalSlopeCovariance::Diagonal(diag) => {
                    if a == b {
                        diag[a]
                    } else {
                        0.0
                    }
                }
                MarginalSlopeCovariance::Full(cov) => cov[[a, b]],
                MarginalSlopeCovariance::LowRank(factor) => {
                    let mut value = 0.0;
                    for r in 0..factor.ncols() {
                        value += factor[[a, r]] * factor[[b, r]];
                    }
                    value
                }
            };
            c2[[a, b]] = s2 * sigma_ab / c - (s2 * sigma_g[a]) * (s2 * sigma_g[b]) / (c * c * c);
        }
    }

    let linear = probit_scale
        * slopes
            .iter()
            .zip(z.iter())
            .map(|(&g, &zi)| g * zi)
            .sum::<f64>();
    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let ad1 = qd1 * c;
    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }
    if !(ad1.is_finite() && ad1 > 0.0) {
        return Err(SurvivalMarginalSlopeError::NumericalFailure {
            reason: format!(
                "survival marginal-slope transformed derivative must be positive, got {ad1}"
            ),
        }
        .into());
    }

    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let nll =
        w * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * ad1.ln());

    let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w)?;
    let (e1_k1, e1_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;
    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;
    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    let dim = 3 + k;
    let mut grad = Array1::<f64>::zeros(dim);
    let mut hess = Array2::<f64>::zeros((dim, dim));
    grad[0] = u1_eta0 * c;
    grad[1] = u1_eta1 * c;
    grad[2] = u1_ad1 * c;
    hess[[0, 0]] = u2_eta0 * c * c;
    hess[[1, 1]] = u2_eta1 * c * c;
    hess[[2, 2]] = u2_ad1 * c * c;
    for a in 0..k {
        let idx = 3 + a;
        let dlin = probit_scale * z[a];
        let deta0 = q0 * c1[a] + dlin;
        let deta1 = q1 * c1[a] + dlin;
        let dad1 = qd1 * c1[a];
        grad[idx] = u1_eta0 * deta0 + u1_eta1 * deta1 + u1_ad1 * dad1;
        hess[[0, idx]] = u2_eta0 * c * deta0 + u1_eta0 * c1[a];
        hess[[idx, 0]] = hess[[0, idx]];
        hess[[1, idx]] = u2_eta1 * c * deta1 + u1_eta1 * c1[a];
        hess[[idx, 1]] = hess[[1, idx]];
        hess[[2, idx]] = u2_ad1 * c * dad1 + u1_ad1 * c1[a];
        hess[[idx, 2]] = hess[[2, idx]];
        for b in 0..k {
            let jdx = 3 + b;
            let dlin_b = probit_scale * z[b];
            let deta0_b = q0 * c1[b] + dlin_b;
            let deta1_b = q1 * c1[b] + dlin_b;
            let dad1_b = qd1 * c1[b];
            hess[[idx, jdx]] = u2_eta0 * deta0 * deta0_b
                + u1_eta0 * q0 * c2[[a, b]]
                + u2_eta1 * deta1 * deta1_b
                + u1_eta1 * q1 * c2[[a, b]]
                + u2_ad1 * dad1 * dad1_b
                + u1_ad1 * qd1 * c2[[a, b]];
        }
    }
    Ok((nll, grad, hess))
}

fn standardize_latent_z_matrix_with_policy(
    z: &Array2<f64>,
    weights: &Array1<f64>,
    context: &str,
    policy: &LatentZPolicy,
) -> Result<(Array2<f64>, LatentZNormalization), String> {
    if z.ncols() == 0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!("{context} requires at least one z column"),
        }
        .into());
    }
    let mut out = Array2::<f64>::zeros(z.raw_dim());
    let mut first_norm = LatentZNormalization { mean: 0.0, sd: 1.0 };
    for col in 0..z.ncols() {
        let input = z.column(col).to_owned();
        let (standardized, normalization) =
            standardize_latent_z_with_policy(&input, weights, context, policy)?;
        if col == 0 {
            first_norm = normalization;
        }
        out.column_mut(col).assign(&standardized);
    }
    Ok((out, first_norm))
}

/// Derivatives of c(g) = √(1 + (s_f g)^2) up to 4th order in the raw slope g.
#[inline]
fn c_derivatives(g: f64, probit_scale: f64) -> (f64, f64, f64, f64, f64) {
    let observed_g = rigid_observed_logslope(g, probit_scale);
    let g2 = observed_g * observed_g;
    let s2 = probit_scale * probit_scale;
    let s4 = s2 * s2;
    let c = (1.0 + g2).sqrt();
    let c2 = c * c;
    let c3 = c2 * c;
    let c5 = c3 * c2;
    let c7 = c5 * c2;
    let c1 = s2 * g / c;
    let c2d = s2 / c3;
    let c3d = -3.0 * s4 * g / c5;
    let c4d = s4 * (12.0 * g2 - 3.0) / c7;
    (c, c1, c2d, c3d, c4d)
}

/// Derivatives of neglog(x) = -log(x): [-1/x, 1/x², -2/x³, 6/x⁴].
#[inline]
fn neglog_derivatives(x: f64) -> (f64, f64, f64, f64) {
    let x1 = x.max(1e-300);
    let inv = 1.0 / x1;
    let inv2 = inv * inv;
    (-inv, inv2, -2.0 * inv2 * inv, 6.0 * inv2 * inv2)
}

/// Row-level primary gradient (4-vector) and Hessian (4×4 symmetric)
/// computed entirely from closed-form scalar formulas.
///
/// Returns (nll, gradient[4], hessian[4][4]) on the stack.
#[inline]
fn row_primary_closed_form(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
    let (c, c1, c2, ..) = c_derivatives(g, probit_scale);
    let observed_g = rigid_observed_logslope(g, probit_scale);

    // Linear predictors
    let eta0 = q0 * c + observed_g * z;
    let eta1 = q1 * c + observed_g * z;
    let ad1 = qd1 * c;

    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }

    // ── NLL terms ──
    // Entry survival: -neglogΦ(-η₀) = logΦ(-η₀)
    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    // Exit survival: (1-d)·neglogΦ(-η₁)
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    // Event density: d·logφ(η₁)
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    // Time derivative: d·log(ad1)
    let log_ad1 = ad1.max(1e-300).ln();

    let nll =
        w * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * log_ad1);

    // ── First and second derivatives of each NLL component ──
    // signed_probit_neglog_derivatives gives derivatives with respect to m for
    // -weight * logΦ(m). Here m = -η, so odd derivatives flip sign when mapped
    // back to derivatives with respect to η.
    // For entry: m = -η₀, weight = -w because the NLL contains +w logΦ(-η₀)
    let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w)?;
    // For exit: m = -η₁, weight = w(1-d)
    let (e1_k1, e1_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
    // Event density: -d·logφ(η₁) = d·(η₁²/2 + const).
    // d/dη₁ = d·w·η₁, d²/dη₁² = d·w.
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    // Time derivative: -d·log(ad1).
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;

    // ── Chain rule to primary space ──
    // η₀ depends on (q₀, g): ∂η₀/∂q₀ = c, ∂η₀/∂g = q₀c₁ + s_f z
    // η₁ depends on (q₁, g): ∂η₁/∂q₁ = c, ∂η₁/∂g = q₁c₁ + s_f z
    // ad1 depends on (qd1, g): ∂ad1/∂qd1 = c, ∂ad1/∂g = qd1·c₁
    let deta0_dq0 = c;
    let deta0_dg = q0 * c1 + probit_scale * z;
    let deta1_dq1 = c;
    let deta1_dg = q1 * c1 + probit_scale * z;
    let dad1_dqd1 = c;
    let dad1_dg = qd1 * c1;

    // Combined first derivatives of total NLL:
    // u1 for η₀ terms = -e0_k1 (chain rule through m = -η₀)
    // u1 for η₁ terms = -e1_k1 + phi_u1 (chain rule through m = -η₁)
    // u1 for ad1 term = td_u1 (time derivative)
    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;

    let mut grad = [0.0_f64; N_PRIMARY];
    grad[0] = u1_eta0 * deta0_dq0; // ∂ℓ/∂q₀
    grad[1] = u1_eta1 * deta1_dq1; // ∂ℓ/∂q₁
    grad[2] = u1_ad1 * dad1_dqd1; // ∂ℓ/∂qd₁
    grad[3] = u1_eta0 * deta0_dg + u1_eta1 * deta1_dg + u1_ad1 * dad1_dg; // ∂ℓ/∂g

    // Combined second derivatives:
    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    // Second mixed derivatives of η w.r.t. primary scalars:
    let d2eta0_dq0dg = c1;
    let d2eta1_dq1dg = c1;
    let d2ad1_dqd1dg = c1;
    // d²η₀/dg² = q₀·c₂ (z is linear in g, so its second derivative is 0)
    let d2eta0_dg2 = q0 * c2;
    let d2eta1_dg2 = q1 * c2;
    let d2ad1_dg2 = qd1 * c2;

    let mut hess = [[0.0_f64; N_PRIMARY]; N_PRIMARY];

    // (q0, q0)
    hess[0][0] = u2_eta0 * deta0_dq0 * deta0_dq0;
    // (q1, q1)
    hess[1][1] = u2_eta1 * deta1_dq1 * deta1_dq1;
    // (qd1, qd1)
    hess[2][2] = u2_ad1 * dad1_dqd1 * dad1_dqd1;
    // (q0, q1) = 0 (η₀ and η₁ share no primary scalars except g)
    hess[0][1] = 0.0;
    hess[1][0] = 0.0;
    // (q0, qd1) = 0
    hess[0][2] = 0.0;
    hess[2][0] = 0.0;
    // (q1, qd1) = 0
    hess[1][2] = 0.0;
    hess[2][1] = 0.0;
    // (q0, g) = u2_η₀ · (∂η₀/∂q₀)(∂η₀/∂g) + u1_η₀ · (∂²η₀/∂q₀∂g)
    hess[0][3] = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    hess[3][0] = hess[0][3];
    // (q1, g)
    hess[1][3] = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    hess[3][1] = hess[1][3];
    // (qd1, g)
    hess[2][3] = u2_ad1 * dad1_dqd1 * dad1_dg + u1_ad1 * d2ad1_dqd1dg;
    hess[3][2] = hess[2][3];
    // (g, g) = Σ_terms [u2·(dterm/dg)² + u1·(d²term/dg²)]
    hess[3][3] = u2_eta0 * deta0_dg * deta0_dg
        + u1_eta0 * d2eta0_dg2
        + u2_eta1 * deta1_dg * deta1_dg
        + u1_eta1 * d2eta1_dg2
        + u2_ad1 * dad1_dg * dad1_dg
        + u1_ad1 * d2ad1_dg2;

    Ok((nll, grad, hess))
}

/// Shared-slope multi-z reduction for the rigid 4-primary row calculus.
///
/// When K observed scores share one raw log-slope value `g`, the probit
/// gradient vector is `r(g) = s_f g 1_K` and the row index is
///
///     eta_j = q_j sqrt(1 + r(g)' Sigma r(g)) + r(g)' z_i
///           = q_j sqrt(1 + g^2 s_f^2 1' Sigma 1) + g s_f (1'z_i).
///
/// The exact row kernel can therefore remain four-dimensional
/// `(q0, q1, qd1, g)` if, and only if, the log-slope surface is shared across
/// z coordinates.  The derivatives below are the scalar chain rule with
/// `c(g) = sqrt(1 + g^2 s_f^2 1' Sigma 1)` and
/// `d(r'z_i)/dg = s_f 1'z_i`.  K=1 with `1' Sigma 1 = 1` is exactly the
/// existing scalar path handled by `row_primary_closed_form`.
#[inline]
fn row_primary_closed_form_shared_score(
    q0: f64,
    q1: f64,
    qd1: f64,
    g: f64,
    z_sum: f64,
    covariance_ones: f64,
    w: f64,
    d: f64,
    derivative_guard: f64,
    probit_scale: f64,
) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
    if !(covariance_ones.is_finite() && covariance_ones >= 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!(
                "survival marginal-slope shared-score covariance scale must be finite and non-negative, got {covariance_ones}"
            ),
        }
        .into());
    }
    let effective_scale = probit_scale * covariance_ones.sqrt();
    let (c, c1, c2, ..) = c_derivatives(g, effective_scale);
    let linear = rigid_observed_logslope(g, probit_scale) * z_sum;
    let linear_dg = probit_scale * z_sum;

    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let ad1 = qd1 * c;

    if survival_derivative_guard_violated(qd1, derivative_guard) {
        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
            reason: format!(
                "survival marginal-slope monotonicity violated: qd1={qd1:.3e} < guard={derivative_guard:.3e}"
            ),
        }
        .into());
    }

    let (logcdf_neg_eta0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
    let (logcdf_neg_eta1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let log_ad1 = ad1.max(1e-300).ln();

    let nll =
        w * ((1.0 - d) * (-logcdf_neg_eta1) + logcdf_neg_eta0 - d * log_phi_eta1 - d * log_ad1);

    let (e0_k1, e0_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta0, -w)?;
    let (e1_k1, e1_k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-eta1, w * (1.0 - d))?;
    let phi_u1 = w * d * eta1;
    let phi_u2 = w * d;
    let (nl_u1, nl_u2, _, _) = neglog_derivatives(ad1);
    let td_u1 = w * d * nl_u1;
    let td_u2 = w * d * nl_u2;

    let deta0_dq0 = c;
    let deta0_dg = q0 * c1 + linear_dg;
    let deta1_dq1 = c;
    let deta1_dg = q1 * c1 + linear_dg;
    let dad1_dqd1 = c;
    let dad1_dg = qd1 * c1;

    let u1_eta0 = -e0_k1;
    let u1_eta1 = -e1_k1 + phi_u1;
    let u1_ad1 = td_u1;

    let mut grad = [0.0_f64; N_PRIMARY];
    grad[0] = u1_eta0 * deta0_dq0;
    grad[1] = u1_eta1 * deta1_dq1;
    grad[2] = u1_ad1 * dad1_dqd1;
    grad[3] = u1_eta0 * deta0_dg + u1_eta1 * deta1_dg + u1_ad1 * dad1_dg;

    let u2_eta0 = e0_k2;
    let u2_eta1 = e1_k2 + phi_u2;
    let u2_ad1 = td_u2;

    let d2eta0_dq0dg = c1;
    let d2eta1_dq1dg = c1;
    let d2ad1_dqd1dg = c1;
    let d2eta0_dg2 = q0 * c2;
    let d2eta1_dg2 = q1 * c2;
    let d2ad1_dg2 = qd1 * c2;

    let mut hess = [[0.0_f64; N_PRIMARY]; N_PRIMARY];
    hess[0][0] = u2_eta0 * deta0_dq0 * deta0_dq0;
    hess[1][1] = u2_eta1 * deta1_dq1 * deta1_dq1;
    hess[2][2] = u2_ad1 * dad1_dqd1 * dad1_dqd1;
    hess[0][3] = u2_eta0 * deta0_dq0 * deta0_dg + u1_eta0 * d2eta0_dq0dg;
    hess[3][0] = hess[0][3];
    hess[1][3] = u2_eta1 * deta1_dq1 * deta1_dg + u1_eta1 * d2eta1_dq1dg;
    hess[3][1] = hess[1][3];
    hess[2][3] = u2_ad1 * dad1_dqd1 * dad1_dg + u1_ad1 * d2ad1_dqd1dg;
    hess[3][2] = hess[2][3];
    hess[3][3] = u2_eta0 * deta0_dg * deta0_dg
        + u1_eta0 * d2eta0_dg2
        + u2_eta1 * deta1_dg * deta1_dg
        + u1_eta1 * d2eta1_dg2
        + u2_ad1 * dad1_dg * dad1_dg
        + u1_ad1 * d2ad1_dg2;

    Ok((nll, grad, hess))
}

// ── Eval cache ────────────────────────────────────────────────────────
//
// Third and fourth order contracted derivatives for the outer REML path
// continue to use the MultiDirJet engine via row_neglog_directional.
// That path is called O(n_rho²) times, not O(n × inner_iters) times,
// so the jet overhead is acceptable there.

#[derive(Clone)]
struct RowPrimaryBase {
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

struct EvalCache {
    row_bases: Vec<RowPrimaryBase>,
}

// ── Row-level NLL computation ─────────────────────────────────────────

impl SurvivalMarginalSlopeFamily {
    #[inline]
    fn probit_frailty_scale(&self) -> f64 {
        probit_frailty_scale(self.gaussian_frailty_sd)
    }

    fn z_subsample_key(&self) -> Array1<f64> {
        self.z.column(0).to_owned()
    }

    #[inline]
    fn score_dim(&self) -> usize {
        debug_assert_eq!(self.score_covariance.dim(), self.z.ncols());
        self.z.ncols()
    }

    fn score_warp_basis_dim(&self) -> usize {
        self.score_warp
            .as_ref()
            .map_or(0, DeviationRuntime::basis_dim)
    }

    fn score_warp_coord_basis_index(&self, local_idx: usize) -> Result<(usize, usize), String> {
        let basis_dim = self.score_warp_basis_dim();
        if basis_dim == 0 {
            return Err(
                "survival score-warp coordinate lookup without score-warp runtime".to_string(),
            );
        }
        let coord = local_idx / basis_dim;
        if coord >= self.score_dim() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival score-warp local index {local_idx} exceeds K={} per-z blocks with basis dim {basis_dim}",
                    self.score_dim()
                ),
            }
            .into());
        }
        Ok((coord, local_idx % basis_dim))
    }

    fn score_warp_beta_for_coord(
        &self,
        beta_h: &Array1<f64>,
        coord: usize,
    ) -> Result<Array1<f64>, String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        score_warp_component_beta(runtime, beta_h, coord)
    }

    #[inline]
    fn zero_score_warp_span() -> exact_kernel::LocalSpanCubic {
        exact_kernel::LocalSpanCubic {
            left: 0.0,
            right: 1.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
        }
    }

    fn add_local_span_cubic(
        left: f64,
        right: f64,
        target: &mut exact_kernel::LocalSpanCubic,
        span: exact_kernel::LocalSpanCubic,
    ) {
        target.left = left;
        target.right = right;
        target.c0 += span.c0;
        target.c1 += span.c1;
        target.c2 += span.c2;
        target.c3 += span.c3;
    }

    fn score_warp_local_cubic_at(
        &self,
        beta_h: Option<&Array1<f64>>,
        value: f64,
    ) -> Result<exact_kernel::LocalSpanCubic, String> {
        let (Some(runtime), Some(beta_h)) = (self.score_warp.as_ref(), beta_h) else {
            return Ok(Self::zero_score_warp_span());
        };
        if self.score_dim() == 1 {
            return runtime.local_cubic_at(beta_h, value);
        }
        let mut sum = Self::zero_score_warp_span();
        for coord in 0..self.score_dim() {
            let local_beta = score_warp_component_beta(runtime, beta_h, coord)?;
            let span = runtime.local_cubic_at(&local_beta, value)?;
            if coord == 0 {
                sum = exact_kernel::LocalSpanCubic {
                    left: span.left,
                    right: span.right,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                };
            }
            Self::add_local_span_cubic(span.left, span.right, &mut sum, span);
        }
        Ok(sum)
    }

    fn score_warp_observed_value(
        &self,
        row: usize,
        beta_h: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let (Some(runtime), Some(beta_h)) = (self.score_warp.as_ref(), beta_h) else {
            return Ok(0.0);
        };
        let mut value = 0.0;
        for coord in 0..self.score_dim() {
            let local_beta = score_warp_component_beta(runtime, beta_h, coord)?;
            let z_coord = self.z[[row, coord]];
            value += runtime
                .local_cubic_at(&local_beta, z_coord)?
                .evaluate(z_coord);
        }
        Ok(value)
    }

    fn observed_score_projection(&self, row: usize) -> f64 {
        if self.score_dim() == 1 {
            self.z[[row, 0]]
        } else {
            self.z.row(row).sum()
        }
    }

    fn integration_score_basis_coefficients(
        &self,
        local_idx: usize,
        z_basis: f64,
        multiplier: f64,
    ) -> Result<[f64; 4], String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        let (_, basis_idx) = self.score_warp_coord_basis_index(local_idx)?;
        let basis_span = runtime.basis_cubic_at(basis_idx, z_basis)?;
        Ok(exact_kernel::score_basis_cell_coefficients(
            basis_span, multiplier,
        ))
    }

    fn observed_score_basis_coefficients(
        &self,
        row: usize,
        local_idx: usize,
        z_obs: f64,
        multiplier: f64,
    ) -> Result<[f64; 4], String> {
        let runtime = self
            .score_warp
            .as_ref()
            .ok_or_else(|| "missing survival score-warp runtime".to_string())?;
        if self.score_dim() == 1 {
            let basis_span = runtime.basis_cubic_at(local_idx, z_obs)?;
            return Ok(exact_kernel::score_basis_cell_coefficients(
                basis_span, multiplier,
            ));
        }
        let (coord, basis_idx) = self.score_warp_coord_basis_index(local_idx)?;
        let z_coord = self.z[[row, coord]];
        let basis_span = runtime.basis_cubic_at(basis_idx, z_coord)?;
        Ok([multiplier * basis_span.evaluate(z_coord), 0.0, 0.0, 0.0])
    }

    fn logslope_vector_for_row(
        &self,
        row: usize,
        logslope_eta: &Array1<f64>,
    ) -> Result<Vec<f64>, String> {
        let k = self.score_dim();
        if self.logslope_surface_ranges.len() == k && k > 1 {
            if logslope_eta.len() == self.n {
                return Err(
                    "survival marginal-slope internal logslope vector requested scalar eta for a per-z surface layout"
                        .to_string(),
                );
            }
        }
        if logslope_eta.len() == self.n {
            return Ok(vec![logslope_eta[row]; k]);
        }
        if logslope_eta.len() == self.n * k {
            let start = row * k;
            return Ok(logslope_eta.slice(s![start..start + k]).to_vec());
        }
        Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope logslope eta length {} is incompatible with n={} and score dim K={k}",
                logslope_eta.len(),
                self.n
            ),
        }
        .into())
    }

    fn per_z_logslope_active(&self) -> bool {
        self.score_dim() > 1 && self.logslope_surface_ranges.len() == self.score_dim()
    }

    fn logslope_surface_values_for_row(
        &self,
        row: usize,
        beta_logslope: &Array1<f64>,
    ) -> Result<Vec<f64>, String> {
        let k = self.score_dim();
        if !self.per_z_logslope_active() {
            return self.logslope_vector_for_row(row, beta_logslope);
        }
        let g_row = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("logslope_surface_values_for_row logslope row chunk: {e}"))?;
        let row_view = g_row.row(0);
        let mut out = Vec::with_capacity(k);
        for range in &self.logslope_surface_ranges {
            out.push(
                row_view
                    .slice(s![range.clone()])
                    .dot(&beta_logslope.slice(s![range.clone()])),
            );
        }
        Ok(out)
    }

    fn logslope_surface_row(&self, row: usize) -> Result<Array1<f64>, String> {
        let chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("logslope_surface_row logslope row chunk: {e}"))?;
        Ok(chunk.row(0).to_owned())
    }

    fn shared_logslope_covariance_scale(&self) -> Result<f64, String> {
        let k = self.score_dim();
        if k == 1 {
            return Ok(1.0);
        }
        let ones = vec![1.0; k];
        self.score_covariance.quadratic_form(&ones).map_err(|err| {
            format!("survival marginal-slope shared log-slope covariance scale: {err}")
        })
    }

    fn exact_shared_score_summary(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        context: &str,
    ) -> Result<(f64, f64), String> {
        let k = self.score_dim();
        if k == 1 {
            return Ok((self.z[[row, 0]], 1.0));
        }
        let logslope_eta_len = block_states[2].eta.len();
        if logslope_eta_len != self.n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "{context}: survival marginal-slope exact shared-slope calculus for K={k} requires one log-slope eta per row (n={}); got eta len {logslope_eta_len}. Per-z log-slope derivatives require a {}-primary row kernel.",
                    self.n,
                    3 + k
                ),
            }
            .into());
        }
        Ok((
            self.z.row(row).sum(),
            self.shared_logslope_covariance_scale()?,
        ))
    }

    fn row_primary_closed_form_rigid(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        block_states: &[ParameterBlockState],
        probit_scale: f64,
    ) -> Result<(f64, [f64; N_PRIMARY], [[f64; N_PRIMARY]; N_PRIMARY]), String> {
        let logslope_eta = &block_states[2].eta;
        let k = self.score_dim();
        if k == 1 {
            return row_primary_closed_form(
                q0,
                q1,
                qd1,
                logslope_eta[row],
                self.z[[row, 0]],
                self.weights[row],
                self.event[row],
                self.derivative_guard,
                probit_scale,
            );
        }
        if logslope_eta.len() != self.n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope exact rigid row calculus for K={k} requires one shared log-slope surface (eta len n={}); got eta len {}. Per-z log-slope derivatives require a {}-primary row kernel.",
                    self.n,
                    logslope_eta.len(),
                    3 + k
                ),
            }
            .into());
        }
        let (z_sum, covariance_ones) =
            self.exact_shared_score_summary(row, block_states, "row_primary_closed_form_rigid")?;
        row_primary_closed_form_shared_score(
            q0,
            q1,
            qd1,
            logslope_eta[row],
            z_sum,
            covariance_ones,
            self.weights[row],
            self.event[row],
            self.derivative_guard,
            probit_scale,
        )
    }

    fn ensure_scalar_flex_exact_score_geometry(&self, context: &str) -> Result<(), String> {
        if self.score_dim() == 1 {
            return Ok(());
        }
        Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "{context}: survival marginal-slope exact flexible row calculus is scalar-z only; K={} must use the rigid shared-slope vector kernel or a widened per-z primary kernel",
                self.score_dim()
            ),
        }
        .into())
    }

    fn row_neglog_rigid_vector_value(
        &self,
        row: usize,
        q_geom: SurvivalMarginalSlopeDynamicRowValues,
        block_states: &[ParameterBlockState],
        probit_scale: f64,
    ) -> Result<f64, String> {
        let slopes = if self.per_z_logslope_active() {
            self.logslope_surface_values_for_row(row, &block_states[2].beta)?
        } else {
            self.logslope_vector_for_row(row, &block_states[2].eta)?
        };
        let z = self.z.row(row).to_vec();
        survival_marginal_slope_vector_neglog(
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            &slopes,
            &z,
            &self.score_covariance,
            self.weights[row],
            self.event[row],
            self.derivative_guard,
            probit_scale,
        )
    }

    /// Two-phase auto-subsample entry: when `options.auto_outer_subsample` is
    /// enabled and Phase 1 still has budget, this
    /// returns a cloned `BlockwiseFitOptions` carrying a freshly built
    /// stratified Horvitz-Thompson mask. Otherwise returns `None` and
    /// the caller uses the original options unchanged.
    ///
    /// Survival entry points (`*_workspace_with_options`,
    /// `log_likelihood_only_with_options`) do not receive the outer ρ
    /// directly, so we derive a per-outer-eval key by concatenating the
    /// joint coefficient vector across blocks. Within a single outer
    /// eval downstream calls share the same betas, so retries do not
    /// double-bump; across outer evals the betas change so the counter
    /// increments cleanly. This is the documented fallback for
    /// families that lack an explicit ρ at the entry point.
    fn install_auto_outer_subsample_options(
        &self,
        options: &BlockwiseFitOptions,
        block_states: &[ParameterBlockState],
    ) -> Option<BlockwiseFitOptions> {
        let mut total_len = 0usize;
        for state in block_states {
            total_len += state.beta.len();
        }
        let mut beta_proxy = Array1::<f64>::zeros(total_len);
        let mut cursor = 0usize;
        for state in block_states {
            let len = state.beta.len();
            if len > 0 {
                beta_proxy
                    .slice_mut(s![cursor..cursor + len])
                    .assign(&state.beta);
                cursor += len;
            }
        }
        let event_secondary: Vec<u8> = self
            .event
            .iter()
            .map(|v| if *v > 0.5 { 1u8 } else { 0u8 })
            .collect();
        let z_key = self.z_subsample_key();
        crate::families::marginal_slope_shared::maybe_install_auto_outer_subsample(
            options,
            z_key.as_slice().expect("z key must be contiguous"),
            Some(&event_secondary),
            &beta_proxy,
            &self.auto_subsample_phase_counter,
            &self.auto_subsample_last_rho,
            SURVIVAL_MGS_AUTO_SUBSAMPLE_PHASE1_BUDGET,
            "survival-mgs",
        )
    }

    /// Outer-aware variant of `log_likelihood_only`. When
    /// `options.outer_score_subsample` is `None` this iterates over all rows
    /// and matches the legacy full-data implementation. When it is `Some`,
    /// only the sampled rows contribute, with their Horvitz-Thompson
    /// inverse-inclusion weights taken from `OuterScoreSubsample::rows`. Lets outer-only
    /// score/gradient passes scale to biobank `n` without distorting the
    /// full-data inner-PIRLS or covariance code paths.
    pub(crate) fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let row_iter = outer_weighted_rows(options, self.n).to_vec();
        if flex_active {
            self.validate_exact_monotonicity(block_states)?;
            let total: Result<f64, String> = row_iter
                .into_par_iter()
                .try_fold(
                    || 0.0,
                    |mut ll, weighted| -> Result<_, String> {
                        ll -= weighted.weight
                            * self.row_neglog_flex_value(weighted.index, block_states)?;
                        Ok(ll)
                    },
                )
                .try_reduce(
                    || 0.0,
                    |left, right| -> Result<_, String> { Ok(left + right) },
                );
            return total;
        }
        // True fast path: K=1 keeps the original scalar closed form; K>1
        // uses the covariance-aware vector likelihood. Exact rigid
        // gradient/Hessian paths below use the same c(a)=sqrt(1+r'Sigma r)
        // algebra through `row_primary_closed_form_rigid`.
        let guard = self.derivative_guard;
        let probit_scale = self.probit_frailty_scale();
        let score_dim = self.score_dim();
        let total: Result<f64, String> = row_iter
            .into_par_iter()
            .try_fold(
                || 0.0,
                |mut ll, weighted| -> Result<_, String> {
                    let i = weighted.index;
                    let q_geom = self.row_dynamic_q_values(i, block_states)?;
                    if score_dim > 1 {
                        ll -= weighted.weight
                            * self.row_neglog_rigid_vector_value(
                                i,
                                q_geom,
                                block_states,
                                probit_scale,
                            )?;
                        return Ok(ll);
                    }
                    let g = block_states[2].eta[i];
                    let (nll, _, _) = row_primary_closed_form(
                        q_geom.q0,
                        q_geom.q1,
                        q_geom.qd1,
                        g,
                        self.z[[i, 0]],
                        self.weights[i],
                        self.event[i],
                        guard,
                        probit_scale,
                    )?;
                    ll -= weighted.weight * nll;
                    Ok(ll)
                },
            )
            .try_reduce(
                || 0.0,
                |left, right| -> Result<_, String> { Ok(left + right) },
            );
        total
    }

    fn is_sigma_aux_index(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> bool {
        shared_is_sigma_aux_index(self.gaussian_frailty_sd, derivative_blocks, psi_index)
    }

    fn sigma_scale_jet(
        &self,
        n_dirs: usize,
        first_masks: &[usize],
        second_masks: &[usize],
    ) -> Result<MultiDirJet, String> {
        probit_frailty_scale_multi_dir_jet(
            self.gaussian_frailty_sd,
            "survival marginal-slope log-sigma auxiliary requested without GaussianShift sigma",
            n_dirs,
            first_masks,
            second_masks,
        )
    }

    fn row_neglog_directional_with_scale_jet(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[&Array1<f64>],
        scale_jet: &MultiDirJet,
    ) -> Result<f64, String> {
        let k = dirs.len();
        if k > 4 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope sigma row directional expects 0..=4 directions, got {k}"
                ),
            }
            .into());
        }
        if scale_jet.coeffs.len() != (1usize << k) {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope sigma scale jet dimension mismatch: coeffs={}, dirs={k}",
                    scale_jet.coeffs.len()
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (z_sum, covariance_ones) = self.exact_shared_score_summary(
            row,
            block_states,
            "row_neglog_directional_with_scale_jet",
        )?;
        let q_geom = self.row_dynamic_q_values(row, block_states)?;

        let first = |idx: usize| -> Vec<f64> { dirs.iter().map(|dir| dir[idx]).collect() };
        let q0_jet = MultiDirJet::linear(k, q_geom.q0, &first(0));
        let q1_jet = MultiDirJet::linear(k, q_geom.q1, &first(1));
        let qd1_jet = MultiDirJet::linear(k, q_geom.qd1, &first(2));
        let g_jet = MultiDirJet::linear(k, block_states[2].eta[row], &first(3));

        let observed_g_jet = g_jet.mul(scale_jet);
        let one_plus_b2 = MultiDirJet::constant(k, 1.0)
            .add(&observed_g_jet.mul(&observed_g_jet).scale(covariance_ones));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);
        let z_jet = MultiDirJet::constant(k, z_sum);
        let eta0_jet = a0_jet.add(&observed_g_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&observed_g_jet.mul(&z_jet));

        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0);

        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let qd1_lower = self.time_derivative_lower_bound();
        let qd1_val = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if survival_derivative_guard_violated(qd1_val, qd1_lower) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
                ),
            }
            .into());
        }
        let time_deriv_term = if di > 0.0 {
            ad1_jet
                .compose_unary(unary_derivatives_log(ad1_val))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        Ok(exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term)
            .coeff((1usize << k) - 1))
    }

    fn row_sigma_primary_terms(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        second_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let primary_dim = N_PRIMARY;
        let zero = zero_primary_direction_ref();
        // sigma_scale_jet only depends on the multi-dir spec, not on the
        // row — so resolve each variant once and reuse across the
        // objective + grad + hess sweeps for this row. Previously this
        // was rebuilt 1 + N_PRIMARY + N_PRIMARY*(N_PRIMARY+1)/2 = 15
        // times per row.
        let (scale_obj, scale_grad, scale_hess) = if second_sigma {
            (
                self.sigma_scale_jet(2, &[1, 2], &[3])?,
                self.sigma_scale_jet(3, &[1, 2], &[3])?,
                self.sigma_scale_jet(4, &[1, 2], &[3])?,
            )
        } else {
            (
                self.sigma_scale_jet(1, &[1], &[])?,
                self.sigma_scale_jet(2, &[1], &[])?,
                self.sigma_scale_jet(3, &[1], &[])?,
            )
        };
        let objective = if second_sigma {
            self.row_neglog_directional_with_scale_jet(
                row,
                block_states,
                &[zero, zero],
                &scale_obj,
            )?
        } else {
            self.row_neglog_directional_with_scale_jet(row, block_states, &[zero], &scale_obj)?
        };

        let mut grad = Array1::<f64>::zeros(primary_dim);
        for a in 0..primary_dim {
            let da = unit_primary_direction_ref(a);
            let value = if second_sigma {
                self.row_neglog_directional_with_scale_jet(
                    row,
                    block_states,
                    &[zero, zero, da],
                    &scale_grad,
                )?
            } else {
                self.row_neglog_directional_with_scale_jet(
                    row,
                    block_states,
                    &[zero, da],
                    &scale_grad,
                )?
            };
            grad[a] = value;
        }

        let mut hess = Array2::<f64>::zeros((primary_dim, primary_dim));
        for a in 0..primary_dim {
            let da = unit_primary_direction_ref(a);
            for b in a..primary_dim {
                let db = unit_primary_direction_ref(b);
                let value = if second_sigma {
                    self.row_neglog_directional_with_scale_jet(
                        row,
                        block_states,
                        &[zero, zero, da, db],
                        &scale_hess,
                    )?
                } else {
                    self.row_neglog_directional_with_scale_jet(
                        row,
                        block_states,
                        &[zero, da, db],
                        &scale_hess,
                    )?
                };
                hess[[a, b]] = value;
                hess[[b, a]] = value;
            }
        }

        Ok((objective, grad, hess))
    }

    fn sigma_exact_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.sigma_exact_joint_psi_terms_with_options(
            block_states,
            specs,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psi_terms`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, per-block score vectors, Hessian operator blocks) is accumulated
    /// with the row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn sigma_exact_joint_psi_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        drop(specs);
        if self.flex_active() {
            return Err(
                "survival marginal-slope log-sigma hyperderivatives are implemented for the rigid probit marginal-slope kernel; flex score/link/timewiggle kernels still require the analytic cell-tensor sigma path"
                    .to_string(),
            );
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Bit-deterministic reduction: see `chunked_row_reduction`.
        let (objective_psi, score_t, score_m, score_g, score_h, score_w, acc) =
            chunked_row_reduction(
                row_iter.as_slice(),
                || {
                    (
                        0.0,
                        Array1::zeros(p_t),
                        Array1::zeros(p_m),
                        Array1::zeros(p_g),
                        Array1::zeros(p_h),
                        Array1::zeros(p_w),
                        BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
                    )
                },
                |row, a| -> Result<(), String> {
                    let (mut obj, mut grad, mut hess) =
                        self.row_sigma_primary_terms(row, block_states, false)?;
                    let w = row_weights[row];
                    if w != 1.0 {
                        obj *= w;
                        grad.mapv_inplace(|v| v * w);
                        hess.mapv_inplace(|v| v * w);
                    }
                    a.0 += obj;
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    self.accumulate_score_with_q_geometry(
                        row, &q_geom, &grad, &mut a.1, &mut a.2, &mut a.3,
                    )?;
                    a.6.add_pullback_with_q_geometry(self, row, &q_geom, &grad, &hess)?;
                    Ok(())
                },
                |total, chunk| {
                    total.0 += chunk.0;
                    total.1 += &chunk.1;
                    total.2 += &chunk.2;
                    total.3 += &chunk.3;
                    total.4 += &chunk.4;
                    total.5 += &chunk.5;
                    total.6.add(&chunk.6);
                },
            )?;

        let mut score_psi = Array1::zeros(slices.total);
        score_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(Arc::new(acc.into_operator(slices))),
        }))
    }

    fn sigma_exact_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.sigma_exact_joint_psisecond_order_terms_with_options(
            block_states,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psisecond_order_terms`. See
    /// `sigma_exact_joint_psi_terms_with_options` for the row-iter / weighting
    /// contract.
    pub(crate) fn sigma_exact_joint_psisecond_order_terms_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.flex_active() {
            return Ok(None);
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Bit-deterministic reduction: see `chunked_row_reduction`.
        let (objective_psi_psi, score_t, score_m, score_g, score_h, score_w, acc) =
            chunked_row_reduction(
                row_iter.as_slice(),
                || {
                    (
                        0.0,
                        Array1::zeros(p_t),
                        Array1::zeros(p_m),
                        Array1::zeros(p_g),
                        Array1::zeros(p_h),
                        Array1::zeros(p_w),
                        BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
                    )
                },
                |row, a| -> Result<(), String> {
                    let (mut obj, mut grad, mut hess) =
                        self.row_sigma_primary_terms(row, block_states, true)?;
                    let w = row_weights[row];
                    if w != 1.0 {
                        obj *= w;
                        grad.mapv_inplace(|v| v * w);
                        hess.mapv_inplace(|v| v * w);
                    }
                    a.0 += obj;
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    self.accumulate_score_with_q_geometry(
                        row, &q_geom, &grad, &mut a.1, &mut a.2, &mut a.3,
                    )?;
                    a.6.add_pullback_with_q_geometry(self, row, &q_geom, &grad, &hess)?;
                    Ok(())
                },
                |total, chunk| {
                    total.0 += chunk.0;
                    total.1 += &chunk.1;
                    total.2 += &chunk.2;
                    total.3 += &chunk.3;
                    total.4 += &chunk.4;
                    total.5 += &chunk.5;
                    total.6.add(&chunk.6);
                },
            )?;

        let mut score_psi_psi = Array1::zeros(slices.total);
        score_psi_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(acc.into_operator(slices))),
        }))
    }

    fn sigma_exact_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.sigma_exact_joint_psihessian_directional_derivative_with_options(
            block_states,
            d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `sigma_exact_joint_psihessian_directional_derivative`.
    /// See `sigma_exact_joint_psi_terms_with_options` for the row-iter /
    /// weighting contract — the returned dense Hessian-derivative matrix is
    /// accumulated with per-row inverse-inclusion weights when a subsample is active.
    pub(crate) fn sigma_exact_joint_psihessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.flex_active() {
            return Ok(None);
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let primary_dim = N_PRIMARY;
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Sigma scale jets and the zero primary direction are constant
        // across rows; resolve once outside the fold instead of rebuilding
        // per-row (and per (a,b) pair) inside it.
        let scale_grad = self.sigma_scale_jet(3, &[1], &[])?;
        let scale_hess = self.sigma_scale_jet(4, &[1], &[])?;
        let zero = zero_primary_direction_ref();
        // Bit-deterministic reduction: see `chunked_row_reduction`.
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
            |row, acc| -> Result<(), String> {
                let row_dir = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let mut grad = Array1::<f64>::zeros(primary_dim);
                for a in 0..primary_dim {
                    let da = unit_primary_direction_ref(a);
                    grad[a] = self.row_neglog_directional_with_scale_jet(
                        row,
                        block_states,
                        &[zero, &row_dir, da],
                        &scale_grad,
                    )?;
                }
                let mut hess = Array2::<f64>::zeros((primary_dim, primary_dim));
                for a in 0..primary_dim {
                    let da = unit_primary_direction_ref(a);
                    for b in a..primary_dim {
                        let db = unit_primary_direction_ref(b);
                        let value = self.row_neglog_directional_with_scale_jet(
                            row,
                            block_states,
                            &[zero, &row_dir, da, db],
                            &scale_hess,
                        )?;
                        hess[[a, b]] = value;
                        hess[[b, a]] = value;
                    }
                }
                let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                let w = row_weights[row];
                if w != 1.0 {
                    grad.mapv_inplace(|v| v * w);
                    hess.mapv_inplace(|v| v * w);
                }
                acc.add_pullback_with_q_geometry(self, row, &q_geom, &grad, &hess)?;
                Ok(())
            },
            |total, chunk| {
                total.add(&chunk);
            },
        )?;
        Ok(Some(acc.to_dense(&slices)))
    }

    fn flex_timewiggle_active(&self) -> bool {
        self.time_wiggle_ncols > 0
    }

    fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.design_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        (p_total - p_w)..p_total
    }

    fn time_wiggle_first_order_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleFirstOrderGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
        let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
        let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle basis/beta mismatch: B..B''={},{},{} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq_dq0 = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2q_dq02 = fast_av(&basis_d2, &beta_w);
        Ok(Some(SurvivalTimeWiggleFirstOrderGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
        }))
    }

    fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalTimeWiggleGeometry>, String> {
        let first = self.time_wiggle_first_order_geometry(h0, beta_w)?;
        let Some(first) = first else {
            return Ok(None);
        };
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis_d3 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 3)?;
        let basis_d4 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 4)?;
        let basis_d5 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 5)?;
        if basis_d3.ncols() != beta_w.len()
            || basis_d4.ncols() != beta_w.len()
            || basis_d5.ncols() != beta_w.len()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope timewiggle high-order basis/beta mismatch: B'''..B'''''={},{},{} betaw={}",
                    basis_d3.ncols(),
                    basis_d4.ncols(),
                    basis_d5.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let d3q_dq03 = fast_av(&basis_d3, &beta_w);
        let d4q_dq04 = fast_av(&basis_d4, &beta_w);
        let d5q_dq05 = fast_av(&basis_d5, &beta_w);
        Ok(Some(SurvivalTimeWiggleGeometry {
            basis: first.basis,
            basis_d1: first.basis_d1,
            basis_d2: first.basis_d2,
            basis_d3,
            basis_d4,
            dq_dq0: first.dq_dq0,
            d2q_dq02: first.d2q_dq02,
            d3q_dq03,
            d4q_dq04,
            d5q_dq05,
        }))
    }

    fn row_dynamic_q_values(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowValues, String> {
        let beta_time = &block_states[0].beta;
        if !self.flex_timewiggle_active() {
            return Ok(SurvivalMarginalSlopeDynamicRowValues {
                q0: self.design_entry.dot_row(row, beta_time)
                    + self.offset_entry[row]
                    + block_states[1].eta[row],
                q1: self.design_exit.dot_row(row, beta_time)
                    + self.offset_exit[row]
                    + block_states[1].eta[row],
                qd1: self.design_derivative_exit.dot_row(row, beta_time)
                    + self.derivative_offset_exit[row],
            });
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_values design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but value geometry could not be built at exit"
                    .to_string()
            })?;

        Ok(SurvivalMarginalSlopeDynamicRowValues {
            q0: h0 + entry_geom.basis.row(0).dot(&beta_time_w),
            q1: h1 + exit_geom.basis.row(0).dot(&beta_time_w),
            qd1: exit_geom.dq_dq0[0] * d_raw,
        })
    }

    fn row_dynamic_q_gradient(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRowGradient, String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        let mut out = SurvivalMarginalSlopeDynamicRowGradient {
            q0: 0.0,
            q1: 0.0,
            qd1: 0.0,
            dq0_time: Array1::zeros(p_time),
            dq1_time: Array1::zeros(p_time),
            dqd1_time: Array1::zeros(p_time),
            dq0_marginal: Array1::zeros(p_marginal),
            dq1_marginal: Array1::zeros(p_marginal),
            dqd1_marginal: Array1::zeros(p_marginal),
        };

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
            out.dq0_time.assign(&time_entry_chunk.row(0));
            out.dq1_time.assign(&time_exit_chunk.row(0));
            out.dqd1_time.assign(&time_deriv_chunk.row(0));
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0);
                out.dq0_marginal.assign(&marginal_row);
                out.dq1_marginal.assign(&marginal_row);
            }
            return Ok(out);
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_gradient design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_row = if p_marginal > 0 {
            let marginal_chunk = self
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_gradient marginal_design: {e}"))?;
            Some(marginal_chunk.row(0).to_owned())
        } else {
            None
        };

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_first_order_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but gradient geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
            }
        }

        Ok(out)
    }

    fn row_dynamic_q_geometry(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalMarginalSlopeDynamicRow, String> {
        let mut out = SurvivalMarginalSlopeDynamicRow::empty_workspace();
        self.row_dynamic_q_geometry_into(row, block_states, &mut out)?;
        Ok(out)
    }

    /// Pooled-buffer variant of [`row_dynamic_q_geometry`]. Resizes and
    /// zero-fills `out` in place, then writes the same row geometry into the
    /// caller-owned workspace. Used by hot rayon `try_fold` accumulators that
    /// thread one `SurvivalMarginalSlopeDynamicRow` per worker thread to
    /// eliminate the ~250 KB-per-call allocator traffic the fresh-allocation
    /// path would incur at biobank-scale `n`.
    fn row_dynamic_q_geometry_into(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        out: &mut SurvivalMarginalSlopeDynamicRow,
    ) -> Result<(), String> {
        let beta_time = &block_states[0].beta;
        let beta_marginal = &block_states[1].beta;
        let p_time = beta_time.len();
        let p_marginal = beta_marginal.len();

        out.reset(p_time, p_marginal);

        if !self.flex_timewiggle_active() {
            out.q0 = self.design_entry.dot_row(row, beta_time)
                + self.offset_entry[row]
                + block_states[1].eta[row];
            out.q1 = self.design_exit.dot_row(row, beta_time)
                + self.offset_exit[row]
                + block_states[1].eta[row];
            out.qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                + self.derivative_offset_exit[row];
            let time_entry_chunk = self
                .design_entry
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
            let time_exit_chunk = self
                .design_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
            let time_deriv_chunk = self
                .design_derivative_exit
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
            let time_row_entry = time_entry_chunk.row(0).to_owned();
            let time_row_exit = time_exit_chunk.row(0).to_owned();
            let time_row_deriv = time_deriv_chunk.row(0).to_owned();
            out.dq0_time.assign(&time_row_entry.view());
            out.dq1_time.assign(&time_row_exit.view());
            out.dqd1_time.assign(&time_row_deriv.view());
            if p_marginal > 0 {
                let marginal_chunk = self
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?;
                let marginal_row = marginal_chunk.row(0).to_owned();
                out.dq0_marginal.assign(&marginal_row.view());
                out.dq1_marginal.assign(&marginal_row.view());
            }
            return Ok(());
        }

        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_base = beta_time.slice(s![..p_base]);
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("row_dynamic_q_geometry design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_row = if p_marginal > 0 {
            let marginal_chunk = self
                .marginal_design
                .try_row_chunk(row..row + 1)
                .map_err(|e| format!("row_dynamic_q_geometry marginal_design: {e}"))?;
            Some(marginal_chunk.row(0).to_owned())
        } else {
            None
        };

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time_base) + self.offset_entry[row] + base_marginal;
        let h1 = x_exit_base.dot(&beta_time_base) + self.offset_exit[row] + base_marginal;
        let d_raw = x_deriv_base.dot(&beta_time_base) + self.derivative_offset_exit[row];

        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at entry"
                    .to_string()
            })?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| {
                "survival marginal-slope timewiggle metadata is present but geometry could not be built at exit"
                    .to_string()
            })?;

        out.q0 = h0 + entry_geom.basis.row(0).dot(&beta_time_w);
        out.q1 = h1 + exit_geom.basis.row(0).dot(&beta_time_w);
        out.qd1 = exit_geom.dq_dq0[0] * d_raw;

        for j in 0..p_base {
            out.dq0_time[j] = entry_geom.dq_dq0[0] * x_entry_base[j];
            out.dq1_time[j] = exit_geom.dq_dq0[0] * x_exit_base[j];
            out.dqd1_time[j] = exit_geom.d2q_dq02[0] * d_raw * x_exit_base[j]
                + exit_geom.dq_dq0[0] * x_deriv_base[j];
            for k in 0..p_base {
                out.d2q0_time_time[[j, k]] =
                    entry_geom.d2q_dq02[0] * x_entry_base[j] * x_entry_base[k];
                out.d2q1_time_time[[j, k]] =
                    exit_geom.d2q_dq02[0] * x_exit_base[j] * x_exit_base[k];
                out.d2qd1_time_time[[j, k]] =
                    exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j] * x_exit_base[k]
                        + exit_geom.d2q_dq02[0]
                            * (x_exit_base[j] * x_deriv_base[k] + x_deriv_base[j] * x_exit_base[k]);
            }
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            out.dq0_time[coeff_idx] = entry_geom.basis[[0, local_idx]];
            out.dq1_time[coeff_idx] = exit_geom.basis[[0, local_idx]];
            out.dqd1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * d_raw;
            for j in 0..p_base {
                let q0_tw = entry_geom.basis_d1[[0, local_idx]] * x_entry_base[j];
                let q1_tw = exit_geom.basis_d1[[0, local_idx]] * x_exit_base[j];
                out.d2q0_time_time[[j, coeff_idx]] = q0_tw;
                out.d2q0_time_time[[coeff_idx, j]] = q0_tw;
                out.d2q1_time_time[[j, coeff_idx]] = q1_tw;
                out.d2q1_time_time[[coeff_idx, j]] = q1_tw;
                let qd1_cross = exit_geom.basis_d2[[0, local_idx]] * d_raw * x_exit_base[j]
                    + exit_geom.basis_d1[[0, local_idx]] * x_deriv_base[j];
                out.d2qd1_time_time[[j, coeff_idx]] = qd1_cross;
                out.d2qd1_time_time[[coeff_idx, j]] = qd1_cross;
            }
        }

        if let Some(marginal_row) = marginal_row.as_ref() {
            for j in 0..p_marginal {
                out.dq0_marginal[j] = entry_geom.dq_dq0[0] * marginal_row[j];
                out.dq1_marginal[j] = exit_geom.dq_dq0[0] * marginal_row[j];
                out.dqd1_marginal[j] = exit_geom.d2q_dq02[0] * d_raw * marginal_row[j];
                for k in 0..p_marginal {
                    out.d2q0_marginal_marginal[[j, k]] =
                        entry_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2q1_marginal_marginal[[j, k]] =
                        exit_geom.d2q_dq02[0] * marginal_row[j] * marginal_row[k];
                    out.d2qd1_marginal_marginal[[j, k]] =
                        exit_geom.d3q_dq03[0] * d_raw * marginal_row[j] * marginal_row[k];
                }
                for k in 0..p_base {
                    out.d2q0_time_marginal[[k, j]] =
                        entry_geom.d2q_dq02[0] * x_entry_base[k] * marginal_row[j];
                    out.d2q1_time_marginal[[k, j]] =
                        exit_geom.d2q_dq02[0] * x_exit_base[k] * marginal_row[j];
                    out.d2qd1_time_marginal[[k, j]] =
                        exit_geom.d3q_dq03[0] * d_raw * x_exit_base[k] * marginal_row[j]
                            + exit_geom.d2q_dq02[0] * x_deriv_base[k] * marginal_row[j];
                }
                for local_idx in 0..time_tail.len() {
                    let coeff_idx = time_tail.start + local_idx;
                    out.d2q0_time_marginal[[coeff_idx, j]] =
                        entry_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2q1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d1[[0, local_idx]] * marginal_row[j];
                    out.d2qd1_time_marginal[[coeff_idx, j]] =
                        exit_geom.basis_d2[[0, local_idx]] * d_raw * marginal_row[j];
                }
            }
        }

        Ok(())
    }

    fn timewiggle_marginal_psi_row_lift(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        primary_layout: Option<&FlexPrimarySlices>,
        psi_row: &Array1<f64>,
        beta_marginal: &Array1<f64>,
    ) -> Result<TimewiggleMarginalPsiRowLift, String> {
        let beta_time = &block_states[0].beta;
        let p_time = beta_time.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let entry_chunk = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_entry: {e}"))?;
        let exit_chunk = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_exit: {e}"))?;
        let deriv_chunk = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift design_derivative_exit: {e}"))?;
        let x_entry_base = entry_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_exit_base = exit_chunk.row(0).slice(s![..p_base]).to_owned();
        let x_deriv_base = deriv_chunk.row(0).slice(s![..p_base]).to_owned();
        let marginal_chunk = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_marginal_psi_row_lift marginal_design: {e}"))?;
        let marginal_row = marginal_chunk.row(0).to_owned();

        let base_marginal = block_states[1].eta[row];
        let h0 = x_entry_base.dot(&beta_time.slice(s![..p_base]))
            + self.offset_entry[row]
            + base_marginal;
        let h1 =
            x_exit_base.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + base_marginal;
        let d_raw =
            x_deriv_base.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let entry_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "missing entry timewiggle geometry for marginal psi lift".to_string())?;
        let exit_geom = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "missing exit timewiggle geometry for marginal psi lift".to_string())?;

        let mu = psi_row.dot(beta_marginal);
        let mut dir =
            Array1::<f64>::zeros(primary_layout.map_or(N_PRIMARY, |primary| primary.total));
        let q0_idx = primary_layout.map_or(0, |primary| primary.q0);
        let q1_idx = primary_layout.map_or(1, |primary| primary.q1);
        let qd1_idx = primary_layout.map_or(2, |primary| primary.qd1);
        dir[q0_idx] = entry_geom.dq_dq0[0] * mu;
        dir[q1_idx] = exit_geom.dq_dq0[0] * mu;
        dir[qd1_idx] = exit_geom.d2q_dq02[0] * d_raw * mu;

        let mut u_q0_time = Array1::<f64>::zeros(p_time);
        let mut u_q1_time = Array1::<f64>::zeros(p_time);
        let mut u_qd1_time = Array1::<f64>::zeros(p_time);
        for j in 0..p_base {
            u_q0_time[j] = entry_geom.d2q_dq02[0] * mu * x_entry_base[j];
            u_q1_time[j] = exit_geom.d2q_dq02[0] * mu * x_exit_base[j];
            u_qd1_time[j] = mu
                * (exit_geom.d3q_dq03[0] * d_raw * x_exit_base[j]
                    + exit_geom.d2q_dq02[0] * x_deriv_base[j]);
        }
        for local_idx in 0..time_tail.len() {
            let coeff_idx = time_tail.start + local_idx;
            u_q0_time[coeff_idx] = entry_geom.basis_d1[[0, local_idx]] * mu;
            u_q1_time[coeff_idx] = exit_geom.basis_d1[[0, local_idx]] * mu;
            u_qd1_time[coeff_idx] = exit_geom.basis_d2[[0, local_idx]] * d_raw * mu;
        }

        let u_q0_marginal =
            psi_row * entry_geom.dq_dq0[0] + &marginal_row * (entry_geom.d2q_dq02[0] * mu);
        let u_q1_marginal =
            psi_row * exit_geom.dq_dq0[0] + &marginal_row * (exit_geom.d2q_dq02[0] * mu);
        let u_qd1_marginal = psi_row * (exit_geom.d2q_dq02[0] * d_raw)
            + &marginal_row * (exit_geom.d3q_dq03[0] * d_raw * mu);

        Ok(TimewiggleMarginalPsiRowLift {
            dir,
            u_q0_time,
            u_q1_time,
            u_qd1_time,
            u_q0_marginal,
            u_q1_marginal,
            u_qd1_marginal,
            x_entry_base,
            x_exit_base,
            x_deriv_base,
            marginal_row,
            entry_basis_d1: entry_geom.basis_d1.row(0).to_owned(),
            entry_basis_d2: entry_geom.basis_d2.row(0).to_owned(),
            exit_basis_d1: exit_geom.basis_d1.row(0).to_owned(),
            exit_basis_d2: exit_geom.basis_d2.row(0).to_owned(),
            exit_basis_d3: exit_geom.basis_d3.row(0).to_owned(),
            entry_m2: entry_geom.d2q_dq02[0],
            entry_m3: entry_geom.d3q_dq03[0],
            exit_m2: exit_geom.d2q_dq02[0],
            exit_m3: exit_geom.d3q_dq03[0],
            exit_m4: exit_geom.d4q_dq04[0],
            d_raw,
            mu,
            psi_row: psi_row.clone(),
        })
    }

    fn time_derivative_lower_bound(&self) -> f64 {
        assert!(self.derivative_guard.is_finite() && self.derivative_guard > 0.0);
        self.derivative_guard
    }

    fn flex_active(&self) -> bool {
        self.score_warp.is_some() || self.link_dev.is_some()
    }

    fn effective_flex_active(&self, block_states: &[ParameterBlockState]) -> Result<bool, String> {
        if self.score_warp.is_some() && self.flex_score_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival score-warp block state".to_string(),
            }
            .into());
        }
        if self.link_dev.is_some() && self.flex_link_beta(block_states)?.is_none() {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "missing survival link-deviation block state".to_string(),
            }
            .into());
        }
        Ok(self.flex_active())
    }

    fn flex_score_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.score_warp.is_none() {
            return Ok(None);
        }
        block_states
            .get(3)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival score-warp block state".to_string())
    }

    fn flex_link_beta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<Option<&'a Array1<f64>>, String> {
        if self.link_dev.is_none() {
            return Ok(None);
        }
        let idx = if self.score_warp.is_some() { 4 } else { 3 };
        block_states
            .get(idx)
            .map(|state| Some(&state.beta))
            .ok_or_else(|| "missing survival link-deviation block state".to_string())
    }

    fn denested_partition_cells(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<Vec<exact_kernel::DenestedPartitionCell>, String> {
        if self.score_dim() == 1 {
            return shared_denested_partition_cells(
                a,
                b,
                self.score_warp.as_ref(),
                beta_h,
                self.link_dev.as_ref(),
                beta_w,
                self.probit_frailty_scale(),
            );
        }
        let score_breaks = self
            .score_warp
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let link_breaks = self
            .link_dev
            .as_ref()
            .map(|runtime| runtime.breakpoints().to_vec())
            .unwrap_or_default();
        let mut cells = exact_kernel::build_denested_partition_cells_with_tails(
            a,
            b,
            &score_breaks,
            &link_breaks,
            |z| self.score_warp_local_cubic_at(beta_h, z),
            |u| {
                if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w) {
                    runtime.local_cubic_at(beta_w, u)
                } else {
                    Ok(Self::zero_score_warp_span())
                }
            },
        )?;
        let scale = self.probit_frailty_scale();
        if scale != 1.0 {
            for partition_cell in &mut cells {
                partition_cell.cell.c0 *= scale;
                partition_cell.cell.c1 *= scale;
                partition_cell.cell.c2 *= scale;
                partition_cell.cell.c3 *= scale;
            }
        }
        Ok(cells)
    }

    fn evaluate_denested_survival_calibration(
        &self,
        a: f64,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, f64), String> {
        let cells = self.denested_partition_cells(a, slope, beta_h, beta_w)?;
        let scale = self.probit_frailty_scale();
        let mut f = -crate::probability::normal_cdf(-q);
        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        for partition_cell in cells {
            let pos_cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: pos_cell.left,
                right: pos_cell.right,
                c0: -pos_cell.c0,
                c1: -pos_cell.c1,
                c2: -pos_cell.c2,
                c3: -pos_cell.c3,
            };
            let state = exact_kernel::evaluate_cell_moments(neg_cell, 9)?;
            f += state.value;
            let (dc_da_pos, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let (dc_daa_pos, _, _) = exact_kernel::denested_cell_second_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                slope,
            );
            let dc_da = scale_coeff4(dc_da_pos, -scale);
            let dc_daa = scale_coeff4(dc_daa_pos, -scale);
            f_a += exact_kernel::cell_first_derivative_from_moments(&dc_da, &state.moments)?;
            f_aa += exact_kernel::cell_second_derivative_from_moments(
                neg_cell,
                &dc_da,
                &dc_da,
                &dc_daa,
                &state.moments,
            )?;
        }
        Ok((f, f_a, f_aa))
    }

    #[cfg(test)]
    fn solve_row_survival_intercept(
        &self,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        self.solve_row_survival_intercept_with_slot(q, slope, beta_h, beta_w, None)
    }

    fn solve_row_survival_intercept_with_slot(
        &self,
        q: f64,
        slope: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        slot: Option<(usize, SurvivalInterceptSlotKind)>,
    ) -> Result<(f64, f64), String> {
        let eval = |a: f64| -> Result<(f64, f64, f64), String> {
            #[cfg(test)]
            {
                survival_intercept_test_counter::bump();
            }
            self.evaluate_denested_survival_calibration(a, q, slope, beta_h, beta_w)
        };
        let probit_scale = self.probit_frailty_scale();
        let a_closed_form = q * rigid_observed_scale(slope, probit_scale) / probit_scale;

        // Prefer the previous PIRLS iter's converged intercept as the initial
        // guess; β changes only a little between consecutive PIRLS iterations,
        // so the previous answer is typically within a few root-solver steps
        // of the new one. If the slot is None (no cache wired) or the stored
        // bits decode to a non-finite value (uninitialised NaN sentinel /
        // stale), fall back to the closed-form rigid seed — preserving the
        // exact pre-warm-start behaviour.
        let cached_a = slot.and_then(|(row, kind)| {
            self.intercept_warm_starts
                .as_ref()
                .and_then(|cache| cache.load(row, kind))
        });
        let a_init = cached_a.unwrap_or(a_closed_form);
        let mut solve_result = super::monotone_root::solve_monotone_root_detailed(
            &eval,
            a_init,
            "survival intercept",
            1e-12,
            64,
            64,
        );
        // If the warm-started solve failed, retry once from the closed-form
        // seed. Cached `a` from a prior PIRLS iter can be far enough from the
        // current root (e.g., after a large β step) that the bracketing search
        // exhausts; the closed-form seed always sits in the correct basin.
        if cached_a.is_some() && solve_result.is_err() {
            solve_result = super::monotone_root::solve_monotone_root_detailed(
                &eval,
                a_closed_form,
                "survival intercept",
                1e-12,
                64,
                64,
            );
        }
        // This routine also emits its own format!()-based String errors below
        // (non-finite derivative, residual rejection), so the enclosing return
        // type stays Result<_, String>; convert the typed solver error here.
        let solution = solve_result.map_err(|e| e.to_string())?;
        let a = solution.root;
        // The solver already evaluated `eval` at `solution.root` during the
        // refine loop and returned the resulting `residual` (best_f) and
        // `abs_deriv` (best_abs_deriv). Reusing them here saves one full
        // calibration evaluation per row × 2 (entry + exit) per joint-Newton
        // sweep — at biobank n=320k this is 640k spared evaluations per pass.
        let residual = solution.residual;
        let abs_deriv = solution.abs_deriv;
        if !abs_deriv.is_finite() || abs_deriv == 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope intercept solve failed: \
                     zero or non-finite derivative at a={a:.6}"
                ),
            }
            .into());
        }

        let target_survival = crate::probability::normal_cdf(-q);
        let achieved_survival = target_survival + residual;
        let tail_mass = target_survival.min(1.0 - target_survival).max(0.0);
        let probability_tol = SURVIVAL_INTERCEPT_ABS_RESIDUAL_TOL
            .max(SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL * tail_mass);
        let mut log_tail_residual = None;
        // Always accept if probability-space residual is within tolerance:
        // a perfectly-converged probability solve (residual=0) is the best
        // achievable answer, and rejecting it because the deep-tail log
        // computation has its own floating-point noise (~6e-8 at |q|>=7)
        // would discard a correct intercept. When tail_mass is small we
        // *additionally* accept tight log-space agreement, so well-resolved
        // tails that drift slightly outside the absolute probability_tol
        // (which can be ulp-bounded) still validate.
        let residual_ok = if tail_mass < SURVIVAL_INTERCEPT_LOG_TAIL_THRESHOLD {
            let probability_pass = residual.abs() <= probability_tol;
            let (achieved_tail, target_log_tail) = if target_survival <= 0.5 {
                let (target_log_survival, _) = signed_probit_logcdf_and_mills_ratio(-q);
                (achieved_survival, target_log_survival)
            } else {
                let (target_log_failure, _) = signed_probit_logcdf_and_mills_ratio(q);
                (1.0 - achieved_survival, target_log_failure)
            };
            let log_pass = if target_log_tail.is_finite()
                && achieved_tail.is_finite()
                && achieved_tail > 0.0
            {
                let log_residual = achieved_tail.ln() - target_log_tail;
                log_tail_residual = Some(log_residual);
                log_residual.abs() <= SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL
            } else {
                false
            };
            probability_pass || log_pass
        } else {
            residual.abs() <= probability_tol
        };

        if !residual_ok {
            let log_tail_detail = log_tail_residual
                .map(|value| format!(", log_tail_residual={value:.3e}"))
                .unwrap_or_default();
            return Err(SurvivalMarginalSlopeError::IntegrationFailed {
                reason: format!(
                    "survival marginal-slope intercept solve failed: \
                     residual={residual:.3e} at a={a:.6}, target survival={target_survival:.6e}, \
                     achieved survival={achieved_survival:.6e}, probability_tol={probability_tol:.3e}\
                     {log_tail_detail}"
                ),
            }
            .into());
        }

        // Cache the converged intercept for the next PIRLS iter, if a slot
        // was provided. When `slot` is None this is a no-op, preserving the
        // exact pre-warm-start behaviour.
        if let Some((row, kind)) = slot {
            if let Some(cache) = self.intercept_warm_starts.as_ref() {
                cache.store(row, kind, a);
            }
        }

        Ok((a, abs_deriv))
    }

    fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.time_linear_constraints.as_ref() else {
            return Ok(None);
        };
        if beta.len() != constraints.a.ncols() || delta.len() != constraints.a.ncols() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time-step dimension mismatch: beta={}, delta={}, expected {}",
                    beta.len(),
                    delta.len(),
                    constraints.a.ncols()
                ),
            }
            .into());
        }
        let mut alpha = 1.0f64;
        for row in 0..constraints.a.nrows() {
            let a_row = constraints.a.row(row);
            let slack = a_row.dot(beta) - constraints.b[row];
            if slack < -1e-10 {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival marginal-slope current time block violates derivative guard at row {row}: slack={slack:.3e}"
                    ),
                }
                .into());
            }
            let drift = a_row.dot(delta);
            if drift < 0.0 {
                alpha = alpha.min((slack / -drift).clamp(0.0, 1.0));
            }
        }
        if alpha >= 1.0 {
            Ok(Some(1.0))
        } else {
            Ok(Some((0.995 * alpha).clamp(0.0, 1.0)))
        }
    }

    /// Project a proposed time-block β so that every row's
    /// `qd1 = D · β + offset ≥ derivative_guard` is satisfied with a
    /// strictly positive margin. Bisects from `current` (assumed feasible)
    /// toward `proposed` along the line segment, returning the largest
    /// `α ∈ [0, 1]` such that `current + α·(proposed − current)` clears
    /// the guard on all rows, then pulls back by 0.5 % to stay strictly
    /// interior.
    ///
    /// Mirrors the role `project_monotone_feasible_beta` plays for
    /// `score_warp` and `link_dev`. Without this projection, PIRLS' trust
    /// region accepts time-block steps that walk right up to the
    /// `qd1 ≥ guard` cliff (and across it under any subsequent
    /// re-evaluation), because `row_primary_closed_form` enforces the
    /// constraint only as an `Err` cliff at evaluation time — leaving no
    /// repulsion from the boundary. Closed-form max-α suffices because
    /// `qd1` is linear in β, so per-row feasibility along the segment is
    /// `α · drift_r ≥ guard − qd1_current_r` for each row `r`.
    fn project_time_qd1_feasible(
        &self,
        current: &Array1<f64>,
        proposed: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let p = current.len();
        if p == 0 || proposed.len() == 0 {
            return Ok(proposed);
        }
        if proposed.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time-block projection length mismatch: current={p}, proposed={}",
                    proposed.len()
                ),
            }
            .into());
        }
        let n_rows = self.derivative_offset_exit.len();
        if n_rows == 0 {
            return Ok(proposed);
        }
        let qd_design_current = self.design_derivative_exit.matrixvectormultiply(current);
        let qd_design_proposed = self.design_derivative_exit.matrixvectormultiply(&proposed);
        if qd_design_current.len() != n_rows || qd_design_proposed.len() != n_rows {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time-block projection row count mismatch: design rows={} vs offset rows={n_rows}",
                    qd_design_current.len()
                ),
            }
            .into());
        }
        let guard = self.derivative_guard;
        let mut alpha = 1.0_f64;
        let mut violated = false;
        let mut worst_current_qd1 = f64::INFINITY;
        let mut worst_current_row = 0usize;
        for row in 0..n_rows {
            let offset = self.derivative_offset_exit[row];
            let qd1_current = qd_design_current[row] + offset;
            if qd1_current < worst_current_qd1 {
                worst_current_qd1 = qd1_current;
                worst_current_row = row;
            }
            let qd1_proposed = qd_design_proposed[row] + offset;
            if !survival_derivative_guard_violated(qd1_proposed, guard) {
                continue;
            }
            violated = true;
            let drift = qd1_proposed - qd1_current;
            if drift >= 0.0 {
                // Proposed violates but current is at-or-above the guard
                // (caught below) and drift is non-negative — only possible
                // if `current` itself violates by more than `proposed`,
                // which the explicit feasibility check below handles.
                continue;
            }
            // qd1(α) = qd1_current + α · drift, drift < 0.
            // qd1(α) ≥ guard ⟺ α ≤ (qd1_current − guard) / (−drift).
            let row_max = ((qd1_current - guard) / -drift).clamp(0.0, 1.0);
            if row_max < alpha {
                alpha = row_max;
            }
        }
        if survival_derivative_guard_violated(worst_current_qd1, guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope time-block current beta violates monotonicity at row {worst_current_row}: \
                     qd1={worst_current_qd1:.3e} < guard={guard:.3e}"
                ),
            }
            .into());
        }
        if !violated {
            return Ok(proposed);
        }
        // Stay strictly interior to the constraint surface so that
        // downstream re-evaluations (PIRLS gradient reload, REML
        // coord-corrections trace, BFGS line-search probes) survive ULP
        // jitter without crossing the cliff.
        let alpha_safe = (0.995 * alpha).clamp(0.0, 1.0);
        let direction = &proposed - current;
        Ok(current + &direction.mapv(|v| v * alpha_safe))
    }

    fn validate_exact_monotonicity(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(), String> {
        if let Some(runtime) = &self.score_warp {
            let beta_h = self
                .flex_score_beta(block_states)?
                .ok_or_else(|| "missing survival score-warp coefficients".to_string())?;
            let expected = runtime.basis_dim() * self.score_dim();
            if beta_h.len() != expected {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: format!(
                        "survival score-warp beta length mismatch: got {}, expected {expected} for K={} and basis dim {}",
                        beta_h.len(),
                        self.score_dim(),
                        runtime.basis_dim()
                    ),
                }
                .into());
            }
            for coord in 0..self.score_dim() {
                let local_beta = self.score_warp_beta_for_coord(beta_h, coord)?;
                runtime.monotonicity_feasible(
                    &local_beta,
                    &format!("survival marginal-slope score-warp[z{coord}]"),
                )?;
            }
        }
        if let Some(runtime) = &self.link_dev {
            let beta_w = self
                .flex_link_beta(block_states)?
                .ok_or_else(|| "missing survival link-deviation coefficients".to_string())?;
            runtime.monotonicity_feasible(beta_w, "survival marginal-slope link deviation")?;
        }
        Ok(())
    }

    fn observed_denested_eta_chi(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<(f64, f64), String> {
        let z_obs = self.observed_score_projection(row);
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta = eval_coeff4_at(&obs.coeff, z_obs);
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        Ok((eta, chi))
    }

    fn observed_denested_cell_partials(
        &self,
        row: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<ObservedDenestedCellPartials, String> {
        let z_obs = self.observed_score_projection(row);
        if self.score_dim() == 1 {
            return shared_observed_denested_cell_partials(
                z_obs,
                a,
                b,
                self.score_warp.as_ref(),
                beta_h,
                self.link_dev.as_ref(),
                beta_w,
                self.probit_frailty_scale(),
            );
        }

        // Observed vector-z contribution is the row-wise direct-sum value
        //     h_i = sum_k W_k(z_ik) beta_k.
        // In the denested additive transport, h_i enters as b*h_i at the
        // observed row.  Holding z_i fixed makes this a constant coefficient
        // in the observed polynomial while preserving the standard link
        // partials in a and b.
        let h_obs = self.score_warp_observed_value(row, beta_h)?;
        let u_obs = a + b * z_obs;
        let link_span = if let (Some(runtime), Some(beta_w)) = (self.link_dev.as_ref(), beta_w) {
            runtime.local_cubic_at(beta_w, u_obs)?
        } else {
            Self::zero_score_warp_span()
        };
        let (d0, d1, d2, d3) = exact_kernel::transformed_link_cubic(link_span, a, b);
        let coeff_raw = [a + b * h_obs + d0, b + d1, d2, d3];
        let shift = a - link_span.left;
        let alpha1 = link_span.c1;
        let alpha2 = link_span.c2;
        let alpha3 = link_span.c3;
        let dc_da_raw = [
            1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
            b * (2.0 * alpha2 + 6.0 * alpha3 * shift),
            3.0 * alpha3 * b * b,
            0.0,
        ];
        let dc_db_raw = [
            h_obs,
            1.0 + alpha1 + 2.0 * alpha2 * shift + 3.0 * alpha3 * shift * shift,
            2.0 * b * (alpha2 + 3.0 * alpha3 * shift),
            3.0 * alpha3 * b * b,
        ];
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact_kernel::link_basis_cell_second_partials(link_span, a, b);
        let (dc_daaa_raw, dc_daab_raw, dc_dabb_raw, dc_dbbb_raw) =
            exact_kernel::link_basis_cell_third_partials(link_span);
        let scale = self.probit_frailty_scale();
        Ok(ObservedDenestedCellPartials {
            coeff: scale_coeff4(coeff_raw, scale),
            dc_da: scale_coeff4(dc_da_raw, scale),
            dc_db: scale_coeff4(dc_db_raw, scale),
            dc_daa: scale_coeff4(dc_daa_raw, scale),
            dc_dab: scale_coeff4(dc_dab_raw, scale),
            dc_dbb: scale_coeff4(dc_dbb_raw, scale),
            dc_daaa: scale_coeff4(dc_daaa_raw, scale),
            dc_daab: scale_coeff4(dc_daab_raw, scale),
            dc_dabb: scale_coeff4(dc_dabb_raw, scale),
            dc_dbbb: scale_coeff4(dc_dbbb_raw, scale),
        })
    }

    fn denested_cell_primary_fixed_partials(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        score_span: exact_kernel::LocalSpanCubic,
        link_span: exact_kernel::LocalSpanCubic,
        z_basis: f64,
        u_basis: f64,
    ) -> Result<DenestedCellPrimaryFixedPartials, String> {
        let scale = self.probit_frailty_scale();
        let r = primary.total;
        let mut coeff_u = vec![[0.0; 4]; r];
        let mut coeff_au = vec![[0.0; 4]; r];
        let mut coeff_bu = vec![[0.0; 4]; r];
        let mut coeff_aau = vec![[0.0; 4]; r];
        let mut coeff_abu = vec![[0.0; 4]; r];
        let mut coeff_bbu = vec![[0.0; 4]; r];
        let mut coeff_aaau = vec![[0.0; 4]; r];
        let mut coeff_aabu = vec![[0.0; 4]; r];
        let mut coeff_abbu = vec![[0.0; 4]; r];
        let mut coeff_bbbu = vec![[0.0; 4]; r];

        let (dc_da_raw, dc_db_raw) =
            exact_kernel::denested_cell_coefficient_partials(score_span, link_span, a, b);
        let (dc_daa_raw, dc_dab_raw, dc_dbb_raw) =
            exact_kernel::denested_cell_second_partials(score_span, link_span, a, b);
        let (dc_daaa_raw, dc_daab_raw, dc_dabb_raw, dc_dbbb_raw) =
            exact_kernel::denested_cell_third_partials(link_span);
        let dc_da = scale_coeff4(dc_da_raw, scale);
        let dc_db = scale_coeff4(dc_db_raw, scale);
        let dc_daa = scale_coeff4(dc_daa_raw, scale);
        let dc_dab = scale_coeff4(dc_dab_raw, scale);
        let dc_dbb = scale_coeff4(dc_dbb_raw, scale);
        let dc_daaa = scale_coeff4(dc_daaa_raw, scale);
        let dc_daab = scale_coeff4(dc_daab_raw, scale);
        let dc_dabb = scale_coeff4(dc_dabb_raw, scale);
        let dc_dbbb = scale_coeff4(dc_dbbb_raw, scale);

        coeff_u[primary.g] = dc_db;
        coeff_au[primary.g] = dc_dab;
        coeff_bu[primary.g] = dc_dbb;
        coeff_aau[primary.g] = dc_daab;
        coeff_abu[primary.g] = dc_dabb;
        coeff_bbu[primary.g] = dc_dbbb;
        coeff_aaau[primary.g] = [0.0; 4];
        coeff_aabu[primary.g] = [0.0; 4];
        coeff_abbu[primary.g] = [0.0; 4];
        coeff_bbbu[primary.g] = [0.0; 4];

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                coeff_u[idx] = scale_coeff4(
                    self.integration_score_basis_coefficients(local_idx, z_basis, b)?,
                    scale,
                );
                coeff_bu[idx] = scale_coeff4(
                    self.integration_score_basis_coefficients(local_idx, z_basis, 1.0)?,
                    scale,
                );
            }
        }

        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_basis)?;
                let idx = w_range.start + local_idx;
                coeff_u[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                let (dc_aw_raw, dc_bw_raw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let link_second = exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaw_raw, dc_abw_raw, dc_bbw_raw) =
                    (link_second.0, link_second.1, link_second.2);
                let (dc_aaaw_raw, dc_aabw_raw, dc_abbw_raw, dc_bbbw_raw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                coeff_au[idx] = scale_coeff4(dc_aw_raw, scale);
                coeff_bu[idx] = scale_coeff4(dc_bw_raw, scale);
                coeff_aau[idx] = scale_coeff4(dc_aaw_raw, scale);
                coeff_abu[idx] = scale_coeff4(dc_abw_raw, scale);
                coeff_bbu[idx] = scale_coeff4(dc_bbw_raw, scale);
                coeff_aaau[idx] = scale_coeff4(dc_aaaw_raw, scale);
                coeff_aabu[idx] = scale_coeff4(dc_aabw_raw, scale);
                coeff_abbu[idx] = scale_coeff4(dc_abbw_raw, scale);
                coeff_bbbu[idx] = scale_coeff4(dc_bbbw_raw, scale);
            }
        }

        Ok(DenestedCellPrimaryFixedPartials {
            dc_da,
            dc_daa,
            dc_daaa,
            coeff_u,
            coeff_au,
            coeff_bu,
            coeff_aau,
            coeff_abu,
            coeff_bbu,
            coeff_aaau,
            coeff_aabu,
            coeff_abbu,
            coeff_bbbu,
        })
    }

    fn observed_fixed_eta_second_partial(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        row: usize,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(eval_coeff4_at(&obs.dc_dbb, z_obs));
        }
        if u == primary.g {
            if let Some(h_range) = primary.h.as_ref() {
                if v >= h_range.start && v < h_range.end {
                    let local_idx = v - h_range.start;
                    return Ok(eval_coeff4_at(
                        &scale_coeff4(
                            self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                            scale,
                        ),
                        z_obs,
                    ));
                }
            }
            if let Some(w_range) = primary.w.as_ref() {
                if v >= w_range.start && v < w_range.end {
                    let local_idx = v - w_range.start;
                    let runtime = self
                        .link_dev
                        .as_ref()
                        .ok_or_else(|| "missing survival link runtime".to_string())?;
                    let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                    let (_, dc_bw) =
                        exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                    return Ok(eval_coeff4_at(&scale_coeff4(dc_bw, scale), z_obs));
                }
            }
        }
        if v == primary.g {
            return self
                .observed_fixed_eta_second_partial(primary, obs, row, v, u, z_obs, u_obs, a, b);
        }
        Ok(0.0)
    }

    fn observed_fixed_chi_second_partial(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(eval_coeff4_at(&obs.dc_dabb, z_obs));
        }
        if u == primary.g {
            if let Some(w_range) = primary.w.as_ref() {
                if v >= w_range.start && v < w_range.end {
                    let local_idx = v - w_range.start;
                    let runtime = self
                        .link_dev
                        .as_ref()
                        .ok_or_else(|| "missing survival link runtime".to_string())?;
                    let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                    let (_, dc_abw, _) =
                        exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                    return Ok(eval_coeff4_at(&scale_coeff4(dc_abw, scale), z_obs));
                }
            }
        }
        if v == primary.g {
            return self.observed_fixed_chi_second_partial(primary, obs, v, u, z_obs, u_obs, a, b);
        }
        Ok(0.0)
    }

    fn evaluate_survival_denom_d(
        &self,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        // Density normalization is |F'(a)| for the same calibration equation
        // solved by `solve_row_survival_intercept`. Reusing that exact
        // derivative convention avoids sign drift between the solver path and
        // the direct-check path.
        let (_, f_a, _) = self.evaluate_denested_survival_calibration(a, 0.0, b, beta_h, beta_w)?;
        let d = f_a.abs();
        if !d.is_finite() || d <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope produced non-positive calibration derivative |F'(a)|={d:.3e}"
                ),
            }
            .into());
        }
        Ok(d)
    }

    fn row_neglog_flex_value(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<f64, String> {
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        self.row_neglog_flex_value_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w,
        )
    }

    fn row_neglog_flex_value_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }
        let (a0, _) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        if !d1.is_finite() || d1 <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D1={d1:.3e} (calibration derivative {:.3e})",
                    d1
                ),
            }
            .into());
        }
        let (eta0, _) = self.observed_denested_eta_chi(row, a0, g, beta_h, beta_w)?;
        let (eta1, chi1) = self.observed_denested_eta_chi(row, a1, g, beta_h, beta_w)?;
        if !chi1.is_finite() || chi1 <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={chi1:.3e}"
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-eta0);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-eta1);
        let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        Ok(wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * chi1.ln()
                - di * log_phi_q1
                + di * d1.ln()
                - di * qd1.ln()))
    }

    /// Build a cached partition: cells + moment states + fixed partials,
    /// computed once per (a, b, β_h, β_w) and reused across the three
    /// integration passes (F, D, D_uv).
    fn build_cached_partition_with_moment_order(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        moment_order: usize,
    ) -> Result<CachedPartitionCells, String> {
        let raw_cells = self.denested_partition_cells(a, b, beta_h, beta_w)?;
        let mut cells = Vec::with_capacity(raw_cells.len());
        for partition_cell in raw_cells {
            let cell = partition_cell.cell;
            let neg_cell = exact_kernel::DenestedCubicCell {
                left: cell.left,
                right: cell.right,
                c0: -cell.c0,
                c1: -cell.c1,
                c2: -cell.c2,
                c3: -cell.c3,
            };
            let z_mid = exact_kernel::interval_probe_point(cell.left, cell.right)?;
            let u_mid = a + b * z_mid;
            let state = exact_kernel::evaluate_cell_moments(cell, moment_order)?;
            let fixed = self.denested_cell_primary_fixed_partials(
                primary,
                a,
                b,
                partition_cell.score_span,
                partition_cell.link_span,
                z_mid,
                u_mid,
            )?;
            cells.push(CachedCellEntry {
                partition_cell,
                neg_cell,
                state,
                fixed,
            });
        }
        Ok(CachedPartitionCells { cells })
    }

    fn build_cached_partition(
        &self,
        primary: &FlexPrimarySlices,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<CachedPartitionCells, String> {
        self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 24)
    }

    fn compute_survival_timepoint_first_order_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        d_calibration: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<SurvivalFlexTimepointFirstOrderExact, String> {
        let p = primary.total;
        let cached =
            self.build_cached_partition_with_moment_order(primary, a, b, beta_h, beta_w, 9)?;

        struct FirstOrderCellAccum {
            f_u: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<FirstOrderCellAccum, String> {
                let state = &entry.state;
                let fixed = &entry.fixed;
                let mut f_u = vec![0.0; p];
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                        &neg_coeff_u,
                        &state.moments,
                    )?;
                }
                Ok(FirstOrderCellAccum { f_u })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_u = Array1::<f64>::zeros(p);
        for acc in cell_accums {
            for u in 0..p {
                f_u[u] += acc.f_u[u];
            }
        }
        f_u[q_index] += crate::probability::normal_pdf(q);

        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        if !d_check.is_finite() || d_check <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D={d_check:.3e}"
                ),
            }
            .into());
        }
        let d_rel_err = (d_check - d_calibration).abs() / d_check.max(d_calibration.abs()).max(1.0);
        if !d_calibration.is_finite() || d_calibration <= 0.0 || d_rel_err > 1e-8 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced inconsistent calibration derivative: solve={d_calibration:.12e}, direct={d_check:.12e}"
                ),
            }
            .into());
        }

        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = f_u[u] / d_check;
        }

        let d_u_cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<Vec<f64>, String> {
                let cell = entry.partition_cell.cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let eta_aa_poly = fixed.dc_daa.to_vec();
                let mut d_u = vec![0.0; p];
                for u in 0..p {
                    let eta_u_poly = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                    let chi_u_poly =
                        poly_add(&poly_scale(&eta_aa_poly, a_u[u]), &fixed.coeff_au[u]);
                    let integrand = poly_sub(
                        &chi_u_poly,
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly),
                    );
                    d_u[u] = exact_kernel::cell_polynomial_integral_from_moments(
                        &integrand,
                        &state.moments,
                        "survival D_t first derivative",
                    )?;
                }
                Ok(d_u)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u = Array1::<f64>::zeros(p);
        for cell_d_u in d_u_cell_accums {
            for u in 0..p {
                d_u[u] += cell_d_u[u];
            }
        }

        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta = eval_coeff4_at(&obs.coeff, z_obs);
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);

        let mut rho = Array1::<f64>::zeros(p);
        let mut tau = Array1::<f64>::zeros(p);
        let scale = self.probit_frailty_scale();
        rho[primary.g] = eval_coeff4_at(&obs.dc_db, z_obs);
        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                        scale,
                    ),
                    z_obs,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                        scale,
                    ),
                    z_obs,
                );
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
            }
        }

        let mut eta_u = Array1::<f64>::zeros(p);
        let mut chi_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            eta_u[u] = chi * a_u[u] + rho[u];
            chi_u[u] = eta_aa * a_u[u] + tau[u];
        }

        Ok(SurvivalFlexTimepointFirstOrderExact {
            eta,
            chi,
            d: d_check,
            eta_u,
            chi_u,
            d_u,
        })
    }

    fn compute_survival_timepoint_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        d_calibration: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        need_d_uv: bool,
    ) -> Result<SurvivalFlexTimepointExact, String> {
        let p = primary.total;
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct ExactTimepointCellAccum {
            f_aa: f64,
            f_u: Vec<f64>,
            f_au: Vec<f64>,
            f_uv: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<ExactTimepointCellAccum, String> {
                let neg_cell = entry.neg_cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let neg_dc_da = fixed.dc_da.map(|value| -value);
                let neg_dc_daa = fixed.dc_daa.map(|value| -value);
                let mut f_u = vec![0.0; p];
                let mut f_au = vec![0.0; p];
                let mut f_uv = vec![0.0; p * p];
                let f_aa = exact_kernel::cell_second_derivative_from_moments(
                    neg_cell,
                    &neg_dc_da,
                    &neg_dc_da,
                    &neg_dc_daa,
                    &state.moments,
                )?;
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    let neg_coeff_au = fixed.coeff_au[u].map(|value| -value);
                    f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                        &neg_coeff_u,
                        &state.moments,
                    )?;
                    f_au[u] = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_coeff_u,
                        &neg_coeff_au,
                        &state.moments,
                    )?;
                }
                for u in 0..p {
                    let neg_coeff_u = fixed.coeff_u[u].map(|value| -value);
                    for v in u..p {
                        let second_coeff = if u == primary.g {
                            fixed.coeff_bu[v]
                        } else if v == primary.g {
                            fixed.coeff_bu[u]
                        } else {
                            [0.0; 4]
                        };
                        let neg_coeff_v = fixed.coeff_u[v].map(|value| -value);
                        let neg_second_coeff = second_coeff.map(|value| -value);
                        let value = exact_kernel::cell_second_derivative_from_moments(
                            neg_cell,
                            &neg_coeff_u,
                            &neg_coeff_v,
                            &neg_second_coeff,
                            &state.moments,
                        )?;
                        f_uv[u * p + v] = value;
                        f_uv[v * p + u] = value;
                    }
                }
                Ok(ExactTimepointCellAccum {
                    f_aa,
                    f_u,
                    f_au,
                    f_uv,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_aa = 0.0;
        for acc in cell_accums {
            f_aa += acc.f_aa;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[u * p + v];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;

        let d_check = self.evaluate_survival_denom_d(a, b, beta_h, beta_w)?;
        if !d_check.is_finite() || d_check <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive density normalization D={d_check:.3e}"
                ),
            }
            .into());
        }
        let d_rel_err = (d_check - d_calibration).abs() / d_check.max(d_calibration.abs()).max(1.0);
        if !d_calibration.is_finite() || d_calibration <= 0.0 || d_rel_err > 1e-8 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced inconsistent calibration derivative: solve={d_calibration:.12e}, direct={d_check:.12e}"
                ),
            }
            .into());
        }

        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = f_u[u] / d_check;
        }

        let d_u_cell_accums = cached
            .cells
            .iter()
            .map(|entry| -> Result<Vec<f64>, String> {
                let cell = entry.partition_cell.cell;
                let state = &entry.state;
                let fixed = &entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let mut d_u = vec![0.0; p];
                for u in 0..p {
                    let eta_u_poly = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                    let chi_u_poly =
                        poly_add(&poly_scale(&fixed.dc_daa, a_u[u]), &fixed.coeff_au[u]);
                    let integrand = poly_sub(
                        &chi_u_poly,
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly),
                    );
                    d_u[u] = exact_kernel::cell_polynomial_integral_from_moments(
                        &integrand,
                        &state.moments,
                        "survival D_t first derivative",
                    )?;
                }
                Ok(d_u)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u = Array1::<f64>::zeros(p);
        for cell_d_u in d_u_cell_accums {
            for u in 0..p {
                d_u[u] += cell_d_u[u];
            }
        }

        let mut a_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let value =
                    (f_uv[[u, v]] - d_u[u] * a_u[v] - d_u[v] * a_u[u] - f_aa * a_u[u] * a_u[v])
                        / d_check;
                a_uv[[u, v]] = value;
                a_uv[[v, u]] = value;
            }
        }

        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let eta = eval_coeff4_at(&obs.coeff, z_obs);
        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);

        let mut rho = Array1::<f64>::zeros(p);
        let mut tau = Array1::<f64>::zeros(p);
        let mut tau_a = Array1::<f64>::zeros(p);
        let scale = self.probit_frailty_scale();
        rho[primary.g] = eval_coeff4_at(&obs.dc_db, z_obs);
        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        tau_a[primary.g] = eval_coeff4_at(&obs.dc_daab, z_obs);

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                        scale,
                    ),
                    z_obs,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                rho[idx] = eval_coeff4_at(
                    &scale_coeff4(
                        exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                        scale,
                    ),
                    z_obs,
                );
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, _, _) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
                tau_a[idx] = eval_coeff4_at(&scale_coeff4(dc_aaw, scale), z_obs);
            }
        }

        let mut eta_u = Array1::<f64>::zeros(p);
        let mut chi_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            eta_u[u] = chi * a_u[u] + rho[u];
            chi_u[u] = eta_aa * a_u[u] + tau[u];
        }

        let mut eta_uv = Array2::<f64>::zeros((p, p));
        let mut chi_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let r_uv = self.observed_fixed_eta_second_partial(
                    primary, &obs, row, u, v, z_obs, u_obs, a, b,
                )?;
                let chi_uv_fixed = self
                    .observed_fixed_chi_second_partial(primary, &obs, u, v, z_obs, u_obs, a, b)?;

                let eta_val = chi * a_uv[[u, v]]
                    + eta_aa * a_u[u] * a_u[v]
                    + tau[u] * a_u[v]
                    + tau[v] * a_u[u]
                    + r_uv;
                eta_uv[[u, v]] = eta_val;
                eta_uv[[v, u]] = eta_val;

                let chi_val = eta_aa * a_uv[[u, v]]
                    + eta_aaa * a_u[u] * a_u[v]
                    + tau_a[u] * a_u[v]
                    + tau_a[v] * a_u[u]
                    + chi_uv_fixed;
                chi_uv[[u, v]] = chi_val;
                chi_uv[[v, u]] = chi_val;
            }
        }

        let mut d_uv = Array2::<f64>::zeros((p, p));
        if need_d_uv {
            let d_uv_cell_accums = cached
                .cells
                .iter()
                .map(|entry| -> Result<Vec<f64>, String> {
                    let cell = entry.partition_cell.cell;
                    let state = &entry.state;
                    let fixed = &entry.fixed;
                    let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                    let chi_poly = fixed.dc_da.to_vec();
                    let eta_aa_poly = fixed.dc_daa.to_vec();
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();
                    let mut eta_u_poly = vec![PolyVec::new(); p];
                    let mut chi_u_poly = vec![PolyVec::new(); p];
                    let mut d_uv = vec![0.0; p * p];
                    for u in 0..p {
                        eta_u_poly[u] = poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u]);
                        chi_u_poly[u] =
                            poly_add(&poly_scale(&eta_aa_poly, a_u[u]), &fixed.coeff_au[u]);
                    }
                    for u in 0..p {
                        for v in u..p {
                            let r_uv_fixed = if u == primary.g {
                                fixed.coeff_bu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_bu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };
                            let chi_uv_fixed = if u == primary.g {
                                fixed.coeff_abu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_abu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };
                            let eta_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&chi_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(&fixed.coeff_au[u], a_u[v]),
                                ),
                                &poly_add(&poly_scale(&fixed.coeff_au[v], a_u[u]), &r_uv_fixed),
                            );
                            let chi_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&eta_aa_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aaa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(&fixed.coeff_aau[u], a_u[v]),
                                ),
                                &poly_add(&poly_scale(&fixed.coeff_aau[v], a_u[u]), &chi_uv_fixed),
                            );
                            let term2 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[v], &eta_poly), &eta_u_poly[u]),
                                -1.0,
                            );
                            let term3 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[u], &eta_poly), &eta_u_poly[v]),
                                -1.0,
                            );
                            let term4 = poly_scale(
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        &poly_mul(&eta_poly, &eta_uv_poly),
                                    ),
                                ),
                                -1.0,
                            );
                            let term5 = poly_mul(
                                &chi_poly,
                                &poly_mul(
                                    &poly_mul(&eta_poly, &eta_poly),
                                    &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                ),
                            );
                            let integrand = poly_add(
                                &poly_add(&poly_add(&chi_uv_poly, &term2), &term3),
                                &poly_add(&term4, &term5),
                            );
                            let value = exact_kernel::cell_polynomial_integral_from_moments(
                                &integrand,
                                &state.moments,
                                "survival D_t second derivative",
                            )?;
                            d_uv[u * p + v] = value;
                            d_uv[v * p + u] = value;
                        }
                    }
                    Ok(d_uv)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for cell_d_uv in d_uv_cell_accums {
                for u in 0..p {
                    for v in 0..p {
                        d_uv[[u, v]] += cell_d_uv[u * p + v];
                    }
                }
            }
        }

        Ok(SurvivalFlexTimepointExact {
            eta,
            chi,
            d: d_check,
            eta_u,
            eta_uv,
            chi_u,
            chi_uv,
            d_u,
            d_uv,
        })
    }

    fn compute_row_flex_primary_gradient_hessian_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        self.ensure_scalar_flex_exact_score_geometry(
            "compute_row_flex_primary_gradient_hessian_exact",
        )?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        self.compute_row_flex_primary_gradient_hessian_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, primary,
        )
    }

    fn compute_row_flex_primary_gradient_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        q_geom: &SurvivalMarginalSlopeDynamicRowGradient,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>), String> {
        self.ensure_scalar_flex_exact_score_geometry("compute_row_flex_primary_gradient_exact")?;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;
        self.compute_row_flex_primary_gradient_from_parts(
            row, q_geom.q0, q_geom.q1, q_geom.qd1, g, beta_h, beta_w, primary,
        )
    }

    fn compute_row_flex_primary_gradient_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>), String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        let entry = self.compute_survival_timepoint_first_order_exact(
            row, primary, q0, primary.q0, a0, g, d0, beta_h, beta_w,
        )?;
        let exit = self.compute_survival_timepoint_first_order_exact(
            row, primary, q1, primary.q1, a1, g, d1, beta_h, beta_w,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={:.3e}",
                    exit.chi
                ),
            }
            .into());
        }

        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-entry.eta);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-exit.eta);
        let (entry_k1, _, _, _) = signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
        let (exit_k1, _, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;
        let log_phi_eta1 = -0.5 * (exit.eta * exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        let row_nll = wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * exit.chi.ln()
                - di * log_phi_q1
                + di * exit.d.ln()
                - di * qd1.ln());

        let p = primary.total;
        let mut grad = Array1::<f64>::zeros(p);
        let entry_u1 = -entry_k1;
        let exit_surv_u1 = -exit_k1;

        for u in 0..p {
            grad[u] += entry_u1 * entry.eta_u[u];
            grad[u] += exit_surv_u1 * exit.eta_u[u];
            grad[u] += wi * di * exit.eta * exit.eta_u[u];
            grad[u] -= wi * di * exit.chi_u[u] / exit.chi;
            if u == primary.q1 {
                grad[u] += wi * di * q1;
            }
            grad[u] += wi * di * exit.d_u[u] / exit.d;
            if u == primary.qd1 {
                grad[u] -= wi * di / qd1;
            }
        }

        Ok((row_nll, grad))
    }

    fn compute_row_flex_primary_gradient_hessian_from_parts(
        &self,
        row: usize,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        primary: &FlexPrimarySlices,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: qd1={qd1:.3e} < guard={:.3e}",
                    self.derivative_guard
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;
        let entry = self.compute_survival_timepoint_exact(
            row, primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, false,
        )?;
        let exit = self.compute_survival_timepoint_exact(
            row, primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, true,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival marginal-slope row {row} produced non-positive observed chi1={:.3e}",
                    exit.chi
                ),
            }
            .into());
        }

        let wi = self.weights[row];
        let di = self.event[row];
        let (log_surv0, _) = signed_probit_logcdf_and_mills_ratio(-entry.eta);
        let (log_surv1, _) = signed_probit_logcdf_and_mills_ratio(-exit.eta);
        let (entry_k1, entry_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
        let (exit_k1, exit_k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;
        let log_phi_eta1 = -0.5 * (exit.eta * exit.eta + std::f64::consts::TAU.ln());
        let log_phi_q1 = -0.5 * (q1 * q1 + std::f64::consts::TAU.ln());
        let row_nll = wi
            * (log_surv0
                - (1.0 - di) * log_surv1
                - di * log_phi_eta1
                - di * exit.chi.ln()
                - di * log_phi_q1
                + di * exit.d.ln()
                - di * qd1.ln());

        let p = primary.total;
        let mut grad = Array1::<f64>::zeros(p);
        let mut hess = Array2::<f64>::zeros((p, p));
        let entry_u1 = -entry_k1;
        let entry_u2 = entry_k2;
        let exit_surv_u1 = -exit_k1;
        let exit_surv_u2 = exit_k2;

        for u in 0..p {
            grad[u] += entry_u1 * entry.eta_u[u];
            grad[u] += exit_surv_u1 * exit.eta_u[u];
            grad[u] += wi * di * exit.eta * exit.eta_u[u];
            grad[u] -= wi * di * exit.chi_u[u] / exit.chi;
            if u == primary.q1 {
                grad[u] += wi * di * q1;
            }
            grad[u] += wi * di * exit.d_u[u] / exit.d;
            if u == primary.qd1 {
                grad[u] -= wi * di / qd1;
            }
        }

        for u in 0..p {
            for v in u..p {
                let mut value = 0.0;
                value +=
                    entry_u2 * entry.eta_u[u] * entry.eta_u[v] + entry_u1 * entry.eta_uv[[u, v]];
                value += exit_surv_u2 * exit.eta_u[u] * exit.eta_u[v]
                    + exit_surv_u1 * exit.eta_uv[[u, v]];
                value += wi * di * (exit.eta_u[u] * exit.eta_u[v] + exit.eta * exit.eta_uv[[u, v]]);
                value -= wi
                    * di
                    * (exit.chi_uv[[u, v]] / exit.chi
                        - (exit.chi_u[u] * exit.chi_u[v]) / (exit.chi * exit.chi));
                if u == primary.q1 && v == primary.q1 {
                    value += wi * di;
                }
                value += wi
                    * di
                    * (exit.d_uv[[u, v]] / exit.d
                        - (exit.d_u[u] * exit.d_u[v]) / (exit.d * exit.d));
                if u == primary.qd1 && v == primary.qd1 {
                    value += wi * di / (qd1 * qd1);
                }
                hess[[u, v]] = value;
                hess[[v, u]] = value;
            }
        }

        Ok((row_nll, grad, hess))
    }

    /// Per-row NLL and its directional derivatives through 4 primary scalars.
    ///
    /// NLL_i = w_i * [ (1-d)·neglogΦ(-η₁) + logΦ(-η₀) − d·logφ(η₁) − d·log(a'₁) ]
    ///
    /// where η = a(t) + β·z, a(t) = q(t)·√(1+β²), β = g.
    ///
    /// block_states[0].eta is from the exit design and is NOT used here;
    /// all 3 time-block linear predictors are recomputed from beta_time
    /// because the time block has 3 design matrices sharing one coefficient vector.
    ///
    /// Note: hot callers (third/fourth contracted tensor builders) now route
    /// through `row_neglog_directional_jet_batched` to share q-geometry and
    /// jet construction across the 10 upper-triangular (a, b) cells. This
    /// per-cell variant is retained as a regression baseline for the
    /// batched-vs-legacy equivalence test below.
    #[cfg(test)]
    fn row_neglog_directional_refs(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[ArrayView1<'_, f64>],
    ) -> Result<f64, String> {
        crate::families::jet_partitions::ROW_NEGLOG_CALLS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let k = dirs.len();
        if k > 4 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope row directional expects 0..=4 directions, got {k}"
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (z_sum, covariance_ones) =
            self.exact_shared_score_summary(row, block_states, "row_neglog_directional_refs")?;

        // Primary scalar jets: q0, q1, qd1, g.
        // Stack-allocated fixed-capacity buffers (k ≤ 4 by precondition above)
        // avoid heap allocation in the per-row hot loop.
        let mut q0_first_buf = [0.0f64; 4];
        let mut q1_first_buf = [0.0f64; 4];
        let mut qd1_first_buf = [0.0f64; 4];
        let mut g_first_buf = [0.0f64; 4];
        for (i, dir) in dirs.iter().enumerate() {
            q0_first_buf[i] = dir[0];
            q1_first_buf[i] = dir[1];
            qd1_first_buf[i] = dir[2];
            g_first_buf[i] = dir[3];
        }
        let q0_first: &[f64] = &q0_first_buf[..k];
        let q1_first: &[f64] = &q1_first_buf[..k];
        let qd1_first: &[f64] = &qd1_first_buf[..k];
        let g_first: &[f64] = &g_first_buf[..k];

        // Reuse the realized q-geometry so exact directional derivatives stay on
        // the same timewiggle-aware/frailty-aware manifold as the closed-form
        // primary gradient/Hessian and the exact outer-derivative paths.
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let q0_val = q_geom.q0;
        let q1_val = q_geom.q1;
        let qd1_val = q_geom.qd1;
        let g_val = block_states[2].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, g_first);

        // The observed slope seen by the probit likelihood is the frailty-scaled
        // raw logslope coefficient.
        let observed_g_jet = g_jet.scale(self.probit_frailty_scale());
        // c = sqrt(1 + observed_g^2 * 1' Sigma 1) for a shared K-vector slope.
        let one_plus_b2 = MultiDirJet::constant(k, 1.0)
            .add(&observed_g_jet.mul(&observed_g_jet).scale(covariance_ones));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        // a0 = q0 * c, a1 = q1 * c, ad1 = qd1 * c
        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);

        // eta0 = a0 + observed_g * 1'z, eta1 = a1 + observed_g * 1'z.
        let z_jet = MultiDirJet::constant(k, z_sum);
        let eta0_jet = a0_jet.add(&observed_g_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&observed_g_jet.mul(&z_jet));

        // NLL_i = w_i * {
        //   (1-d_i) * neglogphi(-eta1)  [exit survival for censored]
        //   + log Phi(-eta0)            [entry survival from left truncation]
        //   - d_i * log_phi(eta1)        [event log-density of normal]
        //   - d_i * log(ad1)             [event log time-derivative]
        // }

        // Entry survival term: +log Phi(-eta0) = log S(t0|z)
        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0); // note: -w * neglogphi(-eta0) = w * log Phi(-eta0)

        // Exit survival term: (1-d)*neglogphi(-eta1) = -(1-d)*log Phi(-eta1)
        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        // Event density: -d * log phi(eta1)
        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        // Time derivative: -d * log(ad1)
        // The model domain is enforced on qd1 itself, which is the same quantity
        // constrained by the time monotonicity inequalities. Since c = sqrt(1+beta^2)
        // is strictly positive, qd1 >= guard implies ad1 > 0 as required by log(ad1).
        let qd1_lower = self.time_derivative_lower_bound();
        let qd1_val = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if survival_derivative_guard_violated(qd1_val, qd1_lower) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
                ),
            }
            .into());
        }
        let time_deriv_term = if di > 0.0 {
            ad1_jet
                .compose_unary(unary_derivatives_log(ad1_val))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let total = exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term);

        if k == 0 {
            Ok(total.coeff(0))
        } else {
            Ok(total.coeff(total.full_mask()))
        }
    }

    /// Batched variant of `row_neglog_directional_refs` returning the full
    /// `MultiDirJet` for the per-row NLL composed against an arbitrary
    /// ordered list of `dirs` (one direction per jet bit).
    ///
    /// Callers reading higher-order contracted tensors (third/fourth)
    /// previously looped the 10 upper-triangular (a, b) pairs and called
    /// `row_neglog_directional_refs` ten times, each rebuilding the same
    /// q-geometry, the same primary jets, and rerunning the same composed
    /// unary derivatives. By packing the basis directions e_0..e_3 alongside
    /// the user-supplied `dir` (and optional `dir_v`) into one larger jet, the
    /// per-row q-geometry + jet-construction work is shared across all 10
    /// upper-triangular cells. Off-diagonal entries are then a single mask
    /// read; diagonal entries fall back to a small per-axis jet that still
    /// reuses no work between cells.
    ///
    /// `dirs.len()` must be in `1..=8` (MAX_DIRS in `jet_partitions`).
    fn row_neglog_directional_jet_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dirs: &[ArrayView1<'_, f64>],
    ) -> Result<MultiDirJet, String> {
        crate::families::jet_partitions::ROW_NEGLOG_CALLS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let k = dirs.len();
        if k == 0 || k > 8 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival marginal-slope row directional jet expects 1..=8 directions, got {k}"
                ),
            }
            .into());
        }
        let wi = self.weights[row];
        let di = self.event[row];
        let (z_sum, covariance_ones) = self.exact_shared_score_summary(
            row,
            block_states,
            "row_neglog_directional_jet_batched",
        )?;

        // Primary scalar jets: q0, q1, qd1, g. Fixed-capacity stack buffers
        // (k ≤ 8) avoid heap allocation in the per-row hot loop.
        let mut q0_first_buf = [0.0f64; 8];
        let mut q1_first_buf = [0.0f64; 8];
        let mut qd1_first_buf = [0.0f64; 8];
        let mut g_first_buf = [0.0f64; 8];
        for (i, dir) in dirs.iter().enumerate() {
            q0_first_buf[i] = dir[0];
            q1_first_buf[i] = dir[1];
            qd1_first_buf[i] = dir[2];
            g_first_buf[i] = dir[3];
        }
        let q0_first: &[f64] = &q0_first_buf[..k];
        let q1_first: &[f64] = &q1_first_buf[..k];
        let qd1_first: &[f64] = &qd1_first_buf[..k];
        let g_first: &[f64] = &g_first_buf[..k];

        // Reuse the realized q-geometry (timewiggle / frailty manifold).
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let q0_val = q_geom.q0;
        let q1_val = q_geom.q1;
        let qd1_val = q_geom.qd1;
        let g_val = block_states[2].eta[row];

        let q0_jet = MultiDirJet::linear(k, q0_val, q0_first);
        let q1_jet = MultiDirJet::linear(k, q1_val, q1_first);
        let qd1_jet = MultiDirJet::linear(k, qd1_val, qd1_first);
        let g_jet = MultiDirJet::linear(k, g_val, g_first);

        // observed_g = frailty_scale * g
        let observed_g_jet = g_jet.scale(self.probit_frailty_scale());
        // c = sqrt(1 + observed_g^2 * 1' Sigma 1) for a shared K-vector slope.
        let one_plus_b2 = MultiDirJet::constant(k, 1.0)
            .add(&observed_g_jet.mul(&observed_g_jet).scale(covariance_ones));
        let c_jet = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.coeff(0)));

        let a0_jet = q0_jet.mul(&c_jet);
        let a1_jet = q1_jet.mul(&c_jet);
        let ad1_jet = qd1_jet.mul(&c_jet);

        let z_jet = MultiDirJet::constant(k, z_sum);
        let eta0_jet = a0_jet.add(&observed_g_jet.mul(&z_jet));
        let eta1_jet = a1_jet.add(&observed_g_jet.mul(&z_jet));

        let neg_eta0 = eta0_jet.scale(-1.0);
        let entry_term = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.coeff(0), wi))
            .scale(-1.0);

        let neg_eta1 = eta1_jet.scale(-1.0);
        let exit_term = neg_eta1.compose_unary(unary_derivatives_neglog_phi(
            neg_eta1.coeff(0),
            wi * (1.0 - di),
        ));

        let event_density_term = if di > 0.0 {
            eta1_jet
                .compose_unary(unary_derivatives_log_normal_pdf(eta1_jet.coeff(0)))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let qd1_lower = self.time_derivative_lower_bound();
        let qd1_val_chk = qd1_jet.coeff(0);
        let ad1_val = ad1_jet.coeff(0);
        if survival_derivative_guard_violated(qd1_val_chk, qd1_lower) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival marginal-slope monotonicity violated at row {row}: raw time derivative={qd1_val_chk:.3e} must be at least derivative_guard={qd1_lower:.3e}; transformed time derivative={ad1_val:.3e}"
                ),
            }
            .into());
        }
        let time_deriv_term = if di > 0.0 {
            ad1_jet
                .compose_unary(unary_derivatives_log(ad1_val))
                .scale(-wi * di)
        } else {
            MultiDirJet::zero(k)
        };

        let total = exit_term
            .add(&entry_term)
            .add(&event_density_term)
            .add(&time_deriv_term);

        Ok(total)
    }

    /// Build the row's third-order contracted tensor `T[a][b] = ∂_{e_a} ∂_{e_b} ∂_{dir}` of NLL_i
    /// against the four primary axes (a, b ∈ {0..N_PRIMARY}). Six off-diagonal
    /// entries are read from a single batched k=5 jet over [e_0..e_3, dir];
    /// four diagonal entries (a == b) need order-3 in a single basis axis and
    /// fall back to a k=3 jet [e_a, e_a, dir]. Result is symmetric.
    fn row_primary_third_contracted_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        let e0 = unit_primary_direction_ref(0).view();
        let e1 = unit_primary_direction_ref(1).view();
        let e2 = unit_primary_direction_ref(2).view();
        let e3 = unit_primary_direction_ref(3).view();
        // One k=5 jet covers all 6 off-diagonal (a < b) entries via masks
        // `(1<<a) | (1<<b) | (1<<4)`.
        let batched =
            self.row_neglog_directional_jet_batched(row, block_states, &[e0, e1, e2, e3, dir])?;
        let dir_bit = 1usize << 4;
        let mut r = [[0.0_f64; 4]; 4];
        for a in 0..N_PRIMARY {
            for b in (a + 1)..N_PRIMARY {
                let mask = (1usize << a) | (1usize << b) | dir_bit;
                let value = batched.coeff(mask);
                r[a][b] = value;
                r[b][a] = value;
            }
        }
        // Diagonal entries: ∂²_{e_a} ∂_{dir} require two copies of e_a in
        // distinct bit positions. A k=3 jet [e_a, e_a, dir] read at full
        // mask (7) supplies that.
        for a in 0..N_PRIMARY {
            let ea = unit_primary_direction_ref(a).view();
            let diag_jet =
                self.row_neglog_directional_jet_batched(row, block_states, &[ea, ea, dir])?;
            r[a][a] = diag_jet.coeff(diag_jet.full_mask());
        }
        Ok(r)
    }

    /// Build the row's fourth-order contracted tensor
    /// `T[a][b] = ∂_{e_a} ∂_{e_b} ∂_{dir_u} ∂_{dir_v}` of NLL_i, mirrored.
    fn row_primary_fourth_contracted_batched(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<[[f64; 4]; 4], String> {
        let e0 = unit_primary_direction_ref(0).view();
        let e1 = unit_primary_direction_ref(1).view();
        let e2 = unit_primary_direction_ref(2).view();
        let e3 = unit_primary_direction_ref(3).view();
        // One k=6 jet covers all 6 off-diagonal (a < b) entries via masks
        // `(1<<a) | (1<<b) | (1<<4) | (1<<5)`.
        let batched = self.row_neglog_directional_jet_batched(
            row,
            block_states,
            &[e0, e1, e2, e3, dir_u, dir_v],
        )?;
        let u_bit = 1usize << 4;
        let v_bit = 1usize << 5;
        let mut r = [[0.0_f64; 4]; 4];
        for a in 0..N_PRIMARY {
            for b in (a + 1)..N_PRIMARY {
                let mask = (1usize << a) | (1usize << b) | u_bit | v_bit;
                let value = batched.coeff(mask);
                r[a][b] = value;
                r[b][a] = value;
            }
        }
        for a in 0..N_PRIMARY {
            let ea = unit_primary_direction_ref(a).view();
            let diag_jet = self.row_neglog_directional_jet_batched(
                row,
                block_states,
                &[ea, ea, dir_u, dir_v],
            )?;
            r[a][a] = diag_jet.coeff(diag_jet.full_mask());
        }
        Ok(r)
    }

    /// Compute per-row primary gradient and Hessian using the closed-form
    /// scalar kernel.  The hot inner computation uses stack arrays only;
    /// conversion to Array1/Array2 happens once at the boundary for API
    /// compatibility with outer-derivative paths.
    fn compute_row_primary_gradient_hessian_uncached(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let q_geom = self.row_dynamic_q_values(row, block_states)?;
        let (nll, grad_arr, hess_arr) = self.row_primary_closed_form_rigid(
            row,
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            block_states,
            self.probit_frailty_scale(),
        )?;
        // Convert stack arrays to ndarray types at the boundary.
        let grad = Array1::from_vec(grad_arr.to_vec());
        let mut hess = Array2::zeros((N_PRIMARY, N_PRIMARY));
        for i in 0..N_PRIMARY {
            for j in 0..N_PRIMARY {
                hess[[i, j]] = hess_arr[i][j];
            }
        }
        Ok((nll, grad, hess))
    }

    fn build_eval_cache(&self, block_states: &[ParameterBlockState]) -> Result<EvalCache, String> {
        let row_bases = (0..self.n)
            .into_par_iter()
            .map(|row| {
                let (_, gradient, hessian) =
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                Ok(RowPrimaryBase { gradient, hessian })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(EvalCache { row_bases })
    }

    fn row_primary_gradient_hessian<'a>(
        &self,
        row: usize,
        cache: &'a EvalCache,
    ) -> (&'a Array1<f64>, &'a Array2<f64>) {
        let base = &cache.row_bases[row];
        (&base.gradient, &base.hessian)
    }

    fn offset_channel_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(OffsetChannelResiduals, OffsetChannelCurvatures), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let primary = flex_active.then(|| flex_primary_slices(self));
        let rows = (0..self.n)
            .into_par_iter()
            .map(
                |row| -> Result<(usize, f64, f64, f64, [[f64; 3]; 3]), String> {
                    if self.weights[row] <= 0.0 {
                        return Ok((row, 0.0, 0.0, 0.0, [[0.0; 3]; 3]));
                    }
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (gradient, hessian) = if let Some(primary) = primary.as_ref() {
                        let (_, gradient, hessian) = self
                            .compute_row_flex_primary_gradient_hessian_exact(
                                row,
                                block_states,
                                &q_geom,
                                primary,
                            )?;
                        (gradient, hessian)
                    } else {
                        let (_, gradient, hessian) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (gradient, hessian)
                    };
                    let channel = [0usize, 1usize, 2usize];
                    let mut curvature = [[0.0; 3]; 3];
                    for a in 0..3 {
                        for b in 0..3 {
                            curvature[a][b] = hessian[[channel[a], channel[b]]];
                        }
                    }
                    Ok((row, gradient[1], gradient[0], gradient[2], curvature))
                },
            )
            .collect::<Result<Vec<_>, String>>()?;
        let mut exit = Array1::<f64>::zeros(self.n);
        let mut entry = Array1::<f64>::zeros(self.n);
        let mut derivative = Array1::<f64>::zeros(self.n);
        let mut curvatures = vec![[[0.0; 3]; 3]; self.n];
        for (row, r_exit, r_entry, r_derivative, curvature) in rows {
            exit[row] = r_exit;
            entry[row] = r_entry;
            derivative[row] = r_derivative;
            curvatures[row] = curvature;
        }
        Ok((
            OffsetChannelResiduals {
                exit,
                entry,
                derivative,
            },
            OffsetChannelCurvatures { rows: curvatures },
        ))
    }

    fn accumulate_dynamic_q_core_gradient(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        joint_gradient: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.time.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.marginal.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut joint_gradient.slice_mut(s![slices.logslope.clone()]),
        )?;
        Ok(())
    }

    fn accumulate_dynamic_q_core_gradient_first_order(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRowGradient,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        joint_gradient: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.time.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                joint_gradient[slices.marginal.start + coeff_idx] -=
                    primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut joint_gradient.slice_mut(s![slices.logslope.clone()]),
        )?;
        Ok(())
    }

    fn dynamic_q_core_hessian_blocks(
        &self,
        row: usize,
        p_t: usize,
        p_m: usize,
        p_g: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
    ) -> Result<DynamicQCoreHessianBlocks, String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        let d2q_time_time = [
            &q_geom.d2q0_time_time,
            &q_geom.d2q1_time_time,
            &q_geom.d2qd1_time_time,
        ];
        let d2q_marginal_marginal = [
            &q_geom.d2q0_marginal_marginal,
            &q_geom.d2q1_marginal_marginal,
            &q_geom.d2qd1_marginal_marginal,
        ];
        let d2q_time_marginal = [
            &q_geom.d2q0_time_marginal,
            &q_geom.d2q1_time_marginal,
            &q_geom.d2qd1_time_marginal,
        ];
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("dynamic_q_core_hessian_blocks logslope try_row_chunk: {e}"))?;
        let logslope_row = logslope_chunk.row(0).to_owned();

        // The caller is itself an outer per-row `into_par_iter` over n rows, so
        // any inner `rayon::join` here just oversubscribes a saturated pool —
        // each block is at most p_m^2 ≈ 9k FMAs (microseconds), far below the
        // join+steal overhead. Build all blocks sequentially.
        let mut hess_time = Array2::<f64>::zeros((p_t, p_t));
        for a in 0..p_t {
            for b in 0..p_t {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value += primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_time[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_time[q_u][[a, b]];
                }
                hess_time[[a, b]] = value;
            }
        }
        let mut hess_marginal = Array2::<f64>::zeros((p_m, p_m));
        for a in 0..p_m {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_marginal[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_marginal_marginal[q_u][[a, b]];
                }
                hess_marginal[[a, b]] = value;
            }
        }
        let mut hess_logslope = Array2::<f64>::zeros((p_g, p_g));
        let h_gg_scale = primary_hessian[[3, 3]];
        if h_gg_scale != 0.0 {
            for a in 0..p_g {
                let xa = logslope_row[a];
                if xa == 0.0 {
                    continue;
                }
                let row_scale = h_gg_scale * xa;
                for b in 0..p_g {
                    hess_logslope[[a, b]] = row_scale * logslope_row[b];
                }
            }
        }
        let mut hess_time_marginal = Array2::<f64>::zeros((p_t, p_m));
        for a in 0..p_t {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_marginal[q_u][[a, b]];
                }
                hess_time_marginal[[a, b]] = value;
            }
        }
        let mut hess_time_logslope = Array2::<f64>::zeros((p_t, p_g));
        for a in 0..p_t {
            let mut weight = 0.0;
            for q_u in 0..3 {
                weight += primary_hessian[[q_u, 3]] * dq_time[q_u][a];
            }
            if weight != 0.0 {
                for b in 0..p_g {
                    hess_time_logslope[[a, b]] = weight * logslope_row[b];
                }
            }
        }
        let mut hess_marginal_logslope = Array2::<f64>::zeros((p_m, p_g));
        for a in 0..p_m {
            let mut weight = 0.0;
            for q_u in 0..3 {
                weight += primary_hessian[[q_u, 3]] * dq_marginal[q_u][a];
            }
            if weight != 0.0 {
                for b in 0..p_g {
                    hess_marginal_logslope[[a, b]] = weight * logslope_row[b];
                }
            }
        }

        Ok(DynamicQCoreHessianBlocks {
            hess_time,
            hess_marginal,
            hess_logslope,
            hess_time_marginal,
            hess_time_logslope,
            hess_marginal_logslope,
        })
    }

    fn dynamic_q_core_diagonal_hessian_blocks(
        &self,
        row: usize,
        p_t: usize,
        p_m: usize,
        p_g: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        let d2q_time_time = [
            &q_geom.d2q0_time_time,
            &q_geom.d2q1_time_time,
            &q_geom.d2qd1_time_time,
        ];
        let d2q_marginal_marginal = [
            &q_geom.d2q0_marginal_marginal,
            &q_geom.d2q1_marginal_marginal,
            &q_geom.d2qd1_marginal_marginal,
        ];
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| {
                format!("dynamic_q_core_diagonal_hessian_blocks logslope try_row_chunk: {e}")
            })?;
        let logslope_row = logslope_chunk.row(0).to_owned();

        // Outer caller is `(0..n).into_par_iter()` — inner rayon::join here
        // would oversubscribe a saturated pool for ≤9k-FMA blocks. Sequential.
        let mut hess_time = Array2::<f64>::zeros((p_t, p_t));
        for a in 0..p_t {
            for b in 0..p_t {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value += primary_hessian[[q_u, q_v]] * dq_time[q_u][a] * dq_time[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_time_time[q_u][[a, b]];
                }
                hess_time[[a, b]] = value;
            }
        }
        let mut hess_marginal = Array2::<f64>::zeros((p_m, p_m));
        for a in 0..p_m {
            for b in 0..p_m {
                let mut value = 0.0;
                for q_u in 0..3 {
                    for q_v in 0..3 {
                        value +=
                            primary_hessian[[q_u, q_v]] * dq_marginal[q_u][a] * dq_marginal[q_v][b];
                    }
                    value += primary_gradient[q_u] * d2q_marginal_marginal[q_u][[a, b]];
                }
                hess_marginal[[a, b]] = value;
            }
        }
        let mut hess_logslope = Array2::<f64>::zeros((p_g, p_g));
        let h_gg_scale = primary_hessian[[3, 3]];
        if h_gg_scale != 0.0 {
            for a in 0..p_g {
                let xa = logslope_row[a];
                if xa == 0.0 {
                    continue;
                }
                let row_scale = h_gg_scale * xa;
                for b in 0..p_g {
                    hess_logslope[[a, b]] = row_scale * logslope_row[b];
                }
            }
        }
        Ok((hess_time, hess_marginal, hess_logslope))
    }

    fn accumulate_dynamic_q_core_hessian(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let blocks = self.dynamic_q_core_hessian_blocks(
            row,
            p_t,
            p_m,
            p_g,
            q_geom,
            primary_gradient,
            primary_hessian,
        )?;

        for a in 0..p_t {
            for b in 0..p_t {
                joint_hessian[[slices.time.start + a, slices.time.start + b]] +=
                    blocks.hess_time[[a, b]];
            }
        }
        for a in 0..p_m {
            for b in 0..p_m {
                joint_hessian[[slices.marginal.start + a, slices.marginal.start + b]] +=
                    blocks.hess_marginal[[a, b]];
            }
        }
        for a in 0..p_g {
            for b in 0..p_g {
                joint_hessian[[slices.logslope.start + a, slices.logslope.start + b]] +=
                    blocks.hess_logslope[[a, b]];
            }
        }
        for a in 0..p_t {
            for b in 0..p_m {
                let value = blocks.hess_time_marginal[[a, b]];
                joint_hessian[[slices.time.start + a, slices.marginal.start + b]] += value;
                joint_hessian[[slices.marginal.start + b, slices.time.start + a]] += value;
            }
        }
        for a in 0..p_t {
            for b in 0..p_g {
                let value = blocks.hess_time_logslope[[a, b]];
                joint_hessian[[slices.time.start + a, slices.logslope.start + b]] += value;
                joint_hessian[[slices.logslope.start + b, slices.time.start + a]] += value;
            }
        }
        for a in 0..p_m {
            for b in 0..p_g {
                let value = blocks.hess_marginal_logslope[[a, b]];
                joint_hessian[[slices.marginal.start + a, slices.logslope.start + b]] += value;
                joint_hessian[[slices.logslope.start + b, slices.marginal.start + a]] += value;
            }
        }
        Ok(())
    }

    fn accumulate_dynamic_q_blockwise_gradient(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        grad_time: &mut Array1<f64>,
        grad_marginal: &mut Array1<f64>,
        grad_logslope: &mut Array1<f64>,
    ) -> Result<(), String> {
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        for (q_idx, dq) in dq_time.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                grad_time[coeff_idx] -= primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        for (q_idx, dq) in dq_marginal.iter().enumerate() {
            for coeff_idx in 0..dq.len() {
                grad_marginal[coeff_idx] -= primary_gradient[q_idx] * dq[coeff_idx];
            }
        }
        self.logslope_design.axpy_row_into(
            row,
            -primary_gradient[3],
            &mut grad_logslope.view_mut(),
        )?;
        Ok(())
    }

    fn accumulate_dynamic_q_core_block_hessians(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        hess_time: &mut Array2<f64>,
        hess_marginal: &mut Array2<f64>,
        hess_logslope: &mut Array2<f64>,
    ) -> Result<(), String> {
        let (local_time, local_marginal, local_logslope) = self
            .dynamic_q_core_diagonal_hessian_blocks(
                row,
                hess_time.nrows(),
                hess_marginal.nrows(),
                hess_logslope.nrows(),
                q_geom,
                primary_gradient,
                primary_hessian,
            )?;

        *hess_time += &local_time;
        *hess_marginal += &local_marginal;
        *hess_logslope += &local_logslope;
        Ok(())
    }

    fn accumulate_dynamic_q_blockwise_row(
        &self,
        row: usize,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary: &FlexPrimarySlices,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        acc: &mut DynamicQBlockwiseAccumulator,
    ) -> Result<(), String> {
        self.accumulate_dynamic_q_blockwise_gradient(
            row,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            &mut acc.grad_time,
            &mut acc.grad_marginal,
            &mut acc.grad_logslope,
        )?;
        self.accumulate_dynamic_q_core_block_hessians(
            row,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            primary_hessian.slice(s![0..N_PRIMARY, 0..N_PRIMARY]),
            &mut acc.hess_time,
            &mut acc.hess_marginal,
            &mut acc.hess_logslope,
        )?;
        if let (Some(primary_range), Some(gradient), Some(hessian)) = (
            primary.h.as_ref(),
            acc.grad_score_warp.as_mut(),
            acc.hess_score_warp.as_mut(),
        ) {
            *gradient -= &primary_gradient.slice(s![primary_range.clone()]);
            *hessian += &primary_hessian
                .slice(s![primary_range.clone(), primary_range.clone()])
                .to_owned();
        }
        if let (Some(primary_range), Some(gradient), Some(hessian)) = (
            primary.w.as_ref(),
            acc.grad_link_dev.as_mut(),
            acc.hess_link_dev.as_mut(),
        ) {
            *gradient -= &primary_gradient.slice(s![primary_range.clone()]);
            *hessian += &primary_hessian
                .slice(s![primary_range.clone(), primary_range.clone()])
                .to_owned();
        }
        Ok(())
    }

    fn accumulate_identity_primary_cross_hessian(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        core_hessian_column: ndarray::ArrayView1<'_, f64>,
        joint_block: &std::ops::Range<usize>,
        joint_local: usize,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        let joint_idx = joint_block.start + joint_local;
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];

        for coeff_idx in 0..slices.time.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += core_hessian_column[q_idx] * dq_time[q_idx][coeff_idx];
            }
            joint_hessian[[slices.time.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.time.start + coeff_idx]] += value;
        }
        for coeff_idx in 0..slices.marginal.len() {
            let mut value = 0.0;
            for q_idx in 0..3 {
                value += core_hessian_column[q_idx] * dq_marginal[q_idx][coeff_idx];
            }
            joint_hessian[[slices.marginal.start + coeff_idx, joint_idx]] += value;
            joint_hessian[[joint_idx, slices.marginal.start + coeff_idx]] += value;
        }
        let logslope_chunk = self
            .logslope_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| {
                format!("accumulate_identity_primary_cross_hessian logslope try_row_chunk: {e}")
            })?;
        let logslope_row = logslope_chunk.row(0);
        let logslope_weight = core_hessian_column[3];
        if logslope_weight != 0.0 {
            for coeff_idx in 0..slices.logslope.len() {
                let value = logslope_weight * logslope_row[coeff_idx];
                joint_hessian[[slices.logslope.start + coeff_idx, joint_idx]] += value;
                joint_hessian[[joint_idx, slices.logslope.start + coeff_idx]] += value;
            }
        }
        Ok(())
    }

    fn add_dense_submatrix(
        &self,
        joint_hessian: &mut Array2<f64>,
        target_rows: &std::ops::Range<usize>,
        target_cols: &std::ops::Range<usize>,
        source: ArrayView2<'_, f64>,
    ) {
        for row_local in 0..target_rows.len() {
            for col_local in 0..target_cols.len() {
                joint_hessian[[target_rows.start + row_local, target_cols.start + col_local]] +=
                    source[[row_local, col_local]];
            }
        }
    }

    fn add_dense_symmetric_cross_submatrix(
        &self,
        joint_hessian: &mut Array2<f64>,
        left_range: &std::ops::Range<usize>,
        right_range: &std::ops::Range<usize>,
        source: ArrayView2<'_, f64>,
    ) {
        for left_local in 0..left_range.len() {
            for right_local in 0..right_range.len() {
                let value = source[[left_local, right_local]];
                joint_hessian[[
                    left_range.start + left_local,
                    right_range.start + right_local,
                ]] += value;
                joint_hessian[[
                    right_range.start + right_local,
                    left_range.start + left_local,
                ]] += value;
            }
        }
    }

    fn accumulate_dynamic_q_joint_row(
        &self,
        row: usize,
        slices: &BlockSlices,
        q_geom: &SurvivalMarginalSlopeDynamicRow,
        primary_gradient: ndarray::ArrayView1<'_, f64>,
        primary_hessian: ArrayView2<'_, f64>,
        identity_blocks: &[(std::ops::Range<usize>, std::ops::Range<usize>)],
        joint_gradient: &mut Array1<f64>,
        joint_hessian: &mut Array2<f64>,
    ) -> Result<(), String> {
        self.accumulate_dynamic_q_core_gradient(
            row,
            slices,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            joint_gradient,
        )?;
        self.accumulate_dynamic_q_core_hessian(
            row,
            slices,
            q_geom,
            primary_gradient.slice(s![0..N_PRIMARY]),
            primary_hessian.slice(s![0..N_PRIMARY, 0..N_PRIMARY]),
            joint_hessian,
        )?;

        for (primary_range, joint_range) in identity_blocks {
            for local in 0..primary_range.len() {
                joint_gradient[joint_range.start + local] -=
                    primary_gradient[primary_range.start + local];
                self.accumulate_identity_primary_cross_hessian(
                    row,
                    slices,
                    q_geom,
                    primary_hessian.slice(s![0..N_PRIMARY, primary_range.start + local]),
                    joint_range,
                    local,
                    joint_hessian,
                )?;
            }
            self.add_dense_submatrix(
                joint_hessian,
                joint_range,
                joint_range,
                primary_hessian.slice(s![primary_range.clone(), primary_range.clone()]),
            );
        }

        for left_idx in 0..identity_blocks.len() {
            for right_idx in left_idx + 1..identity_blocks.len() {
                let (left_primary, left_joint) = &identity_blocks[left_idx];
                let (right_primary, right_joint) = &identity_blocks[right_idx];
                self.add_dense_symmetric_cross_submatrix(
                    joint_hessian,
                    left_joint,
                    right_joint,
                    primary_hessian.slice(s![left_primary.clone(), right_primary.clone()]),
                );
            }
        }

        Ok(())
    }

    fn evaluate_blockwise_exact_newton_dynamic_q<RowTerms>(
        &self,
        block_states: &[ParameterBlockState],
        primary: &FlexPrimarySlices,
        row_terms: RowTerms,
    ) -> Result<FamilyEvaluation, String>
    where
        RowTerms: Fn(
                usize,
                &SurvivalMarginalSlopeDynamicRow,
            ) -> Result<(f64, Array1<f64>, Array2<f64>), String>
            + Sync,
    {
        let slices = block_slices(self, block_states);
        let make_acc = || DynamicQBlockwiseAccumulator::new(&slices);
        // See `evaluate_exact_newton_joint_dynamic_q_dense` for rationale.
        let make_acc_ws = || {
            (
                make_acc(),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };

        let acc = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let (row_nll, primary_gradient, primary_hessian) = row_terms(row, q_geom)?;
                state.log_likelihood -= row_nll;
                self.accumulate_dynamic_q_blockwise_row(
                    row,
                    q_geom,
                    primary,
                    primary_gradient.view(),
                    primary_hessian.view(),
                    state,
                )?;
                Ok(acc)
            })
            .try_reduce(make_acc_ws, |mut left, right| -> Result<_, String> {
                left.0.add_assign(&right.0);
                Ok(left)
            })?
            .0;

        Ok(acc.into_family_evaluation())
    }

    fn row_primary_third_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=5 jet helper. The stack
        // tensor is then copied once into Array2 at the API boundary.
        let r = self.row_primary_third_contracted_batched(row, block_states, dir)?;
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            for b in 0..N_PRIMARY {
                out[[a, b]] = r[a][b];
            }
        }
        Ok(out)
    }

    /// Second-order cross-coefficient for parameter pair (u, v) from the
    /// b-family.  In survival, the only b-coupling is through g (the slope
    /// parameter), so this is nonzero only when u==g or v==g.
    fn cell_pair_second_coeff(
        &self,
        primary: &FlexPrimarySlices,
        coeff_bu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_bu[v]
        } else if v == primary.g {
            coeff_bu[u]
        } else {
            [0.0; 4]
        }
    }

    /// Third-order a-cross-coefficient for parameter pair (u, v) from the
    /// ab-family. Nonzero only when u==g or v==g.
    fn cell_pair_third_coeff_a(
        &self,
        primary: &FlexPrimarySlices,
        coeff_abu: &[[f64; 4]],
        u: usize,
        v: usize,
    ) -> [f64; 4] {
        if u == primary.g {
            coeff_abu[v]
        } else if v == primary.g {
            coeff_abu[u]
        } else {
            [0.0; 4]
        }
    }

    #[inline]
    fn primary_param_supported(&self, primary: &FlexPrimarySlices, idx: usize) -> bool {
        idx == primary.g
            || primary
                .h
                .as_ref()
                .map(|range| range.contains(&idx))
                .unwrap_or(false)
            || primary
                .w
                .as_ref()
                .map(|range| range.contains(&idx))
                .unwrap_or(false)
    }

    fn cell_directional_coeff_family(
        &self,
        primary: &FlexPrimarySlices,
        family: &[[f64; 4]],
        dir: &Array1<f64>,
    ) -> [f64; 4] {
        let mut out = [0.0; 4];
        add_scaled_coeff4(&mut out, &family[primary.g], dir[primary.g]);
        if let Some(h_range) = primary.h.as_ref() {
            for idx in h_range.clone() {
                add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
            }
        }
        if let Some(w_range) = primary.w.as_ref() {
            for idx in w_range.clone() {
                add_scaled_coeff4(&mut out, &family[idx], dir[idx]);
            }
        }
        out
    }

    fn cell_mixed_directional_from_b_family(
        &self,
        primary: &FlexPrimarySlices,
        family: &[[f64; 4]],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> [f64; 4] {
        let mut out = [0.0; 4];
        add_scaled_coeff4(
            &mut out,
            &family[primary.g],
            dir_u[primary.g] * dir_v[primary.g],
        );
        if let Some(h_range) = primary.h.as_ref() {
            for idx in h_range.clone() {
                add_scaled_coeff4(
                    &mut out,
                    &family[idx],
                    dir_u[primary.g] * dir_v[idx] + dir_v[primary.g] * dir_u[idx],
                );
            }
        }
        if let Some(w_range) = primary.w.as_ref() {
            for idx in w_range.clone() {
                add_scaled_coeff4(
                    &mut out,
                    &family[idx],
                    dir_u[primary.g] * dir_v[idx] + dir_v[primary.g] * dir_u[idx],
                );
            }
        }
        out
    }

    fn cell_param_directional_from_b_family(
        &self,
        primary: &FlexPrimarySlices,
        family: &[[f64; 4]],
        param: usize,
        dir: &Array1<f64>,
    ) -> [f64; 4] {
        if param == primary.g {
            return self.cell_directional_coeff_family(primary, family, dir);
        }
        if self.primary_param_supported(primary, param) {
            return scale_coeff4(family[param], dir[primary.g]);
        }
        [0.0; 4]
    }

    fn cell_param_mixed_from_bb_family(
        &self,
        primary: &FlexPrimarySlices,
        family: &[[f64; 4]],
        param: usize,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> [f64; 4] {
        if param == primary.g {
            return self.cell_mixed_directional_from_b_family(primary, family, dir_u, dir_v);
        }
        if self.primary_param_supported(primary, param) {
            return scale_coeff4(family[param], dir_u[primary.g] * dir_v[primary.g]);
        }
        [0.0; 4]
    }

    fn cell_pair_directional_from_bb_family(
        &self,
        primary: &FlexPrimarySlices,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        dir: &Array1<f64>,
    ) -> [f64; 4] {
        if u == primary.g && v == primary.g {
            return self.cell_directional_coeff_family(primary, family, dir);
        }
        if u == primary.g && self.primary_param_supported(primary, v) {
            return scale_coeff4(family[v], dir[primary.g]);
        }
        if v == primary.g && self.primary_param_supported(primary, u) {
            return scale_coeff4(family[u], dir[primary.g]);
        }
        [0.0; 4]
    }

    fn cell_pair_mixed_from_bbb_family(
        &self,
        primary: &FlexPrimarySlices,
        family: &[[f64; 4]],
        u: usize,
        v: usize,
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> [f64; 4] {
        if u == primary.g && v == primary.g {
            return self.cell_mixed_directional_from_b_family(primary, family, dir_u, dir_v);
        }
        if u == primary.g && self.primary_param_supported(primary, v) {
            return scale_coeff4(family[v], dir_u[primary.g] * dir_v[primary.g]);
        }
        if v == primary.g && self.primary_param_supported(primary, u) {
            return scale_coeff4(family[u], dir_u[primary.g] * dir_v[primary.g]);
        }
        [0.0; 4]
    }

    /// Directional derivative of the fixed eta second partial r_uv w.r.t.
    /// a contraction direction.  Only (g,g), (g,h), (g,w) entries are nonzero.
    fn observed_fixed_eta_second_partial_dir(
        &self,
        primary: &FlexPrimarySlices,
        obs: &ObservedDenestedCellPartials,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a: f64,
        b: f64,
        a_dir: f64,
        dir: &Array1<f64>,
        beta_w: Option<&Array1<f64>>,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            let dc_dbbb_val = {
                let zero_span = exact_kernel::LocalSpanCubic {
                    left: 0.0,
                    right: 1.0,
                    c0: 0.0,
                    c1: 0.0,
                    c2: 0.0,
                    c3: 0.0,
                };
                let link_span_obs = if let (Some(rt), Some(bw)) = (self.link_dev.as_ref(), beta_w) {
                    rt.local_cubic_at(bw, u_obs)?
                } else {
                    zero_span
                };
                let (_, _, _, dc_dbbb_raw) =
                    exact_kernel::denested_cell_third_partials(link_span_obs);
                eval_coeff4_at(&scale_coeff4(dc_dbbb_raw, scale), z_obs)
            };
            return Ok(eval_coeff4_at(&obs.dc_dabb, z_obs) * a_dir + dc_dbbb_val * dir[primary.g]);
        }
        if u == primary.g || v == primary.g {
            let other = if u == primary.g { v } else { u };
            if let Some(h_range) = primary.h.as_ref() {
                if other >= h_range.start && other < h_range.end {
                    return Ok(0.0);
                }
            }
            if let Some(w_range) = primary.w.as_ref() {
                if other >= w_range.start && other < w_range.end {
                    let local_idx = other - w_range.start;
                    let runtime = self
                        .link_dev
                        .as_ref()
                        .ok_or_else(|| "missing link runtime".to_string())?;
                    let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                    let (_, dc_abw, dc_bbw) =
                        exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                    return Ok(eval_coeff4_at(&scale_coeff4(dc_abw, scale), z_obs) * a_dir
                        + eval_coeff4_at(&scale_coeff4(dc_bbw, scale), z_obs) * dir[primary.g]);
                }
            }
        }
        Ok(0.0)
    }

    /// Directional derivative of the fixed chi second partial.
    fn observed_fixed_chi_second_partial_dir(
        &self,
        primary: &FlexPrimarySlices,
        u: usize,
        v: usize,
        z_obs: f64,
        u_obs: f64,
        a_dir: f64,
        dir: &Array1<f64>,
    ) -> Result<f64, String> {
        let scale = self.probit_frailty_scale();
        if u == primary.g && v == primary.g {
            return Ok(0.0);
        }
        if u == primary.g || v == primary.g {
            let other = if u == primary.g { v } else { u };
            if let Some(w_range) = primary.w.as_ref() {
                if other >= w_range.start && other < w_range.end {
                    let local_idx = other - w_range.start;
                    let runtime = self
                        .link_dev
                        .as_ref()
                        .ok_or_else(|| "missing link runtime".to_string())?;
                    let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                    let (_, dc_aabw, dc_abbw, _) =
                        exact_kernel::link_basis_cell_third_partials(basis_span);
                    return Ok(eval_coeff4_at(&scale_coeff4(dc_aabw, scale), z_obs) * a_dir
                        + eval_coeff4_at(&scale_coeff4(dc_abbw, scale), z_obs) * dir[primary.g]);
                }
            }
        }
        Ok(0.0)
    }

    /// Compute directional extensions of a timepoint's exact quantities.
    /// Given the base `SurvivalFlexTimepointExact`, returns the directional
    /// derivatives eta_uv_dir, chi_uv_dir, d_u_dir, d_uv_dir contracted
    /// with `dir`.
    fn compute_survival_timepoint_directional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir: &Array1<f64>,
        need_d_uv_dir: bool,
    ) -> Result<SurvivalFlexTimepointDirectionalExact, String> {
        let p = primary.total;
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct DirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Vec<f64>,
            f_au: Vec<f64>,
            f_uv: Vec<f64>,
            f_a_dir: f64,
            f_aa_dir: f64,
            f_au_dir: Vec<f64>,
            f_uv_dir: Vec<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(
                |cell_entry| -> Result<DirectionalTimepointCellAccum, String> {
                    let neg_cell = cell_entry.neg_cell;
                    let state = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let neg_dc_da: [f64; 4] = fixed.dc_da.map(|v| -v);
                    let neg_dc_daa: [f64; 4] = fixed.dc_daa.map(|v| -v);

                    let f_a = exact_kernel::cell_first_derivative_from_moments(
                        &neg_dc_da,
                        &state.moments,
                    )?;
                    let f_aa = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_dc_daa,
                        &state.moments,
                    )?;

                    let mut neg_coeff_dir = [0.0; 4];
                    let mut neg_coeff_a_dir = [0.0; 4];
                    let mut neg_coeff_aa_dir = [0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            neg_coeff_dir[k] -= fixed.coeff_u[c][k] * dir[c];
                            neg_coeff_a_dir[k] -= fixed.coeff_au[c][k] * dir[c];
                            neg_coeff_aa_dir[k] -= fixed.coeff_aau[c][k] * dir[c];
                        }
                    }

                    let f_a_dir = exact_kernel::cell_second_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_coeff_a_dir,
                        &state.moments,
                    )?;
                    let f_aa_dir = exact_kernel::cell_third_derivative_from_moments(
                        neg_cell,
                        &neg_dc_da,
                        &neg_dc_da,
                        &neg_coeff_dir,
                        &neg_dc_daa,
                        &neg_coeff_a_dir,
                        &neg_coeff_a_dir,
                        &neg_coeff_aa_dir,
                        &state.moments,
                    )?;

                    let mut f_u = vec![0.0; p];
                    let mut f_au = vec![0.0; p];
                    let mut f_uv = vec![0.0; p * p];
                    let mut f_au_dir = vec![0.0; p];
                    let mut f_uv_dir = vec![0.0; p * p];
                    for u in 0..p {
                        let neg_coeff_u = fixed.coeff_u[u].map(|v| -v);
                        let neg_coeff_au = fixed.coeff_au[u].map(|v| -v);

                        f_u[u] = exact_kernel::cell_first_derivative_from_moments(
                            &neg_coeff_u,
                            &state.moments,
                        )?;
                        f_au[u] = exact_kernel::cell_second_derivative_from_moments(
                            neg_cell,
                            &neg_dc_da,
                            &neg_coeff_u,
                            &neg_coeff_au,
                            &state.moments,
                        )?;

                        let mut neg_coeff_u_dir = [0.0; 4];
                        let mut neg_coeff_au_dir = [0.0; 4];
                        for c in 0..p {
                            if dir[c] == 0.0 {
                                continue;
                            }
                            let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                            let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                            for k in 0..4 {
                                neg_coeff_u_dir[k] -= sc[k] * dir[c];
                                neg_coeff_au_dir[k] -= sca[k] * dir[c];
                            }
                        }

                        f_au_dir[u] = exact_kernel::cell_third_derivative_from_moments(
                            neg_cell,
                            &neg_dc_da,
                            &neg_coeff_u,
                            &neg_coeff_dir,
                            &neg_coeff_au,
                            &neg_coeff_a_dir,
                            &neg_coeff_u_dir,
                            &neg_coeff_au_dir,
                            &state.moments,
                        )?;
                    }

                    for u in 0..p {
                        for v in u..p {
                            let neg_coeff_u = fixed.coeff_u[u].map(|val| -val);
                            let neg_coeff_v = fixed.coeff_u[v].map(|val| -val);
                            let sc_uv = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, v);
                            let neg_sc_uv = sc_uv.map(|val| -val);

                            let base_val = exact_kernel::cell_second_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_sc_uv,
                                &state.moments,
                            )?;
                            f_uv[u * p + v] = base_val;
                            f_uv[v * p + u] = base_val;

                            let mut neg_coeff_u_dir = [0.0; 4];
                            let mut neg_coeff_v_dir = [0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_uc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_vc =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                for k in 0..4 {
                                    neg_coeff_u_dir[k] -= sc_uc[k] * dir[c];
                                    neg_coeff_v_dir[k] -= sc_vc[k] * dir[c];
                                }
                            }

                            let dir_val = exact_kernel::cell_third_derivative_from_moments(
                                neg_cell,
                                &neg_coeff_u,
                                &neg_coeff_v,
                                &neg_coeff_dir,
                                &neg_sc_uv,
                                &neg_coeff_u_dir,
                                &neg_coeff_v_dir,
                                &[0.0; 4], // third cross vanishes for cubic cells
                                &state.moments,
                            )?;
                            f_uv_dir[u * p + v] = dir_val;
                            f_uv_dir[v * p + u] = dir_val;
                        }
                    }

                    Ok(DirectionalTimepointCellAccum {
                        f_a,
                        f_aa,
                        f_u,
                        f_au,
                        f_uv,
                        f_a_dir,
                        f_aa_dir,
                        f_au_dir,
                        f_uv_dir,
                    })
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0;
        let mut f_aa = 0.0;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_dir = 0.0;
        let mut f_aa_dir = 0.0;
        let mut f_au_dir = Array1::<f64>::zeros(p);
        let mut f_uv_dir = Array2::<f64>::zeros((p, p));
        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_dir += acc.f_a_dir;
            f_aa_dir += acc.f_aa_dir;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_dir[u] += acc.f_au_dir[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[u * p + v];
                    f_uv_dir[[u, v]] += acc.f_uv_dir[u * p + v];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        f_uv_dir[[q_index, q_index]] += dir[q_index] * (1.0 - q * q) * phi_q;

        let inv_f_a = 1.0 / f_a;
        let mut a_u = Array1::<f64>::zeros(p);
        for u in 0..p {
            a_u[u] = -f_u[u] * inv_f_a;
        }
        let mut a_uv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * a_u[v] + f_au[v] * a_u[u] + f_aa * a_u[u] * a_u[v])
                        * inv_f_a;
                a_uv[[u, v]] = val;
                a_uv[[v, u]] = val;
            }
        }
        let a_dir = a_u.dot(dir);
        let a_u_dir = a_uv.dot(dir);
        let mut a_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n_dir = f_uv_dir[[u, v]]
                    + f_au_dir[u] * a_u[v]
                    + f_au[u] * a_u_dir[v]
                    + f_au_dir[v] * a_u[u]
                    + f_au[v] * a_u_dir[u]
                    + f_aa_dir * a_u[u] * a_u[v]
                    + f_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v]);
                let val = -(n_dir + f_a_dir * a_uv[[u, v]]) * inv_f_a;
                a_uv_dir[[u, v]] = val;
                a_uv_dir[[v, u]] = val;
            }
        }

        // Observed-point quantities and their dir-extensions
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let chi_val = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut tau = Array1::<f64>::zeros(p);
        let mut g_au_fixed = vec![[0.0; 4]; p];
        let mut tau_a = Array1::<f64>::zeros(p);
        let mut g_bu_fixed = vec![[0.0; 4]; p];
        let mut g_aau_fixed = vec![[0.0; 4]; p];
        let mut g_abu_fixed = vec![[0.0; 4]; p];
        let mut g_bbu_fixed = vec![[0.0; 4]; p];
        let mut g_aaau_fixed = vec![[0.0; 4]; p];
        let mut g_aabu_fixed = vec![[0.0; 4]; p];
        let mut g_abbu_fixed = vec![[0.0; 4]; p];
        let mut g_bbbu_fixed = vec![[0.0; 4]; p];

        g_u_fixed[primary.g] = obs.dc_db;
        g_au_fixed[primary.g] = obs.dc_dab;
        g_bu_fixed[primary.g] = obs.dc_dbb;
        g_aau_fixed[primary.g] = obs.dc_daab;
        g_abu_fixed[primary.g] = obs.dc_dabb;
        g_bbu_fixed[primary.g] = obs.dc_dbbb;
        g_aaau_fixed[primary.g] = [0.0; 4];
        g_aabu_fixed[primary.g] = [0.0; 4];
        g_abbu_fixed[primary.g] = [0.0; 4];
        g_bbbu_fixed[primary.g] = [0.0; 4];

        tau[primary.g] = eval_coeff4_at(&obs.dc_dab, z_obs);
        tau_a[primary.g] = eval_coeff4_at(&obs.dc_daab, z_obs);
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, _) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                g_u_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                g_au_fixed[idx] = scale_coeff4(dc_aw, scale);
                g_bu_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b).1,
                    scale,
                );
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
                tau[idx] = eval_coeff4_at(&scale_coeff4(dc_aw, scale), z_obs);
                tau_a[idx] = eval_coeff4_at(&scale_coeff4(dc_aaw, scale), z_obs);
            }
        }

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                    scale,
                );
                g_bu_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                    scale,
                );
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let chi_dir = eta_aa * a_dir + tau.dot(dir);
        let eta_aa_dir = eta_aaa * a_dir
            + eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir, COEFF_SUPPORT_GW),
                z_obs,
            );
        let eta_aaa_dir = eval_coeff4_at(
            &g_jet.directional_family(g_jet.aaa_first, dir, COEFF_SUPPORT_GW),
            z_obs,
        );

        let mut tau_dir = Array1::<f64>::zeros(p);
        let mut tau_a_dir = Array1::<f64>::zeros(p);
        for u in 0..p {
            let fixed_tau_dir =
                g_jet.param_directional_from_b_family(g_jet.ab_first, u, dir, COEFF_SUPPORT_GW);
            tau_dir[u] = eval_coeff4_at(&g_jet.aa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_dir, z_obs);

            let fixed_tau_a_dir =
                g_jet.param_directional_from_b_family(g_jet.aab_first, u, dir, COEFF_SUPPORT_GW);
            tau_a_dir[u] = eval_coeff4_at(&g_jet.aaa_first[u], z_obs) * a_dir
                + eval_coeff4_at(&fixed_tau_a_dir, z_obs);
        }

        let mut eta_uv_dir = Array2::<f64>::zeros((p, p));
        let mut chi_uv_dir = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let r_uv_dir = self.observed_fixed_eta_second_partial_dir(
                    primary, &obs, u, v, z_obs, u_obs, a, b, a_dir, dir, beta_w,
                )?;
                let chi_uv_fixed_dir = self.observed_fixed_chi_second_partial_dir(
                    primary, u, v, z_obs, u_obs, a_dir, dir,
                )?;

                let eta_val = chi_dir * a_uv[[u, v]]
                    + chi_val * a_uv_dir[[u, v]]
                    + eta_aa_dir * a_u[u] * a_u[v]
                    + eta_aa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_dir[u] * a_u[v]
                    + tau[u] * a_u_dir[v]
                    + tau_dir[v] * a_u[u]
                    + tau[v] * a_u_dir[u]
                    + r_uv_dir;
                eta_uv_dir[[u, v]] = eta_val;
                eta_uv_dir[[v, u]] = eta_val;

                let chi_v = eta_aa_dir * a_uv[[u, v]]
                    + eta_aa * a_uv_dir[[u, v]]
                    + eta_aaa_dir * a_u[u] * a_u[v]
                    + eta_aaa * (a_u_dir[u] * a_u[v] + a_u[u] * a_u_dir[v])
                    + tau_a_dir[u] * a_u[v]
                    + tau_a[u] * a_u_dir[v]
                    + tau_a_dir[v] * a_u[u]
                    + tau_a[v] * a_u_dir[u]
                    + chi_uv_fixed_dir;
                chi_uv_dir[[u, v]] = chi_v;
                chi_uv_dir[[v, u]] = chi_v;
            }
        }

        // D_u_dir: directional derivative of the density normalization first derivative.
        let d_u_dir_cell_accums = cached
            .cells
            .iter()
            .map(|cell_entry| -> Result<Array1<f64>, String> {
                let mut d_u_dir = Array1::<f64>::zeros(p);
                let cell = cell_entry.partition_cell.cell;
                let state_ref = &cell_entry.state;
                let fixed = &cell_entry.fixed;
                let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                let chi_poly = fixed.dc_da.to_vec();
                let eta_aa_poly = fixed.dc_daa.to_vec();

                let mut eta_u_poly = vec![PolyVec::new(); p];
                let mut chi_u_poly = vec![PolyVec::new(); p];
                for u in 0..p {
                    eta_u_poly[u] =
                        poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u].to_vec());
                    chi_u_poly[u] = poly_add(
                        &poly_scale(&eta_aa_poly, a_u[u]),
                        &fixed.coeff_au[u].to_vec(),
                    );
                }

                let mut coeff_dir_poly = vec![0.0; 4];
                let mut coeff_a_dir_poly = vec![0.0; 4];
                for c in 0..p {
                    if dir[c] == 0.0 {
                        continue;
                    }
                    for k in 0..4 {
                        coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                        coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                    }
                }
                let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);

                for u in 0..p {
                    let mut eta_u_dir_fixed = vec![0.0; 4];
                    let mut chi_u_dir_fixed = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                        for k in 0..4 {
                            eta_u_dir_fixed[k] += sc[k] * dir[c];
                            chi_u_dir_fixed[k] += sca[k] * dir[c];
                        }
                    }
                    let eta_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&chi_poly, a_u_dir[u]),
                            &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                        ),
                        &eta_u_dir_fixed,
                    );
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();
                    let chi_u_dir_poly = poly_add(
                        &poly_add(
                            &poly_scale(&eta_aa_poly, a_u_dir[u]),
                            &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                        ),
                        &chi_u_dir_fixed,
                    );

                    // D_u integrand: chi_u - chi * eta * eta_u
                    let integrand_base = poly_sub(
                        &chi_u_poly[u],
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_poly[u]),
                    );
                    // Polynomial derivative of integrand w.r.t. dir
                    let integrand_dir = poly_sub(
                        &poly_sub(
                            &poly_sub(
                                &chi_u_dir_poly,
                                &poly_mul(&poly_mul(&coeff_a_dir_poly, &eta_poly), &eta_u_poly[u]),
                            ),
                            &poly_mul(&poly_mul(&chi_poly, &eta_dir_poly), &eta_u_poly[u]),
                        ),
                        &poly_mul(&poly_mul(&chi_poly, &eta_poly), &eta_u_dir_poly),
                    );
                    // Moment-weighting correction: -eta*eta_dir * integrand_base
                    let full_integrand = poly_sub(
                        &integrand_dir,
                        &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &integrand_base),
                    );

                    d_u_dir[u] += exact_kernel::cell_polynomial_integral_from_moments(
                        &full_integrand,
                        &state_ref.moments,
                        "survival D_t first derivative directional",
                    )?;
                }
                Ok(d_u_dir)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut d_u_dir = Array1::<f64>::zeros(p);
        for cell_d_u_dir in d_u_dir_cell_accums {
            for u in 0..p {
                d_u_dir[u] += cell_d_u_dir[u];
            }
        }

        // D_uv_dir
        let mut d_uv_dir = Array2::<f64>::zeros((p, p));
        if need_d_uv_dir {
            let d_uv_dir_cell_accums = cached
                .cells
                .iter()
                .map(|cell_entry| -> Result<Array2<f64>, String> {
                    let mut d_uv_dir = Array2::<f64>::zeros((p, p));
                    let cell = cell_entry.partition_cell.cell;
                    let state_ref = &cell_entry.state;
                    let fixed = &cell_entry.fixed;
                    let eta_poly = vec![cell.c0, cell.c1, cell.c2, cell.c3];
                    let chi_poly = fixed.dc_da.to_vec();
                    let eta_aa_poly = fixed.dc_daa.to_vec();
                    let eta_aaa_poly = fixed.dc_daaa.to_vec();

                    let mut eta_u_poly = vec![PolyVec::new(); p];
                    let mut chi_u_poly = vec![PolyVec::new(); p];
                    for u in 0..p {
                        eta_u_poly[u] =
                            poly_add(&poly_scale(&chi_poly, a_u[u]), &fixed.coeff_u[u].to_vec());
                        chi_u_poly[u] = poly_add(
                            &poly_scale(&eta_aa_poly, a_u[u]),
                            &fixed.coeff_au[u].to_vec(),
                        );
                    }
                    let mut coeff_dir_poly = vec![0.0; 4];
                    let mut coeff_a_dir_poly = vec![0.0; 4];
                    for c in 0..p {
                        if dir[c] == 0.0 {
                            continue;
                        }
                        for k in 0..4 {
                            coeff_dir_poly[k] += fixed.coeff_u[c][k] * dir[c];
                            coeff_a_dir_poly[k] += fixed.coeff_au[c][k] * dir[c];
                        }
                    }
                    let eta_dir_poly = poly_add(&poly_scale(&chi_poly, a_dir), &coeff_dir_poly);
                    let chi_dir_poly =
                        poly_add(&poly_scale(&eta_aa_poly, a_dir), &coeff_a_dir_poly);

                    for u in 0..p {
                        for v in u..p {
                            let r_uv_fixed = if u == primary.g {
                                fixed.coeff_bu[v].to_vec()
                            } else if v == primary.g {
                                fixed.coeff_bu[u].to_vec()
                            } else {
                                vec![0.0; 4]
                            };

                            let eta_uv_poly = poly_add(
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&chi_poly, a_uv[[u, v]]),
                                        &poly_scale(&eta_aa_poly, a_u[u] * a_u[v]),
                                    ),
                                    &poly_scale(&fixed.coeff_au[u].to_vec(), a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(&fixed.coeff_au[v].to_vec(), a_u[u]),
                                    &r_uv_fixed,
                                ),
                            );

                            // D_uv integrand: 5 terms
                            let t1 = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_uv[[u, v]]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_u[v]),
                                ),
                                &poly_add(
                                    &poly_scale(&fixed.coeff_aau[u].to_vec(), a_u[v]),
                                    &poly_add(
                                        &poly_scale(&fixed.coeff_aau[v].to_vec(), a_u[u]),
                                        &if u == primary.g {
                                            fixed.coeff_abu[v].to_vec()
                                        } else if v == primary.g {
                                            fixed.coeff_abu[u].to_vec()
                                        } else {
                                            vec![0.0; 4]
                                        },
                                    ),
                                ),
                            );
                            let t2 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[v], &eta_poly), &eta_u_poly[u]),
                                -1.0,
                            );
                            let t3 = poly_scale(
                                &poly_mul(&poly_mul(&chi_u_poly[u], &eta_poly), &eta_u_poly[v]),
                                -1.0,
                            );
                            let t4 = poly_scale(
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        &poly_mul(&eta_poly, &eta_uv_poly),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5 = poly_mul(
                                &chi_poly,
                                &poly_mul(
                                    &poly_mul(&eta_poly, &eta_poly),
                                    &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                ),
                            );
                            let i_base =
                                poly_add(&poly_add(&poly_add(&t1, &t2), &t3), &poly_add(&t4, &t5));

                            // Polynomial dir-derivatives of per-u quantities
                            let mut eu_dir_fixed_u = vec![0.0; 4];
                            let mut eu_dir_fixed_v = vec![0.0; 4];
                            let mut cu_dir_fixed_u = vec![0.0; 4];
                            let mut cu_dir_fixed_v = vec![0.0; 4];
                            for c in 0..p {
                                if dir[c] == 0.0 {
                                    continue;
                                }
                                let sc_u =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, u, c);
                                let sc_v =
                                    self.cell_pair_second_coeff(primary, &fixed.coeff_bu, v, c);
                                let sca_u =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, u, c);
                                let sca_v =
                                    self.cell_pair_third_coeff_a(primary, &fixed.coeff_abu, v, c);
                                for k in 0..4 {
                                    eu_dir_fixed_u[k] += sc_u[k] * dir[c];
                                    eu_dir_fixed_v[k] += sc_v[k] * dir[c];
                                    cu_dir_fixed_u[k] += sca_u[k] * dir[c];
                                    cu_dir_fixed_v[k] += sca_v[k] * dir[c];
                                }
                            }
                            let eta_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aa_poly, a_u[u] * a_dir),
                                ),
                                &eu_dir_fixed_u,
                            );
                            let eta_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aa_poly, a_u[v] * a_dir),
                                ),
                                &eu_dir_fixed_v,
                            );
                            let chi_u_dir_poly_u = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[u]),
                                    &poly_scale(&eta_aaa_poly, a_u[u] * a_dir),
                                ),
                                &cu_dir_fixed_u,
                            );
                            let chi_u_dir_poly_v = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_u_dir[v]),
                                    &poly_scale(&eta_aaa_poly, a_u[v] * a_dir),
                                ),
                                &cu_dir_fixed_v,
                            );
                            let eta_uv_dir_poly = poly_add(
                                &poly_add(
                                    &poly_scale(&chi_poly, a_uv_dir[[u, v]]),
                                    &poly_scale(
                                        &eta_aa_poly,
                                        a_u_dir[u] * a_u[v]
                                            + a_u[u] * a_u_dir[v]
                                            + a_uv[[u, v]] * a_dir,
                                    ),
                                ),
                                &poly_add(
                                    &poly_add(
                                        &poly_scale(&fixed.coeff_au[u].to_vec(), a_u_dir[v]),
                                        &poly_scale(&fixed.coeff_au[v].to_vec(), a_u_dir[u]),
                                    ),
                                    &{
                                        let mut fp = vec![0.0; 4];
                                        for c in 0..p {
                                            if dir[c] == 0.0 {
                                                continue;
                                            }
                                            let sca_u = self.cell_pair_third_coeff_a(
                                                primary,
                                                &fixed.coeff_abu,
                                                u,
                                                c,
                                            );
                                            let sca_v = self.cell_pair_third_coeff_a(
                                                primary,
                                                &fixed.coeff_abu,
                                                v,
                                                c,
                                            );
                                            for k in 0..4 {
                                                fp[k] += sca_u[k] * dir[c] * a_u[v]
                                                    + sca_v[k] * dir[c] * a_u[u];
                                            }
                                        }
                                        fp
                                    },
                                ),
                            );

                            // Differentiate each of the 5 integrand terms
                            let t1_dir = poly_add(
                                &poly_add(
                                    &poly_scale(&eta_aa_poly, a_uv_dir[[u, v]]),
                                    &poly_scale(
                                        &eta_aaa_poly,
                                        a_u_dir[u] * a_u[v]
                                            + a_u[u] * a_u_dir[v]
                                            + a_uv[[u, v]] * a_dir,
                                    ),
                                ),
                                &poly_add(
                                    &poly_scale(&fixed.coeff_aau[u].to_vec(), a_u_dir[v]),
                                    &poly_scale(&fixed.coeff_aau[v].to_vec(), a_u_dir[u]),
                                ),
                            );
                            let t2_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_v, &eta_poly),
                                            &eta_u_poly[u],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[v], &eta_dir_poly),
                                            &eta_u_poly[u],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[v], &eta_poly),
                                        &eta_u_dir_poly_u,
                                    ),
                                ),
                                -1.0,
                            );
                            let t3_dir = poly_scale(
                                &poly_add(
                                    &poly_add(
                                        &poly_mul(
                                            &poly_mul(&chi_u_dir_poly_u, &eta_poly),
                                            &eta_u_poly[v],
                                        ),
                                        &poly_mul(
                                            &poly_mul(&chi_u_poly[u], &eta_dir_poly),
                                            &eta_u_poly[v],
                                        ),
                                    ),
                                    &poly_mul(
                                        &poly_mul(&chi_u_poly[u], &eta_poly),
                                        &eta_u_dir_poly_v,
                                    ),
                                ),
                                -1.0,
                            );
                            let t4_dir = poly_scale(
                                &poly_add(
                                    &poly_mul(
                                        &chi_dir_poly,
                                        &poly_add(
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                            &poly_mul(&eta_poly, &eta_uv_poly),
                                        ),
                                    ),
                                    &poly_mul(
                                        &chi_poly,
                                        &poly_add(
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                            &poly_add(
                                                &poly_mul(&eta_dir_poly, &eta_uv_poly),
                                                &poly_mul(&eta_poly, &eta_uv_dir_poly),
                                            ),
                                        ),
                                    ),
                                ),
                                -1.0,
                            );
                            let t5_dir = poly_add(
                                &poly_mul(
                                    &chi_dir_poly,
                                    &poly_mul(
                                        &poly_mul(&eta_poly, &eta_poly),
                                        &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                    ),
                                ),
                                &poly_mul(
                                    &chi_poly,
                                    &poly_add(
                                        &poly_mul(
                                            &poly_scale(&poly_mul(&eta_dir_poly, &eta_poly), 2.0),
                                            &poly_mul(&eta_u_poly[u], &eta_u_poly[v]),
                                        ),
                                        &poly_mul(
                                            &poly_mul(&eta_poly, &eta_poly),
                                            &poly_add(
                                                &poly_mul(&eta_u_dir_poly_u, &eta_u_poly[v]),
                                                &poly_mul(&eta_u_poly[u], &eta_u_dir_poly_v),
                                            ),
                                        ),
                                    ),
                                ),
                            );

                            let i_base_dir = poly_add(
                                &poly_add(&poly_add(&t1_dir, &t2_dir), &t3_dir),
                                &poly_add(&t4_dir, &t5_dir),
                            );
                            let full_integrand = poly_sub(
                                &i_base_dir,
                                &poly_mul(&poly_mul(&eta_poly, &eta_dir_poly), &i_base),
                            );

                            let value = exact_kernel::cell_polynomial_integral_from_moments(
                                &full_integrand,
                                &state_ref.moments,
                                "survival D_t second derivative directional",
                            )?;
                            d_uv_dir[[u, v]] += value;
                            d_uv_dir[[v, u]] = d_uv_dir[[u, v]];
                        }
                    }
                    Ok(d_uv_dir)
                })
                .collect::<Result<Vec<_>, String>>()?;
            for cell_d_uv_dir in d_uv_dir_cell_accums {
                for u in 0..p {
                    for v in 0..p {
                        d_uv_dir[[u, v]] += cell_d_uv_dir[[u, v]];
                    }
                }
            }
        }

        Ok(SurvivalFlexTimepointDirectionalExact {
            eta_uv_dir,
            chi_uv_dir,
            d_u_dir,
            d_uv_dir,
        })
    }

    /// Exact mixed bidirectional extension D_{d1} D_{d2} of the timepoint
    /// quantities. This carries the calibration solve, observed eta/chi
    /// transport, and density-normalization transport analytically.
    fn compute_survival_timepoint_bidirectional_exact(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        self.compute_survival_timepoint_bidirectional_exact_full(
            row, primary, q, q_index, a, b, beta_h, beta_w, dir1, dir2,
        )
    }

    fn compute_survival_timepoint_bidirectional_exact_full(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        q: f64,
        q_index: usize,
        a: f64,
        b: f64,
        beta_h: Option<&Array1<f64>>,
        beta_w: Option<&Array1<f64>>,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<SurvivalFlexTimepointBiDirectionalExact, String> {
        let p = primary.total;
        let zero4 = [0.0; 4];
        let cached = self.build_cached_partition(primary, a, b, beta_h, beta_w)?;

        struct BiDirectionalTimepointCellAccum {
            f_a: f64,
            f_aa: f64,
            f_u: Array1<f64>,
            f_au: Array1<f64>,
            f_uv: Array2<f64>,
            f_a_d1: f64,
            f_aa_d1: f64,
            f_au_d1: Array1<f64>,
            f_uv_d1: Array2<f64>,
            f_a_d2: f64,
            f_aa_d2: f64,
            f_au_d2: Array1<f64>,
            f_uv_d2: Array2<f64>,
            f_a_d12: f64,
            f_aa_d12: f64,
            f_au_d12: Array1<f64>,
            f_uv_d12: Array2<f64>,
        }

        let cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<BiDirectionalTimepointCellAccum, String> {
                let mut f_a = 0.0f64;
                let mut f_aa = 0.0f64;
                let mut f_u = Array1::<f64>::zeros(p);
                let mut f_au = Array1::<f64>::zeros(p);
                let mut f_uv = Array2::<f64>::zeros((p, p));
                let mut f_a_d1 = 0.0f64;
                let mut f_aa_d1 = 0.0f64;
                let mut f_au_d1 = Array1::<f64>::zeros(p);
                let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
                let mut f_a_d2 = 0.0f64;
                let mut f_aa_d2 = 0.0f64;
                let mut f_au_d2 = Array1::<f64>::zeros(p);
                let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
                let mut f_a_d12 = 0.0f64;
                let mut f_aa_d12 = 0.0f64;
                let mut f_au_d12 = Array1::<f64>::zeros(p);
                let mut f_uv_d12 = Array2::<f64>::zeros((p, p));
                let nc = ce.neg_cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let da = fx.dc_da.map(|v| -v);
                let daa = fx.dc_daa.map(|v| -v);

                f_a += exact_kernel::cell_first_derivative_from_moments(&da, &st.moments)?;
                f_aa += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &daa,
                    &st.moments,
                )?;

                let mut cd1 = [0.0; 4];
                let mut ca1 = [0.0; 4];
                let mut caa1 = [0.0; 4];
                let mut cd2 = [0.0; 4];
                let mut ca2 = [0.0; 4];
                let mut caa2 = [0.0; 4];
                let mut cd12 = [0.0; 4];
                let mut ca12 = [0.0; 4];
                for c in 0..p {
                    for k in 0..4 {
                        if dir1[c] != 0.0 {
                            cd1[k] -= fx.coeff_u[c][k] * dir1[c];
                            ca1[k] -= fx.coeff_au[c][k] * dir1[c];
                            caa1[k] -= fx.coeff_aau[c][k] * dir1[c];
                        }
                        if dir2[c] != 0.0 {
                            cd2[k] -= fx.coeff_u[c][k] * dir2[c];
                            ca2[k] -= fx.coeff_au[c][k] * dir2[c];
                            caa2[k] -= fx.coeff_aau[c][k] * dir2[c];
                        }
                    }
                }
                for c1 in 0..p {
                    if dir1[c1] == 0.0 {
                        continue;
                    }
                    for c2 in 0..p {
                        if dir2[c2] == 0.0 {
                            continue;
                        }
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, c1, c2);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, c1, c2);
                        for k in 0..4 {
                            cd12[k] -= sc[k] * dir1[c1] * dir2[c2];
                            ca12[k] -= sca[k] * dir1[c1] * dir2[c2];
                        }
                    }
                }

                f_a_d1 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &ca1,
                    &st.moments,
                )?;
                f_a_d2 += exact_kernel::cell_second_derivative_from_moments(
                    nc,
                    &da,
                    &cd2,
                    &ca2,
                    &st.moments,
                )?;
                f_a_d12 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &cd1,
                    &cd2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &ca12,
                    &st.moments,
                )?;
                f_aa_d1 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &daa,
                    &ca1,
                    &ca1,
                    &caa1,
                    &st.moments,
                )?;
                f_aa_d2 += exact_kernel::cell_third_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd2,
                    &daa,
                    &ca2,
                    &ca2,
                    &caa2,
                    &st.moments,
                )?;
                f_aa_d12 += exact_kernel::cell_fourth_derivative_from_moments(
                    nc,
                    &da,
                    &da,
                    &cd1,
                    &cd2,
                    &daa,
                    &ca1,
                    &ca2,
                    &ca1,
                    &ca2,
                    &cd12,
                    &caa1,
                    &caa2,
                    &ca12,
                    &ca12,
                    &[0.0; 4],
                    &st.moments,
                )?;

                for u in 0..p {
                    let cu = fx.coeff_u[u].map(|v| -v);
                    let cau = fx.coeff_au[u].map(|v| -v);
                    f_u[u] += exact_kernel::cell_first_derivative_from_moments(&cu, &st.moments)?;
                    f_au[u] += exact_kernel::cell_second_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cau,
                        &st.moments,
                    )?;
                    let mut cu1 = [0.0; 4];
                    let mut cau1 = [0.0; 4];
                    let mut cu2 = [0.0; 4];
                    let mut cau2 = [0.0; 4];
                    for c in 0..p {
                        let sc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                        let sca = self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, c);
                        for k in 0..4 {
                            if dir1[c] != 0.0 {
                                cu1[k] -= sc[k] * dir1[c];
                                cau1[k] -= sca[k] * dir1[c];
                            }
                            if dir2[c] != 0.0 {
                                cu2[k] -= sc[k] * dir2[c];
                                cau2[k] -= sca[k] * dir2[c];
                            }
                        }
                    }
                    f_au_d1[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cau,
                        &ca1,
                        &cu1,
                        &cau1,
                        &st.moments,
                    )?;
                    f_au_d2[u] += exact_kernel::cell_third_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd2,
                        &cau,
                        &ca2,
                        &cu2,
                        &cau2,
                        &st.moments,
                    )?;
                    f_au_d12[u] += exact_kernel::cell_fourth_derivative_from_moments(
                        nc,
                        &da,
                        &cu,
                        &cd1,
                        &cd2,
                        &cau,
                        &ca1,
                        &ca2,
                        &cu1,
                        &cu2,
                        &cd12,
                        &cau1,
                        &cau2,
                        &ca12,
                        &[0.0; 4],
                        &[0.0; 4],
                        &st.moments,
                    )?;
                }
                for u in 0..p {
                    for v in u..p {
                        let cu = fx.coeff_u[u].map(|x| -x);
                        let cv = fx.coeff_u[v].map(|x| -x);
                        let sc = self
                            .cell_pair_second_coeff(primary, &fx.coeff_bu, u, v)
                            .map(|x| -x);
                        let bv = exact_kernel::cell_second_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &sc,
                            &st.moments,
                        )?;
                        f_uv[[u, v]] += bv;
                        if u != v {
                            f_uv[[v, u]] += bv;
                        }
                        let mut cu1 = [0.0; 4];
                        let mut cv1 = [0.0; 4];
                        let mut cu2 = [0.0; 4];
                        let mut cv2 = [0.0; 4];
                        for c in 0..p {
                            let suc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, c);
                            let svc = self.cell_pair_second_coeff(primary, &fx.coeff_bu, v, c);
                            for k in 0..4 {
                                if dir1[c] != 0.0 {
                                    cu1[k] -= suc[k] * dir1[c];
                                    cv1[k] -= svc[k] * dir1[c];
                                }
                                if dir2[c] != 0.0 {
                                    cu2[k] -= suc[k] * dir2[c];
                                    cv2[k] -= svc[k] * dir2[c];
                                }
                            }
                        }
                        let d1v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &sc,
                            &cu1,
                            &cv1,
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d1[[u, v]] += d1v;
                        if u != v {
                            f_uv_d1[[v, u]] += d1v;
                        }
                        let d2v = exact_kernel::cell_third_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd2,
                            &sc,
                            &cu2,
                            &cv2,
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d2[[u, v]] += d2v;
                        if u != v {
                            f_uv_d2[[v, u]] += d2v;
                        }
                        let d12v = exact_kernel::cell_fourth_derivative_from_moments(
                            nc,
                            &cu,
                            &cv,
                            &cd1,
                            &cd2,
                            &sc,
                            &cu1,
                            &cu2,
                            &cv1,
                            &cv2,
                            &cd12,
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &[0.0; 4],
                            &st.moments,
                        )?;
                        f_uv_d12[[u, v]] += d12v;
                        if u != v {
                            f_uv_d12[[v, u]] += d12v;
                        }
                    }
                }

                Ok(BiDirectionalTimepointCellAccum {
                    f_a,
                    f_aa,
                    f_u,
                    f_au,
                    f_uv,
                    f_a_d1,
                    f_aa_d1,
                    f_au_d1,
                    f_uv_d1,
                    f_a_d2,
                    f_aa_d2,
                    f_au_d2,
                    f_uv_d2,
                    f_a_d12,
                    f_aa_d12,
                    f_au_d12,
                    f_uv_d12,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut f_a = 0.0f64;
        let mut f_aa = 0.0f64;
        let mut f_u = Array1::<f64>::zeros(p);
        let mut f_au = Array1::<f64>::zeros(p);
        let mut f_uv = Array2::<f64>::zeros((p, p));
        let mut f_a_d1 = 0.0f64;
        let mut f_aa_d1 = 0.0f64;
        let mut f_au_d1 = Array1::<f64>::zeros(p);
        let mut f_uv_d1 = Array2::<f64>::zeros((p, p));
        let mut f_a_d2 = 0.0f64;
        let mut f_aa_d2 = 0.0f64;
        let mut f_au_d2 = Array1::<f64>::zeros(p);
        let mut f_uv_d2 = Array2::<f64>::zeros((p, p));
        let mut f_a_d12 = 0.0f64;
        let mut f_aa_d12 = 0.0f64;
        let mut f_au_d12 = Array1::<f64>::zeros(p);
        let mut f_uv_d12 = Array2::<f64>::zeros((p, p));

        for acc in cell_accums {
            f_a += acc.f_a;
            f_aa += acc.f_aa;
            f_a_d1 += acc.f_a_d1;
            f_aa_d1 += acc.f_aa_d1;
            f_a_d2 += acc.f_a_d2;
            f_aa_d2 += acc.f_aa_d2;
            f_a_d12 += acc.f_a_d12;
            f_aa_d12 += acc.f_aa_d12;
            for u in 0..p {
                f_u[u] += acc.f_u[u];
                f_au[u] += acc.f_au[u];
                f_au_d1[u] += acc.f_au_d1[u];
                f_au_d2[u] += acc.f_au_d2[u];
                f_au_d12[u] += acc.f_au_d12[u];
                for v in 0..p {
                    f_uv[[u, v]] += acc.f_uv[[u, v]];
                    f_uv_d1[[u, v]] += acc.f_uv_d1[[u, v]];
                    f_uv_d2[[u, v]] += acc.f_uv_d2[[u, v]];
                    f_uv_d12[[u, v]] += acc.f_uv_d12[[u, v]];
                }
            }
        }

        let phi_q = crate::probability::normal_pdf(q);
        f_u[q_index] += phi_q;
        f_uv[[q_index, q_index]] += -q * phi_q;
        f_uv_d1[[q_index, q_index]] += dir1[q_index] * (1.0 - q * q) * phi_q;
        f_uv_d2[[q_index, q_index]] += dir2[q_index] * (1.0 - q * q) * phi_q;
        f_uv_d12[[q_index, q_index]] += dir1[q_index] * dir2[q_index] * q * (q * q - 3.0) * phi_q;

        let inv = 1.0 / f_a;
        let mut au = Array1::<f64>::zeros(p);
        for u in 0..p {
            au[u] = -f_u[u] * inv;
        }
        let mut auv = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let val =
                    -(f_uv[[u, v]] + f_au[u] * au[v] + f_au[v] * au[u] + f_aa * au[u] * au[v])
                        * inv;
                auv[[u, v]] = val;
                auv[[v, u]] = val;
            }
        }
        let ad1 = au.dot(dir1);
        let ad2 = au.dot(dir2);
        let aud1 = auv.dot(dir1);
        let aud2 = auv.dot(dir2);

        let mut auvd1 = Array2::<f64>::zeros((p, p));
        let mut auvd2 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n1 = f_uv_d1[[u, v]]
                    + f_au_d1[u] * au[v]
                    + f_au[u] * aud1[v]
                    + f_au_d1[v] * au[u]
                    + f_au[v] * aud1[u]
                    + f_aa_d1 * au[u] * au[v]
                    + f_aa * (aud1[u] * au[v] + au[u] * aud1[v]);
                let v1 = -(n1 + f_a_d1 * auv[[u, v]]) * inv;
                auvd1[[u, v]] = v1;
                auvd1[[v, u]] = v1;

                let n2 = f_uv_d2[[u, v]]
                    + f_au_d2[u] * au[v]
                    + f_au[u] * aud2[v]
                    + f_au_d2[v] * au[u]
                    + f_au[v] * aud2[u]
                    + f_aa_d2 * au[u] * au[v]
                    + f_aa * (aud2[u] * au[v] + au[u] * aud2[v]);
                let v2 = -(n2 + f_a_d2 * auv[[u, v]]) * inv;
                auvd2[[u, v]] = v2;
                auvd2[[v, u]] = v2;
            }
        }

        let ad12 = aud2.dot(dir1);
        let aud12 = auvd2.dot(dir1);
        let mut auvd12 = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let n = f_uv_d12[[u, v]]
                    + f_au_d12[u] * au[v]
                    + f_au_d1[u] * aud2[v]
                    + f_au_d2[u] * aud1[v]
                    + f_au[u] * aud12[v]
                    + f_au_d12[v] * au[u]
                    + f_au_d1[v] * aud2[u]
                    + f_au_d2[v] * aud1[u]
                    + f_au[v] * aud12[u]
                    + f_aa_d12 * au[u] * au[v]
                    + f_aa_d1 * (aud2[u] * au[v] + au[u] * aud2[v])
                    + f_aa_d2 * (aud1[u] * au[v] + au[u] * aud1[v])
                    + f_aa
                        * (aud12[u] * au[v]
                            + aud1[u] * aud2[v]
                            + aud2[u] * aud1[v]
                            + au[u] * aud12[v]);
                let val =
                    -(n + f_a_d12 * auv[[u, v]] + f_a_d1 * auvd2[[u, v]] + f_a_d2 * auvd1[[u, v]])
                        * inv;
                auvd12[[u, v]] = val;
                auvd12[[v, u]] = val;
            }
        }

        let obs = self.observed_denested_cell_partials(row, a, b, beta_h, beta_w)?;
        let z_obs = self.observed_score_projection(row);
        let u_obs = a + b * z_obs;
        let scale = self.probit_frailty_scale();

        let mut g_u_fixed = vec![[0.0; 4]; p];
        let mut g_au_fixed = vec![[0.0; 4]; p];
        let mut g_bu_fixed = vec![[0.0; 4]; p];
        let mut g_aau_fixed = vec![[0.0; 4]; p];
        let mut g_abu_fixed = vec![[0.0; 4]; p];
        let mut g_bbu_fixed = vec![[0.0; 4]; p];
        let mut g_aaau_fixed = vec![[0.0; 4]; p];
        let mut g_aabu_fixed = vec![[0.0; 4]; p];
        let mut g_abbu_fixed = vec![[0.0; 4]; p];
        let mut g_bbbu_fixed = vec![[0.0; 4]; p];

        g_u_fixed[primary.g] = obs.dc_db;
        g_au_fixed[primary.g] = obs.dc_dab;
        g_bu_fixed[primary.g] = obs.dc_dbb;
        g_aau_fixed[primary.g] = obs.dc_daab;
        g_abu_fixed[primary.g] = obs.dc_dabb;
        g_bbu_fixed[primary.g] = obs.dc_dbbb;
        g_aaau_fixed[primary.g] = [0.0; 4];
        g_aabu_fixed[primary.g] = [0.0; 4];
        g_abbu_fixed[primary.g] = [0.0; 4];
        g_bbbu_fixed[primary.g] = [0.0; 4];

        if let Some(h_range) = primary.h.as_ref().filter(|_| self.score_warp.is_some()) {
            for local_idx in 0..h_range.len() {
                let idx = h_range.start + local_idx;
                g_u_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, b)?,
                    scale,
                );
                g_bu_fixed[idx] = scale_coeff4(
                    self.observed_score_basis_coefficients(row, local_idx, z_obs, 1.0)?,
                    scale,
                );
            }
        }
        if let (Some(w_range), Some(runtime)) = (primary.w.as_ref(), self.link_dev.as_ref()) {
            for local_idx in 0..w_range.len() {
                let basis_span = runtime.basis_cubic_at(local_idx, u_obs)?;
                let idx = w_range.start + local_idx;
                let (dc_aw, dc_bw) =
                    exact_kernel::link_basis_cell_coefficient_partials(basis_span, a, b);
                let (dc_aaw, dc_abw, dc_bbw) =
                    exact_kernel::link_basis_cell_second_partials(basis_span, a, b);
                let (dc_aaaw, dc_aabw, dc_abbw, dc_bbbw) =
                    exact_kernel::link_basis_cell_third_partials(basis_span);
                g_u_fixed[idx] = scale_coeff4(
                    exact_kernel::link_basis_cell_coefficients(basis_span, a, b),
                    scale,
                );
                g_au_fixed[idx] = scale_coeff4(dc_aw, scale);
                g_bu_fixed[idx] = scale_coeff4(dc_bw, scale);
                g_aau_fixed[idx] = scale_coeff4(dc_aaw, scale);
                g_abu_fixed[idx] = scale_coeff4(dc_abw, scale);
                g_bbu_fixed[idx] = scale_coeff4(dc_bbw, scale);
                g_aaau_fixed[idx] = scale_coeff4(dc_aaaw, scale);
                g_aabu_fixed[idx] = scale_coeff4(dc_aabw, scale);
                g_abbu_fixed[idx] = scale_coeff4(dc_abbw, scale);
                g_bbbu_fixed[idx] = scale_coeff4(dc_bbbw, scale);
            }
        }

        let g_jet = SparsePrimaryCoeffJetView::new(
            primary.g,
            primary.h.as_ref(),
            primary.w.as_ref(),
            &[],
            &g_au_fixed,
            &g_bu_fixed,
            &g_aau_fixed,
            &g_abu_fixed,
            &g_bbu_fixed,
            &g_aaau_fixed,
            &g_aabu_fixed,
            &g_abbu_fixed,
            &g_bbbu_fixed,
        );

        let chi = eval_coeff4_at(&obs.dc_da, z_obs);
        let eta_aa = eval_coeff4_at(&obs.dc_daa, z_obs);
        let eta_aaa = eval_coeff4_at(&obs.dc_daaa, z_obs);
        let chi_jet = scalar_composite_bilinear(
            chi,
            eta_aa,
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.a_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.ab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aa_jet = scalar_composite_bilinear(
            eta_aa,
            eta_aaa,
            0.0,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            ad1,
            ad2,
            ad12,
        );
        let eta_aaa_jet = MultiDirJet::bilinear(
            eta_aaa,
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir1, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.directional_family(g_jet.aaa_first, dir2, COEFF_SUPPORT_GW),
                z_obs,
            ),
            eval_coeff4_at(
                &g_jet.mixed_directional_from_b_family(
                    g_jet.aab_first,
                    dir1,
                    dir2,
                    COEFF_SUPPORT_GW,
                ),
                z_obs,
            ),
        );

        let mut a_u_jets = Vec::with_capacity(p);
        let mut tau_jets = Vec::with_capacity(p);
        let mut tau_a_jets = Vec::with_capacity(p);
        for u in 0..p {
            a_u_jets.push(MultiDirJet::bilinear(au[u], aud1[u], aud2[u], aud12[u]));
            tau_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_au_fixed[u], z_obs),
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.ab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                ad1,
                ad2,
                ad12,
            ));
            tau_a_jets.push(scalar_composite_bilinear(
                eval_coeff4_at(&g_aau_fixed[u], z_obs),
                eval_coeff4_at(&g_aaau_fixed[u], z_obs),
                0.0,
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir1,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_directional_from_b_family(
                        g_jet.aab_first,
                        u,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                eval_coeff4_at(
                    &g_jet.param_mixed_from_bb_family(
                        g_jet.abb_first,
                        u,
                        dir1,
                        dir2,
                        COEFF_SUPPORT_GW,
                    ),
                    z_obs,
                ),
                0.0,
                0.0,
                ad1,
                ad2,
                ad12,
            ));
        }

        let mut eta_uv_uv = Array2::<f64>::zeros((p, p));
        let mut chi_uv_uv = Array2::<f64>::zeros((p, p));
        let mut d_uv_uv = Array2::<f64>::zeros((p, p));

        for u in 0..p {
            for v in u..p {
                let a_uv_jet = MultiDirJet::bilinear(
                    auv[[u, v]],
                    auvd1[[u, v]],
                    auvd2[[u, v]],
                    auvd12[[u, v]],
                );
                let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                let r_uv_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.b_first, u, v, COEFF_SUPPORT_GHW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.bb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GHW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_uv_fixed_jet = scalar_composite_bilinear(
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.ab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_from_b_family(g_jet.aab_first, u, v, COEFF_SUPPORT_GW),
                        z_obs,
                    ),
                    0.0,
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir1,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_directional_from_bb_family(
                            g_jet.abb_first,
                            u,
                            v,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    eval_coeff4_at(
                        &g_jet.pair_mixed_from_bbb_family(
                            g_jet.bbb_first,
                            u,
                            v,
                            dir1,
                            dir2,
                            COEFF_SUPPORT_GW,
                        ),
                        z_obs,
                    ),
                    0.0,
                    0.0,
                    ad1,
                    ad2,
                    ad12,
                );

                let eta_uv_jet = chi_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aa_jet.mul(&a_u_prod))
                    .add(&tau_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_jets[v].mul(&a_u_jets[u]))
                    .add(&r_uv_jet);
                let chi_uv_jet = eta_aa_jet
                    .mul(&a_uv_jet)
                    .add(&eta_aaa_jet.mul(&a_u_prod))
                    .add(&tau_a_jets[u].mul(&a_u_jets[v]))
                    .add(&tau_a_jets[v].mul(&a_u_jets[u]))
                    .add(&chi_uv_fixed_jet);

                eta_uv_uv[[u, v]] = eta_uv_jet.coeff(3);
                eta_uv_uv[[v, u]] = eta_uv_uv[[u, v]];
                chi_uv_uv[[u, v]] = chi_uv_jet.coeff(3);
                chi_uv_uv[[v, u]] = chi_uv_uv[[u, v]];
            }
        }

        let d_uv_uv_cell_accums = cached
            .cells
            .iter()
            .map(|ce| -> Result<Array2<f64>, String> {
                let mut d_uv_uv = Array2::<f64>::zeros((p, p));
                let cell = ce.partition_cell.cell;
                let st = &ce.state;
                let fx = &ce.fixed;
                let eta_base = [cell.c0, cell.c1, cell.c2, cell.c3];

                let coeff_dir1 = self.cell_directional_coeff_family(primary, &fx.coeff_u, dir1);
                let coeff_dir2 = self.cell_directional_coeff_family(primary, &fx.coeff_u, dir2);
                let coeff_dir12 =
                    self.cell_mixed_directional_from_b_family(primary, &fx.coeff_bu, dir1, dir2);
                let coeff_a_dir1 = self.cell_directional_coeff_family(primary, &fx.coeff_au, dir1);
                let coeff_a_dir2 = self.cell_directional_coeff_family(primary, &fx.coeff_au, dir2);
                let coeff_a_dir12 =
                    self.cell_mixed_directional_from_b_family(primary, &fx.coeff_abu, dir1, dir2);
                let coeff_aa_dir1 =
                    self.cell_directional_coeff_family(primary, &fx.coeff_aau, dir1);
                let coeff_aa_dir2 =
                    self.cell_directional_coeff_family(primary, &fx.coeff_aau, dir2);
                let coeff_aa_dir12 =
                    self.cell_mixed_directional_from_b_family(primary, &fx.coeff_aabu, dir1, dir2);
                let coeff_aaa_dir1 =
                    self.cell_directional_coeff_family(primary, &fx.coeff_aaau, dir1);
                let coeff_aaa_dir2 =
                    self.cell_directional_coeff_family(primary, &fx.coeff_aaau, dir2);
                let coeff_aaa_dir12 =
                    self.cell_mixed_directional_from_b_family(primary, &fx.coeff_aabu, dir1, dir2);

                let eta_poly_jet = coeff4_composite_bilinear(
                    &eta_base,
                    &fx.dc_da,
                    &fx.dc_daa,
                    &coeff_dir1,
                    &coeff_dir2,
                    &coeff_dir12,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let chi_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_da,
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &coeff_a_dir1,
                    &coeff_a_dir2,
                    &coeff_a_dir12,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aa_poly_jet = coeff4_composite_bilinear(
                    &fx.dc_daa,
                    &fx.dc_daaa,
                    &zero4,
                    &coeff_aa_dir1,
                    &coeff_aa_dir2,
                    &coeff_aa_dir12,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    ad1,
                    ad2,
                    ad12,
                );
                let eta_aaa_poly_jet = coeff4_fixed_bilinear(
                    &fx.dc_daaa,
                    &coeff_aaa_dir1,
                    &coeff_aaa_dir2,
                    &coeff_aaa_dir12,
                );

                let mut eta_u_poly_jets = Vec::with_capacity(p);
                let mut chi_u_poly_jets = Vec::with_capacity(p);
                let mut coeff_au_fixed_jets = Vec::with_capacity(p);
                let mut coeff_aau_fixed_jets = Vec::with_capacity(p);
                for u in 0..p {
                    let coeff_u_dir1 =
                        self.cell_param_directional_from_b_family(primary, &fx.coeff_bu, u, dir1);
                    let coeff_u_dir2 =
                        self.cell_param_directional_from_b_family(primary, &fx.coeff_bu, u, dir2);
                    let coeff_u_dir12 =
                        self.cell_param_mixed_from_bb_family(primary, &fx.coeff_bbu, u, dir1, dir2);
                    let coeff_au_dir1 =
                        self.cell_param_directional_from_b_family(primary, &fx.coeff_abu, u, dir1);
                    let coeff_au_dir2 =
                        self.cell_param_directional_from_b_family(primary, &fx.coeff_abu, u, dir2);
                    let coeff_au_dir12 = self.cell_param_mixed_from_bb_family(
                        primary,
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                    );
                    let coeff_aau_dir1 =
                        self.cell_param_directional_from_b_family(primary, &fx.coeff_aabu, u, dir1);
                    let coeff_aau_dir2 =
                        self.cell_param_directional_from_b_family(primary, &fx.coeff_aabu, u, dir2);
                    let coeff_aau_dir12 = self.cell_param_mixed_from_bb_family(
                        primary,
                        &fx.coeff_abbu,
                        u,
                        dir1,
                        dir2,
                    );

                    let coeff_u_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_u[u],
                        &coeff_u_dir1,
                        &coeff_u_dir2,
                        &coeff_u_dir12,
                    );
                    let coeff_au_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_au[u],
                        &coeff_au_dir1,
                        &coeff_au_dir2,
                        &coeff_au_dir12,
                    );
                    let coeff_aau_fixed_jet = coeff4_fixed_bilinear(
                        &fx.coeff_aau[u],
                        &coeff_aau_dir1,
                        &coeff_aau_dir2,
                        &coeff_aau_dir12,
                    );

                    eta_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&chi_poly_jet, &a_u_jets[u]),
                        &coeff_u_fixed_jet,
                    ));
                    chi_u_poly_jets.push(poly_add_jets(
                        &poly_scale_jets(&eta_aa_poly_jet, &a_u_jets[u]),
                        &coeff_au_fixed_jet,
                    ));
                    coeff_au_fixed_jets.push(coeff_au_fixed_jet);
                    coeff_aau_fixed_jets.push(coeff_aau_fixed_jet);
                }

                for u in 0..p {
                    for v in u..p {
                        let a_uv_jet = MultiDirJet::bilinear(
                            auv[[u, v]],
                            auvd1[[u, v]],
                            auvd2[[u, v]],
                            auvd12[[u, v]],
                        );
                        let a_u_prod = a_u_jets[u].mul(&a_u_jets[v]);
                        let r_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_second_coeff(primary, &fx.coeff_bu, u, v),
                            &self.cell_pair_directional_from_bb_family(
                                primary,
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir1,
                            ),
                            &self.cell_pair_directional_from_bb_family(
                                primary,
                                &fx.coeff_bbu,
                                u,
                                v,
                                dir2,
                            ),
                            &self.cell_pair_mixed_from_bbb_family(
                                primary,
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                            ),
                        );
                        let chi_uv_fixed_jet = coeff4_fixed_bilinear(
                            &self.cell_pair_third_coeff_a(primary, &fx.coeff_abu, u, v),
                            &self.cell_pair_directional_from_bb_family(
                                primary,
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir1,
                            ),
                            &self.cell_pair_directional_from_bb_family(
                                primary,
                                &fx.coeff_abbu,
                                u,
                                v,
                                dir2,
                            ),
                            &self.cell_pair_mixed_from_bbb_family(
                                primary,
                                &fx.coeff_bbbu,
                                u,
                                v,
                                dir1,
                                dir2,
                            ),
                        );

                        let eta_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&chi_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_au_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_au_fixed_jets[v], &a_u_jets[u]),
                                    &r_uv_fixed_jet,
                                ),
                            ),
                        );
                        let chi_uv_poly_jet = poly_add_jets(
                            &poly_add_jets(
                                &poly_scale_jets(&eta_aa_poly_jet, &a_uv_jet),
                                &poly_scale_jets(&eta_aaa_poly_jet, &a_u_prod),
                            ),
                            &poly_add_jets(
                                &poly_scale_jets(&coeff_aau_fixed_jets[u], &a_u_jets[v]),
                                &poly_add_jets(
                                    &poly_scale_jets(&coeff_aau_fixed_jets[v], &a_u_jets[u]),
                                    &chi_uv_fixed_jet,
                                ),
                            ),
                        );

                        let t1 = chi_uv_poly_jet.clone();
                        let t2 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[v], &eta_poly_jet),
                                &eta_u_poly_jets[u],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t3 = poly_scale_jets(
                            &poly_mul_jets(
                                &poly_mul_jets(&chi_u_poly_jets[u], &eta_poly_jet),
                                &eta_u_poly_jets[v],
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t4 = poly_scale_jets(
                            &poly_mul_jets(
                                &chi_poly_jet,
                                &poly_add_jets(
                                    &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                                    &poly_mul_jets(&eta_poly_jet, &eta_uv_poly_jet),
                                ),
                            ),
                            &MultiDirJet::constant(2, -1.0),
                        );
                        let t5 = poly_mul_jets(
                            &chi_poly_jet,
                            &poly_mul_jets(
                                &poly_mul_jets(&eta_poly_jet, &eta_poly_jet),
                                &poly_mul_jets(&eta_u_poly_jets[u], &eta_u_poly_jets[v]),
                            ),
                        );
                        let i_base_jet = poly_add_jets(
                            &poly_add_jets(&poly_add_jets(&t1, &t2), &t3),
                            &poly_add_jets(&t4, &t5),
                        );

                        let i_base = poly_coeff_mask(&i_base_jet, 0);
                        let i_base_d2 = poly_coeff_mask(&i_base_jet, 2);
                        let i_base_d12 = poly_coeff_mask(&i_base_jet, 3);
                        let eta_poly = poly_coeff_mask(&eta_poly_jet, 0);
                        let eta_d1_poly = poly_coeff_mask(&eta_poly_jet, 1);
                        let eta_d2_poly = poly_coeff_mask(&eta_poly_jet, 2);
                        let eta_d12_poly = poly_coeff_mask(&eta_poly_jet, 3);

                        let correction = poly_add(
                            &poly_mul(
                                &poly_add(
                                    &poly_mul(&eta_d2_poly, &eta_d1_poly),
                                    &poly_mul(&eta_poly, &eta_d12_poly),
                                ),
                                &i_base,
                            ),
                            &poly_mul(&poly_mul(&eta_poly, &eta_d1_poly), &i_base_d2),
                        );
                        let full_integrand = poly_sub(&i_base_d12, &correction);
                        let value = exact_kernel::cell_polynomial_integral_from_moments(
                            &full_integrand,
                            &st.moments,
                            "survival D_t second derivative bidirectional",
                        )?;
                        d_uv_uv[[u, v]] += value;
                        d_uv_uv[[v, u]] = d_uv_uv[[u, v]];
                    }
                }
                Ok(d_uv_uv)
            })
            .collect::<Result<Vec<_>, String>>()?;
        for cell_d_uv_uv in d_uv_uv_cell_accums {
            for u in 0..p {
                for v in 0..p {
                    d_uv_uv[[u, v]] += cell_d_uv_uv[[u, v]];
                }
            }
        }

        Ok(SurvivalFlexTimepointBiDirectionalExact {
            eta_uv_uv,
            chi_uv_uv,
            d_uv_uv,
        })
    }

    /// Exact third-order directional contraction for the flexible survival
    /// path.  Returns D_dir H[u,v] where H is the primary-space NLL Hessian.
    fn row_flex_primary_third_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_third_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival third contracted: dir length {} != primary dimension {p}",
                    dir.len()
                ),
            }
            .into());
        }
        if dir.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival third contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry = self.compute_survival_timepoint_exact(
            row, &primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, false,
        )?;
        let exit = self.compute_survival_timepoint_exact(
            row, &primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, true,
        )?;

        if !exit.chi.is_finite() || exit.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival third contracted row {row}: non-positive chi1={:.3e}",
                    exit.chi,
                ),
            }
            .into());
        }

        let entry_ext = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir, false,
        )?;
        let exit_ext = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir, true,
        )?;

        let wi = self.weights[row];
        let di = self.event[row];

        let (entry_k1, entry_k2, entry_k3, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-entry.eta, -wi)?;
        let (exit_k1, exit_k2, exit_k3, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit.eta, wi * (1.0 - di))?;

        let entry_u1 = -entry_k1;
        let entry_u2 = entry_k2;
        let entry_u3 = -entry_k3;
        let exit_u1 = -exit_k1;
        let exit_u2 = exit_k2;
        let exit_u3 = -exit_k3;

        let entry_eta_dir = entry.eta_u.dot(dir);
        let exit_eta_dir = exit.eta_u.dot(dir);
        let exit_chi_dir = exit.chi_u.dot(dir);
        let exit_d_dir = exit.d_u.dot(dir);
        let qd1_dir = dir[primary.qd1];

        let entry_eta_u_dir = entry.eta_uv.dot(dir);
        let exit_eta_u_dir = exit.eta_uv.dot(dir);
        let exit_chi_u_dir = exit.chi_uv.dot(dir);
        let exit_d_u_dir = exit.d_uv.dot(dir);

        let chi = exit.chi;
        let chi_inv = 1.0 / chi;
        let chi_inv2 = chi_inv * chi_inv;
        let chi_inv3 = chi_inv2 * chi_inv;
        let d_val = exit.d;
        let d_inv = 1.0 / d_val;
        let d_inv2 = d_inv * d_inv;
        let d_inv3 = d_inv2 * d_inv;

        let mut out = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let mut val = 0.0;

                // Entry probit
                val += entry_u3 * entry_eta_dir * entry.eta_u[u] * entry.eta_u[v];
                val += entry_u2
                    * (entry_eta_u_dir[u] * entry.eta_u[v] + entry.eta_u[u] * entry_eta_u_dir[v]);
                val += entry_u2 * entry_eta_dir * entry.eta_uv[[u, v]];
                val += entry_u1 * entry_ext.eta_uv_dir[[u, v]];

                // Exit probit survival
                val += exit_u3 * exit_eta_dir * exit.eta_u[u] * exit.eta_u[v];
                val += exit_u2
                    * (exit_eta_u_dir[u] * exit.eta_u[v] + exit.eta_u[u] * exit_eta_u_dir[v]);
                val += exit_u2 * exit_eta_dir * exit.eta_uv[[u, v]];
                val += exit_u1 * exit_ext.eta_uv_dir[[u, v]];

                // Event density
                val += wi
                    * di
                    * (exit_eta_u_dir[u] * exit.eta_u[v]
                        + exit.eta_u[u] * exit_eta_u_dir[v]
                        + exit_eta_dir * exit.eta_uv[[u, v]]
                        + exit.eta * exit_ext.eta_uv_dir[[u, v]]);

                // Event chi
                let chi_uv_over_chi_dir = (exit_ext.chi_uv_dir[[u, v]] * chi
                    - exit.chi_uv[[u, v]] * exit_chi_dir)
                    * chi_inv2;
                let chi_u_chi_v_over_chi2_dir = (exit_chi_u_dir[u] * exit.chi_u[v]
                    + exit.chi_u[u] * exit_chi_u_dir[v])
                    * chi_inv2
                    - 2.0 * exit.chi_u[u] * exit.chi_u[v] * exit_chi_dir * chi_inv3;
                val -= wi * di * (chi_uv_over_chi_dir - chi_u_chi_v_over_chi2_dir);

                // Event D
                let d_uv_over_d_dir =
                    (exit_ext.d_uv_dir[[u, v]] * d_val - exit.d_uv[[u, v]] * exit_d_dir) * d_inv2;
                let d_u_d_v_over_d2_dir =
                    (exit_d_u_dir[u] * exit.d_u[v] + exit.d_u[u] * exit_d_u_dir[v]) * d_inv2
                        - 2.0 * exit.d_u[u] * exit.d_u[v] * exit_d_dir * d_inv3;
                val += wi * di * (d_uv_over_d_dir - d_u_d_v_over_d2_dir);

                // qd1 term
                if u == primary.qd1 && v == primary.qd1 {
                    val += wi * di * (-2.0 / (qd1 * qd1 * qd1)) * qd1_dir;
                }

                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    /// Fourth-order directional contraction for the flexible survival path.
    ///
    /// The mixed second-directional timepoint transport is carried exactly
    /// through the implicit intercept solve, the observed-point eta/chi jets,
    /// and the cellwise density-normalization integrand.
    fn row_flex_primary_fourth_contracted_exact(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.ensure_scalar_flex_exact_score_geometry("row_flex_primary_fourth_contracted_exact")?;
        let primary = flex_primary_slices(self);
        let p = primary.total;
        if dir_u.len() != p || dir_v.len() != p {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival fourth contracted: dir lengths ({},{}) != {p}",
                    dir_u.len(),
                    dir_v.len(),
                ),
            }
            .into());
        }
        if dir_u.iter().all(|v| v.abs() == 0.0) || dir_v.iter().all(|v| v.abs() == 0.0) {
            return Ok(Array2::<f64>::zeros((p, p)));
        }

        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
        let q0 = q_geom.q0;
        let q1 = q_geom.q1;
        let qd1 = q_geom.qd1;
        let g = block_states[2].eta[row];
        let beta_h = self.flex_score_beta(block_states)?;
        let beta_w = self.flex_link_beta(block_states)?;

        if survival_derivative_guard_violated(qd1, self.derivative_guard) {
            return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                reason: format!(
                    "survival fourth contracted monotonicity violated at row {row}: qd1={qd1:.3e}"
                ),
            }
            .into());
        }

        let (a0, d0) = self.solve_row_survival_intercept_with_slot(
            q0,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Entry)),
        )?;
        let (a1, d1) = self.solve_row_survival_intercept_with_slot(
            q1,
            g,
            beta_h,
            beta_w,
            Some((row, SurvivalInterceptSlotKind::Exit)),
        )?;

        let entry_base = self.compute_survival_timepoint_exact(
            row, &primary, q0, primary.q0, a0, g, d0, beta_h, beta_w, false,
        )?;
        let exit_base = self.compute_survival_timepoint_exact(
            row, &primary, q1, primary.q1, a1, g, d1, beta_h, beta_w, true,
        )?;

        if !exit_base.chi.is_finite() || exit_base.chi <= 0.0 {
            return Err(SurvivalMarginalSlopeError::NumericalFailure {
                reason: format!(
                    "survival fourth contracted row {row}: non-positive chi1={:.3e}",
                    exit_base.chi,
                ),
            }
            .into());
        }

        let entry_ext_u = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_u, false,
        )?;
        let entry_ext_v = self.compute_survival_timepoint_directional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_v, false,
        )?;
        let exit_ext_u = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_u, true,
        )?;
        let exit_ext_v = self.compute_survival_timepoint_directional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_v, true,
        )?;

        // Bidirectional extensions D_{d1} D_{d2} (η_uv, χ_uv, D_uv) via exact
        // IFT second-order recursion through the cell kernel.
        let entry_bi = self.compute_survival_timepoint_bidirectional_exact(
            row, &primary, q0, primary.q0, a0, g, beta_h, beta_w, dir_u, dir_v,
        )?;
        let exit_bi = self.compute_survival_timepoint_bidirectional_exact(
            row, &primary, q1, primary.q1, a1, g, beta_h, beta_w, dir_u, dir_v,
        )?;

        let ordered_uv = self.compute_survival_fourth_contracted_ordered(
            row,
            &primary,
            qd1,
            &entry_base,
            &exit_base,
            &entry_ext_u,
            &entry_ext_v,
            &exit_ext_u,
            &exit_ext_v,
            &entry_bi,
            &exit_bi,
            dir_u,
            dir_v,
        )?;
        let ordered_vu = self.compute_survival_fourth_contracted_ordered(
            row,
            &primary,
            qd1,
            &entry_base,
            &exit_base,
            &entry_ext_v,
            &entry_ext_u,
            &exit_ext_v,
            &exit_ext_u,
            &entry_bi,
            &exit_bi,
            dir_v,
            dir_u,
        )?;

        let mut out = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                out[[i, j]] = 0.5 * (ordered_uv[[i, j]] + ordered_vu[[i, j]]);
            }
        }
        Ok(out)
    }

    /// Compute the ordered fourth contracted D_{dir2}(D_{dir1}(H[a,b])).
    fn compute_survival_fourth_contracted_ordered(
        &self,
        row: usize,
        primary: &FlexPrimarySlices,
        qd1: f64,
        entry_base: &SurvivalFlexTimepointExact,
        exit_base: &SurvivalFlexTimepointExact,
        entry_ext1: &SurvivalFlexTimepointDirectionalExact,
        entry_ext2: &SurvivalFlexTimepointDirectionalExact,
        exit_ext1: &SurvivalFlexTimepointDirectionalExact,
        exit_ext2: &SurvivalFlexTimepointDirectionalExact,
        entry_bi: &SurvivalFlexTimepointBiDirectionalExact,
        exit_bi: &SurvivalFlexTimepointBiDirectionalExact,
        dir1: &Array1<f64>,
        dir2: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let p = primary.total;
        let wi = self.weights[row];
        let di = self.event[row];

        let (entry_k1, entry_k2, entry_k3, entry_k4) =
            signed_probit_neglog_derivatives_up_to_fourth(-entry_base.eta, -wi)?;
        let (exit_k1, exit_k2, exit_k3, exit_k4) =
            signed_probit_neglog_derivatives_up_to_fourth(-exit_base.eta, wi * (1.0 - di))?;

        let entry_u1 = -entry_k1;
        let entry_u2 = entry_k2;
        let entry_u3 = -entry_k3;
        let exit_u1 = -exit_k1;
        let exit_u2 = exit_k2;
        let exit_u3 = -exit_k3;

        let entry_eta_d1 = entry_base.eta_u.dot(dir1);
        let entry_eta_d2 = entry_base.eta_u.dot(dir2);
        let exit_eta_d1 = exit_base.eta_u.dot(dir1);
        let exit_eta_d2 = exit_base.eta_u.dot(dir2);
        let exit_chi_d1 = exit_base.chi_u.dot(dir1);
        let exit_chi_d2 = exit_base.chi_u.dot(dir2);
        let exit_d_d1 = exit_base.d_u.dot(dir1);
        let exit_d_d2 = exit_base.d_u.dot(dir2);
        let qd1_d1 = dir1[primary.qd1];
        let qd1_d2 = dir2[primary.qd1];

        let entry_eta_u_d1 = entry_base.eta_uv.dot(dir1);
        let entry_eta_u_d2 = entry_base.eta_uv.dot(dir2);
        let exit_eta_u_d1 = exit_base.eta_uv.dot(dir1);
        let exit_eta_u_d2 = exit_base.eta_uv.dot(dir2);
        let exit_chi_u_d1 = exit_base.chi_uv.dot(dir1);
        let exit_chi_u_d2 = exit_base.chi_uv.dot(dir2);
        let exit_d_u_d2 = exit_base.d_uv.dot(dir2);

        let entry_eta_d12 = entry_eta_u_d2.dot(dir1);
        let exit_eta_d12 = exit_eta_u_d2.dot(dir1);
        let exit_chi_d12 = exit_chi_u_d2.dot(dir1);
        let exit_d_d12 = exit_d_u_d2.dot(dir1);

        let entry_eta_u_d12: Array1<f64> = (0..p)
            .map(|u| entry_ext2.eta_uv_dir.row(u).dot(dir1))
            .collect::<Vec<_>>()
            .into();
        let exit_eta_u_d12: Array1<f64> = (0..p)
            .map(|u| exit_ext2.eta_uv_dir.row(u).dot(dir1))
            .collect::<Vec<_>>()
            .into();
        let exit_chi_u_d12: Array1<f64> = (0..p)
            .map(|u| exit_ext2.chi_uv_dir.row(u).dot(dir1))
            .collect::<Vec<_>>()
            .into();
        let exit_d_u_d12: Array1<f64> = (0..p)
            .map(|u| exit_ext2.d_uv_dir.row(u).dot(dir1))
            .collect::<Vec<_>>()
            .into();

        // Mixed second-directional D_{d1} D_{d2} η_uv etc. computed by the
        // caller and passed via entry_bi / exit_bi (BiDirectionalExact).
        let entry_eta_uv_d12 = &entry_bi.eta_uv_uv;
        let exit_eta_uv_d12 = &exit_bi.eta_uv_uv;

        let chi = exit_base.chi;
        let chi_inv = 1.0 / chi;
        let chi_inv2 = chi_inv * chi_inv;
        let chi_inv3 = chi_inv2 * chi_inv;
        let chi_inv4 = chi_inv3 * chi_inv;
        let d_val = exit_base.d;
        let d_inv = 1.0 / d_val;
        let d_inv2 = d_inv * d_inv;
        let d_inv3 = d_inv2 * d_inv;
        let d_inv4 = d_inv3 * d_inv;

        let mut out = Array2::<f64>::zeros((p, p));
        for u in 0..p {
            for v in u..p {
                let mut val = 0.0;

                // Entry probit
                let eu = &entry_base.eta_u;
                let euv = &entry_base.eta_uv;

                let a_term = eu[u] * eu[v] * entry_eta_d1;
                let a_term_d2 = entry_eta_u_d2[u] * eu[v] * entry_eta_d1
                    + eu[u] * entry_eta_u_d2[v] * entry_eta_d1
                    + eu[u] * eu[v] * entry_eta_d12;
                let b_term = entry_ext1.eta_uv_dir[[u, v]];
                let b_term_d2 = entry_eta_uv_d12[[u, v]];
                let c_term = entry_eta_u_d1[u] * eu[v]
                    + eu[u] * entry_eta_u_d1[v]
                    + entry_eta_d1 * euv[[u, v]];
                let c_term_d2 = entry_eta_u_d12[u] * eu[v]
                    + entry_eta_u_d1[u] * entry_eta_u_d2[v]
                    + entry_eta_u_d2[u] * entry_eta_u_d1[v]
                    + eu[u] * entry_eta_u_d12[v]
                    + entry_eta_d12 * euv[[u, v]]
                    + entry_eta_d1 * entry_ext2.eta_uv_dir[[u, v]];

                val += entry_k4 * entry_eta_d2 * a_term
                    + entry_u3 * a_term_d2
                    + entry_u3 * entry_eta_d2 * c_term
                    + entry_u2 * c_term_d2
                    + entry_u2 * entry_eta_d2 * b_term
                    + entry_u1 * b_term_d2;

                // Exit probit
                let xu = &exit_base.eta_u;
                let xuv = &exit_base.eta_uv;

                let xa = xu[u] * xu[v] * exit_eta_d1;
                let xa_d2 = exit_eta_u_d2[u] * xu[v] * exit_eta_d1
                    + xu[u] * exit_eta_u_d2[v] * exit_eta_d1
                    + xu[u] * xu[v] * exit_eta_d12;
                let xb = exit_ext1.eta_uv_dir[[u, v]];
                let xb_d2 = exit_eta_uv_d12[[u, v]];
                let xc =
                    exit_eta_u_d1[u] * xu[v] + xu[u] * exit_eta_u_d1[v] + exit_eta_d1 * xuv[[u, v]];
                let xc_d2 = exit_eta_u_d12[u] * xu[v]
                    + exit_eta_u_d1[u] * exit_eta_u_d2[v]
                    + exit_eta_u_d2[u] * exit_eta_u_d1[v]
                    + xu[u] * exit_eta_u_d12[v]
                    + exit_eta_d12 * xuv[[u, v]]
                    + exit_eta_d1 * exit_ext2.eta_uv_dir[[u, v]];

                val += exit_k4 * exit_eta_d2 * xa
                    + exit_u3 * xa_d2
                    + exit_u3 * exit_eta_d2 * xc
                    + exit_u2 * xc_d2
                    + exit_u2 * exit_eta_d2 * xb
                    + exit_u1 * xb_d2;

                // Event density
                val += wi
                    * di
                    * (exit_eta_u_d12[u] * xu[v]
                        + exit_eta_u_d1[u] * exit_eta_u_d2[v]
                        + exit_eta_u_d2[u] * exit_eta_u_d1[v]
                        + xu[u] * exit_eta_u_d12[v]
                        + exit_eta_d12 * xuv[[u, v]]
                        + exit_eta_d1 * exit_ext2.eta_uv_dir[[u, v]]
                        + exit_eta_d2 * exit_ext1.eta_uv_dir[[u, v]]
                        + exit_base.eta * exit_eta_uv_d12[[u, v]]);

                // Event chi
                let chi_uv_val = exit_base.chi_uv[[u, v]];
                let chi_u_val = exit_base.chi_u[u];
                let chi_v_val = exit_base.chi_u[v];
                let chi_uv_d1 = exit_ext1.chi_uv_dir[[u, v]];
                let chi_uv_d2 = exit_ext2.chi_uv_dir[[u, v]];
                let chi_u_d1 = exit_chi_u_d1[u];
                let chi_v_d1 = exit_chi_u_d1[v];
                let chi_u_d2 = exit_chi_u_d2[u];
                let chi_v_d2 = exit_chi_u_d2[v];
                let chi_u_d12v = exit_chi_u_d12[u];
                let chi_v_d12v = exit_chi_u_d12[v];

                let chi_uv_d12_val = exit_bi.chi_uv_uv[[u, v]];
                let d2_r_chi = chi_uv_d12_val * chi_inv
                    - chi_uv_d1 * exit_chi_d2 * chi_inv2
                    - chi_uv_d2 * exit_chi_d1 * chi_inv2
                    - chi_uv_val * exit_chi_d12 * chi_inv2
                    + 2.0 * chi_uv_val * exit_chi_d1 * exit_chi_d2 * chi_inv3;

                let d2_s_chi = (chi_u_d12v * chi_v_val
                    + chi_u_d1 * chi_v_d2
                    + chi_u_d2 * chi_v_d1
                    + chi_u_val * chi_v_d12v)
                    * chi_inv2
                    - 2.0 * (chi_u_d1 * chi_v_val + chi_u_val * chi_v_d1) * exit_chi_d2 * chi_inv3
                    - 2.0 * (chi_u_d2 * chi_v_val + chi_u_val * chi_v_d2) * exit_chi_d1 * chi_inv3
                    - 2.0 * chi_u_val * chi_v_val * exit_chi_d12 * chi_inv3
                    + 6.0 * chi_u_val * chi_v_val * exit_chi_d1 * exit_chi_d2 * chi_inv4;
                val -= wi * di * (d2_r_chi - d2_s_chi);

                // Event D
                let d_uv_val = exit_base.d_uv[[u, v]];
                let d_u_val = exit_base.d_u[u];
                let d_v_val = exit_base.d_u[v];
                let d_uv_d1 = exit_ext1.d_uv_dir[[u, v]];
                let d_uv_d2 = exit_ext2.d_uv_dir[[u, v]];
                let d_u_d1 = exit_ext1.d_u_dir[u];
                let d_v_d1 = exit_ext1.d_u_dir[v];
                let d_u_d2 = exit_ext2.d_u_dir[u];
                let d_v_d2 = exit_ext2.d_u_dir[v];
                let d_u_d12v = exit_d_u_d12[u];
                let d_v_d12v = exit_d_u_d12[v];

                let d_uv_d12_val = exit_bi.d_uv_uv[[u, v]];
                let d2_r_d = d_uv_d12_val * d_inv
                    - d_uv_d1 * exit_d_d2 * d_inv2
                    - d_uv_d2 * exit_d_d1 * d_inv2
                    - d_uv_val * exit_d_d12 * d_inv2
                    + 2.0 * d_uv_val * exit_d_d1 * exit_d_d2 * d_inv3;

                let d2_s_d =
                    (d_u_d12v * d_v_val + d_u_d1 * d_v_d2 + d_u_d2 * d_v_d1 + d_u_val * d_v_d12v)
                        * d_inv2
                        - 2.0 * (d_u_d1 * d_v_val + d_u_val * d_v_d1) * exit_d_d2 * d_inv3
                        - 2.0 * (d_u_d2 * d_v_val + d_u_val * d_v_d2) * exit_d_d1 * d_inv3
                        - 2.0 * d_u_val * d_v_val * exit_d_d12 * d_inv3
                        + 6.0 * d_u_val * d_v_val * exit_d_d1 * exit_d_d2 * d_inv4;
                val += wi * di * (d2_r_d - d2_s_d);

                // qd1 term
                if u == primary.qd1 && v == primary.qd1 {
                    val += wi * di * (6.0 / (qd1 * qd1 * qd1 * qd1)) * qd1_d1 * qd1_d2;
                }

                out[[u, v]] = val;
                out[[v, u]] = val;
            }
        }
        Ok(out)
    }

    fn row_primary_third_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_third_contracted_exact(row, block_states, dir)
        } else {
            self.row_primary_third_contracted(row, block_states, dir.view())
        }
    }

    fn row_primary_fourth_contracted(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: ArrayView1<'_, f64>,
        dir_v: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        // Batched path delegating to the shared k=6 jet helper.
        let r = self.row_primary_fourth_contracted_batched(row, block_states, dir_u, dir_v)?;
        let mut out = Array2::<f64>::zeros((N_PRIMARY, N_PRIMARY));
        for a in 0..N_PRIMARY {
            for b in 0..N_PRIMARY {
                out[[a, b]] = r[a][b];
            }
        }
        Ok(out)
    }

    fn row_primary_fourth_contracted_general(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        dir_u: &Array1<f64>,
        dir_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if self.effective_flex_active(block_states)? {
            self.row_flex_primary_fourth_contracted_exact(row, block_states, dir_u, dir_v)
        } else {
            self.row_primary_fourth_contracted(row, block_states, dir_u.view(), dir_v.view())
        }
    }

    // ── Pullback through design matrices ──────────────────────────────

    /// Accumulate the pullback of a primary-space Hessian into coefficient-space.
    ///
    /// Writes directly into `target` subslices via sparse-aware row primitives —
    /// no dense row buffers or temporary blocks are allocated.
    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &BlockSlices,
        primary_hessian: &Array2<f64>,
    ) {
        let h = primary_hessian;
        let time_designs = [
            &self.design_entry,
            &self.design_exit,
            &self.design_derivative_exit,
        ];

        // time-time block: Σ_{a,b} H[a,b] * time_a_row ⊗ time_b_row
        for a in 0..3 {
            for b in 0..3 {
                let alpha = h[[a, b]];
                if alpha == 0.0 {
                    continue;
                }
                time_designs[a]
                    .row_outer_into_view(
                        row,
                        time_designs[b],
                        alpha,
                        target.slice_mut(s![slices.time.clone(), slices.time.clone()]),
                    )
                    .expect("time block row_outer_into dimension mismatch");
            }
        }

        // marginal-marginal block: (H[0,0]+H[0,1]+H[1,0]+H[1,1]) * m_row ⊗ m_row
        self.marginal_design
            .syr_row_into_view(
                row,
                h[[0, 0]] + h[[0, 1]] + h[[1, 0]] + h[[1, 1]],
                target.slice_mut(s![slices.marginal.clone(), slices.marginal.clone()]),
            )
            .expect("marginal syr_row_into dimension mismatch");

        // logslope-logslope block: H[3,3] * g_row ⊗ g_row
        self.logslope_design
            .syr_row_into_view(
                row,
                h[[3, 3]],
                target.slice_mut(s![slices.logslope.clone(), slices.logslope.clone()]),
            )
            .expect("logslope syr_row_into dimension mismatch");

        // marginal-logslope block: (H[0,3]+H[1,3]) * m_row ⊗ g_row  (+ transpose)
        {
            let alpha_mg = h[[0, 3]] + h[[1, 3]];
            if alpha_mg != 0.0 {
                self.marginal_design
                    .row_outer_into_view(
                        row,
                        &self.logslope_design,
                        alpha_mg,
                        target.slice_mut(s![slices.marginal.clone(), slices.logslope.clone()]),
                    )
                    .expect("marginal-logslope row_outer_into dimension mismatch");
                self.logslope_design
                    .row_outer_into_view(
                        row,
                        &self.marginal_design,
                        alpha_mg,
                        target.slice_mut(s![slices.logslope.clone(), slices.marginal.clone()]),
                    )
                    .expect("logslope-marginal row_outer_into dimension mismatch");
            }
        }

        // time-logslope block: H[a,3] * time_a_row ⊗ g_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 3]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.logslope_design,
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.logslope.clone()]),
                )
                .expect("time-logslope row_outer_into dimension mismatch");
            self.logslope_design
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.logslope.clone(), slices.time.clone()]),
                )
                .expect("logslope-time row_outer_into dimension mismatch");
        }

        // time-marginal block: (H[a,0]+H[a,1]) * time_a_row ⊗ m_row  (+ transpose)
        for a in 0..3 {
            let alpha = h[[a, 0]] + h[[a, 1]];
            if alpha == 0.0 {
                continue;
            }
            time_designs[a]
                .row_outer_into_view(
                    row,
                    &self.marginal_design,
                    alpha,
                    target.slice_mut(s![slices.time.clone(), slices.marginal.clone()]),
                )
                .expect("time-marginal row_outer_into dimension mismatch");
            self.marginal_design
                .row_outer_into_view(
                    row,
                    time_designs[a],
                    alpha,
                    target.slice_mut(s![slices.marginal.clone(), slices.time.clone()]),
                )
                .expect("marginal-time row_outer_into dimension mismatch");
        }
    }

    /// Block-diagonal-only pullback: writes only the principal time-time,
    /// marginal-marginal, and logslope-logslope rowwise contributions into
    /// per-block targets. Used by `evaluate()` to populate per-block working
    /// sets without ever materializing the cross blocks.
    fn add_pullback_block_diagonals(
        &self,
        row: usize,
        primary_hessian: &Array2<f64>,
        time_target: &mut Array2<f64>,
        marginal_target: &mut Array2<f64>,
        logslope_target: &mut Array2<f64>,
    ) {
        let h = primary_hessian;
        let time_designs = [
            &self.design_entry,
            &self.design_exit,
            &self.design_derivative_exit,
        ];
        for a in 0..3 {
            for b in 0..3 {
                let alpha = h[[a, b]];
                if alpha == 0.0 {
                    continue;
                }
                time_designs[a]
                    .row_outer_into_view(row, time_designs[b], alpha, time_target.view_mut())
                    .expect("time block row_outer_into dimension mismatch");
            }
        }
        let alpha_mm = h[[0, 0]] + h[[0, 1]] + h[[1, 0]] + h[[1, 1]];
        self.marginal_design
            .syr_row_into_view(row, alpha_mm, marginal_target.view_mut())
            .expect("marginal syr_row_into dimension mismatch");
        self.logslope_design
            .syr_row_into_view(row, h[[3, 3]], logslope_target.view_mut())
            .expect("logslope syr_row_into dimension mismatch");
    }

    fn row_primary_direction_from_flat_dynamic(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let flex_primary = self
            .effective_flex_active(block_states)?
            .then(|| flex_primary_slices(self));
        let mut out = Array1::<f64>::zeros(flex_primary.as_ref().map_or(N_PRIMARY, |p| p.total));
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let q_geom = self.row_dynamic_q_geometry(row, block_states)?;

        let q0_dir = q_geom.dq0_time.dot(&d_time) + q_geom.dq0_marginal.dot(&d_marginal);
        let q1_dir = q_geom.dq1_time.dot(&d_time) + q_geom.dq1_marginal.dot(&d_marginal);
        let qd1_dir = q_geom.dqd1_time.dot(&d_time) + q_geom.dqd1_marginal.dot(&d_marginal);
        let g_dir = self
            .logslope_design
            .dot_row_view(row, d_beta_flat.slice(s![slices.logslope.clone()]));

        if let Some(primary) = flex_primary.as_ref() {
            out[primary.q0] = q0_dir;
            out[primary.q1] = q1_dir;
            out[primary.qd1] = qd1_dir;
            out[primary.g] = g_dir;
            for (primary_range, block_range) in flex_identity_block_pairs(primary, slices) {
                out.slice_mut(s![primary_range])
                    .assign(&d_beta_flat.slice(s![block_range]));
            }
        } else {
            out[0] = q0_dir;
            out[1] = q1_dir;
            out[2] = qd1_dir;
            out[3] = g_dir;
        }
        Ok(out)
    }

    // ── Psi (spatial length-scale) derivatives ────────────────────────

    // ── Psi terms (first and second order) ────────────────────────────
    //
    // All three psi methods (first-order, second-order, directional derivative)
    // use block-local accumulation via BlockHessianAccumulator. Per-row work is
    // O(max(p_block²)) instead of O(p²), eliminating the dense p×p bottleneck
    // that breaks multi-axis Duchon / per-axis length scaling.

    /// Resolve psi block info: (block_idx, local_idx, p_block, label).
    fn psi_block_info(
        &self,
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<(usize, usize, usize, &'static str)>, String> {
        let Some((block_idx, local_idx)) = psi_derivative_location(derivative_blocks, psi_index)
        else {
            return Ok(None);
        };
        match block_idx {
            1 => Ok(Some((
                block_idx,
                local_idx,
                self.marginal_design.ncols(),
                "SurvivalMarginalSlope marginal",
            ))),
            2 => Ok(Some((
                block_idx,
                local_idx,
                self.logslope_design.ncols(),
                "SurvivalMarginalSlope logslope",
            ))),
            _ => Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: format!(
                    "survival marginal-slope psi: only baseline/slope spatial blocks are supported, got block {block_idx}"
                ),
            }
            .into()),
        }
    }

    /// Accumulate block-local score from a primary-space vector (replaces
    /// pullback_primary_vector + score += for score accumulation).
    fn accumulate_score_blockwise(
        &self,
        row: usize,
        primary: &Array1<f64>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
        score_g: &mut Array1<f64>,
    ) -> Result<(), String> {
        {
            let mut st = score_t.view_mut();
            self.design_entry
                .axpy_row_into(row, primary[0], &mut st)
                .expect("time entry axpy dim mismatch");
            self.design_exit
                .axpy_row_into(row, primary[1], &mut st)
                .expect("time exit axpy dim mismatch");
            self.design_derivative_exit
                .axpy_row_into(row, primary[2], &mut st)
                .expect("time deriv axpy dim mismatch");
        }
        self.marginal_design.axpy_row_into(
            row,
            primary[0] + primary[1],
            &mut score_m.view_mut(),
        )?;
        self.logslope_design
            .axpy_row_into(row, primary[3], &mut score_g.view_mut())?;
        Ok(())
    }

    fn accumulate_score_identity_blocks(
        &self,
        primary_layout: Option<&FlexPrimarySlices>,
        primary: &Array1<f64>,
        score_h: Option<&mut Array1<f64>>,
        score_w: Option<&mut Array1<f64>>,
    ) {
        if let Some(primary_layout) = primary_layout {
            if let (Some(range), Some(score_h)) = (primary_layout.h.as_ref(), score_h) {
                *score_h = &*score_h + &primary.slice(s![range.clone()]);
            }
            if let (Some(range), Some(score_w)) = (primary_layout.w.as_ref(), score_w) {
                *score_w = &*score_w + &primary.slice(s![range.clone()]);
            }
        }
    }

    /// Score pullback using actual Jacobians from q-geometry (timewiggle-correct).
    fn accumulate_score_with_q_geometry(
        &self,
        row: usize,
        qg: &SurvivalMarginalSlopeDynamicRow,
        primary: &Array1<f64>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
        score_g: &mut Array1<f64>,
    ) -> Result<(), String> {
        let jt = [&qg.dq0_time, &qg.dq1_time, &qg.dqd1_time];
        let jm = [&qg.dq0_marginal, &qg.dq1_marginal, &qg.dqd1_marginal];
        for q in 0..3 {
            if primary[q] != 0.0 {
                score_t.scaled_add(primary[q], jt[q]);
            }
        }
        for q in 0..3 {
            if primary[q] != 0.0 {
                score_m.scaled_add(primary[q], jm[q]);
            }
        }
        self.logslope_design
            .axpy_row_into(row, primary[3], &mut score_g.view_mut())?;
        Ok(())
    }

    /// U_{iB}^α contribution to score (eq 46, term 1) for timewiggle + marginal ψ.
    fn accumulate_score_timewiggle_psi_u(
        lift: &TimewiggleMarginalPsiRowLift,
        f_pi: &Array1<f64>,
        score_t: &mut Array1<f64>,
        score_m: &mut Array1<f64>,
    ) {
        let ut = [&lift.u_q0_time, &lift.u_q1_time, &lift.u_qd1_time];
        let um = [
            &lift.u_q0_marginal,
            &lift.u_q1_marginal,
            &lift.u_qd1_marginal,
        ];
        for q in 0..3 {
            if f_pi[q] != 0.0 {
                score_t.scaled_add(f_pi[q], ut[q]);
            }
        }
        for q in 0..3 {
            if f_pi[q] != 0.0 {
                score_m.scaled_add(f_pi[q], um[q]);
            }
        }
    }

    /// Compute u_i^{ψε} = D_ψ D_β π_i · d_beta for hessian directional derivative.
    /// With timewiggle + marginal ψ, includes T''(h) cross-terms.
    fn timewiggle_psi_action(
        &self,
        row: usize,
        block_states: &[ParameterBlockState],
        slices: &BlockSlices,
        primary_layout: Option<&FlexPrimarySlices>,
        psi_row: &Array1<f64>,
        beta_psi: &Array1<f64>,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let beta_time = &block_states[0].beta;
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);
        let ec = self
            .design_entry
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action design_entry: {e}"))?;
        let xc = self
            .design_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action design_exit: {e}"))?;
        let dc = self
            .design_derivative_exit
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action design_derivative_exit: {e}"))?;
        let x_e = ec.row(0).slice(s![..p_base]).to_owned();
        let x_x = xc.row(0).slice(s![..p_base]).to_owned();
        let x_d = dc.row(0).slice(s![..p_base]).to_owned();
        let mc = self
            .marginal_design
            .try_row_chunk(row..row + 1)
            .map_err(|e| format!("timewiggle_psi_action marginal_design: {e}"))?;
        let x_m = mc.row(0).to_owned();
        let bm = block_states[1].eta[row];
        let h0 = x_e.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
        let h1 = x_x.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
        let d_raw = x_d.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
        let eg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
            .ok_or_else(|| "missing entry timewiggle for psi action".to_string())?;
        let xg = self
            .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
            .ok_or_else(|| "missing exit timewiggle for psi action".to_string())?;
        let mu = psi_row.dot(beta_psi);
        let dt = d_beta_flat.slice(s![slices.time.clone()]);
        let dm = d_beta_flat.slice(s![slices.marginal.clone()]);
        let dh0 = x_e.dot(&dt.slice(s![..p_base])) + x_m.dot(&dm);
        let dh1 = x_x.dot(&dt.slice(s![..p_base])) + x_m.dot(&dm);
        let dd_raw = x_d.dot(&dt.slice(s![..p_base]));
        let dmu = psi_row.dot(&dm);
        let mut out = Array1::zeros(primary_layout.map_or(N_PRIMARY, |primary| primary.total));
        let q0_idx = primary_layout.map_or(0, |primary| primary.q0);
        let q1_idx = primary_layout.map_or(1, |primary| primary.q1);
        let qd1_idx = primary_layout.map_or(2, |primary| primary.qd1);
        out[q0_idx] = eg.d2q_dq02[0] * mu * dh0 + eg.dq_dq0[0] * dmu;
        out[q1_idx] = xg.d2q_dq02[0] * mu * dh1 + xg.dq_dq0[0] * dmu;
        out[qd1_idx] =
            xg.d3q_dq03[0] * d_raw * mu * dh1 + xg.d2q_dq02[0] * (dd_raw * mu + d_raw * dmu);
        Ok(out)
    }

    fn psi_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `psi_terms_inner`. When
    /// `options.outer_score_subsample` is `None`, iterates all rows and is
    /// bit-for-bit equivalent to the legacy implementation. When `Some`, only
    /// the sampled rows contribute and every row-summed component (objective
    /// scalar, per-block score vectors, Hessian operator blocks) is accumulated
    /// with the row's Horvitz-Thompson inverse-inclusion weight.
    pub(crate) fn psi_terms_inner_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        cache: Option<&EvalCache>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx)?
        } else {
            spatial_block_primary_loading(block_idx)?
        };
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let timewiggle_active = self.flex_timewiggle_active();
        let timewiggle_psi = timewiggle_active && block_idx == 1;

        // NOTE: old dense timewiggle early return removed; now handled by
        // block path with lifted Jacobians below.
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());

        // Build the psi design map once; rowwise loop does direct row_vector(row)
        // calls via the PsiDesignMap API.
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            self.n,
            p_psi,
            0..self.n,
            psi_label,
            &policy,
        )?;

        // Parallel accumulation: each worker gets its own block-local accumulators.
        type Acc = (
            f64,                     // objective_psi
            Array1<f64>,             // score_t
            Array1<f64>,             // score_m
            Array1<f64>,             // score_g
            Array1<f64>,             // score_h
            Array1<f64>,             // score_w
            BlockHessianAccumulator, // Hessian blocks
        );
        let make_acc = || -> Acc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                Array1::zeros(p_h),
                Array1::zeros(p_w),
                BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
            )
        };

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let (objective_psi, score_t, score_m, score_g, score_h, score_w, acc) =
            chunked_row_reduction(
                row_iter.as_slice(),
                make_acc,
                |row, a| -> Result<(), String> {
                    let psi_row = psi_map
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                    let q_geom = if timewiggle_active {
                        Some(self.row_dynamic_q_geometry(row, block_states)?)
                    } else {
                        None
                    };

                    let psi_lift = if timewiggle_psi {
                        Some(self.timewiggle_marginal_psi_row_lift(
                            row,
                            block_states,
                            flex_primary.as_ref(),
                            &psi_row,
                            beta_psi,
                        )?)
                    } else {
                        None
                    };

                    let dir = if let Some(lift) = psi_lift.as_ref() {
                        lift.dir.clone()
                    } else if let Some(primary) = flex_primary.as_ref() {
                        primary_direction_from_psi_row_flex(primary, block_idx, &psi_row, beta_psi)
                    } else {
                        primary_direction_from_psi_row(block_idx, &psi_row, beta_psi)
                    };

                    let q_geom_lazy;
                    let (mut f_pi, mut f_pipi) = if let Some(primary) = flex_primary.as_ref() {
                        let q_ref = match q_geom.as_ref() {
                            Some(q) => q,
                            None => {
                                q_geom_lazy = self.row_dynamic_q_geometry(row, block_states)?;
                                &q_geom_lazy
                            }
                        };
                        let (_, g, h) = self.compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            q_ref,
                            primary,
                        )?;
                        (g, h)
                    } else if let Some(c) = cache {
                        let (g, h) = self.row_primary_gradient_hessian(row, c);
                        (g.clone(), h.clone())
                    } else {
                        let (_, g, h) =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (g, h)
                    };

                    // Third contracted derivative T_i[u^α].
                    let w = row_weights[row];
                    if w != 1.0 {
                        f_pi.mapv_inplace(|v| v * w);
                        f_pipi.mapv_inplace(|v| v * w);
                    }

                    let mut third =
                        self.row_primary_third_contracted_general(row, block_states, &dir)?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }

                    // ── Eq (45): objective_psi += f_i^T u_i^α ──
                    a.0 += f_pi.dot(&dir);

                    let s1 = f_pi.dot(&loading);
                    match block_idx {
                        1 => a.2.scaled_add(s1, &psi_row),
                        _ => a.3.scaled_add(s1, &psi_row),
                    }
                    let pb = f_pipi.dot(&dir);
                    if let Some(lift) = psi_lift.as_ref() {
                        Self::accumulate_score_timewiggle_psi_u(lift, &f_pi, &mut a.1, &mut a.2);
                    }
                    if let Some(q) = q_geom.as_ref() {
                        self.accumulate_score_with_q_geometry(
                            row, q, &pb, &mut a.1, &mut a.2, &mut a.3,
                        )?;
                    } else {
                        self.accumulate_score_blockwise(row, &pb, &mut a.1, &mut a.2, &mut a.3)?;
                    }
                    self.accumulate_score_identity_blocks(
                        flex_primary.as_ref(),
                        &pb,
                        Some(&mut a.4),
                        Some(&mut a.5),
                    );

                    let right_primary = f_pipi.dot(&loading);
                    if let Some(q) = q_geom.as_ref() {
                        a.6.add_rank1_psi_cross_with_q_geometry(
                            self,
                            row,
                            q,
                            block_idx,
                            &psi_row,
                            &right_primary,
                        )?;
                    } else {
                        a.6.add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary)?;
                    }
                    if let Some(q) = q_geom.as_ref() {
                        let zero_grad = Array1::zeros(third.nrows());
                        a.6.add_pullback_with_q_geometry(self, row, q, &zero_grad, &third)?;
                    } else {
                        a.6.add_pullback(self, row, &third)?;
                    }
                    if let Some(lift) = psi_lift.as_ref() {
                        let q = q_geom.as_ref().unwrap();
                        a.6.add_timewiggle_psi_u_cross(self, row, q, lift, &f_pipi)?;
                        a.6.add_second_pullback_weighted(q, &pb);
                        a.6.add_timewiggle_psi_kappa_alpha(self, lift, &f_pi);
                    }

                    Ok(())
                },
                |total, chunk| {
                    total.0 += chunk.0;
                    total.1 += &chunk.1;
                    total.2 += &chunk.2;
                    total.3 += &chunk.3;
                    total.4 += &chunk.4;
                    total.5 += &chunk.5;
                    total.6.add(&chunk.6);
                },
            )?;

        // Assemble score into flat vector
        let mut score_psi = Array1::zeros(slices.total);
        score_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(std::sync::Arc::new(acc.into_operator(slices))),
        }))
    }

    /// Batched per-axis variant of `psi_terms_inner_with_options`. Performs a
    /// single rayon row pass that fetches the per-row primary gradient and
    /// Hessian once and folds them into every axis in lock-step, instead of
    /// the K serial row passes the per-axis path would issue.
    ///
    /// Returns `Ok(None)` when any branch the fast path does not cover is
    /// active (effective flex, timewiggle, or a sigma-aux index in the
    /// request list); callers fall back to the per-axis path. The simple
    /// spatial-only path (the biobank survival marginal-slope workload) is
    /// the case this fast path targets.
    pub(crate) fn psi_terms_inner_batched_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_indices: &[usize],
        cache: Option<&EvalCache>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            return Ok(None);
        }
        for &psi_index in psi_indices {
            if self.is_sigma_aux_index(derivative_blocks, psi_index) {
                return Ok(None);
            }
        }

        let k = psi_indices.len();
        if k == 0 {
            return Ok(Some(Vec::new()));
        }
        let slices = block_slices(self, block_states);

        // Per-axis context: psi map, primary-space loading, beta. Resolved
        // once outside the row pass so the row hot loop sees only borrows.
        struct AxisCtx<'a> {
            block_idx: usize,
            psi_map: crate::families::custom_family::PsiDesignMap,
            loading: Array1<f64>,
            beta_psi: &'a Array1<f64>,
        }
        let policy = crate::resource::ResourcePolicy::default_library();
        let mut axes: Vec<AxisCtx<'_>> = Vec::with_capacity(k);
        for &psi_index in psi_indices {
            let Some((block_idx, local_idx, p_psi, psi_label)) =
                self.psi_block_info(derivative_blocks, psi_index)?
            else {
                // psi_block_info returning None means caller passed an index
                // that does not resolve to a known block; defer to per-axis
                // so behaviour matches `first_order_terms(psi_index)`.
                return Ok(None);
            };
            let deriv = &derivative_blocks[block_idx][local_idx];
            let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
                deriv,
                self.n,
                p_psi,
                0..self.n,
                psi_label,
                &policy,
            )?;
            let loading = spatial_block_primary_loading(block_idx)?;
            let beta_psi: &Array1<f64> = match block_idx {
                1 => &block_states[1].beta,
                _ => &block_states[2].beta,
            };
            axes.push(AxisCtx {
                block_idx,
                psi_map,
                loading,
                beta_psi,
            });
        }

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());

        struct BatchedPsiAxisAcc {
            objective_psi: f64,
            score_t: Array1<f64>,
            score_m: Array1<f64>,
            score_g: Array1<f64>,
            score_h: Array1<f64>,
            score_w: Array1<f64>,
            hessian: BlockHessianAccumulator,
        }
        let make_accs = || -> Vec<BatchedPsiAxisAcc> {
            (0..k)
                .map(|_| BatchedPsiAxisAcc {
                    objective_psi: 0.0,
                    score_t: Array1::zeros(p_t),
                    score_m: Array1::zeros(p_m),
                    score_g: Array1::zeros(p_g),
                    score_h: Array1::zeros(p_h),
                    score_w: Array1::zeros(p_w),
                    hessian: BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
                })
                .collect()
        };

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);

        let folded = chunked_row_reduction(
            row_iter.as_slice(),
            make_accs,
            |row, accs: &mut Vec<BatchedPsiAxisAcc>| -> Result<(), String> {
                let w = row_weights[row];

                // Fetch (f_pi, f_pipi) UNWEIGHTED once per row. Mutating
                // them in place between axes would double-weight every
                // axis after the first; instead each axis applies `w`
                // inline below.
                let (f_pi, f_pipi) = if let Some(c) = cache {
                    let (g, h) = self.row_primary_gradient_hessian(row, c);
                    (g.clone(), h.clone())
                } else {
                    let (_, g, h) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    (g, h)
                };

                for (axis_idx, axis) in axes.iter().enumerate() {
                    let psi_row = axis
                        .psi_map
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map (batched): {e}"))?;
                    let dir =
                        primary_direction_from_psi_row(axis.block_idx, &psi_row, axis.beta_psi);
                    let mut third =
                        self.row_primary_third_contracted_general(row, block_states, &dir)?;
                    if w != 1.0 {
                        third.mapv_inplace(|v| v * w);
                    }

                    let acc = &mut accs[axis_idx];

                    // objective_psi += w * (f_pi · dir)
                    acc.objective_psi += w * f_pi.dot(&dir);

                    // score_psi += w * (f_pi · loading) * psi_row, routed to
                    // the marginal or logslope block depending on axis.
                    let s1 = w * f_pi.dot(&axis.loading);
                    match axis.block_idx {
                        1 => acc.score_m.scaled_add(s1, &psi_row),
                        _ => acc.score_g.scaled_add(s1, &psi_row),
                    }

                    let mut pb = f_pipi.dot(&dir);
                    if w != 1.0 {
                        pb.mapv_inplace(|v| v * w);
                    }
                    self.accumulate_score_blockwise(
                        row,
                        &pb,
                        &mut acc.score_t,
                        &mut acc.score_m,
                        &mut acc.score_g,
                    )?;
                    self.accumulate_score_identity_blocks(
                        None,
                        &pb,
                        Some(&mut acc.score_h),
                        Some(&mut acc.score_w),
                    );

                    let mut right_primary = f_pipi.dot(&axis.loading);
                    if w != 1.0 {
                        right_primary.mapv_inplace(|v| v * w);
                    }
                    acc.hessian.add_rank1_psi_cross(
                        self,
                        row,
                        axis.block_idx,
                        &psi_row,
                        &right_primary,
                    )?;
                    acc.hessian.add_pullback(self, row, &third)?;
                }
                Ok(())
            },
            |total: &mut Vec<BatchedPsiAxisAcc>, chunk: Vec<BatchedPsiAxisAcc>| {
                for (t, c) in total.iter_mut().zip(chunk.into_iter()) {
                    t.objective_psi += c.objective_psi;
                    t.score_t += &c.score_t;
                    t.score_m += &c.score_m;
                    t.score_g += &c.score_g;
                    t.score_h += &c.score_h;
                    t.score_w += &c.score_w;
                    t.hessian.add(&c.hessian);
                }
            },
        )?;

        let mut out: Vec<ExactNewtonJointPsiTerms> = Vec::with_capacity(k);
        for acc in folded.into_iter() {
            let mut score_psi = Array1::zeros(slices.total);
            score_psi
                .slice_mut(s![slices.time.clone()])
                .assign(&acc.score_t);
            score_psi
                .slice_mut(s![slices.marginal.clone()])
                .assign(&acc.score_m);
            score_psi
                .slice_mut(s![slices.logslope.clone()])
                .assign(&acc.score_g);
            if let Some(range) = slices.score_warp.as_ref() {
                score_psi.slice_mut(s![range.clone()]).assign(&acc.score_h);
            }
            if let Some(range) = slices.link_dev.as_ref() {
                score_psi.slice_mut(s![range.clone()]).assign(&acc.score_w);
            }
            out.push(ExactNewtonJointPsiTerms {
                objective_psi: acc.objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(
                    acc.hessian.into_operator(slices.clone()),
                )),
            });
        }
        Ok(Some(out))
    }

    fn psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        self.psi_terms_inner(block_states, derivative_blocks, psi_index, None)
    }

    fn psi_second_order_terms_inner(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: Option<&EvalCache>,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner_with_options(
            block_states,
            derivative_blocks,
            psi_i,
            psi_j,
            cache,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `psi_second_order_terms_inner`. See
    /// `psi_terms_inner_with_options` for the row-iter / weighting contract.
    pub(crate) fn psi_second_order_terms_inner_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
        cache: Option<&EvalCache>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx_i, local_idx_i, p_psi_i, label_i)) =
            self.psi_block_info(derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some((block_idx_j, local_idx_j, p_psi_j, label_j)) =
            self.psi_block_info(derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let deriv_i = &derivative_blocks[block_idx_i][local_idx_i];
        let deriv_j = &derivative_blocks[block_idx_j][local_idx_j];
        let loading_i = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx_i)?
        } else {
            spatial_block_primary_loading(block_idx_i)?
        };
        let loading_j = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx_j)?
        } else {
            spatial_block_primary_loading(block_idx_j)?
        };
        let beta_i = match block_idx_i {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let beta_j = match block_idx_j {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };

        let timewiggle_active = self.flex_timewiggle_active();
        let timewiggle_psi_i = timewiggle_active && block_idx_i == 1;
        let timewiggle_psi_j = timewiggle_active && block_idx_j == 1;

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let same_block = block_idx_i == block_idx_j;

        // Build psi design maps once outside the row loop; rowwise calls use
        // the direct row_vector(row) API.
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map_i = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_i,
            self.n,
            p_psi_i,
            0..self.n,
            label_i,
            &policy,
        )?;
        let psi_map_j = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv_j,
            self.n,
            p_psi_j,
            0..self.n,
            label_j,
            &policy,
        )?;
        let psi_map_ij = if same_block {
            Some(
                crate::families::custom_family::resolve_custom_family_x_psi_psi_map(
                    deriv_i,
                    deriv_j,
                    local_idx_j,
                    self.n,
                    p_psi_i,
                    0..self.n,
                    label_i,
                    &policy,
                )?,
            )
        } else {
            None
        };

        struct JointPsiSecondOrderAcc {
            objective_psi_psi: f64,
            score_t: Array1<f64>,
            score_m: Array1<f64>,
            score_g: Array1<f64>,
            score_h: Array1<f64>,
            score_w: Array1<f64>,
            hessian: BlockHessianAccumulator,
        }
        let make_acc = || -> JointPsiSecondOrderAcc {
            JointPsiSecondOrderAcc {
                objective_psi_psi: 0.0,
                score_t: Array1::zeros(p_t),
                score_m: Array1::zeros(p_m),
                score_g: Array1::zeros(p_g),
                score_h: Array1::zeros(p_h),
                score_w: Array1::zeros(p_w),
                hessian: BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
            }
        };

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let JointPsiSecondOrderAcc {
            objective_psi_psi,
            score_t,
            score_m,
            score_g,
            score_h,
            score_w,
            hessian,
        } = chunked_row_reduction(
            row_iter.as_slice(),
            make_acc,
            |row, a| -> Result<(), String> {
                // Compute psi design rows once; derive directions from them.
                let psi_row_i = psi_map_i
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                let psi_row_j = psi_map_j
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                let q_geom = if timewiggle_active {
                    Some(self.row_dynamic_q_geometry(row, block_states)?)
                } else {
                    None
                };

                let psi_lift_i = if timewiggle_psi_i {
                    Some(self.timewiggle_marginal_psi_row_lift(
                        row,
                        block_states,
                        flex_primary.as_ref(),
                        &psi_row_i,
                        beta_i,
                    )?)
                } else {
                    None
                };
                let psi_lift_j = if timewiggle_psi_j {
                    Some(self.timewiggle_marginal_psi_row_lift(
                        row,
                        block_states,
                        flex_primary.as_ref(),
                        &psi_row_j,
                        beta_j,
                    )?)
                } else {
                    None
                };

                let dir_i = if let Some(lift) = psi_lift_i.as_ref() {
                    lift.dir.clone()
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_direction_from_psi_row_flex(primary, block_idx_i, &psi_row_i, beta_i)
                } else {
                    primary_direction_from_psi_row(block_idx_i, &psi_row_i, beta_i)
                };
                let dir_j = if let Some(lift) = psi_lift_j.as_ref() {
                    lift.dir.clone()
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_direction_from_psi_row_flex(primary, block_idx_j, &psi_row_j, beta_j)
                } else {
                    primary_direction_from_psi_row(block_idx_j, &psi_row_j, beta_j)
                };

                let (psi_row_ij, dir_ij) = if same_block {
                    let r = psi_map_ij
                        .as_ref()
                        .expect("psi_map_ij built when same_block")
                        .row_vector(row)
                        .map_err(|e| format!("survival rowwise psi map: {e}"))?;
                    let d = if let Some(primary) = flex_primary.as_ref() {
                        primary_second_direction_from_psi_row_flex(primary, block_idx_i, &r, beta_i)
                    } else {
                        primary_second_direction_from_psi_row(block_idx_i, &r, beta_i)
                    };
                    (Some(r), d)
                } else {
                    (
                        None,
                        Array1::<f64>::zeros(
                            flex_primary
                                .as_ref()
                                .map_or(N_PRIMARY, |primary| primary.total),
                        ),
                    )
                };
                let has_ij = psi_row_ij
                    .as_ref()
                    .is_some_and(|r| r.iter().any(|v| v.abs() > 0.0));

                let q_geom_lazy;
                let (mut f_pi, mut f_pipi) = if let Some(primary) = flex_primary.as_ref() {
                    let q_ref = match q_geom.as_ref() {
                        Some(q) => q,
                        None => {
                            q_geom_lazy = self.row_dynamic_q_geometry(row, block_states)?;
                            &q_geom_lazy
                        }
                    };
                    let (_, g, h) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_ref,
                        primary,
                    )?;
                    (g, h)
                } else if let Some(c) = cache {
                    let (g, h) = self.row_primary_gradient_hessian(row, c);
                    (g.clone(), h.clone())
                } else {
                    let (_, g, h) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    (g, h)
                };
                let w = row_weights[row];
                if w != 1.0 {
                    f_pi.mapv_inplace(|v| v * w);
                    f_pipi.mapv_inplace(|v| v * w);
                }
                let mut third_i =
                    self.row_primary_third_contracted_general(row, block_states, &dir_i)?;
                let mut third_j =
                    self.row_primary_third_contracted_general(row, block_states, &dir_j)?;
                let mut fourth =
                    self.row_primary_fourth_contracted_general(row, block_states, &dir_i, &dir_j)?;
                if w != 1.0 {
                    third_i.mapv_inplace(|v| v * w);
                    third_j.mapv_inplace(|v| v * w);
                    fourth.mapv_inplace(|v| v * w);
                }

                a.objective_psi_psi += dir_i.dot(&f_pipi.dot(&dir_j)) + f_pi.dot(&dir_ij);

                // Score
                if has_ij {
                    let s_ij = f_pi.dot(&loading_i);
                    let psi_ij = psi_row_ij.as_ref().unwrap();
                    match block_idx_i {
                        1 => a.score_m.scaled_add(s_ij, psi_ij),
                        _ => a.score_g.scaled_add(s_ij, psi_ij),
                    }
                }
                let s_i = loading_i.dot(&f_pipi.dot(&dir_j));
                match block_idx_i {
                    1 => a.score_m.scaled_add(s_i, &psi_row_i),
                    _ => a.score_g.scaled_add(s_i, &psi_row_i),
                }
                let s_j = loading_j.dot(&f_pipi.dot(&dir_i));
                match block_idx_j {
                    1 => a.score_m.scaled_add(s_j, &psi_row_j),
                    _ => a.score_g.scaled_add(s_j, &psi_row_j),
                }
                let pb1 = f_pipi.dot(&dir_ij);
                if let Some(q) = q_geom.as_ref() {
                    self.accumulate_score_with_q_geometry(
                        row,
                        q,
                        &pb1,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                } else {
                    self.accumulate_score_blockwise(
                        row,
                        &pb1,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                }
                self.accumulate_score_identity_blocks(
                    flex_primary.as_ref(),
                    &pb1,
                    Some(&mut a.score_h),
                    Some(&mut a.score_w),
                );
                let pb2 = third_i.dot(&dir_j);
                if let Some(q) = q_geom.as_ref() {
                    self.accumulate_score_with_q_geometry(
                        row,
                        q,
                        &pb2,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                } else {
                    self.accumulate_score_blockwise(
                        row,
                        &pb2,
                        &mut a.score_t,
                        &mut a.score_m,
                        &mut a.score_g,
                    )?;
                }
                self.accumulate_score_identity_blocks(
                    flex_primary.as_ref(),
                    &pb2,
                    Some(&mut a.score_h),
                    Some(&mut a.score_w),
                );

                // Hessian
                if has_ij {
                    let rp_ij = f_pipi.dot(&loading_i);
                    if let Some(q) = q_geom.as_ref() {
                        a.hessian.add_rank1_psi_cross_with_q_geometry(
                            self,
                            row,
                            q,
                            block_idx_i,
                            psi_row_ij.as_ref().unwrap(),
                            &rp_ij,
                        )?;
                    } else {
                        a.hessian.add_rank1_psi_cross(
                            self,
                            row,
                            block_idx_i,
                            psi_row_ij.as_ref().unwrap(),
                            &rp_ij,
                        )?;
                    }
                }
                let scalar_ij = loading_i.dot(&f_pipi.dot(&loading_j));
                a.hessian.add_psi_psi_outer(
                    block_idx_i,
                    &psi_row_i,
                    block_idx_j,
                    &psi_row_j,
                    scalar_ij,
                );
                let rp_i = third_j.t().dot(&loading_i);
                if let Some(q) = q_geom.as_ref() {
                    a.hessian.add_rank1_psi_cross_with_q_geometry(
                        self,
                        row,
                        q,
                        block_idx_i,
                        &psi_row_i,
                        &rp_i,
                    )?;
                } else {
                    a.hessian
                        .add_rank1_psi_cross(self, row, block_idx_i, &psi_row_i, &rp_i)?;
                }
                let rp_j = third_i.t().dot(&loading_j);
                if let Some(q) = q_geom.as_ref() {
                    a.hessian.add_rank1_psi_cross_with_q_geometry(
                        self,
                        row,
                        q,
                        block_idx_j,
                        &psi_row_j,
                        &rp_j,
                    )?;
                } else {
                    a.hessian
                        .add_rank1_psi_cross(self, row, block_idx_j, &psi_row_j, &rp_j)?;
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(fourth.nrows());
                    a.hessian
                        .add_pullback_with_q_geometry(self, row, q, &zero_grad, &fourth)?;
                } else {
                    a.hessian.add_pullback(self, row, &fourth)?;
                }
                let mut third_ij =
                    self.row_primary_third_contracted_general(row, block_states, &dir_ij)?;
                if w != 1.0 {
                    third_ij.mapv_inplace(|v| v * w);
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(third_ij.nrows());
                    a.hessian
                        .add_pullback_with_q_geometry(self, row, q, &zero_grad, &third_ij)?;
                } else {
                    a.hessian.add_pullback(self, row, &third_ij)?;
                }

                // Timewiggle psi corrections for ψ_i (terms 1,2,4,5 of eq 47)
                if let Some(lift_i) = psi_lift_i.as_ref() {
                    let q = q_geom.as_ref().unwrap();
                    // U_i^α cross terms with third_j Hessian
                    a.hessian
                        .add_timewiggle_psi_u_cross(self, row, q, lift_i, &third_j)?;
                    // Second pullback weighted by T_j[dir_j] applied to dir_i
                    let hu_i = f_pipi.dot(&dir_i);
                    a.hessian.add_second_pullback_weighted(q, &hu_i);
                    // K^{BC,α_i} weighted by gradient
                    a.hessian
                        .add_timewiggle_psi_kappa_alpha(self, lift_i, &f_pi);
                }
                // Timewiggle psi corrections for ψ_j
                if let Some(lift_j) = psi_lift_j.as_ref() {
                    let q = q_geom.as_ref().unwrap();
                    a.hessian
                        .add_timewiggle_psi_u_cross(self, row, q, lift_j, &third_i)?;
                    let hu_j = f_pipi.dot(&dir_j);
                    a.hessian.add_second_pullback_weighted(q, &hu_j);
                    if psi_lift_i.is_none() {
                        // Only add gradient-weighted K^α for j if we didn't already
                        // add it for i (when both are marginal, it's already covered)
                        a.hessian
                            .add_timewiggle_psi_kappa_alpha(self, lift_j, &f_pi);
                    }
                }

                Ok(())
            },
            |total, chunk| {
                total.objective_psi_psi += chunk.objective_psi_psi;
                total.score_t += &chunk.score_t;
                total.score_m += &chunk.score_m;
                total.score_g += &chunk.score_g;
                total.score_h += &chunk.score_h;
                total.score_w += &chunk.score_w;
                total.hessian.add(&chunk.hessian);
            },
        )?;

        let mut score_psi_psi = Array1::zeros(slices.total);
        score_psi_psi
            .slice_mut(s![slices.time.clone()])
            .assign(&score_t);
        score_psi_psi
            .slice_mut(s![slices.marginal.clone()])
            .assign(&score_m);
        score_psi_psi
            .slice_mut(s![slices.logslope.clone()])
            .assign(&score_g);
        if let Some(range) = slices.score_warp.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_h);
        }
        if let Some(range) = slices.link_dev.as_ref() {
            score_psi_psi.slice_mut(s![range.clone()]).assign(&score_w);
        }

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(Box::new(hessian.into_operator(slices))),
        }))
    }

    fn psi_second_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        self.psi_second_order_terms_inner(block_states, derivative_blocks, psi_i, psi_j, None)
    }

    fn psi_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.psi_hessian_directional_derivative_with_options(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
    }

    /// Outer-aware variant of `psi_hessian_directional_derivative` that
    /// returns the dense block Hessian directional derivative. When
    /// `options.outer_score_subsample` is `Some`, only the masked rows are
    /// visited and the accumulator uses per-row Horvitz-Thompson
    /// inverse-inclusion weights before being densified.
    pub(crate) fn psi_hessian_directional_derivative_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Array2<f64>>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx)?
        } else {
            spatial_block_primary_loading(block_idx)?
        };
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let d_beta_block = match block_idx {
            1 => d_beta_flat.slice(s![slices.marginal.clone()]),
            _ => d_beta_flat.slice(s![slices.logslope.clone()]),
        };

        let timewiggle_active = self.flex_timewiggle_active();
        let timewiggle_psi = timewiggle_active && block_idx == 1;

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            self.n,
            p_psi,
            0..self.n,
            psi_label,
            &policy,
        )?;

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
            |row, acc| -> Result<(), String> {
                let psi_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                let q_geom = if timewiggle_active {
                    Some(self.row_dynamic_q_geometry(row, block_states)?)
                } else {
                    None
                };

                let psi_lift = if timewiggle_psi {
                    Some(self.timewiggle_marginal_psi_row_lift(
                        row,
                        block_states,
                        flex_primary.as_ref(),
                        &psi_row,
                        beta_psi,
                    )?)
                } else {
                    None
                };

                let psi_dir = if let Some(lift) = psi_lift.as_ref() {
                    lift.dir.clone()
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_direction_from_psi_row_flex(primary, block_idx, &psi_row, beta_psi)
                } else {
                    primary_direction_from_psi_row(block_idx, &psi_row, beta_psi)
                };
                let psi_action = if psi_lift.is_some() {
                    self.timewiggle_psi_action(
                        row,
                        block_states,
                        &slices,
                        flex_primary.as_ref(),
                        &psi_row,
                        beta_psi,
                        d_beta_flat,
                    )?
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_psi_action_from_psi_row_flex(primary, block_idx, &psi_row, d_beta_block)
                } else {
                    primary_psi_action_from_psi_row(block_idx, &psi_row, d_beta_block)
                };
                let row_dir = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let q_geom_lazy;
                let mut h_pi = if let Some(primary) = flex_primary.as_ref() {
                    let q_ref = match q_geom.as_ref() {
                        Some(q) => q,
                        None => {
                            q_geom_lazy = self.row_dynamic_q_geometry(row, block_states)?;
                            &q_geom_lazy
                        }
                    };
                    self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_ref,
                        primary,
                    )?
                    .2
                } else {
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?
                        .2
                };
                let w = row_weights[row];
                if w != 1.0 {
                    h_pi.mapv_inplace(|v| v * w);
                }
                let mut third_beta =
                    self.row_primary_third_contracted_general(row, block_states, &row_dir)?;
                let mut fourth = self.row_primary_fourth_contracted_general(
                    row,
                    block_states,
                    &row_dir,
                    &psi_dir,
                )?;
                if w != 1.0 {
                    third_beta.mapv_inplace(|v| v * w);
                    fourth.mapv_inplace(|v| v * w);
                }

                let right_primary = third_beta.t().dot(&loading);
                if let Some(q) = q_geom.as_ref() {
                    acc.add_rank1_psi_cross_with_q_geometry(
                        self,
                        row,
                        q,
                        block_idx,
                        &psi_row,
                        &right_primary,
                    )?;
                } else {
                    acc.add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary)?;
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(fourth.nrows());
                    acc.add_pullback_with_q_geometry(self, row, q, &zero_grad, &fourth)?;
                } else {
                    acc.add_pullback(self, row, &fourth)?;
                }
                let mut third_action =
                    self.row_primary_third_contracted_general(row, block_states, &psi_action)?;
                if w != 1.0 {
                    third_action.mapv_inplace(|v| v * w);
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(third_action.nrows());
                    acc.add_pullback_with_q_geometry(self, row, q, &zero_grad, &third_action)?;
                } else {
                    acc.add_pullback(self, row, &third_action)?;
                }
                // Timewiggle psi corrections
                if let Some(lift) = psi_lift.as_ref() {
                    let q = q_geom.as_ref().unwrap();
                    // U^α cross with third_beta (D_β H term)
                    acc.add_timewiggle_psi_u_cross(self, row, q, lift, &third_beta)?;
                    let second_pullback_weight = third_beta.dot(&psi_dir) + h_pi.dot(&psi_action);
                    acc.add_second_pullback_weighted(q, &second_pullback_weight);
                    let kappa_weight = h_pi.dot(&row_dir);
                    acc.add_timewiggle_psi_kappa_alpha(self, lift, &kappa_weight);
                }
                Ok(())
            },
            |total, chunk| {
                total.add(&chunk);
            },
        )?;

        Ok(Some(acc.to_dense(&slices)))
    }

    /// Outer-aware operator builder for the per-ψ Hessian directional
    /// derivative. When `options.outer_score_subsample` is `Some`, only the
    /// sampled rows are visited and the accumulator uses per-row
    /// Horvitz-Thompson inverse-inclusion weights before being wrapped in the
    /// `HyperOperator`.
    pub(crate) fn psi_hessian_directional_derivative_operator_with_options(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let flex_primary = flex_active.then(|| flex_primary_slices(self));
        let slices = block_slices(self, block_states);
        let Some((block_idx, local_idx, p_psi, psi_label)) =
            self.psi_block_info(derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let deriv = &derivative_blocks[block_idx][local_idx];
        let loading = if let Some(primary) = flex_primary.as_ref() {
            spatial_block_primary_loading_flex(primary, block_idx)?
        } else {
            spatial_block_primary_loading(block_idx)?
        };
        let beta_psi = match block_idx {
            1 => &block_states[1].beta,
            _ => &block_states[2].beta,
        };
        let d_beta_block = match block_idx {
            1 => d_beta_flat.slice(s![slices.marginal.clone()]),
            _ => d_beta_flat.slice(s![slices.logslope.clone()]),
        };

        let timewiggle_active = self.flex_timewiggle_active();
        let timewiggle_psi = timewiggle_active && block_idx == 1;

        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());

        // Build the psi design map once; rowwise calls use direct row_vector(row).
        let policy = crate::resource::ResourcePolicy::default_library();
        let psi_map = crate::families::custom_family::resolve_custom_family_x_psi_map(
            deriv,
            self.n,
            p_psi,
            0..self.n,
            psi_label,
            &policy,
        )?;

        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        // Process fixed row chunks in parallel and merge local cross-block
        // accumulators in row-chunk order for deterministic timewiggle assembly.
        let acc = chunked_row_reduction(
            row_iter.as_slice(),
            || BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
            |row, acc| -> Result<(), String> {
                let psi_row = psi_map
                    .row_vector(row)
                    .map_err(|e| format!("survival rowwise psi map: {e}"))?;

                let q_geom = if timewiggle_active {
                    Some(self.row_dynamic_q_geometry(row, block_states)?)
                } else {
                    None
                };

                let psi_lift = if timewiggle_psi {
                    Some(self.timewiggle_marginal_psi_row_lift(
                        row,
                        block_states,
                        flex_primary.as_ref(),
                        &psi_row,
                        beta_psi,
                    )?)
                } else {
                    None
                };

                let psi_dir = if let Some(lift) = psi_lift.as_ref() {
                    lift.dir.clone()
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_direction_from_psi_row_flex(primary, block_idx, &psi_row, beta_psi)
                } else {
                    primary_direction_from_psi_row(block_idx, &psi_row, beta_psi)
                };
                let psi_action = if psi_lift.is_some() {
                    self.timewiggle_psi_action(
                        row,
                        block_states,
                        &slices,
                        flex_primary.as_ref(),
                        &psi_row,
                        beta_psi,
                        d_beta_flat,
                    )?
                } else if let Some(primary) = flex_primary.as_ref() {
                    primary_psi_action_from_psi_row_flex(primary, block_idx, &psi_row, d_beta_block)
                } else {
                    primary_psi_action_from_psi_row(block_idx, &psi_row, d_beta_block)
                };
                let row_dir = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let q_geom_lazy;
                let mut h_pi = if let Some(primary) = flex_primary.as_ref() {
                    let q_ref = match q_geom.as_ref() {
                        Some(q) => q,
                        None => {
                            q_geom_lazy = self.row_dynamic_q_geometry(row, block_states)?;
                            &q_geom_lazy
                        }
                    };
                    self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_ref,
                        primary,
                    )?
                    .2
                } else {
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?
                        .2
                };
                let w = row_weights[row];
                if w != 1.0 {
                    h_pi.mapv_inplace(|v| v * w);
                }
                let mut third_beta =
                    self.row_primary_third_contracted_general(row, block_states, &row_dir)?;
                let mut fourth = self.row_primary_fourth_contracted_general(
                    row,
                    block_states,
                    &row_dir,
                    &psi_dir,
                )?;
                if w != 1.0 {
                    third_beta.mapv_inplace(|v| v * w);
                    fourth.mapv_inplace(|v| v * w);
                }

                let right_primary = third_beta.t().dot(&loading);
                if let Some(q) = q_geom.as_ref() {
                    acc.add_rank1_psi_cross_with_q_geometry(
                        self,
                        row,
                        q,
                        block_idx,
                        &psi_row,
                        &right_primary,
                    )?;
                } else {
                    acc.add_rank1_psi_cross(self, row, block_idx, &psi_row, &right_primary)?;
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(fourth.nrows());
                    acc.add_pullback_with_q_geometry(self, row, q, &zero_grad, &fourth)?;
                } else {
                    acc.add_pullback(self, row, &fourth)?;
                }
                let mut third_action =
                    self.row_primary_third_contracted_general(row, block_states, &psi_action)?;
                if w != 1.0 {
                    third_action.mapv_inplace(|v| v * w);
                }
                if let Some(q) = q_geom.as_ref() {
                    let zero_grad = Array1::zeros(third_action.nrows());
                    acc.add_pullback_with_q_geometry(self, row, q, &zero_grad, &third_action)?;
                } else {
                    acc.add_pullback(self, row, &third_action)?;
                }
                if let Some(lift) = psi_lift.as_ref() {
                    let q = q_geom.as_ref().unwrap();
                    acc.add_timewiggle_psi_u_cross(self, row, q, lift, &third_beta)?;
                    let second_pullback_weight = third_beta.dot(&psi_dir) + h_pi.dot(&psi_action);
                    acc.add_second_pullback_weighted(q, &second_pullback_weight);
                    let kappa_weight = h_pi.dot(&row_dir);
                    acc.add_timewiggle_psi_kappa_alpha(self, lift, &kappa_weight);
                }
                Ok(())
            },
            |total, chunk| {
                total.add(&chunk);
            },
        )?;

        Ok(Some(
            Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>
        ))
    }

    fn exact_newton_joint_hessian_operator(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(Arc<dyn HyperOperator>, Array1<f64>), String> {
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let make_acc = || BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w);

        // See `evaluate_exact_newton_joint_dynamic_q_dense` for rationale: the
        // per-thread accumulator embeds a `SurvivalMarginalSlopeDynamicRow`
        // workspace so the nine Array2/Array1 buffers are reused across all
        // rows handled by one rayon worker.
        let make_acc_ws = || {
            (
                make_acc(),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };
        let acc = if self.effective_flex_active(block_states)? {
            let primary = flex_primary_slices(self);
            (0..self.n)
                .into_par_iter()
                .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                    let (state, q_geom) = &mut acc;
                    self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                    let (_, g, h) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_geom,
                        &primary,
                    )?;
                    state.add_pullback_with_q_geometry(self, row, q_geom, &g, &h)?;
                    Ok(acc)
                })
                .try_reduce(make_acc_ws, |mut a, b| -> Result<_, String> {
                    a.0.add(&b.0);
                    Ok(a)
                })?
                .0
        } else {
            (0..self.n)
                .into_par_iter()
                .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                    let (state, q_geom) = &mut acc;
                    self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                    let (_, g, h) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    state.add_pullback_with_q_geometry(self, row, q_geom, &g, &h)?;
                    Ok(acc)
                })
                .try_reduce(make_acc_ws, |mut a, b| -> Result<_, String> {
                    a.0.add(&b.0);
                    Ok(a)
                })?
                .0
        };

        let diagonal = acc.diagonal(&slices);
        Ok((
            Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>,
            diagonal,
        ))
    }

    /// Outer-aware operator builder for the flex-no-wiggle joint-Hessian
    /// directional derivative. The default-options shim is omitted because
    /// the `SurvivalMarginalSlopeExactNewtonJointHessianWorkspace` always
    /// threads its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is
    /// `Some`, only the sampled rows are visited and the accumulator uses
    /// per-row Horvitz-Thompson inverse-inclusion weights before being wrapped
    /// in the `HyperOperator`.
    pub(crate) fn exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Arc<dyn HyperOperator>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let make_acc_ws = || {
            (
                BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };
        let acc = row_iter
            .into_par_iter()
            .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let h_pi = self
                    .compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_geom,
                        &primary,
                    )?
                    .2;
                let u_d = self.row_primary_direction_from_flat_dynamic(
                    row,
                    block_states,
                    &slices,
                    d_beta_flat,
                )?;
                let mut t_ud =
                    self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                let mut h_ud = h_pi.dot(&u_d);
                let w = row_weights[row];
                if w != 1.0 {
                    h_ud.mapv_inplace(|v| v * w);
                    t_ud.mapv_inplace(|v| v * w);
                }
                state.add_pullback_with_q_geometry(self, row, q_geom, &h_ud, &t_ud)?;
                Ok(acc)
            })
            .try_reduce(make_acc_ws, |mut a, b| -> Result<_, String> {
                a.0.add(&b.0);
                Ok(a)
            })?
            .0;
        Ok(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>)
    }

    /// Outer-aware operator builder for the flex-no-wiggle joint-Hessian
    /// second directional derivative. The default-options shim is omitted
    /// because the `SurvivalMarginalSlopeExactNewtonJointHessianWorkspace`
    /// always threads its own `BlockwiseFitOptions`. When `options.outer_score_subsample` is
    /// `Some`, only the sampled rows are visited and the accumulator uses
    /// per-row Horvitz-Thompson inverse-inclusion weights before being wrapped
    /// in the `HyperOperator`.
    pub(crate) fn exact_newton_joint_hessiansecond_directional_derivative_operator_flex_no_wiggle_with_options(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
        options: &BlockwiseFitOptions,
    ) -> Result<Arc<dyn HyperOperator>, String> {
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let p_h = slices.score_warp.as_ref().map_or(0, |range| range.len());
        let p_w = slices.link_dev.as_ref().map_or(0, |range| range.len());
        let row_iter = outer_row_indices(options, self.n).to_vec();
        let row_weights = outer_row_weights_by_index(options, self.n);
        let make_acc_ws = || {
            (
                BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };
        let acc = row_iter
            .into_par_iter()
            .try_fold(make_acc_ws, |mut acc, row| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let ud =
                    self.row_primary_direction_from_flat_dynamic(row, block_states, &slices, d_u)?;
                let ue =
                    self.row_primary_direction_from_flat_dynamic(row, block_states, &slices, d_v)?;
                let mut q_de =
                    self.row_flex_primary_fourth_contracted_exact(row, block_states, &ud, &ue)?;
                let t_d = self.row_flex_primary_third_contracted_exact(row, block_states, &ud)?;
                let mut gamma = t_d.dot(&ue);
                let w = row_weights[row];
                if w != 1.0 {
                    gamma.mapv_inplace(|v| v * w);
                    q_de.mapv_inplace(|v| v * w);
                }
                state.add_pullback_with_q_geometry(self, row, q_geom, &gamma, &q_de)?;
                Ok(acc)
            })
            .try_reduce(make_acc_ws, |mut a, b| -> Result<_, String> {
                a.0.add(&b.0);
                Ok(a)
            })?
            .0;
        Ok(Arc::new(acc.into_operator(slices)) as Arc<dyn HyperOperator>)
    }
}

// ── Workspace structs ─────────────────────────────────────────────────

struct SurvivalMarginalSlopePsiWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    specs: Vec<ParameterBlockSpec>,
    derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
    cache: Option<EvalCache>,
    /// Outer-only ψ-calculus options. The `outer_score_subsample` field is
    /// the row mask threaded through `sigma_exact_joint_psi_terms_with_options`
    /// and the second-order / Hessian-drift counterparts to make the cached
    /// ψ calculus subsample-aware.
    options: BlockwiseFitOptions,
}

struct SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    joint_hessian_operator: Arc<dyn HyperOperator>,
    joint_hessian_diagonal: Array1<f64>,
    /// Cached per-row primary gradient + Hessian for timewiggle directional
    /// derivative reuse.  Built once during workspace construction so that
    /// repeated directional-derivative calls do not recompute them.
    eval_cache: Option<EvalCache>,
    /// Outer-only joint-Hessian directional-derivative options. The
    /// `outer_score_subsample` field is the row mask threaded through the
    /// `_with_options` directional-derivative helpers so the cached joint
    /// Hessian Hv-action paths can downscale to the stratified subsample at
    /// biobank scale. When `None`, the row iteration is identical to the
    /// legacy full-data path.
    options: BlockwiseFitOptions,
}

impl SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let (joint_hessian_operator, joint_hessian_diagonal) =
            family.exact_newton_joint_hessian_operator(&block_states)?;
        let eval_cache = if family.flex_timewiggle_active() && !family.flex_active() {
            Some(family.build_eval_cache(&block_states)?)
        } else {
            None
        };
        Ok(Self {
            family,
            block_states,
            joint_hessian_operator,
            joint_hessian_diagonal,
            eval_cache,
            options,
        })
    }
}

impl ExactNewtonJointHessianWorkspace for SurvivalMarginalSlopeExactNewtonJointHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        // The operator we already built in `Self::new` carries every block
        // (h_tt, h_mm, h_gg, h_tm, …) of the joint Hessian. Asking the family
        // to re-materialize a dense p×p Hessian via
        // `evaluate_exact_newton_joint_dynamic_q_dense` would re-walk all n
        // rows just to repeat the J^T H J + Σ f K pullback we just finished;
        // at biobank scale that is the same n-row sweep twice per inner
        // joint-Newton cycle. Reuse the operator's `to_dense()` instead — an
        // O(p²) block copy. Numerically identical to the dense path modulo
        // FMA summation order.
        Ok(Some(self.joint_hessian_operator.to_dense()))
    }

    fn hessian_matvec(&self, beta_flat: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_hessian_operator.mul_vec(beta_flat)))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        // Forward to HyperOperator's existing `mul_vec_into`, which writes the
        // matvec result directly into the caller-owned buffer with no
        // intermediate allocation. Used by inner-Newton PCG so each CG iter
        // avoids a fresh Array1<f64> on the survival biobank-scale hot path.
        if v.len() != self.joint_hessian_operator.dim()
            || out.len() != self.joint_hessian_operator.dim()
        {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "hessian_matvec_into: dim mismatch v={} out={} op={}",
                    v.len(),
                    out.len(),
                    self.joint_hessian_operator.dim()
                ),
            }
            .into());
        }
        self.joint_hessian_operator
            .mul_vec_into(v.view(), out.view_mut());
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.joint_hessian_diagonal.clone()))
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        if self.family.effective_flex_active(&self.block_states)?
            && !self.family.flex_timewiggle_active()
        {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
                    &self.block_states,
                    d_beta_flat,
                    &self.options,
                )
                .map(Some);
        }
        if let Some(cache) = self.eval_cache.as_ref() {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
                    &self.block_states,
                    d_beta_flat,
                    cache,
                )
                .map(|matrix| {
                    Some(Arc::new(
                        crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                    ) as Arc<dyn HyperOperator>)
                });
        }
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
            .map(|result| {
                result.map(|matrix| {
                    Arc::new(
                        crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                    ) as Arc<dyn HyperOperator>
                })
            })
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if let Some(cache) = self.eval_cache.as_ref() {
            return self
                .family
                .exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
                    &self.block_states,
                    d_beta_flat,
                    cache,
                )
                .map(Some);
        }
        self.family
            .exact_newton_joint_hessian_directional_derivative(&self.block_states, d_beta_flat)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        if self.family.effective_flex_active(&self.block_states)?
            && !self.family.flex_timewiggle_active()
        {
            return self
                .family
                .exact_newton_joint_hessiansecond_directional_derivative_operator_flex_no_wiggle_with_options(
                    &self.block_states,
                    d_beta_u_flat,
                    d_beta_v_flat,
                    &self.options,
                )
                .map(Some);
        }
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
            .map(|result| {
                result.map(|matrix| {
                    Arc::new(
                        crate::solver::estimate::reml::unified::DenseMatrixHyperOperator { matrix },
                    ) as Arc<dyn HyperOperator>
                })
            })
    }

    fn second_directional_derivative(
        &self,
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &self.block_states,
                d_beta_u_flat,
                d_beta_v_flat,
            )
    }
}

impl SurvivalMarginalSlopePsiWorkspace {
    fn new(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
        specs: Vec<ParameterBlockSpec>,
        derivative_blocks: Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>,
        options: BlockwiseFitOptions,
    ) -> Result<Self, String> {
        let cache = if family.flex_active() {
            None
        } else {
            Some(family.build_eval_cache(&block_states)?)
        };
        Ok(Self {
            family,
            block_states,
            specs,
            derivative_blocks,
            cache,
            options,
        })
    }
}

impl ExactNewtonJointPsiWorkspace for SurvivalMarginalSlopePsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_index)
        {
            return self.family.sigma_exact_joint_psi_terms_with_options(
                &self.block_states,
                &self.specs,
                &self.options,
            );
        }
        self.family.psi_terms_inner_with_options(
            &self.block_states,
            &self.derivative_blocks,
            psi_index,
            self.cache.as_ref(),
            &self.options,
        )
    }

    fn first_order_terms_all(&self) -> Result<Option<Vec<ExactNewtonJointPsiTerms>>, String> {
        let total: usize = self.derivative_blocks.iter().map(Vec::len).sum();
        if total == 0 {
            return Ok(Some(Vec::new()));
        }
        let psi_indices: Vec<usize> = (0..total).collect();
        self.family.psi_terms_inner_batched_with_options(
            &self.block_states,
            &self.derivative_blocks,
            &psi_indices,
            self.cache.as_ref(),
            &self.options,
        )
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_i)
            || self
                .family
                .is_sigma_aux_index(&self.derivative_blocks, psi_j)
        {
            if psi_i == psi_j {
                return self
                    .family
                    .sigma_exact_joint_psisecond_order_terms_with_options(
                        &self.block_states,
                        &self.options,
                    );
            }
            return Ok(None);
        }
        self.family.psi_second_order_terms_inner_with_options(
            &self.block_states,
            &self.derivative_blocks,
            psi_i,
            psi_j,
            self.cache.as_ref(),
            &self.options,
        )
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        if self
            .family
            .is_sigma_aux_index(&self.derivative_blocks, psi_index)
        {
            return self
                .family
                .sigma_exact_joint_psihessian_directional_derivative_with_options(
                    &self.block_states,
                    d_beta_flat,
                    &self.options,
                )
                .map(|result| {
                    result.map(|matrix| {
                        crate::solver::estimate::reml::unified::DriftDerivResult::Dense(matrix)
                    })
                });
        }
        self.family
            .psi_hessian_directional_derivative_operator_with_options(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
                d_beta_flat,
                &self.options,
            )
            .map(|result| {
                result.map(crate::solver::estimate::reml::unified::DriftDerivResult::Operator)
            })
    }
}

// ── RowKernel<4> implementation ───────────────────────────────────────

struct SurvivalMarginalSlopeRowKernel {
    family: SurvivalMarginalSlopeFamily,
    block_states: Vec<ParameterBlockState>,
    slices: BlockSlices,
}

impl SurvivalMarginalSlopeRowKernel {
    fn new(family: SurvivalMarginalSlopeFamily, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = block_slices(&family, &block_states);
        Self {
            family,
            block_states,
            slices,
        }
    }
}

impl RowKernel<4> for SurvivalMarginalSlopeRowKernel {
    fn n_rows(&self) -> usize {
        self.family.n
    }
    fn n_coefficients(&self) -> usize {
        self.slices.total
    }

    fn row_kernel(&self, row: usize) -> Result<(f64, [f64; 4], [[f64; 4]; 4]), String> {
        let beta_time = &self.block_states[0].beta;
        let q0 = self.family.design_entry.dot_row(row, beta_time)
            + self.family.offset_entry[row]
            + self.block_states[1].eta[row];
        let q1 = self.family.design_exit.dot_row(row, beta_time)
            + self.family.offset_exit[row]
            + self.block_states[1].eta[row];
        let qd1 = self.family.design_derivative_exit.dot_row(row, beta_time)
            + self.family.derivative_offset_exit[row];
        self.family.row_primary_closed_form_rigid(
            row,
            q0,
            q1,
            qd1,
            &self.block_states,
            self.family.probit_frailty_scale(),
        )
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; 4] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.slices.time.clone()]);
        let d_marginal = d_beta.slice(s![self.slices.marginal.clone()]);
        let d_logslope = d_beta.slice(s![self.slices.logslope.clone()]);
        [
            self.family.design_entry.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_exit.dot_row_view(row, d_time)
                + self.family.marginal_design.dot_row_view(row, d_marginal),
            self.family.design_derivative_exit.dot_row_view(row, d_time),
            self.family.logslope_design.dot_row_view(row, d_logslope),
        ]
    }

    fn jacobian_action_matrix(&self, factor: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        if factor.nrows() != self.slices.total {
            return None;
        }
        let n_rows = self.family.n;
        let rank = factor.ncols();
        if rank == 0 {
            return Some(Array2::<f64>::zeros((n_rows, 0)));
        }

        let f_time = factor.slice(s![self.slices.time.clone(), ..]);
        let f_marginal = factor.slice(s![self.slices.marginal.clone(), ..]);
        let f_logslope = factor.slice(s![self.slices.logslope.clone(), ..]);

        let jf_entry = survival_axis_jf_via_design(&self.family.design_entry, f_time, n_rows);
        let jf_exit = survival_axis_jf_via_design(&self.family.design_exit, f_time, n_rows);
        let jf_derivative =
            survival_axis_jf_via_design(&self.family.design_derivative_exit, f_time, n_rows);
        let jf_marginal =
            survival_axis_jf_via_design(&self.family.marginal_design, f_marginal, n_rows);
        let jf_logslope =
            survival_axis_jf_via_design(&self.family.logslope_design, f_logslope, n_rows);

        let mut jf = Array2::<f64>::zeros((n_rows, 4 * rank));
        {
            let mut axis0 = jf.slice_mut(s![.., 0..rank]);
            axis0.assign(&jf_entry);
            axis0 += &jf_marginal;
        }
        {
            let mut axis1 = jf.slice_mut(s![.., rank..2 * rank]);
            axis1.assign(&jf_exit);
            axis1 += &jf_marginal;
        }
        jf.slice_mut(s![.., 2 * rank..3 * rank])
            .assign(&jf_derivative);
        jf.slice_mut(s![.., 3 * rank..4 * rank])
            .assign(&jf_logslope);
        Some(jf)
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; 4], out: &mut [f64]) {
        {
            let mut time = ndarray::ArrayViewMut1::from(&mut out[self.slices.time.clone()]);
            self.family
                .design_entry
                .axpy_row_into(row, v[0], &mut time)
                .expect("time entry axpy dim mismatch");
            self.family
                .design_exit
                .axpy_row_into(row, v[1], &mut time)
                .expect("time exit axpy dim mismatch");
            self.family
                .design_derivative_exit
                .axpy_row_into(row, v[2], &mut time)
                .expect("time deriv axpy dim mismatch");
        }
        {
            let mut marginal = ndarray::ArrayViewMut1::from(&mut out[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .axpy_row_into(row, v[0] + v[1], &mut marginal)
                .expect("marginal axpy dim mismatch");
        }
        {
            let mut logslope = ndarray::ArrayViewMut1::from(&mut out[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .axpy_row_into(row, v[3], &mut logslope)
                .expect("logslope axpy dim mismatch");
        }
    }

    fn add_pullback_hessian(&self, row: usize, h: &[[f64; 4]; 4], target: &mut Array2<f64>) {
        let mut h_arr = Array2::<f64>::zeros((4, 4));
        for a in 0..4 {
            for b in 0..4 {
                h_arr[[a, b]] = h[a][b];
            }
        }
        self.family
            .add_pullback_primary_hessian(target, row, &self.slices, &h_arr);
    }

    fn add_diagonal_quadratic(&self, row: usize, h: &[[f64; 4]; 4], diag: &mut [f64]) {
        let designs: [(usize, &DesignMatrix); 3] = [
            (0, &self.family.design_entry),
            (1, &self.family.design_exit),
            (2, &self.family.design_derivative_exit),
        ];
        for &(pi, des) in &designs {
            {
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.squared_axpy_row_into(row, h[pi][pi], &mut td)
                    .expect("time squared_axpy dim mismatch");
            }
            for &(pj, des_j) in &designs {
                if pj <= pi {
                    continue;
                }
                let mut td = ndarray::ArrayViewMut1::from(&mut diag[self.slices.time.clone()]);
                des.crossdiag_axpy_row_into(row, des_j, 2.0 * h[pi][pj], &mut td)
                    .expect("time crossdiag dim mismatch");
            }
        }
        {
            let alpha = h[0][0] + 2.0 * h[0][1] + h[1][1];
            let mut md = ndarray::ArrayViewMut1::from(&mut diag[self.slices.marginal.clone()]);
            self.family
                .marginal_design
                .squared_axpy_row_into(row, alpha, &mut md)
                .expect("marginal squared_axpy dim mismatch");
        }
        {
            let mut gd = ndarray::ArrayViewMut1::from(&mut diag[self.slices.logslope.clone()]);
            self.family
                .logslope_design
                .squared_axpy_row_into(row, h[3][3], &mut gd)
                .expect("logslope squared_axpy dim mismatch");
        }
    }

    fn row_third_contracted(&self, row: usize, dir: &[f64; 4]) -> Result<[[f64; 4]; 4], String> {
        // Batched path: one k=5 MultiDirJet [e_0..e_3, dir] covers all 6
        // off-diagonal (a,b) entries via mask reads; 4 small k=3 jets handle
        // diagonal ∂²_{e_a} ∂_{dir} entries. Replaces the legacy 10 separate
        // calls into row_neglog_directional_refs.
        let dir_view = ndarray::aview1(&dir[..]);
        self.family
            .row_primary_third_contracted_batched(row, &self.block_states, dir_view)
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; 4],
        dir_v: &[f64; 4],
    ) -> Result<[[f64; 4]; 4], String> {
        // Batched path: one k=6 MultiDirJet [e_0..e_3, dir_u, dir_v] covers
        // the 6 off-diagonal (a,b) entries; 4 k=4 jets handle the diagonal
        // ∂²_{e_a} ∂_{dir_u} ∂_{dir_v} entries.
        let u_view = ndarray::aview1(&dir_u[..]);
        let v_view = ndarray::aview1(&dir_v[..]);
        self.family
            .row_primary_fourth_contracted_batched(row, &self.block_states, u_view, v_view)
    }
}

fn survival_axis_jf_via_design(
    design: &DesignMatrix,
    factor_block: ArrayView2<'_, f64>,
    n_rows: usize,
) -> Array2<f64> {
    let rank = factor_block.ncols();
    if rank == 0 {
        return Array2::<f64>::zeros((n_rows, 0));
    }
    let factor = factor_block.as_standard_layout().into_owned();
    match design.as_dense_ref() {
        Some(dense) => fast_ab(dense, &factor),
        None => {
            let mut out = Array2::<f64>::zeros((n_rows, rank));
            for c in 0..rank {
                let result = design.dot(&factor.column(c).to_owned());
                out.column_mut(c).assign(&result);
            }
            out
        }
    }
}

impl SurvivalMarginalSlopeFamily {
    /// Unified dense joint Hessian assembly for flex and timewiggle paths.
    /// Both paths use q-geometry Jacobians via accumulate_dynamic_q_joint_row.
    /// The rigid path (no flex, no timewiggle) uses the RowKernel fast path.
    fn evaluate_exact_newton_joint_dynamic_q_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        if flex_active {
            self.validate_exact_monotonicity(block_states)?;
        }
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = if flex_active {
            flex_identity_block_pairs(&primary, &slices)
        } else {
            vec![]
        };

        type Acc = (f64, Array1<f64>, Array2<f64>);
        // Per-thread accumulator carries a `SurvivalMarginalSlopeDynamicRow`
        // workspace alongside the (nll, gradient, hessian) tuple so the nine
        // Array2/Array1 buffers inside it are reused across all rows assigned
        // to a single rayon worker. At biobank scale this eliminates the
        // ~80 GB-per-sweep allocator traffic the fresh-allocation path used.
        type AccWithWs = (Acc, SurvivalMarginalSlopeDynamicRow);
        let make_acc = || -> AccWithWs {
            (
                (
                    0.0,
                    Array1::zeros(p_total),
                    Array2::zeros((p_total, p_total)),
                ),
                SurvivalMarginalSlopeDynamicRow::empty_workspace(),
            )
        };

        let final_acc = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                let (state, q_geom) = &mut acc;
                self.row_dynamic_q_geometry_into(row, block_states, q_geom)?;
                let (row_nll, f_pi, f_pipi) = if flex_active {
                    self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        q_geom,
                        &primary,
                    )?
                } else {
                    self.compute_row_primary_gradient_hessian_uncached(row, block_states)?
                };
                state.0 -= row_nll;
                self.accumulate_dynamic_q_joint_row(
                    row,
                    &slices,
                    q_geom,
                    f_pi.view(),
                    f_pipi.view(),
                    &identity_blocks,
                    &mut state.1,
                    &mut state.2,
                )?;
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.0.0 += right.0.0;
                left.0.1 += &right.0.1;
                left.0.2 += &right.0.2;
                Ok(left)
            })?;
        Ok(final_acc.0)
    }

    fn evaluate_exact_newton_joint_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            self.evaluate_exact_newton_joint_dynamic_q_dense(block_states)
        } else {
            let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
            let rows = crate::families::row_kernel::RowSet::All;
            let cache = build_row_kernel_cache(&kern, &rows)?;
            Ok((
                row_kernel_log_likelihood(&cache, &rows),
                -row_kernel_gradient(&kern, &cache, &rows),
                row_kernel_hessian_dense(&kern, &cache, &rows),
            ))
        }
    }

    fn evaluate_exact_newton_joint_gradient_dynamic_q(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>), String> {
        let flex_active = self.effective_flex_active(block_states)?;
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let identity_blocks = if flex_active {
            flex_identity_block_pairs(&primary, &slices)
        } else {
            vec![]
        };
        type Acc = (f64, Array1<f64>);
        let make_acc = || -> Acc { (0.0, Array1::zeros(slices.total)) };

        (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                let q_geom = self.row_dynamic_q_gradient(row, block_states)?;
                let (row_nll, f_pi) = if flex_active {
                    self.compute_row_flex_primary_gradient_exact(
                        row,
                        block_states,
                        &q_geom,
                        &primary,
                    )?
                } else {
                    let (nll, grad_arr, _) = self.row_primary_closed_form_rigid(
                        row,
                        q_geom.q0,
                        q_geom.q1,
                        q_geom.qd1,
                        block_states,
                        self.probit_frailty_scale(),
                    )?;
                    (nll, Array1::from_vec(grad_arr.to_vec()))
                };
                acc.0 -= row_nll;
                self.accumulate_dynamic_q_core_gradient_first_order(
                    row,
                    &slices,
                    &q_geom,
                    f_pi.slice(s![0..N_PRIMARY]),
                    &mut acc.1,
                )?;
                for (primary_range, joint_range) in &identity_blocks {
                    for local in 0..primary_range.len() {
                        acc.1[joint_range.start + local] -= f_pi[primary_range.start + local];
                    }
                }
                Ok(acc)
            })
            .try_reduce(make_acc, |mut left, right| -> Result<_, String> {
                left.0 += right.0;
                left.1 += &right.1;
                Ok(left)
            })
    }

    /// Exact directional derivative of the joint Hessian for timewiggle-only
    /// models (no score-warp / link-deviation).  Computes the derivative by
    /// differentiating the J^T H J + f·K pullback through the timewiggle
    /// q-map geometry (equation 47 of the unified pullback framework).
    fn exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: Option<&EvalCache>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let p_total = slices.total;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);

        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let primary_owned;
                    let (f_pi, h_pi) = if let Some(cached) = cache {
                        self.row_primary_gradient_hessian(row, cached)
                    } else {
                        primary_owned =
                            self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                        (&primary_owned.1, &primary_owned.2)
                    };
                    // Inline primary direction from already-computed q_geom
                    // (avoids redundant row_dynamic_q_geometry call)
                    let d_logslope = d_beta_flat.slice(s![slices.logslope.clone()]);
                    let u_d = Array1::from_vec(vec![
                        q_geom.dq0_time.dot(&d_time) + q_geom.dq0_marginal.dot(&d_marginal),
                        q_geom.dq1_time.dot(&d_time) + q_geom.dq1_marginal.dot(&d_marginal),
                        q_geom.dqd1_time.dot(&d_time) + q_geom.dqd1_marginal.dot(&d_marginal),
                        self.logslope_design.dot_row_view(row, d_logslope),
                    ]);
                    let t_ud = self.row_primary_third_contracted(row, block_states, u_d.view())?;
                    let h_ud = h_pi.dot(&u_d);

                    // Term 1 + 3: reuse core accumulator with (H·u^d, T[u^d])
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &mut acc,
                    )?;

                    // ── Timewiggle Jacobian derivatives ────────────────
                    let ec = self
                        .design_entry
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
                    let xc = self
                        .design_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
                    let dc = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
                    let xe = ec.row(0).slice(s![..p_base]).to_owned();
                    let xx = xc.row(0).slice(s![..p_base]).to_owned();
                    let xd = dc.row(0).slice(s![..p_base]).to_owned();
                    let mc = self
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
                    let mr = mc.row(0).to_owned();
                    let dh0 = xe.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
                    let dh1 = xx.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
                    let ddr = xd.dot(&d_time.slice(s![..p_base]));
                    let bm = block_states[1].eta[row];
                    let h0 = xe.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
                    let h1 = xx.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
                    let dr =
                        xd.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
                    let eg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
                        .ok_or_else(|| "timewiggle geometry missing at entry".to_string())?;
                    let xg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
                        .ok_or_else(|| "timewiggle geometry missing at exit".to_string())?;
                    let (m2e, m3e) = (eg.d2q_dq02[0], eg.d3q_dq03[0]);
                    let (m2x, m3x, m4x) = (xg.d2q_dq02[0], xg.d3q_dq03[0], xg.d4q_dq04[0]);

                    // dJ_{q,time}[a] / dβ[d]
                    let mut dj0t = vec![0.0f64; p_time];
                    let mut dj1t = vec![0.0f64; p_time];
                    let mut djdt = vec![0.0f64; p_time];
                    for a in 0..p_base {
                        dj0t[a] = m2e * dh0 * xe[a];
                        dj1t[a] = m2x * dh1 * xx[a];
                        djdt[a] = m3x * dh1 * dr * xx[a] + m2x * ddr * xx[a] + m2x * dh1 * xd[a];
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        dj0t[ci] = eg.basis_d1[[0, li]] * dh0;
                        dj1t[ci] = xg.basis_d1[[0, li]] * dh1;
                        djdt[ci] = xg.basis_d2[[0, li]] * dh1 * dr + xg.basis_d1[[0, li]] * ddr;
                    }
                    let djt = [&dj0t[..], &dj1t[..], &djdt[..]];
                    let mut dj0m = vec![0.0f64; p_marginal];
                    let mut dj1m = vec![0.0f64; p_marginal];
                    let mut djdm = vec![0.0f64; p_marginal];
                    for a in 0..p_marginal {
                        dj0m[a] = m2e * dh0 * mr[a];
                        dj1m[a] = m2x * dh1 * mr[a];
                        djdm[a] = m3x * dh1 * dr * mr[a] + m2x * ddr * mr[a];
                    }
                    let djm = [&dj0m[..], &dj1m[..], &djdm[..]];
                    let jt: [&Array1<f64>; 3] =
                        [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
                    let jm: [&Array1<f64>; 3] = [
                        &q_geom.dq0_marginal,
                        &q_geom.dq1_marginal,
                        &q_geom.dqd1_marginal,
                    ];

                    // Term 2: (dJ/d)^T H J + J^T H (dJ/d)
                    for a in 0..p_time {
                        for b in 0..p_time {
                            let mut v = 0.0;
                            for qu in 0..3 {
                                for qv in 0..3 {
                                    v += h_pi[[qu, qv]]
                                        * (djt[qu][a] * jt[qv][b] + jt[qu][a] * djt[qv][b]);
                                }
                            }
                            acc[[slices.time.start + a, slices.time.start + b]] += v;
                        }
                    }
                    for a in 0..p_marginal {
                        for b in 0..p_marginal {
                            let mut v = 0.0;
                            for qu in 0..3 {
                                for qv in 0..3 {
                                    v += h_pi[[qu, qv]]
                                        * (djm[qu][a] * jm[qv][b] + jm[qu][a] * djm[qv][b]);
                                }
                            }
                            acc[[slices.marginal.start + a, slices.marginal.start + b]] += v;
                        }
                    }
                    for a in 0..p_time {
                        for b in 0..p_marginal {
                            let mut v = 0.0;
                            for qu in 0..3 {
                                for qv in 0..3 {
                                    v += h_pi[[qu, qv]]
                                        * (djt[qu][a] * jm[qv][b] + jt[qu][a] * djm[qv][b]);
                                }
                            }
                            acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                        }
                    }
                    let gc = self
                        .logslope_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("logslope_design try_row_chunk: {e}"))?;
                    let gr = gc.row(0);
                    for a in 0..p_time {
                        let mut w = 0.0;
                        for qu in 0..3 {
                            w += h_pi[[qu, 3]] * djt[qu][a];
                        }
                        for b in 0..slices.logslope.len() {
                            let v = w * gr[b];
                            acc[[slices.time.start + a, slices.logslope.start + b]] += v;
                            acc[[slices.logslope.start + b, slices.time.start + a]] += v;
                        }
                    }
                    for a in 0..p_marginal {
                        let mut w = 0.0;
                        for qu in 0..3 {
                            w += h_pi[[qu, 3]] * djm[qu][a];
                        }
                        for b in 0..slices.logslope.len() {
                            let v = w * gr[b];
                            acc[[slices.marginal.start + a, slices.logslope.start + b]] += v;
                            acc[[slices.logslope.start + b, slices.marginal.start + a]] += v;
                        }
                    }

                    // Term 4: Σ_r f_r dK_r/d
                    for a in 0..p_base {
                        for b in 0..p_base {
                            let dk0 = m3e * dh0 * xe[a] * xe[b];
                            let dk1 = m3x * dh1 * xx[a] * xx[b];
                            let dkd = m4x * dh1 * dr * xx[a] * xx[b]
                                + m3x * ddr * xx[a] * xx[b]
                                + m3x * dh1 * (xx[a] * xd[b] + xd[a] * xx[b]);
                            acc[[slices.time.start + a, slices.time.start + b]] +=
                                f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                        }
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for a in 0..p_base {
                            let dk0 = eg.basis_d2[[0, li]] * dh0 * xe[a];
                            let dk1 = xg.basis_d2[[0, li]] * dh1 * xx[a];
                            let dkd = xg.basis_d3[[0, li]] * dh1 * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddr * xx[a]
                                + xg.basis_d2[[0, li]] * dh1 * xd[a];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + a, slices.time.start + ci]] += v;
                            acc[[slices.time.start + ci, slices.time.start + a]] += v;
                        }
                    }
                    for a in 0..p_base {
                        for b in 0..p_marginal {
                            let dk0 = m3e * dh0 * xe[a] * mr[b];
                            let dk1 = m3x * dh1 * xx[a] * mr[b];
                            let dkd = m4x * dh1 * dr * xx[a] * mr[b]
                                + m3x * ddr * xx[a] * mr[b]
                                + m3x * dh1 * xd[a] * mr[b];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                        }
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for b in 0..p_marginal {
                            let dk0 = eg.basis_d2[[0, li]] * dh0 * mr[b];
                            let dk1 = xg.basis_d2[[0, li]] * dh1 * mr[b];
                            let dkd = xg.basis_d3[[0, li]] * dh1 * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddr * mr[b];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
                        }
                    }
                    for a in 0..p_marginal {
                        for b in 0..p_marginal {
                            let dk0 = m3e * dh0 * mr[a] * mr[b];
                            let dk1 = m3x * dh1 * mr[a] * mr[b];
                            let dkd = m4x * dh1 * dr * mr[a] * mr[b] + m3x * ddr * mr[a] * mr[b];
                            acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                                f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    /// Exact directional derivative of the joint Hessian for simultaneous
    /// timewiggle + flexible score/link warps.
    ///
    /// This extends the timewiggle-only transport by keeping the full flexible
    /// primary Hessian/third contraction live while only differentiating the
    /// q-geometry Jacobian and K tensors for the dynamic q coordinates. The
    /// score/link primary coordinates remain identity-mapped, so their
    /// contribution enters through the shared pullback term plus cross-columns
    /// against the dJ correction.
    fn exact_newton_joint_hessian_directional_derivative_timewiggle_flex(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let p_total = slices.total;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        let d_marginal = d_beta_flat.slice(s![slices.marginal.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_time_w = beta_time.slice(s![time_tail.clone()]);

        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) = self.compute_row_flex_primary_gradient_hessian_exact(
                        row,
                        block_states,
                        &q_geom,
                        &primary,
                    )?;
                    let u_d = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_beta_flat,
                    )?;
                    let t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);

                    self.accumulate_dynamic_q_joint_row(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &identity_blocks,
                        &mut Array1::zeros(p_total),
                        &mut acc,
                    )?;

                    let ec = self
                        .design_entry
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
                    let xc = self
                        .design_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
                    let dc = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
                    let xe = ec.row(0).slice(s![..p_base]).to_owned();
                    let xx = xc.row(0).slice(s![..p_base]).to_owned();
                    let xd = dc.row(0).slice(s![..p_base]).to_owned();
                    let mc = self
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
                    let mr = mc.row(0).to_owned();
                    let dh0 = xe.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
                    let dh1 = xx.dot(&d_time.slice(s![..p_base])) + mr.dot(&d_marginal);
                    let ddr = xd.dot(&d_time.slice(s![..p_base]));
                    let bm = block_states[1].eta[row];
                    let h0 = xe.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
                    let h1 = xx.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
                    let dr =
                        xd.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];
                    let eg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_time_w)?
                        .ok_or_else(|| "timewiggle geometry missing at entry".to_string())?;
                    let xg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_time_w)?
                        .ok_or_else(|| "timewiggle geometry missing at exit".to_string())?;
                    let (m2e, m3e) = (eg.d2q_dq02[0], eg.d3q_dq03[0]);
                    let (m2x, m3x, m4x) = (xg.d2q_dq02[0], xg.d3q_dq03[0], xg.d4q_dq04[0]);

                    let mut dj0t = vec![0.0f64; p_time];
                    let mut dj1t = vec![0.0f64; p_time];
                    let mut djdt = vec![0.0f64; p_time];
                    for a in 0..p_base {
                        dj0t[a] = m2e * dh0 * xe[a];
                        dj1t[a] = m2x * dh1 * xx[a];
                        djdt[a] = m3x * dh1 * dr * xx[a] + m2x * ddr * xx[a] + m2x * dh1 * xd[a];
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        dj0t[ci] = eg.basis_d1[[0, li]] * dh0;
                        dj1t[ci] = xg.basis_d1[[0, li]] * dh1;
                        djdt[ci] = xg.basis_d2[[0, li]] * dh1 * dr + xg.basis_d1[[0, li]] * ddr;
                    }
                    let djt = [&dj0t[..], &dj1t[..], &djdt[..]];
                    let mut dj0m = vec![0.0f64; p_marginal];
                    let mut dj1m = vec![0.0f64; p_marginal];
                    let mut djdm = vec![0.0f64; p_marginal];
                    for a in 0..p_marginal {
                        dj0m[a] = m2e * dh0 * mr[a];
                        dj1m[a] = m2x * dh1 * mr[a];
                        djdm[a] = m3x * dh1 * dr * mr[a] + m2x * ddr * mr[a];
                    }
                    let djm = [&dj0m[..], &dj1m[..], &djdm[..]];
                    let jt: [&Array1<f64>; 3] =
                        [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
                    let jm: [&Array1<f64>; 3] = [
                        &q_geom.dq0_marginal,
                        &q_geom.dq1_marginal,
                        &q_geom.dqd1_marginal,
                    ];

                    for a in 0..p_time {
                        for b in 0..p_time {
                            let mut v = 0.0;
                            for qu in 0..3 {
                                for qv in 0..3 {
                                    v += h_pi[[qu, qv]]
                                        * (djt[qu][a] * jt[qv][b] + jt[qu][a] * djt[qv][b]);
                                }
                            }
                            acc[[slices.time.start + a, slices.time.start + b]] += v;
                        }
                    }
                    for a in 0..p_marginal {
                        for b in 0..p_marginal {
                            let mut v = 0.0;
                            for qu in 0..3 {
                                for qv in 0..3 {
                                    v += h_pi[[qu, qv]]
                                        * (djm[qu][a] * jm[qv][b] + jm[qu][a] * djm[qv][b]);
                                }
                            }
                            acc[[slices.marginal.start + a, slices.marginal.start + b]] += v;
                        }
                    }
                    for a in 0..p_time {
                        for b in 0..p_marginal {
                            let mut v = 0.0;
                            for qu in 0..3 {
                                for qv in 0..3 {
                                    v += h_pi[[qu, qv]]
                                        * (djt[qu][a] * jm[qv][b] + jt[qu][a] * djm[qv][b]);
                                }
                            }
                            acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                        }
                    }
                    let gc = self
                        .logslope_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("logslope_design try_row_chunk: {e}"))?;
                    let gr = gc.row(0);
                    for a in 0..p_time {
                        let mut w = 0.0;
                        for qu in 0..3 {
                            w += h_pi[[qu, 3]] * djt[qu][a];
                        }
                        for b in 0..slices.logslope.len() {
                            let v = w * gr[b];
                            acc[[slices.time.start + a, slices.logslope.start + b]] += v;
                            acc[[slices.logslope.start + b, slices.time.start + a]] += v;
                        }
                    }
                    for a in 0..p_marginal {
                        let mut w = 0.0;
                        for qu in 0..3 {
                            w += h_pi[[qu, 3]] * djm[qu][a];
                        }
                        for b in 0..slices.logslope.len() {
                            let v = w * gr[b];
                            acc[[slices.marginal.start + a, slices.logslope.start + b]] += v;
                            acc[[slices.logslope.start + b, slices.marginal.start + a]] += v;
                        }
                    }

                    for (primary_range, joint_range) in &identity_blocks {
                        for local in 0..primary_range.len() {
                            let primary_idx = primary_range.start + local;
                            let joint_idx = joint_range.start + local;
                            for a in 0..p_time {
                                let mut value = 0.0;
                                for qu in 0..3 {
                                    value += h_pi[[qu, primary_idx]] * djt[qu][a];
                                }
                                acc[[slices.time.start + a, joint_idx]] += value;
                                acc[[joint_idx, slices.time.start + a]] += value;
                            }
                            for a in 0..p_marginal {
                                let mut value = 0.0;
                                for qu in 0..3 {
                                    value += h_pi[[qu, primary_idx]] * djm[qu][a];
                                }
                                acc[[slices.marginal.start + a, joint_idx]] += value;
                                acc[[joint_idx, slices.marginal.start + a]] += value;
                            }
                        }
                    }

                    for a in 0..p_base {
                        for b in 0..p_base {
                            let dk0 = m3e * dh0 * xe[a] * xe[b];
                            let dk1 = m3x * dh1 * xx[a] * xx[b];
                            let dkd = m4x * dh1 * dr * xx[a] * xx[b]
                                + m3x * ddr * xx[a] * xx[b]
                                + m3x * dh1 * (xx[a] * xd[b] + xd[a] * xx[b]);
                            acc[[slices.time.start + a, slices.time.start + b]] +=
                                f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                        }
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for a in 0..p_base {
                            let dk0 = eg.basis_d2[[0, li]] * dh0 * xe[a];
                            let dk1 = xg.basis_d2[[0, li]] * dh1 * xx[a];
                            let dkd = xg.basis_d3[[0, li]] * dh1 * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddr * xx[a]
                                + xg.basis_d2[[0, li]] * dh1 * xd[a];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + a, slices.time.start + ci]] += v;
                            acc[[slices.time.start + ci, slices.time.start + a]] += v;
                        }
                    }
                    for a in 0..p_base {
                        for b in 0..p_marginal {
                            let dk0 = m3e * dh0 * xe[a] * mr[b];
                            let dk1 = m3x * dh1 * xx[a] * mr[b];
                            let dkd = m4x * dh1 * dr * xx[a] * mr[b]
                                + m3x * ddr * xx[a] * mr[b]
                                + m3x * dh1 * xd[a] * mr[b];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                        }
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for b in 0..p_marginal {
                            let dk0 = eg.basis_d2[[0, li]] * dh0 * mr[b];
                            let dk1 = xg.basis_d2[[0, li]] * dh1 * mr[b];
                            let dkd = xg.basis_d3[[0, li]] * dh1 * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddr * mr[b];
                            let v = f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                            acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
                        }
                    }
                    for a in 0..p_marginal {
                        for b in 0..p_marginal {
                            let dk0 = m3e * dh0 * mr[a] * mr[b];
                            let dk1 = m3x * dh1 * mr[a] * mr[b];
                            let dkd = m4x * dh1 * dr * mr[a] * mr[b] + m3x * ddr * mr[a] * mr[b];
                            acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                                f_pi[0] * dk0 + f_pi[1] * dk1 + f_pi[2] * dkd;
                        }
                    }

                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    fn exact_newton_joint_hessian_directional_derivative_timewiggle_cached(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        cache: &EvalCache,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
            block_states,
            d_beta_flat,
            Some(cache),
        )
    }

    fn exact_newton_joint_hessian_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_timewiggle_inner(
            block_states,
            d_beta_flat,
            None,
        )
    }
    /// Fully exact second directional derivative D²H[d,e] for timewiggle-only.
    /// Differentiates DH[e] along d analytically using m₂–m₅ scalars.
    ///
    /// D²H[d,e] = J^T Ψ J  +  Σ γ_r K_r
    ///   + Σ bilinear(W_k, left_k, right_k)  for k in {T_e×dJ_d, T_d×dJ_e, H×d²J, H×dJ_d×dJ_e}
    ///   + dK cross-terms: (Hu_d)·dK_e + (Hu_e)·dK_d + f·d²K
    ///
    /// where Ψ = Q[u_d,u_e] + T[du_e/dd], γ = T_d·u_e + H·du_e/dd.
    fn exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let p_total = slices.total;
        let p_time = slices.time.len();
        let p_marginal = slices.marginal.len();
        let time_tail = self.time_wiggle_range();
        let p_base = time_tail.start;
        let du_t = d_u.slice(s![slices.time.clone()]);
        let du_m = d_u.slice(s![slices.marginal.clone()]);
        let du_g = d_u.slice(s![slices.logslope.clone()]);
        let dv_t = d_v.slice(s![slices.time.clone()]);
        let dv_m = d_v.slice(s![slices.marginal.clone()]);
        let dv_g = d_v.slice(s![slices.logslope.clone()]);
        let beta_time = &block_states[0].beta;
        let beta_tw = beta_time.slice(s![time_tail.clone()]);

        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let (_, f_pi, h_pi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;

                    // Primary directions
                    let ud = Array1::from_vec(vec![
                        q_geom.dq0_time.dot(&du_t) + q_geom.dq0_marginal.dot(&du_m),
                        q_geom.dq1_time.dot(&du_t) + q_geom.dq1_marginal.dot(&du_m),
                        q_geom.dqd1_time.dot(&du_t) + q_geom.dqd1_marginal.dot(&du_m),
                        self.logslope_design.dot_row_view(row, du_g),
                    ]);
                    let ue = Array1::from_vec(vec![
                        q_geom.dq0_time.dot(&dv_t) + q_geom.dq0_marginal.dot(&dv_m),
                        q_geom.dq1_time.dot(&dv_t) + q_geom.dq1_marginal.dot(&dv_m),
                        q_geom.dqd1_time.dot(&dv_t) + q_geom.dqd1_marginal.dot(&dv_m),
                        self.logslope_design.dot_row_view(row, dv_g),
                    ]);

                    let t_d = self.row_primary_third_contracted(row, block_states, ud.view())?;
                    let t_e = self.row_primary_third_contracted(row, block_states, ue.view())?;
                    let q_de = self.row_primary_fourth_contracted(
                        row,
                        block_states,
                        ud.view(),
                        ue.view(),
                    )?;
                    let h_ud = h_pi.dot(&ud);
                    let h_ue = h_pi.dot(&ue);

                    // ── Timewiggle geometry ─────────────────────────────
                    let ec = self
                        .design_entry
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_entry try_row_chunk: {e}"))?;
                    let xc = self
                        .design_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_exit try_row_chunk: {e}"))?;
                    let dc = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("design_derivative_exit try_row_chunk: {e}"))?;
                    let xe = ec.row(0).slice(s![..p_base]).to_owned();
                    let xx = xc.row(0).slice(s![..p_base]).to_owned();
                    let xd = dc.row(0).slice(s![..p_base]).to_owned();
                    let mc = self
                        .marginal_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("marginal_design try_row_chunk: {e}"))?;
                    let mr = mc.row(0).to_owned();

                    let bm = block_states[1].eta[row];
                    let h0 = xe.dot(&beta_time.slice(s![..p_base])) + self.offset_entry[row] + bm;
                    let h1 = xx.dot(&beta_time.slice(s![..p_base])) + self.offset_exit[row] + bm;
                    let dr =
                        xd.dot(&beta_time.slice(s![..p_base])) + self.derivative_offset_exit[row];

                    let eg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h0]).view(), beta_tw)?
                        .ok_or_else(|| "timewiggle geometry missing".to_string())?;
                    let xg = self
                        .time_wiggle_geometry(Array1::from_vec(vec![h1]).view(), beta_tw)?
                        .ok_or_else(|| "timewiggle geometry missing".to_string())?;

                    let m2_en = eg.d2q_dq02[0];
                    let m3_en = eg.d3q_dq03[0];
                    let m4_en = eg.d4q_dq04[0];
                    let m2_ex = xg.d2q_dq02[0];
                    let m3_ex = xg.d3q_dq03[0];
                    let m4_ex = xg.d4q_dq04[0];
                    let m5_ex = xg.d5q_dq05[0];

                    // Direction scalars (h linear in β ⇒ d²h/ded = 0)
                    let dh0d = xe.dot(&du_t.slice(s![..p_base])) + mr.dot(&du_m);
                    let dh1d = xx.dot(&du_t.slice(s![..p_base])) + mr.dot(&du_m);
                    let ddrd = xd.dot(&du_t.slice(s![..p_base]));
                    let dh0e = xe.dot(&dv_t.slice(s![..p_base])) + mr.dot(&dv_m);
                    let dh1e = xx.dot(&dv_t.slice(s![..p_base])) + mr.dot(&dv_m);
                    let ddre = xd.dot(&dv_t.slice(s![..p_base]));

                    // du_e/dd = (dJ/dd)·e_v — primary direction of e perturbed by d
                    // dJ[q0,time_a]/dd = m2_en*dh0d*xe[a] for base, basis_d1*dh0d for wiggle
                    let due_d = {
                        let mut v = [0.0f64; 4];
                        for a in 0..p_base {
                            v[0] += m2_en * dh0d * xe[a] * dv_t[a];
                            v[1] += m2_ex * dh1d * xx[a] * dv_t[a];
                            v[2] += (m3_ex * dh1d * dr * xx[a]
                                + m2_ex * ddrd * xx[a]
                                + m2_ex * dh1d * xd[a])
                                * dv_t[a];
                        }
                        for li in 0..time_tail.len() {
                            let ci = time_tail.start + li;
                            v[0] += eg.basis_d1[[0, li]] * dh0d * dv_t[ci];
                            v[1] += xg.basis_d1[[0, li]] * dh1d * dv_t[ci];
                            v[2] += (xg.basis_d2[[0, li]] * dh1d * dr
                                + xg.basis_d1[[0, li]] * ddrd)
                                * dv_t[ci];
                        }
                        for a in 0..p_marginal {
                            v[0] += m2_en * dh0d * mr[a] * dv_m[a];
                            v[1] += m2_ex * dh1d * mr[a] * dv_m[a];
                            v[2] += (m3_ex * dh1d * dr * mr[a] + m2_ex * ddrd * mr[a]) * dv_m[a];
                        }
                        // v[3] = 0 (logslope J is constant)
                        Array1::from_vec(v.to_vec())
                    };

                    // Ψ = Q[ud,ue] + T[due_d]
                    let t_due =
                        self.row_primary_third_contracted(row, block_states, due_d.view())?;
                    let psi = &q_de + &t_due;

                    // γ = T_d·ue + H·due_d
                    let gamma = &t_d.dot(&ue) + &h_pi.dot(&due_d);

                    // ── Term A: J^T Ψ J + γ·K ─────────────────────────
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        gamma.view(),
                        psi.view(),
                        &mut acc,
                    )?;

                    let jt = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
                    let jm = [
                        &q_geom.dq0_marginal,
                        &q_geom.dq1_marginal,
                        &q_geom.dqd1_marginal,
                    ];
                    let gc = self
                        .logslope_design
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| format!("logslope_design try_row_chunk: {e}"))?;
                    let gr = gc.row(0);

                    // ── Helper: accumulate a symmetrized bilinear term ──
                    // Adds Σ W[qu,qv] * (left[qu,a]*right[qv,b] + right[qu,a]*left[qv,b])
                    // for all block pairs into acc.
                    macro_rules! accum_bilinear {
                        ($w:expr, $lt:expr, $lm:expr, $rt:expr, $rm:expr) => {
                            for a in 0..p_time {
                                for b in 0..p_time {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lt[qu][a] * $rt[qv][b]
                                                    + $rt[qu][a] * $lt[qv][b]);
                                        }
                                    }
                                    acc[[slices.time.start + a, slices.time.start + b]] += v;
                                }
                            }
                            for a in 0..p_marginal {
                                for b in 0..p_marginal {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lm[qu][a] * $rm[qv][b]
                                                    + $rm[qu][a] * $lm[qv][b]);
                                        }
                                    }
                                    acc[[slices.marginal.start + a, slices.marginal.start + b]] +=
                                        v;
                                }
                            }
                            for a in 0..p_time {
                                for b in 0..p_marginal {
                                    let mut v = 0.0;
                                    for qu in 0..3 {
                                        for qv in 0..3 {
                                            v += $w[[qu, qv]]
                                                * ($lt[qu][a] * $rm[qv][b]
                                                    + $rt[qu][a] * $lm[qv][b]);
                                        }
                                    }
                                    acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                                    acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                                }
                            }
                            for a in 0..p_time {
                                let mut w2 = 0.0;
                                for qu in 0..3 {
                                    w2 += $w[[qu, 3]] * $lt[qu][a];
                                }
                                for b in 0..slices.logslope.len() {
                                    let v = w2 * gr[b];
                                    acc[[slices.time.start + a, slices.logslope.start + b]] += v;
                                    acc[[slices.logslope.start + b, slices.time.start + a]] += v;
                                }
                            }
                            for a in 0..p_marginal {
                                let mut w2 = 0.0;
                                for qu in 0..3 {
                                    w2 += $w[[qu, 3]] * $lm[qu][a];
                                }
                                for b in 0..slices.logslope.len() {
                                    let v = w2 * gr[b];
                                    acc[[slices.marginal.start + a, slices.logslope.start + b]] +=
                                        v;
                                    acc[[slices.logslope.start + b, slices.marginal.start + a]] +=
                                        v;
                                }
                            }
                        };
                    }

                    // ── Build dJ arrays for both directions ────────────
                    // (same code as first directional, for d and e)
                    type DjArrays = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
                    let build_dj = |dh0: f64, dh1: f64, ddr_val: f64| -> DjArrays {
                        let mut j0t = vec![0.0f64; p_time];
                        let mut j1t = vec![0.0f64; p_time];
                        let mut jdt = vec![0.0f64; p_time];
                        for a in 0..p_base {
                            j0t[a] = m2_en * dh0 * xe[a];
                            j1t[a] = m2_ex * dh1 * xx[a];
                            jdt[a] = m3_ex * dh1 * dr * xx[a]
                                + m2_ex * ddr_val * xx[a]
                                + m2_ex * dh1 * xd[a];
                        }
                        for li in 0..time_tail.len() {
                            let ci = time_tail.start + li;
                            j0t[ci] = eg.basis_d1[[0, li]] * dh0;
                            j1t[ci] = xg.basis_d1[[0, li]] * dh1;
                            jdt[ci] =
                                xg.basis_d2[[0, li]] * dh1 * dr + xg.basis_d1[[0, li]] * ddr_val;
                        }
                        let mut j0m = vec![0.0f64; p_marginal];
                        let mut j1m = vec![0.0f64; p_marginal];
                        let mut jdm = vec![0.0f64; p_marginal];
                        for a in 0..p_marginal {
                            j0m[a] = m2_en * dh0 * mr[a];
                            j1m[a] = m2_ex * dh1 * mr[a];
                            jdm[a] = m3_ex * dh1 * dr * mr[a] + m2_ex * ddr_val * mr[a];
                        }
                        (j0t, j1t, jdt, j0m, j1m, jdm)
                    };

                    let (djd0t, djd1t, djddt, djd0m, djd1m, djddm) = build_dj(dh0d, dh1d, ddrd);
                    let djd_t = [&djd0t[..], &djd1t[..], &djddt[..]];
                    let djd_m = [&djd0m[..], &djd1m[..], &djddm[..]];

                    let (dje0t, dje1t, djedt, dje0m, dje1m, djedm) = build_dj(dh0e, dh1e, ddre);
                    let dje_t = [&dje0t[..], &dje1t[..], &djedt[..]];
                    let dje_m = [&dje0m[..], &dje1m[..], &djedm[..]];

                    // ── d²J/ded (cross derivative, h linear ⇒ d²h=0) ──
                    let mut d2j0t = vec![0.0f64; p_time];
                    let mut d2j1t = vec![0.0f64; p_time];
                    let mut d2jdt = vec![0.0f64; p_time];
                    for a in 0..p_base {
                        d2j0t[a] = m3_en * dh0d * dh0e * xe[a];
                        d2j1t[a] = m3_ex * dh1d * dh1e * xx[a];
                        d2jdt[a] = m4_ex * dh1d * dh1e * dr * xx[a]
                            + m3_ex * (dh1d * ddre + dh1e * ddrd) * xx[a]
                            + m3_ex * dh1d * dh1e * xd[a];
                    }
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        d2j0t[ci] = eg.basis_d2[[0, li]] * dh0d * dh0e;
                        d2j1t[ci] = xg.basis_d2[[0, li]] * dh1d * dh1e;
                        d2jdt[ci] = xg.basis_d3[[0, li]] * dh1d * dh1e * dr
                            + xg.basis_d2[[0, li]] * (dh1d * ddre + dh1e * ddrd);
                    }
                    let d2j_t = [&d2j0t[..], &d2j1t[..], &d2jdt[..]];
                    let mut d2j0m = vec![0.0f64; p_marginal];
                    let mut d2j1m = vec![0.0f64; p_marginal];
                    let mut d2jdm = vec![0.0f64; p_marginal];
                    for a in 0..p_marginal {
                        d2j0m[a] = m3_en * dh0d * dh0e * mr[a];
                        d2j1m[a] = m3_ex * dh1d * dh1e * mr[a];
                        d2jdm[a] = m4_ex * dh1d * dh1e * dr * mr[a]
                            + m3_ex * (dh1d * ddre + dh1e * ddrd) * mr[a];
                    }
                    let d2j_m = [&d2j0m[..], &d2j1m[..], &d2jdm[..]];

                    let jt_s: [&[f64]; 3] = [
                        jt[0].as_slice().unwrap(),
                        jt[1].as_slice().unwrap(),
                        jt[2].as_slice().unwrap(),
                    ];
                    let jm_s: [&[f64]; 3] = [
                        jm[0].as_slice().unwrap(),
                        jm[1].as_slice().unwrap(),
                        jm[2].as_slice().unwrap(),
                    ];

                    // ── Term B: bilinear cross-terms ────────────────────
                    // (dJ_d)^T T_e J + J^T T_e (dJ_d) — differentiated from Term 1
                    accum_bilinear!(t_e, djd_t, djd_m, jt_s, jm_s);
                    // (dJ_e)^T T_d J + J^T T_d (dJ_e) — symmetry partner
                    accum_bilinear!(t_d, dje_t, dje_m, jt_s, jm_s);
                    // (d²J)^T H J + J^T H (d²J) — from Term 2
                    accum_bilinear!(h_pi, d2j_t, d2j_m, jt_s, jm_s);
                    // (dJ_d)^T H (dJ_e) + (dJ_e)^T H (dJ_d) — from Term 2
                    accum_bilinear!(h_pi, djd_t, djd_m, dje_t, dje_m);

                    // ── Term C: dK cross-terms ──────────────────────────
                    // (H·ud)_r dK_r/de + (H·ue)_r dK_r/dd + f_r d²K_r/ded
                    //
                    // dK[q,a,b]/dd = d(K[q,a,b])/dd where K = m_{k}*product-of-design-rows
                    // d²K[q,a,b]/ded = m_{k+2}*dh_d*dh_e*(...) since d²h/ded=0
                    //
                    // For q0 base×base: K = m2_en*xe[a]*xe[b]
                    //   dK/dd = m3_en*dh0d*xe[a]*xe[b]
                    //   d²K/ded = m4_en*dh0d*dh0e*xe[a]*xe[b]
                    // For q1 base×base: K = m2_ex*xx[a]*xx[b]
                    //   dK/dd = m3_ex*dh1d*xx[a]*xx[b]
                    //   d²K/ded = m4_ex*dh1d*dh1e*xx[a]*xx[b]
                    // For qd1 base×base: K = m3_ex*dr*xx[a]*xx[b] + m2_ex*(xx[a]*xd[b]+xd[a]*xx[b])
                    //   dK/dd = m4_ex*dh1d*dr*xx[a]*xx[b] + m3_ex*ddrd*xx[a]*xx[b]
                    //         + m3_ex*dh1d*(xx[a]*xd[b]+xd[a]*xx[b])
                    //   d²K/ded = m5_ex*dh1d*dh1e*dr*xx[a]*xx[b]
                    //           + m4_ex*(dh1d*ddre+dh1e*ddrd)*xx[a]*xx[b]
                    //           + m4_ex*dh1d*dh1e*(xx[a]*xd[b]+xd[a]*xx[b])

                    // base×base time×time
                    for a in 0..p_base {
                        for b in 0..p_base {
                            let dke_0 = m3_en * dh0e * xe[a] * xe[b];
                            let dke_1 = m3_ex * dh1e * xx[a] * xx[b];
                            let dke_d = m4_ex * dh1e * dr * xx[a] * xx[b]
                                + m3_ex * ddre * xx[a] * xx[b]
                                + m3_ex * dh1e * (xx[a] * xd[b] + xd[a] * xx[b]);
                            let dkd_0 = m3_en * dh0d * xe[a] * xe[b];
                            let dkd_1 = m3_ex * dh1d * xx[a] * xx[b];
                            let dkd_d = m4_ex * dh1d * dr * xx[a] * xx[b]
                                + m3_ex * ddrd * xx[a] * xx[b]
                                + m3_ex * dh1d * (xx[a] * xd[b] + xd[a] * xx[b]);
                            let d2k_0 = m4_en * dh0d * dh0e * xe[a] * xe[b];
                            let d2k_1 = m4_ex * dh1d * dh1e * xx[a] * xx[b];
                            let d2k_d = m5_ex * dh1d * dh1e * dr * xx[a] * xx[b]
                                + m4_ex * (dh1d * ddre + dh1e * ddrd) * xx[a] * xx[b]
                                + m4_ex * dh1d * dh1e * (xx[a] * xd[b] + xd[a] * xx[b]);
                            acc[[slices.time.start + a, slices.time.start + b]] += h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                        }
                    }

                    // base×wiggle time×time
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for a in 0..p_base {
                            // K[q0] for base×wiggle: d2q0/dβ_base[a] dβ_wiggle[li]
                            //   = basis_d1[li]*xe[a] at entry  (m2 * x * basis is wrong; correct is basis_d1*x)
                            // Actually from q_geom: d2q0_time_time[[a, ci]] = basis_d1_entry[li]*xe[a]
                            // dK/dd = basis_d2[li]*dh0d*xe[a]
                            // d²K/ded = basis_d3[li]*dh0d*dh0e*xe[a]
                            let dke_0 = eg.basis_d2[[0, li]] * dh0e * xe[a];
                            let dke_1 = xg.basis_d2[[0, li]] * dh1e * xx[a];
                            let dke_d = xg.basis_d3[[0, li]] * dh1e * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddre * xx[a]
                                + xg.basis_d2[[0, li]] * dh1e * xd[a];
                            let dkd_0 = eg.basis_d2[[0, li]] * dh0d * xe[a];
                            let dkd_1 = xg.basis_d2[[0, li]] * dh1d * xx[a];
                            let dkd_d = xg.basis_d3[[0, li]] * dh1d * dr * xx[a]
                                + xg.basis_d2[[0, li]] * ddrd * xx[a]
                                + xg.basis_d2[[0, li]] * dh1d * xd[a];
                            let d2k_0 = eg.basis_d3[[0, li]] * dh0d * dh0e * xe[a];
                            let d2k_1 = xg.basis_d3[[0, li]] * dh1d * dh1e * xx[a];
                            let d2k_d = xg.basis_d4[[0, li]] * dh1d * dh1e * dr * xx[a]
                                + xg.basis_d3[[0, li]] * (dh1d * ddre + dh1e * ddrd) * xx[a]
                                + xg.basis_d3[[0, li]] * dh1d * dh1e * xd[a];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + a, slices.time.start + ci]] += v;
                            acc[[slices.time.start + ci, slices.time.start + a]] += v;
                        }
                    }

                    // base×marginal time×marginal
                    for a in 0..p_base {
                        for b in 0..p_marginal {
                            let dke_0 = m3_en * dh0e * xe[a] * mr[b];
                            let dke_1 = m3_ex * dh1e * xx[a] * mr[b];
                            let dke_d = m4_ex * dh1e * dr * xx[a] * mr[b]
                                + m3_ex * ddre * xx[a] * mr[b]
                                + m3_ex * dh1e * xd[a] * mr[b];
                            let dkd_0 = m3_en * dh0d * xe[a] * mr[b];
                            let dkd_1 = m3_ex * dh1d * xx[a] * mr[b];
                            let dkd_d = m4_ex * dh1d * dr * xx[a] * mr[b]
                                + m3_ex * ddrd * xx[a] * mr[b]
                                + m3_ex * dh1d * xd[a] * mr[b];
                            let d2k_0 = m4_en * dh0d * dh0e * xe[a] * mr[b];
                            let d2k_1 = m4_ex * dh1d * dh1e * xx[a] * mr[b];
                            let d2k_d = m5_ex * dh1d * dh1e * dr * xx[a] * mr[b]
                                + m4_ex * (dh1d * ddre + dh1e * ddrd) * xx[a] * mr[b]
                                + m4_ex * dh1d * dh1e * xd[a] * mr[b];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + a, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + a]] += v;
                        }
                    }

                    // wiggle×marginal
                    for li in 0..time_tail.len() {
                        let ci = time_tail.start + li;
                        for b in 0..p_marginal {
                            let dke_0 = eg.basis_d2[[0, li]] * dh0e * mr[b];
                            let dke_1 = xg.basis_d2[[0, li]] * dh1e * mr[b];
                            let dke_d = xg.basis_d3[[0, li]] * dh1e * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddre * mr[b];
                            let dkd_0 = eg.basis_d2[[0, li]] * dh0d * mr[b];
                            let dkd_1 = xg.basis_d2[[0, li]] * dh1d * mr[b];
                            let dkd_d = xg.basis_d3[[0, li]] * dh1d * dr * mr[b]
                                + xg.basis_d2[[0, li]] * ddrd * mr[b];
                            let d2k_0 = eg.basis_d3[[0, li]] * dh0d * dh0e * mr[b];
                            let d2k_1 = xg.basis_d3[[0, li]] * dh1d * dh1e * mr[b];
                            let d2k_d = xg.basis_d4[[0, li]] * dh1d * dh1e * dr * mr[b]
                                + xg.basis_d3[[0, li]] * (dh1d * ddre + dh1e * ddrd) * mr[b];
                            let v = h_ud[0] * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                            acc[[slices.time.start + ci, slices.marginal.start + b]] += v;
                            acc[[slices.marginal.start + b, slices.time.start + ci]] += v;
                        }
                    }

                    // marginal×marginal
                    for a in 0..p_marginal {
                        for b in 0..p_marginal {
                            let dke_0 = m3_en * dh0e * mr[a] * mr[b];
                            let dke_1 = m3_ex * dh1e * mr[a] * mr[b];
                            let dke_d =
                                m4_ex * dh1e * dr * mr[a] * mr[b] + m3_ex * ddre * mr[a] * mr[b];
                            let dkd_0 = m3_en * dh0d * mr[a] * mr[b];
                            let dkd_1 = m3_ex * dh1d * mr[a] * mr[b];
                            let dkd_d =
                                m4_ex * dh1d * dr * mr[a] * mr[b] + m3_ex * ddrd * mr[a] * mr[b];
                            let d2k_0 = m4_en * dh0d * dh0e * mr[a] * mr[b];
                            let d2k_1 = m4_ex * dh1d * dh1e * mr[a] * mr[b];
                            let d2k_d = m5_ex * dh1d * dh1e * dr * mr[a] * mr[b]
                                + m4_ex * (dh1d * ddre + dh1e * ddrd) * mr[a] * mr[b];
                            acc[[slices.marginal.start + a, slices.marginal.start + b]] += h_ud[0]
                                * dke_0
                                + h_ud[1] * dke_1
                                + h_ud[2] * dke_d
                                + h_ue[0] * dkd_0
                                + h_ue[1] * dkd_1
                                + h_ue[2] * dkd_d
                                + f_pi[0] * d2k_0
                                + f_pi[1] * d2k_1
                                + f_pi[2] * d2k_d;
                        }
                    }

                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    /// Exact first directional derivative for flex without timewiggle.
    /// J is constant (no wiggle), so DH[d] = J^T T[u^d] J + Σ (Hu^d)_r K_r.
    fn exact_newton_joint_hessian_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let h_pi = self
                        .compute_row_flex_primary_gradient_hessian_exact(
                            row,
                            block_states,
                            &q_geom,
                            &primary,
                        )?
                        .2;
                    let u_d = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_beta_flat,
                    )?;
                    let t_ud =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &u_d)?;
                    let h_ud = h_pi.dot(&u_d);
                    // Core q-geometry pullback (Hessian only)
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        h_ud.view(),
                        t_ud.view(),
                        &mut acc,
                    )?;
                    // Identity block Hessian: cross + diagonal + cross-cross
                    for (primary_range, joint_range) in &identity_blocks {
                        for local in 0..primary_range.len() {
                            self.accumulate_identity_primary_cross_hessian(
                                row,
                                &slices,
                                &q_geom,
                                t_ud.slice(s![0..N_PRIMARY, primary_range.start + local]),
                                joint_range,
                                local,
                                &mut acc,
                            )?;
                        }
                        self.add_dense_submatrix(
                            &mut acc,
                            joint_range,
                            joint_range,
                            t_ud.slice(s![primary_range.clone(), primary_range.clone()]),
                        );
                    }
                    for li in 0..identity_blocks.len() {
                        for ri in li + 1..identity_blocks.len() {
                            let (lp, lj) = &identity_blocks[li];
                            let (rp, rj) = &identity_blocks[ri];
                            self.add_dense_symmetric_cross_submatrix(
                                &mut acc,
                                lj,
                                rj,
                                t_ud.slice(s![lp.clone(), rp.clone()]),
                            );
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    /// Exact second directional derivative for flex without timewiggle.
    /// J constant ⇒ D²H[d,e] = J^T Q[u^d,u^e] J + Σ (T_d·u^e)_r K_r.
    fn exact_newton_joint_hessiansecond_directional_derivative_flex_no_wiggle(
        &self,
        block_states: &[ParameterBlockState],
        d_u: &Array1<f64>,
        d_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let slices = block_slices(self, block_states);
        let primary = flex_primary_slices(self);
        let p_total = slices.total;
        let identity_blocks = flex_identity_block_pairs(&primary, &slices);
        let result = (0..self.n)
            .into_par_iter()
            .try_fold(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut acc, row| -> Result<Array2<f64>, String> {
                    let q_geom = self.row_dynamic_q_geometry(row, block_states)?;
                    let ud = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_u,
                    )?;
                    let ue = self.row_primary_direction_from_flat_dynamic(
                        row,
                        block_states,
                        &slices,
                        d_v,
                    )?;
                    let q_de =
                        self.row_flex_primary_fourth_contracted_exact(row, block_states, &ud, &ue)?;
                    let t_d =
                        self.row_flex_primary_third_contracted_exact(row, block_states, &ud)?;
                    let gamma = t_d.dot(&ue);
                    // Hessian-only: accumulate q-core + identity block Hessian
                    self.accumulate_dynamic_q_core_hessian(
                        row,
                        &slices,
                        &q_geom,
                        gamma.view(),
                        q_de.view(),
                        &mut acc,
                    )?;
                    for (primary_range, joint_range) in &identity_blocks {
                        for local in 0..primary_range.len() {
                            self.accumulate_identity_primary_cross_hessian(
                                row,
                                &slices,
                                &q_geom,
                                q_de.slice(s![0..N_PRIMARY, primary_range.start + local]),
                                joint_range,
                                local,
                                &mut acc,
                            )?;
                        }
                        self.add_dense_submatrix(
                            &mut acc,
                            joint_range,
                            joint_range,
                            q_de.slice(s![primary_range.clone(), primary_range.clone()]),
                        );
                    }
                    for li in 0..identity_blocks.len() {
                        for ri in li + 1..identity_blocks.len() {
                            let (lp, lj) = &identity_blocks[li];
                            let (rp, rj) = &identity_blocks[ri];
                            self.add_dense_symmetric_cross_submatrix(
                                &mut acc,
                                lj,
                                rj,
                                q_de.slice(s![lp.clone(), rp.clone()]),
                            );
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || Array2::<f64>::zeros((p_total, p_total)),
                |mut a, b| -> Result<_, String> {
                    a += &b;
                    Ok(a)
                },
            )?;
        Ok(result)
    }

    fn evaluate_blockwise_exact_newton(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if self.per_z_logslope_active() {
            return self.evaluate_blockwise_exact_newton_per_z(block_states);
        }
        if self.effective_flex_active(block_states)? {
            return self.evaluate_blockwise_exact_newton_flexible(block_states);
        }
        if self.flex_timewiggle_active() {
            return self.evaluate_blockwise_exact_newton_timewiggle(block_states);
        }

        // For all non-flex, non-timewiggle modes: use the dense joint path
        // when p is small enough.  This guarantees block Hessians are
        // principal blocks of the joint Hessian regardless of whether the
        // underlying designs happen to be sparse.
        let slices = block_slices(self, block_states);
        if slices.total < 512 {
            return self.evaluate_blockwise_exact_newton_dense(block_states);
        }

        // Large p (>= 512): joint dense Hessian is too expensive.
        // Fall back to blockwise sparse/mixed assembly for memory efficiency.
        let time_csrs = match (
            self.design_entry.as_sparse(),
            self.design_exit.as_sparse(),
            self.design_derivative_exit.as_sparse(),
        ) {
            (Some(e), Some(x), Some(d)) => Some((
                e.to_csr_arc().expect("entry CSR"),
                x.to_csr_arc().expect("exit CSR"),
                d.to_csr_arc().expect("deriv CSR"),
            )),
            _ => None,
        };
        let marginal_csr = self
            .marginal_design
            .as_sparse()
            .and_then(|s| s.to_csr_arc());
        let logslope_csr = self
            .logslope_design
            .as_sparse()
            .and_then(|s| s.to_csr_arc());

        let time_sparse = time_csrs.is_some();
        let marginal_sparse = marginal_csr.is_some();
        let logslope_sparse = logslope_csr.is_some();

        if time_sparse && marginal_sparse && logslope_sparse {
            self.evaluate_blockwise_exact_newton_sparse(
                block_states,
                &time_csrs.unwrap(),
                &marginal_csr.unwrap(),
                &logslope_csr.unwrap(),
            )
        } else if !time_sparse && !marginal_sparse && !logslope_sparse {
            self.evaluate_blockwise_exact_newton_dense(block_states)
        } else {
            self.evaluate_blockwise_exact_newton_mixed(
                block_states,
                time_csrs.as_ref(),
                marginal_csr.as_ref(),
                logslope_csr.as_ref(),
            )
        }
    }

    fn evaluate_blockwise_exact_newton_per_z(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            return Err(
                "survival marginal-slope per-z logslope surfaces currently require the rigid row kernel"
                    .to_string(),
            );
        }
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let beta_time = &block_states[0].beta;
        let beta_logslope = &block_states[2].beta;
        let probit_scale = self.probit_frailty_scale();
        type PerZBlockAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
        );
        let (ll, grad_t, grad_m, grad_g, hess_t, hess_m, hess_g): PerZBlockAcc = (0..self.n)
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(p_t),
                        Array1::<f64>::zeros(p_m),
                        Array1::<f64>::zeros(p_g),
                        Array2::<f64>::zeros((p_t, p_t)),
                        Array2::<f64>::zeros((p_m, p_m)),
                        Array2::<f64>::zeros((p_g, p_g)),
                    )
                },
                |mut acc, row| -> Result<_, String> {
                    let q0 = self.design_entry.dot_row(row, beta_time)
                        + self.offset_entry[row]
                        + block_states[1].eta[row];
                    let q1 = self.design_exit.dot_row(row, beta_time)
                        + self.offset_exit[row]
                        + block_states[1].eta[row];
                    let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                        + self.derivative_offset_exit[row];
                    let slopes = self.logslope_surface_values_for_row(row, beta_logslope)?;
                    let z = self.z.row(row).to_vec();
                    let (nll, f_pi, f_pipi) = row_primary_closed_form_vector(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &self.score_covariance,
                        self.weights[row],
                        self.event[row],
                        self.derivative_guard,
                        probit_scale,
                    )?;
                    acc.0 -= nll;
                    self.design_entry
                        .axpy_row_into(row, -f_pi[0], &mut acc.1.view_mut())?;
                    self.design_exit
                        .axpy_row_into(row, -f_pi[1], &mut acc.1.view_mut())?;
                    self.design_derivative_exit.axpy_row_into(
                        row,
                        -f_pi[2],
                        &mut acc.1.view_mut(),
                    )?;
                    self.marginal_design.axpy_row_into(
                        row,
                        -(f_pi[0] + f_pi[1]),
                        &mut acc.2.view_mut(),
                    )?;
                    let g_row = self.logslope_surface_row(row)?;
                    for (coord, range) in self.logslope_surface_ranges.iter().enumerate() {
                        let alpha = -f_pi[3 + coord];
                        for col in range.clone() {
                            acc.3[col] += alpha * g_row[col];
                        }
                    }
                    let time_designs = [
                        &self.design_entry,
                        &self.design_exit,
                        &self.design_derivative_exit,
                    ];
                    for a in 0..3 {
                        for b in 0..3 {
                            time_designs[a].row_outer_into(
                                row,
                                time_designs[b],
                                f_pipi[[a, b]],
                                &mut acc.4,
                            )?;
                        }
                    }
                    let alpha_mm =
                        f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    self.marginal_design
                        .syr_row_into(row, alpha_mm, &mut acc.5)?;
                    for (a, range_a) in self.logslope_surface_ranges.iter().enumerate() {
                        for (b, range_b) in self.logslope_surface_ranges.iter().enumerate() {
                            let alpha = f_pipi[[3 + a, 3 + b]];
                            if alpha == 0.0 {
                                continue;
                            }
                            for ca in range_a.clone() {
                                let va = g_row[ca] * alpha;
                                for cb in range_b.clone() {
                                    acc.6[[ca, cb]] += va * g_row[cb];
                                }
                            }
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(p_t),
                        Array1::<f64>::zeros(p_m),
                        Array1::<f64>::zeros(p_g),
                        Array2::<f64>::zeros((p_t, p_t)),
                        Array2::<f64>::zeros((p_m, p_m)),
                        Array2::<f64>::zeros((p_g, p_g)),
                    )
                },
                |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4 += &b.4;
                    a.5 += &b.5;
                    a.6 += &b.6;
                    Ok(a)
                },
            )?;
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_t,
                    hessian: SymmetricMatrix::Dense(hess_t),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_m,
                    hessian: SymmetricMatrix::Dense(hess_m),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_g,
                    hessian: SymmetricMatrix::Dense(hess_g),
                },
            ],
        })
    }

    fn evaluate_exact_newton_joint_dense_per_z(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let slices = block_slices(self, block_states);
        let total = slices.total;
        let k = self.score_dim();
        let dim = 3 + k;
        let beta_time = &block_states[0].beta;
        let beta_logslope = &block_states[2].beta;
        let probit_scale = self.probit_frailty_scale();
        type PerZJointAcc = (f64, Array1<f64>, Array2<f64>);
        let (ll, grad, hess): PerZJointAcc = (0..self.n)
            .into_par_iter()
            .try_fold(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(total),
                        Array2::<f64>::zeros((total, total)),
                    )
                },
                |mut acc, row| -> Result<_, String> {
                    let q0 = self.design_entry.dot_row(row, beta_time)
                        + self.offset_entry[row]
                        + block_states[1].eta[row];
                    let q1 = self.design_exit.dot_row(row, beta_time)
                        + self.offset_exit[row]
                        + block_states[1].eta[row];
                    let qd1 = self.design_derivative_exit.dot_row(row, beta_time)
                        + self.derivative_offset_exit[row];
                    let slopes = self.logslope_surface_values_for_row(row, beta_logslope)?;
                    let z = self.z.row(row).to_vec();
                    let (nll, f_pi, f_pipi) = row_primary_closed_form_vector(
                        q0,
                        q1,
                        qd1,
                        &slopes,
                        &z,
                        &self.score_covariance,
                        self.weights[row],
                        self.event[row],
                        self.derivative_guard,
                        probit_scale,
                    )?;
                    acc.0 -= nll;
                    let mut j = Array2::<f64>::zeros((dim, total));
                    let entry = self.design_entry.try_row_chunk(row..row + 1).map_err(|e| {
                        format!("evaluate_exact_newton_joint_dense_per_z entry row: {e}")
                    })?;
                    let exit = self.design_exit.try_row_chunk(row..row + 1).map_err(|e| {
                        format!("evaluate_exact_newton_joint_dense_per_z exit row: {e}")
                    })?;
                    let deriv = self
                        .design_derivative_exit
                        .try_row_chunk(row..row + 1)
                        .map_err(|e| {
                            format!("evaluate_exact_newton_joint_dense_per_z derivative row: {e}")
                        })?;
                    let marginal =
                        self.marginal_design
                            .try_row_chunk(row..row + 1)
                            .map_err(|e| {
                                format!("evaluate_exact_newton_joint_dense_per_z marginal row: {e}")
                            })?;
                    j.slice_mut(s![0, slices.time.clone()])
                        .assign(&entry.row(0));
                    j.slice_mut(s![1, slices.time.clone()]).assign(&exit.row(0));
                    j.slice_mut(s![2, slices.time.clone()])
                        .assign(&deriv.row(0));
                    j.slice_mut(s![0, slices.marginal.clone()])
                        .assign(&marginal.row(0));
                    j.slice_mut(s![1, slices.marginal.clone()])
                        .assign(&marginal.row(0));
                    let g_row = self.logslope_surface_row(row)?;
                    for (coord, range) in self.logslope_surface_ranges.iter().enumerate() {
                        let global_range = (slices.logslope.start + range.start)
                            ..(slices.logslope.start + range.end);
                        j.slice_mut(s![3 + coord, global_range])
                            .assign(&g_row.slice(s![range.clone()]));
                    }
                    for a in 0..dim {
                        for col in 0..total {
                            acc.1[col] -= f_pi[a] * j[[a, col]];
                        }
                    }
                    for a in 0..dim {
                        for b in 0..dim {
                            let alpha = f_pipi[[a, b]];
                            if alpha == 0.0 {
                                continue;
                            }
                            for ca in 0..total {
                                let va = j[[a, ca]] * alpha;
                                if va == 0.0 {
                                    continue;
                                }
                                for cb in 0..total {
                                    acc.2[[ca, cb]] += va * j[[b, cb]];
                                }
                            }
                        }
                    }
                    Ok(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        0.0,
                        Array1::<f64>::zeros(total),
                        Array2::<f64>::zeros((total, total)),
                    )
                },
                |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    Ok(a)
                },
            )?;
        Ok((ll, grad, hess))
    }

    /// Blockwise exact-Newton for the flexible (score-warp / link-deviation)
    /// model.
    ///
    /// Accumulates exact per-block coefficient gradients and Hessians directly
    /// from the dynamic-q row jets. This preserves the exact block Newton
    /// update while avoiding dense full-joint assembly when the caller only
    /// needs block-local working sets.
    fn evaluate_blockwise_exact_newton_flexible(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        self.validate_exact_monotonicity(block_states)?;
        let primary = flex_primary_slices(self);
        self.evaluate_blockwise_exact_newton_dynamic_q(block_states, &primary, |row, q_geom| {
            self.compute_row_flex_primary_gradient_hessian_exact(
                row,
                block_states,
                q_geom,
                &primary,
            )
        })
    }

    /// Blockwise exact-Newton for the time-wiggle model.
    ///
    /// Accumulates exact block-local Hessians directly from the 4D primary
    /// row calculus instead of materializing and slicing a dense joint Hessian.
    fn evaluate_blockwise_exact_newton_timewiggle(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        let primary = flex_primary_slices(self);
        self.evaluate_blockwise_exact_newton_dynamic_q(block_states, &primary, |row, _| {
            self.compute_row_primary_gradient_hessian_uncached(row, block_states)
        })
    }

    fn evaluate_blockwise_exact_newton_mixed(
        &self,
        block_states: &[ParameterBlockState],
        time_csrs: Option<&(
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
        )>,
        marginal_csr: Option<&Arc<faer::sparse::SparseRowMat<usize, f64>>>,
        logslope_csr: Option<&Arc<faer::sparse::SparseRowMat<usize, f64>>>,
    ) -> Result<FamilyEvaluation, String> {
        use crate::matrix::SparseHessianAccumulator;

        enum BlockwiseHessianAccumulator {
            Dense(Array2<f64>),
            Sparse(SparseHessianAccumulator),
        }

        impl BlockwiseHessianAccumulator {
            fn add_assign(&mut self, other: &Self) {
                match (self, other) {
                    (Self::Dense(lhs), Self::Dense(rhs)) => *lhs += rhs,
                    (Self::Sparse(lhs), Self::Sparse(rhs)) => lhs.add_values(&rhs.values),
                    // Per-block accumulators all share one storage decision
                    // (marginal_csr / logslope_csr) made at the top of
                    // `family_evaluate_blockwise`.
                    // SAFETY: mismatch ⇒ a newly added partial picked the
                    // wrong storage variant — invariant violation.
                    _ => panic!("blockwise Hessian accumulator kind mismatch"),
                }
            }

            fn into_symmetric(self) -> SymmetricMatrix {
                match self {
                    Self::Dense(mat) => SymmetricMatrix::Dense(mat),
                    Self::Sparse(acc) => SymmetricMatrix::Sparse(acc.into_sparse_col_mat()),
                }
            }
        }

        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let time_pattern = time_csrs.map(|(entry, exit, deriv)| {
            SparseHessianAccumulator::from_multi_csr(
                &[entry.as_ref(), exit.as_ref(), deriv.as_ref()],
                p_t,
            )
        });
        let marginal_pattern =
            marginal_csr.map(|csr| SparseHessianAccumulator::from_single_csr(csr.as_ref(), p_m));
        let logslope_pattern =
            logslope_csr.map(|csr| SparseHessianAccumulator::from_single_csr(csr.as_ref(), p_g));

        let e_sparse = time_csrs.map(|(entry, _, _)| {
            let sym = entry.symbolic();
            (sym.row_ptr(), sym.col_idx(), entry.val())
        });
        let x_sparse = time_csrs.map(|(_, exit, _)| {
            let sym = exit.symbolic();
            (sym.row_ptr(), sym.col_idx(), exit.val())
        });
        let d_sparse = time_csrs.map(|(_, _, deriv)| {
            let sym = deriv.symbolic();
            (sym.row_ptr(), sym.col_idx(), deriv.val())
        });
        let m_sparse = marginal_csr.map(|csr| {
            let sym = csr.symbolic();
            (sym.row_ptr(), sym.col_idx(), csr.val())
        });
        let g_sparse = logslope_csr.map(|csr| {
            let sym = csr.symbolic();
            (sym.row_ptr(), sym.col_idx(), csr.val())
        });

        type MixedAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            BlockwiseHessianAccumulator,
            BlockwiseHessianAccumulator,
            BlockwiseHessianAccumulator,
        );

        let make_acc = || -> MixedAcc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                time_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_t, p_t))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
                marginal_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_m, p_m))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
                logslope_pattern.as_ref().map_or_else(
                    || BlockwiseHessianAccumulator::Dense(Array2::zeros((p_g, p_g))),
                    |pattern| BlockwiseHessianAccumulator::Sparse(pattern.empty_clone()),
                ),
            )
        };

        let (
            ll,
            grad_time,
            grad_marginal,
            grad_logslope,
            hess_time,
            hess_marginal,
            hess_logslope,
        ): MixedAcc = (0..self.n)
            .into_par_iter()
                .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    match &e_sparse {
                        Some((e_rp, e_ci, e_v)) => {
                            let gt = &mut acc.1;
                            for p in e_rp[row]..e_rp[row + 1] {
                                gt[e_ci[p]] -= f_pi[0] * e_v[p];
                            }
                            let (x_rp, x_ci, x_v) = x_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for exit design");
                            for p in x_rp[row]..x_rp[row + 1] {
                                gt[x_ci[p]] -= f_pi[1] * x_v[p];
                            }
                            let (d_rp, d_ci, d_v) = d_sparse.as_ref().expect(
                                "time sparse metadata should be present for derivative design",
                            );
                            for p in d_rp[row]..d_rp[row + 1] {
                                gt[d_ci[p]] -= f_pi[2] * d_v[p];
                            }
                        }
                        None => {
                            let mut time = acc.1.view_mut();
                            self.design_entry
                                .axpy_row_into(row, -f_pi[0], &mut time)
                                .expect("time entry axpy dim mismatch");
                            self.design_exit
                                .axpy_row_into(row, -f_pi[1], &mut time)
                                .expect("time exit axpy dim mismatch");
                            self.design_derivative_exit
                                .axpy_row_into(row, -f_pi[2], &mut time)
                                .expect("time deriv axpy dim mismatch");
                        }
                    }

                    match &m_sparse {
                        Some((m_rp, m_ci, m_v)) => {
                            let gm = &mut acc.2;
                            let alpha_m = -(f_pi[0] + f_pi[1]);
                            for p in m_rp[row]..m_rp[row + 1] {
                                gm[m_ci[p]] += alpha_m * m_v[p];
                            }
                        }
                        None => {
                            self.marginal_design
                                .axpy_row_into(row, -(f_pi[0] + f_pi[1]), &mut acc.2.view_mut())
                                .expect(
                                    "survival marginal block axpy should match block dimensions",
                                );
                        }
                    }

                    match &g_sparse {
                        Some((g_rp, g_ci, g_v)) => {
                            let gg = &mut acc.3;
                            for p in g_rp[row]..g_rp[row + 1] {
                                gg[g_ci[p]] -= f_pi[3] * g_v[p];
                            }
                        }
                        None => {
                            self.logslope_design
                                .axpy_row_into(row, -f_pi[3], &mut acc.3.view_mut())
                                .expect(
                                    "survival logslope block axpy should match block dimensions",
                                );
                        }
                    }

                    match &mut acc.4 {
                        BlockwiseHessianAccumulator::Dense(hess_time) => {
                            let designs = [
                                &self.design_entry,
                                &self.design_exit,
                                &self.design_derivative_exit,
                            ];
                            for a in 0..3 {
                                for b in 0..3 {
                                    designs[a]
                                        .row_outer_into(
                                            row,
                                            designs[b],
                                            f_pipi[[a, b]],
                                            &mut *hess_time,
                                        )
                                        .expect("time row_outer_into dim mismatch");
                                }
                            }
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_time) => {
                            let (e_rp, e_ci, e_v) = e_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for entry design");
                            let (x_rp, x_ci, x_v) = x_sparse
                                .as_ref()
                                .expect("time sparse metadata should be present for exit design");
                            let (d_rp, d_ci, d_v) = d_sparse.as_ref().expect(
                                "time sparse metadata should be present for derivative design",
                            );
                            let row_slices: [(std::ops::Range<usize>, &[usize], &[f64]); 3] = [
                                (e_rp[row]..e_rp[row + 1], e_ci, e_v),
                                (x_rp[row]..x_rp[row + 1], x_ci, x_v),
                                (d_rp[row]..d_rp[row + 1], d_ci, d_v),
                            ];
                            for a in 0..3 {
                                for b in 0..3 {
                                    let alpha = f_pipi[[a, b]];
                                    if alpha == 0.0 {
                                        continue;
                                    }
                                    let (ref ra, cia, va) = row_slices[a];
                                    let (ref rb, cib, vb) = row_slices[b];
                                    for pi in ra.clone() {
                                        let ca = cia[pi];
                                        let xia = va[pi] * alpha;
                                        for pj in rb.clone() {
                                            let cb = cib[pj];
                                            if ca <= cb {
                                                hess_time.add_upper(ca, cb, xia * vb[pj]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let alpha_m = f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    match &mut acc.5 {
                        BlockwiseHessianAccumulator::Dense(hess_marginal) => {
                            self.marginal_design
                                .syr_row_into(row, alpha_m, &mut *hess_marginal)
                                .expect(
                                    "survival marginal block syr should match block dimensions",
                                );
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_marginal) => {
                            if alpha_m != 0.0 {
                                let (m_rp, m_ci, m_v) = m_sparse.as_ref().expect(
                                    "marginal sparse metadata should be present for sparse block",
                                );
                                for pi in m_rp[row]..m_rp[row + 1] {
                                    let ca = m_ci[pi];
                                    let xia = m_v[pi] * alpha_m;
                                    for pj in m_rp[row]..m_rp[row + 1] {
                                        let cb = m_ci[pj];
                                        if ca <= cb {
                                            hess_marginal.add_upper(ca, cb, xia * m_v[pj]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    let alpha_g = f_pipi[[3, 3]];
                    match &mut acc.6 {
                        BlockwiseHessianAccumulator::Dense(hess_logslope) => {
                            self.logslope_design
                                .syr_row_into(row, alpha_g, &mut *hess_logslope)
                                .expect(
                                    "survival logslope block syr should match block dimensions",
                                );
                        }
                        BlockwiseHessianAccumulator::Sparse(hess_logslope) => {
                            if alpha_g != 0.0 {
                                let (g_rp, g_ci, g_v) = g_sparse.as_ref().expect(
                                    "logslope sparse metadata should be present for sparse block",
                                );
                                for pi in g_rp[row]..g_rp[row + 1] {
                                    let ca = g_ci[pi];
                                    let xia = g_v[pi] * alpha_g;
                                    for pj in g_rp[row]..g_rp[row + 1] {
                                        let cb = g_ci[pj];
                                        if ca <= cb {
                                            hess_logslope.add_upper(ca, cb, xia * g_v[pj]);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4.add_assign(&b.4);
                    a.5.add_assign(&b.5);
                    a.6.add_assign(&b.6);
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: {
                let slices = block_slices(self, block_states);
                let mut sets = vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_time,
                        hessian: hess_time.into_symmetric(),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: hess_marginal.into_symmetric(),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: hess_logslope.into_symmetric(),
                    },
                ];
                if let Some(range) = slices.score_warp.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                if let Some(range) = slices.link_dev.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                sets
            },
        })
    }

    // ── Dense path (original) ────────────────────────────────────────

    fn evaluate_blockwise_exact_newton_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<FamilyEvaluation, String> {
        // Build RowKernel — the single source of truth for all exact-Newton
        // quantities.  The cache evaluates every row kernel once and stores
        // (nll_i, g_i[4], H_i[4×4]).
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let rows = crate::families::row_kernel::RowSet::All;
        let cache = build_row_kernel_cache(&kern, &rows)?;

        let ll = row_kernel_log_likelihood(&cache, &rows);

        // Joint gradient:  g = -Σ_i Jᵢᵀ gᵢ  (sign: gᵢ are NLL gradients,
        // we negate to get log-likelihood gradient).
        let nll_grad = row_kernel_gradient(&kern, &cache, &rows);
        let joint_gradient = -nll_grad;

        // Block-diagonal Hessians only — the inner solver consumes per-block
        // working sets, so we accumulate the principal time-time, m-m, and
        // g-g blocks directly instead of building the full joint Hessian and
        // slicing.  Cost falls from Θ(n·(p_t+p_m+p_g)²) to
        // Θ(n·(p_t²+p_m²+p_g²)).
        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();
        let mut hess_time = Array2::<f64>::zeros((p_t, p_t));
        let mut hess_marginal = Array2::<f64>::zeros((p_m, p_m));
        let mut hess_logslope = Array2::<f64>::zeros((p_g, p_g));
        for row in 0..cache.n {
            let h = &cache.hessians[row];
            let mut h_arr = Array2::<f64>::zeros((4, 4));
            for a in 0..4 {
                for b in 0..4 {
                    h_arr[[a, b]] = h[a][b];
                }
            }
            self.add_pullback_block_diagonals(
                row,
                &h_arr,
                &mut hess_time,
                &mut hess_marginal,
                &mut hess_logslope,
            );
        }

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.time.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.marginal.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_marginal),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![slices.logslope.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_logslope),
            },
        ];
        if let Some(range) = slices.score_warp.as_ref() {
            // The 4-D row kernel does not span score_warp / link_dev primary
            // dimensions, so these blocks contribute zero gradient/Hessian
            // here — exactly what the joint-then-slice path produced.
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
            });
        }
        if let Some(range) = slices.link_dev.as_ref() {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range.clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    // ── Sparse path ──────────────────────────────────────────────────

    fn evaluate_blockwise_exact_newton_sparse(
        &self,
        block_states: &[ParameterBlockState],
        time_csrs: &(
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
            Arc<faer::sparse::SparseRowMat<usize, f64>>,
        ),
        marginal_csr: &Arc<faer::sparse::SparseRowMat<usize, f64>>,
        logslope_csr: &Arc<faer::sparse::SparseRowMat<usize, f64>>,
    ) -> Result<FamilyEvaluation, String> {
        use crate::matrix::SparseHessianAccumulator;

        let slices = block_slices(self, block_states);
        let p_t = slices.time.len();
        let p_m = slices.marginal.len();
        let p_g = slices.logslope.len();

        let (ref csr_entry, ref csr_exit, ref csr_deriv) = *time_csrs;

        // Build symbolic sparsity patterns once.
        let pattern_time = SparseHessianAccumulator::from_multi_csr(
            &[csr_entry.as_ref(), csr_exit.as_ref(), csr_deriv.as_ref()],
            p_t,
        );
        let pattern_marginal =
            SparseHessianAccumulator::from_single_csr(marginal_csr.as_ref(), p_m);
        let pattern_logslope =
            SparseHessianAccumulator::from_single_csr(logslope_csr.as_ref(), p_g);

        // Pre-extract CSR symbolic parts for zero-overhead inner loop access.
        let e_sym = csr_entry.symbolic();
        let e_rp = e_sym.row_ptr();
        let e_ci = e_sym.col_idx();
        let e_v = csr_entry.val();

        let x_sym = csr_exit.symbolic();
        let x_rp = x_sym.row_ptr();
        let x_ci = x_sym.col_idx();
        let x_v = csr_exit.val();

        let d_sym = csr_deriv.symbolic();
        let d_rp = d_sym.row_ptr();
        let d_ci = d_sym.col_idx();
        let d_v = csr_deriv.val();

        let m_sym = marginal_csr.symbolic();
        let m_rp = m_sym.row_ptr();
        let m_ci = m_sym.col_idx();
        let m_v = marginal_csr.val();

        let g_sym = logslope_csr.symbolic();
        let g_rp = g_sym.row_ptr();
        let g_ci = g_sym.col_idx();
        let g_v = logslope_csr.val();

        // Accumulator type: gradients dense, Hessians sparse value buffers.
        type SAcc = (
            f64,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            SparseHessianAccumulator,
            SparseHessianAccumulator,
            SparseHessianAccumulator,
        );

        let make_acc = || -> SAcc {
            (
                0.0,
                Array1::zeros(p_t),
                Array1::zeros(p_m),
                Array1::zeros(p_g),
                pattern_time.empty_clone(),
                pattern_marginal.empty_clone(),
                pattern_logslope.empty_clone(),
            )
        };

        let (
            ll,
            grad_time,
            grad_marginal,
            grad_logslope,
            acc_time,
            acc_marginal,
            acc_logslope,
        ): SAcc = (0..self.n)
            .into_par_iter()
            .try_fold(make_acc, |mut acc, row| -> Result<_, String> {
                    let (row_nll, f_pi, f_pipi) =
                        self.compute_row_primary_gradient_hessian_uncached(row, block_states)?;
                    acc.0 -= row_nll;

                    // ── gradients (dense axpy via CSR scatter) ────────────
                    {
                        let gt = &mut acc.1;
                        for p in e_rp[row]..e_rp[row + 1] {
                            gt[e_ci[p]] -= f_pi[0] * e_v[p];
                        }
                        for p in x_rp[row]..x_rp[row + 1] {
                            gt[x_ci[p]] -= f_pi[1] * x_v[p];
                        }
                        for p in d_rp[row]..d_rp[row + 1] {
                            gt[d_ci[p]] -= f_pi[2] * d_v[p];
                        }
                    }
                    {
                        let gm = &mut acc.2;
                        let alpha_m = -(f_pi[0] + f_pi[1]);
                        for p in m_rp[row]..m_rp[row + 1] {
                            gm[m_ci[p]] += alpha_m * m_v[p];
                        }
                    }
                    {
                        let gg = &mut acc.3;
                        for p in g_rp[row]..g_rp[row + 1] {
                            gg[g_ci[p]] -= f_pi[3] * g_v[p];
                        }
                    }

                    // ── time Hessian: 3×3 cross-product scatter ──────────
                    // Only emit upper-triangle entries (ca <= cb) to avoid
                    // double-counting: SymmetricMatrix::Sparse mirrors the
                    // upper triangle into the lower.
                    let row_slices: [(std::ops::Range<usize>, &[usize], &[f64]); 3] = [
                        (e_rp[row]..e_rp[row + 1], e_ci, e_v),
                        (x_rp[row]..x_rp[row + 1], x_ci, x_v),
                        (d_rp[row]..d_rp[row + 1], d_ci, d_v),
                    ];
                    let ht = &mut acc.4;
                    for a in 0..3 {
                        for b in 0..3 {
                            let alpha = f_pipi[[a, b]];
                            if alpha == 0.0 {
                                continue;
                            }
                            let (ref ra, cia, va) = row_slices[a];
                            let (ref rb, cib, vb) = row_slices[b];
                            for pi in ra.clone() {
                                let ca = cia[pi];
                                let xia = va[pi] * alpha;
                                for pj in rb.clone() {
                                    let cb = cib[pj];
                                    if ca <= cb {
                                        ht.add_upper(ca, cb, xia * vb[pj]);
                                    }
                                }
                            }
                        }
                    }

                    // ── marginal Hessian: symmetric rank-1 scatter ───────
                    let alpha_m = f_pipi[[0, 0]] + f_pipi[[0, 1]] + f_pipi[[1, 0]] + f_pipi[[1, 1]];
                    if alpha_m != 0.0 {
                        let hm = &mut acc.5;
                        let m_start = m_rp[row];
                        let m_end = m_rp[row + 1];
                        for pi in m_start..m_end {
                            let ca = m_ci[pi];
                            let xia = m_v[pi] * alpha_m;
                            for pj in m_start..m_end {
                                let cb = m_ci[pj];
                                if ca <= cb {
                                    hm.add_upper(ca, cb, xia * m_v[pj]);
                                }
                            }
                        }
                    }

                    // ── logslope Hessian: symmetric rank-1 scatter ───────
                    let alpha_g = f_pipi[[3, 3]];
                    if alpha_g != 0.0 {
                        let hg = &mut acc.6;
                        let g_start = g_rp[row];
                        let g_end = g_rp[row + 1];
                        for pi in g_start..g_end {
                            let ca = g_ci[pi];
                            let xia = g_v[pi] * alpha_g;
                            for pj in g_start..g_end {
                                let cb = g_ci[pj];
                                if ca <= cb {
                                    hg.add_upper(ca, cb, xia * g_v[pj]);
                                }
                            }
                        }
                    }

                    Ok(acc)
                })
                .try_reduce(make_acc, |mut a, b| -> Result<_, String> {
                    a.0 += b.0;
                    a.1 += &b.1;
                    a.2 += &b.2;
                    a.3 += &b.3;
                    a.4.add_values(&b.4.values);
                    a.5.add_values(&b.5.values);
                    a.6.add_values(&b.6.values);
                    Ok(a)
                })?;

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: {
                let slices = block_slices(self, block_states);
                let mut sets = vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_time,
                        hessian: SymmetricMatrix::Sparse(acc_time.into_sparse_col_mat()),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_marginal,
                        hessian: SymmetricMatrix::Sparse(acc_marginal.into_sparse_col_mat()),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: grad_logslope,
                        hessian: SymmetricMatrix::Sparse(acc_logslope.into_sparse_col_mat()),
                    },
                ];
                if let Some(range) = slices.score_warp.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                if let Some(range) = slices.link_dev.as_ref() {
                    sets.push(BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(range.len()),
                        hessian: SymmetricMatrix::Dense(Array2::zeros((range.len(), range.len()))),
                    });
                }
                sets
            },
        })
    }
}

// ── CustomFamily impl ─────────────────────────────────────────────────

fn time_wiggle_basis_ncols(knots: &Array1<f64>, degree: usize) -> Result<usize, String> {
    if knots.is_empty() {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope timewiggle requires at least one knot".to_string(),
        }
        .into());
    }
    let probe = 0.5 * (knots[0] + knots[knots.len() - 1]);
    let h0 = Array1::from_vec(vec![probe]);
    Ok(monotone_wiggle_basis_with_derivative_order(h0.view(), knots, degree, 0)?.ncols())
}

impl CustomFamily for SurvivalMarginalSlopeFamily {
    fn persistent_warm_start_fingerprint(
        &self,
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Option<String> {
        drop(specs);
        drop(options);
        let mut hasher = crate::solver::persistent_warm_start::StableHasher::new();
        hasher.write_str("survival-marginal-slope-family");
        hasher.write_usize(self.n);
        hasher.write_usize(self.event.len());
        for &value in self.event.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.weights.len());
        for &value in self.weights.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.z.nrows());
        hasher.write_usize(self.z.ncols());
        for &value in self.z.iter() {
            hasher.write_f64(value);
        }
        match self.gaussian_frailty_sd {
            Some(value) => {
                hasher.write_bool(true);
                hasher.write_f64(value);
            }
            None => hasher.write_bool(false),
        }
        hasher.write_f64(self.derivative_guard);
        hasher.write_usize(self.offset_entry.len());
        for &value in self.offset_entry.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.offset_exit.len());
        for &value in self.offset_exit.iter() {
            hasher.write_f64(value);
        }
        hasher.write_usize(self.derivative_offset_exit.len());
        for &value in self.derivative_offset_exit.iter() {
            hasher.write_f64(value);
        }
        Some(hasher.finish_hex())
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // Operator-aware: the rigid K=4 RowKernel + RowKernelHessianWorkspace
        // adapter (see `exact_newton_joint_hessian_workspace`) applies joint
        // Hv at O(n · (p_time + p_marginal + p_logslope + p_flex)) per call.
        // Report the operator work model so diagnostics and first-order-only
        // policies reflect the representation that actually executes.
        let n = self.n as u64;
        let p_total: u64 = specs
            .iter()
            .map(|s| s.design.ncols() as u64)
            .fold(0u64, |a, p| a.saturating_add(p));
        if crate::custom_family::use_joint_matrix_free_path(p_total as usize, n as usize) {
            n.saturating_mul(p_total)
        } else {
            crate::custom_family::joint_coupled_coefficient_hessian_cost(n, specs)
        }
    }

    fn outer_derivative_policy(
        &self,
        specs: &[ParameterBlockSpec],
        psi_dim: usize,
        options: &BlockwiseFitOptions,
    ) -> crate::custom_family::OuterDerivativePolicy {
        use crate::custom_family::OuterDerivativePolicy;

        let capability = self.exact_outer_derivative_order(specs, options);
        let rho_dim = specs
            .iter()
            .map(|spec| spec.penalties.len() as u128)
            .sum::<u128>();
        let k = rho_dim.saturating_add(psi_dim as u128).max(1);

        let predicted_hessian_work = if !self.flex_active() && !self.flex_timewiggle_active() {
            // Rigid survival marginal-slope evaluates outer rho/psi
            // coordinate corrections and projected logdet traces through
            // row-kernel/HVP paths.  The projected subspace trace reductions
            // are batched in `reml::unified`, so the shared X·U_S work is
            // paid once per derivative group rather than once per coordinate.
            // Model the work that actually executes: one row-kernel pass per
            // outer coordinate and coefficient axis, plus the fixed four
            // primary survival channels.
            let p_total = specs
                .iter()
                .map(|spec| spec.design.ncols() as u128)
                .sum::<u128>();
            (self.n as u128)
                .saturating_mul(k)
                .saturating_mul(p_total.saturating_add(N_PRIMARY as u128))
        } else {
            // Flex/time-wiggle survival paths have higher-order dynamic-q
            // row geometry. Keep the generic dense policy there until those
            // paths have their own measured row-work model.
            let (gradient_work, hessian_work) =
                crate::custom_family::default_outer_derivative_policy_costs(
                    specs,
                    psi_dim,
                    self.coefficient_gradient_cost(specs),
                    self.coefficient_hessian_cost(specs),
                );
            return OuterDerivativePolicy {
                capability,
                predicted_gradient_work: gradient_work,
                predicted_hessian_work: hessian_work,
                subsample_capable: true,
            };
        };

        OuterDerivativePolicy {
            capability,
            predicted_gradient_work: predicted_hessian_work / 2,
            predicted_hessian_work,
            subsample_capable: true,
        }
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.evaluate_blockwise_exact_newton(block_states)
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let options = BlockwiseFitOptions {
            auto_outer_subsample: false,
            ..BlockwiseFitOptions::default()
        };
        SurvivalMarginalSlopeFamily::log_likelihood_only_with_options(self, block_states, &options)
    }

    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        let owned;
        let options: &BlockwiseFitOptions =
            match self.install_auto_outer_subsample_options(options, block_states) {
                Some(cloned) => {
                    owned = cloned;
                    &owned
                }
                None => options,
            };
        SurvivalMarginalSlopeFamily::log_likelihood_only_with_options(self, block_states, options)
    }

    fn joint_line_search_log_likelihood_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        line_search_options: &BlockwiseFitOptions,
        workspace_options: &BlockwiseFitOptions,
    ) -> Result<Option<(f64, Arc<dyn ExactNewtonJointHessianWorkspace>)>, String> {
        drop(line_search_options);
        if self.per_z_logslope_active()
            || self.effective_flex_active(block_states)?
            || self.flex_timewiggle_active()
        {
            return Ok(None);
        }
        let Some(workspace) = self.exact_newton_joint_hessian_workspace_with_options(
            block_states,
            specs,
            workspace_options,
        )?
        else {
            return Ok(None);
        };
        let Some(log_likelihood) = workspace.joint_log_likelihood_evaluation()? else {
            return Ok(None);
        };
        Ok(Some((log_likelihood, workspace)))
    }

    fn joint_line_search_log_likelihood_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        line_search_options: &BlockwiseFitOptions,
        workspace_options: &BlockwiseFitOptions,
    ) -> Result<Option<(f64, Option<Arc<dyn ExactNewtonJointHessianWorkspace>>)>, String> {
        self.joint_line_search_log_likelihood_workspace(
            block_states,
            specs,
            line_search_options,
            workspace_options,
        )
        .map(|maybe| maybe.map(|(log_likelihood, workspace)| (log_likelihood, Some(workspace))))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let total = block_states
            .iter()
            .map(|state| state.beta.len())
            .sum::<usize>();
        if total >= 512 {
            return Ok(None);
        }
        if self.per_z_logslope_active() {
            return Ok(Some(
                self.evaluate_exact_newton_joint_dense_per_z(block_states)?
                    .2,
            ));
        }
        Ok(Some(
            self.evaluate_exact_newton_joint_dense(block_states)?.2,
        ))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        if self.per_z_logslope_active() {
            let (log_likelihood, gradient, _) =
                self.evaluate_exact_newton_joint_dense_per_z(block_states)?;
            return Ok(Some(ExactNewtonJointGradientEvaluation {
                log_likelihood,
                gradient,
            }));
        }
        if self.effective_flex_active(block_states)? || self.flex_timewiggle_active() {
            let (log_likelihood, gradient) =
                self.evaluate_exact_newton_joint_gradient_dynamic_q(block_states)?;
            return Ok(Some(ExactNewtonJointGradientEvaluation {
                log_likelihood,
                gradient,
            }));
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let rows = crate::families::row_kernel::RowSet::All;
        let cache = build_row_kernel_cache(&kern, &rows)?;
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: row_kernel_log_likelihood(&cache, &rows),
            gradient: -row_kernel_gradient(&kern, &cache, &rows),
        }))
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if !self.effective_flex_active(block_states)? && !self.flex_timewiggle_active() {
            let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
            return Ok(Some(Arc::new(RowKernelHessianWorkspace::new(kern)?)));
        }
        Ok(Some(Arc::new(
            SurvivalMarginalSlopeExactNewtonJointHessianWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                BlockwiseFitOptions::default(),
            )?,
        )))
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if !self.effective_flex_active(block_states)? && !self.flex_timewiggle_active() {
            // Rigid path: RowKernel<4> operator wired through the supplied
            // `RowSet`. The cache and every assembly function honour the
            // mask uniformly through the Horvitz–Thompson weights on each
            // `WeightedOuterRow`.
            let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
            let rows = crate::families::row_kernel::RowSet::from_options(options, self.n);
            return Ok(Some(Arc::new(RowKernelHessianWorkspace::with_rows(
                kern, rows,
            )?)));
        }
        // Flex / timewiggle path: store the options on the workspace so the
        // directional-derivative methods can pick up the outer-row subsample.
        let owned;
        let options: &BlockwiseFitOptions =
            match self.install_auto_outer_subsample_options(options, block_states) {
                Some(cloned) => {
                    owned = cloned;
                    &owned
                }
                None => options,
            };
        Ok(Some(Arc::new(
            SurvivalMarginalSlopeExactNewtonJointHessianWorkspace::new(
                self.clone(),
                block_states.to_vec(),
                options.clone(),
            )?,
        )))
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        drop(specs);
        // The workspace impl above unconditionally returns `Some(workspace)`
        // — the rigid path produces a `RowKernelHessianWorkspace` and the
        // flex path produces a
        // `SurvivalMarginalSlopeExactNewtonJointHessianWorkspace`. Both
        // route the joint Hessian through Hv operators rather than dense
        // assembly.
        !self.per_z_logslope_active()
    }

    fn outer_hyper_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        drop(specs);
        // The exact outer Hessian over θ=(ρ,ψ[,log σ]) can be applied without
        // pairwise θθ materialization: coefficient-Hessian drift terms use the
        // joint-Hessian workspace's directional-derivative operators, and ψ
        // drift terms use `SurvivalMarginalSlopePsiWorkspace` to return
        // `DriftDerivResult::Operator`. Advertising this capability lets the
        // outer planner keep ARC/Newton curvature at large n or large ψ_dim
        // while routing the representation through matrix-free HVPs.
        !self.per_z_logslope_active()
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if self.effective_flex_active(block_states)? {
            return if self.flex_timewiggle_active() {
                self.exact_newton_joint_hessian_directional_derivative_timewiggle_flex(
                    block_states,
                    d_beta_flat,
                )
            } else {
                self.exact_newton_joint_hessian_directional_derivative_flex_no_wiggle(
                    block_states,
                    d_beta_flat,
                )
            }
            .map(Some);
        }
        if self.flex_timewiggle_active() {
            return self
                .exact_newton_joint_hessian_directional_derivative_timewiggle(
                    block_states,
                    d_beta_flat,
                )
                .map(Some);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let sl = d_beta_flat.as_slice().ok_or("non-contiguous d_beta")?;
        crate::families::row_kernel::row_kernel_directional_derivative(
            &kern,
            &crate::families::row_kernel::RowSet::All,
            sl,
        )
        .map(Some)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        if self.effective_flex_active(block_states)? {
            if self.flex_timewiggle_active() {
                return self
                    .exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
                        block_states,
                        d_beta_u_flat,
                        d_beta_v_flat,
                    )
                    .map(Some);
            }
            return self
                .exact_newton_joint_hessiansecond_directional_derivative_flex_no_wiggle(
                    block_states,
                    d_beta_u_flat,
                    d_beta_v_flat,
                )
                .map(Some);
        }
        if self.flex_timewiggle_active() {
            return self
                .exact_newton_joint_hessiansecond_directional_derivative_timewiggle(
                    block_states,
                    d_beta_u_flat,
                    d_beta_v_flat,
                )
                .map(Some);
        }
        let kern = SurvivalMarginalSlopeRowKernel::new(self.clone(), block_states.to_vec());
        let su = d_beta_u_flat.as_slice().ok_or("non-contiguous d_beta_u")?;
        let sv = d_beta_v_flat.as_slice().ok_or("non-contiguous d_beta_v")?;
        crate::families::row_kernel::row_kernel_second_directional_derivative(
            &kern,
            &crate::families::row_kernel::RowSet::All,
            su,
            sv,
        )
        .map(Some)
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self.sigma_exact_joint_psi_terms(block_states, specs);
        }
        self.psi_terms(block_states, derivative_blocks, psi_index)
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_i)
            || self.is_sigma_aux_index(derivative_blocks, psi_j)
        {
            if psi_i == psi_j {
                return self.sigma_exact_joint_psisecond_order_terms(block_states);
            }
            return Ok(None);
        }
        self.psi_second_order_terms(block_states, derivative_blocks, psi_i, psi_j)
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if self.is_sigma_aux_index(derivative_blocks, psi_index) {
            return self
                .sigma_exact_joint_psihessian_directional_derivative(block_states, d_beta_flat);
        }
        self.psi_hessian_directional_derivative(
            block_states,
            derivative_blocks,
            psi_index,
            d_beta_flat,
        )
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if self.per_z_logslope_active() {
            return Ok(None);
        }
        Ok(Some(Arc::new(SurvivalMarginalSlopePsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            derivative_blocks.to_vec(),
            BlockwiseFitOptions::default(),
        )?)))
    }

    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        let owned;
        let options: &BlockwiseFitOptions =
            match self.install_auto_outer_subsample_options(options, block_states) {
                Some(cloned) => {
                    owned = cloned;
                    &owned
                }
                None => options,
            };
        Ok(Some(Arc::new(SurvivalMarginalSlopePsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            specs.to_vec(),
            derivative_blocks.to_vec(),
            options.clone(),
        )?)))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == 0 {
            return Ok(self.time_linear_constraints.clone());
        }
        if self.score_warp.is_some() && block_idx == 3 {
            return Ok(self
                .score_warp
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        let link_block_idx = if self.score_warp.is_some() { 4 } else { 3 };
        if self.link_dev.is_some() && block_idx == link_block_idx {
            return Ok(self
                .link_dev
                .as_ref()
                .map(DeviationRuntime::structural_monotonicity_constraints));
        }
        Ok(None)
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_idx == 0 {
            return self.max_feasible_time_step(&block_states[0].beta, delta);
        }
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if block_idx >= block_states.len() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "post-update block index {} out of range for {} blocks",
                    block_idx,
                    block_states.len()
                ),
            }
            .into());
        }
        if block_idx == 0 {
            let proposed = if let Some(constraints) = self.time_linear_constraints.as_ref() {
                project_onto_linear_constraints(beta.len(), constraints, Some(&beta))
            } else {
                beta
            };
            let current = &block_states[0].beta;
            return self.project_time_qd1_feasible(current, proposed);
        }
        if self.score_warp.is_some() && block_idx == 3 {
            if let Some(runtime) = &self.score_warp {
                let current = &block_states[3].beta;
                if current.len() != beta.len() {
                    return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival score-warp post-update beta length mismatch: current={}, proposed={}",
                            current.len(),
                            beta.len()
                        ),
                    }
                    .into());
                }
                let expected = runtime.basis_dim() * self.score_dim();
                if beta.len() != expected {
                    return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival score-warp post-update beta length mismatch: proposed={}, expected {expected} for K={} and basis dim {}",
                            beta.len(),
                            self.score_dim(),
                            runtime.basis_dim()
                        ),
                    }
                    .into());
                }
                let mut projected = Array1::<f64>::zeros(beta.len());
                for coord in 0..self.score_dim() {
                    let range = score_warp_component_range(runtime, coord);
                    let current_local = current.slice(s![range.clone()]).to_owned();
                    let proposed_local = beta.slice(s![range.clone()]).to_owned();
                    let local = project_monotone_feasible_beta(
                        runtime,
                        &current_local,
                        &proposed_local,
                        &format!("score_warp_dev[z{coord}]"),
                    )?;
                    projected.slice_mut(s![range]).assign(&local);
                }
                return Ok(projected);
            }
        }
        let link_block_idx = if self.score_warp.is_some() { 4 } else { 3 };
        if self.link_dev.is_some() && block_idx == link_block_idx {
            if let Some(runtime) = &self.link_dev {
                let current = block_states
                    .get(link_block_idx)
                    .map(|state| &state.beta)
                    .ok_or_else(|| "missing survival link-deviation block state".to_string())?;
                if current.len() != beta.len() {
                    return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival link-deviation post-update beta length mismatch: current={}, proposed={}",
                            current.len(),
                            beta.len()
                        ),
                    }
                    .into());
                }
                return project_monotone_feasible_beta(runtime, current, &beta, "link_dev");
            }
        }
        Ok(beta)
    }
}

// ── Building block specs ──────────────────────────────────────────────

fn build_time_blockspec(
    time_block: &TimeBlockInput,
    design_exit: &DesignMatrix,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: design_exit.clone(),
        offset: Array1::zeros(design_exit.nrows()),
        penalties: time_block
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_block.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_logslope_blockspec(
    design: &TermCollectionDesign,
    baseline: f64,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: design.design.clone(),
        offset: offset + baseline,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn build_marginal_blockspec(
    design: &TermCollectionDesign,
    offset: &Array1<f64>,
    rho: Array1<f64>,
    beta_hint: Option<Array1<f64>>,
) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: design.design.clone(),
        offset: offset.clone(),
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: rho,
        initial_beta: beta_hint,
    }
}

fn inner_fit(
    family: &SurvivalMarginalSlopeFamily,
    blocks: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
) -> Result<UnifiedFitResult, String> {
    fit_custom_family(family, blocks, options).map_err(|e| e.to_string())
}

fn append_timewiggle_tail_nonnegative_constraints(
    base: Option<LinearInequalityConstraints>,
    p_total: usize,
    time_wiggle_ncols: usize,
) -> Result<Option<LinearInequalityConstraints>, String> {
    let p_wiggle = time_wiggle_ncols.min(p_total);
    if p_wiggle == 0 {
        return Ok(base);
    }
    if let Some(base_constraints) = base.as_ref() {
        if base_constraints.a.ncols() != p_total {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time constraint width mismatch: constraints={}, time block={p_total}",
                    base_constraints.a.ncols()
                ),
            }
            .into());
        }
        if base_constraints.a.nrows() != base_constraints.b.len() {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival marginal-slope time constraint row mismatch: A rows={}, b len={}",
                    base_constraints.a.nrows(),
                    base_constraints.b.len()
                ),
            }
            .into());
        }
    }

    let base_rows = base.as_ref().map_or(0, |constraints| constraints.a.nrows());
    let rows = base_rows + p_wiggle;
    let mut a = Array2::<f64>::zeros((rows, p_total));
    let mut b = Array1::<f64>::zeros(rows);

    if let Some(base_constraints) = base {
        a.slice_mut(s![..base_rows, ..]).assign(&base_constraints.a);
        b.slice_mut(s![..base_rows]).assign(&base_constraints.b);
    }

    let tail_start = p_total - p_wiggle;
    for (row_offset, col) in (tail_start..p_total).enumerate() {
        a[[base_rows + row_offset, col]] = 1.0;
    }
    Ok(Some(LinearInequalityConstraints { a, b }))
}

fn mean_abs(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in values {
        sum += v.abs();
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

fn block_log_lambda_seeds<'a, I>(design: &DesignMatrix, penalty_locals: I) -> Vec<f64>
where
    I: IntoIterator<Item = &'a Array2<f64>>,
{
    let unit_weights = Array1::<f64>::ones(design.nrows());
    let likelihood_scale = match design.diag_gram(&unit_weights) {
        Ok(d) => mean_abs(d.iter().copied()).max(1.0e-8),
        Err(_) => 1.0,
    };
    penalty_locals
        .into_iter()
        .map(|s| {
            let penalty_scale = mean_abs(s.diag().iter().copied()).max(1.0e-8);
            (likelihood_scale / penalty_scale).ln().clamp(-12.0, 12.0)
        })
        .collect()
}

fn joint_setup(
    data: ArrayView2<'_, f64>,
    time_penalties: usize,
    marginalspec: &TermCollectionSpec,
    marginal_penalties: usize,
    logslopespec: &TermCollectionSpec,
    logslope_penalties: usize,
    core_rho0_seed: &[f64],
    extra_rho0: &[f64],
    initial_sigma: Option<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    let marginal_terms = spatial_length_scale_term_indices(marginalspec);
    let logslope_terms = spatial_length_scale_term_indices(logslopespec);
    let core_len = time_penalties + marginal_penalties + logslope_penalties;
    let rho_dim = core_len + extra_rho0.len();
    let mut rho0vec = Array1::<f64>::zeros(rho_dim);
    debug_assert_eq!(
        core_rho0_seed.len(),
        core_len,
        "core_rho0_seed length must equal time+marginal+logslope penalty count"
    );
    for (idx, value) in core_rho0_seed.iter().copied().enumerate().take(core_len) {
        rho0vec[idx] = value;
    }
    if !extra_rho0.is_empty() {
        let start = core_len;
        for (idx, value) in extra_rho0.iter().copied().enumerate() {
            rho0vec[start + idx] = value;
        }
    }
    let rho_lower = Array1::<f64>::from_elem(rho_dim, -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho_dim, 12.0);
    // Time block has no spatial length scales (pure B-spline on time)
    let empty_kappa = SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), vec![]);
    let marginal_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        marginalspec,
        &marginal_terms,
        kappa_options,
    )
    .reseed_from_data(data, marginalspec, &marginal_terms, kappa_options);
    let logslope_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        logslopespec,
        &logslope_terms,
        kappa_options,
    )
    .reseed_from_data(data, logslopespec, &logslope_terms, kappa_options);
    let mut values = empty_kappa.as_array().to_vec();
    values.extend(marginal_kappa.as_array().iter());
    values.extend(logslope_kappa.as_array().iter());
    let marginal_dims = marginal_kappa.dims_per_term().to_vec();
    let logslope_dims = logslope_kappa.dims_per_term().to_vec();
    let mut dims = empty_kappa.dims_per_term().to_vec();
    dims.extend(marginal_dims.iter().copied());
    dims.extend(logslope_dims.iter().copied());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(values.clone()), dims.clone());
    // Bounds: concatenate [empty | marginal data-aware | logslope data-aware]
    let marginal_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut lower_vals = Vec::with_capacity(dims.iter().sum());
    lower_vals.extend(marginal_lower.as_array().iter());
    lower_vals.extend(logslope_lower.as_array().iter());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), dims.clone());
    let marginal_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        marginalspec,
        &marginal_terms,
        &marginal_dims,
        kappa_options,
    );
    let logslope_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        logslopespec,
        &logslope_terms,
        &logslope_dims,
        kappa_options,
    );
    let mut upper_vals = Vec::with_capacity(dims.iter().sum());
    upper_vals.extend(marginal_upper.as_array().iter());
    upper_vals.extend(logslope_upper.as_array().iter());
    let log_kappa_upper = SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), dims);
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);
    let setup = ExactJointHyperSetup::new(
        rho0vec,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
    );
    if let Some(sigma) = initial_sigma.filter(|sigma| *sigma > 0.0) {
        setup.with_auxiliary(
            Array1::from_vec(vec![sigma.ln()]),
            Array1::from_vec(vec![-12.0]),
            Array1::from_vec(vec![6.0]),
        )
    } else {
        setup
    }
}

fn validate_spec(spec: &SurvivalMarginalSlopeTermSpec) -> Result<(), String> {
    let n = spec.age_entry.len();
    log::info!(
        "[survival-marginal-slope] fit start n={} marginal_terms={} logslope_terms={}",
        n,
        spec.marginalspec.linear_terms.len()
            + spec.marginalspec.random_effect_terms.len()
            + spec.marginalspec.smooth_terms.len(),
        spec.logslopespec.linear_terms.len()
            + spec.logslopespec.random_effect_terms.len()
            + spec.logslopespec.smooth_terms.len(),
    );
    if spec.age_exit.len() != n
        || spec.event_target.len() != n
        || spec.weights.len() != n
        || spec.z.nrows() != n
        || spec.z.ncols() == 0
        || spec.marginal_offset.len() != n
        || spec.logslope_offset.len() != n
    {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope row mismatch: entry={}, exit={}, event={}, weights={}, z={}x{}, marginal_offset={}, logslope_offset={}",
                n,
                spec.age_exit.len(),
                spec.event_target.len(),
                spec.weights.len(),
                spec.z.nrows(),
                spec.z.ncols(),
                spec.marginal_offset.len(),
                spec.logslope_offset.len()
            ),
        }
        .into());
    }
    if spec.weights.iter().any(|&w| !w.is_finite() || w < 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite non-negative weights".to_string(),
        }
        .into());
    }
    if spec.z.iter().any(|&zi| !zi.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite z values".to_string(),
        }
        .into());
    }
    if spec.marginal_offset.iter().any(|&value| !value.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite marginal offsets".to_string(),
        }
        .into());
    }
    if spec.logslope_offset.iter().any(|&value| !value.is_finite()) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires finite logslope offsets".to_string(),
        }
        .into());
    }
    spec.frailty.validate_for_marginal_slope()?;
    match &spec.frailty {
        FrailtySpec::None => {}
        FrailtySpec::GaussianShift { sigma_fixed } => {
            let Some(sigma) = sigma_fixed else {
                return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                    reason:
                        "survival-marginal-slope requires GaussianShift sigma_fixed or FrailtySpec::None; learnable GaussianShift sigma is not implemented for the exact marginal-slope outer solver"
                            .to_string(),
                }
                .into());
            };
            if !sigma.is_finite() || *sigma < 0.0 {
                return Err(SurvivalMarginalSlopeError::InvalidInput {
                    reason: format!(
                        "survival-marginal-slope requires GaussianShift sigma >= 0, got {sigma}"
                    ),
                }
                .into());
            }
        }
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival-marginal-slope does not support FrailtySpec::HazardMultiplier"
                    .to_string(),
            }
            .into());
        }
    }
    if spec.event_target.iter().any(|&d| d != 0.0 && d != 1.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires binary event indicators (0.0 or 1.0)"
                .to_string(),
        }
        .into());
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: format!(
                "survival-marginal-slope requires derivative_guard > 0, got {}",
                spec.derivative_guard
            ),
        }
        .into());
    }
    for i in 0..n {
        if spec.age_exit[i] < spec.age_entry[i] {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: format!(
                    "survival-marginal-slope row {i}: exit time ({}) < entry time ({})",
                    spec.age_exit[i], spec.age_entry[i]
                ),
            }
            .into());
        }
    }
    let n_entry = spec.time_block.design_entry.nrows();
    let n_exit = spec.time_block.design_exit.nrows();
    let n_deriv = spec.time_block.design_derivative_exit.nrows();
    if n_entry != n || n_exit != n || n_deriv != n {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope time block design row mismatch: \
                 data={n}, design_entry={n_entry}, design_exit={n_exit}, design_derivative_exit={n_deriv}"
            ),
        }
        .into());
    }
    let p_entry = spec.time_block.design_entry.ncols();
    let p_exit = spec.time_block.design_exit.ncols();
    let p_deriv = spec.time_block.design_derivative_exit.ncols();
    if p_exit != p_entry || p_deriv != p_entry {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival-marginal-slope time block design column mismatch: entry={p_entry}, exit={p_exit}, deriv={p_deriv}"
            ),
        }
        .into());
    }
    if !spec.time_block.structural_monotonicity {
        return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason:
                "survival-marginal-slope requires structural time monotonicity by construction; non-structural time transforms are no longer supported"
                    .to_string(),
        }
        .into());
    }
    if let Some(beta0) = &spec.time_block.initial_beta {
        let derivative_constraints = structural_time_coefficient_constraints(
            &spec.time_block.design_derivative_exit,
            &spec.time_block.derivative_offset_exit,
            spec.derivative_guard,
        )?;
        if let Some(constraints) = derivative_constraints.as_ref() {
            if beta0.len() != constraints.a.ncols() {
                return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                    reason: format!(
                        "survival-marginal-slope time_block initial_beta length mismatch: got {}, expected {}",
                        beta0.len(),
                        constraints.a.ncols()
                    ),
                }
                .into());
            }
            for row in 0..constraints.a.nrows() {
                let slack = constraints.a.row(row).dot(beta0) - constraints.b[row];
                if slack < -1e-10 {
                    return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                        reason: format!(
                            "survival-marginal-slope time_block initial_beta violates structural coefficient non-negativity at row {row}: slack={slack:.3e}"
                        ),
                    }
                    .into());
                }
            }
        }
    }
    if let Some(timewiggle) = spec.timewiggle_block.as_ref() {
        if timewiggle.degree != 3 {
            return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
                reason: format!(
                    "survival-marginal-slope timewiggle requires cubic degree=3, got {}",
                    timewiggle.degree
                ),
            }
            .into());
        }
        let derived_ncols = time_wiggle_basis_ncols(&timewiggle.knots, timewiggle.degree)?;
        if derived_ncols == 0 {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason:
                    "survival-marginal-slope timewiggle requires at least one wiggle coefficient"
                        .to_string(),
            }
            .into());
        }
        if timewiggle.ncols != derived_ncols {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope timewiggle metadata width mismatch: metadata={}, basis={derived_ncols}",
                    timewiggle.ncols
                ),
            }
            .into());
        }
        if spec.time_block.design_exit.ncols() < derived_ncols {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope timewiggle requests {} tail columns but time block only has {} columns",
                    derived_ncols,
                    spec.time_block.design_exit.ncols()
                ),
            }
            .into());
        }
    }
    Ok(())
}

fn concatenate_term_specs(specs: &[TermCollectionSpec]) -> TermCollectionSpec {
    let mut out = TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    };
    for spec in specs {
        out.linear_terms.extend(spec.linear_terms.clone());
        out.random_effect_terms
            .extend(spec.random_effect_terms.clone());
        out.smooth_terms.extend(spec.smooth_terms.clone());
    }
    out
}

fn shift_penalty(mut penalty: BlockwisePenalty, offset: usize) -> BlockwisePenalty {
    penalty.col_range = (penalty.col_range.start + offset)..(penalty.col_range.end + offset);
    penalty
}

fn combine_logslope_surface_designs(
    mut designs: Vec<TermCollectionDesign>,
    specs: &[TermCollectionSpec],
) -> Result<
    (
        TermCollectionDesign,
        TermCollectionSpec,
        Vec<std::ops::Range<usize>>,
    ),
    String,
> {
    if designs.is_empty() {
        return Err(
            "survival marginal-slope requires at least one logslope surface design".to_string(),
        );
    }
    if designs.len() == 1 {
        let design = designs.remove(0);
        let range = 0..design.design.ncols();
        let spec = specs
            .first()
            .cloned()
            .ok_or_else(|| "missing logslope surface spec".to_string())?;
        return Ok((design, spec, vec![range]));
    }
    if designs.iter().any(|design| {
        design.linear_constraints.is_some() || design.coefficient_lower_bounds.is_some()
    }) {
        return Err(
            "per-z logslope surface concatenation does not support coefficient bounds or linear constraints"
                .to_string(),
        );
    }

    let mut ranges = Vec::with_capacity(designs.len());
    let mut offset = 0usize;
    let mut blocks = Vec::with_capacity(designs.len());
    let mut penalties = Vec::new();
    let mut nullspace_dims = Vec::new();
    let mut penaltyinfo = Vec::new();
    let mut dropped_penaltyinfo = Vec::new();
    let mut linear_ranges = Vec::new();
    let mut random_effect_ranges = Vec::new();
    let mut random_effect_levels = Vec::new();
    let mut combined = designs[0].clone();
    combined.smooth.term_designs.clear();
    combined.smooth.penalties.clear();
    combined.smooth.nullspace_dims.clear();
    combined.smooth.penaltyinfo.clear();
    combined.smooth.dropped_penaltyinfo.clear();
    combined.smooth.terms.clear();
    combined.smooth.coefficient_lower_bounds = None;
    combined.smooth.linear_constraints = None;

    for (surface_idx, design) in designs.into_iter().enumerate() {
        let width = design.design.ncols();
        ranges.push(offset..offset + width);
        blocks.push(design.design.clone());
        for (local_penalty_idx, penalty) in design.penalties.iter().cloned().enumerate() {
            let global_index = penalties.len();
            penalties.push(shift_penalty(penalty, offset));
            if let Some(info) = design.penaltyinfo.get(local_penalty_idx) {
                let mut info = info.clone();
                info.global_index = global_index;
                if let Some(termname) = info.termname.as_mut() {
                    *termname = format!("logslope[z{surface_idx}]::{termname}");
                }
                penaltyinfo.push(info);
            }
        }
        nullspace_dims.extend(design.nullspace_dims.iter().copied());
        dropped_penaltyinfo.extend(design.dropped_penaltyinfo.iter().cloned());
        linear_ranges.extend(design.linear_ranges.iter().cloned().map(|(name, range)| {
            (
                format!("logslope[z{surface_idx}]::{name}"),
                (range.start + offset)..(range.end + offset),
            )
        }));
        random_effect_ranges.extend(design.random_effect_ranges.iter().cloned().map(
            |(name, range)| {
                (
                    format!("logslope[z{surface_idx}]::{name}"),
                    (range.start + offset)..(range.end + offset),
                )
            },
        ));
        random_effect_levels.extend(design.random_effect_levels.iter().cloned());
        offset += width;
    }
    combined.design = DesignMatrix::hstack(blocks)
        .map_err(|e| format!("survival marginal-slope logslope hstack: {e}"))?;
    combined.penalties = penalties;
    combined.nullspace_dims = nullspace_dims;
    combined.penaltyinfo = penaltyinfo;
    combined.dropped_penaltyinfo = dropped_penaltyinfo;
    combined.coefficient_lower_bounds = None;
    combined.linear_constraints = None;
    combined.intercept_range = 0..0;
    combined.linear_ranges = linear_ranges;
    combined.random_effect_ranges = random_effect_ranges;
    combined.random_effect_levels = random_effect_levels;
    Ok((combined, concatenate_term_specs(specs), ranges))
}

/// Compute a baseline slope from the actual survival marginal-slope likelihood,
/// using the baseline offsets alone as a time-only pilot q(t).
///
/// This is a safeguarded 1D Newton solve on the true row objective. It does not
/// use a coarse fixed grid scan.
fn pooled_survival_baseline(
    event: &Array1<f64>,
    weights: &Array1<f64>,
    z: &Array1<f64>,
    q0: &Array1<f64>,
    q1: &Array1<f64>,
    qd1: &Array1<f64>,
    probit_scale: f64,
) -> f64 {
    let n = event.len();
    if n == 0 {
        return 0.0;
    }
    let objective_grad_hess = |slope: f64| -> Option<(f64, f64, f64)> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let triples: Option<Vec<(f64, f64, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (row_obj, row_grad, row_hess) = row_primary_closed_form(
                    q0[i],
                    q1[i],
                    qd1[i],
                    slope,
                    z[i],
                    weights[i],
                    event[i],
                    0.0,
                    probit_scale,
                )
                .ok()?;
                Some((row_obj, row_grad[3], row_hess[3][3]))
            })
            .collect();
        let triples = triples?;
        Some(
            triples
                .into_iter()
                .fold((0.0_f64, 0.0_f64, 0.0_f64), |(o, g, h), (oi, gi, hi)| {
                    (o + oi, g + gi, h + hi)
                }),
        )
    };

    let Some(state0) = objective_grad_hess(0.0) else {
        return 0.0;
    };
    if !state0.0.is_finite() {
        return 0.0;
    }
    if state0.1.abs() < 1e-8 {
        return 0.0;
    }

    let mut best_slope = 0.0;
    let mut best = state0;

    let mut bracket_lo = if state0.1 <= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut bracket_hi = if state0.1 >= 0.0 {
        Some((0.0, state0))
    } else {
        None
    };
    let mut step = 0.5f64;
    for _ in 0..48 {
        for &candidate in &[-step, step] {
            if let Some(state) = objective_grad_hess(candidate) {
                if state.0 < best.0 {
                    best_slope = candidate;
                    best = state;
                }
                if state.1 <= 0.0 {
                    bracket_lo = Some((candidate, state));
                }
                if state.1 >= 0.0 {
                    bracket_hi = Some((candidate, state));
                }
                if let (Some((lo, lo_state)), Some((hi, hi_state))) = (bracket_lo, bracket_hi)
                    && lo < hi
                    && lo_state.1 <= 0.0
                    && hi_state.1 >= 0.0
                {
                    let mut slope = best_slope.clamp(lo, hi);
                    let mut state = if (slope - lo).abs() < f64::EPSILON {
                        lo_state
                    } else if (slope - hi).abs() < f64::EPSILON {
                        hi_state
                    } else {
                        match objective_grad_hess(slope) {
                            Some(s) => s,
                            None => {
                                slope = 0.5 * (lo + hi);
                                objective_grad_hess(slope).unwrap_or(best)
                            }
                        }
                    };

                    let mut bracket_lo = (lo, lo_state);
                    let mut bracket_hi = (hi, hi_state);
                    for _ in 0..60 {
                        if state.1.abs() < 1e-8 || (bracket_hi.0 - bracket_lo.0).abs() < 1e-8 {
                            break;
                        }
                        let mut candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                        if state.2.is_finite() && state.2 > 0.0 {
                            let newton = slope - state.1 / state.2;
                            if newton > bracket_lo.0 && newton < bracket_hi.0 {
                                candidate = newton;
                            }
                        }
                        let Some(candidate_state) = objective_grad_hess(candidate) else {
                            candidate = 0.5 * (bracket_lo.0 + bracket_hi.0);
                            let Some(mid_state) = objective_grad_hess(candidate) else {
                                break;
                            };
                            if mid_state.0 < best.0 {
                                best_slope = candidate;
                                best = mid_state;
                            }
                            if mid_state.1 <= 0.0 {
                                bracket_lo = (candidate, mid_state);
                            } else {
                                bracket_hi = (candidate, mid_state);
                            }
                            slope = candidate;
                            state = mid_state;
                            continue;
                        };
                        if candidate_state.0 < best.0 {
                            best_slope = candidate;
                            best = candidate_state;
                        }
                        if candidate_state.1 <= 0.0 {
                            bracket_lo = (candidate, candidate_state);
                        } else {
                            bracket_hi = (candidate, candidate_state);
                        }
                        slope = candidate;
                        state = candidate_state;
                    }
                    return if best.0.is_finite() { best_slope } else { 0.0 };
                }
            }
        }
        step *= 2.0;
    }
    if best.0.is_finite() { best_slope } else { 0.0 }
}

// ── Public fitting function ───────────────────────────────────────────

pub fn fit_survival_marginal_slope_terms(
    data: ArrayView2<'_, f64>,
    spec: SurvivalMarginalSlopeTermSpec,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalMarginalSlopeFitResult, String> {
    let fit_started = std::time::Instant::now();
    let mut spec = spec;
    validate_spec(&spec)?;
    if spec.base_link != InverseLink::Standard(LinkFunction::Probit) {
        return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival-marginal-slope currently supports only probit base_link, got {:?}",
                spec.base_link
            ),
        }
        .into());
    }
    let (z_standardized, z_normalization) = standardize_latent_z_matrix_with_policy(
        &spec.z,
        &spec.weights,
        "survival-marginal-slope",
        &spec.latent_z_policy,
    )?;
    spec.z = z_standardized;
    let score_covariance = marginal_slope_covariance_from_scores(spec.z.view(), &spec.weights)?;
    let z_primary = spec.z.column(0).to_owned();
    let n = spec.age_entry.len();
    let initial_sigma = match &spec.frailty {
        FrailtySpec::GaussianShift {
            sigma_fixed: Some(s),
        } => Some(*s),
        FrailtySpec::None => None,
        FrailtySpec::GaussianShift { sigma_fixed: None } | FrailtySpec::HazardMultiplier { .. } => {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason:
                    "internal: validate_spec should have rejected unsupported marginal-slope frailty"
                        .to_string(),
            }
            .into());
        }
    };
    let probit_scale = probit_frailty_scale(initial_sigma);
    let baseline_started = std::time::Instant::now();
    let baseline_slope = pooled_survival_baseline(
        &spec.event_target,
        &spec.weights,
        &z_primary,
        &spec.time_block.offset_entry,
        &spec.time_block.offset_exit,
        &spec.time_block.derivative_offset_exit,
        probit_scale,
    );
    log::info!(
        "[survival-marginal-slope] baseline seed slope={:.6e} elapsed={:.3}s",
        baseline_slope,
        baseline_started.elapsed().as_secs_f64(),
    );

    let logslope_specs_input = spec
        .logslopespecs
        .clone()
        .unwrap_or_else(|| vec![spec.logslopespec.clone()]);
    if logslope_specs_input.len() != spec.z.ncols() && logslope_specs_input.len() != 1 {
        return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
            reason: format!(
                "survival marginal-slope expected either one shared logslope spec or one spec per z coordinate (K={}); got {}",
                spec.z.ncols(),
                logslope_specs_input.len()
            ),
        }
        .into());
    }
    let mut design_specs = Vec::with_capacity(1 + logslope_specs_input.len());
    design_specs.push(spec.marginalspec.clone());
    design_specs.extend(logslope_specs_input.iter().cloned());
    let (mut joint_designs, mut joint_specs) =
        build_term_collection_designs_and_freeze_joint(data, &design_specs)
            .map_err(|e| e.to_string())?;
    let marginal_design = joint_designs.remove(0);
    let marginalspec_boot = joint_specs.remove(0);
    let (logslope_design, logslopespec_boot, logslope_surface_ranges) =
        combine_logslope_surface_designs(joint_designs, &joint_specs)?;

    let time_penalties_len = spec.time_block.penalties.len();
    let score_warp_prepared = spec
        .score_warp
        .as_ref()
        .map(|cfg| build_per_z_score_warp_deviation_block_from_seed(&spec.z, cfg))
        .transpose()?;
    let link_dev_prepared = spec
        .link_dev
        .as_ref()
        .map(|cfg| {
            let q0_seed = Array1::from_iter((0..n).map(|row| {
                let q_exit = spec.time_block.offset_exit[row] + spec.marginal_offset[row];
                let slope = baseline_slope + spec.logslope_offset[row];
                rigid_observed_eta(q_exit, slope, z_primary[row], probit_scale)
            }));
            let padded_seed = padded_deviation_seed(&q0_seed, 1.0, 0.5);
            build_link_deviation_block_from_knots_design_seed_and_weights(
                &padded_seed,
                &q0_seed,
                &spec.weights,
                cfg,
            )
        })
        .transpose()?;
    let extra_rho0 = {
        let mut out = Vec::new();
        if let Some(ref prepared) = score_warp_prepared {
            out.extend(std::iter::repeat(0.0).take(prepared.block.penalties.len()));
        }
        if let Some(ref prepared) = link_dev_prepared {
            out.extend(std::iter::repeat(0.0).take(prepared.block.penalties.len()));
        }
        out
    };
    let core_rho0_seed: Vec<f64> = {
        let mut seeds = Vec::with_capacity(
            time_penalties_len + marginal_design.penalties.len() + logslope_design.penalties.len(),
        );
        seeds.extend(block_log_lambda_seeds(
            &spec.time_block.design_exit,
            spec.time_block.penalties.iter(),
        ));
        seeds.extend(block_log_lambda_seeds(
            &marginal_design.design,
            marginal_design.penalties.iter().map(|bp| &bp.local),
        ));
        seeds.extend(block_log_lambda_seeds(
            &logslope_design.design,
            logslope_design.penalties.iter().map(|bp| &bp.local),
        ));
        seeds
    };
    let setup = joint_setup(
        data,
        time_penalties_len,
        &marginalspec_boot,
        marginal_design.penalties.len(),
        &logslopespec_boot,
        logslope_design.penalties.len(),
        &core_rho0_seed,
        &extra_rho0,
        initial_sigma,
        kappa_options,
    );

    let hints = RefCell::new(ThetaHints::default());
    let sigma_hint = RefCell::new(initial_sigma);
    let exact_warm_start = RefCell::new(None::<CustomFamilyWarmStart>);

    let event = Arc::new(spec.event_target.clone());
    let weights = Arc::new(spec.weights.clone());
    let z = Arc::new(spec.z.clone());
    let derivative_guard = spec.derivative_guard;
    // Time designs arrive as DesignMatrix (already sparse for local-support
    // bases like B-spline, or dense for I-spline).  No post-hoc scan needed.
    let design_entry = spec.time_block.design_entry.clone();
    let design_exit = spec.time_block.design_exit.clone();
    let design_derivative_exit = spec.time_block.design_derivative_exit.clone();
    let offset_entry = Arc::new(spec.time_block.offset_entry.clone());
    let offset_exit = Arc::new(spec.time_block.offset_exit.clone());
    let derivative_offset_exit = Arc::new(spec.time_block.derivative_offset_exit.clone());
    let time_block_ref = spec.time_block.clone();
    let score_warp_runtime = score_warp_prepared.as_ref().map(|p| p.runtime.clone());
    let link_dev_runtime = link_dev_prepared.as_ref().map(|p| p.runtime.clone());
    let derived_time_wiggle_ncols = spec
        .timewiggle_block
        .as_ref()
        .map(|timewiggle| time_wiggle_basis_ncols(&timewiggle.knots, timewiggle.degree))
        .transpose()?;
    let structural_time_constraints = structural_time_coefficient_constraints(
        &design_derivative_exit,
        derivative_offset_exit.as_ref(),
        derivative_guard,
    )?;
    let time_linear_constraints = append_timewiggle_tail_nonnegative_constraints(
        structural_time_constraints,
        design_exit.ncols(),
        derived_time_wiggle_ncols.unwrap_or(0),
    )?;

    let intercept_warm_starts = new_intercept_warm_start_cache(n);
    let make_family = |marginal_design: &TermCollectionDesign,
                       logslope_design: &TermCollectionDesign,
                       sigma: Option<f64>|
     -> SurvivalMarginalSlopeFamily {
        SurvivalMarginalSlopeFamily {
            n,
            event: Arc::clone(&event),
            weights: Arc::clone(&weights),
            z: Arc::clone(&z),
            score_covariance: score_covariance.clone(),
            gaussian_frailty_sd: sigma,
            derivative_guard,
            design_entry: design_entry.clone(),
            design_exit: design_exit.clone(),
            design_derivative_exit: design_derivative_exit.clone(),
            offset_entry: Arc::clone(&offset_entry),
            offset_exit: Arc::clone(&offset_exit),
            derivative_offset_exit: Arc::clone(&derivative_offset_exit),
            marginal_design: marginal_design.design.clone(),
            logslope_design: logslope_design.design.clone(),
            logslope_surface_ranges: logslope_surface_ranges.clone(),
            score_warp: score_warp_runtime.clone(),
            link_dev: link_dev_runtime.clone(),
            time_linear_constraints: time_linear_constraints.clone(),
            time_wiggle_knots: spec.timewiggle_block.as_ref().map(|w| w.knots.clone()),
            time_wiggle_degree: spec.timewiggle_block.as_ref().map(|w| w.degree),
            time_wiggle_ncols: derived_time_wiggle_ncols.unwrap_or(0),
            intercept_warm_starts: Some(Arc::clone(&intercept_warm_starts)),
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    };

    let build_blocks = |rho: &Array1<f64>,
                        marginal_design: &TermCollectionDesign,
                        logslope_design: &TermCollectionDesign|
     -> Result<Vec<ParameterBlockSpec>, String> {
        let hints = hints.borrow();
        let mut cursor = 0usize;
        let rho_time = rho
            .slice(s![cursor..cursor + time_penalties_len])
            .to_owned();
        cursor += time_penalties_len;
        let rho_marginal = rho
            .slice(s![cursor..cursor + marginal_design.penalties.len()])
            .to_owned();
        cursor += marginal_design.penalties.len();
        let rho_logslope = rho
            .slice(s![cursor..cursor + logslope_design.penalties.len()])
            .to_owned();
        cursor += logslope_design.penalties.len();
        let time_beta_hint = if let Some(constraints) = time_linear_constraints.as_ref() {
            Some(project_onto_linear_constraints(
                design_exit.ncols(),
                constraints,
                hints
                    .time_beta
                    .as_ref()
                    .or(time_block_ref.initial_beta.as_ref()),
            ))
        } else {
            hints
                .time_beta
                .clone()
                .or_else(|| time_block_ref.initial_beta.clone())
        };
        let mut blocks = vec![
            build_time_blockspec(&time_block_ref, &design_exit, rho_time, time_beta_hint),
            build_marginal_blockspec(
                marginal_design,
                &spec.marginal_offset,
                rho_marginal,
                hints.marginal_beta.clone(),
            ),
            build_logslope_blockspec(
                logslope_design,
                baseline_slope,
                &spec.logslope_offset,
                rho_logslope,
                hints.logslope_beta.clone(),
            ),
        ];
        if let Some(prepared) = score_warp_prepared.as_ref() {
            let rho_h = rho
                .slice(s![cursor..cursor + prepared.block.penalties.len()])
                .to_owned();
            cursor += prepared.block.penalties.len();
            blocks.push(build_per_z_score_warp_aux_blockspec(
                prepared,
                rho_h,
                hints.score_warp_beta.clone(),
            )?);
        }
        push_deviation_aux_blockspecs(
            &mut blocks,
            rho,
            &mut cursor,
            None,
            link_dev_prepared.as_ref(),
            None,
            hints.link_dev_beta.clone(),
        )?;
        Ok(blocks)
    };

    // ── Pilot fit: rigid (zero-penalty) to seed coefficients ────────────
    //
    // The pilot is only a cold-start coefficient initializer. If the workflow
    // dispatcher already attached an exact or prefix warm-start entry, the
    // outer optimizer will consume that ρ seed and the first real inner solve
    // will immediately overwrite these hints at the cached smoothing point.
    // Running the rigid pilot in that regime is pure latency at biobank scale
    // (the log shows ~15s for n≈196k), and worse, it seeds β at ρ=0 while the
    // cached outer seed may be far from ρ=0. Do a non-consuming peek so the
    // optimizer still receives the cached entry via `try_load`.
    //
    // The peek must use the same validity criterion as the outer optimizer's
    // cache loader. A poisoned all-boundary checkpoint is not a usable seed:
    // skipping the pilot for such an entry leaves the subsequent cold seed
    // validation without coefficient hints, which is exactly the failure mode
    // this pilot exists to prevent.
    const CUSTOM_FAMILY_RHO_BOUND: f64 = 10.0;
    let outer_cache_seed_available = options
        .cache_session
        .as_ref()
        .and_then(|session| session.peek_load_with_source())
        .is_some_and(|loaded| {
            crate::solver::outer_strategy::cache_entry_would_help_outer(
                &loaded,
                setup.rho_dim(),
                CUSTOM_FAMILY_RHO_BOUND,
            )
        });
    if outer_cache_seed_available {
        log::info!(
            "[survival-marginal-slope/pilot] skip reason=outer-cache-seed-present n={} rho_dim={}",
            n,
            setup.rho_dim(),
        );
    } else {
        let pilot_started = std::time::Instant::now();
        log::info!(
            "[survival-marginal-slope/pilot] start n={} time_p={} marginal_p={} logslope_p={}",
            n,
            design_exit.ncols(),
            marginal_design.design.ncols(),
            logslope_design.design.ncols(),
        );
        let rigid_rho = Array1::<f64>::zeros(
            time_penalties_len
                + marginal_design.penalties.len()
                + logslope_design.penalties.len()
                + score_warp_prepared
                    .as_ref()
                    .map_or(0, |prepared| prepared.block.penalties.len())
                + link_dev_prepared
                    .as_ref()
                    .map_or(0, |prepared| prepared.block.penalties.len()),
        );
        let rigid_blocks = build_blocks(&rigid_rho, &marginal_design, &logslope_design)?;
        let rigid_family = make_family(&marginal_design, &logslope_design, initial_sigma);
        let mut pilot_options = options.clone();
        // The pilot is only a warm start. Avoid production covariance assembly
        // and cap inner cycles so a bad seed cannot silently consume minutes
        // before the real outer optimizer starts. Empirically, biobank-scale
        // survival pilots descend the joint objective by ~5 orders of
        // magnitude in the first 10 cycles and then enter a trust-region-
        // clipped tail; 30 cycles is a budget that catches the descent
        // shoulder without burning into the long tail. At ~0.5s/cycle for
        // a 350k-row LOSO fold that's ~15s — within the "no silent
        // minutes" envelope this cap protects.
        pilot_options.compute_covariance = false;
        pilot_options.inner_max_cycles = pilot_options.inner_max_cycles.min(30);
        match fit_custom_family_fixed_log_lambda_warm_start(
            &rigid_family,
            &rigid_blocks,
            &pilot_options,
        ) {
            Ok((block_beta, converged, cycles)) => {
                // Only install the pilot's β as warm-start hints if the pilot
                // actually reached a KKT certificate. The blockwise inner
                // logger at custom_family.rs:12136 emits the warning
                //   "returning non-converged warm-start iterate and rejecting
                //    this outer REML/LAML evaluation"
                // when its cycle budget is exhausted without convergence; the
                // matching outer-side contract is `nonconverged_outer_eval_result`
                // (custom_family.rs:5993), which surfaces zero gradient and
                // HessianResult::Unavailable so the optimizer backs off. A
                // partial pilot β can still be far from the cold-start optimum
                // (the warning literally exists to signal that), so seeding
                // the real outer optimizer with it can drag the first true
                // inner solve to a degenerate region of (ρ, β)-space from
                // which the analytic envelope gradient is no longer reliable.
                // Discarding the partial β reverts the first real inner solve
                // to a clean cold start at whatever ρ the outer optimizer
                // picks (cached seed or initial_theta), which is the
                // behaviour the warning text already promises.
                if converged {
                    let mut hints_mut = hints.borrow_mut();
                    if let Some(beta) = block_beta.first() {
                        hints_mut.time_beta = Some(beta.clone());
                    }
                    if let Some(beta) = block_beta.get(1) {
                        hints_mut.marginal_beta = Some(beta.clone());
                    }
                    if let Some(beta) = block_beta.get(2) {
                        hints_mut.logslope_beta = Some(beta.clone());
                    }
                    if score_warp_prepared.is_some() {
                        if let Some(beta) = block_beta.get(3) {
                            hints_mut.score_warp_beta = Some(beta.clone());
                        }
                    }
                    if link_dev_prepared.is_some() {
                        let link_idx = if score_warp_prepared.is_some() { 4 } else { 3 };
                        if let Some(beta) = block_beta.get(link_idx) {
                            hints_mut.link_dev_beta = Some(beta.clone());
                        }
                    }
                }
                log::info!(
                    "[survival-marginal-slope/pilot] end status={} cycles={} elapsed={:.3}s hints_installed={}",
                    if converged { "converged" } else { "partial" },
                    cycles,
                    pilot_started.elapsed().as_secs_f64(),
                    converged,
                );
            }
            Err(err) => {
                log::warn!(
                    "[survival-marginal-slope/pilot] end status=ignored-error elapsed={:.3}s error={}",
                    pilot_started.elapsed().as_secs_f64(),
                    err,
                );
            }
        }
    }

    let marginal_terms = spatial_length_scale_term_indices(&marginalspec_boot);
    let logslope_terms = spatial_length_scale_term_indices(&logslopespec_boot);
    let marginal_has_spatial = !marginal_terms.is_empty();
    let logslope_has_spatial = !logslope_terms.is_empty();
    let analytic_joint_derivatives_available =
        marginal_has_spatial || logslope_has_spatial || setup.log_kappa_dim() == 0;

    if setup.log_kappa_dim() > 0 && !analytic_joint_derivatives_available {
        return Err(
            "exact survival marginal-slope spatial optimization requires analytic joint psi derivatives"
                .to_string(),
        );
    }

    let derivative_probe_started = std::time::Instant::now();
    log::info!(
        "[survival-marginal-slope] initial derivative probe start rho_dim={} log_kappa_dim={}",
        setup.rho_dim(),
        setup.log_kappa_dim(),
    );
    let initial_rho = setup.theta0().slice(s![..setup.rho_dim()]).to_owned();
    let initial_blocks = build_blocks(&initial_rho, &marginal_design, &logslope_design)?;
    let initial_family = make_family(&marginal_design, &logslope_design, initial_sigma);
    let (joint_gradient, joint_hessian) =
        custom_family_outer_derivatives(&initial_family, &initial_blocks, options);
    let analytic_joint_gradient_available = analytic_joint_derivatives_available
        && matches!(
            joint_gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
    // Survival marginal-slope now exposes exact coefficient-space and ψ-space
    // Hessian directional derivatives as HyperOperators (see the workspace
    // overrides below). Keep analytic curvature advertised at biobank scale;
    // the unified REML/LAML planner chooses the matrix-free outer-HVP route for
    // large `(n, p, K)` shapes instead of falling back to first-order BFGS.
    let analytic_joint_hessian_available =
        analytic_joint_derivatives_available && joint_hessian.is_analytic();
    log::info!(
        "[survival-marginal-slope] initial derivative probe end gradient_analytic={} hessian_analytic={} elapsed={:.3}s",
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        derivative_probe_started.elapsed().as_secs_f64(),
    );
    let kappa_options_ref: &SpatialLengthScaleOptimizationOptions = kappa_options;
    let derivative_block_cache = RefCell::new(
        None::<(
            Array1<f64>,
            Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        )>,
    );
    let theta_matches = |left: &Array1<f64>, right: &Array1<f64>| -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right.iter())
                .all(|(lhs, rhs)| (*lhs - *rhs).abs() <= 1e-12 * (1.0 + lhs.abs().max(rhs.abs())))
    };
    let sigma_from_theta = |theta: &Array1<f64>| -> Option<f64> {
        initial_sigma.map(|_| theta[setup.rho_dim() + setup.log_kappa_dim()].exp())
    };
    let get_derivative_blocks = |theta: &Array1<f64>,
                                 specs: &[TermCollectionSpec],
                                 designs: &[TermCollectionDesign]|
     -> Result<
        Arc<Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>>>,
        String,
    > {
        if let Some((cached_theta, cached_blocks)) = derivative_block_cache.borrow().as_ref()
            && theta_matches(cached_theta, theta)
        {
            return Ok(Arc::clone(cached_blocks));
        }

        let mut derivative_blocks = vec![
            Vec::new(),
            if marginal_has_spatial {
                build_block_spatial_psi_derivatives(data, &specs[0], &designs[0])?.ok_or_else(
                    || {
                        "survival marginal-slope: marginal block has spatial terms but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            },
            if logslope_has_spatial {
                build_block_spatial_psi_derivatives(data, &specs[1], &designs[1])?.ok_or_else(
                    || {
                        "survival marginal-slope: logslope block has spatial terms but spatial psi derivatives are unavailable"
                            .to_string()
                    },
                )?
            } else {
                Vec::new()
            },
        ];
        if score_warp_runtime.is_some() {
            derivative_blocks.push(Vec::new());
        }
        if link_dev_runtime.is_some() {
            derivative_blocks.push(Vec::new());
        }
        if initial_sigma.is_some_and(|sigma| sigma > 0.0) {
            let sigma_aux = crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                Array2::zeros((0, 0)),
                Array2::zeros((0, 0)),
                None,
                None,
                None,
                None,
            );
            derivative_blocks
                .last_mut()
                .ok_or_else(|| "survival marginal-slope missing derivative blocks".to_string())?
                .push(sigma_aux);
        }
        let derivative_blocks = Arc::new(derivative_blocks);
        derivative_block_cache.replace(Some((theta.clone(), Arc::clone(&derivative_blocks))));
        Ok(derivative_blocks)
    };

    log::info!(
        "[survival-marginal-slope/outer] solve start rho_dim={} log_kappa_dim={} aux_dim={}",
        setup.rho_dim(),
        setup.log_kappa_dim(),
        setup.auxiliary_dim(),
    );

    // Survival marginal-slope is a multi-block family with β-dependent
    // joint Hessian (hazard multipliers depend on current β); the
    // Wood-Fasiolo PSD invariant that justifies EFS fails here, so
    // disable fixed-point at plan time.
    let outer_policy = {
        let psi_dim = setup.theta0().len() - setup.rho_dim();
        initial_family.outer_derivative_policy(&initial_blocks, psi_dim, options)
    };
    let exact_spatial_outer_tol = kappa_options_ref.rel_tol.max(1e-6);
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[marginalspec_boot.clone(), logslopespec_boot.clone()],
        &[marginal_terms.clone(), logslope_terms.clone()],
        kappa_options_ref,
        &setup,
        crate::seeding::SeedRiskProfile::Survival,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        None,
        outer_policy,
        |theta, _: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let eval_started = std::time::Instant::now();
            log::info!(
                "[survival-marginal-slope/outer-inner-fit] start theta_dim={}",
                theta.len(),
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            sigma_hint.replace(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let fit = inner_fit(&family, &blocks, options)?;
            let mut hints_mut = hints.borrow_mut();
            if let Some(block) = fit.block_states.get(0) {
                hints_mut.time_beta = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(1) {
                hints_mut.marginal_beta = Some(block.beta.clone());
            }
            if let Some(block) = fit.block_states.get(2) {
                hints_mut.logslope_beta = Some(block.beta.clone());
            }
            if score_warp_prepared.is_some() {
                if let Some(block) = fit.block_states.get(3) {
                    hints_mut.score_warp_beta = Some(block.beta.clone());
                }
            }
            if link_dev_prepared.is_some() {
                let link_idx = if score_warp_prepared.is_some() { 4 } else { 3 };
                if let Some(block) = fit.block_states.get(link_idx) {
                    hints_mut.link_dev_beta = Some(block.beta.clone());
                }
            }
            log::info!(
                "[survival-marginal-slope/outer-inner-fit] end elapsed={:.3}s",
                eval_started.elapsed().as_secs_f64(),
            );
            Ok(fit)
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         _row_set: &crate::families::row_kernel::RowSet| {
            use crate::solver::estimate::reml::unified::EvalMode;
            let eval_started = std::time::Instant::now();
            log::info!(
                "[survival-marginal-slope/outer-eval] start mode={:?} theta_dim={}",
                eval_mode,
                theta.len(),
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            sigma_hint.replace(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            // Preserve ValueOnly probes and request the Hessian exactly when
            // this realized family advertised analytic joint second-order
            // support.
            let effective_mode = match eval_mode {
                EvalMode::ValueGradientHessian if !analytic_joint_hessian_available => {
                    EvalMode::ValueAndGradient
                }
                other => other,
            };
            let eval = evaluate_custom_family_joint_hyper_shared(
                &family,
                &blocks,
                &joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol),
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                effective_mode,
            )?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "exact survival marginal-slope inner solve did not converge".to_string()
                );
            }
            log::info!(
                "[survival-marginal-slope/outer-eval] end objective={:.6e} mode={:?} elapsed={:.3}s",
                eval.objective,
                eval_mode,
                eval_started.elapsed().as_secs_f64(),
            );
            if matches!(eval_mode, EvalMode::ValueGradientHessian)
                && analytic_joint_hessian_available
                && !eval.outer_hessian.is_analytic()
            {
                return Err(
                    "exact survival marginal-slope joint objective did not return an outer Hessian"
                        .to_string(),
                );
            }
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let eval_started = std::time::Instant::now();
            log::info!(
                "[survival-marginal-slope/outer-efs] start theta_dim={}",
                theta.len(),
            );
            let rho = theta.slice(s![..setup.rho_dim()]).to_owned();
            let blocks = build_blocks(&rho, &designs[0], &designs[1])?;
            let sigma = sigma_from_theta(theta);
            sigma_hint.replace(sigma);
            let family = make_family(&designs[0], &designs[1], sigma);
            let derivative_blocks = get_derivative_blocks(theta, specs, designs)?;
            let eval = evaluate_custom_family_joint_hyper_efs_shared(
                &family,
                &blocks,
                &joint_hyper_options_for_outer_tolerance(options, exact_spatial_outer_tol),
                &rho,
                derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )?;
            exact_warm_start.replace(Some(eval.warm_start.clone()));
            if !eval.inner_converged {
                return Err(
                    "exact survival marginal-slope EFS inner solve did not converge".to_string(),
                );
            }
            log::info!(
                "[survival-marginal-slope/outer-efs] end elapsed={:.3}s",
                eval_started.elapsed().as_secs_f64(),
            );
            Ok(eval.efs_eval)
        },
    )?;
    log::info!(
        "[survival-marginal-slope/outer] solve end elapsed={:.3}s",
        fit_started.elapsed().as_secs_f64(),
    );
    if !solved.fit.outer_converged {
        return Err(SurvivalMarginalSlopeError::IntegrationFailed {
            reason: format!(
                "survival marginal-slope outer optimization did not converge: \
                 iterations={} final_objective={:.6e} |g|_inf={:?}",
                solved.fit.outer_iterations, solved.fit.reml_score, solved.fit.outer_gradient_norm
            ),
        }
        .into());
    }

    let (baseline_offset_residuals, baseline_offset_curvatures) = {
        let final_family =
            make_family(&solved.designs[0], &solved.designs[1], *sigma_hint.borrow());
        final_family.offset_channel_geometry(&solved.fit.block_states)?
    };
    let mut resolved_specs = solved.resolved_specs;
    let designs = solved.designs;
    Ok(SurvivalMarginalSlopeFitResult {
        fit: solved.fit,
        marginalspec_resolved: resolved_specs.remove(0),
        logslopespec_resolved: resolved_specs.remove(0),
        marginal_design: designs[0].clone(),
        logslope_design: designs[1].clone(),
        gaussian_frailty_sd: *sigma_hint.borrow(),
        baseline_slope,
        baseline_offset_residuals,
        baseline_offset_curvatures,
        z_normalization,
        time_block_penalties_len: time_penalties_len,
        score_warp_runtime,
        link_dev_runtime,
    })
}

/// Thread-local invocation counter used by the warm-start unit test to
/// assert that pre-populating the per-row intercept cache reduces the number
/// of `evaluate_denested_survival_calibration` calls. Compiled into the
/// `solve_row_survival_intercept_with_slot` evaluator only under `cfg(test)`,
/// so the production hot path is byte-identical.
#[cfg(test)]
mod survival_intercept_test_counter {
    use std::cell::Cell;
    thread_local! {
        static COUNT: Cell<u64> = const { Cell::new(0) };
    }
    pub(super) fn bump() {
        COUNT.with(|c| c.set(c.get() + 1));
    }
    pub(super) fn reset() {
        COUNT.with(|c| c.set(0));
    }
    pub(super) fn get() -> u64 {
        COUNT.with(|c| c.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::{CustomFamily, ExactOuterDerivativeOrder};
    use crate::matrix::{DenseDesignMatrix, SymmetricMatrix};
    use approx::assert_relative_eq;
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::array;

    fn poly_mul_legacy(lhs: &[f64], rhs: &[f64]) -> PolyVec {
        if lhs.is_empty() || rhs.is_empty() {
            return PolyVec::new();
        }
        let mut out: PolyVec = smallvec![0.0; lhs.len() + rhs.len() - 1];
        for (i, &lv) in lhs.iter().enumerate() {
            for (j, &rv) in rhs.iter().enumerate() {
                out[i + j] += lv * rv;
            }
        }
        out
    }

    fn poly_add_legacy(lhs: &[f64], rhs: &[f64]) -> PolyVec {
        let mut out: PolyVec = smallvec![0.0; lhs.len().max(rhs.len())];
        for (idx, value) in lhs.iter().enumerate() {
            out[idx] += value;
        }
        for (idx, value) in rhs.iter().enumerate() {
            out[idx] += value;
        }
        out
    }

    fn poly_sub_legacy(lhs: &[f64], rhs: &[f64]) -> PolyVec {
        let mut out: PolyVec = smallvec![0.0; lhs.len().max(rhs.len())];
        for (idx, value) in lhs.iter().enumerate() {
            out[idx] += value;
        }
        for (idx, value) in rhs.iter().enumerate() {
            out[idx] -= value;
        }
        out
    }

    fn poly_scale_legacy(poly: &[f64], scale: f64) -> PolyVec {
        poly.iter().map(|value| scale * value).collect()
    }

    #[test]
    fn poly_kernels_match_legacy_implementations() {
        // Locks numerical agreement with the pre-tightening implementations of
        // poly_mul/add/sub/scale. Bit-for-bit identical (same FMA pairings,
        // same summation order). Covers same-length, lhs-longer, rhs-longer,
        // empty, length-1, and the typical cubic×cubic shape.
        let cases: &[(&[f64], &[f64])] = &[
            (&[], &[]),
            (&[], &[1.0, 2.0, 3.0]),
            (&[2.0], &[3.0]),
            (&[1.0, 2.0, 3.0, 4.0], &[5.0, -6.0, 7.0, -8.0]),
            (&[0.5, -0.25, 0.125], &[1.0, 2.0, 3.0, 4.0, 5.0]),
            (&[1.0, 0.0, -1.0], &[0.0, 1.0, 0.0]),
            (
                &[1e-12, 1e-6, 1.0, 1e6, 1e12],
                &[-1e-12, -1e-6, -1.0, -1e6, -1e12],
            ),
        ];
        for (lhs, rhs) in cases {
            let mul_new: Vec<f64> = poly_mul(lhs, rhs).to_vec();
            let mul_old: Vec<f64> = poly_mul_legacy(lhs, rhs).to_vec();
            assert_eq!(
                mul_new, mul_old,
                "poly_mul mismatch for lhs={lhs:?}, rhs={rhs:?}"
            );
            let add_new: Vec<f64> = poly_add(lhs, rhs).to_vec();
            let add_old: Vec<f64> = poly_add_legacy(lhs, rhs).to_vec();
            assert_eq!(
                add_new, add_old,
                "poly_add mismatch for lhs={lhs:?}, rhs={rhs:?}"
            );
            let sub_new: Vec<f64> = poly_sub(lhs, rhs).to_vec();
            let sub_old: Vec<f64> = poly_sub_legacy(lhs, rhs).to_vec();
            assert_eq!(
                sub_new, sub_old,
                "poly_sub mismatch for lhs={lhs:?}, rhs={rhs:?}"
            );
            for &scale in &[0.0, 1.0, -1.0, 2.5, -0.125] {
                let scl_new: Vec<f64> = poly_scale(lhs, scale).to_vec();
                let scl_old: Vec<f64> = poly_scale_legacy(lhs, scale).to_vec();
                assert_eq!(
                    scl_new, scl_old,
                    "poly_scale mismatch for poly={lhs:?}, scale={scale}"
                );
            }
        }
    }

    fn empty_termspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![],
        }
    }

    fn unit_score_covariance() -> MarginalSlopeCovariance {
        MarginalSlopeCovariance::Diagonal(array![1.0])
    }

    // Mirrors the production single-z convention produced by
    // `combine_logslope_surface_designs` for `designs.len() == 1`, where the
    // emitted ranges vector is `vec![0..ncols]`. Test fixtures use empty
    // logslope designs (`n × 0`), so the single placeholder range is `0..0`.
    fn empty_logslope_surface_ranges() -> Vec<std::ops::Range<usize>> {
        let placeholder = 0..0;
        vec![placeholder]
    }

    fn base_time_block() -> TimeBlockInput {
        TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Array1::zeros(1),
            offset_exit: Array1::zeros(1),
            derivative_offset_exit: Array1::from_elem(
                1,
                DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
            ),
            structural_monotonicity: true,
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: Some(Array1::zeros(1)),
        }
    }

    fn sparse_design(dense: &Array2<f64>) -> DesignMatrix {
        let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                let value = dense[[i, j]];
                if value != 0.0 {
                    triplets.push(Triplet::new(i, j, value));
                }
            }
        }
        let sparse = SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
            .expect("assemble sparse design");
        DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse))
    }

    /// Build an n-row closed-form survival family with empty time/marginal/
    /// logslope blocks (so q0/q1/qd1 come from offsets only). No flex
    /// deviations are configured, so `log_likelihood_only` takes the
    /// closed-form fast path.
    fn make_closed_form_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
        // Pseudo-random rows uncorrelated with row parity, so an even-only
        // subsample is representative for the Horvitz-Thompson rescaling
        // check.
        let event: Array1<f64> =
            Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
        let weights: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
        let z: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_entry: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_exit: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
        );
        // qd1 must remain strictly above the derivative guard.
        let derivative_offset_exit: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
        SurvivalMarginalSlopeFamily {
            n,
            event: Arc::new(event),
            weights: Arc::new(weights),
            z: Arc::new(z.insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            // Empty time/marginal/logslope designs: `n_rows × 0` so the
            // closed-form q geometry is driven entirely by offsets.
            design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            offset_entry: Arc::new(offset_entry),
            offset_exit: Arc::new(offset_exit),
            derivative_offset_exit: Arc::new(derivative_offset_exit),
            marginal_design: DesignMatrix::from(Array2::zeros((n, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((n, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    }

    fn closed_form_block_states(
        family: &SurvivalMarginalSlopeFamily,
        g: f64,
    ) -> Vec<ParameterBlockState> {
        let n = family.n;
        vec![
            // Time block: empty beta; per-row eta entries unused (designs
            // are zero-column).
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(n),
            },
            // Marginal block: empty beta.
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(n),
            },
            // Log-slope block: empty beta with per-row eta = g (constant
            // log-slope across rows).
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::from_elem(n, g),
            },
        ]
    }

    #[test]
    fn survival_primary_g_fourth_cell_partials_are_zero() {
        let family = make_closed_form_test_family(1);
        let primary = flex_primary_slices(&family);
        let score_span = exact_kernel::LocalSpanCubic {
            left: -1.0,
            right: 1.0,
            c0: 0.2,
            c1: -0.1,
            c2: 0.05,
            c3: -0.02,
        };
        let link_span = exact_kernel::LocalSpanCubic {
            left: -0.5,
            right: 0.5,
            c0: 0.1,
            c1: 0.3,
            c2: -0.2,
            c3: 0.4,
        };
        let fixed = family
            .denested_cell_primary_fixed_partials(
                &primary, 0.2, 0.7, score_span, link_span, 0.0, 0.0,
            )
            .expect("primary fixed partials");
        let (_, dc_daab, dc_dabb, dc_dbbb) = exact_kernel::denested_cell_third_partials(link_span);

        assert_eq!(fixed.coeff_aau[primary.g], dc_daab);
        assert_eq!(fixed.coeff_abu[primary.g], dc_dabb);
        assert_eq!(fixed.coeff_bbu[primary.g], dc_dbbb);
        assert!(dc_daab.iter().any(|value| *value != 0.0));
        assert!(dc_dabb.iter().any(|value| *value != 0.0));
        assert!(dc_dbbb.iter().any(|value| *value != 0.0));
        assert_eq!(fixed.coeff_aaau[primary.g], [0.0; 4]);
        assert_eq!(fixed.coeff_aabu[primary.g], [0.0; 4]);
        assert_eq!(fixed.coeff_abbu[primary.g], [0.0; 4]);
        assert_eq!(fixed.coeff_bbbu[primary.g], [0.0; 4]);
    }

    #[test]
    fn survival_log_likelihood_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);

        let baseline = family
            .log_likelihood_only(&states)
            .expect("baseline ll (no subsample)");

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full_mask = family
            .log_likelihood_only_with_options(&states, &opts_full)
            .expect("ll with mask=full");

        let rel = ((with_full_mask - baseline) / baseline.abs().max(1.0)).abs();
        assert!(
            rel < 1e-12,
            "subsample(mask=full) {} differs from baseline {} by rel {}",
            with_full_mask,
            baseline,
            rel
        );
    }

    #[test]
    fn survival_log_likelihood_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .log_likelihood_only_with_options(&states, &opts_half)
            .expect("ll with mask=even");

        // Raw even-row sum: same mask but weight_scale = 1.0.
        let mut opts_even_unscaled = BlockwiseFitOptions::default();
        opts_even_unscaled.outer_score_subsample = Some(Arc::new(
            OuterScoreSubsample::with_uniform_weight(even_mask, m, 0, 1.0),
        ));
        let raw_even_sum = family
            .log_likelihood_only_with_options(&states, &opts_even_unscaled)
            .expect("raw even-row ll sum");

        let expected_scaled = (n as f64 / m as f64) * raw_even_sum;
        let rel = ((scaled - expected_scaled) / expected_scaled.abs().max(1.0)).abs();
        assert!(
            rel < 1e-12,
            "scaled {} != 2*even_sum {} (rel {})",
            scaled,
            expected_scaled,
            rel
        );

        // Horvitz-Thompson check: 2 * Σ_even ≈ full-data sum.
        let baseline = family.log_likelihood_only(&states).expect("baseline ll");
        let ht_rel = ((scaled - baseline) / baseline.abs().max(1.0)).abs();
        assert!(
            ht_rel < 0.05,
            "Horvitz-Thompson scaled {} not near baseline {} (rel {})",
            scaled,
            baseline,
            ht_rel
        );
    }

    #[test]
    fn poly_mul_treats_empty_inputs_as_zero_polynomials() {
        assert!(poly_mul(&[], &[1.0, 2.0]).is_empty());
        assert!(poly_mul(&[1.0, 2.0], &[]).is_empty());
        assert_eq!(
            poly_mul(&[1.0, 2.0], &[3.0, 4.0]).as_slice(),
            &[3.0, 10.0, 8.0][..]
        );
    }

    fn dummy_blockspec(cols: usize) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: "dummy".to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((1, cols)))),
            offset: Array1::zeros(1),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::zeros(cols)),
        }
    }

    fn dummy_penalized_blockspec(cols: usize, penalties: usize) -> ParameterBlockSpec {
        let mut spec = dummy_blockspec(cols);
        spec.penalties = (0..penalties)
            .map(|_| PenaltyMatrix::Dense(Array2::eye(cols)))
            .collect();
        spec.nullspace_dims = vec![0; penalties];
        spec.initial_log_lambdas = Array1::zeros(penalties);
        spec
    }

    fn test_deviation_runtime() -> DeviationRuntime {
        build_score_warp_deviation_block_from_seed(
            &array![-1.0, 0.0, 1.0],
            &DeviationBlockConfig {
                degree: 3,
                num_internal_knots: 1,
                penalty_order: 2,
                penalty_orders: vec![1, 2, 3],
                double_penalty: false,
                monotonicity_eps: 1e-4,
            },
        )
        .expect("build test deviation runtime")
        .runtime
    }

    fn max_abs_diff_vec(lhs: &Array1<f64>, rhs: &Array1<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    fn max_abs_diff_mat(lhs: &Array2<f64>, rhs: &Array2<f64>) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn rigid_shared_multi_z_row_kernel_matches_vector_value_and_finite_differences() {
        let covariance = MarginalSlopeCovariance::Full(array![[1.3, 0.4], [0.4, 0.7]]);
        let z = [0.6, -1.1];
        let params = [0.15, 0.55, 0.9, -0.22];
        let weight = 1.3;
        let event = 1.0;
        let derivative_guard = 1e-6;
        let probit_scale = 0.85;
        let covariance_ones = covariance.quadratic_form(&[1.0, 1.0]).expect("1'Sigma1");
        let z_sum = z.iter().sum::<f64>();

        let eval = |p: [f64; 4]| {
            row_primary_closed_form_shared_score(
                p[0],
                p[1],
                p[2],
                p[3],
                z_sum,
                covariance_ones,
                weight,
                event,
                derivative_guard,
                probit_scale,
            )
            .expect("shared-score row kernel")
        };

        let (nll, grad, hess) = eval(params);
        let vector_nll = survival_marginal_slope_vector_neglog(
            params[0],
            params[1],
            params[2],
            &[params[3], params[3]],
            &z,
            &covariance,
            weight,
            event,
            derivative_guard,
            probit_scale,
        )
        .expect("vector row value");
        assert!(
            (nll - vector_nll).abs() <= 1e-14,
            "shared row nll={nll:.17e} vector nll={vector_nll:.17e}"
        );

        let step = 1e-5;
        for axis in 0..4 {
            let mut plus = params;
            let mut minus = params;
            plus[axis] += step;
            minus[axis] -= step;
            let (nll_plus, grad_plus, _) = eval(plus);
            let (nll_minus, grad_minus, _) = eval(minus);
            let fd_grad = (nll_plus - nll_minus) / (2.0 * step);
            assert!(
                (grad[axis] - fd_grad).abs() <= 2e-6,
                "grad[{axis}] analytic={:.12e} fd={:.12e}",
                grad[axis],
                fd_grad
            );
            for row in 0..4 {
                let fd_hess = (grad_plus[row] - grad_minus[row]) / (2.0 * step);
                assert!(
                    (hess[row][axis] - fd_hess).abs() <= 3e-5,
                    "hess[{row},{axis}] analytic={:.12e} fd={:.12e}",
                    hess[row][axis],
                    fd_hess
                );
            }
        }
    }

    fn assert_blockwise_matches_joint_principal_blocks(
        family: &SurvivalMarginalSlopeFamily,
        block_states: &[ParameterBlockState],
    ) {
        let eval = family
            .evaluate_blockwise_exact_newton(block_states)
            .expect("blockwise exact-newton evaluation");
        let (joint_ll, joint_gradient, joint_hessian) = family
            .evaluate_exact_newton_joint_dense(block_states)
            .expect("joint dense exact-newton evaluation");
        let slices = block_slices(family, block_states);
        let mut block_ranges = vec![
            slices.time.clone(),
            slices.marginal.clone(),
            slices.logslope.clone(),
        ];
        if let Some(range) = slices.score_warp.clone() {
            block_ranges.push(range);
        }
        if let Some(range) = slices.link_dev.clone() {
            block_ranges.push(range);
        }

        assert!((eval.log_likelihood - joint_ll).abs() <= 1e-10);
        assert_eq!(eval.blockworking_sets.len(), block_ranges.len());
        for (work, range) in eval.blockworking_sets.iter().zip(block_ranges.iter()) {
            let BlockWorkingSet::ExactNewton { gradient, hessian } = work else {
                panic!("expected exact-newton block working set");
            };
            let expected_gradient = joint_gradient.slice(s![range.clone()]).to_owned();
            let expected_hessian = joint_hessian
                .slice(s![range.clone(), range.clone()])
                .to_owned();
            assert!(
                max_abs_diff_vec(gradient, &expected_gradient) <= 1e-10,
                "gradient block mismatch"
            );
            assert!(
                max_abs_diff_mat(&hessian.to_dense(), &expected_hessian) <= 1e-10,
                "hessian block mismatch"
            );
        }
    }

    fn test_family(
        score_warp: Option<DeviationRuntime>,
        link_dev: Option<DeviationRuntime>,
    ) -> SurvivalMarginalSlopeFamily {
        SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(Array1::zeros(1)),
            offset_exit: Arc::new(Array1::zeros(1)),
            derivative_offset_exit: Arc::new(Array1::from_elem(1, 1e-6)),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 2))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 3))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp,
            link_dev,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    }

    #[test]
    fn validate_spec_rejects_nonstructural_time_block() {
        let spec = SurvivalMarginalSlopeTermSpec {
            age_entry: array![0.0, 0.0],
            age_exit: array![1.0, 1.0],
            event_target: array![0.0, 1.0],
            weights: array![1.0, 1.0],
            z: array![-1.0, 1.0].insert_axis(Axis(1)),
            base_link: InverseLink::Standard(LinkFunction::Probit),
            marginalspec: empty_termspec(),
            marginal_offset: Array1::zeros(2),
            frailty: FrailtySpec::None,
            derivative_guard: 1e-4,
            time_block: TimeBlockInput {
                design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
                design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
                design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
                offset_entry: Array1::zeros(2),
                offset_exit: Array1::zeros(2),
                derivative_offset_exit: Array1::zeros(2),
                structural_monotonicity: false,
                ..base_time_block()
            },
            timewiggle_block: None,
            logslopespec: empty_termspec(),
            logslopespecs: None,
            logslope_offset: Array1::zeros(2),
            score_warp: None,
            link_dev: None,
            latent_z_policy: LatentZPolicy::default(),
        };

        let err = validate_spec(&spec).expect_err("non-structural time block should fail");
        assert!(
            err.contains("requires structural time monotonicity"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_spec_rejects_learnable_gaussian_shift_sigma() {
        let spec = SurvivalMarginalSlopeTermSpec {
            age_entry: array![0.0, 0.0],
            age_exit: array![1.0, 1.0],
            event_target: array![0.0, 1.0],
            weights: array![1.0, 1.0],
            z: array![-1.0, 1.0].insert_axis(Axis(1)),
            base_link: InverseLink::Standard(LinkFunction::Probit),
            marginalspec: empty_termspec(),
            marginal_offset: Array1::zeros(2),
            frailty: FrailtySpec::GaussianShift { sigma_fixed: None },
            derivative_guard: 1e-4,
            time_block: base_time_block(),
            timewiggle_block: None,
            logslopespec: empty_termspec(),
            logslopespecs: None,
            logslope_offset: Array1::zeros(2),
            score_warp: None,
            link_dev: None,
            latent_z_policy: LatentZPolicy::default(),
        };

        let err =
            validate_spec(&spec).expect_err("learnable GaussianShift sigma should be rejected");
        assert!(err.contains("learnable GaussianShift sigma is not implemented"));
    }

    #[test]
    fn block_slices_handles_link_only_survival_flex_layout() {
        let link_runtime = test_deviation_runtime();
        let family = test_family(None, Some(link_runtime.clone()));
        let block_states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(3),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];

        let slices = block_slices(&family, &block_states);
        assert!(slices.score_warp.is_none());
        assert_eq!(
            slices.link_dev.as_ref().expect("link-only slice").len(),
            link_runtime.basis_dim()
        );
        assert_eq!(slices.total, 1 + 2 + 3 + link_runtime.basis_dim());
    }

    #[test]
    fn exact_flex_row_matches_rigid_closed_form_without_deviations() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.7]),
            z: Arc::new(array![0.25].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![0.2]),
            offset_exit: Arc::new(array![0.4]),
            derivative_offset_exit: Arc::new(array![0.8]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.6],
            },
        ];
        let q_geom = family
            .row_dynamic_q_geometry(0, &block_states)
            .expect("row geometry");
        let primary = flex_primary_slices(&family);
        let (nll_exact, grad_exact, hess_exact) = family
            .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
            .expect("exact flex row");
        let (nll_rigid, grad_rigid, hess_rigid) = row_primary_closed_form(
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            block_states[2].eta[0],
            family.z[[0, 0]],
            family.weights[0],
            family.event[0],
            family.derivative_guard,
            family.probit_frailty_scale(),
        )
        .expect("rigid row");

        assert!((nll_exact - nll_rigid).abs() < 1e-10);
        for idx in 0..N_PRIMARY {
            assert!((grad_exact[idx] - grad_rigid[idx]).abs() < 1e-8);
        }
        for i in 0..N_PRIMARY {
            for j in 0..N_PRIMARY {
                assert!((hess_exact[[i, j]] - hess_rigid[i][j]).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn row_primary_closed_form_rejects_negative_infinite_signed_margin() {
        let err = row_primary_closed_form(f64::INFINITY, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1e-6, 1.0)
            .expect_err("exact closed-form row should reject -inf signed margins");
        assert!(err.contains("non-finite signed margin"));
    }

    /// Mechanism-of-ρ=2 proof for the inner-PIRLS pathology on biobank-scale
    /// saturated probit fits.
    ///
    /// `add_pullback_primary_hessian` (line 9753) sums `h[0,0] + h[1,1]`
    /// over rows into the marginal-block joint Hessian (β_marg enters both
    /// `q0` and `q1` with the SAME `marginal_design` row, so the Jacobian
    /// pullback adds the q0 and q1 second derivatives at the same slot).
    ///
    /// Mathematical facts pinned here:
    ///   - Censored rows: `h[0,0] + h[1,1] = 0` to ULP at every η ≥ 0.
    ///     The entry term `+w·log Φ(−η₀)` (concave, curvature −w·c²) and
    ///     the exit term `−w·log Φ(−η₁)` (convex, curvature +w·c²) have
    ///     equal-and-opposite second derivatives when q₀ = q₁.
    ///   - Event rows: residual `+w·c²·(1/η² + O(1/η⁴))` from the Mills
    ///     asymptotic. The event-density term `+w·η₁²/2` contributes
    ///     exactly `+w·c²`; the entry-survival term contributes
    ///     `−w·c²·(1 − 1/η²)`. Sum = `+w·c²/η²`.
    ///
    /// At biobank saturation (η ~ 988), the marginal-block joint Hessian
    /// collapses to `O(w/η²) = O(1e-6)` per event row from the likelihood
    /// side; censored contributions are 0 to ULP. The Newton step in that
    /// block is then dominated by the smoothing penalty `S_marg`. When
    /// the saturating direction lies in the null space of `S_marg`
    /// (typical for the duchon-smooth's polynomial null space), the
    /// effective curvature drops to the f64 ridge floor, the inner
    /// Newton step is set by ridge alone, and `actual = rhs·δ` while
    /// `predicted = ½·rhs·δ` — yielding ρ ≡ 2 to floating-point precision
    /// as observed.
    #[test]
    fn marginal_block_hessian_cancels_in_saturated_regime() {
        let probit_scale = 1.0_f64;
        let w = 1.0_f64;
        let derivative_guard = 1e-6;
        let qd1 = 1.0_f64;
        let g = 0.0_f64;
        let z = 0.0_f64;

        // Censored rows, q0 = q1 = η, at a wide range of saturations:
        // cancellation must be ULP-exact for every η.
        for &eta in &[0.5_f64, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0, 500.0, 988.0] {
            let (_nll, _grad, hess) = row_primary_closed_form(
                eta,
                eta,
                qd1,
                g,
                z,
                w,
                0.0,
                derivative_guard,
                probit_scale,
            )
            .expect("rigid censored row");
            let sum = hess[0][0] + hess[1][1];
            assert!(
                sum.abs() <= 1e-12 * (hess[0][0].abs() + hess[1][1].abs()).max(1.0),
                "censored cancellation broke at η={eta}: h[0,0]={:.3e} h[1,1]={:.3e} sum={:.3e}",
                hess[0][0],
                hess[1][1],
                sum,
            );
        }

        // Event rows, q0 = q1 = η, deep saturation: residual scales as
        // 1/η² by Mills asymptotic M(−η) = η + 1/η + O(1/η³).
        for &eta in &[40.0_f64, 100.0, 500.0, 988.0] {
            let (_nll, _grad, hess) = row_primary_closed_form(
                eta,
                eta,
                qd1,
                g,
                z,
                w,
                1.0,
                derivative_guard,
                probit_scale,
            )
            .expect("rigid event row");
            let sum = hess[0][0] + hess[1][1];
            let bound = 2.0 / (eta * eta);
            assert!(
                sum > 0.0 && sum <= bound,
                "event cancellation residual at η={eta}: sum={:.3e} expected (0, {:.3e}]",
                sum,
                bound,
            );
        }

        // Cross-check at η = 988 (the user's biobank-scale saturation):
        // both kinds of rows hit the predicted floor exactly.
        let (_, _, ev) =
            row_primary_closed_form(988.0, 988.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1e-6, 1.0).unwrap();
        let (_, _, ce) =
            row_primary_closed_form(988.0, 988.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1e-6, 1.0).unwrap();
        let ev_sum = ev[0][0] + ev[1][1];
        let ce_sum = ce[0][0] + ce[1][1];
        assert!(
            ev_sum > 0.0 && ev_sum < 2.0e-6,
            "event saturated h[0,0]+h[1,1] = {ev_sum:.3e}, expected ~1/988² ≈ 1e-6",
        );
        assert_eq!(
            ce_sum, 0.0,
            "censored saturated h[0,0]+h[1,1] must be EXACTLY 0, got {ce_sum:.3e}",
        );
    }

    #[test]
    fn row_primary_closed_form_rejects_nan_signed_margin() {
        let err = row_primary_closed_form(f64::NAN, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1e-6, 1.0)
            .expect_err("exact closed-form row should reject NaN signed margins");
        assert!(err.contains("non-finite signed margin"));
    }

    #[test]
    fn rigid_row_kernel_propagates_invalid_nonfinite_signed_margin_errors() {
        let mut family = test_family(None, None);
        family.offset_entry = Arc::new(array![f64::INFINITY]);
        family.offset_exit = Arc::new(array![0.0]);
        family.derivative_offset_exit = Arc::new(array![1.0]);
        family.event = Arc::new(array![1.0]);

        let kernel = SurvivalMarginalSlopeRowKernel::new(
            family,
            vec![
                ParameterBlockState {
                    beta: Array1::zeros(1),
                    eta: Array1::zeros(1),
                },
                ParameterBlockState {
                    beta: Array1::zeros(2),
                    eta: Array1::zeros(1),
                },
                ParameterBlockState {
                    beta: Array1::zeros(3),
                    eta: Array1::zeros(1),
                },
            ],
        );

        let err =
            <SurvivalMarginalSlopeRowKernel as crate::families::row_kernel::RowKernel<4>>::row_kernel(
                &kernel, 0,
            )
            .expect_err("row kernel should propagate exact probit boundary failures");
        assert!(err.contains("non-finite signed margin"));
    }

    #[test]
    fn rigid_row_kernel_propagates_nan_signed_margin_errors() {
        let mut family = test_family(None, None);
        family.offset_entry = Arc::new(array![f64::NAN]);
        family.offset_exit = Arc::new(array![0.0]);
        family.derivative_offset_exit = Arc::new(array![1.0]);
        family.event = Arc::new(array![1.0]);

        let kernel = SurvivalMarginalSlopeRowKernel::new(
            family,
            vec![
                ParameterBlockState {
                    beta: Array1::zeros(1),
                    eta: Array1::zeros(1),
                },
                ParameterBlockState {
                    beta: Array1::zeros(2),
                    eta: Array1::zeros(1),
                },
                ParameterBlockState {
                    beta: Array1::zeros(3),
                    eta: Array1::zeros(1),
                },
            ],
        );

        let err =
            <SurvivalMarginalSlopeRowKernel as crate::families::row_kernel::RowKernel<4>>::row_kernel(
                &kernel, 0,
            )
            .expect_err("row kernel should propagate NaN probit boundary failures");
        assert!(err.contains("non-finite signed margin"));
    }

    #[test]
    fn exact_flex_row_value_matches_rigid_with_zero_score_and_link_coefficients() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![0.9]),
            z: Arc::new(array![-0.35].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![-0.1]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.6]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.45],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let q_geom = family
            .row_dynamic_q_geometry(0, &block_states)
            .expect("row geometry");
        let primary = flex_primary_slices(&family);
        let (nll_exact, grad_exact, hess_exact) = family
            .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
            .expect("exact flex row");
        let (nll_rigid, grad_rigid, hess_rigid) = row_primary_closed_form(
            q_geom.q0,
            q_geom.q1,
            q_geom.qd1,
            block_states[2].eta[0],
            family.z[[0, 0]],
            family.weights[0],
            family.event[0],
            family.derivative_guard,
            family.probit_frailty_scale(),
        )
        .expect("rigid row");

        assert!((nll_exact - nll_rigid).abs() < 1e-10);
        assert!((grad_exact[primary.q0] - grad_rigid[0]).abs() < 1e-8);
        assert!((grad_exact[primary.q1] - grad_rigid[1]).abs() < 1e-8);
        assert!((grad_exact[primary.qd1] - grad_rigid[2]).abs() < 1e-8);
        assert!((grad_exact[primary.g] - grad_rigid[3]).abs() < 1e-8);
        assert!((hess_exact[[primary.q0, primary.q0]] - hess_rigid[0][0]).abs() < 1e-7);
        assert!((hess_exact[[primary.q0, primary.g]] - hess_rigid[0][3]).abs() < 1e-7);
        assert!((hess_exact[[primary.q1, primary.q1]] - hess_rigid[1][1]).abs() < 1e-7);
        assert!((hess_exact[[primary.q1, primary.g]] - hess_rigid[1][3]).abs() < 1e-7);
        assert!((hess_exact[[primary.qd1, primary.qd1]] - hess_rigid[2][2]).abs() < 1e-7);
        assert!((hess_exact[[primary.g, primary.g]] - hess_rigid[3][3]).abs() < 1e-7);
    }

    #[test]
    fn link_flex_family_supports_second_order_exact_outer_path() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(Array1::zeros(1)),
            offset_exit: Arc::new(Array1::zeros(1)),
            derivative_offset_exit: Arc::new(Array1::ones(1)),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let specs = vec![
            dummy_blockspec(1),
            dummy_blockspec(0),
            dummy_blockspec(score_runtime.basis_dim()),
            dummy_blockspec(link_runtime.basis_dim()),
        ];
        assert_eq!(
            family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn timewiggle_scorewarp_family_supports_second_order_exact_outer_path() {
        let score_runtime = test_deviation_runtime();
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 5))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 5))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 5))),
            offset_entry: Arc::new(Array1::zeros(1)),
            offset_exit: Arc::new(Array1::zeros(1)),
            derivative_offset_exit: Arc::new(Array1::ones(1)),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let specs = vec![
            dummy_blockspec(5),
            dummy_blockspec(0),
            dummy_blockspec(score_runtime.basis_dim()),
        ];
        assert_eq!(
            family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn survival_marginal_slope_advertises_outer_hvp_at_large_psi_dim() {
        let n = 2usize;
        let family = make_block_psi_test_family(n);
        let specs = vec![
            dummy_penalized_blockspec(0, 0),
            dummy_penalized_blockspec(1, 31),
            dummy_penalized_blockspec(1, 1),
        ];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            ..BlockwiseFitOptions::default()
        };

        let (gradient, hessian) = custom_family_outer_derivatives(&family, &specs, &options);

        assert!(family.inner_coefficient_hessian_hvp_available(&specs));
        assert!(family.outer_hyper_hessian_hvp_available(&specs));
        assert_eq!(
            family.exact_outer_derivative_order(&specs, &options),
            ExactOuterDerivativeOrder::Second
        );
        assert!(
            crate::solver::estimate::reml::unified::prefer_outer_hessian_operator(50_001, 2, 32,)
        );
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(
            hessian,
            crate::solver::outer_strategy::DeclaredHessianForm::Either
        );
    }

    #[test]
    fn survival_marginal_slope_coefficient_cost_uses_joint_coupled_formula() {
        // Rigid three-block shape: time p=12, marginal p=20, log-slope p=8.
        // The row kernel couples all three blocks, so the joint Hessian is
        // dense over (12+20+8)²=1600 entries per row. The override must
        // return n·(Σ p_b)², not the block-diagonal Σ n·p_b².
        let n = 200usize;
        let p_time = 12usize;
        let p_marg = 20usize;
        let p_log = 8usize;
        let family = SurvivalMarginalSlopeFamily {
            n,
            event: Arc::new(Array1::zeros(n)),
            weights: Arc::new(Array1::from_elem(n, 1.0)),
            z: Arc::new(Array1::zeros(n).insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((n, p_time))),
            design_exit: DesignMatrix::from(Array2::zeros((n, p_time))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((n, p_time))),
            offset_entry: Arc::new(Array1::zeros(n)),
            offset_exit: Arc::new(Array1::zeros(n)),
            derivative_offset_exit: Arc::new(Array1::ones(n)),
            marginal_design: DesignMatrix::from(Array2::zeros((n, p_marg))),
            logslope_design: DesignMatrix::from(Array2::zeros((n, p_log))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let specs = vec![
            dummy_penalized_blockspec(p_time, 1),
            dummy_penalized_blockspec(p_marg, 1),
            dummy_penalized_blockspec(p_log, 1),
        ];
        let p_total = (p_time + p_marg + p_log) as u64;
        let expected_joint = (n as u64) * p_total * p_total;
        let expected_block_diag_at_full_n =
            (n as u64) * ((p_time * p_time + p_marg * p_marg + p_log * p_log) as u64);
        assert_eq!(family.coefficient_hessian_cost(&specs), expected_joint);
        // Joint coupling exceeds block-diagonal by the cross-block fill
        // 2·n·(p_t·p_m + p_t·p_l + p_m·p_l).
        assert!(expected_joint > expected_block_diag_at_full_n);
    }

    #[test]
    fn exact_outer_row_work_gate_keeps_large_timewiggle_link_models_under_linear_flex_budget() {
        let link_runtime = test_deviation_runtime();
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 80,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 12))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 12))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 12))),
            offset_entry: Arc::new(Array1::zeros(1)),
            offset_exit: Arc::new(Array1::zeros(1)),
            derivative_offset_exit: Arc::new(Array1::ones(1)),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 20))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 20))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let specs = vec![
            dummy_penalized_blockspec(12, 2),
            dummy_penalized_blockspec(20, 2),
            dummy_penalized_blockspec(link_runtime.basis_dim(), 2),
        ];
        assert_eq!(
            family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
            ExactOuterDerivativeOrder::Second
        );
    }

    #[test]
    fn timewiggle_scorewarp_beta_hessian_directional_derivative_returns_finite_matrix() {
        let score_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.2],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let slices = block_slices(&family, &block_states);
        let mut d_beta_flat = Array1::zeros(slices.total);
        d_beta_flat[slices.time.start] = 0.07;
        d_beta_flat[slices.time.start + 1] = -0.03;
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;
        if let Some(h_range) = slices.score_warp.as_ref() {
            d_beta_flat[h_range.start] = 0.02;
        }

        let directional = family
            .exact_newton_joint_hessian_directional_derivative(&block_states, &d_beta_flat)
            .expect("timewiggle flex beta-Hessian directional derivative should evaluate")
            .expect("directional derivative should exist");
        assert_eq!(directional.nrows(), slices.total);
        assert_eq!(directional.ncols(), slices.total);
        assert!(directional.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn timewiggle_scorewarp_beta_hessian_second_directional_derivative_returns_finite_matrix() {
        let score_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.2],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let slices = block_slices(&family, &block_states);
        let mut d_beta_u = Array1::zeros(slices.total);
        let mut d_beta_v = Array1::zeros(slices.total);
        d_beta_u[slices.time.start] = 0.07;
        d_beta_u[slices.time.start + 1] = -0.03;
        d_beta_u[slices.marginal.start] = 0.05;
        d_beta_u[slices.logslope.start] = -0.04;
        d_beta_v[slices.time.start + 2] = 0.06;
        d_beta_v[slices.marginal.start + 1] = -0.02;
        d_beta_v[slices.logslope.start] = 0.03;
        if let Some(h_range) = slices.score_warp.as_ref() {
            d_beta_u[h_range.start] = 0.02;
            d_beta_v[h_range.start] = -0.01;
        }

        let second = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &block_states,
                &d_beta_u,
                &d_beta_v,
            )
            .expect("timewiggle flex beta-Hessian second directional derivative should evaluate")
            .expect("second directional derivative should exist");
        assert_eq!(second.nrows(), slices.total);
        assert_eq!(second.ncols(), slices.total);
        assert!(second.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn link_flex_bidirectional_timepoint_returns_finite_transport() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.2],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];

        let q_geom = family
            .row_dynamic_q_geometry(0, &block_states)
            .expect("row geometry");
        let primary = flex_primary_slices(&family);
        let g = block_states[2].eta[0];
        let beta_h = family.flex_score_beta(&block_states).expect("score beta");
        let beta_w = family.flex_link_beta(&block_states).expect("link beta");
        let (a1, _) = family
            .solve_row_survival_intercept(q_geom.q1, g, beta_h, beta_w)
            .expect("solve intercept");

        let mut dir_u = Array1::zeros(primary.total);
        let mut dir_v = Array1::zeros(primary.total);
        dir_u[primary.q1] = 0.07;
        dir_u[primary.g] = -0.04;
        dir_v[primary.q0] = 0.03;
        if let Some(h_range) = primary.h.as_ref() {
            dir_u[h_range.start] = 0.02;
        }
        if let Some(w_range) = primary.w.as_ref() {
            dir_v[w_range.start] = -0.01;
        }

        let active = family
            .compute_survival_timepoint_bidirectional_exact(
                0, &primary, q_geom.q1, primary.q1, a1, g, beta_h, beta_w, &dir_u, &dir_v,
            )
            .expect("active bidirectional transport");
        assert!(active.eta_uv_uv.iter().all(|value| value.is_finite()));
        assert!(active.chi_uv_uv.iter().all(|value| value.is_finite()));
        assert!(active.d_uv_uv.iter().all(|value| value.is_finite()));
        assert_eq!(active.eta_uv_uv.nrows(), primary.total);
        assert_eq!(active.eta_uv_uv.ncols(), primary.total);
        assert_eq!(active.chi_uv_uv.nrows(), primary.total);
        assert_eq!(active.chi_uv_uv.ncols(), primary.total);
        assert_eq!(active.d_uv_uv.nrows(), primary.total);
        assert_eq!(active.d_uv_uv.ncols(), primary.total);
        for u in 0..primary.total {
            for v in 0..primary.total {
                assert_eq!(active.eta_uv_uv[[u, v]], active.eta_uv_uv[[v, u]]);
                assert_eq!(active.chi_uv_uv[[u, v]], active.chi_uv_uv[[v, u]]);
                assert_eq!(active.d_uv_uv[[u, v]], active.d_uv_uv[[v, u]]);
            }
        }
    }

    #[test]
    fn link_flex_blockwise_exact_newton_matches_joint_principal_blocks() {
        assert!(file!().ends_with(".rs"));
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
        let logslope_design = array![[1.0], [0.5]];
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 0.8]),
            z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
            offset_entry: Arc::new(array![0.05, -0.02]),
            offset_exit: Arc::new(array![0.15, 0.08]),
            derivative_offset_exit: Arc::new(array![0.9, 1.1]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let marginal_beta = array![0.35, -0.1];
        let logslope_beta = array![0.2];
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
        ];

        assert_blockwise_matches_joint_principal_blocks(&family, &block_states);
    }

    #[test]
    fn link_flex_marginal_psi_terms_return_finite_joint_terms() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_beta = array![0.2];
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_beta.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(score_runtime.basis_dim()),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(link_runtime.basis_dim()),
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[1.0, -0.4]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            Vec::new(),
        ];

        let slices = block_slices(&family, &block_states);
        let terms = family
            .psi_terms(&block_states, &derivative_blocks, 0)
            .expect("link flex psi terms should evaluate")
            .expect("psi terms should exist");
        assert!(terms.objective_psi.is_finite());
        assert_eq!(terms.score_psi.len(), slices.total);
        assert!(terms.score_psi.iter().all(|value| value.is_finite()));
        assert!(terms.hessian_psi_operator.is_some());
    }

    #[test]
    fn link_flex_marginal_psi_second_order_returns_finite_joint_terms() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_design = array![[1.2, -0.3]];
        let logslope_beta = array![0.2, -0.05];
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[1.0, -0.4]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[0.3, 0.8]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            Vec::new(),
        ];

        let slices = block_slices(&family, &block_states);
        let terms = family
            .psi_second_order_terms(&block_states, &derivative_blocks, 0, 1)
            .expect("link flex psi second-order path should evaluate")
            .expect("psi second-order terms should exist");
        assert!(terms.objective_psi_psi.is_finite());
        assert_eq!(terms.score_psi_psi.len(), slices.total);
        assert!(terms.score_psi_psi.iter().all(|value| value.is_finite()));
        assert!(terms.hessian_psi_psi_operator.is_some());
    }

    #[test]
    fn link_flex_marginal_psi_hessian_directional_returns_finite_matrix() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.2],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[1.0, -0.4]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            Vec::new(),
        ];
        let slices = block_slices(&family, &block_states);
        let mut d_beta_flat = Array1::zeros(slices.total);
        d_beta_flat[slices.time.start] = 0.07;
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;
        if let Some(h_range) = slices.score_warp.as_ref() {
            d_beta_flat[h_range.start] = 0.02;
        }
        if let Some(w_range) = slices.link_dev.as_ref() {
            d_beta_flat[w_range.start] = -0.03;
        }

        let hess_dir = family
            .psi_hessian_directional_derivative(&block_states, &derivative_blocks, 0, &d_beta_flat)
            .expect("link flex psi-Hessian directional path should evaluate")
            .expect("psi-Hessian directional derivative should exist");
        assert_eq!(hess_dir.nrows(), slices.total);
        assert_eq!(hess_dir.ncols(), slices.total);
        assert!(hess_dir.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn timewiggle_marginal_psi_terms_return_finite_joint_terms() {
        let score_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.2],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[1.0, -0.4]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            Vec::new(),
        ];

        let slices = block_slices(&family, &block_states);
        let terms = family
            .psi_terms(&block_states, &derivative_blocks, 0)
            .expect("timewiggle psi terms should evaluate")
            .expect("psi terms should exist");
        assert!(terms.objective_psi.is_finite());
        assert_eq!(terms.score_psi.len(), slices.total);
        assert!(terms.score_psi.iter().all(|value| value.is_finite()));
        assert!(terms.hessian_psi_operator.is_some());
    }

    #[test]
    fn timewiggle_blockwise_exact_newton_matches_joint_principal_blocks() {
        assert!(file!().ends_with(".rs"));
        let score_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_beta = array![0.2];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 0.8]),
            z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_exit: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_derivative_exit: DesignMatrix::from(array![
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            offset_entry: Arc::new(array![0.05, -0.02]),
            offset_exit: Arc::new(array![0.15, 0.08]),
            derivative_offset_exit: Arc::new(array![0.9, 1.1]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0], [0.5]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: array![0.2, 0.1],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
        ];

        assert_blockwise_matches_joint_principal_blocks(&family, &block_states);
    }

    #[test]
    fn flex_timewiggle_fast_gradient_matches_dense_joint_gradient() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_design = array![[1.0]];
        let logslope_beta = array![0.2];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];

        let (fast_ll, fast_grad) = family
            .evaluate_exact_newton_joint_gradient_dynamic_q(&block_states)
            .expect("fast gradient should evaluate");
        let (dense_ll, dense_grad, _) = family
            .evaluate_exact_newton_joint_dynamic_q_dense(&block_states)
            .expect("dense joint derivatives should evaluate");

        assert_close(fast_ll, dense_ll, 1e-10, "log-likelihood");
        assert_eq!(fast_grad.len(), dense_grad.len());
        for idx in 0..fast_grad.len() {
            assert_close(
                fast_grad[idx],
                dense_grad[idx],
                1e-8,
                &format!("gradient[{idx}]"),
            );
        }
    }

    #[test]
    fn row_dynamic_q_geometry_into_pooled_matches_fresh_allocation_bitwise() {
        // Regression: `row_dynamic_q_geometry_into` reuses a caller-owned
        // `SurvivalMarginalSlopeDynamicRow` workspace (resized + zero-filled
        // in place) instead of allocating nine fresh Array2/Array1 buffers
        // per row. Both code paths must return bit-for-bit identical
        // contents, with the workspace path additionally verified to leave
        // the same answer when re-entered against an already-populated
        // buffer (so the in-place `reset` correctly wipes stale state).
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_design = array![[1.0], [0.5]];
        let logslope_beta = array![0.2];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 0.8]),
            z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_exit: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_derivative_exit: DesignMatrix::from(array![
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            offset_entry: Arc::new(array![0.05, -0.02]),
            offset_exit: Arc::new(array![0.15, 0.08]),
            derivative_offset_exit: Arc::new(array![0.9, 1.1]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
        ];

        // Compare each row, twice — first into a virgin workspace, then
        // into the same workspace re-used after a different row has
        // already populated it. Both must equal the fresh-allocation path
        // bit-for-bit.
        let mut workspace = SurvivalMarginalSlopeDynamicRow::empty_workspace();
        // Pre-load the workspace with row 1 so the row 0 comparison
        // exercises the buffer-reuse zeroing logic on a non-trivial state.
        family
            .row_dynamic_q_geometry_into(1, &block_states, &mut workspace)
            .expect("preload workspace with row 1");
        for row in [0usize, 1usize, 0usize] {
            let fresh = family
                .row_dynamic_q_geometry(row, &block_states)
                .expect("fresh-allocation row geometry");
            family
                .row_dynamic_q_geometry_into(row, &block_states, &mut workspace)
                .expect("pooled-workspace row geometry");
            assert_eq!(workspace.q0.to_bits(), fresh.q0.to_bits(), "row {row} q0");
            assert_eq!(workspace.q1.to_bits(), fresh.q1.to_bits(), "row {row} q1");
            assert_eq!(
                workspace.qd1.to_bits(),
                fresh.qd1.to_bits(),
                "row {row} qd1"
            );
            let array1_pairs: [(&Array1<f64>, &Array1<f64>, &str); 6] = [
                (&workspace.dq0_time, &fresh.dq0_time, "dq0_time"),
                (&workspace.dq1_time, &fresh.dq1_time, "dq1_time"),
                (&workspace.dqd1_time, &fresh.dqd1_time, "dqd1_time"),
                (&workspace.dq0_marginal, &fresh.dq0_marginal, "dq0_marginal"),
                (&workspace.dq1_marginal, &fresh.dq1_marginal, "dq1_marginal"),
                (
                    &workspace.dqd1_marginal,
                    &fresh.dqd1_marginal,
                    "dqd1_marginal",
                ),
            ];
            for (lhs, rhs, label) in array1_pairs {
                assert_eq!(lhs.shape(), rhs.shape(), "row {row} {label} shape");
                for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
                    assert_eq!(
                        l.to_bits(),
                        r.to_bits(),
                        "row {row} {label}[{i}] lhs={l:.17e} rhs={r:.17e}",
                    );
                }
            }
            let array2_pairs: [(&Array2<f64>, &Array2<f64>, &str); 9] = [
                (
                    &workspace.d2q0_time_time,
                    &fresh.d2q0_time_time,
                    "d2q0_time_time",
                ),
                (
                    &workspace.d2q1_time_time,
                    &fresh.d2q1_time_time,
                    "d2q1_time_time",
                ),
                (
                    &workspace.d2qd1_time_time,
                    &fresh.d2qd1_time_time,
                    "d2qd1_time_time",
                ),
                (
                    &workspace.d2q0_time_marginal,
                    &fresh.d2q0_time_marginal,
                    "d2q0_time_marginal",
                ),
                (
                    &workspace.d2q1_time_marginal,
                    &fresh.d2q1_time_marginal,
                    "d2q1_time_marginal",
                ),
                (
                    &workspace.d2qd1_time_marginal,
                    &fresh.d2qd1_time_marginal,
                    "d2qd1_time_marginal",
                ),
                (
                    &workspace.d2q0_marginal_marginal,
                    &fresh.d2q0_marginal_marginal,
                    "d2q0_marginal_marginal",
                ),
                (
                    &workspace.d2q1_marginal_marginal,
                    &fresh.d2q1_marginal_marginal,
                    "d2q1_marginal_marginal",
                ),
                (
                    &workspace.d2qd1_marginal_marginal,
                    &fresh.d2qd1_marginal_marginal,
                    "d2qd1_marginal_marginal",
                ),
            ];
            for (lhs, rhs, label) in array2_pairs {
                assert_eq!(lhs.shape(), rhs.shape(), "row {row} {label} shape");
                for ((idx, l), r) in lhs.indexed_iter().zip(rhs.iter()) {
                    assert_eq!(
                        l.to_bits(),
                        r.to_bits(),
                        "row {row} {label}[{idx:?}] lhs={l:.17e} rhs={r:.17e}",
                    );
                }
            }
        }
    }

    #[test]
    fn flex_timewiggle_operator_to_dense_matches_evaluate_dense_joint_hessian() {
        // Regression: SurvivalMarginalSlopeExactNewtonJointHessianWorkspace::
        // hessian_dense() now returns the already-built operator's
        // to_dense() instead of re-running
        // evaluate_exact_newton_joint_dynamic_q_dense (a second full n-row
        // sweep). Both code paths must agree on the joint p×p Hessian.
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_design = array![[1.0], [0.5]];
        let logslope_beta = array![0.2];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 0.8]),
            z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_exit: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_derivative_exit: DesignMatrix::from(array![
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            offset_entry: Arc::new(array![0.05, -0.02]),
            offset_exit: Arc::new(array![0.15, 0.08]),
            derivative_offset_exit: Arc::new(array![0.9, 1.1]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
        ];

        let (_, _, dense) = family
            .evaluate_exact_newton_joint_dynamic_q_dense(&block_states)
            .expect("dense joint Hessian");
        let (operator, _) = family
            .exact_newton_joint_hessian_operator(&block_states)
            .expect("joint Hessian operator");
        let op_dense = operator.to_dense();

        assert_eq!(op_dense.shape(), dense.shape());
        let diff = max_abs_diff_mat(&op_dense, &dense);
        assert!(
            diff <= 1e-10,
            "operator.to_dense() differs from evaluate_dense by {diff:.3e}",
        );
    }

    #[test]
    fn timewiggle_marginal_logslope_psi_second_order_returns_finite_joint_terms() {
        let score_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_design = array![[1.2, -0.3]];
        let logslope_beta = array![0.2, -0.05];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[1.0, -0.4]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[0.3, 0.8]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            Vec::new(),
        ];

        let slices = block_slices(&family, &block_states);
        let terms = family
            .psi_second_order_terms(&block_states, &derivative_blocks, 0, 1)
            .expect("timewiggle scorewarp psi second-order path should evaluate")
            .expect("psi second-order terms should exist");
        assert!(terms.objective_psi_psi.is_finite());
        assert_eq!(terms.score_psi_psi.len(), slices.total);
        assert!(terms.score_psi_psi.iter().all(|value| value.is_finite()));
        assert!(terms.hessian_psi_psi_operator.is_some());
    }

    #[test]
    fn timewiggle_marginal_psi_hessian_directional_returns_finite_matrix() {
        let score_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
            design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.2],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];
        let derivative_blocks = vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[1.0, -0.4]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            Vec::new(),
        ];
        let slices = block_slices(&family, &block_states);
        let mut d_beta_flat = Array1::zeros(slices.total);
        d_beta_flat[slices.time.start] = 0.07;
        d_beta_flat[slices.time.start + 1] = -0.03;
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;
        if let Some(h_range) = slices.score_warp.as_ref() {
            d_beta_flat[h_range.start] = 0.02;
        }

        let slices = block_slices(&family, &block_states);
        let hess_dir = family
            .psi_hessian_directional_derivative(&block_states, &derivative_blocks, 0, &d_beta_flat)
            .expect("timewiggle scorewarp psi-Hessian directional path should evaluate")
            .expect("psi-Hessian directional derivative should exist");
        assert_eq!(hess_dir.nrows(), slices.total);
        assert_eq!(hess_dir.ncols(), slices.total);
        assert!(hess_dir.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn sigma_exact_joint_psi_terms_returns_analytic_terms() {
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_beta = array![0.2];
        let sigma = 0.65;
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: Some(sigma),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_beta.clone(),
            },
        ];
        let specs = vec![
            dummy_blockspec(1),
            dummy_blockspec(marginal_design.ncols()),
            dummy_blockspec(1),
        ];

        let terms = family
            .sigma_exact_joint_psi_terms(&block_states, &specs)
            .expect("sigma psi terms should evaluate analytically")
            .expect("sigma psi terms should be present");
        assert!(terms.objective_psi.is_finite());
        assert_eq!(
            terms.score_psi.len(),
            block_slices(&family, &block_states).total
        );
        assert!(terms.score_psi.iter().all(|value| value.is_finite()));
        assert!(terms.hessian_psi_operator.is_some());
    }

    #[test]
    fn censored_rows_still_reject_invalid_time_derivative() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 1)),
            )),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 1)),
            )),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::ones((1, 1)),
            )),
            offset_entry: Arc::new(Array1::zeros(1)),
            offset_exit: Arc::new(Array1::zeros(1)),
            derivative_offset_exit: Arc::new(Array1::from_elem(1, 1e-6)),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
        ];

        let err = family
            .row_neglog_directional_refs(0, &block_states, &[])
            .expect_err("censored rows must still enforce the time-derivative domain");
        assert!(
            err.contains("monotonicity violated at row 0"),
            "unexpected error: {err}"
        );
    }

    fn assert_close(lhs: f64, rhs: f64, tol: f64, label: &str) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "{label} mismatch: lhs={lhs:.12e}, rhs={rhs:.12e}, tol={tol:.3e}"
        );
    }

    fn standard_test_time_wiggle() -> (Array1<f64>, usize, usize) {
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let degree = 3usize;
        let ncols = time_wiggle_basis_ncols(&knots, degree).expect("timewiggle basis width");
        (knots, degree, ncols)
    }

    fn assert_closed_form_row_matches_exact_directional_derivatives(
        family: SurvivalMarginalSlopeFamily,
        block_states: Vec<ParameterBlockState>,
    ) {
        let (nll_closed, grad_closed, hess_closed) = family
            .compute_row_primary_gradient_hessian_uncached(0, &block_states)
            .expect("closed-form row derivatives");
        let nll_exact = family
            .row_neglog_directional_refs(0, &block_states, &[])
            .expect("exact row objective");
        assert_close(nll_closed, nll_exact, 1e-12, "nll");

        for a in 0..N_PRIMARY {
            let dir_a = unit_primary_direction(a);
            let grad_exact = family
                .row_neglog_directional_refs(0, &block_states, &[dir_a.view()])
                .expect("exact row gradient");
            assert_close(grad_closed[a], grad_exact, 1e-10, &format!("grad[{a}]"));
            for b in 0..N_PRIMARY {
                let dir_b = unit_primary_direction(b);
                let hess_exact = family
                    .row_neglog_directional_refs(0, &block_states, &[dir_a.view(), dir_b.view()])
                    .expect("exact row hessian");
                assert_close(
                    hess_closed[[a, b]],
                    hess_exact,
                    1e-9,
                    &format!("hess[{a},{b}]"),
                );
            }
        }
    }

    #[test]
    fn closed_form_row_matches_exact_directional_derivatives() {
        assert!(file!().ends_with(".rs"));
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.2]),
            z: Arc::new(array![0.3].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.8]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.4]],
            )),
            offset_entry: Arc::new(array![0.1]),
            offset_exit: Arc::new(array![-0.2]),
            derivative_offset_exit: Arc::new(array![0.05]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.4],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![-0.1],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![0.7],
                eta: array![0.7],
            },
        ];
        assert_closed_form_row_matches_exact_directional_derivatives(family, block_states);
    }

    #[test]
    fn closed_form_row_matches_exact_directional_derivatives_with_gaussian_frailty() {
        assert!(file!().ends_with(".rs"));
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.2]),
            z: Arc::new(array![0.3].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: Some(0.65),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.8]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.4]],
            )),
            offset_entry: Arc::new(array![0.1]),
            offset_exit: Arc::new(array![-0.2]),
            derivative_offset_exit: Arc::new(array![0.05]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0
            ]])),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.4],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![-0.1],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![0.7],
                eta: array![0.7],
            },
        ];
        assert_closed_form_row_matches_exact_directional_derivatives(family, block_states);
    }

    #[test]
    fn closed_form_row_matches_exact_directional_derivatives_with_timewiggle() {
        assert!(file!().ends_with(".rs"));
        let time_wiggle_knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let time_wiggle_degree = 3;
        let base_q0 = array![0.09];
        let time_wiggle_ncols = monotone_wiggle_basis_with_derivative_order(
            base_q0.view(),
            &time_wiggle_knots,
            time_wiggle_degree,
            0,
        )
        .expect("timewiggle basis")
        .ncols();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.1]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: Some(0.4),
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![[0.2, 0.0, 0.0, 0.0]]),
            design_exit: DesignMatrix::from(array![[0.35, 0.0, 0.0, 0.0]]),
            design_derivative_exit: DesignMatrix::from(array![[1.1, 0.0, 0.0, 0.0]]),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.12]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(array![[0.6]]),
            logslope_design: DesignMatrix::from(array![[1.0]]),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.2, 0.08, -0.03, 0.02],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: array![0.05],
                eta: array![0.03],
            },
            ParameterBlockState {
                beta: array![0.35],
                eta: array![0.35],
            },
        ];
        assert_closed_form_row_matches_exact_directional_derivatives(family, block_states);
    }

    #[test]
    fn exact_newton_evaluation_propagates_invalid_rows() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![1e-6]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
        ];

        let err = family
            .evaluate(&block_states)
            .expect_err("invalid rows must abort exact-newton evaluation");
        assert!(
            err.contains("monotonicity violated"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn structural_time_constraints_use_derivative_lower_bound_rows() {
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![0.0, 1.0]),
            weights: Arc::new(array![1.0, 1.0]),
            z: Arc::new(array![0.0, 0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 2)),
            )),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 2)),
            )),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0, 2.0], [3.0, 4.0]],
            )),
            offset_entry: Arc::new(Array1::zeros(2)),
            offset_exit: Arc::new(Array1::zeros(2)),
            derivative_offset_exit: Arc::new(array![0.25, 0.5]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((2, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            time_linear_constraints: structural_time_coefficient_constraints(
                &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                    [1.0, 2.0],
                    [3.0, 4.0]
                ])),
                &array![0.25, 0.5],
                1e-4,
            )
            .expect("time coefficient constraints"),
            score_warp: None,
            link_dev: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let spec = ParameterBlockSpec {
            name: "time_surface".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [0.0, 0.0],
                [0.0, 0.0]
            ])),
            offset: Array1::zeros(2),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let constraints = family
            .block_linear_constraints(&[], 0, &spec)
            .expect("constraint lookup")
            .expect("time constraints");
        assert_eq!(constraints.a, Array2::<f64>::eye(2));
        assert_eq!(constraints.b, Array1::<f64>::zeros(2));
    }

    #[test]
    fn timewiggle_tail_constraints_are_part_of_time_block_feasibility() {
        let structural = structural_time_coefficient_constraints(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.0, 0.0
            ]])),
            &array![1e-6],
            1e-6,
        )
        .expect("structural constraints");
        let constraints = append_timewiggle_tail_nonnegative_constraints(structural, 3, 2)
            .expect("combined constraints")
            .expect("time constraints");

        assert_eq!(constraints.a, Array2::<f64>::eye(3));
        assert_eq!(constraints.b, Array1::<f64>::zeros(3));
    }

    #[test]
    fn timewiggle_tail_step_is_clipped_before_it_can_flip_derivative() {
        assert!(file!().ends_with(".rs"));
        let constraints = append_timewiggle_tail_nonnegative_constraints(None, 2, 1)
            .expect("tail constraints")
            .expect("time constraints");
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0, 0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0, 0.0
            ]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[0.0, 0.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![1e-6]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            time_linear_constraints: Some(constraints),
            score_warp: None,
            link_dev: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 1,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let states = vec![ParameterBlockState {
            beta: array![0.0, 0.5],
            eta: array![0.0],
        }];
        let alpha = family
            .max_feasible_step_size(&states, 0, &array![0.0, -1.0])
            .expect("timewiggle tail step ceiling")
            .expect("negative tail step should be bounded");
        assert_relative_eq!(alpha, 0.4975, epsilon = 1e-12);
    }

    #[test]
    fn time_block_post_update_projects_structural_and_timewiggle_constraints() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.0
            ]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0, 0.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![1e-6]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            time_linear_constraints: append_timewiggle_tail_nonnegative_constraints(
                structural_time_coefficient_constraints(
                    &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                        1.0, 0.0
                    ]])),
                    &array![1e-6],
                    1e-6,
                )
                .expect("time coefficient constraints"),
                2,
                1,
            )
            .expect("combined time constraints"),
            score_warp: None,
            link_dev: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 1,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let spec = ParameterBlockSpec {
            name: "time_surface".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0, 0.0]])),
            offset: Array1::zeros(1),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let beta = family
            .post_update_block_beta(
                &[ParameterBlockState {
                    beta: array![0.0, 0.0],
                    eta: array![0.0],
                }],
                0,
                &spec,
                array![-0.3, -0.2],
            )
            .expect("return time beta");
        assert_eq!(beta, array![0.0, 0.0]);
    }

    /// Reproduces the biobank-scale failure mode: no `time_linear_constraints`
    /// (no time-wiggle), so the only barrier between PIRLS and the
    /// `qd1 ≥ derivative_guard` cliff is `project_time_qd1_feasible`.
    /// The proposed β drives a row's qd1 from feasible (`+0.4`) deep into
    /// infeasibility (`-0.6`); the projection must pull it back to the
    /// feasible side with a strictly positive margin.
    #[test]
    fn time_block_post_update_projects_qd1_when_no_linear_constraints() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
            // qd1 = 1.0 · β[0] + 0.0 · β[1] + offset
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0, 0.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            // offset = derivative_guard exactly (the production setup).
            derivative_offset_exit: Arc::new(array![1e-6]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let spec = ParameterBlockSpec {
            name: "time_surface".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0, 0.0]])),
            offset: Array1::zeros(1),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let current = array![0.4, 7.0];
        // qd1 at current = 1.0·0.4 + 0.0·7.0 + 1e-6 ≈ 0.4 (feasible)
        // qd1 at proposed = 1.0·-0.6 + 0.0·-3.0 + 1e-6 ≈ -0.6 (infeasible)
        let proposed = array![-0.6, -3.0];
        let projected = family
            .post_update_block_beta(
                &[ParameterBlockState {
                    beta: current.clone(),
                    eta: array![0.0],
                }],
                0,
                &spec,
                proposed.clone(),
            )
            .expect("qd1 projection");
        // Verify: projected sits strictly above the guard, NOT at proposed.
        let qd1_projected = 1.0 * projected[0] + 0.0 * projected[1] + 1e-6;
        assert!(
            qd1_projected >= 1e-6,
            "projected qd1 must clear guard, got {qd1_projected:.3e}"
        );
        assert!(
            (projected[0] - proposed[0]).abs() > 1e-9,
            "projection must pull β[0] back from {} (proposed)",
            proposed[0]
        );
        // Bit-exact reproduction of the code path's max-α:
        //   qd1_current = D·current + offset = 0.4 + 1e-6
        //   qd1_proposed = D·proposed + offset = -0.6 + 1e-6
        //   drift = qd1_proposed − qd1_current = −1.0
        //   row_max = (qd1_current − guard) / −drift = (0.4 + 1e-6 − 1e-6) / 1.0 = 0.4
        // Then α_safe = 0.995 · 0.4 = 0.398, so β = current + 0.398·(proposed−current).
        let qd1_current_row = 1.0 * current[0] + 0.0 * current[1] + 1e-6;
        let qd1_proposed_row = 1.0 * proposed[0] + 0.0 * proposed[1] + 1e-6;
        let drift = qd1_proposed_row - qd1_current_row;
        let row_max = (qd1_current_row - 1e-6) / -drift;
        let alpha_safe = 0.995 * row_max;
        let expected_beta_0 = current[0] + alpha_safe * (proposed[0] - current[0]);
        let expected_beta_1 = current[1] + alpha_safe * (proposed[1] - current[1]);
        assert!(
            (projected[0] - expected_beta_0).abs() < 1e-12,
            "projected[0]={:.12} expected={expected_beta_0:.12}",
            projected[0]
        );
        assert!(
            (projected[1] - expected_beta_1).abs() < 1e-12,
            "projected[1]={:.12} expected={expected_beta_1:.12}",
            projected[1]
        );
    }

    /// When `current` already violates the qd1 monotonicity, the projection
    /// surfaces a structured error (with the row index and qd1 value)
    /// instead of silently returning a still-infeasible β. This is the
    /// invariant the score_warp / link_dev projection enforces; the time
    /// block now matches.
    #[test]
    fn time_block_post_update_errors_when_current_violates_qd1() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![1e-6]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let spec = ParameterBlockSpec {
            name: "time_surface".to_string(),
            design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
            offset: Array1::zeros(1),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        // current qd1 = -1.0 + 1e-6 < guard → infeasible.
        let err = family
            .post_update_block_beta(
                &[ParameterBlockState {
                    beta: array![-1.0],
                    eta: array![0.0],
                }],
                0,
                &spec,
                array![0.5],
            )
            .expect_err("infeasible current must surface an error");
        assert!(
            err.contains("violates monotonicity") && err.contains("row 0"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn time_block_feasible_step_stays_inside_derivative_guard() {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![0.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.0].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-4,
            design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0, 0.0
            ]])),
            design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                0.0, 0.0
            ]])),
            design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                array![[1.0, 0.0]],
            )),
            offset_entry: Arc::new(array![0.0]),
            offset_exit: Arc::new(array![0.0]),
            derivative_offset_exit: Arc::new(array![0.2]),
            marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 0)),
            )),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: structural_time_coefficient_constraints(
                &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0, 0.0]])),
                &array![0.2],
                1e-4,
            )
            .expect("time coefficient constraints"),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
        ];
        let alpha = family
            .max_feasible_step_size(&states, 0, &array![-1.0, 0.0])
            .expect("time step ceiling")
            .expect("time step should be bounded");
        assert_eq!(alpha, 0.0);
        let feasible = &states[0].beta + &(array![-1.0, 0.0] * alpha);
        let slack = family
            .time_linear_constraints
            .as_ref()
            .expect("constraints")
            .a
            .row(0)
            .dot(&feasible)
            - family
                .time_linear_constraints
                .as_ref()
                .expect("constraints")
                .b[0];
        assert!(slack >= 0.0);
    }

    #[test]
    fn mixed_blockwise_exact_newton_preserves_sparse_block_hessians() {
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 0.8]),
            z: Arc::new(array![0.1, -0.2].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.6]])),
            design_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![[0.9], [0.5]])),
            design_derivative_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![
                [1.0],
                [1.0]
            ])),
            offset_entry: Arc::new(array![0.0, 0.0]),
            offset_exit: Arc::new(array![0.0, 0.0]),
            derivative_offset_exit: Arc::new(array![0.05, 0.05]),
            marginal_design: sparse_design(&array![[1.0, 0.0], [0.0, 1.0]]),
            logslope_design: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.5]])),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.4],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.2, -0.1],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.3],
                eta: array![0.3, 0.3],
            },
        ];

        let eval = family
            .evaluate_blockwise_exact_newton(&block_states)
            .expect("mixed exact-newton evaluation");

        assert!(matches!(
            &eval.blockworking_sets[0],
            BlockWorkingSet::ExactNewton {
                hessian: SymmetricMatrix::Dense(_),
                ..
            }
        ));
        assert!(matches!(
            &eval.blockworking_sets[1],
            BlockWorkingSet::ExactNewton {
                hessian: SymmetricMatrix::Dense(_) | SymmetricMatrix::Sparse(_),
                ..
            }
        ));
        assert!(matches!(
            &eval.blockworking_sets[2],
            BlockWorkingSet::ExactNewton {
                hessian: SymmetricMatrix::Dense(_),
                ..
            }
        ));
    }

    /// Pairwise oracle for survival marginal-slope outer-HVP cross-checking.
    ///
    /// Builds a small survival fixture (n=1, no time-wiggle, no link-dev,
    /// no score-warp) with two ψ axes spanning the marginal and logslope
    /// blocks. Calls `psi_second_order_terms(i, j)` for every (i, j) pair,
    /// probes the operator returned via
    /// `psi_hessian_directional_derivative_operator` at each axis i with
    /// a fixed direction `d_beta_flat`, and writes the result to
    /// `/tmp/survival_pairwise_oracle.json`.
    ///
    /// Verification anchor for the operator-form survival outer-HVP that
    /// pairs the existing ψψ likelihood-side directional pullback with
    /// the directional trace-correction helpers being added under the
    /// CTN HVP Phase 2 work. The oracle uses only existing public APIs.
    #[test]
    fn survival_marginal_slope_pairwise_oracle_dumps_json() {
        let marginal_design = array![[0.7, -0.2]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_design = array![[1.0, 0.4]];
        let logslope_beta = array![0.2, -0.05];
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![1.0]),
            weights: Arc::new(array![1.0]),
            z: Arc::new(array![0.15].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
            offset_entry: Arc::new(array![0.05]),
            offset_exit: Arc::new(array![0.15]),
            derivative_offset_exit: Arc::new(array![0.9]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
        ];
        // Two ψ axes: one in marginal (block 1), one in logslope (block 2).
        let derivative_blocks = vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[1.0, -0.4]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                array![[0.6, 0.3]],
                Array2::zeros((2, 2)),
                None,
                None,
                None,
                None,
            )],
        ];

        let psi_dim = 2usize;

        let mut pair_records = Vec::new();
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                let terms = family
                    .psi_second_order_terms(&block_states, &derivative_blocks, i, j)
                    .expect("survival pairwise call ok")
                    .expect("pairwise returns Some for valid i,j");
                let g_inf = terms
                    .score_psi_psi
                    .iter()
                    .fold(0.0f64, |m, x| m.max(x.abs()));
                assert!(
                    terms.objective_psi_psi.is_finite(),
                    "objective_psi_psi non-finite at (i={i},j={j})"
                );
                assert!(
                    terms.score_psi_psi.iter().all(|v| v.is_finite()),
                    "score_psi_psi non-finite at (i={i},j={j})"
                );
                pair_records.push(serde_json::json!({
                    "i": i,
                    "j": j,
                    "a": terms.objective_psi_psi,
                    "g_inf": g_inf,
                    "g": terms.score_psi_psi.to_vec(),
                    "operator_present": terms.hessian_psi_psi_operator.is_some(),
                }));
            }
        }

        let slices = block_slices(&family, &block_states);
        let mut d_beta_flat = Array1::zeros(slices.total);
        d_beta_flat[slices.time.start] = 0.07;
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.marginal.start + 1] = -0.02;
        d_beta_flat[slices.logslope.start] = -0.04;
        d_beta_flat[slices.logslope.start + 1] = 0.03;

        let mut op_records = Vec::new();
        for i in 0..psi_dim {
            let op = family
                .psi_hessian_directional_derivative_operator_with_options(
                    &block_states,
                    &derivative_blocks,
                    i,
                    &d_beta_flat,
                    &BlockwiseFitOptions::default(),
                )
                .expect("operator call ok")
                .expect("operator returns Some");
            let dim = op.dim();
            let mut probes: Vec<(&'static str, Array1<f64>)> = Vec::new();
            let mut e0 = Array1::<f64>::zeros(dim);
            if dim > 0 {
                e0[0] = 1.0;
            }
            probes.push(("e0", e0));
            let scale = 1.0 / (dim.max(1) as f64).sqrt();
            probes.push(("uniform", Array1::from_elem(dim, scale)));
            let alt: Array1<f64> = (0..dim)
                .map(|k| if k % 2 == 0 { 0.5 } else { -0.3 })
                .collect();
            probes.push(("alt", alt));

            let mut probe_outputs = Vec::new();
            for (label, v) in probes.iter() {
                let out = op.mul_vec(v);
                let v_inf = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let out_inf = out.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                assert!(
                    out.iter().all(|x| x.is_finite()),
                    "operator output non-finite at axis {i} probe {label}"
                );
                probe_outputs.push(serde_json::json!({
                    "label": label,
                    "v_inf": v_inf,
                    "out_inf": out_inf,
                }));
            }
            op_records.push(serde_json::json!({
                "i": i,
                "dim": dim,
                "probes": probe_outputs,
            }));
        }

        let payload = serde_json::json!({
            "version": 1,
            "fixture": "survival_marginal_slope:n=1,no_wiggle,no_warp,psi_dim=2",
            "psi_dim": psi_dim,
            "p_total": slices.total,
            "pair_records": pair_records,
            "operator_records": op_records,
        });

        let path = std::path::Path::new("/tmp/survival_pairwise_oracle.json");
        std::fs::write(path, serde_json::to_string_pretty(&payload).unwrap())
            .expect("write oracle JSON");
        eprintln!(
            "[oracle] wrote {} pair records, {} operator records to {}",
            psi_dim * psi_dim,
            psi_dim,
            path.display()
        );
    }

    /// Closed-form test family with `gaussian_frailty_sd` set so the sigma-aware
    /// joint psi paths fire. Same pseudo-random row layout as
    /// `make_closed_form_test_family` so half-row subsamples remain
    /// representative.
    fn make_sigma_aware_closed_form_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
        let mut family = make_closed_form_test_family(n);
        family.gaussian_frailty_sd = Some(0.6);
        family
    }

    fn rel_diff_array1_survival(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let mut max = 0.0f64;
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs() / b[i].abs().max(1.0);
            if d > max {
                max = d;
            }
        }
        max
    }

    fn rel_diff_array2_survival(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        let mut max = 0.0f64;
        for ((i, j), &av) in a.indexed_iter() {
            let bv = b[[i, j]];
            let d = (av - bv).abs() / bv.abs().max(1.0);
            if d > max {
                max = d;
            }
        }
        max
    }

    #[test]
    fn survival_sigma_psi_terms_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_sigma_aware_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);
        let specs = vec![dummy_blockspec(0), dummy_blockspec(0), dummy_blockspec(0)];

        let baseline = family
            .sigma_exact_joint_psi_terms(&states, &specs)
            .expect("baseline psi terms")
            .expect("some");

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_full)
            .expect("with full mask")
            .expect("some");

        let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
            / baseline.objective_psi.abs().max(1.0))
        .abs();
        assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
        let score_rel = rel_diff_array1_survival(&with_full.score_psi, &baseline.score_psi);
        assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
    }

    #[test]
    fn survival_sigma_psi_terms_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_sigma_aware_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);
        let specs = vec![dummy_blockspec(0), dummy_blockspec(0), dummy_blockspec(0)];

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_half)
            .expect("scaled")
            .expect("some");

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_raw)
            .expect("raw")
            .expect("some");

        let factor = n as f64 / m as f64;
        let exp_obj = factor * raw.objective_psi;
        let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
        assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
        let exp_score = &raw.score_psi * factor;
        let score_rel = rel_diff_array1_survival(&scaled.score_psi, &exp_score);
        assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
    }

    #[test]
    fn survival_sigma_psi_second_order_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_sigma_aware_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);

        let baseline = family
            .sigma_exact_joint_psisecond_order_terms(&states)
            .expect("baseline")
            .expect("some");

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_full)
            .expect("with full mask")
            .expect("some");

        let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
            / baseline.objective_psi_psi.abs().max(1.0))
        .abs();
        assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
        let score_rel = rel_diff_array1_survival(&with_full.score_psi_psi, &baseline.score_psi_psi);
        assert!(score_rel < 1e-12, "score rel {}", score_rel);
    }

    #[test]
    fn survival_sigma_psi_second_order_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_sigma_aware_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_half)
            .expect("scaled")
            .expect("some");

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_raw)
            .expect("raw")
            .expect("some");

        let factor = n as f64 / m as f64;
        let exp_obj = factor * raw.objective_psi_psi;
        let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
        assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
        let exp_score = &raw.score_psi_psi * factor;
        let score_rel = rel_diff_array1_survival(&scaled.score_psi_psi, &exp_score);
        assert!(score_rel < 1e-12, "score rel {}", score_rel);
    }

    #[test]
    fn survival_sigma_psihessian_directional_derivative_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_sigma_aware_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);
        let slices = block_slices(&family, &states);
        let d_beta_flat = Array1::<f64>::zeros(slices.total);

        let baseline = family
            .sigma_exact_joint_psihessian_directional_derivative(&states, &d_beta_flat)
            .expect("baseline")
            .expect("some");

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .sigma_exact_joint_psihessian_directional_derivative_with_options(
                &states,
                &d_beta_flat,
                &opts_full,
            )
            .expect("with full")
            .expect("some");

        let rel = rel_diff_array2_survival(&with_full, &baseline);
        assert!(rel < 1e-12, "drift rel {}", rel);
    }

    #[test]
    fn survival_sigma_psihessian_directional_derivative_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_sigma_aware_closed_form_test_family(n);
        let states = closed_form_block_states(&family, 0.25);
        let slices = block_slices(&family, &states);
        let d_beta_flat = Array1::<f64>::zeros(slices.total);

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .sigma_exact_joint_psihessian_directional_derivative_with_options(
                &states,
                &d_beta_flat,
                &opts_half,
            )
            .expect("scaled")
            .expect("some");

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .sigma_exact_joint_psihessian_directional_derivative_with_options(
                &states,
                &d_beta_flat,
                &opts_raw,
            )
            .expect("raw")
            .expect("some");

        let factor = n as f64 / m as f64;
        let exp = &raw * factor;
        let rel = rel_diff_array2_survival(&scaled, &exp);
        assert!(rel < 1e-12, "drift rel {}", rel);
    }

    /// Multi-row test family with non-empty marginal/logslope designs but no
    /// score_warp / link_dev / time_wiggle. Drives the rigid block path of
    /// `psi_terms_inner` so we can subsample-check Horvitz-Thompson scaling.
    fn make_block_psi_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
        let event: Array1<f64> =
            Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
        let weights: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
        let z: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_entry: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_exit: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
        );
        let derivative_offset_exit: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
        // Single-column marginal/logslope designs with row-varying entries.
        let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
        });
        let logslope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
        });
        SurvivalMarginalSlopeFamily {
            n,
            event: Arc::new(event),
            weights: Arc::new(weights),
            z: Arc::new(z.insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            offset_entry: Arc::new(offset_entry),
            offset_exit: Arc::new(offset_exit),
            derivative_offset_exit: Arc::new(derivative_offset_exit),
            marginal_design: DesignMatrix::from(marginal_design),
            logslope_design: DesignMatrix::from(logslope_design),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: None,
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    }

    fn block_psi_test_block_states(
        family: &SurvivalMarginalSlopeFamily,
        m_beta: f64,
        g_beta: f64,
    ) -> Vec<ParameterBlockState> {
        let n = family.n;
        let m_design = family.marginal_design.to_dense().to_owned();
        let g_design = family.logslope_design.to_dense().to_owned();
        let m_eta = m_design.dot(&array![m_beta]);
        let g_eta = g_design.dot(&array![g_beta]);
        vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: array![m_beta],
                eta: m_eta,
            },
            ParameterBlockState {
                beta: array![g_beta],
                eta: g_eta,
            },
        ]
    }

    /// Derivative blocks with a single ψ on the marginal block (block_idx=1).
    /// `x_psi` has shape (n, 1) so the test family gets a per-row psi map.
    fn block_psi_test_marginal_derivative_blocks(
        n: usize,
    ) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
        let x_psi = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
        });
        vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                x_psi,
                Array2::zeros((1, 1)),
                None,
                None,
                None,
                None,
            )],
            Vec::new(),
        ]
    }

    /// Derivative blocks with one ψ on marginal (block 1) and one on logslope
    /// (block 2), so second-order terms can mix.
    fn block_psi_test_dual_derivative_blocks(
        n: usize,
    ) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
        let x_psi_m = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
        });
        let x_psi_g = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.2 + 0.5 * (((i * 43 + 17) % n) as f64) / (n as f64)
        });
        vec![
            Vec::new(),
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                x_psi_m,
                Array2::zeros((1, 1)),
                None,
                None,
                None,
                None,
            )],
            vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
                None,
                x_psi_g,
                Array2::zeros((1, 1)),
                None,
                None,
                None,
                None,
            )],
        ]
    }

    #[test]
    fn survival_psi_terms_inner_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);

        let baseline = family
            .psi_terms_inner(&states, &derivative_blocks, 0, None)
            .expect("baseline psi terms")
            .expect("some");

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_full)
            .expect("with full mask")
            .expect("some");

        let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
            / baseline.objective_psi.abs().max(1.0))
        .abs();
        assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
        let score_rel = rel_diff_array1_survival(&with_full.score_psi, &baseline.score_psi);
        assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
    }

    #[test]
    fn survival_psi_terms_inner_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_half)
            .expect("scaled")
            .expect("some");

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_raw)
            .expect("raw")
            .expect("some");

        let factor = n as f64 / m as f64;
        let exp_obj = factor * raw.objective_psi;
        let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
        assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
        let exp_score = &raw.score_psi * factor;
        let score_rel = rel_diff_array1_survival(&scaled.score_psi, &exp_score);
        assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
    }

    #[test]
    fn survival_psi_terms_inner_batched_matches_per_axis() {
        // The batched first-order ψ row pass shares the per-row primary
        // gradient/Hessian across axes; this asserts it produces the same
        // ExactNewtonJointPsiTerms (objective, score, Hessian-operator action)
        // as K serial calls to `psi_terms_inner_with_options`.
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_dual_derivative_blocks(n);
        let opts = BlockwiseFitOptions::default();

        let per_axis_0 = family
            .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts)
            .expect("per-axis 0")
            .expect("some");
        let per_axis_1 = family
            .psi_terms_inner_with_options(&states, &derivative_blocks, 1, None, &opts)
            .expect("per-axis 1")
            .expect("some");

        let batched = family
            .psi_terms_inner_batched_with_options(&states, &derivative_blocks, &[0, 1], None, &opts)
            .expect("batched")
            .expect("batched simple-spatial path returned None unexpectedly");
        assert_eq!(batched.len(), 2, "batched should yield one term per axis");

        let per_axis = [&per_axis_0, &per_axis_1];
        for (i, (lhs, rhs)) in per_axis.iter().zip(batched.iter()).enumerate() {
            let obj_rel =
                ((rhs.objective_psi - lhs.objective_psi) / lhs.objective_psi.abs().max(1.0)).abs();
            assert!(
                obj_rel < 1e-12,
                "axis {i} objective_psi rel {obj_rel} (per-axis={}, batched={})",
                lhs.objective_psi,
                rhs.objective_psi,
            );
            let score_rel = rel_diff_array1_survival(&rhs.score_psi, &lhs.score_psi);
            assert!(score_rel < 1e-12, "axis {i} score_psi rel {score_rel}");

            let op_a = lhs
                .hessian_psi_operator
                .as_ref()
                .expect("per-axis Hessian operator");
            let op_b = rhs
                .hessian_psi_operator
                .as_ref()
                .expect("batched Hessian operator");
            assert_eq!(op_a.dim(), op_b.dim(), "axis {i} operator dim mismatch");
            let dim = op_a.dim();
            let probe = Array1::from_shape_fn(dim, |j| {
                ((j as i64 * 37 + 11).rem_euclid(7)) as f64 * 0.1 - 0.3
            });
            let a = op_a.mul_vec(&probe);
            let b = op_b.mul_vec(&probe);
            let op_rel = rel_diff_array1_survival(&a, &b);
            assert!(op_rel < 1e-12, "axis {i} Hessian-action rel {op_rel}");
        }
    }

    #[test]
    fn survival_psi_terms_inner_batched_subsample_matches_per_axis() {
        // Same equivalence under a half-row Horvitz-Thompson mask, exercising
        // the per-row weight branch of the batched fast path.
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let mut opts = BlockwiseFitOptions::default();
        opts.outer_score_subsample =
            Some(Arc::new(OuterScoreSubsample::new(even_mask, n, 0xC0FFEE)));

        let per_axis_0 = family
            .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts)
            .expect("per-axis 0")
            .expect("some");
        let per_axis_1 = family
            .psi_terms_inner_with_options(&states, &derivative_blocks, 1, None, &opts)
            .expect("per-axis 1")
            .expect("some");

        let batched = family
            .psi_terms_inner_batched_with_options(&states, &derivative_blocks, &[0, 1], None, &opts)
            .expect("batched")
            .expect("batched simple-spatial path returned None under subsample");
        assert_eq!(batched.len(), 2);

        let per_axis = [&per_axis_0, &per_axis_1];
        for (i, (lhs, rhs)) in per_axis.iter().zip(batched.iter()).enumerate() {
            let obj_rel =
                ((rhs.objective_psi - lhs.objective_psi) / lhs.objective_psi.abs().max(1.0)).abs();
            assert!(
                obj_rel < 1e-12,
                "axis {i} subsample objective_psi rel {obj_rel}"
            );
            let score_rel = rel_diff_array1_survival(&rhs.score_psi, &lhs.score_psi);
            assert!(
                score_rel < 1e-12,
                "axis {i} subsample score_psi rel {score_rel}"
            );

            let op_a = lhs.hessian_psi_operator.as_ref().unwrap();
            let op_b = rhs.hessian_psi_operator.as_ref().unwrap();
            let dim = op_a.dim();
            let probe = Array1::from_shape_fn(dim, |j| {
                ((j as i64 * 41 + 5).rem_euclid(11)) as f64 * 0.07 - 0.4
            });
            let a = op_a.mul_vec(&probe);
            let b = op_b.mul_vec(&probe);
            let op_rel = rel_diff_array1_survival(&a, &b);
            assert!(
                op_rel < 1e-12,
                "axis {i} subsample Hessian-action rel {op_rel}"
            );
        }
    }

    #[test]
    fn survival_psi_second_order_terms_inner_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

        let baseline = family
            .psi_second_order_terms_inner(&states, &derivative_blocks, 0, 1, None)
            .expect("baseline psi second-order")
            .expect("some");

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .psi_second_order_terms_inner_with_options(
                &states,
                &derivative_blocks,
                0,
                1,
                None,
                &opts_full,
            )
            .expect("with full")
            .expect("some");

        let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
            / baseline.objective_psi_psi.abs().max(1.0))
        .abs();
        assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
        let score_rel = rel_diff_array1_survival(&with_full.score_psi_psi, &baseline.score_psi_psi);
        assert!(score_rel < 1e-12, "score rel {}", score_rel);
    }

    #[test]
    fn survival_psi_second_order_terms_inner_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .psi_second_order_terms_inner_with_options(
                &states,
                &derivative_blocks,
                0,
                1,
                None,
                &opts_half,
            )
            .expect("scaled")
            .expect("some");

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .psi_second_order_terms_inner_with_options(
                &states,
                &derivative_blocks,
                0,
                1,
                None,
                &opts_raw,
            )
            .expect("raw")
            .expect("some");

        let factor = n as f64 / m as f64;
        let exp_obj = factor * raw.objective_psi_psi;
        let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
        assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
        let exp_score = &raw.score_psi_psi * factor;
        let score_rel = rel_diff_array1_survival(&scaled.score_psi_psi, &exp_score);
        assert!(score_rel < 1e-12, "score rel {}", score_rel);
    }

    #[test]
    fn survival_psi_hessian_directional_derivative_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
        let slices = block_slices(&family, &states);
        let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;

        let baseline = family
            .psi_hessian_directional_derivative(&states, &derivative_blocks, 0, &d_beta_flat)
            .expect("baseline psi-Hessian directional derivative")
            .expect("some");

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .psi_hessian_directional_derivative_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &opts_full,
            )
            .expect("with full mask")
            .expect("some");

        let rel = rel_diff_array2_survival(&with_full, &baseline);
        assert!(rel < 1e-12, "drift rel {}", rel);
    }

    #[test]
    fn survival_psi_hessian_directional_derivative_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
        let slices = block_slices(&family, &states);
        let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .psi_hessian_directional_derivative_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &opts_half,
            )
            .expect("scaled")
            .expect("some");

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .psi_hessian_directional_derivative_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &opts_raw,
            )
            .expect("raw")
            .expect("some");

        let factor = n as f64 / m as f64;
        let exp = &raw * factor;
        let rel = rel_diff_array2_survival(&scaled, &exp);
        assert!(rel < 1e-12, "drift rel {}", rel);
    }

    #[test]
    fn survival_psi_workspace_hessian_directional_derivative_is_operator_and_matches_dense() {
        let n = 40usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
        let specs = vec![dummy_blockspec(0), dummy_blockspec(1), dummy_blockspec(1)];
        let slices = block_slices(&family, &states);
        let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;

        let dense = family
            .psi_hessian_directional_derivative_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &BlockwiseFitOptions::default(),
            )
            .expect("dense drift")
            .expect("dense drift available");
        let workspace = family
            .exact_newton_joint_psi_workspace_with_options(
                &states,
                &specs,
                &derivative_blocks,
                &BlockwiseFitOptions::default(),
            )
            .expect("workspace")
            .expect("workspace available");
        let result = workspace
            .hessian_directional_derivative(0, &d_beta_flat)
            .expect("workspace drift")
            .expect("workspace drift available");

        let crate::solver::estimate::reml::unified::DriftDerivResult::Operator(op) = result else {
            panic!("survival psi drift should use operator representation");
        };
        assert_eq!(op.dim(), dense.nrows());
        let operator_dense = op.to_dense();
        let rel = rel_diff_array2_survival(&operator_dense, &dense);
        assert!(rel < 1e-12, "operator/dense drift rel {rel}");
    }

    #[test]
    fn survival_psi_hessian_directional_derivative_operator_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
        let slices = block_slices(&family, &states);
        let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;

        let baseline = family
            .psi_hessian_directional_derivative_operator_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &BlockwiseFitOptions::default(),
            )
            .expect("baseline operator")
            .expect("some");
        let baseline_dense = baseline.to_dense();

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .psi_hessian_directional_derivative_operator_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &opts_full,
            )
            .expect("with full mask")
            .expect("some");
        let with_full_dense = with_full.to_dense();

        let rel = rel_diff_array2_survival(&with_full_dense, &baseline_dense);
        assert!(rel < 1e-12, "operator drift rel {}", rel);
    }

    #[test]
    fn survival_psi_hessian_directional_derivative_operator_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 200usize;
        let family = make_block_psi_test_family(n);
        let states = block_psi_test_block_states(&family, 0.15, 0.25);
        let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
        let slices = block_slices(&family, &states);
        let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .psi_hessian_directional_derivative_operator_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &opts_half,
            )
            .expect("scaled")
            .expect("some");
        let scaled_dense = scaled.to_dense();

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .psi_hessian_directional_derivative_operator_with_options(
                &states,
                &derivative_blocks,
                0,
                &d_beta_flat,
                &opts_raw,
            )
            .expect("raw")
            .expect("some");
        let raw_dense = raw.to_dense();

        let factor = n as f64 / m as f64;
        let exp = &raw_dense * factor;
        let rel = rel_diff_array2_survival(&scaled_dense, &exp);
        assert!(rel < 1e-12, "operator drift rel {}", rel);
    }

    // ── Phase 7: joint-Hessian flex-no-wiggle directional-derivative
    // operator subsample tests. The flex-no-wiggle helpers are the path
    // taken by the joint-Hessian workspace's `directional_derivative_operator`
    // when `effective_flex_active(states)` is true and timewiggle is off.
    // We exercise the helpers directly through a flex-active fixture so the
    // outer subsample threading is verified end-to-end.

    fn make_flex_no_wiggle_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
        let score_runtime = test_deviation_runtime();
        let event: Array1<f64> =
            Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
        let weights: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
        let z: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_entry: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
        );
        let offset_exit: Array1<f64> = Array1::from_iter(
            (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
        );
        let derivative_offset_exit: Array1<f64> =
            Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
        let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
        });
        let logslope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
            0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
        });
        SurvivalMarginalSlopeFamily {
            n,
            event: Arc::new(event),
            weights: Arc::new(weights),
            z: Arc::new(z.insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
            design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
            offset_entry: Arc::new(offset_entry),
            offset_exit: Arc::new(offset_exit),
            derivative_offset_exit: Arc::new(derivative_offset_exit),
            marginal_design: DesignMatrix::from(marginal_design),
            logslope_design: DesignMatrix::from(logslope_design),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime),
            link_dev: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        }
    }

    fn flex_no_wiggle_test_block_states(
        family: &SurvivalMarginalSlopeFamily,
    ) -> Vec<ParameterBlockState> {
        let n = family.n;
        let m_design = family.marginal_design.to_dense().to_owned();
        let g_design = family.logslope_design.to_dense().to_owned();
        let m_beta = 0.15_f64;
        let g_beta = 0.25_f64;
        let m_eta = m_design.dot(&array![m_beta]);
        let g_eta = g_design.dot(&array![g_beta]);
        let score_dim = family
            .score_warp
            .as_ref()
            .map(|w| w.basis_dim())
            .unwrap_or(0);
        vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: array![m_beta],
                eta: m_eta,
            },
            ParameterBlockState {
                beta: array![g_beta],
                eta: g_eta,
            },
            ParameterBlockState {
                beta: Array1::zeros(score_dim),
                eta: Array1::zeros(n),
            },
        ]
    }

    #[test]
    fn survival_jointhessian_flex_no_wiggle_operator_subsample_full_equals_unsampled() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 40usize;
        let family = make_flex_no_wiggle_test_family(n);
        let states = flex_no_wiggle_test_block_states(&family);
        assert!(family.effective_flex_active(&states).unwrap());
        assert!(!family.flex_timewiggle_active());
        let slices = block_slices(&family, &states);
        let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;

        let baseline = family
            .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
                &states,
                &d_beta_flat,
                &BlockwiseFitOptions::default(),
            )
            .expect("baseline operator");
        let baseline_dense = baseline.to_dense();

        let mut opts_full = BlockwiseFitOptions::default();
        opts_full.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            (0..n).collect(),
            n,
            0xDEADBEEF,
        )));
        let with_full = family
            .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
                &states,
                &d_beta_flat,
                &opts_full,
            )
            .expect("with full mask");
        let with_full_dense = with_full.to_dense();

        let rel = rel_diff_array2_survival(&with_full_dense, &baseline_dense);
        assert!(
            rel < 1e-10,
            "joint Hessian flex-no-wiggle dH operator drift rel {}",
            rel
        );
    }

    #[test]
    fn survival_jointhessian_flex_no_wiggle_operator_subsample_half_scales_correctly() {
        use crate::families::marginal_slope_shared::OuterScoreSubsample;
        let n = 40usize;
        let family = make_flex_no_wiggle_test_family(n);
        let states = flex_no_wiggle_test_block_states(&family);
        assert!(family.effective_flex_active(&states).unwrap());
        assert!(!family.flex_timewiggle_active());
        let slices = block_slices(&family, &states);
        let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
        d_beta_flat[slices.marginal.start] = 0.05;
        d_beta_flat[slices.logslope.start] = -0.04;

        let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
        let m = even_mask.len();

        let mut opts_half = BlockwiseFitOptions::default();
        opts_half.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::new(
            even_mask.clone(),
            n,
            0xCAFE,
        )));
        let scaled = family
            .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
                &states,
                &d_beta_flat,
                &opts_half,
            )
            .expect("scaled");
        let scaled_dense = scaled.to_dense();

        let mut opts_raw = BlockwiseFitOptions::default();
        opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
            even_mask, m, 0, 1.0,
        )));
        let raw = family
            .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
                &states,
                &d_beta_flat,
                &opts_raw,
            )
            .expect("raw");
        let raw_dense = raw.to_dense();

        let factor = n as f64 / m as f64;
        let exp = &raw_dense * factor;
        let rel = rel_diff_array2_survival(&scaled_dense, &exp);
        assert!(
            rel < 1e-10,
            "joint Hessian flex-no-wiggle dH operator HT rel {}",
            rel
        );
    }

    /// Build a small flex fixture wired to a fresh `SurvivalInterceptWarmStartCache`.
    /// `n` is small so we can iterate over every row in the assertion loop, and
    /// the score-warp deviation is configured non-trivially so the calibration
    /// solver has to actually iterate (no exact-zero short-circuit applies in
    /// survival: there's no degenerate fast path equivalent to bernoulli's
    /// `exact_zero_deviation`, so every cold solve goes through the bracket).
    fn make_warm_start_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
        let mut family = make_flex_no_wiggle_test_family(n);
        family.intercept_warm_starts = Some(new_intercept_warm_start_cache(n));
        family
    }

    /// Exercise the warm-start cache end-to-end on the gradient/Hessian path:
    /// 1. Solve every row's (entry, exit) intercepts on a cold family to get
    ///    the exact converged `a` values.
    /// 2. Build a warm family with the cache pre-populated to those values.
    /// 3. Build a cold family with the cache initialised but empty (NaN).
    /// 4. Assert the per-row (NLL, gradient, Hessian) produced by both
    ///    families are numerically identical to ~1e-12 relative — the cache
    ///    must not perturb the answer.
    /// 5. Assert the warm family's evaluator-call count is strictly lower
    ///    than the cold family's — if warm starts are silently bypassed
    ///    (e.g., None slot threaded through) the counts would match and this
    ///    assertion would fire.
    /// Build flex block states with a *non-trivial* score-warp coefficient
    /// vector so the cold closed-form rigid seed is meaningfully off-root.
    /// Without this, beta_h = 0 leaves the score warp as the identity and the
    /// rigid seed equals the converged intercept exactly — the cold and warm
    /// paths both finish in one evaluator call and the count assertion can't
    /// distinguish them. We perturb a single basis coordinate by a small
    /// amount (well inside the monotone-feasible region for the test basis).
    fn flex_no_wiggle_perturbed_test_block_states(
        family: &SurvivalMarginalSlopeFamily,
    ) -> Vec<ParameterBlockState> {
        let mut states = flex_no_wiggle_test_block_states(family);
        if let Some(score_state) = states.get_mut(3) {
            if score_state.beta.len() >= 2 {
                score_state.beta[0] = 0.18;
                score_state.beta[1] = -0.12;
            } else if !score_state.beta.is_empty() {
                score_state.beta[0] = 0.18;
            }
        }
        states
    }

    #[test]
    fn survival_intercept_warm_starts_match_cold_and_reduce_eval_count() {
        let n = 8usize;
        let states =
            flex_no_wiggle_perturbed_test_block_states(&make_flex_no_wiggle_test_family(n));

        // Step 1: cold family without any cache — capture the converged (a0, a1)
        // per row so we can pre-populate the warm cache.
        let cold_no_cache = make_flex_no_wiggle_test_family(n);
        assert!(cold_no_cache.intercept_warm_starts.is_none());
        assert!(cold_no_cache.effective_flex_active(&states).unwrap());
        let beta_h = cold_no_cache.flex_score_beta(&states).unwrap();
        let beta_w = cold_no_cache.flex_link_beta(&states).unwrap();
        let g_block = &states[2].eta;

        let mut q0_per_row = Vec::with_capacity(n);
        let mut q1_per_row = Vec::with_capacity(n);
        let mut qd1_per_row = Vec::with_capacity(n);
        let mut a0_per_row = Vec::with_capacity(n);
        let mut a1_per_row = Vec::with_capacity(n);
        for row in 0..n {
            let q_geom = cold_no_cache.row_dynamic_q_values(row, &states).unwrap();
            q0_per_row.push(q_geom.q0);
            q1_per_row.push(q_geom.q1);
            qd1_per_row.push(q_geom.qd1);
            let (a0, _) = cold_no_cache
                .solve_row_survival_intercept(q_geom.q0, g_block[row], beta_h, beta_w)
                .unwrap();
            let (a1, _) = cold_no_cache
                .solve_row_survival_intercept(q_geom.q1, g_block[row], beta_h, beta_w)
                .unwrap();
            a0_per_row.push(a0);
            a1_per_row.push(a1);
        }

        // Step 2: warm family with the cache pre-populated to the converged
        // intercepts. The solver should accept these as the seed and converge
        // in ~0–1 refine iters instead of the cold bracket walk.
        let warm_family = make_warm_start_test_family(n);
        {
            let cache = warm_family
                .intercept_warm_starts
                .as_ref()
                .expect("warm cache present");
            for row in 0..n {
                cache.store(row, SurvivalInterceptSlotKind::Entry, a0_per_row[row]);
                cache.store(row, SurvivalInterceptSlotKind::Exit, a1_per_row[row]);
            }
        }

        // Step 3: cold family with cache attached but NaN-initialised. This
        // exercises the warm-start code path on the cold-start side, so any
        // diff vs the warm-populated family isolates the seed behaviour.
        let cold_family = make_warm_start_test_family(n);

        let primary = flex_primary_slices(&warm_family);

        // Step 4: per-row (NLL, gradient, Hessian) must match to ~1e-12 rel.
        survival_intercept_test_counter::reset();
        let mut warm_results = Vec::with_capacity(n);
        for row in 0..n {
            let res = warm_family
                .compute_row_flex_primary_gradient_hessian_from_parts(
                    row,
                    q0_per_row[row],
                    q1_per_row[row],
                    qd1_per_row[row],
                    g_block[row],
                    beta_h,
                    beta_w,
                    &primary,
                )
                .expect("warm gradient/hessian");
            warm_results.push(res);
        }
        let warm_evals = survival_intercept_test_counter::get();

        survival_intercept_test_counter::reset();
        let mut cold_results = Vec::with_capacity(n);
        for row in 0..n {
            let res = cold_family
                .compute_row_flex_primary_gradient_hessian_from_parts(
                    row,
                    q0_per_row[row],
                    q1_per_row[row],
                    qd1_per_row[row],
                    g_block[row],
                    beta_h,
                    beta_w,
                    &primary,
                )
                .expect("cold gradient/hessian");
            cold_results.push(res);
        }
        let cold_evals = survival_intercept_test_counter::get();

        for row in 0..n {
            let (warm_nll, warm_grad, warm_hess) = &warm_results[row];
            let (cold_nll, cold_grad, cold_hess) = &cold_results[row];

            let nll_rel =
                (warm_nll - cold_nll).abs() / (1e-300_f64).max(cold_nll.abs().max(warm_nll.abs()));
            assert!(
                nll_rel <= 1e-12,
                "row {row}: warm NLL {warm_nll:.17e} vs cold NLL {cold_nll:.17e}, rel {nll_rel:.3e}",
            );

            let grad_max_abs = cold_grad.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            let grad_rel = warm_grad
                .iter()
                .zip(cold_grad.iter())
                .map(|(w, c)| (w - c).abs())
                .fold(0.0_f64, f64::max)
                / (1e-300_f64).max(grad_max_abs);
            assert!(
                grad_rel <= 1e-12,
                "row {row}: warm vs cold gradient drift rel {grad_rel:.3e}",
            );

            let hess_max_abs = cold_hess.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            let hess_rel = warm_hess
                .iter()
                .zip(cold_hess.iter())
                .map(|(w, c)| (w - c).abs())
                .fold(0.0_f64, f64::max)
                / (1e-300_f64).max(hess_max_abs);
            assert!(
                hess_rel <= 1e-12,
                "row {row}: warm vs cold Hessian drift rel {hess_rel:.3e}",
            );
        }

        // Step 5: warm-start path must spend strictly fewer evaluator calls
        // than the cold path. If a future refactor accidentally threads `None`
        // through `solve_row_survival_intercept_with_slot` (silently bypassing
        // the cache), these two counts would be identical and the test fails
        // loudly. The exact ratio depends on the bracket-walk dynamics but
        // pre-seeding at the converged root collapses the safeguarded refine
        // loop into a 1–2-step Newton confirmation.
        assert!(
            warm_evals > 0,
            "warm-start path made zero evaluator calls — counter not wired?"
        );
        assert!(
            cold_evals > 0,
            "cold-start path made zero evaluator calls — counter not wired?"
        );
        assert!(
            warm_evals < cold_evals,
            "warm-start eval count {warm_evals} is not lower than cold {cold_evals}; \
             warm starts may be silently bypassed",
        );
    }

    /// Regression: the batched k=5 / k=6 jet path for `row_primary_third_contracted`
    /// and `row_primary_fourth_contracted` must agree to ~1e-12 relative with
    /// the legacy 10-call path that built a fresh k=3 / k=4 jet per
    /// upper-triangular (a, b) cell. Exercises both score_warp and link_dev
    /// in the same fixture as the operator-vs-dense regression so the
    /// composed-unary derivative chain is non-trivial.
    #[test]
    fn row_third_and_fourth_contracted_batched_matches_legacy_per_cell() {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
        let marginal_beta = array![0.35, -0.1];
        let logslope_design = array![[1.0], [0.5]];
        let logslope_beta = array![0.2];
        let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) =
            standard_test_time_wiggle();
        let family = SurvivalMarginalSlopeFamily {
            n: 2,
            event: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 0.8]),
            z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_exit: DesignMatrix::from(array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            design_derivative_exit: DesignMatrix::from(array![
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0]
            ]),
            offset_entry: Arc::new(array![0.05, -0.02]),
            offset_exit: Arc::new(array![0.15, 0.08]),
            derivative_offset_exit: Arc::new(array![0.9, 1.1]),
            marginal_design: DesignMatrix::from(marginal_design.clone()),
            logslope_design: DesignMatrix::from(logslope_design.clone()),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            time_linear_constraints: None,
            time_wiggle_knots: Some(time_wiggle_knots),
            time_wiggle_degree: Some(time_wiggle_degree),
            time_wiggle_ncols,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let block_states = vec![
            ParameterBlockState {
                beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
                eta: array![0.0, 0.0],
            },
            ParameterBlockState {
                beta: marginal_beta.clone(),
                eta: marginal_design.dot(&marginal_beta),
            },
            ParameterBlockState {
                beta: logslope_beta.clone(),
                eta: logslope_design.dot(&logslope_beta),
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(2),
            },
        ];

        // Use two non-axis-aligned, non-orthogonal directions so the
        // contracted tensors stress every term of the Faà di Bruno expansion
        // (especially the diagonal cells that hit a degenerate `[e_a, e_a, …]`
        // jet in the batched path).
        let dir_u = array![0.31, -0.42, 0.18, 0.27];
        let dir_v = array![-0.11, 0.55, -0.07, 0.14];

        for row in 0..family.n {
            // Legacy third: 10 k=3 jet calls.
            let mut legacy_third = [[0.0_f64; 4]; 4];
            for a in 0..N_PRIMARY {
                let da = unit_primary_direction_ref(a).view();
                for b in a..N_PRIMARY {
                    let db = unit_primary_direction_ref(b).view();
                    let value = family
                        .row_neglog_directional_refs(row, &block_states, &[da, db, dir_u.view()])
                        .expect("legacy third per-cell");
                    legacy_third[a][b] = value;
                    legacy_third[b][a] = value;
                }
            }
            let batched_third = family
                .row_primary_third_contracted_batched(row, &block_states, dir_u.view())
                .expect("batched third");
            for a in 0..N_PRIMARY {
                for b in 0..N_PRIMARY {
                    let l = legacy_third[a][b];
                    let r = batched_third[a][b];
                    let denom = l.abs().max(1.0);
                    let rel = (l - r).abs() / denom;
                    assert!(
                        rel <= 1e-12,
                        "row {row} third[{a}][{b}]: legacy={l:.17e} batched={r:.17e} rel={rel:.3e}",
                    );
                }
            }

            // Legacy fourth: 10 k=4 jet calls.
            let mut legacy_fourth = [[0.0_f64; 4]; 4];
            for a in 0..N_PRIMARY {
                let da = unit_primary_direction_ref(a).view();
                for b in a..N_PRIMARY {
                    let db = unit_primary_direction_ref(b).view();
                    let value = family
                        .row_neglog_directional_refs(
                            row,
                            &block_states,
                            &[da, db, dir_u.view(), dir_v.view()],
                        )
                        .expect("legacy fourth per-cell");
                    legacy_fourth[a][b] = value;
                    legacy_fourth[b][a] = value;
                }
            }
            let batched_fourth = family
                .row_primary_fourth_contracted_batched(
                    row,
                    &block_states,
                    dir_u.view(),
                    dir_v.view(),
                )
                .expect("batched fourth");
            for a in 0..N_PRIMARY {
                for b in 0..N_PRIMARY {
                    let l = legacy_fourth[a][b];
                    let r = batched_fourth[a][b];
                    let denom = l.abs().max(1.0);
                    let rel = (l - r).abs() / denom;
                    assert!(
                        rel <= 1e-12,
                        "row {row} fourth[{a}][{b}]: legacy={l:.17e} batched={r:.17e} rel={rel:.3e}",
                    );
                }
            }
        }
    }

    #[test]
    fn survival_auto_subsample_phase_counter_field_initializes_to_zero() {
        let family = make_closed_form_test_family(8);
        assert_eq!(
            family
                .auto_subsample_phase_counter
                .load(std::sync::atomic::Ordering::SeqCst),
            0,
            "fresh family must start at Phase-1 step 0"
        );
        assert!(
            family
                .auto_subsample_last_rho
                .lock()
                .expect("auto_subsample_last_rho mutex poisoned")
                .is_none(),
            "fresh family must have no recorded last-rho proxy"
        );
    }
}
