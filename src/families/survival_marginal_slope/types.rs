use poly_arith::{
    poly_add, poly_add_jets, poly_coeff_mask, poly_mul, poly_mul_jets, poly_scale, poly_scale_jets,
    poly_sub,
};


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


/// Block tag used by the joint training-row preflight diagnostic.
/// Names a single block in the joint design layout
/// `[time | marginal | logslope | score_warp? | link_dev?]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JointPreflightBlock {
    Time,
    Marginal,
    Logslope,
    ScoreWarp,
    LinkDev,
}


impl std::fmt::Display for JointPreflightBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            JointPreflightBlock::Time => "time",
            JointPreflightBlock::Marginal => "marginal",
            JointPreflightBlock::Logslope => "logslope",
            JointPreflightBlock::ScoreWarp => "score_warp",
            JointPreflightBlock::LinkDev => "link_dev",
        };
        f.write_str(name)
    }
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
    /// Out-of-fold Stage-1 score-influence Jacobian `J = ∂z/∂θ₁` (n × p₁) for a
    /// CTN → marginal-slope chain (issue #461, §3 of
    /// `marginal_slope_orthogonal_design.md`). When `Some`, the score-warp build
    /// site installs the absorbed influence block
    /// `Z_infl = diag(s_f · β̂₀(x_i)) · J` instead of the free-spline score-warp:
    /// the realized x-dependent Stage-1 leakage directions in η-space are
    /// appended as a null-penalized absorbed block (gauge priority 80,
    /// orthogonalized against marginal ⊕ logslope), making the β estimating
    /// equation Neyman-orthogonal to `span(Z_infl)`. When `None` (raw `z` with
    /// no Stage-1 model), the free-warp `score_warp` path is used unchanged.
    /// Populated out-of-fold by `crossfit_score_calibration` in
    /// `solver/workflow.rs`; mirrors the BMS spec field of the same name.
    pub score_influence_jacobian: Option<Array2<f64>>,
    pub latent_z_policy: LatentZPolicy,
}


pub const DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD: f64 = 1e-6;

const SURVIVAL_INTERCEPT_ABS_RESIDUAL_TOL: f64 = 1e-12;

const SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL: f64 = 1e-8;

const SURVIVAL_INTERCEPT_LOG_TAIL_THRESHOLD: f64 = 1e-8;


#[inline]
fn survival_derivative_guard_tolerance(qd1: f64, derivative_guard: f64) -> f64 {
    // The monotonicity bound q'(t) >= derivative_guard is enforced by the inner
    // active-set solver against SCALED constraint rows (each scaled by
    // max(||row||, |guard-offset|, 1) >= 1) to ACTIVE_SET_PRIMAL_FEASIBILITY_TOL,
    // so a converged active constraint legitimately sits up to that tolerance on
    // the infeasible side of the exact bound. The likelihood-domain predicate must
    // admit the same band the solver can certify -- matching validate_time_qd1_feasible
    // -- otherwise boundary-feasible oversmoothed iterates are spuriously rejected and
    // every outer seed fails (#788). log(c*qd1) is finite for any qd1 > 0, so this
    // admits no numerically unsafe iterate; it only stops rejecting boundary-feasible
    // points. The raw 256*eps band remains as a floor.
    let magnitude = 1.0 + qd1.abs().max(derivative_guard.abs());
    let solver_band = 4.0 * crate::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL * magnitude;
    let eps_floor = 256.0 * f64::EPSILON * magnitude;
    solver_band.max(eps_floor)
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
    /// Width `p₁` of the absorbed Stage-1 influence block (#461) when the fit
    /// hosted a dedicated additive absorber (the trailing block). `None` when no
    /// CTN Stage-1 chain produced an influence Jacobian. The predictor drops the
    /// absorber's `γ`; this width lets it account for the extra trailing block
    /// and slice `γ` out of the joint covariance.
    pub influence_absorber_width: Option<usize>,
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
    /// large scale (B-spline/I-spline rows have only degree+1 nonzeros).
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
    /// Absorbed Stage-1 influence columns `Z̃_infl` at the training rows
    /// (`n × p₁`), residualized against the marginal location span in the
    /// rigid-pilot row metric (#461, design §3). When `Some`, the family hosts a
    /// dedicated additive absorber block whose coefficient `γ` shifts the
    /// de-nested observed index `η₁` by `+Z̃_infl[row,:]·γ` (sibling of the
    /// per-row calibration intercept — un-`c(g)`-scaled, unlike the marginal
    /// block which enters the time-quantile location through `q·c(g)`). The
    /// block carries a fixed small ridge and is dropped at predict. `None` ⇒ raw
    /// `z` with no CTN Stage-1; the free-warp `score_warp` is the fallback basis.
    influence_absorber: Option<Array2<f64>>,
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
///
/// Two slots per row (entry intercept against `q0`, exit intercept against
/// `q1`). Each slot stores the converged intercept `a` alongside a
/// `beta_tag: u64` — a 64-bit hash of the joint coefficient vector at the
/// time of write. Reads return `Some(a)` only when the caller's tag matches
/// the stored tag AND the stored value is finite. This makes the cache
/// transactional with respect to trust-region trials and subsampled probes:
/// a rejected trial at β_A and an accepted full-data eval at β_B key under
/// distinct tags, so writes from one cannot poison reads from the other.
///
/// The "never written" sentinel is `beta_tag == 0`. Callers compute their
/// tag with `hash_intercept_warm_start_key` and remap `0` to `1` so that the
/// sentinel can never collide with a real key. Two consecutive evaluations
/// at the same β share the same tag and reuse the cached root.
///
/// Memory ordering: the writer stores `value` with `Relaxed` and then `tag`
/// with `Release`. The reader loads `tag` with `Acquire`, reads `value`
/// with `Relaxed`, and re-checks `tag` with `Acquire`. The double-check
/// detects a torn read where another thread interleaved a tag bump between
/// the value read and the second tag load.
struct SurvivalInterceptWarmStartCache {
    entry_value: Vec<std::sync::atomic::AtomicU64>,
    entry_tag: Vec<std::sync::atomic::AtomicU64>,
    exit_value: Vec<std::sync::atomic::AtomicU64>,
    exit_tag: Vec<std::sync::atomic::AtomicU64>,
}


impl SurvivalInterceptWarmStartCache {
    #[inline]
    fn slots_for(
        &self,
        kind: SurvivalInterceptSlotKind,
    ) -> (
        &[std::sync::atomic::AtomicU64],
        &[std::sync::atomic::AtomicU64],
    ) {
        match kind {
            SurvivalInterceptSlotKind::Entry => (&self.entry_value, &self.entry_tag),
            SurvivalInterceptSlotKind::Exit => (&self.exit_value, &self.exit_tag),
        }
    }

    /// Return the cached intercept iff the slot's stored `beta_tag` matches
    /// the caller's `beta_tag` and the stored value is finite. Otherwise
    /// returns `None` (cache miss — caller falls back to closed-form seed).
    #[inline]
    fn load(&self, row: usize, kind: SurvivalInterceptSlotKind, beta_tag: u64) -> Option<f64> {
        let (values, tags) = self.slots_for(kind);
        let value_slot = values.get(row)?;
        let tag_slot = tags.get(row)?;
        let tag_before = tag_slot.load(std::sync::atomic::Ordering::Acquire);
        if tag_before != beta_tag {
            return None;
        }
        let bits = value_slot.load(std::sync::atomic::Ordering::Relaxed);
        let tag_after = tag_slot.load(std::sync::atomic::Ordering::Acquire);
        if tag_after != beta_tag {
            return None;
        }
        let value = f64::from_bits(bits);
        value.is_finite().then_some(value)
    }

    /// Stamp the slot with the converged intercept under `beta_tag`. Concurrent
    /// writers from different trials race; the last writer wins, which is fine
    /// because every reader gates on its own tag and only accepts a match.
    #[inline]
    fn store(&self, row: usize, kind: SurvivalInterceptSlotKind, a: f64, beta_tag: u64) {
        let (values, tags) = self.slots_for(kind);
        if let (Some(value_slot), Some(tag_slot)) = (values.get(row), tags.get(row)) {
            // Invalidate before writing the new value so an interleaved
            // reader cannot see the new tag paired with the old value.
            tag_slot.store(0, std::sync::atomic::Ordering::Release);
            value_slot.store(a.to_bits(), std::sync::atomic::Ordering::Relaxed);
            tag_slot.store(beta_tag, std::sync::atomic::Ordering::Release);
        }
    }
}


fn new_intercept_warm_start_cache(n: usize) -> Arc<SurvivalInterceptWarmStartCache> {
    Arc::new(SurvivalInterceptWarmStartCache {
        entry_value: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
        entry_tag: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect(),
        exit_value: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(f64::NAN.to_bits()))
            .collect(),
        exit_tag: (0..n)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect(),
    })
}


/// FNV-1a 64-bit hash of the joint coefficient slices `(beta_h, beta_w)`.
/// Returned tag is guaranteed non-zero (zero is remapped to one) so that
/// the cache's "never written" sentinel cannot collide with a real key.
/// At 64 bits, false collisions across distinct β are astronomically rare;
/// on a miss we just re-solve from the closed-form seed.
#[inline]
fn hash_intercept_warm_start_key(
    beta_h: Option<&Array1<f64>>,
    beta_w: Option<&Array1<f64>>,
) -> u64 {
    let mut hash = Fnv1a::new();
    hash.mix_opt_beta(0xa1, beta_h);
    hash.mix_opt_beta(0xa2, beta_w);
    hash.finish_nonzero()
}


#[derive(Clone, Default)]
struct ThetaHints {
    time_beta: Option<Array1<f64>>,
    marginal_beta: Option<Array1<f64>>,
    logslope_beta: Option<Array1<f64>>,
    score_warp_beta: Option<Array1<f64>>,
    link_dev_beta: Option<Array1<f64>>,
    influence_beta: Option<Array1<f64>>,
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


fn score_warp_component_beta(
    runtime: &DeviationRuntime,
    beta: &Array1<f64>,
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


/// Stripe a (post-reparam) scalar score-warp `base` across K z coordinates
/// to produce the direct-sum block `Φ_total = [Φ(z_1) | Φ(z_2) | ... | Φ(z_K)]`.
///
/// Caller is responsible for any cross-block reparameterisation on
/// `base.runtime` BEFORE calling this — the install path has already
/// updated `base.runtime.basis_dim()` to the kept dimension `p_kept`,
/// and `base.block.design / penalties / nullspace_dims` reflect that
/// reparam. Each per-z stripe then evaluates `runtime.design_at_training_with_residual`
/// at `z[:, k]` so the cached parametric anchor rows are folded into every
/// stripe, giving a striped design that is jointly orthogonal (in the W-
/// metric used during reparam) to span(anchors) at training rows.
fn stripe_score_warp_across_z_coords(
    base: ParameterBlockInput,
    base_runtime: DeviationRuntime,
    z: &Array2<f64>,
) -> Result<PerZScoreWarpPrepared, String> {
    let score_dim = z.ncols();
    if score_dim == 0 {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival score-warp requires at least one z coordinate".to_string(),
        }
        .into());
    }
    if score_dim == 1 {
        return Ok(PerZScoreWarpPrepared {
            block: base,
            runtime: base_runtime,
            score_dim,
        });
    }

    // Vector-z score warp is the direct sum of K scalar warp spaces:
    //
    //     h_i = sum_{k=1}^K W_k(z_{ik}) beta_k .
    //
    // The coefficient vector is laid out as [beta_1 | ... | beta_K],
    // and the row design is the horizontal concatenation of the K scalar
    // designs. Per-z stripes go through `design_at_training_with_residual`
    // so that any installed anchor residual (from cross-block reparam) is
    // subtracted uniformly across all K stripes — each stripe lives in
    // the SAME orthogonal complement of the parametric anchor span as the
    // primary z coordinate's basis. Penalties are block-local on each
    // coordinate slice, giving each W_k its own smoothing parameter unless
    // a later grouping layer intentionally ties precision labels.
    let p = base_runtime.basis_dim();
    let n = z.nrows();
    let mut design = Array2::<f64>::zeros((n, p * score_dim));
    for coord in 0..score_dim {
        let z_coord = z.column(coord).to_owned();
        let coord_design = base_runtime.design_at_training_with_residual(&z_coord)?;
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

    let mut block = base.clone();
    block.design = DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(design));
    block.offset = Array1::zeros(n);
    block.initial_beta = Some(Array1::zeros(p * score_dim));
    block.initial_log_lambdas = None;
    let base_penalties = base.penalties.clone();
    let base_nullspaces = base.nullspace_dims.clone();
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
        runtime: base_runtime,
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
    // Survival marginal-slope gauge ownership: score-warp deviations
    // are pure shape modifications around the latent score axis and
    // should never own a shared affine direction with time, marginal,
    // or logslope blocks. Set a low priority so the canonical-gauge
    // selector drops shared directions from score_warp_dev before any
    // parametric block loses a column.
    spec.gauge_priority = 80;
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
    /// Absorbed Stage-1 influence block (#461), trailing the flex blocks. Its
    /// width is `p₁` (the Stage-1 coefficient count), `None` when no CTN Stage-1
    /// chain produced an influence Jacobian.
    influence: Option<std::ops::Range<usize>>,
    total: usize,
}


/// Identifies one coefficient block of the survival marginal-slope joint
/// Hessian. The discriminant order *is* the coordinate layout order
/// (`time | marginal | logslope | score_warp? | link_dev?`), so it doubles as
/// the upper-triangle ordering used to pick the stored half of each symmetric
/// off-diagonal block. `ScoreWarp` and `LinkDev` are optional: they are absent
/// from the layout unless the corresponding flex deviation block is active.
///
/// This enum and [`BlockHessianAccumulator::block_view`] are the single source
/// of truth for the block layout. Every Hessian read-out (dense scatter,
/// matvec, bilinear form, diagonal extraction) is driven by them rather than
/// re-listing the fifteen blocks and their transpose relationships by hand, so
/// a layout change lands in exactly one place.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum HessBlock {
    Time,
    Marginal,
    Logslope,
    ScoreWarp,
    LinkDev,
    /// Absorbed Stage-1 influence block (#461). Placed LAST in the canonical
    /// layout so the existing `score_warp` (index 3) / `link_dev` (index 4)
    /// block-state positions are undisturbed; the absorber's coordinate range
    /// trails them and its β is dropped at predict.
    Influence,
}


impl HessBlock {
    /// Blocks in canonical coordinate-layout order. Iterating this array is how
    /// every assembler visits blocks; the order fixes floating-point
    /// accumulation order in the matvec / bilinear paths.
    const ALL: [HessBlock; 6] = [
        HessBlock::Time,
        HessBlock::Marginal,
        HessBlock::Logslope,
        HessBlock::ScoreWarp,
        HessBlock::LinkDev,
        HessBlock::Influence,
    ];
}


impl BlockSlices {
    /// Coordinate range occupied by `block`, or `None` when the (optional)
    /// flex block is inactive. `time`/`marginal`/`logslope` are always present.
    #[inline]
    fn range_of(&self, block: HessBlock) -> Option<std::ops::Range<usize>> {
        match block {
            HessBlock::Time => Some(self.time.clone()),
            HessBlock::Marginal => Some(self.marginal.clone()),
            HessBlock::Logslope => Some(self.logslope.clone()),
            HessBlock::ScoreWarp => self.score_warp.clone(),
            HessBlock::LinkDev => self.link_dev.clone(),
            HessBlock::Influence => self.influence.clone(),
        }
    }
}


fn block_slices(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
) -> BlockSlices {
    if !block_states.is_empty() {
        let expected_blocks = 3
            + usize::from(family.score_warp.is_some())
            + usize::from(family.link_dev.is_some())
            + usize::from(family.influence_absorber.is_some());
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
    // Absorbed influence block trails the flex blocks: its width is the
    // Stage-1 coefficient count `p₁` (= `Z̃_infl.ncols()`).
    let influence = family.influence_absorber.as_ref().map(|z_tilde| {
        let range = cursor..cursor + z_tilde.ncols();
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
        influence,
        total,
    }
}


/// Owned scratch buffers backing a
/// [`crate::families::survival_marginal_slope_gpu::SurvivalFlexGpuRowInputs`] descriptor.
///
/// Built per-call by
/// [`SurvivalMarginalSlopeFamily::build_survival_flex_gpu_row_batch`];
/// callers hold the batch by value across the GPU `try_*` entry so the
/// borrowed slices returned by [`Self::as_inputs`] live for the dispatch.
struct SurvivalFlexGpuRowBatch {
    n: usize,
    p: usize,
    q0: Vec<f64>,
    q1: Vec<f64>,
    qd1: Vec<f64>,
    z: Vec<f64>,
    g: Vec<f64>,
    beta: Vec<f64>,
    weights: Vec<f64>,
    event: Vec<f64>,
}


impl SurvivalFlexGpuRowBatch {
    /// Borrow the buffers as a
    /// [`crate::families::survival_marginal_slope_gpu::SurvivalFlexGpuRowInputs`] descriptor.
    fn as_inputs<'a>(
        &'a self,
        family: &SurvivalMarginalSlopeFamily,
    ) -> crate::families::survival_marginal_slope_gpu::SurvivalFlexGpuRowInputs<'a> {
        crate::families::survival_marginal_slope_gpu::SurvivalFlexGpuRowInputs {
            n: self.n,
            r: N_PRIMARY,
            p: self.p,
            score_dim: family.score_dim(),
            beta: &self.beta,
            q0: &self.q0,
            q1: &self.q1,
            qd1: &self.qd1,
            z: &self.z,
            g: &self.g,
            weights: &self.weights,
            event: &self.event,
            derivative_guard: family.derivative_guard,
            probit_scale: family.probit_frailty_scale(),
        }
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
    /// Single trailing primary index for the absorbed Stage-1 influence offset
    /// `o_infl` (#461). Unlike `g`/`h`/`w`, `o_infl` does NOT enter the de-nested
    /// calibration cells — it is a pure additive shift of the OBSERVED index η₁,
    /// so its only non-zero primary partial is `∂η₁/∂o_infl = 1` injected at the
    /// observed-timepoint reconstruction (cell-coefficient partials stay zero).
    infl: Option<usize>,
    total: usize,
}


/// Pack a private `SurvivalFlexTimepointExact` into the Block 10
/// pub-substrate input type so the shared CPU/GPU pure assembler in
/// `crate::families::survival_marginal_slope_gpu` can consume it without taking a
/// dependency on the family's private jet structs.
pub(crate) fn block10_pack_base(
    base: &SurvivalFlexTimepointExact,
) -> crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10TimepointBase {
    crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10TimepointBase {
        eta: base.eta,
        chi: base.chi,
        d: base.d,
        eta_u: base.eta_u.to_vec(),
        eta_uv: base.eta_uv.iter().copied().collect(),
        chi_u: base.chi_u.to_vec(),
        chi_uv: base.chi_uv.iter().copied().collect(),
        d_u: base.d_u.to_vec(),
        d_uv: base.d_uv.iter().copied().collect(),
    }
}


pub(crate) fn block10_pack_dir(
    ext: &SurvivalFlexTimepointDirectionalExact,
) -> crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10TimepointDirectional {
    crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10TimepointDirectional {
        eta_uv_dir: ext.eta_uv_dir.iter().copied().collect(),
        chi_uv_dir: ext.chi_uv_dir.iter().copied().collect(),
        d_u_dir: ext.d_u_dir.to_vec(),
        d_uv_dir: ext.d_uv_dir.iter().copied().collect(),
    }
}


pub(crate) fn block10_pack_bi(
    bi: &SurvivalFlexTimepointBiDirectionalExact,
) -> crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10TimepointBiDirectional {
    crate::families::survival_marginal_slope_gpu::SurvivalFlexBlock10TimepointBiDirectional {
        eta_uv_uv: bi.eta_uv_uv.iter().copied().collect(),
        chi_uv_uv: bi.chi_uv_uv.iter().copied().collect(),
        d_uv_uv: bi.d_uv_uv.iter().copied().collect(),
    }
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
    // The absorber contributes a single primary scalar `o_infl` (trailing all
    // flex bases). Its full coefficient block lives in the `Influence`
    // ParameterBlockSpec; here it is one primary channel whose row-design is
    // `Z̃_infl[row,:]`, projected by `add_pullback`.
    let infl = family.influence_absorber.as_ref().map(|_| {
        let idx = cursor;
        cursor += 1;
        idx
    });
    FlexPrimarySlices {
        q0,
        q1,
        qd1,
        g,
        h,
        w,
        infl,
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
    /// Absorbed Stage-1 influence block (#461): the trailing block-diagonal
    /// grad/Hess over the `p₁` absorber coefficients `γ`, projected from the
    /// single `o_infl` primary scalar through the `Z̃_infl` design row.
    grad_influence: Option<Array1<f64>>,
    hess_influence: Option<Array2<f64>>,
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
            grad_influence: slices
                .influence
                .as_ref()
                .map(|range| Array1::zeros(range.len())),
            hess_influence: slices
                .influence
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
        add_optional_vector(&mut self.grad_influence, &other.grad_influence);
        add_optional_matrix(&mut self.hess_score_warp, &other.hess_score_warp);
        add_optional_matrix(&mut self.hess_link_dev, &other.hess_link_dev);
        add_optional_matrix(&mut self.hess_influence, &other.hess_influence);
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
        if let (Some(gradient), Some(hessian)) = (self.grad_influence, self.hess_influence) {
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


impl DenestedCellPrimaryFixedPartials {
    /// Reconstruct the struct from the device-flat layout emitted by
    /// `crate::families::survival_marginal_slope_gpu_prep::DENESTED_CELL_PRIMARY_FIXED_PARTIALS_KERNEL_SRC`.
    ///
    /// Layout (per cell):
    ///
    /// ```text
    ///   dc_da[4], dc_daa[4], dc_daaa[4]                       // 12 doubles
    ///   coeff_u[r][4]                                          // 4r
    ///   coeff_au[r][4], coeff_bu[r][4]                         // 8r
    ///   coeff_aau[r][4], coeff_abu[r][4], coeff_bbu[r][4]      // 12r
    ///   coeff_aaau[r][4], coeff_aabu[r][4], coeff_abbu[r][4], coeff_bbbu[r][4]
    ///                                                          // 16r
    /// ```
    ///
    /// Total length: `12 + 40 * r`.
    fn from_flat_slice(flat: &[f64], r: usize) -> Result<Self, String> {
        let expected = 12 + 40 * r;
        if flat.len() != expected {
            return Err(format!(
                "DenestedCellPrimaryFixedPartials::from_flat_slice: expected {expected} doubles \
                 (12 + 40·r with r={r}), got {}",
                flat.len()
            ));
        }
        let read4 =
            |off: usize| -> [f64; 4] { [flat[off], flat[off + 1], flat[off + 2], flat[off + 3]] };
        let dc_da = read4(0);
        let dc_daa = read4(4);
        let dc_daaa = read4(8);
        let mut cursor = 12;
        let read_run = |start: usize| -> Vec<[f64; 4]> {
            let mut out = Vec::with_capacity(r);
            for slot in 0..r {
                let off = start + slot * 4;
                out.push([flat[off], flat[off + 1], flat[off + 2], flat[off + 3]]);
            }
            out
        };
        let coeff_u = read_run(cursor);
        cursor += 4 * r;
        let coeff_au = read_run(cursor);
        cursor += 4 * r;
        let coeff_bu = read_run(cursor);
        cursor += 4 * r;
        let coeff_aau = read_run(cursor);
        cursor += 4 * r;
        let coeff_abu = read_run(cursor);
        cursor += 4 * r;
        let coeff_bbu = read_run(cursor);
        cursor += 4 * r;
        let coeff_aaau = read_run(cursor);
        cursor += 4 * r;
        let coeff_aabu = read_run(cursor);
        cursor += 4 * r;
        let coeff_abbu = read_run(cursor);
        cursor += 4 * r;
        let coeff_bbbu = read_run(cursor);
        cursor += 4 * r;
        assert_eq!(cursor, expected);
        Ok(Self {
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


pub(crate) struct SurvivalFlexTimepointExact {
    pub(crate) eta: f64,
    pub(crate) chi: f64,
    pub(crate) d: f64,
    pub(crate) eta_u: Array1<f64>,
    pub(crate) eta_uv: Array2<f64>,
    pub(crate) chi_u: Array1<f64>,
    pub(crate) chi_uv: Array2<f64>,
    pub(crate) d_u: Array1<f64>,
    pub(crate) d_uv: Array2<f64>,
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
pub(crate) struct SurvivalFlexTimepointDirectionalExact {
    pub(crate) eta_uv_dir: Array2<f64>,
    pub(crate) chi_uv_dir: Array2<f64>,
    pub(crate) d_u_dir: Array1<f64>,
    pub(crate) d_uv_dir: Array2<f64>,
}


pub(crate) struct SurvivalFlexTimepointBiDirectionalExact {
    pub(crate) eta_uv_uv: Array2<f64>,
    pub(crate) chi_uv_uv: Array2<f64>,
    pub(crate) d_uv_uv: Array2<f64>,
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
    /// Absorbed-influence diagonal block (#461), `p_i × p_i` (`p_i = Z̃.ncols()`).
    h_ii: Array2<f64>,
    h_tm: Array2<f64>,
    h_tg: Array2<f64>,
    h_th: Array2<f64>,
    h_tw: Array2<f64>,
    /// time × influence cross-block.
    h_ti: Array2<f64>,
    h_mg: Array2<f64>,
    h_mh: Array2<f64>,
    h_mw: Array2<f64>,
    /// marginal × influence cross-block.
    h_mi: Array2<f64>,
    h_gh: Array2<f64>,
    h_gw: Array2<f64>,
    /// logslope × influence cross-block.
    h_gi: Array2<f64>,
    h_hw: Array2<f64>,
    /// score_warp × influence cross-block.
    h_hi: Array2<f64>,
    /// link_dev × influence cross-block.
    h_wi: Array2<f64>,
}


const PULLBACK_PARALLEL_MIN_CELLS: usize = 16_384;

const PULLBACK_PARALLEL_TARGET_CELLS: usize = 65_536;


impl BlockHessianAccumulator {
    fn new(p_t: usize, p_m: usize, p_g: usize, p_h: usize, p_w: usize, p_i: usize) -> Self {
        Self {
            h_tt: Array2::zeros((p_t, p_t)),
            h_mm: Array2::zeros((p_m, p_m)),
            h_gg: Array2::zeros((p_g, p_g)),
            h_hh: Array2::zeros((p_h, p_h)),
            h_ww: Array2::zeros((p_w, p_w)),
            h_ii: Array2::zeros((p_i, p_i)),
            h_tm: Array2::zeros((p_t, p_m)),
            h_tg: Array2::zeros((p_t, p_g)),
            h_th: Array2::zeros((p_t, p_h)),
            h_tw: Array2::zeros((p_t, p_w)),
            h_ti: Array2::zeros((p_t, p_i)),
            h_mg: Array2::zeros((p_m, p_g)),
            h_mh: Array2::zeros((p_m, p_h)),
            h_mw: Array2::zeros((p_m, p_w)),
            h_mi: Array2::zeros((p_m, p_i)),
            h_gh: Array2::zeros((p_g, p_h)),
            h_gw: Array2::zeros((p_g, p_w)),
            h_gi: Array2::zeros((p_g, p_i)),
            h_hw: Array2::zeros((p_h, p_w)),
            h_hi: Array2::zeros((p_h, p_i)),
            h_wi: Array2::zeros((p_w, p_i)),
        }
    }

    fn block_dims(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.h_tt.nrows(),
            self.h_mm.nrows(),
            self.h_gg.nrows(),
            self.h_hh.nrows(),
            self.h_ww.nrows(),
            self.h_ii.nrows(),
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
        let (p_t, _, _, _, _, _) = self.block_dims();
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

        // Absorbed-influence block (#461). The absorber contributes a single
        // primary scalar `o_infl` at index `primary.infl`, whose design row is
        // `Z̃_infl[row,:]` (the residualized leakage columns). It projects exactly
        // like the single-scalar logslope `g` index (primary 3 → logslope_design)
        // but through `Z̃` and crossed with every block, plus the diagonal
        // `h_ii = primary_hessian[[infl, infl]]·Z̃Z̃ᵀ`. `o_infl` is an additive η₁
        // shift (∂η₁/∂o_infl = 1), so the per-block primary weights mirror the
        // existing channels: time uses {q0,q1,qd1}, marginal {q0,q1}, logslope
        // {g}, score_warp/link_dev their own primary ranges.
        if let Some(infl_idx) = primary.infl {
            let z_tilde = family.influence_absorber.as_ref().ok_or_else(|| {
                "add_pullback: influence primary index present but no Z̃ design".to_string()
            })?;
            let z_row = z_tilde.row(row);
            let p_i = z_row.len();

            // Influence × influence diagonal: primary_hessian[[infl,infl]]·Z̃Z̃ᵀ.
            let ii_weight = primary_hessian[[infl_idx, infl_idx]];
            if ii_weight != 0.0 {
                let z_col = z_row.view().insert_axis(Axis(1));
                ndarray::linalg::general_mat_mul(
                    ii_weight,
                    &z_col,
                    &z_row.view().insert_axis(Axis(0)),
                    1.0,
                    &mut self.h_ii,
                );
            }

            // Time × influence: each time sub-design (entry/exit/deriv) crossed
            // with Z̃ at the matching primary weight.
            let ti_weights = [
                primary_hessian[[0, infl_idx]],
                primary_hessian[[1, infl_idx]],
                primary_hessian[[2, infl_idx]],
            ];
            for (des, alpha) in time_designs.iter().zip(ti_weights.iter()) {
                if *alpha == 0.0 {
                    continue;
                }
                let t_chunk = des
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("add_pullback time design try_row_chunk: {e}"))?;
                let t_row = t_chunk.row(0);
                for (t_coeff, &t_val) in t_row.iter().enumerate() {
                    for i_coeff in 0..p_i {
                        self.h_ti[[t_coeff, i_coeff]] += *alpha * t_val * z_row[i_coeff];
                    }
                }
            }

            // Marginal × influence (marginal uses q0 + q1).
            let mi_weight = primary_hessian[[0, infl_idx]] + primary_hessian[[1, infl_idx]];
            if mi_weight != 0.0 {
                let m_chunk = family
                    .marginal_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("add_pullback marginal_design try_row_chunk: {e}"))?;
                let m_row = m_chunk.row(0);
                for (m_coeff, &m_val) in m_row.iter().enumerate() {
                    for i_coeff in 0..p_i {
                        self.h_mi[[m_coeff, i_coeff]] += mi_weight * m_val * z_row[i_coeff];
                    }
                }
            }

            // Logslope × influence (logslope uses g, primary index 3).
            let gi_weight = primary_hessian[[3, infl_idx]];
            if gi_weight != 0.0 {
                let g_chunk = family
                    .logslope_design
                    .try_row_chunk(row..row + 1)
                    .map_err(|e| format!("add_pullback logslope_design try_row_chunk: {e}"))?;
                let g_row = g_chunk.row(0);
                for (g_coeff, &g_val) in g_row.iter().enumerate() {
                    for i_coeff in 0..p_i {
                        self.h_gi[[g_coeff, i_coeff]] += gi_weight * g_val * z_row[i_coeff];
                    }
                }
            }

            // Score-warp × influence (score_warp basis is itself in the primary
            // vector, so its row design is the identity on `h_range`).
            if let Some(h_range) = primary.h.as_ref() {
                for h_local in 0..h_range.len() {
                    let weight = primary_hessian[[h_range.start + h_local, infl_idx]];
                    if weight == 0.0 {
                        continue;
                    }
                    for i_coeff in 0..p_i {
                        self.h_hi[[h_local, i_coeff]] += weight * z_row[i_coeff];
                    }
                }
            }

            // Link-dev × influence (link_dev basis is in the primary vector too).
            if let Some(w_range) = primary.w.as_ref() {
                for w_local in 0..w_range.len() {
                    let weight = primary_hessian[[w_range.start + w_local, infl_idx]];
                    if weight == 0.0 {
                        continue;
                    }
                    for i_coeff in 0..p_i {
                        self.h_wi[[w_local, i_coeff]] += weight * z_row[i_coeff];
                    }
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

    /// Returns the `(row, col)` block of the symmetric joint Hessian as a view
    /// into the stored half, transposing on the fly when the requested pair is
    /// below the diagonal.
    ///
    /// This is the single source of truth for the block layout: it is the only
    /// place that maps a block pair onto one of the fifteen stored matrices and
    /// decides whether a transpose is needed. The match is exhaustive over all
    /// 5×5 pairs, so adding a block produces a hard compile error here rather
    /// than a silent gap in one assembler. Diagonal blocks are symmetric by
    /// construction and returned untransposed.
    #[inline]
    fn block_view(&self, row: HessBlock, col: HessBlock) -> ArrayView2<'_, f64> {
        use HessBlock::*;
        match (row, col) {
            (Time, Time) => self.h_tt.view(),
            (Marginal, Marginal) => self.h_mm.view(),
            (Logslope, Logslope) => self.h_gg.view(),
            (ScoreWarp, ScoreWarp) => self.h_hh.view(),
            (LinkDev, LinkDev) => self.h_ww.view(),

            (Time, Marginal) => self.h_tm.view(),
            (Marginal, Time) => self.h_tm.t(),
            (Time, Logslope) => self.h_tg.view(),
            (Logslope, Time) => self.h_tg.t(),
            (Time, ScoreWarp) => self.h_th.view(),
            (ScoreWarp, Time) => self.h_th.t(),
            (Time, LinkDev) => self.h_tw.view(),
            (LinkDev, Time) => self.h_tw.t(),

            (Marginal, Logslope) => self.h_mg.view(),
            (Logslope, Marginal) => self.h_mg.t(),
            (Marginal, ScoreWarp) => self.h_mh.view(),
            (ScoreWarp, Marginal) => self.h_mh.t(),
            (Marginal, LinkDev) => self.h_mw.view(),
            (LinkDev, Marginal) => self.h_mw.t(),

            (Logslope, ScoreWarp) => self.h_gh.view(),
            (ScoreWarp, Logslope) => self.h_gh.t(),
            (Logslope, LinkDev) => self.h_gw.view(),
            (LinkDev, Logslope) => self.h_gw.t(),

            (ScoreWarp, LinkDev) => self.h_hw.view(),
            (LinkDev, ScoreWarp) => self.h_hw.t(),

            (Influence, Influence) => self.h_ii.view(),
            (Time, Influence) => self.h_ti.view(),
            (Influence, Time) => self.h_ti.t(),
            (Marginal, Influence) => self.h_mi.view(),
            (Influence, Marginal) => self.h_mi.t(),
            (Logslope, Influence) => self.h_gi.view(),
            (Influence, Logslope) => self.h_gi.t(),
            (ScoreWarp, Influence) => self.h_hi.view(),
            (Influence, ScoreWarp) => self.h_hi.t(),
            (LinkDev, Influence) => self.h_wi.view(),
            (Influence, LinkDev) => self.h_wi.t(),
        }
    }

    /// Visits the present off-diagonal block pairs `(lo, hi)` in column-major
    /// upper-triangle order (`hi` outermost), skipping pairs whose blocks are
    /// inactive. This fixes the off-diagonal traversal order shared by
    /// [`Self::to_dense`] and [`BlockHessianOperator::bilinear`].
    #[inline]
    fn for_each_offdiagonal_pair(
        slices: &BlockSlices,
        mut visit: impl FnMut(HessBlock, std::ops::Range<usize>, HessBlock, std::ops::Range<usize>),
    ) {
        for hi_idx in 1..HessBlock::ALL.len() {
            let hi = HessBlock::ALL[hi_idx];
            let Some(r_hi) = slices.range_of(hi) else {
                continue;
            };
            for &lo in &HessBlock::ALL[..hi_idx] {
                let Some(r_lo) = slices.range_of(lo) else {
                    continue;
                };
                visit(lo, r_lo, hi, r_hi.clone());
            }
        }
    }

    /// Assemble into a dense p×p matrix.
    fn to_dense(&self, slices: &BlockSlices) -> Array2<f64> {
        let mut out = Array2::zeros((slices.total, slices.total));
        for block in HessBlock::ALL {
            let Some(range) = slices.range_of(block) else {
                continue;
            };
            out.slice_mut(s![range.clone(), range])
                .assign(&self.block_view(block, block));
        }
        Self::for_each_offdiagonal_pair(slices, |lo, r_lo, hi, r_hi| {
            out.slice_mut(s![r_lo.clone(), r_hi.clone()])
                .assign(&self.block_view(lo, hi));
            out.slice_mut(s![r_hi, r_lo])
                .assign(&self.block_view(hi, lo));
        });
        out
    }

    fn into_operator(self, slices: BlockSlices) -> BlockHessianOperator {
        BlockHessianOperator {
            blocks: self,
            slices,
        }
    }

    fn add(&mut self, other: &BlockHessianAccumulator) {
        self.h_tt += &other.h_tt;
        self.h_mm += &other.h_mm;
        self.h_gg += &other.h_gg;
        self.h_hh += &other.h_hh;
        self.h_ww += &other.h_ww;
        self.h_ii += &other.h_ii;
        self.h_tm += &other.h_tm;
        self.h_tg += &other.h_tg;
        self.h_th += &other.h_th;
        self.h_tw += &other.h_tw;
        self.h_ti += &other.h_ti;
        self.h_mg += &other.h_mg;
        self.h_mh += &other.h_mh;
        self.h_mw += &other.h_mw;
        self.h_mi += &other.h_mi;
        self.h_gh += &other.h_gh;
        self.h_gw += &other.h_gw;
        self.h_gi += &other.h_gi;
        self.h_hw += &other.h_hw;
        self.h_hi += &other.h_hi;
        self.h_wi += &other.h_wi;
    }

    fn diagonal(&self, slices: &BlockSlices) -> Array1<f64> {
        let mut out = Array1::zeros(slices.total);
        for block in HessBlock::ALL {
            let Some(range) = slices.range_of(block) else {
                continue;
            };
            out.slice_mut(s![range])
                .assign(&self.block_view(block, block).diag());
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
        // large-scale N; nesting inner par_iter here adds work-stealing
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
/// `HyperOperator` view over a populated [`BlockHessianAccumulator`].
///
/// The operator owns the *same* block storage as the dense path — there is
/// no second copy of the 15 cross-block matrices and no second block-layout
/// implementation. The matvec/bilinear here and `to_dense` (which simply
/// scatters via [`BlockHessianAccumulator::to_dense`]) are guaranteed to be
/// consistent because they read one accumulator through one `BlockSlices`.
struct BlockHessianOperator {
    blocks: BlockHessianAccumulator,
    slices: BlockSlices,
}


impl HyperOperator for BlockHessianOperator {
    fn dim(&self) -> usize {
        self.slices.total
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(self.slices.total);
        self.mul_vec_into(v.view(), out.view_mut());
        out
    }

    fn mul_vec_view(&self, v: ArrayView1<'_, f64>) -> Array1<f64> {
        let mut out = Array1::zeros(self.slices.total);
        self.mul_vec_into(v, out.view_mut());
        out
    }

    fn mul_vec_into(&self, v: ArrayView1<'_, f64>, mut out: ArrayViewMut1<'_, f64>) {
        let b = &self.blocks;
        let slices = &self.slices;
        out.fill(0.0);
        // Full symmetric block matvec, organised row-block by row-block. The
        // block layout and transpose handling live entirely in `block_view`;
        // the `HessBlock::ALL` order fixes the per-output accumulation order so
        // the result is bit-identical to the previous hand-written scatter.
        for row in HessBlock::ALL {
            let Some(r_row) = slices.range_of(row) else {
                continue;
            };
            let mut o_row = out.slice_mut(s![r_row]);
            for col in HessBlock::ALL {
                let Some(r_col) = slices.range_of(col) else {
                    continue;
                };
                o_row += &b.block_view(row, col).dot(&v.slice(s![r_col]));
            }
        }
    }

    fn bilinear(&self, v: &Array1<f64>, u: &Array1<f64>) -> f64 {
        let b = &self.blocks;
        let slices = &self.slices;
        // vᵀ H u over the same single-source block layout as `mul_vec_into`.
        // Diagonal blocks first, then each off-diagonal pair contributes both
        // vₗₒ·Hₗₒ,ₕᵢ·uₕᵢ and vₕᵢ·Hₕᵢ,ₗₒ·uₗₒ — preserving the previous summation
        // order so the scalar is bit-identical.
        let mut total = 0.0;
        for block in HessBlock::ALL {
            let Some(range) = slices.range_of(block) else {
                continue;
            };
            let v_i = v.slice(s![range.clone()]);
            let u_i = u.slice(s![range]);
            total += v_i.dot(&b.block_view(block, block).dot(&u_i));
        }
        BlockHessianAccumulator::for_each_offdiagonal_pair(slices, |lo, r_lo, hi, r_hi| {
            let v_lo = v.slice(s![r_lo.clone()]);
            let u_lo = u.slice(s![r_lo]);
            let v_hi = v.slice(s![r_hi.clone()]);
            let u_hi = u.slice(s![r_hi]);
            total += v_lo.dot(&b.block_view(lo, hi).dot(&u_hi));
            total += v_hi.dot(&b.block_view(hi, lo).dot(&u_lo));
        });
        total
    }

    fn to_dense(&self) -> Array2<f64> {
        // Single block-layout source of truth: the operator densifies through
        // exactly the same scatter the dense path uses, so the dense and
        // operator Hessians are bit-identical by construction.
        self.blocks.to_dense(&self.slices)
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


/// Survival row Hessian row metric at a β-independent pilot η, used to
/// build the W inner product for cross-block orthogonalisation of the
/// score-warp and link-deviation flex bases. The previous implementation
/// copied the Bernoulli probit IRLS row weight
/// `w · φ(η)² / (Φ(η)·(1 − Φ(η)))` from BMS verbatim, but that is the row
/// curvature for a Bernoulli probit likelihood, not for the survival
/// marginal-slope likelihood. The survival row neg-log Hessian wrt η₁
/// (the dominant linear-predictor channel — both event and censored rows
/// contribute through it) at fixed β is
///
///   u2_eta1[i] = (1 − d_i) · w_i · d²/dη²[−log Φ(−η_i)]
///              + d_i · w_i
///
/// where the first term is the censored-row Mills-ratio second derivative
/// computed via `signed_probit_neglog_derivatives_up_to_fourth(-η, w·(1−d))`
/// (matching `row_primary_closed_form` at ~3483 column `u2_eta1`) and the
/// second term is the event-row contribution from `−log φ(η) = η²/2 + …`,
/// whose η-Hessian is `w·d`. This is exactly the curvature the joint
/// penalised Hessian sees along the dominant linear-predictor direction at
/// the pilot, so `Aᵀ W C̃ = 0` after the cross-block reparam is preserved in
/// the inner product PIRLS uses (modulo β-dependent drift between pilot and
/// running η — second-order in the off-anchor direction and bounded by the
/// Mills-ratio curvature scaling shared between both branches).
///
/// The pilot η is β-independent so this remains a one-shot construction-time
/// step. See the long comment block in bernoulli_marginal_slope.rs:19180 for
/// the BMS analogue and the failure mode the metric prevents (REML can
/// otherwise collapse the alias eigenvalue when `Aᵀ W_pirls C̃ ≠ 0`).
///
/// # Residual approximation
///
/// This is the η₁-channel row curvature. The full survival row Hessian is
/// 4×4 in primary scalars `(q₀, q₁, qd₁, g)` and chains differently into
/// each block (time/marginal share `dη₁/dq = c`; logslope chains through
/// `dη₁/dg = q₁·c₁ + s_f·z`; flex bases chain through `dη₁/dη = 1`). The
/// cross-block orthogonalisation here is therefore exact in the η₁
/// direction and approximate along the η₀ and ad₁ directions and between
/// blocks whose chain factors differ. In practice η₁ dominates because
/// both event and censored rows contribute through it while only entry
/// contributes to η₀; the alias structure the audit reported on the
/// large-scale fit is along the η₁ channel (time ↔ marginal ↔ logslope
/// ↔ score_warp ↔ link_dev all share constant + low-order columns that
/// project onto η₁). A fully chain-corrected metric is exactly what
/// `families::identifiability_compiler` (Phase 3, family-agnostic
/// `RowJacobianOperator` / `RowHessian` driver)
/// provides as the canonical home; this SMGS pre-PIRLS pilot reparam is
/// the principled scope for *alias killing only*. Two blocks with the
/// same chain factor (time ↔ marginal here, both `dη₁/dq = c`) produce
/// identical η contributions iff their bare-design columns are linearly
/// dependent — killed by bare-design orthogonality regardless of chain.
/// Different-chain aliases (marginal ↔ logslope) require alias structure
/// that exactly matches the chain ratio `(q₁·c₁ + s_f·z)/c`, a
/// degenerate case not observed. The large-scale alias chain the audit
/// reported (`time ↔ marginal ↔ logslope ↔ score_warp ↔ link_dev`) is
/// driven by shared constant + low-order columns — chain-independent and
/// killed by this scalar W. The reparam is one-shot and pre-PIRLS by
/// necessity: PIRLS itself cannot proceed on the aliased design.
fn survival_pilot_irls_row_metric_at_eta(
    eta_pilot: &Array1<f64>,
    sample_weights: &Array1<f64>,
    event: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let n = eta_pilot.len();
    if sample_weights.len() != n || event.len() != n {
        return Err(format!(
            "survival cross-block W metric: length mismatch eta={}, weights={}, event={}",
            n,
            sample_weights.len(),
            event.len(),
        ));
    }
    let mut w = Array1::<f64>::zeros(n);
    for i in 0..n {
        let eta = eta_pilot[i];
        let d = event[i];
        let weight = sample_weights[i];
        let (_, k2, _, _) =
            signed_probit_neglog_derivatives_up_to_fourth(-eta, weight * (1.0 - d))?;
        let phi_part = weight * d;
        w[i] = k2 + phi_part;
    }
    Ok(w)
}


/// Build the SMGS rigid pooled-probit pilot η at training rows from the
/// time-block offsets, marginal/logslope offsets, baseline slope and z. This
/// is the survival analog of BMS `rigid_pooled_probit_pilot_eta`; the basis
/// is intentionally β-independent so the resulting W metric depends only on
/// data + spec offsets and the cross-block orthogonalisation remains a one-
/// shot construction-time step. Used to weight the W-metric inner product
/// in `install_compiled_flex_block_into_runtime` (both flex paths).
fn survival_rigid_pilot_eta(
    n: usize,
    z_primary: &Array1<f64>,
    offset_exit: &Array1<f64>,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    baseline_slope: f64,
    probit_scale: f64,
) -> Array1<f64> {
    Array1::from_iter((0..n).map(|row| {
        let q_exit = offset_exit[row] + marginal_offset[row];
        let slope = baseline_slope + logslope_offset[row];
        rigid_observed_eta(q_exit, slope, z_primary[row], probit_scale)
    }))
}


/// One-step IRLS refinement of the rigid pilot η along the dominant η₁ row
/// channel. Starts from `survival_rigid_pilot_eta` (offset+baseline), runs
/// a single weighted Newton step over the joint location-anchor + logslope
/// design `X = [T_exit | M | G]`, and returns the refined per-row η₁ pilot
/// that the cross-block W metric uses. Prevents the flex-anchor bases
/// (`link_dev`, `score_warp`) from collapsing onto the same constant scalar
/// path that the offset-only seed would produce when training offsets are
/// uniform — the failure mode the original large-scale audit pinned to a
/// 13-dimensional unidentified quotient. The η₁ direction matches the doc
/// scope of `survival_pilot_irls_row_metric_at_eta` (see its long block);
/// the chain factor `dη₁/dq = c(g)` is absorbed into a per-row scaling of
/// the location anchor before the solve.
///
/// Returns `(eta1, beta_logslope)`: the per-row observed index `eta1` (used by
/// the cross-block W metric, unchanged from the legacy scalar return) AND the
/// one-step IRLS estimate of the logslope-surface coefficients `beta_logslope`
/// (the `G`-block portion of the joint Newton step). The latter is the #808
/// operating-point WARM START for the logslope block's `initial_beta`: on
/// clustered-PC designs the logslope block is EXACTLY W-null at the `g = 0`
/// seed (the slope-channel IRLS weight vanishes at the null slope), so the
/// inner joint-Newton cannot take its first step and freezes; seeding the
/// block at the pilot's `g ≈ 0.3` operating point (where the slope channel
/// carries information and the block is full-rank) breaks the chicken-and-egg
/// and lets the inner converge to the true data optimum — preserving the
/// log-slope estimand rather than dropping/pinning it. Self-correcting: it is
/// just a warm start, so the converged fit is the data optimum (zero bias).
fn survival_nonrigid_pilot_eta(
    n: usize,
    location_anchor_design: &DesignMatrix,
    logslope_design: &DesignMatrix,
    z_primary: &Array1<f64>,
    offset_exit: &Array1<f64>,
    marginal_offset: &Array1<f64>,
    logslope_offset: &Array1<f64>,
    baseline_slope: f64,
    sample_weights: &Array1<f64>,
    event: &Array1<f64>,
    probit_scale: f64,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    if location_anchor_design.nrows() != n
        || logslope_design.nrows() != n
        || z_primary.len() != n
        || offset_exit.len() != n
        || marginal_offset.len() != n
        || logslope_offset.len() != n
        || sample_weights.len() != n
        || event.len() != n
    {
        return Err(format!(
            "survival_nonrigid_pilot_eta: row-count mismatch (n={n}, location={}, logslope={}, \
             z={}, offset_exit={}, marginal_offset={}, logslope_offset={}, weights={}, event={})",
            location_anchor_design.nrows(),
            logslope_design.nrows(),
            z_primary.len(),
            offset_exit.len(),
            marginal_offset.len(),
            logslope_offset.len(),
            sample_weights.len(),
            event.len(),
        ));
    }
    let p_loc = location_anchor_design.ncols();
    let p_g = logslope_design.ncols();
    let p_joint = p_loc + p_g;
    if p_joint == 0 {
        return Ok((
            survival_rigid_pilot_eta(
                n,
                z_primary,
                offset_exit,
                marginal_offset,
                logslope_offset,
                baseline_slope,
                probit_scale,
            ),
            Array1::<f64>::zeros(p_g),
        ));
    }
    // Starting pilot (offset-only). Decompose into q_exit and slope so the
    // chain rule below can attribute the η₁ Newton step back to each piece.
    let mut q_exit = Array1::<f64>::zeros(n);
    let mut slope = Array1::<f64>::zeros(n);
    let mut eta1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        q_exit[i] = offset_exit[i] + marginal_offset[i];
        slope[i] = baseline_slope + logslope_offset[i];
        eta1[i] = rigid_observed_eta(q_exit[i], slope[i], z_primary[i], probit_scale);
    }
    // Per-row chain factors and IRLS gradient/Hessian along η₁.
    //
    //   η₁ = q·c(g) + s(g)·z   with c(g) = sqrt(1 + s(g)²), s(g) = observed_logslope(g)
    //   ∂η₁/∂q = c(g)
    //   ∂η₁/∂g = q·c'(g) + s'(g)·z
    //
    // The chain factors come from `rigid_observed_eta` via numerical
    // finite-difference on a tiny step (the parametric closed-form
    // derivatives live further down in `c_derivatives`; reusing it would
    // couple this helper to private internals, while finite-difference at
    // 1e-7 is well within the IRLS pilot's accuracy budget — the result is
    // just used to weight the W metric, not propagated into a final
    // coefficient).
    let mut chain_q = Array1::<f64>::zeros(n);
    let mut chain_g = Array1::<f64>::zeros(n);
    let mut grad_eta1 = Array1::<f64>::zeros(n);
    let mut hess_eta1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        let g_i = slope[i];
        let z_i = z_primary[i];
        let h_fd: f64 = 1.0e-7;
        chain_q[i] = (rigid_observed_eta(q_exit[i] + h_fd, g_i, z_i, probit_scale)
            - rigid_observed_eta(q_exit[i] - h_fd, g_i, z_i, probit_scale))
            / (2.0 * h_fd);
        chain_g[i] = (rigid_observed_eta(q_exit[i], g_i + h_fd, z_i, probit_scale)
            - rigid_observed_eta(q_exit[i], g_i - h_fd, z_i, probit_scale))
            / (2.0 * h_fd);
        // Row gradient and Hessian along η₁ (mirror
        // `survival_pilot_irls_row_metric_at_eta`'s formula):
        //   censored:  d(-log Φ(-η))/dη at weight w·(1-d). The primitive
        //              `signed_probit_neglog_derivatives_up_to_fourth(-η, c)`
        //              returns derivatives wrt its first argument; the chain
        //              rule wrt η is therefore  d/dη = -d/d(-η).
        //   event:     d(η²/2)/dη · w·d = w·d·η, Hessian = w·d.
        let (k1, k2, _, _) = signed_probit_neglog_derivatives_up_to_fourth(
            -eta1[i],
            sample_weights[i] * (1.0 - event[i]),
        )
        .map_err(|e| format!("survival_nonrigid_pilot_eta: row {i}: {e}"))?;
        let event_w = sample_weights[i] * event[i];
        grad_eta1[i] = -k1 + event_w * eta1[i];
        hess_eta1[i] = k2 + event_w;
        if !hess_eta1[i].is_finite() || hess_eta1[i] < 0.0 {
            // Defensive: any non-PSD row would corrupt the joint Gram. Clamp
            // to a tiny positive so the IRLS step degenerates to a small
            // proximal update rather than producing NaN.
            hess_eta1[i] = hess_eta1[i].max(0.0);
        }
        if !grad_eta1[i].is_finite() {
            grad_eta1[i] = 0.0;
        }
    }
    // Normal equations: (Xᵀ W X + λI) β = -Xᵀ g, where W = diag(hess_eta1)
    // along η₁, g = grad_eta1, and X is the η₁ chain-corrected joint design
    // (each location column scaled by chain_q, each logslope column by
    // chain_g). X is never materialized at full height: a one-shot dense
    // `(n, p_joint)` build was ~0.7 GiB at biobank scale (n=320k) and sat
    // co-resident with the construction phase's other dense transients —
    // one of the contributors to the #979 large-scale OOM. Instead the Gram
    // and rhs accumulate over fixed-height row chunks, so peak extra memory
    // is `chunk × p_joint` regardless of n.
    let mut gram = Array2::<f64>::zeros((p_joint, p_joint));
    let mut rhs = Array1::<f64>::zeros(p_joint);
    const PILOT_ROW_CHUNK: usize = 4096;
    let mut x_chunk = Array2::<f64>::zeros((PILOT_ROW_CHUNK.min(n), p_joint));
    let mut chunk_start = 0usize;
    while chunk_start < n {
        let chunk_end = (chunk_start + PILOT_ROW_CHUNK).min(n);
        let rows = chunk_end - chunk_start;
        let loc_rows = location_anchor_design
            .try_row_chunk(chunk_start..chunk_end)
            .map_err(|e| format!("survival_nonrigid_pilot_eta: location anchor rows: {e}"))?;
        let g_rows = logslope_design
            .try_row_chunk(chunk_start..chunk_end)
            .map_err(|e| format!("survival_nonrigid_pilot_eta: logslope rows: {e}"))?;
        {
            let mut x_view = x_chunk.slice_mut(s![..rows, ..]);
            for local in 0..rows {
                let i = chunk_start + local;
                for j in 0..p_loc {
                    x_view[[local, j]] = chain_q[i] * loc_rows[[local, j]];
                }
                for j in 0..p_g {
                    x_view[[local, p_loc + j]] = chain_g[i] * g_rows[[local, j]];
                }
            }
        }
        let h_chunk = hess_eta1.slice(s![chunk_start..chunk_end]).to_owned();
        let mut neg_g_chunk = Array1::<f64>::zeros(rows);
        for local in 0..rows {
            neg_g_chunk[local] = -grad_eta1[chunk_start + local];
        }
        if rows == x_chunk.nrows() {
            gram += &fast_xt_diag_x(&x_chunk, &h_chunk);
            rhs += &fast_atv(&x_chunk, &neg_g_chunk);
        } else {
            let x_tail = x_chunk.slice(s![..rows, ..]).to_owned();
            gram += &fast_xt_diag_x(&x_tail, &h_chunk);
            rhs += &fast_atv(&x_tail, &neg_g_chunk);
        }
        chunk_start = chunk_end;
    }
    // Adaptive ridge: 1e-6 × average diagonal, floored at 1e-12. Keeps the
    // Cholesky well-conditioned even when the rigid design has near-null
    // directions (which it often does at construction — the whole point of
    // the eventual cross-block reparam).
    let avg_diag = if p_joint > 0 {
        (0..p_joint).map(|j| gram[[j, j]]).sum::<f64>() / (p_joint as f64)
    } else {
        0.0
    };
    let ridge_eff = (1.0e-6 * avg_diag).max(1.0e-12);
    for j in 0..p_joint {
        gram[[j, j]] += ridge_eff;
    }
    let factor = gram
        .cholesky(faer::Side::Lower)
        .map_err(|e| format!("survival_nonrigid_pilot_eta: Cholesky failed: {e:?}"))?;
    let beta_step = factor.solvevec(&rhs);
    // Apply the Newton update: pilot q_exit ← q_exit + chain_q · β_loc,
    // pilot slope ← slope + chain_g · β_g — but the chain factors were
    // already folded into `x_chain` above so the row delta along η₁ is
    // simply `x_chain[i,:] · β_step`. We split it back into q and g by
    // re-projecting through the bare designs (without chain factors).
    let mut beta_loc = Array1::<f64>::zeros(p_loc);
    let mut beta_g = Array1::<f64>::zeros(p_g);
    for j in 0..p_loc {
        beta_loc[j] = beta_step[j];
    }
    for j in 0..p_g {
        beta_g[j] = beta_step[p_loc + j];
    }
    let q_delta = location_anchor_design.apply(&beta_loc);
    let g_delta = logslope_design.apply(&beta_g);
    // Trust-region cap to prevent a runaway first step on ill-conditioned
    // pilots: limit |Δη₁| per row to 4·σ_η (σ_η ≈ 1 under probit), measured
    // by the rigid pilot's η₁ standard deviation. This keeps the pilot in
    // the regime where the second-order Taylor approximation is meaningful;
    // the cross-block W metric only needs a per-row varying η₁, not the
    // converged β.
    let mut step_cap: f64 = 4.0;
    {
        let mean: f64 = eta1.iter().sum::<f64>() / (n as f64).max(1.0);
        let mut var: f64 = 0.0;
        for i in 0..n {
            let d = eta1[i] - mean;
            var += d * d;
        }
        let sd = (var / (n as f64).max(1.0)).sqrt();
        if sd.is_finite() && sd > 0.0 {
            step_cap = (4.0_f64).max(4.0 * sd);
        }
    }
    let mut pilot_eta = Array1::<f64>::zeros(n);
    for i in 0..n {
        let q_new = q_exit[i] + q_delta[i];
        let g_new = slope[i] + g_delta[i];
        let proposed = rigid_observed_eta(q_new, g_new, z_primary[i], probit_scale);
        let delta = proposed - eta1[i];
        let capped = if delta.abs() > step_cap {
            eta1[i] + step_cap.copysign(delta)
        } else {
            proposed
        };
        pilot_eta[i] = if capped.is_finite() { capped } else { eta1[i] };
    }
    // Logslope-surface warm start (#808): the `G`-block portion of the joint
    // Newton step, used to seed the logslope block's `initial_beta` off the
    // `g = 0` seed where the block is W-null. Sanitise to finite values; the
    // per-row logslope value `baseline_slope + logslope_offset + G·β_g` is the
    // operating point the inner refines from, so a non-finite coefficient
    // (degenerate one-step solve) falls back to the zero warm start rather than
    // poisoning the seed.
    let beta_logslope = if beta_g.iter().all(|v| v.is_finite()) {
        beta_g
    } else {
        Array1::<f64>::zeros(p_g)
    };
    Ok((pilot_eta, beta_logslope))
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


/// Splice `at row {row}` between the reason prefix and the `:` details
/// separator so that errors from the scalar `row_primary_closed_form*`
/// kernels (which are intentionally row-agnostic) match the canonical
/// `<reason> at row N: <details>` shape that downstream consumers match on.
fn with_row_context(err: String, row: usize) -> String {
    if let Some(colon) = err.find(':') {
        let (head, tail) = err.split_at(colon);
        format!("{head} at row {row}{tail}")
    } else {
        format!("{err} at row {row}")
    }
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


/// Crate-visible wrapper around `row_primary_closed_form` so the
/// identifiability-compiler sibling module
/// (`survival_marginal_slope_identifiability`) can build its 4×4
/// `SurvivalRowHessian` without exposing the closed-form kernel publicly.
pub(crate) fn row_primary_for_compiler(
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
    row_primary_closed_form(q0, q1, qd1, g, z, w, d, derivative_guard, probit_scale)
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
