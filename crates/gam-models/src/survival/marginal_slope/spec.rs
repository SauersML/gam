//! Public input term-spec and fit-result types, the derivative-guard
//! tolerance defaults/helpers, and full input validation (`validate_spec`).
//! This is the user-facing data contract plus its integrity checks.

use super::*;

/// Family-owned survival-baseline coordinates for the joint LAML surface.
///
/// `Linear` is structurally fixed and contributes no family hyperparameter
/// axes. `Nonlinear` owns one frozen offset chart; its theta coordinates are
/// optimized jointly with smoothing, spatial, and learned-frailty axes.
#[derive(Clone, Debug)]
pub enum SurvivalMarginalSlopeBaselineHyperSpec {
    Linear,
    Nonlinear {
        chart: crate::survival::construction::SurvivalMarginalSlopeFrozenOffsetChart,
    },
}

impl SurvivalMarginalSlopeBaselineHyperSpec {
    pub(crate) fn initial_theta(&self) -> Option<&Array1<f64>> {
        match self {
            Self::Linear => None,
            Self::Nonlinear { chart } => Some(chart.initial_theta()),
        }
    }
}

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
    pub baseline_hyper: SurvivalMarginalSlopeBaselineHyperSpec,
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

pub(crate) const SURVIVAL_INTERCEPT_ABS_RESIDUAL_TOL: f64 = 1e-12;

pub(crate) const SURVIVAL_INTERCEPT_REL_TAIL_RESIDUAL_TOL: f64 = 1e-8;

pub(crate) const SURVIVAL_INTERCEPT_LOG_TAIL_THRESHOLD: f64 = 1e-8;

#[inline]
pub(crate) fn survival_derivative_guard_tolerance(qd1: f64, derivative_guard: f64) -> f64 {
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
    let solver_band = 4.0 * gam_solve::pirls::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL * magnitude;
    let eps_floor = 256.0 * f64::EPSILON * magnitude;
    solver_band.max(eps_floor)
}

#[inline]
pub(crate) fn survival_derivative_guard_violated(qd1: f64, derivative_guard: f64) -> bool {
    if !qd1.is_finite() {
        return true;
    }
    // NEG_INFINITY is the "no lower bound" sentinel used by callers that want to
    // skip the guard entirely (e.g. GPU rowjet tests that don't compute a bound).
    // Production paths assert derivative_guard is finite and positive before
    // calling this function, so this branch only fires in the unbounded case.
    if derivative_guard == f64::NEG_INFINITY {
        return false;
    }
    !derivative_guard.is_finite()
        || (qd1 + survival_derivative_guard_tolerance(qd1, derivative_guard) < derivative_guard)
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
    pub score_covariance: Array2<f64>,
    pub time_block_penalties_len: usize,
    pub time_wiggle_knots: Option<Array1<f64>>,
    pub time_wiggle_degree: Option<usize>,
    pub time_wiggle_ncols: usize,
    pub score_warp_runtime: Option<DeviationRuntime>,
    pub link_dev_runtime: Option<DeviationRuntime>,
    /// Width `p₁` of the absorbed Stage-1 influence block (#461) when the fit
    /// hosted a dedicated additive absorber (the trailing block). `None` when no
    /// CTN Stage-1 chain produced an influence Jacobian. The predictor drops the
    /// absorber's `γ`; this width lets it account for the extra trailing block
    /// and slice `γ` out of the joint covariance.
    pub influence_absorber_width: Option<usize>,
    /// Exact residualized training-row absorber design.  This is likelihood
    /// state, not prediction state: ordinary prediction drops the fitted
    /// absorber, while saved-model ALO must replay its row Jacobian exactly.
    pub influence_absorber_design: Option<Array2<f64>>,
}

pub(crate) fn validate_spec(spec: &SurvivalMarginalSlopeTermSpec) -> Result<(), String> {
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
    if let Some(jac) = spec.score_influence_jacobian.as_ref() {
        // #461 absorbed influence Jacobian `J = ∂z/∂θ₁` (n × p₁): must align with
        // the fit rows and be finite. A zero-column J carries no leakage
        // directions; the build site treats it as no absorber, but a row
        // mismatch or non-finite entry is a hard error (the residualization Gram
        // and the per-row Z̃ projection both assume `n` aligned finite rows).
        if jac.nrows() != n {
            return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                reason: format!(
                    "survival-marginal-slope score_influence_jacobian has {} rows, expected {n}",
                    jac.nrows()
                ),
            }
            .into());
        }
        if jac.iter().any(|&v| !v.is_finite()) {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival-marginal-slope score_influence_jacobian must be finite"
                    .to_string(),
            }
            .into());
        }
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
        FrailtySpec::GaussianShift { .. } => {}
        FrailtySpec::HazardMultiplier { .. } => {
            return Err(SurvivalMarginalSlopeError::InvalidInput {
                reason: "survival-marginal-slope does not support FrailtySpec::HazardMultiplier"
                    .to_string(),
            }
            .into());
        }
    }
    if matches!(
        &spec.baseline_hyper,
        SurvivalMarginalSlopeBaselineHyperSpec::Nonlinear { .. }
    ) && !spec.time_block.time_monotonicity.is_coordinate_cone()
    {
        return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: "learned survival marginal-slope baseline coordinates require a StructuralISpline coordinate cone; rowwise derivative constraints would move with the baseline offsets"
                .to_string(),
        }
        .into());
    }
    if spec.event_target.iter().any(|&d| d != 0.0 && d != 1.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires binary event indicators (0.0 or 1.0)"
                .to_string(),
        }
        .into());
    }
    // Fast-fail on a degenerate all-censored design: the marginal-slope partial
    // likelihood has no events to anchor the hazard scale, so the outer/inner
    // solve cannot make progress and otherwise spins without termination (#789B).
    if !spec.event_target.is_empty() && spec.event_target.iter().all(|&d| d == 0.0) {
        return Err(SurvivalMarginalSlopeError::InvalidInput {
            reason: "survival-marginal-slope requires at least one event (event==1); the supplied design is entirely censored (all event==0), which has no finite marginal-slope fit"
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
    if !spec.time_block.time_monotonicity.requires_row_constraints()
        && !spec.time_block.time_monotonicity.is_coordinate_cone()
    {
        return Err(SurvivalMarginalSlopeError::UnsupportedConfiguration {
            reason: format!(
                "survival-marginal-slope requires a row-constraint or coordinate-cone time block; got {:?}",
                spec.time_block.time_monotonicity
            ),
        }
        .into());
    }
    if spec.time_block.time_monotonicity.is_coordinate_cone() {
        for (row, &offset) in spec.time_block.derivative_offset_exit.iter().enumerate() {
            if !offset.is_finite() {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival-marginal-slope coordinate-cone time block has non-finite derivative offset at row {row}: {offset}"
                    ),
                }
                .into());
            }
            if offset < spec.derivative_guard - 1e-12 {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival-marginal-slope coordinate-cone time block requires derivative offset >= guard at row {row}: offset={offset:.3e}, guard={:.3e}",
                        spec.derivative_guard
                    ),
                }
                .into());
            }
        }
        let derivative_design = spec
            .time_block
            .design_derivative_exit
            .try_to_dense_by_chunks("survival marginal-slope coordinate-cone derivative audit")
            .map_err(|reason| SurvivalMarginalSlopeError::IncompatibleDimensions { reason })?;
        for ((row, col), &value) in derivative_design.indexed_iter() {
            if !value.is_finite() {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival-marginal-slope coordinate-cone time block has non-finite derivative design entry at row {row}, col {col}: {value}"
                    ),
                }
                .into());
            }
            if value < -1e-12 {
                return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                    reason: format!(
                        "survival-marginal-slope coordinate-cone time block requires nonnegative derivative design entries; row {row}, col {col} = {value:.3e}"
                    ),
                }
                .into());
            }
        }
    }
    if let Some(beta0) = &spec.time_block.initial_beta {
        match spec.time_block.time_monotonicity {
            monotonicity if monotonicity.is_coordinate_cone() => {
                // Under a coordinate-cone time basis, the solver enforces β ≥ 0
                // directly. The row-wise derivative guard is redundant because
                // validation above proves D ≥ 0 and offset ≥ guard.
                if spec.time_block.design_derivative_exit.ncols() != beta0.len() {
                    return Err(SurvivalMarginalSlopeError::IncompatibleDimensions {
                        reason: format!(
                            "survival-marginal-slope time_block initial_beta length mismatch under coordinate-cone monotonicity: got {}, expected {}",
                            beta0.len(),
                            spec.time_block.design_derivative_exit.ncols()
                        ),
                    }
                    .into());
                }
                for (j, &g) in beta0.iter().enumerate() {
                    if !g.is_finite() {
                        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                            reason: format!(
                                "survival-marginal-slope time_block initial_beta is non-finite at coordinate {j} under coordinate-cone monotonicity: got {g}"
                            ),
                        }
                        .into());
                    }
                    if g < -1e-12 {
                        return Err(SurvivalMarginalSlopeError::MonotonicityViolation {
                            reason: format!(
                                "survival-marginal-slope time_block initial_beta violates β ≥ 0 at coordinate {j} under coordinate-cone monotonicity: got {g:.3e}"
                            ),
                        }
                        .into());
                    }
                }
            }
            _ => {
                let derivative_constraints = time_derivative_guard_constraints(
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
                                    "survival-marginal-slope time_block initial_beta violates derivative guard constraint at row {row}: slack={slack:.3e}"
                                ),
                            }
                            .into());
                        }
                    }
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
