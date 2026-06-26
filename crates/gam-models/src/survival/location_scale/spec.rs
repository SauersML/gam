use super::*;

/// How a time block's parameterization enforces the derivative-guard
/// monotonicity `q'(t) ≥ guard`.
///
/// The constraint set fed to the inner active-set / KKT machinery depends on
/// the variant; consuming families dispatch on this to choose the right
/// constraint shape and to refuse a mismatched parameterization (e.g.
/// `survival_marginal_slope` cannot ride a coordinate-cone-only basis
/// without re-introducing the phantom-multiplier bug it solved with the
/// row-wise representation; `survival_location_scale` cannot ride a
/// row-wise representation without making its reduced KKT system
/// rank-deficient on the cone basis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeBlockMonotonicity {
    /// The time block's coefficients are constrained by a per-coordinate
    /// cone `β_j ≥ 0` (with appropriate offsets handled by the family).
    /// Used by location-scale / latent paths whose bases produce a
    /// non-negative derivative whenever the cone holds.
    EnforcedByCoordinateCone,
    /// The time block's coefficients are constrained by row-wise
    /// `D β + o ≥ guard` over every observation row; needed when the
    /// basis admits negative-derivative directions that no coordinate
    /// cone can encode without leaving phantom KKT multipliers when a
    /// row binds. Used by `survival_marginal_slope` under the additive
    /// base.
    EnforcedByRowConstraint,
    /// The base is a structurally-monotone parameterization (e.g.
    /// `q'(t) = guard + I(t)·γ` with `γ ≥ 0`). Monotonicity holds
    /// pointwise from the cone; the family treats this exactly as a
    /// coordinate cone for constraint generation but the geometric
    /// claim is stronger and is recorded here for diagnostics and for
    /// future fast paths (e.g. skipping per-row validation).
    StructuralISpline,
}

impl TimeBlockMonotonicity {
    /// True when the variant can be enforced by a coordinate cone alone
    /// (no row-wise constraints required). Both `EnforcedByCoordinateCone`
    /// and `StructuralISpline` satisfy this; only `EnforcedByRowConstraint`
    /// requires the row-wise `D β ≥ b` constraint matrix.
    #[inline]
    pub fn is_coordinate_cone(self) -> bool {
        matches!(
            self,
            Self::EnforcedByCoordinateCone | Self::StructuralISpline
        )
    }

    /// True when row-wise `D β + o ≥ guard` constraints must be emitted
    /// for the inner active-set/KKT machinery to capture binding
    /// multipliers correctly.
    #[inline]
    pub fn requires_row_constraints(self) -> bool {
        matches!(self, Self::EnforcedByRowConstraint)
    }
}

#[derive(Clone)]
pub struct TimeBlockInput {
    pub design_entry: DesignMatrix,
    pub design_exit: DesignMatrix,
    pub design_derivative_exit: DesignMatrix,
    pub offset_entry: Array1<f64>,
    pub offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    /// How the time block enforces `q'(t) ≥ guard`. The consuming family
    /// dispatches the constraint shape on this and refuses a mismatch
    /// rather than silently producing a degenerate KKT system.
    pub time_monotonicity: TimeBlockMonotonicity,
    pub penalties: Vec<Array2<f64>>,
    /// Structural nullspace dimension of each penalty matrix.
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

/// A covariate block whose linear predictor depends on the survival time axis
/// via a tensor product: covariate design (n x p_cov) ⊗ B-spline on log(time).
///
/// At row i the linear predictor evaluated at time t is
///
///   eta(t) = [ x_cov(i,:) ⊗ B_time(t) ] @ beta
///
/// where B_time(t) is a B-spline basis row evaluated at log(t).
/// The entry and exit tensor designs are precomputed:
///   X_entry[i,:] = x_cov(i,:) ⊗ B_time(t_entry_i)
///   X_exit[i,:]  = x_cov(i,:) ⊗ B_time(t_exit_i)
#[derive(Clone)]
pub struct TimeDependentCovariateBlockInput {
    /// Covariate design matrix (n x p_cov), same for all time points.
    pub design_covariates: DesignMatrix,
    /// B-spline time basis at entry times (n x p_time).
    pub time_basis_entry: Array2<f64>,
    /// B-spline time basis at exit times (n x p_time).
    pub time_basis_exit: Array2<f64>,
    /// Derivative of the time basis with respect to clock time at exit.
    pub time_basis_derivative_exit: Array2<f64>,
    /// Combined Kronecker penalties for the tensor product.
    pub penalties: Vec<PenaltyMatrix>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
    pub offset: Array1<f64>,
}

/// Whether a covariate block (threshold or log-sigma) is time-invariant or
/// depends on the survival time axis via a tensor product.
#[derive(Clone)]
pub enum CovariateBlockKind {
    Static(ParameterBlockInput),
    TimeVarying(TimeDependentCovariateBlockInput),
}

#[derive(Clone)]
pub struct LinkWiggleBlockInput {
    pub design: DesignMatrix,
    pub knots: Array1<f64>,
    pub degree: usize,
    pub penalties: Vec<gam_problem::PenaltySpec>,
    /// Structural nullspace dimension of each penalty matrix.
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct TimeWiggleBlockInput {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub ncols: usize,
}

#[derive(Clone)]
pub(crate) struct SurvivalLocationScaleSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub inverse_link: InverseLink,
    pub derivative_guard: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub time_block: TimeBlockInput,
    pub threshold_block: CovariateBlockKind,
    pub log_sigma_block: CovariateBlockKind,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
    pub linkwiggle_block: Option<LinkWiggleBlockInput>,
    /// Explicit persistent warm-start cache session. See
    /// [`BlockwiseFitOptions::cache_session`].
    pub cache_session: Option<std::sync::Arc<gam_runtime::warm_start::Session>>,
    /// Persistent warm-start mirror sessions; see
    /// [`BlockwiseFitOptions::cache_mirror_sessions`].
    pub cache_mirror_sessions: Vec<std::sync::Arc<gam_runtime::warm_start::Session>>,
}

#[derive(Clone)]
pub enum SurvivalCovariateTermBlockTemplate {
    Static,
    TimeVarying {
        time_basis_entry: Array2<f64>,
        time_basis_exit: Array2<f64>,
        time_basis_derivative_exit: Array2<f64>,
        time_penalties: Vec<Array2<f64>>,
    },
}

#[derive(Clone)]
pub struct SurvivalLocationScaleTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub inverse_link: InverseLink,
    /// Strict lower bound on d_eta/dt used by both the event Jacobian term
    /// and the time monotonicity constraints.
    pub derivative_guard: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub time_block: TimeBlockInput,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub threshold_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
    pub threshold_template: SurvivalCovariateTermBlockTemplate,
    pub log_sigma_template: SurvivalCovariateTermBlockTemplate,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
    pub linkwiggle_block: Option<LinkWiggleBlockInput>,
    /// Optional warm-start seed for the threshold-block log-smoothing parameters (ρ).
    /// When `Some`, its length must equal the number of threshold penalties; values are
    /// clamped to the outer-loop ρ bounds before being injected into `rho0`.
    /// Used by the outer baseline-config optimizer to thread converged smoothing
    /// from one probe into the next.
    pub initial_threshold_log_lambdas: Option<Array1<f64>>,
    /// Optional warm-start seed for the log-sigma-block log-smoothing parameters (ρ).
    /// Same semantics as `initial_threshold_log_lambdas`.
    pub initial_log_sigma_log_lambdas: Option<Array1<f64>>,
    /// Explicit persistent warm-start cache session. See
    /// [`crate::families::custom_family::BlockwiseFitOptions::cache_session`].
    pub cache_session: Option<std::sync::Arc<gam_runtime::warm_start::Session>>,
    /// Explicit persistent warm-start mirror sessions. See
    /// [`crate::families::custom_family::BlockwiseFitOptions::cache_mirror_sessions`].
    pub cache_mirror_sessions: Vec<std::sync::Arc<gam_runtime::warm_start::Session>>,
}

pub const DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD: f64 = 1e-6;

pub struct SurvivalLocationScaleTermFitResult {
    pub fit: UnifiedFitResult,
    pub resolved_thresholdspec: TermCollectionSpec,
    pub resolved_log_sigmaspec: TermCollectionSpec,
    pub threshold_design: TermCollectionDesign,
    pub log_sigma_design: TermCollectionDesign,
    /// Per-row gradient of unpenalized NLL w.r.t. the three additive time-block
    /// offset channels (entry / exit / derivative-at-exit) at the converged β.
    /// Contracted with `∂o/∂θ_baseline` this yields the analytic θ-gradient
    /// used by the with-gradient baseline optimizer.
    pub baseline_offset_residuals: OffsetChannelResiduals,
    /// 3×3 NLL Hessian per row on the offset channels, in
    /// `(entry, exit, derivative)` order. Diagonal under location-scale —
    /// the row likelihood is separable in `(u0, u1, g)`. Used by the analytic
    /// θ-Hessian builder (chain rule second derivative).
    pub baseline_offset_curvatures: OffsetChannelCurvatures,
    /// Exact data-fit gradient `∂(−ℓ)/∂θ_link` of the unpenalized
    /// log-likelihood w.r.t. the inverse-link parameters at the converged β̂
    /// (`None` when the inverse link carries no free parameters). Equals the
    /// envelope-theorem θ_link-gradient of the profile penalized NLL, consumed
    /// by the inverse-link BFGS optimizer.
    pub link_param_data_fit_gradient: Option<Array1<f64>>,
}

/// Helper struct so callers can build a `UnifiedFitResult` from
/// survival-specific fields without knowing about the unified layout.
pub struct SurvivalLocationScaleFitResultParts {
    pub beta_time: Array1<f64>,
    pub beta_threshold: Array1<f64>,
    pub beta_log_sigma: Array1<f64>,
    pub beta_link_wiggle: Option<Array1<f64>>,
    pub link_wiggle_knots: Option<Array1<f64>>,
    pub link_wiggle_degree: Option<usize>,
    pub lambdas_time: Array1<f64>,
    pub lambdas_threshold: Array1<f64>,
    pub lambdas_log_sigma: Array1<f64>,
    pub lambdas_linkwiggle: Option<Array1<f64>>,
    pub log_likelihood: f64,
    pub reml_score: f64,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    /// Whether any GPU device executed part of this fit (GPU-flag propagation).
    /// Survival location-scale runs on the CPU path, so this is `false`; it is
    /// carried so the assembled `UnifiedFitResultParts` reports a real value.
    pub used_device: bool,
    pub outer_iterations: usize,
    /// `None` = no gradient measured at termination; `Some(g)` = measured.
    /// `outer_converged` is the authoritative convergence signal.
    pub outer_gradient_norm: Option<f64>,
    pub outer_converged: bool,
    pub covariance_conditional: Option<Array2<f64>>,
    pub geometry: Option<FitGeometry>,
}

#[derive(Clone, Copy)]
pub(crate) struct SurvivalLambdaLayout {
    pub(crate) k_time: usize,
    pub(crate) k_threshold: usize,
    pub(crate) k_log_sigma: usize,
    pub(crate) k_wiggle: usize,
}

impl SurvivalLambdaLayout {
    pub(crate) fn new(
        k_time: usize,
        k_threshold: usize,
        k_log_sigma: usize,
        k_wiggle: usize,
    ) -> Self {
        Self {
            k_time,
            k_threshold,
            k_log_sigma,
            k_wiggle,
        }
    }

    pub(crate) fn total(&self) -> usize {
        self.k_time + self.k_threshold + self.k_log_sigma + self.k_wiggle
    }

    pub(crate) fn time_range(&self) -> std::ops::Range<usize> {
        0..self.k_time
    }

    pub(crate) fn threshold_range(&self) -> std::ops::Range<usize> {
        self.k_time..self.k_time + self.k_threshold
    }

    pub(crate) fn log_sigma_range(&self) -> std::ops::Range<usize> {
        self.k_time + self.k_threshold..self.k_time + self.k_threshold + self.k_log_sigma
    }

    pub(crate) fn wiggle_range(&self) -> std::ops::Range<usize> {
        self.k_time + self.k_threshold + self.k_log_sigma..self.total()
    }

    pub(crate) fn validate_rho(&self, rho: &Array1<f64>, label: &str) -> Result<(), String> {
        if rho.len() != self.total() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} rho length mismatch: got {}, expected {}",
                    rho.len(),
                    self.total()
                ),
            }
            .into());
        }
        Ok::<(), _>(())
    }

    pub(crate) fn time_from(&self, rho: &Array1<f64>) -> Array1<f64> {
        let range = self.time_range();
        rho.slice(s![range.start..range.end]).to_owned()
    }

    pub(crate) fn threshold_from(&self, rho: &Array1<f64>) -> Array1<f64> {
        let range = self.threshold_range();
        rho.slice(s![range.start..range.end]).to_owned()
    }

    pub(crate) fn log_sigma_from(&self, rho: &Array1<f64>) -> Array1<f64> {
        let range = self.log_sigma_range();
        rho.slice(s![range.start..range.end]).to_owned()
    }

    pub(crate) fn wiggle_from(&self, rho: &Array1<f64>) -> Option<Array1<f64>> {
        if self.k_wiggle == 0 {
            None
        } else {
            let range = self.wiggle_range();
            Some(rho.slice(s![range.start..range.end]).to_owned())
        }
    }
}

/// Build a `UnifiedFitResult` from survival-specific fields.
pub fn survival_fit_from_parts(
    parts: SurvivalLocationScaleFitResultParts,
) -> Result<UnifiedFitResult, String> {
    let SurvivalLocationScaleFitResultParts {
        beta_time,
        beta_threshold,
        beta_log_sigma,
        beta_link_wiggle,
        link_wiggle_knots,
        link_wiggle_degree,
        lambdas_time,
        lambdas_threshold,
        lambdas_log_sigma,
        lambdas_linkwiggle,
        log_likelihood,
        reml_score,
        stable_penalty_term,
        penalized_objective,
        used_device,
        outer_iterations,
        outer_gradient_norm,
        outer_converged,
        covariance_conditional,
        geometry,
    } = parts;

    // Validation (preserved from the old impl).
    validate_all_finite_estimation("survival_fit.beta_time", beta_time.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.beta_threshold",
        beta_threshold.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.beta_log_sigma",
        beta_log_sigma.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    if let Some(beta_wiggle) = beta_link_wiggle.as_ref() {
        validate_all_finite_estimation(
            "survival_fit.beta_link_wiggle",
            beta_wiggle.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        let knots = link_wiggle_knots.as_ref().ok_or_else(|| {
            "survival_fit.beta_link_wiggle requires link_wiggle_knots".to_string()
        })?;
        validate_all_finite_estimation("survival_fit.link_wiggle_knots", knots.iter().copied())
            .map_err(|e| e.to_string())?;
        if link_wiggle_degree.is_none() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival_fit.beta_link_wiggle requires link_wiggle_degree".to_string(),
            }
            .into());
        }
    } else if link_wiggle_knots.is_some() || link_wiggle_degree.is_some() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "survival_fit link-wiggle metadata requires beta_link_wiggle coefficients"
                .to_string(),
        }
        .into());
    }
    validate_all_finite_estimation("survival_fit.lambdas_time", lambdas_time.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.lambdas_threshold",
        lambdas_threshold.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.lambdas_log_sigma",
        lambdas_log_sigma.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    // Each block's smoothing-parameter count counts the number of distinct
    // penalty terms acting on that block's coefficients. A penalty term cannot
    // outnumber the coefficients it penalizes, so reject `lambdas_<block>`
    // vectors longer than the corresponding `beta_<block>`. This catches stale
    // / misaligned lambda slices that would otherwise propagate silently into
    // downstream inference where the per-block penalty bookkeeping is
    // unrecoverable.
    if lambdas_time.len() > beta_time.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival_fit.lambdas_time has {} entries but beta_time has only {} \
                 coefficients; each lambda corresponds to a penalty term on this block",
                lambdas_time.len(),
                beta_time.len()
            ),
        }
        .into());
    }
    if lambdas_threshold.len() > beta_threshold.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival_fit.lambdas_threshold has {} entries but beta_threshold has only {} \
                 coefficients; each lambda corresponds to a penalty term on this block",
                lambdas_threshold.len(),
                beta_threshold.len()
            ),
        }
        .into());
    }
    if lambdas_log_sigma.len() > beta_log_sigma.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival_fit.lambdas_log_sigma has {} entries but beta_log_sigma has only {} \
                 coefficients; each lambda corresponds to a penalty term on this block",
                lambdas_log_sigma.len(),
                beta_log_sigma.len()
            ),
        }
        .into());
    }
    if let Some(lambdas_wiggle) = lambdas_linkwiggle.as_ref() {
        if beta_link_wiggle.is_none() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival_fit.lambdas_linkwiggle requires beta_link_wiggle".to_string(),
            }
            .into());
        }
        validate_all_finite_estimation(
            "survival_fit.lambdas_linkwiggle",
            lambdas_wiggle.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        let wiggle_len = beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
        if lambdas_wiggle.len() > wiggle_len {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival_fit.lambdas_linkwiggle has {} entries but beta_link_wiggle has \
                     only {} coefficients; each lambda corresponds to a penalty term on this block",
                    lambdas_wiggle.len(),
                    wiggle_len
                ),
            }
            .into());
        }
    }
    ensure_finite_scalar_estimation("survival_fit.log_likelihood", log_likelihood)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.reml_score", reml_score)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.stable_penalty_term", stable_penalty_term)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.penalized_objective", penalized_objective)
        .map_err(|e| e.to_string())?;
    if let Some(g) = outer_gradient_norm {
        ensure_finite_scalar_estimation("survival_fit.outer_gradient_norm", g)
            .map_err(|e| e.to_string())?;
    }

    let total_p = beta_time.len()
        + beta_threshold.len()
        + beta_log_sigma.len()
        + beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
    if let Some(cov) = covariance_conditional.as_ref() {
        validate_all_finite_estimation("survival_fit.covariance_conditional", cov.iter().copied())
            .map_err(|e| e.to_string())?;
        let (rows, cols) = cov.dim();
        if rows != total_p || cols != total_p {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "survival_fit.covariance_conditional must be {}x{}, got {}x{}",
                    total_p, total_p, rows, cols
                ),
            }
            .into());
        }
    }
    if let Some(geom) = geometry.as_ref() {
        geom.validate_numeric_finiteness()
            .map_err(|e| e.to_string())?;
        let (rows, cols) = geom.penalized_hessian.dim();
        if rows != total_p || cols != total_p {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "survival_fit.geometry.penalized_hessian must be {}x{}, got {}x{}",
                    total_p, total_p, rows, cols
                ),
            }
            .into());
        }
        if geom.working_weights.len() != geom.working_response.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival_fit.geometry working length mismatch: weights={}, response={}",
                    geom.working_weights.len(),
                    geom.working_response.len()
                ),
            }
            .into());
        }
    }

    // Build blocks for the unified representation.
    use crate::model_types::{BlockRole, FittedBlock, FittedLinkState, UnifiedFitResultParts};
    let mut blocks = vec![
        FittedBlock {
            beta: beta_time.clone(),
            role: BlockRole::Time,
            edf: 0.0,
            lambdas: lambdas_time.clone(),
        },
        FittedBlock {
            beta: beta_threshold.clone(),
            role: BlockRole::Threshold,
            edf: 0.0,
            lambdas: lambdas_threshold.clone(),
        },
        FittedBlock {
            beta: beta_log_sigma.clone(),
            role: BlockRole::Scale,
            edf: 0.0,
            lambdas: lambdas_log_sigma.clone(),
        },
    ];
    if let Some(ref bw) = beta_link_wiggle {
        blocks.push(FittedBlock {
            beta: bw.clone(),
            role: BlockRole::LinkWiggle,
            edf: 0.0,
            lambdas: lambdas_linkwiggle
                .clone()
                .unwrap_or_else(|| Array1::zeros(0)),
        });
    }
    let all_lambdas: Vec<f64> = blocks
        .iter()
        .flat_map(|b| b.lambdas.iter().copied())
        .collect();
    let log_lambdas = Array1::from_vec(
        all_lambdas
            .iter()
            .map(|&v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY })
            .collect(),
    );
    let deviance = -2.0 * log_likelihood;
    crate::model_types::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks,
        log_lambdas,
        lambdas: Array1::from_vec(all_lambdas),
        likelihood_family: None,
        likelihood_scale: crate::types::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: crate::types::LogLikelihoodNormalization::UserProvided,
        log_likelihood,
        deviance,
        reml_score,
        stable_penalty_term,
        penalized_objective,
        used_device,
        outer_iterations,
        outer_converged,
        outer_gradient_norm,
        standard_deviation: 1.0,
        covariance_conditional,
        covariance_corrected: None,
        inference: None,
        fitted_link: FittedLinkState::Standard(None),
        geometry,
        block_states: Vec::new(),
        pirls_status: gam_solve::pirls::PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: crate::model_types::FitArtifacts {
            pirls: None,
            null_space_logdet: None,
            null_space_dim: None,
            survival_link_wiggle_knots: link_wiggle_knots,
            survival_link_wiggle_degree: link_wiggle_degree,
            criterion_certificate: None,
            rho_posterior_certificate: None,
            rho_posterior_escalation: None,
            rho_covariance: None,
        },
        inner_cycles: 0,
    })
    .map_err(|e| e.to_string())
}

#[derive(Clone)]
pub struct SurvivalLocationScalePredictInput {
    pub x_time_exit: Array2<f64>,
    pub eta_time_offset_exit: Array1<f64>,
    pub time_wiggle_knots: Option<Array1<f64>>,
    pub time_wiggle_degree: Option<usize>,
    pub time_wiggle_ncols: usize,
    pub x_threshold: DesignMatrix,
    pub eta_threshold_offset: Array1<f64>,
    pub x_log_sigma: DesignMatrix,
    pub eta_log_sigma_offset: Array1<f64>,
    pub x_link_wiggle: Option<DesignMatrix>,
    pub link_wiggle_knots: Option<Array1<f64>>,
    pub link_wiggle_degree: Option<usize>,
    pub inverse_link: InverseLink,
}

#[derive(Clone, Debug)]
pub struct SurvivalLocationScalePredictResult {
    pub eta: Array1<f64>,
    pub survival_prob: Array1<f64>,
}

#[derive(Clone)]
pub struct SurvivalLocationScalePredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub survival_prob: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub response_standard_error: Option<Array1<f64>>,
}

pub(crate) fn initial_log_lambdas<T>(
    penalties: &[T],
    rho0: Option<Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let k = penalties.len();
    let rho = rho0.unwrap_or_else(|| Array1::zeros(k));
    if rho.len() != k {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "initial_log_lambdas mismatch: got {}, expected {k}",
                rho.len()
            ),
        }
        .into());
    }
    Ok(rho)
}
