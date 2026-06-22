//! Jointly learned latent-frailty survival and binary deployment families with
//! a live time/baseline block.
//!
//! Model:
//!   H_0(a) = exp(q(a)),
//!   h_0(a) = dq(a)/da,
//!   H(a | U) = H_0(a) * exp(U),
//!   U ~ N(mu, sigma^2),
//!   mu = X beta + offset.
//!
//! Unlike the old compiled-row path, the cumulative masses and baseline hazard
//! are rebuilt inside the optimizer from the current time-basis coefficients.
//! The family-level fit surface supports exact events, right censoring, and
//! interval censoring `T ∈ (L, R]` (contribution `log[S(L) − S(R)]`). Interval
//! rows carry the reserved [`LATENT_SURVIVAL_EVENT_INTERVAL`] event code and a
//! dedicated upper-bound time channel (`time_design_right` / `q_right`); the
//! 3-way event dispatch is [`latent_survival_event_type_for`]. Reached from the
//! formula DSL via `SurvInterval(L, R, event) ~ ...`.

use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, fit_custom_family, fit_custom_family_fixed_log_lambdas,
};
use crate::families::gamlss::{FamilyMetadata, ParameterLink};
use crate::families::sigma_link::{exp_sigma_eta_for_sigma_scalar, exp_sigma_from_eta_scalar};
use crate::families::survival::latent::interval::{
    LatentFrailtyResolution, LatentIntervalModel, LatentIntervalRowView,
    validate_latent_interval_inputs,
};
use crate::families::survival::location_scale::{
    TimeBlockInput, project_onto_linear_constraints, structural_time_coefficient_constraints,
};
use crate::families::survival::lognormal_kernel::{
    FrailtySpec, HazardLoading, LatentSurvivalEventType, LatentSurvivalRow, LatentSurvivalRowJet,
    log_kernel_bundle,
};
use crate::matrix::{DenseDesignMatrix, DesignMatrix, SymmetricMatrix};
use crate::model_types::UnifiedFitResult;
use crate::pirls::LinearInequalityConstraints;
use crate::probability::signed_log_sum_exp;
use crate::quadrature::{IntegratedExpectationMode, QuadratureContext};
use crate::smooth::{
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design,
};
use crate::types::MIN_WEIGHT;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Typed error for the latent-survival / latent-binary family kernels and
/// their fit-time and per-row validation helpers. Variants pick the semantic
/// bucket while the inner `reason` carries the original byte-equivalent
/// message so external callers that previously consumed `String` errors keep
/// the same diagnostic text via `Display`.
#[derive(Debug, Clone)]
pub enum LatentSurvivalError {
    /// The frailty spec supplied to a latent-survival or latent-binary
    /// helper is incompatible (wrong variant, missing fixed sigma, non-finite
    /// or negative fixed sigma).
    InvalidFrailty { reason: String },
    /// Per-row dataset validation failed: empty input, size mismatch across
    /// the spec vectors, or invalid age / event / weight / unloaded-mass
    /// values for an individual row.
    InvalidDataset { reason: String },
    /// A parameter-block state, eta vector, or directional-derivative
    /// argument supplied to a family entry point has the wrong length.
    BlockMismatch { reason: String },
    /// A runtime numerical value (sigma, baseline hazard derivative, kernel
    /// sum, event probability) became non-finite or out-of-domain.
    NumericalFailure { reason: String },
    /// The requested combination of time-block structure or event type is
    /// not implemented (non-structural monotonicity, interval-censored rows
    /// on the dynamic-derivative path).
    UnsupportedConfiguration { reason: String },
}

crate::impl_reason_error_boilerplate! {
    LatentSurvivalError {
        InvalidFrailty,
        InvalidDataset,
        BlockMismatch,
        NumericalFailure,
        UnsupportedConfiguration,
    }
}

impl From<crate::families::block_layout::block_count::BlockCountMismatch> for LatentSurvivalError {
    fn from(
        err: crate::families::block_layout::block_count::BlockCountMismatch,
    ) -> LatentSurvivalError {
        LatentSurvivalError::BlockMismatch {
            reason: err.message(),
        }
    }
}

impl From<String> for LatentSurvivalError {
    /// Inbound conversion for the many `Result<_, String>` helpers this
    /// module still calls into (term-collection design assembly, dense
    /// chunk conversion, sparse linear constraints). The text is preserved
    /// verbatim; we only pick a category so external messages flow through
    /// `?` without per-callsite `.map_err`.
    fn from(reason: String) -> LatentSurvivalError {
        LatentSurvivalError::InvalidDataset { reason }
    }
}

/// Reserved [`LatentSurvivalTermSpec::event_target`] code marking an
/// interval-censored row `(L, R]`. Exact-event codes are `>= 1` and right
/// censoring is `0`; the interval code is the sentinel `u8::MAX` so it never
/// collides with an exact-event count and the dispatch is an explicit 3-way map
/// `{0 → RightCensored, INTERVAL → IntervalCensored, k ≥ 1 → ExactEvent}`.
pub const LATENT_SURVIVAL_EVENT_INTERVAL: u8 = u8::MAX;

#[inline]
fn latent_survival_event_type_for(code: u8) -> LatentSurvivalEventType {
    match code {
        0 => LatentSurvivalEventType::RightCensored,
        LATENT_SURVIVAL_EVENT_INTERVAL => LatentSurvivalEventType::IntervalCensored,
        _ => LatentSurvivalEventType::ExactEvent,
    }
}

#[derive(Clone)]
pub struct LatentSurvivalTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    /// Time-basis design evaluated at the interval upper bound `R` (so
    /// `q_right = design_right · β_time + offset_right`). `None` when the data
    /// carries no interval-censored rows; the family then reuses the exit design
    /// for the unused `q_right` channel. When `Some`, rows whose
    /// `event_target == LATENT_SURVIVAL_EVENT_INTERVAL` contribute the interval
    /// likelihood `log[S(L) − S(R)]`.
    pub time_design_right: Option<DesignMatrix>,
    pub time_offset_right: Option<Array1<f64>>,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    /// Unloaded (background) cumulative mass at the interval upper bound `R`.
    /// Length-`n`; entries for non-interval rows are ignored. Empty/`None`
    /// folds to zero (full-loading interval rows).
    pub unloaded_mass_right: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
}

pub struct LatentSurvivalTermFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub latent_sd: f64,
    /// Per-row residuals of the unpenalized NLL w.r.t. the additive baseline
    /// time-block offsets `(entry, exit, derivative)` at the converged β̂.
    /// Contracted against `baseline_offset_theta_partials` by
    /// `baseline_chain_rule_gradient` to give the exact θ-gradient of the
    /// profile penalized NLL for the outer baseline-config optimizer.
    pub baseline_offset_residuals: crate::families::survival::OffsetChannelResiduals,
}

#[derive(Clone)]
pub struct LatentBinaryTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub derivative_guard: f64,
    pub time_block: TimeBlockInput,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
}

pub struct LatentBinaryTermFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    /// Per-row residuals of the unpenalized NLL w.r.t. the additive baseline
    /// time-block offsets `(entry, exit)` at the converged β̂ (the derivative
    /// channel is identically zero for the binary deployment likelihood).
    pub baseline_offset_residuals: crate::families::survival::OffsetChannelResiduals,
}

#[derive(Clone)]
struct PreparedLatentTimeBlock {
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    /// Dense time-basis design at the interval upper bound `R`. Falls back to a
    /// clone of `design_exit` when the spec supplies no interval design, so the
    /// `q_right` channel is always well-defined (and unused for non-interval
    /// rows).
    design_right: Array2<f64>,
    linear_constraints: Option<LinearInequalityConstraints>,
    penalties: Vec<Array2<f64>>,
    initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct LatentSurvivalFamily {
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub latent_sd_fixed: Option<f64>,
    pub hazard_loading: HazardLoading,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub x_time_entry: Array2<f64>,
    pub x_time_exit: Array2<f64>,
    pub x_time_derivative_exit: Array2<f64>,
    /// Time-basis design evaluated at the interval upper bound `R` (so
    /// `q_right = x_time_right · β_time + time_offset_right`). For non-interval
    /// rows this row equals `x_time_exit`'s row (`q_right` is then unused by the
    /// likelihood), so the matrix always has `n` rows and the same column count
    /// as the other time designs.
    pub x_time_right: Array2<f64>,
    /// Time-block offset at the interval upper bound `R` (length `n`).
    pub time_offset_right: Array1<f64>,
    /// Unloaded (background) cumulative mass at the interval upper bound `R`
    /// (length `n`). Ignored for non-interval rows.
    pub unloaded_mass_right: Array1<f64>,
    pub x_mean: DesignMatrix,
    pub time_linear_constraints: Option<LinearInequalityConstraints>,
    pub quadctx: Arc<QuadratureContext>,
}

#[derive(Clone)]
pub struct LatentBinaryFamily {
    pub event_target: Array1<u8>,
    pub weights: Array1<f64>,
    pub latent_sd: f64,
    pub hazard_loading: HazardLoading,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub x_time_entry: Array2<f64>,
    pub x_time_exit: Array2<f64>,
    pub x_mean: DesignMatrix,
    pub time_linear_constraints: Option<LinearInequalityConstraints>,
    pub quadctx: Arc<QuadratureContext>,
}

impl LatentSurvivalFamily {
    pub const BLOCK_TIME: usize = 0;
    pub const BLOCK_MEAN: usize = 1;
    pub const BLOCK_LOG_SIGMA: usize = 2;

    pub fn parameter_names() -> &'static [&'static str] {
        &["time_transform", "mean"]
    }

    pub fn parameter_links() -> &'static [ParameterLink] {
        &[ParameterLink::Identity, ParameterLink::Identity]
    }

    pub fn metadata() -> FamilyMetadata {
        FamilyMetadata {
            name: "latent_survival",
            parameternames: Self::parameter_names(),
            parameter_links: Self::parameter_links(),
        }
    }

    fn split_time_eta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<
        (
            ArrayView1<'a, f64>,
            ArrayView1<'a, f64>,
            ArrayView1<'a, f64>,
            &'a Array1<f64>,
        ),
        LatentSurvivalError,
    > {
        let expected_blocks = if self.latent_sd_fixed.is_some() { 2 } else { 3 };
        crate::families::block_layout::block_count::validate_block_count::<LatentSurvivalError>(
            "LatentSurvivalFamily",
            expected_blocks,
            block_states.len(),
        )?;
        let n = self.event_target.len();
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_mean = &block_states[Self::BLOCK_MEAN].eta;
        if eta_time.len() != 3 * n {
            return Err(LatentSurvivalError::BlockMismatch {
                reason: format!(
                    "latent survival time eta length mismatch: got {}, expected {}",
                    eta_time.len(),
                    3 * n
                ),
            });
        }
        if eta_mean.len() != n || self.weights.len() != n {
            return Err(LatentSurvivalError::BlockMismatch {
                reason: "latent survival mean eta dimension mismatch".to_string(),
            });
        }
        Ok((
            eta_time.slice(s![0..n]),
            eta_time.slice(s![n..2 * n]),
            eta_time.slice(s![2 * n..3 * n]),
            eta_mean,
        ))
    }

    /// Per-row interval upper-bound time transform `q_right = x_time_right · β_time
    /// + time_offset_right`. Shares the time-block coefficients with `q_exit`
    /// (same monotone basis, evaluated at `R`), so it is read off the time
    /// block's `beta` rather than carried as an extra eta channel. For
    /// non-interval rows `x_time_right` equals `x_time_exit`, so the (unused)
    /// value is simply `q_exit`.
    fn time_q_right(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Array1<f64>, LatentSurvivalError> {
        let n = self.event_target.len();
        let beta_time = &block_states[Self::BLOCK_TIME].beta;
        if self.x_time_right.ncols() != beta_time.len() {
            return Err(LatentSurvivalError::BlockMismatch {
                reason: format!(
                    "latent survival interval right design has {} columns but time beta has {}",
                    self.x_time_right.ncols(),
                    beta_time.len()
                ),
            });
        }
        if self.x_time_right.nrows() != n || self.time_offset_right.len() != n {
            return Err(LatentSurvivalError::BlockMismatch {
                reason: "latent survival interval right design/offset row count mismatch"
                    .to_string(),
            });
        }
        let mut q_right = self.x_time_right.dot(beta_time);
        q_right += &self.time_offset_right;
        Ok(q_right)
    }

    fn latent_sd(&self, block_states: &[ParameterBlockState]) -> Result<f64, LatentSurvivalError> {
        if let Some(sigma) = self.latent_sd_fixed {
            return Ok(sigma);
        }
        let eta = *block_states
            .get(Self::BLOCK_LOG_SIGMA)
            .and_then(|state| state.eta.get(0))
            .ok_or_else(|| LatentSurvivalError::BlockMismatch {
                reason: "latent survival learnable log_sigma block is missing".to_string(),
            })?;
        let sigma = exp_sigma_from_eta_scalar(eta);
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!(
                    "latent survival learnable sigma became invalid: log_sigma={eta}, sigma={sigma}"
                ),
            });
        }
        Ok(sigma)
    }
}

impl LatentBinaryFamily {
    pub const BLOCK_TIME: usize = 0;
    pub const BLOCK_MEAN: usize = 1;

    fn split_time_eta<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<(ArrayView1<'a, f64>, ArrayView1<'a, f64>, &'a Array1<f64>), LatentSurvivalError>
    {
        crate::families::block_layout::block_count::validate_block_count::<LatentSurvivalError>(
            "LatentBinaryFamily",
            2,
            block_states.len(),
        )?;
        let n = self.event_target.len();
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_mean = &block_states[Self::BLOCK_MEAN].eta;
        if eta_time.len() != 3 * n {
            return Err(LatentSurvivalError::BlockMismatch {
                reason: format!(
                    "latent binary time eta length mismatch: got {}, expected {}",
                    eta_time.len(),
                    3 * n
                ),
            });
        }
        if eta_mean.len() != n || self.weights.len() != n {
            return Err(LatentSurvivalError::BlockMismatch {
                reason: "latent binary mean eta dimension mismatch".to_string(),
            });
        }
        Ok((
            eta_time.slice(s![0..n]),
            eta_time.slice(s![n..2 * n]),
            eta_mean,
        ))
    }
}

pub fn fixed_latent_hazard_frailty(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<(f64, HazardLoading), String> {
    fixed_latent_hazard_frailty_typed(frailty, context).map_err(Into::into)
}

fn fixed_latent_hazard_frailty_typed(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<(f64, HazardLoading), LatentSurvivalError> {
    match frailty {
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            loading,
        } if sigma.is_finite() && *sigma >= 0.0 => Ok((*sigma, *loading)),
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(sigma),
            ..
        } => Err(LatentSurvivalError::InvalidFrailty {
            reason: format!(
                "{context} requires a finite fixed hazard-multiplier sigma >= 0, got {sigma}"
            ),
        }),
        FrailtySpec::HazardMultiplier {
            sigma_fixed: None, ..
        } => Err(LatentSurvivalError::InvalidFrailty {
            reason: format!("{context} currently requires a fixed hazard-multiplier sigma"),
        }),
        FrailtySpec::GaussianShift { .. } => Err(LatentSurvivalError::InvalidFrailty {
            reason: format!("{context} requires HazardMultiplier frailty, not GaussianShift"),
        }),
        FrailtySpec::None => Err(LatentSurvivalError::InvalidFrailty {
            reason: format!("{context} requires a fixed HazardMultiplier frailty specification"),
        }),
    }
}

pub fn latent_hazard_loading(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<HazardLoading, String> {
    latent_hazard_loading_typed(frailty, context).map_err(Into::into)
}

fn latent_hazard_loading_typed(
    frailty: &FrailtySpec,
    context: &str,
) -> Result<HazardLoading, LatentSurvivalError> {
    match frailty {
        FrailtySpec::HazardMultiplier { loading, .. } => Ok(*loading),
        FrailtySpec::GaussianShift { .. } => Err(LatentSurvivalError::InvalidFrailty {
            reason: format!("{context} requires HazardMultiplier frailty, not GaussianShift"),
        }),
        FrailtySpec::None => Err(LatentSurvivalError::InvalidFrailty {
            reason: format!("{context} requires a HazardMultiplier frailty specification"),
        }),
    }
}

#[derive(Clone, Copy)]
struct LatentSurvivalTimeJet {
    grad_entry: f64,
    grad_exit: f64,
    neg_hess_entry: f64,
    neg_hess_exit: f64,
}

pub fn fit_latent_survival_terms(
    data: ArrayView2<'_, f64>,
    spec: LatentSurvivalTermSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentSurvivalTermFitResult, String> {
    let latent_sd = validate_latent_survival_inputs(data, &spec, &frailty)?;
    let hazard_loading = latent_hazard_loading(&frailty, "latent-survival")?;
    let mean_design =
        build_term_collection_design(data, &spec.meanspec).map_err(|e| e.to_string())?;
    let resolvedspec = freeze_term_collection_from_design(&spec.meanspec, &mean_design)
        .map_err(|e| e.to_string())?;
    let time_prepared = prepare_latent_time_block(
        &spec.time_block,
        spec.time_design_right.as_ref(),
        spec.derivative_guard,
    )?;

    let n = spec.event_target.len();
    let time_offset_right = match spec.time_offset_right.as_ref() {
        Some(offset) => {
            if offset.len() != n {
                return Err(format!(
                    "latent survival interval right time offset must have length {n}, got {}",
                    offset.len()
                ));
            }
            offset.clone()
        }
        None => Array1::zeros(n),
    };
    let unloaded_mass_right = if spec.unloaded_mass_right.is_empty() {
        Array1::zeros(n)
    } else {
        if spec.unloaded_mass_right.len() != n {
            return Err(format!(
                "latent survival interval right unloaded mass must have length {n}, got {}",
                spec.unloaded_mass_right.len()
            ));
        }
        spec.unloaded_mass_right.clone()
    };

    let family = LatentSurvivalFamily {
        event_target: spec.event_target.clone(),
        weights: spec.weights.clone(),
        latent_sd_fixed: latent_sd,
        hazard_loading,
        unloaded_mass_entry: spec.unloaded_mass_entry.clone(),
        unloaded_mass_exit: spec.unloaded_mass_exit.clone(),
        unloaded_hazard_exit: spec.unloaded_hazard_exit.clone(),
        x_time_entry: time_prepared.design_entry.clone(),
        x_time_exit: time_prepared.design_exit.clone(),
        x_time_derivative_exit: time_prepared.design_derivative_exit.clone(),
        x_time_right: time_prepared.design_right.clone(),
        time_offset_right,
        unloaded_mass_right,
        x_mean: mean_design.design.clone(),
        time_linear_constraints: time_prepared.linear_constraints.clone(),
        quadctx: Arc::new(QuadratureContext::new()),
    };

    let mut blocks = vec![
        build_time_blockspec(&time_prepared, &spec.time_block),
        build_mean_blockspec(&mean_design, spec.mean_offset.clone()),
    ];
    if latent_sd.is_none() {
        blocks.push(build_log_sigma_blockspec(
            LEARNABLE_LATENT_SD_SEED,
            mean_design.design.nrows(),
        ));
    }
    // Interval warm start (issue #1108). Interval-censored rows contribute the
    // NON-concave `ℓ = log[S(L) − S(R)]`; the coupled exact-joint inner Newton
    // diverges from the cold seed (β_time = 1e-4, σ = 0.5) — the failure surfaces
    // first as `fit_custom_family`'s outer ρ-seed startup validation rejecting
    // every seed (`solver_started = 0`). We warm-start from a LOG-CONCAVE
    // surrogate whose β/σ land in the interval basin, threaded via `initial_beta`
    // (consumed by every inner solve, including each ρ-seed validation fit).
    //
    // Surrogate = right-censored at the bracket LOWER bound `L`. Its survival
    // mass `S(L) = K_{0,B(L)}` is log-concave (PD Hessian) and — crucially —
    // its time-block design is the SAME fixed-knot I-spline basis the interval
    // fit uses, which is FULL RANK regardless of how heavily the inspection-grid
    // `L` values are TIED (the basis columns are functions of the frozen knots,
    // not of the observed time multiplicities). Unlike an exact-event surrogate
    // it imposes NO per-row `q̇(L) > 0` hazard-derivative feasibility condition
    // (which the tied/degenerate cold-start derivative design can violate), so it
    // is robust where exact-event-at-L is not. The warm σ then refines from the
    // bracket-width spread inside the (now in-basin) interval fit.
    //
    // Failure is NON-SILENT (#1108): a surrogate that errors or returns a
    // non-finite / all-zero degenerate β is surfaced as a hard error rather than
    // silently reverting to the diverging cold start (which masked the real
    // failure across several attempts). Only `initial_beta` is seeded; the EXACT
    // interval objective/gradient/Hessian are unchanged, so σ̂ is the true MLE.
    let has_interval_rows = spec
        .event_target
        .iter()
        .any(|&code| code == LATENT_SURVIVAL_EVENT_INTERVAL);
    if has_interval_rows {
        let warm_event_target = spec.event_target.mapv(|code| {
            if code == LATENT_SURVIVAL_EVENT_INTERVAL {
                0u8
            } else {
                code
            }
        });
        let mut warm_family = family.clone();
        warm_family.event_target = warm_event_target;
        // Right-censored-at-L ignores the interval upper bound `R`, so the
        // (unused) `q_right` channel cannot drift the fit; leaving the right
        // design/mass in place is harmless (no interval row remains to read it).
        let warm_fit = fit_custom_family_fixed_log_lambdas(
            &warm_family,
            &blocks,
            options,
            None,
            0,
            None,
            false,
        )
        .map_err(|e| {
            format!(
                "latent interval warm start: right-censored-at-L surrogate fit failed \
                 (so the interval fit cannot be safely warm-started; this surrogate is \
                 log-concave and should converge — investigate the surrogate, not the \
                 interval kernel): {e}"
            )
        })?;
        let warm_beta_usable = warm_fit
            .block_states
            .iter()
            .any(|s| s.beta.iter().all(|v| v.is_finite()) && s.beta.iter().any(|&v| v != 0.0));
        if !warm_beta_usable {
            return Err(
                "latent interval warm start: right-censored-at-L surrogate returned a \
                 degenerate (non-finite or all-zero) β across every block; the warm start \
                 cannot seed the interval fit. This indicates the surrogate's time-block \
                 design is rank-deficient or the inner solve stalled at the seed — \
                 investigate the surrogate before retrying the interval fit."
                    .to_string(),
            );
        }
        for (block, state) in blocks.iter_mut().zip(warm_fit.block_states.iter()) {
            if state.beta.iter().all(|v| v.is_finite()) {
                block.initial_beta = Some(state.beta.clone());
            }
        }
    }
    let fit = fit_custom_family(&family, &blocks, options).map_err(|e| e.to_string())?;
    let latent_sd = family.latent_sd(&fit.block_states)?;
    let baseline_offset_residuals = family.offset_channel_residuals(&fit.block_states)?;
    Ok(LatentSurvivalTermFitResult {
        fit,
        design: mean_design,
        resolvedspec,
        latent_sd,
        baseline_offset_residuals,
    })
}

pub fn fit_latent_binary_terms(
    data: ArrayView2<'_, f64>,
    spec: LatentBinaryTermSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentBinaryTermFitResult, String> {
    let latent_sd = validate_latent_binary_inputs(data, &spec, &frailty)?;
    let (_, hazard_loading) = fixed_latent_hazard_frailty(&frailty, "latent-binary")?;
    let mean_design =
        build_term_collection_design(data, &spec.meanspec).map_err(|e| e.to_string())?;
    let resolvedspec = freeze_term_collection_from_design(&spec.meanspec, &mean_design)
        .map_err(|e| e.to_string())?;
    let time_prepared = prepare_latent_time_block(&spec.time_block, None, spec.derivative_guard)?;

    let family = LatentBinaryFamily {
        event_target: spec.event_target.clone(),
        weights: spec.weights.clone(),
        latent_sd,
        hazard_loading,
        unloaded_mass_entry: spec.unloaded_mass_entry.clone(),
        unloaded_mass_exit: spec.unloaded_mass_exit.clone(),
        x_time_entry: time_prepared.design_entry.clone(),
        x_time_exit: time_prepared.design_exit.clone(),
        x_mean: mean_design.design.clone(),
        time_linear_constraints: time_prepared.linear_constraints.clone(),
        quadctx: Arc::new(QuadratureContext::new()),
    };

    let blocks = vec![
        build_time_blockspec(&time_prepared, &spec.time_block),
        build_mean_blockspec(&mean_design, spec.mean_offset.clone()),
    ];
    let fit = fit_custom_family(&family, &blocks, options).map_err(|e| e.to_string())?;
    let baseline_offset_residuals = family.offset_channel_residuals(&fit.block_states)?;
    Ok(LatentBinaryTermFitResult {
        fit,
        design: mean_design,
        resolvedspec,
        baseline_offset_residuals,
    })
}

/// Latent-survival adapter for the shared [`LatentIntervalModel`] driver.
///
/// Survival permits a learnable sigma (`sigma_fixed == None`) and carries the
/// per-row unloaded baseline hazard at exit (which feeds the exact-event
/// loaded/unloaded split); everything else is validated by the shared engine.
struct LatentSurvivalModel;

impl LatentIntervalModel for LatentSurvivalModel {
    fn context() -> &'static str {
        "latent-survival"
    }

    fn allows_interval() -> bool {
        true
    }

    fn frailty_policy(
        frailty: &FrailtySpec,
    ) -> Result<LatentFrailtyResolution, LatentSurvivalError> {
        match frailty {
            FrailtySpec::HazardMultiplier {
                sigma_fixed,
                loading,
            } => {
                if let Some(sigma) = sigma_fixed
                    && (!sigma.is_finite() || *sigma < 0.0)
                {
                    return Err(LatentSurvivalError::InvalidFrailty {
                        reason: format!(
                            "latent-survival requires a finite hazard-multiplier sigma >= 0, got {sigma}"
                        ),
                    });
                }
                Ok(LatentFrailtyResolution {
                    sigma: *sigma_fixed,
                    loading: *loading,
                })
            }
            FrailtySpec::GaussianShift { .. } => Err(LatentSurvivalError::InvalidFrailty {
                reason: "latent-survival requires HazardMultiplier frailty, not GaussianShift"
                    .to_string(),
            }),
            FrailtySpec::None => Err(LatentSurvivalError::InvalidFrailty {
                reason: "latent-survival requires a HazardMultiplier frailty specification"
                    .to_string(),
            }),
        }
    }
}

fn validate_latent_survival_inputs(
    data: ArrayView2<'_, f64>,
    spec: &LatentSurvivalTermSpec,
    frailty: &FrailtySpec,
) -> Result<Option<f64>, LatentSurvivalError> {
    let row = LatentIntervalRowView {
        frailty,
        age_entry: &spec.age_entry,
        age_exit: &spec.age_exit,
        event_target: &spec.event_target,
        weights: &spec.weights,
        unloaded_mass_entry: &spec.unloaded_mass_entry,
        unloaded_mass_exit: &spec.unloaded_mass_exit,
        unloaded_hazard_exit: Some(&spec.unloaded_hazard_exit),
        mean_offset: &spec.mean_offset,
        derivative_guard: spec.derivative_guard,
        time_block: &spec.time_block,
    };
    validate_latent_interval_inputs::<LatentSurvivalModel>(data, &row)
}

pub(crate) fn validate_unloaded_components_for_loading(
    context: &str,
    row_index: usize,
    loading: HazardLoading,
    unloaded_entry: f64,
    unloaded_exit: f64,
    unloaded_hazard: Option<f64>,
) -> Result<(), LatentSurvivalError> {
    match loading {
        HazardLoading::Full => {
            if unloaded_entry != 0.0
                || unloaded_exit != 0.0
                || unloaded_hazard.is_some_and(|hazard| hazard != 0.0)
            {
                return Err(LatentSurvivalError::InvalidDataset {
                    reason: format!(
                        "{context} row {} uses full hazard loading, so unloaded components must be exactly zero; got entry_mass={}, exit_mass={}, exit_hazard={}",
                        row_index + 1,
                        unloaded_entry,
                        unloaded_exit,
                        unloaded_hazard.unwrap_or(0.0)
                    ),
                });
            }
        }
        HazardLoading::LoadedVsUnloaded => {}
    }
    Ok(())
}

/// Latent-binary adapter for the shared [`LatentIntervalModel`] driver.
///
/// Binary never evaluates an exact event, so it requires a finite *fixed*
/// latent sigma (via [`fixed_latent_hazard_frailty_typed`]) and carries no
/// per-row unloaded hazard; every other invariant is validated by the shared
/// engine.
struct LatentBinaryModel;

impl LatentIntervalModel for LatentBinaryModel {
    fn context() -> &'static str {
        "latent-binary"
    }

    fn frailty_policy(
        frailty: &FrailtySpec,
    ) -> Result<LatentFrailtyResolution, LatentSurvivalError> {
        let (sigma, loading) = fixed_latent_hazard_frailty_typed(frailty, "latent-binary")?;
        Ok(LatentFrailtyResolution {
            sigma: Some(sigma),
            loading,
        })
    }
}

fn validate_latent_binary_inputs(
    data: ArrayView2<'_, f64>,
    spec: &LatentBinaryTermSpec,
    frailty: &FrailtySpec,
) -> Result<f64, LatentSurvivalError> {
    let row = LatentIntervalRowView {
        frailty,
        age_entry: &spec.age_entry,
        age_exit: &spec.age_exit,
        event_target: &spec.event_target,
        weights: &spec.weights,
        unloaded_mass_entry: &spec.unloaded_mass_entry,
        unloaded_mass_exit: &spec.unloaded_mass_exit,
        unloaded_hazard_exit: None,
        mean_offset: &spec.mean_offset,
        derivative_guard: spec.derivative_guard,
        time_block: &spec.time_block,
    };
    // The binary `frailty_policy` always yields `Some(sigma)` (it rejects the
    // learnable-scale case), so the shared driver's `Option<f64>` is `Some`
    // here; surface a structured error rather than unwrapping if that ever
    // changes.
    validate_latent_interval_inputs::<LatentBinaryModel>(data, &row)?.ok_or_else(|| {
        LatentSurvivalError::InvalidFrailty {
            reason: "latent-binary requires a fixed latent sigma".to_string(),
        }
    })
}

fn prepare_latent_time_block(
    input: &TimeBlockInput,
    design_right: Option<&DesignMatrix>,
    derivative_guard: f64,
) -> Result<PreparedLatentTimeBlock, LatentSurvivalError> {
    if !input.time_monotonicity.is_coordinate_cone() {
        return Err(LatentSurvivalError::UnsupportedConfiguration {
            reason: format!(
                "latent survival requires a coordinate-cone monotonicity strategy; got {:?}",
                input.time_monotonicity
            ),
        });
    }
    let design_entry = input
        .design_entry
        .try_to_dense_by_chunks("latent survival entry time design")?;
    let design_exit = input
        .design_exit
        .try_to_dense_by_chunks("latent survival exit time design")?;
    let design_derivative_exit = input
        .design_derivative_exit
        .try_to_dense_by_chunks("latent survival derivative time design")?;
    // The interval upper-bound design shares the time-block coefficients with
    // the exit design; when the data has no interval rows we reuse the exit
    // design so `q_right` stays well-defined (its likelihood contribution is
    // gated off for non-interval rows). When present it must match the exit
    // design's shape (same basis, evaluated at R).
    let design_right = match design_right {
        Some(matrix) => {
            let dense =
                matrix.try_to_dense_by_chunks("latent survival interval right time design")?;
            if dense.nrows() != design_exit.nrows() || dense.ncols() != design_exit.ncols() {
                return Err(LatentSurvivalError::InvalidDataset {
                    reason: format!(
                        "latent survival interval right time design must match exit design shape \
                         {:?}, got {:?}",
                        design_exit.dim(),
                        dense.dim()
                    ),
                });
            }
            dense
        }
        None => design_exit.clone(),
    };
    let linear_constraints = structural_time_coefficient_constraints(
        &input.design_derivative_exit,
        &input.derivative_offset_exit,
        derivative_guard,
    )?;
    let initial_beta = match linear_constraints.as_ref() {
        // `project_onto_linear_constraints` validates that any supplied
        // `initial_beta` matches `design_exit.ncols()`; surface a mismatch as a
        // structured error rather than letting an ndarray broadcast panic
        // (issue #374).
        Some(constraints) => Some(project_onto_linear_constraints(
            design_exit.ncols(),
            constraints,
            input.initial_beta.as_ref(),
        )?),
        None => None,
    };
    Ok(PreparedLatentTimeBlock {
        design_entry,
        design_exit,
        design_derivative_exit,
        design_right,
        linear_constraints,
        penalties: input.penalties.clone(),
        initial_beta,
    })
}

fn stack_rows(blocks: &[&Array2<f64>]) -> Array2<f64> {
    let ncols = blocks.first().map_or(0, |m| m.ncols());
    let nrows = blocks.iter().map(|m| m.nrows()).sum();
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    let mut row = 0usize;
    for block in blocks {
        let end = row + block.nrows();
        out.slice_mut(s![row..end, ..]).assign(block);
        row = end;
    }
    out
}

fn build_time_blockspec(
    prepared: &PreparedLatentTimeBlock,
    input: &TimeBlockInput,
) -> ParameterBlockSpec {
    // The solver produces a `3·n`-long time `eta` (the `[entry; exit; deriv]`
    // channel stack that `split_time_eta` slices). That stacked operator is
    // the eta-producing matrix and so belongs in `stacked_design`, paired with
    // the matching `3·n`-row stacked offset. The audit / shape-policy invariant
    // `design.nrows() == n_obs` is satisfied by exposing the single-channel
    // n-row exit design as `design`; the audit never inspects `stacked_design`.
    //
    // This mirrors the survival location-scale fix for the same #326 class
    // (`survival_location_scale.rs`): the previous code put the `3·n`-row
    // stack in `design`, which tripped the flat identifiability audit's
    // row-equality invariant (`block 1 (mean) has n rows, expected 3n`).
    let stacked_design = stack_rows(&[
        &prepared.design_entry,
        &prepared.design_exit,
        &prepared.design_derivative_exit,
    ]);
    let stacked_offset = crate::linalg::utils::stack_offsets(&[
        &input.offset_entry,
        &input.offset_exit,
        &input.derivative_offset_exit,
    ]);
    ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
            prepared.design_exit.clone(),
        ))),
        offset: input.offset_exit.clone(),
        penalties: prepared
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: input.nullspace_dims.clone(),
        initial_log_lambdas: input
            .initial_log_lambdas
            .clone()
            .unwrap_or_else(|| Array1::zeros(prepared.penalties.len())),
        initial_beta: prepared.initial_beta.clone(),
        // Canonical-gauge ownership for the latent-survival joint design: the
        // time-transform block carries the structural monotone baseline that
        // anchors the parameterisation, so it owns any shared constant
        // direction (strictly higher than `mean`/`log_sigma` at 100). This
        // matches the survival location-scale gauge contract (time highest).
        gauge_priority: 200,
        jacobian_callback: None,
        stacked_design: Some(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
            stacked_design,
        )))),
        stacked_offset: Some(stacked_offset),
    }
}

fn build_mean_blockspec(design: &TermCollectionDesign, offset: Array1<f64>) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "mean".to_string(),
        design: design.design.clone(),
        offset,
        penalties: design.penalties_as_penalty_matrix(),
        nullspace_dims: design.nullspace_dims.clone(),
        initial_log_lambdas: Array1::zeros(design.penalties.len()),
        initial_beta: None,
        // Strictly below `time_transform` (200) so any constant direction
        // shared between the monotone time baseline and the mean intercept is
        // deterministically attributable to the lower-priority `mean` block by
        // the canonical-gauge RRQR (the descending-priority contract used by
        // survival location-scale; #366/#556 gauge story).
        gauge_priority: 150,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Starting latent-frailty standard deviation when `sigma` is learnable
/// (`sigma_fixed == None`). The log-sigma block is seeded at `log(0.5)` so the
/// optimizer begins from a moderate, well-conditioned dispersion (σ = 0.5,
/// neither a near-degenerate σ → 0 that flattens the frailty integral nor a
/// large σ that makes the Gauss-Hermite quadrature heavy-tailed) and then
/// learns the data's actual scale. Only an initial value, not a constraint.
const LEARNABLE_LATENT_SD_SEED: f64 = 0.5;

fn build_log_sigma_blockspec(initial_sigma: f64, n_obs: usize) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "log_sigma".to_string(),
        // The frailty/dispersion scale is a single GLOBAL hyperparameter (one free
        // coefficient), but the identifiability audit — and the canonical-row
        // architecture generally — require every block's effective Jacobian to carry
        // `n_obs` rows. A global scalar is realised the same way the survival
        // location-scale `log_sigma` block is (see `BinomialLocationScaleFamily`): an
        // `n_obs × 1` constant column of ones, so `eta = design · β` is the same scalar
        // broadcast to every observation. This keeps it a single free parameter while
        // exposing the `n_obs`-row shape the audit checks, and `latent_sd` reads
        // `eta[0]` — identical across rows by construction.
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(Array2::from_elem(
            (n_obs, 1),
            1.0,
        )))),
        offset: Array1::zeros(n_obs),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::from_elem(
            1,
            exp_sigma_eta_for_sigma_scalar(initial_sigma),
        )),
        // Lowest of the three (time=200, mean=150): the learnable-scale channel
        // yields any shared constant to the location blocks.
        gauge_priority: 120,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

const LATENT_SURVIVAL_PRIMARY_Q_ENTRY: usize = 0;
const LATENT_SURVIVAL_PRIMARY_Q_EXIT: usize = 1;
const LATENT_SURVIVAL_PRIMARY_QDOT_EXIT: usize = 2;
// Interval-censored right boundary R: q_right = log B(R) shares the time-block
// coefficients with q_exit (same monotone transform, different time point), so
// it is a fourth linear functional of the time block, NOT an independent eta
// channel. It sits before `mu`/`log_sigma` so the "trailing optional log_sigma"
// invariant used by `active_primary` (= `LATENT_SURVIVAL_PRIMARY_LOG_SIGMA`)
// keeps q_right always active.
const LATENT_SURVIVAL_PRIMARY_Q_RIGHT: usize = 3;
const LATENT_SURVIVAL_PRIMARY_MU: usize = 4;
const LATENT_SURVIVAL_PRIMARY_LOG_SIGMA: usize = 5;
const LATENT_SURVIVAL_PRIMARY_DIM: usize = 6;

use crate::families::jet_partitions::MultiDirJet as LatentMultiDirJet;

/// Derivatives of `log(x)` through 4th order.
///
/// # Contract
///
/// `x` must be strictly positive. This function does NOT clamp: a previous
/// version replaced `x` by `x.max(1e-300)`, which fabricated enormous finite
/// derivatives (`1/1e-300` etc.) that are the derivatives of neither `log(x)`
/// nor `log(max(x, floor))` and would silently mask an upstream domain
/// failure. Both callers guarantee `x > 0`: one composes at the literal `1.0`
/// (the normalised log-sum base); the other passes `base`, which is gated by
/// an explicit `base.is_finite() && base > 0.0` check immediately upstream. A
/// non-positive `x` therefore never reaches here on any supported path; were
/// one to, the function returns the honest IEEE result (`-inf`/`NaN`) —
/// identical in debug and release — rather than a finite fabrication. For all
/// valid `x > 0` the output is bit-identical to the previous clamped version.
#[inline]
fn latent_unary_derivatives_log(x: f64) -> [f64; 5] {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    [x.ln(), 1.0 / x, -1.0 / x2, 2.0 / x3, -6.0 / x4]
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryTerm {
    coeff: f64,
    q_exp: usize,
    qdot_power: usize,
    tau_exp: usize,
    k: usize,
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryDirection {
    dq: f64,
    dqd: f64,
    dmu: f64,
    dtau: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentSurvivalPrimaryDirection {
    dq_entry: f64,
    dq_exit: f64,
    dqdot_exit: f64,
    dq_right: f64,
    dmu: f64,
    dlog_sigma: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryState {
    q: f64,
    qdot: f64,
    mu: f64,
    sigma: f64,
    log_sigma_factor: f64,
}

fn latent_kernel_accumulate_term(
    terms: &mut BTreeMap<(usize, usize, usize, usize), f64>,
    term: LatentKernelPrimaryTerm,
    scale: f64,
) {
    if scale == 0.0 || term.coeff == 0.0 {
        return;
    }
    *terms
        .entry((term.q_exp, term.qdot_power, term.tau_exp, term.k))
        .or_insert(0.0) += scale * term.coeff;
}

fn latent_kernel_differentiate_terms(
    terms: &[LatentKernelPrimaryTerm],
    dir: LatentKernelPrimaryDirection,
) -> Vec<LatentKernelPrimaryTerm> {
    let mut out = BTreeMap::<(usize, usize, usize, usize), f64>::new();
    for term in terms {
        if dir.dq != 0.0 {
            if term.q_exp > 0 {
                latent_kernel_accumulate_term(&mut out, *term, dir.dq * term.q_exp as f64);
            }
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dq,
            );
        }
        if dir.dmu != 0.0 {
            if term.k > 0 {
                latent_kernel_accumulate_term(&mut out, *term, dir.dmu * term.k as f64);
            }
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dmu,
            );
        }
        if dir.dtau != 0.0 {
            if term.tau_exp > 0 {
                latent_kernel_accumulate_term(&mut out, *term, dir.dtau * term.tau_exp as f64);
            }
            let kf = term.k as f64;
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    tau_exp: term.tau_exp + 2,
                    ..*term
                },
                dir.dtau * kf * kf,
            );
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    tau_exp: term.tau_exp + 2,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dtau * (2.0 * kf + 1.0),
            );
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 2,
                    tau_exp: term.tau_exp + 2,
                    k: term.k + 2,
                    ..*term
                },
                dir.dtau,
            );
        }
        if dir.dqd != 0.0 && term.qdot_power > 0 {
            latent_kernel_accumulate_term(
                &mut out,
                LatentKernelPrimaryTerm {
                    qdot_power: term.qdot_power - 1,
                    ..*term
                },
                dir.dqd * term.qdot_power as f64,
            );
        }
    }
    out.into_iter()
        .filter_map(|((q_exp, qdot_power, tau_exp, k), coeff)| {
            (coeff != 0.0).then_some(LatentKernelPrimaryTerm {
                coeff,
                q_exp,
                qdot_power,
                tau_exp,
                k,
            })
        })
        .collect()
}

fn latent_kernel_term_lists_for_directions(
    base_terms: &[LatentKernelPrimaryTerm],
    directions: &[LatentKernelPrimaryDirection],
) -> Vec<Vec<LatentKernelPrimaryTerm>> {
    fn build_mask(
        mask: usize,
        base_terms: &[LatentKernelPrimaryTerm],
        directions: &[LatentKernelPrimaryDirection],
        cache: &mut [Option<Vec<LatentKernelPrimaryTerm>>],
    ) -> Vec<LatentKernelPrimaryTerm> {
        if let Some(existing) = &cache[mask] {
            return existing.clone();
        }
        let built = if mask == 0 {
            base_terms.to_vec()
        } else {
            let bit = 1usize << mask.trailing_zeros();
            let prev = build_mask(mask ^ bit, base_terms, directions, cache);
            latent_kernel_differentiate_terms(&prev, directions[bit.trailing_zeros() as usize])
        };
        cache[mask] = Some(built.clone());
        built
    }

    let mut cache = vec![None; 1usize << directions.len()];
    (0..cache.len())
        .map(|mask| build_mask(mask, base_terms, directions, &mut cache))
        .collect()
}

fn latent_kernel_sum_log_jet(
    quadctx: &QuadratureContext,
    base_terms: &[LatentKernelPrimaryTerm],
    state: LatentKernelPrimaryState,
    directions: &[LatentKernelPrimaryDirection],
    context: &str,
) -> Result<LatentMultiDirJet, LatentSurvivalError> {
    let term_lists = latent_kernel_term_lists_for_directions(base_terms, directions);
    let max_k = term_lists
        .iter()
        .flat_map(|terms| terms.iter().map(|term| term.k))
        .max()
        .unwrap_or(0);
    let bundle =
        log_kernel_bundle(quadctx, state.q.exp(), state.mu, state.sigma, max_k).map_err(|e| {
            LatentSurvivalError::NumericalFailure {
                reason: format!("{context} kernel evaluation failed: {e}"),
            }
        })?;

    let evaluate_terms =
        |terms: &[LatentKernelPrimaryTerm]| -> Result<(f64, f64), LatentSurvivalError> {
            let mut log_mags = Vec::new();
            let mut signs = Vec::new();
            for term in terms {
                if term.coeff == 0.0 {
                    continue;
                }
                if term.qdot_power > 0 && !(state.qdot.is_finite() && state.qdot > 0.0) {
                    return Err(LatentSurvivalError::NumericalFailure {
                        reason: format!(
                            "{context} requires positive finite qdot for exact-event directional terms, got {}",
                            state.qdot
                        ),
                    });
                }
                let log_qdot = if term.qdot_power > 0 {
                    state.qdot.ln()
                } else {
                    0.0
                };
                let log_mag = term.coeff.abs().ln()
                    + term.q_exp as f64 * state.q
                    + term.tau_exp as f64 * state.log_sigma_factor
                    + term.qdot_power as f64 * log_qdot
                    + bundle.get(term.k);
                log_mags.push(log_mag);
                signs.push(term.coeff.signum());
            }
            if log_mags.is_empty() {
                return Ok((f64::NEG_INFINITY, 0.0));
            }
            Ok(signed_log_sum_exp(&log_mags, &signs))
        };

    let (base_log_sum, base_sign) = evaluate_terms(&term_lists[0])?;
    if !(base_log_sum.is_finite() && base_sign > 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!("{context} produced a non-positive signed kernel sum"),
        });
    }

    let mut normalized = LatentMultiDirJet::constant(directions.len(), 1.0);
    for mask in 1..term_lists.len() {
        let (log_abs, sign) = evaluate_terms(&term_lists[mask])?;
        normalized.coeffs[mask] = if !log_abs.is_finite() || sign == 0.0 {
            0.0
        } else {
            sign * (log_abs - base_log_sum).exp()
        };
    }

    let mut out = normalized.compose_unary(latent_unary_derivatives_log(1.0));
    out.coeffs[0] += base_log_sum;
    Ok(out)
}

fn latent_survival_basis_direction(primary_idx: usize) -> LatentSurvivalPrimaryDirection {
    match primary_idx {
        LATENT_SURVIVAL_PRIMARY_Q_ENTRY => LatentSurvivalPrimaryDirection {
            dq_entry: 1.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_Q_EXIT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 1.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_QDOT_EXIT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 1.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_Q_RIGHT => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 1.0,
            dmu: 0.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_MU => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 1.0,
            dlog_sigma: 0.0,
        },
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA => LatentSurvivalPrimaryDirection {
            dq_entry: 0.0,
            dq_exit: 0.0,
            dqdot_exit: 0.0,
            dq_right: 0.0,
            dmu: 0.0,
            dlog_sigma: 1.0,
        },
        // SAFETY: latent survival has exactly `LATENT_SURVIVAL_PRIMARY_DIM`
        // (= 5) primary directions, indexed 0..=4 via the module-private
        // `LATENT_SURVIVAL_PRIMARY_*` constants. All five are matched
        // above, so this wildcard fires only on an out-of-range index,
        // which the internal iteration bounds (`0..LATENT_SURVIVAL_PRIMARY_DIM`)
        // make unreachable.
        // SAFETY: primary_idx is bounded by LATENT_SURVIVAL_PRIMARY_DIM at every internal call site.
        _ => std::panic::panic_any(format!(
            "latent survival primary index out of bounds: primary_idx={primary_idx}, primary_dim={LATENT_SURVIVAL_PRIMARY_DIM}"
        )),
    }
}

fn latent_survival_map_entry_direction(
    direction: LatentSurvivalPrimaryDirection,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_entry,
        dqd: 0.0,
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

fn latent_survival_map_exit_direction(
    direction: LatentSurvivalPrimaryDirection,
    event_type: LatentSurvivalEventType,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_exit,
        dqd: if matches!(event_type, LatentSurvivalEventType::ExactEvent) {
            direction.dqdot_exit
        } else {
            0.0
        },
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

/// Direction map for the interval-censored LEFT boundary state (mass `M_L =
/// exp(q_exit)`). The left boundary tracks the same `q_exit` time functional as
/// right-censoring (no hazard-derivative channel), plus the shared `mu`/`sigma`.
fn latent_survival_map_left_direction(
    direction: LatentSurvivalPrimaryDirection,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_exit,
        dqd: 0.0,
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

/// Direction map for the interval-censored RIGHT boundary state (mass `M_R =
/// exp(q_right)`). The right boundary tracks the dedicated `q_right` functional
/// (which shares the time-block coefficients with `q_exit` but is evaluated at
/// the interval upper bound `R`), plus the shared `mu`/`sigma`.
fn latent_survival_map_right_direction(
    direction: LatentSurvivalPrimaryDirection,
) -> LatentKernelPrimaryDirection {
    LatentKernelPrimaryDirection {
        dq: direction.dq_right,
        dqd: 0.0,
        dmu: direction.dmu,
        dtau: direction.dlog_sigma,
    }
}

fn latent_survival_row_primary_log_jet(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    q_right: f64,
    mu: f64,
    sigma: f64,
    log_sigma_factor: f64,
    directions: &[LatentSurvivalPrimaryDirection],
) -> Result<LatentMultiDirJet, String> {
    let entry_state = LatentKernelPrimaryState {
        q: q_entry,
        qdot: 1.0,
        mu,
        sigma,
        log_sigma_factor,
    };
    let entry_directions = directions
        .iter()
        .copied()
        .map(latent_survival_map_entry_direction)
        .collect::<Vec<_>>();

    let denominator = latent_kernel_sum_log_jet(
        quadctx,
        &[LatentKernelPrimaryTerm {
            coeff: 1.0,
            q_exp: 0,
            qdot_power: 0,
            tau_exp: 0,
            k: 0,
        }],
        entry_state,
        &entry_directions,
        "latent survival denominator",
    )?;

    // The numerator for right-censoring / exact events is a single-state log-sum
    // kernel at the exit mass. Interval censoring is the difference of two
    // single-state kernels at DIFFERENT masses (L at `q_exit`, R at `q_right`),
    // so it is assembled by `latent_survival_interval_numerator_log_jet` below.
    let numerator = match row.event_type {
        LatentSurvivalEventType::RightCensored | LatentSurvivalEventType::ExactEvent => {
            let exit_state = LatentKernelPrimaryState {
                q: q_exit,
                qdot: qdot_exit,
                mu,
                sigma,
                log_sigma_factor,
            };
            let exit_directions = directions
                .iter()
                .copied()
                .map(|dir| latent_survival_map_exit_direction(dir, row.event_type))
                .collect::<Vec<_>>();
            let numerator_terms = match row.event_type {
                LatentSurvivalEventType::RightCensored => vec![LatentKernelPrimaryTerm {
                    coeff: 1.0,
                    q_exp: 0,
                    qdot_power: 0,
                    tau_exp: 0,
                    k: 0,
                }],
                LatentSurvivalEventType::ExactEvent => {
                    let mut terms = Vec::new();
                    if row.hazard_unloaded > 0.0 {
                        terms.push(LatentKernelPrimaryTerm {
                            coeff: row.hazard_unloaded,
                            q_exp: 0,
                            qdot_power: 0,
                            tau_exp: 0,
                            k: 0,
                        });
                    }
                    terms.push(LatentKernelPrimaryTerm {
                        coeff: 1.0,
                        q_exp: 1,
                        qdot_power: 1,
                        tau_exp: 0,
                        k: 1,
                    });
                    terms
                }
                LatentSurvivalEventType::IntervalCensored => {
                    // Interval-censored rows are routed to the dedicated two-state
                    // numerator branch (the outer match arm below), so this inner
                    // arm is not reached; a clean error rather than a panic guards
                    // against a future routing change.
                    return Err(
                        "interval-censored row reached the single-state numerator branch; \
                         it must take the dedicated two-state branch"
                            .to_string(),
                    );
                }
            };
            latent_kernel_sum_log_jet(
                quadctx,
                &numerator_terms,
                exit_state,
                &exit_directions,
                "latent survival numerator",
            )?
        }
        LatentSurvivalEventType::IntervalCensored => latent_survival_interval_numerator_log_jet(
            quadctx,
            row,
            q_exit,
            q_right,
            mu,
            sigma,
            log_sigma_factor,
            directions,
        )?,
    };

    let mut total = numerator.add(&denominator.scale(-1.0));
    // For interval rows the unloaded exit mass is folded into the per-boundary
    // coefficients `exp(-mass_unloaded_{left,right})` inside the two-state
    // numerator, so only the (constant) unloaded-entry term remains here; for
    // right-censoring / exact events the exit/entry unloaded masses are an
    // additive constant on the log-likelihood.
    match row.event_type {
        LatentSurvivalEventType::IntervalCensored => {
            total.coeffs[0] += row.mass_unloaded_entry;
        }
        _ => {
            total.coeffs[0] += -row.mass_unloaded_exit + row.mass_unloaded_entry;
        }
    }
    Ok(total)
}

/// Interval-censored numerator jet `log[ c_L·K_{0,M_L} − c_R·K_{0,M_R} ]` where
/// `M_L = exp(q_exit)`, `M_R = exp(q_right)`, `c_L = exp(-mass_unloaded_left)`
/// and `c_R = exp(-mass_unloaded_right)`.
///
/// This is the dynamic-time analogue of the static
/// [`LatentSurvivalRowJet::interval_censored`] kernel: the interval likelihood
/// is the difference of two BOUNDARY survival masses, each a single-state
/// order-0 kernel, but at two DISTINCT cumulative masses. Because the two
/// boundaries respond to different time functionals (`q_exit` vs `q_right`) we
/// cannot fold them into one `latent_kernel_sum_log_jet` state. Instead we:
///   1. build each boundary's `log K_{0,M}` jet at its own state, with its own
///      direction map (left → `dq_exit`, right → `dq_right`; both share
///      `mu`/`sigma`),
///   2. lift each to the LINEAR domain via `exp` (a unary composition whose five
///      derivatives at value `v` are all `exp(v)`), scaled by its coefficient
///      `c_L` (resp. `−c_R`),
///   3. add the two linear-domain jets, and
///   4. drop back to the log domain via the same `log` unary composition the
///      single-state path uses.
/// Every multi-direction coefficient (value, score, neg-Hessian, 3rd, 4th)
/// follows by the Faà-di-Bruno composition already implemented in
/// `MultiDirJet::compose_unary`, so the derivative reductions are consistent
/// with the exact-event/right-censored branches by construction.
fn latent_survival_interval_numerator_log_jet(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_exit: f64,
    q_right: f64,
    mu: f64,
    sigma: f64,
    log_sigma_factor: f64,
    directions: &[LatentSurvivalPrimaryDirection],
) -> Result<LatentMultiDirJet, String> {
    let single_k0 = [LatentKernelPrimaryTerm {
        coeff: 1.0,
        q_exp: 0,
        qdot_power: 0,
        tau_exp: 0,
        k: 0,
    }];

    let left_state = LatentKernelPrimaryState {
        q: q_exit,
        qdot: 1.0,
        mu,
        sigma,
        log_sigma_factor,
    };
    let right_state = LatentKernelPrimaryState {
        q: q_right,
        qdot: 1.0,
        mu,
        sigma,
        log_sigma_factor,
    };
    let left_directions = directions
        .iter()
        .copied()
        .map(latent_survival_map_left_direction)
        .collect::<Vec<_>>();
    let right_directions = directions
        .iter()
        .copied()
        .map(latent_survival_map_right_direction)
        .collect::<Vec<_>>();

    let log_left = latent_kernel_sum_log_jet(
        quadctx,
        &single_k0,
        left_state,
        &left_directions,
        "latent survival interval left boundary",
    )?;
    let log_right = latent_kernel_sum_log_jet(
        quadctx,
        &single_k0,
        right_state,
        &right_directions,
        "latent survival interval right boundary",
    )?;

    // Lift each boundary's log-kernel jet to the linear domain and scale by the
    // unloaded-mass prefactor. exp''''(v) = exp(v) for all orders, so the unary
    // derivative tower is `[exp(v); exp(v); exp(v); exp(v); exp(v)]`.
    let c_left = (-row.mass_unloaded_left).exp();
    let c_right = (-row.mass_unloaded_right).exp();
    let exp_left_value = log_left.coeff(0).exp();
    let exp_right_value = log_right.coeff(0).exp();
    let linear_left = log_left.compose_unary([exp_left_value; 5]).scale(c_left);
    let linear_right = log_right.compose_unary([exp_right_value; 5]).scale(c_right);

    let linear_numerator = linear_left.add(&linear_right.scale(-1.0));
    let base = linear_numerator.coeff(0);
    if !(base.is_finite() && base > 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent survival interval numerator must be a positive survival-mass difference, \
                 got c_L*K0(M_L) - c_R*K0(M_R) = {base}; require M_L < M_R (i.e. L < R)"
            ),
        }
        .into());
    }
    // Drop back to the log domain. `latent_unary_derivatives_log(base)` is the
    // unary derivative tower of `ln` at the positive base value, so the composed
    // value channel is `ln(base)` and the higher coefficients are the
    // log-of-a-difference score / curvature, consistent with the single-state
    // log-sum path (which composes `ln` at its normalised base of 1).
    Ok(linear_numerator.compose_unary(latent_unary_derivatives_log(base)))
}

fn latent_survival_row_primary_gradient_hessian(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    q_right: f64,
    mu: f64,
    sigma: f64,
    include_log_sigma: bool,
) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
    let log_sigma_factor = if sigma > 0.0 { sigma.ln() } else { 0.0 };
    let mut gradient = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
    let mut neg_hessian =
        Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
    let active_primary = if include_log_sigma {
        LATENT_SURVIVAL_PRIMARY_DIM
    } else {
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
    };
    let log_lik = latent_survival_row_primary_log_jet(
        quadctx,
        row,
        q_entry,
        q_exit,
        qdot_exit,
        q_right,
        mu,
        sigma,
        log_sigma_factor,
        &[],
    )?
    .coeff(0);
    for a in 0..active_primary {
        let dir_a = latent_survival_basis_direction(a);
        gradient[a] = latent_survival_row_primary_log_jet(
            quadctx,
            row,
            q_entry,
            q_exit,
            qdot_exit,
            q_right,
            mu,
            sigma,
            log_sigma_factor,
            &[dir_a],
        )?
        .coeff(1);
        for b in a..active_primary {
            let coeff = latent_survival_row_primary_log_jet(
                quadctx,
                row,
                q_entry,
                q_exit,
                qdot_exit,
                q_right,
                mu,
                sigma,
                log_sigma_factor,
                &[dir_a, latent_survival_basis_direction(b)],
            )?
            .coeff(3);
            neg_hessian[[a, b]] = -coeff;
            neg_hessian[[b, a]] = -coeff;
        }
    }
    Ok((log_lik, gradient, neg_hessian))
}

fn latent_survival_row_primary_third_contracted(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    q_right: f64,
    mu: f64,
    sigma: f64,
    direction: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, String> {
    let log_sigma_factor = if sigma > 0.0 { sigma.ln() } else { 0.0 };
    let active_primary = if include_log_sigma {
        LATENT_SURVIVAL_PRIMARY_DIM
    } else {
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
    };
    let dir = LatentSurvivalPrimaryDirection {
        dq_entry: direction[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
        dq_exit: direction[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
        dqdot_exit: direction[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
        dq_right: direction[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
        dmu: direction[LATENT_SURVIVAL_PRIMARY_MU],
        dlog_sigma: direction[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA],
    };
    let mut out = Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
    for a in 0..active_primary {
        let dir_a = latent_survival_basis_direction(a);
        for b in a..active_primary {
            let coeff = latent_survival_row_primary_log_jet(
                quadctx,
                row,
                q_entry,
                q_exit,
                qdot_exit,
                q_right,
                mu,
                sigma,
                log_sigma_factor,
                &[dir_a, latent_survival_basis_direction(b), dir],
            )?
            .coeff(7);
            out[[a, b]] = -coeff;
            out[[b, a]] = -coeff;
        }
    }
    Ok(out)
}

fn latent_survival_row_primary_fourth_contracted(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    q_right: f64,
    mu: f64,
    sigma: f64,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, String> {
    let log_sigma_factor = if sigma > 0.0 { sigma.ln() } else { 0.0 };
    let active_primary = if include_log_sigma {
        LATENT_SURVIVAL_PRIMARY_DIM
    } else {
        LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
    };
    let dir_u = LatentSurvivalPrimaryDirection {
        dq_entry: direction_u[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
        dq_exit: direction_u[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
        dqdot_exit: direction_u[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
        dq_right: direction_u[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
        dmu: direction_u[LATENT_SURVIVAL_PRIMARY_MU],
        dlog_sigma: direction_u[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA],
    };
    let dir_v = LatentSurvivalPrimaryDirection {
        dq_entry: direction_v[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
        dq_exit: direction_v[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
        dqdot_exit: direction_v[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
        dq_right: direction_v[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
        dmu: direction_v[LATENT_SURVIVAL_PRIMARY_MU],
        dlog_sigma: direction_v[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA],
    };
    let mut out = Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
    for a in 0..active_primary {
        let dir_a = latent_survival_basis_direction(a);
        for b in a..active_primary {
            let coeff = latent_survival_row_primary_log_jet(
                quadctx,
                row,
                q_entry,
                q_exit,
                qdot_exit,
                q_right,
                mu,
                sigma,
                log_sigma_factor,
                &[dir_a, latent_survival_basis_direction(b), dir_u, dir_v],
            )?
            .coeff(15);
            out[[a, b]] = -coeff;
            out[[b, a]] = -coeff;
        }
    }
    Ok(out)
}

#[derive(Clone)]
struct LatentSurvivalJointSlices {
    time: std::ops::Range<usize>,
    mean: std::ops::Range<usize>,
    log_sigma: Option<std::ops::Range<usize>>,
    total: usize,
}

#[derive(Clone)]
struct LatentSurvivalJointGradientAccum {
    ll: f64,
    gradient: Array1<f64>,
}

#[derive(Clone)]
struct LatentSurvivalJointDenseAccum {
    ll: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

#[derive(Clone)]
struct LatentSurvivalDenseHessianAccum {
    hessian: Array2<f64>,
}

/// Process latent-survival rows in fixed contiguous chunks, using one
/// accumulator per rayon task and reducing those accumulators in chunk-index
/// order so gradient/Hessian assembly stays deterministic across runs.
fn deterministic_latent_survival_row_reduction<Acc, Init, Process, Combine>(
    n_rows: usize,
    init: Init,
    process_row: Process,
    mut combine: Combine,
) -> Result<Acc, String>
where
    Acc: Send,
    Init: Fn() -> Acc + Sync,
    Process: Fn(usize, &mut Acc) -> Result<(), String> + Sync,
    Combine: FnMut(&mut Acc, Acc),
{
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    const TARGET_CHUNK_COUNT: usize = 32;
    if n_rows == 0 {
        return Ok(init());
    }
    let chunk_size = n_rows.div_ceil(TARGET_CHUNK_COUNT).max(1);
    let n_chunks = n_rows.div_ceil(chunk_size);
    let chunk_accumulators: Vec<Acc> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| -> Result<Acc, String> {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n_rows);
            let mut acc = init();
            for row_idx in start..end {
                process_row(row_idx, &mut acc)?;
            }
            Ok(acc)
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut total = init();
    for acc in chunk_accumulators {
        combine(&mut total, acc);
    }
    Ok(total)
}

impl LatentSurvivalFamily {
    /// Assemble the per-row [`LatentSurvivalRow`] for `row_idx` from the family's
    /// unloaded-mass/hazard fields and the supplied per-row time quantiles.
    ///
    /// Shared by every per-row reduction (log-likelihood, gradient, Hessian,
    /// directional third derivatives): each previously inlined an identical
    /// `event_type` lookup followed by the same 12-argument
    /// `build_latent_survival_row` call. Behavior is unchanged.
    fn build_row_at(
        &self,
        row_idx: usize,
        q_entry: f64,
        q_exit: f64,
        qdot_exit: f64,
        q_right: f64,
    ) -> Result<LatentSurvivalRow, LatentSurvivalError> {
        let event_type = latent_survival_event_type_for(self.event_target[row_idx]);
        build_latent_survival_row(
            row_idx,
            self.hazard_loading,
            event_type,
            q_entry,
            q_exit,
            qdot_exit,
            q_right,
            self.unloaded_mass_entry[row_idx],
            self.unloaded_mass_exit[row_idx],
            self.unloaded_mass_right[row_idx],
            self.unloaded_hazard_exit[row_idx],
        )
    }

    fn joint_slices(&self) -> LatentSurvivalJointSlices {
        let p_time = self.x_time_exit.ncols();
        let p_mean = self.x_mean.ncols();
        let time = 0..p_time;
        let mean = p_time..p_time + p_mean;
        let log_sigma = self
            .latent_sd_fixed
            .is_none()
            .then_some((p_time + p_mean)..(p_time + p_mean + 1));
        LatentSurvivalJointSlices {
            total: log_sigma
                .as_ref()
                .map_or(p_time + p_mean, |range| range.end),
            time,
            mean,
            log_sigma,
        }
    }

    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        out[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] = self.x_time_entry.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_Q_EXIT] = self.x_time_exit.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT] = self.x_time_derivative_exit.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_Q_RIGHT] = self.x_time_right.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_MU] = self
            .x_mean
            .dot_row_view(row, d_beta_flat.slice(s![slices.mean.clone()]));
        if let Some(range) = &slices.log_sigma {
            out[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA] = d_beta_flat[range.start];
        }
        out
    }

    fn joint_block_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let slices = self.joint_slices();
        let mut ranges = vec![slices.time.clone(), slices.mean.clone()];
        if let Some(log_sigma) = slices.log_sigma {
            ranges.push(log_sigma);
        }
        ranges
    }

    fn add_pullback_primary_gradient(
        &self,
        target: &mut Array1<f64>,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        primary_gradient: &Array1<f64>,
        weight: f64,
    ) -> Result<(), String> {
        for (primary_idx, time_vec) in [
            (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
            (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
            (
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                self.x_time_derivative_exit.row(row),
            ),
            (LATENT_SURVIVAL_PRIMARY_Q_RIGHT, self.x_time_right.row(row)),
        ] {
            let scale = weight * primary_gradient[primary_idx];
            if scale == 0.0 {
                continue;
            }
            for i in 0..time_vec.len() {
                let xi = time_vec[i];
                if xi != 0.0 {
                    target[slices.time.start + i] += scale * xi;
                }
            }
        }

        let mean_scale = weight * primary_gradient[LATENT_SURVIVAL_PRIMARY_MU];
        if mean_scale != 0.0 {
            self.x_mean
                .axpy_row_into(
                    row,
                    mean_scale,
                    &mut target.slice_mut(s![slices.mean.clone()]),
                )
                .map_err(|error| {
                    format!(
                        "latent survival mean gradient pullback dimension mismatch: row={row}, mean_slice={:?}, target_len={}, x_mean_cols={}, error={error}",
                        slices.mean,
                        target.len(),
                        self.x_mean.ncols()
                    )
                })?;
        }

        if let Some(log_sigma) = &slices.log_sigma {
            target[log_sigma.start] += weight * primary_gradient[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA];
        }
        Ok(())
    }

    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        primary_hessian: &Array2<f64>,
    ) -> Result<(), String> {
        let time_weights = [
            primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
            ]],
            primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
            ]],
            primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
            ]],
            primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
            ]],
        ];
        let time_cross_weights = [
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                &self.x_time_entry,
                &self.x_time_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                &self.x_time_entry,
                &self.x_time_derivative_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                &self.x_time_exit,
                &self.x_time_derivative_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                &self.x_time_entry,
                &self.x_time_right,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                &self.x_time_exit,
                &self.x_time_right,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                &self.x_time_derivative_exit,
                &self.x_time_right,
            ),
        ];
        {
            let time_target = &mut target.slice_mut(s![slices.time.clone(), slices.time.clone()]);
            dense_outer_accumulate(time_target, time_weights[0], self.x_time_entry.row(row));
            dense_outer_accumulate(time_target, time_weights[1], self.x_time_exit.row(row));
            dense_outer_accumulate(
                time_target,
                time_weights[2],
                self.x_time_derivative_exit.row(row),
            );
            dense_outer_accumulate(time_target, time_weights[3], self.x_time_right.row(row));
            for (a, b, lhs, rhs) in time_cross_weights {
                let weight = primary_hessian[[a, b]];
                if weight == 0.0 {
                    continue;
                }
                dense_symmetric_cross_accumulate(time_target, weight, lhs.row(row), rhs.row(row));
            }
        }

        let mean_weight = primary_hessian[[LATENT_SURVIVAL_PRIMARY_MU, LATENT_SURVIVAL_PRIMARY_MU]];
        self.x_mean
            .syr_row_into_view(
                row,
                mean_weight,
                target.slice_mut(s![slices.mean.clone(), slices.mean.clone()]),
            )
            .map_err(|error| {
                format!(
                    "latent survival mean Hessian pullback dimension mismatch: row={row}, mean_slice={:?}, target_dim={:?}, x_mean_cols={}, error={error}",
                    slices.mean,
                    target.dim(),
                    self.x_mean.ncols()
                )
            })?;

        let mean_row = self
            .x_mean
            .try_row_chunk(row..row + 1)
            .map_err(|error| {
                format!(
                    "latent survival mean pullback row chunk failed: row={row}, x_mean_rows={}, x_mean_cols={}, error={error}",
                    self.x_mean.nrows(),
                    self.x_mean.ncols()
                )
            })?;
        let mean_vec = mean_row.row(0);
        let time_mean_weights = [
            (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
            (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
            (
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                self.x_time_derivative_exit.row(row),
            ),
            (LATENT_SURVIVAL_PRIMARY_Q_RIGHT, self.x_time_right.row(row)),
        ];
        for (primary_idx, time_vec) in time_mean_weights {
            let weight = primary_hessian[[primary_idx, LATENT_SURVIVAL_PRIMARY_MU]];
            if weight == 0.0 {
                continue;
            }
            for i in 0..time_vec.len() {
                let xi = time_vec[i];
                if xi == 0.0 {
                    continue;
                }
                for j in 0..mean_vec.len() {
                    let xj = mean_vec[j];
                    if xj == 0.0 {
                        continue;
                    }
                    target[[slices.time.start + i, slices.mean.start + j]] += weight * xi * xj;
                    target[[slices.mean.start + j, slices.time.start + i]] += weight * xj * xi;
                }
            }
        }

        if let Some(log_sigma) = &slices.log_sigma {
            let sigma_idx = log_sigma.start;
            target[[sigma_idx, sigma_idx]] += primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            ]];

            for (primary_idx, time_vec) in [
                (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
                (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
                (
                    LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                    self.x_time_derivative_exit.row(row),
                ),
                (LATENT_SURVIVAL_PRIMARY_Q_RIGHT, self.x_time_right.row(row)),
            ] {
                let weight = primary_hessian[[primary_idx, LATENT_SURVIVAL_PRIMARY_LOG_SIGMA]];
                if weight == 0.0 {
                    continue;
                }
                for i in 0..time_vec.len() {
                    let xi = time_vec[i];
                    if xi == 0.0 {
                        continue;
                    }
                    target[[slices.time.start + i, sigma_idx]] += weight * xi;
                    target[[sigma_idx, slices.time.start + i]] += weight * xi;
                }
            }

            let mean_sigma_weight = primary_hessian[[
                LATENT_SURVIVAL_PRIMARY_MU,
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            ]];
            if mean_sigma_weight != 0.0 {
                for j in 0..mean_vec.len() {
                    let xj = mean_vec[j];
                    if xj == 0.0 {
                        continue;
                    }
                    target[[slices.mean.start + j, sigma_idx]] += mean_sigma_weight * xj;
                    target[[sigma_idx, slices.mean.start + j]] += mean_sigma_weight * xj;
                }
            }
        }
        Ok(())
    }

    fn evaluate_exact_newton_joint_gradient_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>), String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        let acc = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || LatentSurvivalJointGradientAccum {
                ll: 0.0,
                gradient: Array1::<f64>::zeros(total),
            },
            |row_idx, acc| {
                let wi = self.weights[row_idx];
                if wi <= MIN_WEIGHT {
                    return Ok(());
                }
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let (row_ll, primary_gradient, _) = latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                    mu[row_idx],
                    sigma,
                    include_log_sigma,
                )?;
                acc.ll += wi * row_ll;
                self.add_pullback_primary_gradient(
                    &mut acc.gradient,
                    row_idx,
                    &slices,
                    &primary_gradient,
                    wi,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.ll += chunk_acc.ll;
                total_acc.gradient += &chunk_acc.gradient;
            },
        )?;
        Ok((acc.ll, acc.gradient))
    }

    /// Per-row residuals of the unpenalized NLL with respect to the three
    /// additive baseline time-block offsets `(entry, exit, derivative)`.
    ///
    /// The baseline configuration θ enters the latent-survival working model
    /// only through the additive offsets on the three time channels
    ///   q_entry = x_time_entry·β_time + o_E(θ),
    ///   q_exit  = x_time_exit·β_time  + o_X(θ),
    ///   q̇_exit = x_time_deriv·β_time + o_D(θ),
    /// exactly the offset channel the transformation path carries through
    /// [`WorkingModelSurvival::offset_channel_residuals`]. Because
    /// `∂q_ch/∂o_ch = 1`, the residual `∂NLL/∂o_ch_i` equals
    /// `−∂(log-likelihood)/∂q_ch_i`, and the per-row primary log-likelihood
    /// gradient over `(q_entry, q_exit, q̇_exit)` is precisely the
    /// `Q_ENTRY`/`Q_EXIT`/`QDOT_EXIT` components returned by
    /// [`latent_survival_row_primary_gradient_hessian`]. Sampleweight-scaled to
    /// match the [`OffsetChannelResiduals`] contract consumed by
    /// `baseline_chain_rule_gradient`.
    ///
    /// At the converged (constrained) β̂ the envelope theorem makes this the
    /// exact θ-gradient of the profile penalized NLL `0.5·deviance + 0.5·βᵀSβ`.
    /// The interval upper-bound `q_right = x_time_right·β_time + o_R(θ)` channel
    /// DOES carry its own baseline-θ offset `o_R(θ)` (the time basis evaluated at
    /// the bracket upper bound `R`), distinct from the exit offset at `L`, so its
    /// residual `−∂(log-likelihood)/∂q_right` is returned in the dedicated
    /// [`OffsetChannelResiduals::right`] channel; it is exactly 0 on every
    /// non-interval row (the `Q_RIGHT` primary channel is inert there) and the
    /// baseline-θ chain rule contracts it against the `age_right`-evaluated
    /// η-partial.
    pub fn offset_channel_residuals(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<crate::families::survival::OffsetChannelResiduals, String> {
        let n = self.event_target.len();
        if block_states.is_empty() {
            // Degraded-fit fallback mirroring the location-scale family: an
            // empty block-state slate (ARC deterministic-replay stall) yields
            // zero residuals so the outer baseline-θ BFGS sees ‖g‖ = 0 and
            // terminates cleanly at the current θ̂ rather than panicking.
            log::warn!(
                "LatentSurvivalFamily::offset_channel_residuals: block_states is empty \
                 (degraded fit); returning zero offset residuals (n={n})"
            );
            return Ok(crate::families::survival::OffsetChannelResiduals {
                exit: Array1::<f64>::zeros(n),
                entry: Array1::<f64>::zeros(n),
                derivative: Array1::<f64>::zeros(n),
                right: Array1::<f64>::zeros(n),
            });
        }
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let include_log_sigma = self.joint_slices().log_sigma.is_some();
        let mut entry = Array1::<f64>::zeros(n);
        let mut exit = Array1::<f64>::zeros(n);
        let mut derivative = Array1::<f64>::zeros(n);
        let mut right = Array1::<f64>::zeros(n);
        for row_idx in 0..n {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row = self.build_row_at(
                row_idx,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                q_right[row_idx],
            )?;
            let (_, primary_gradient, _) = latent_survival_row_primary_gradient_hessian(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                q_right[row_idx],
                mu[row_idx],
                sigma,
                include_log_sigma,
            )?;
            // ∂NLL/∂o_ch = −w · ∂(log-likelihood)/∂q_ch.
            entry[row_idx] = -wi * primary_gradient[LATENT_SURVIVAL_PRIMARY_Q_ENTRY];
            exit[row_idx] = -wi * primary_gradient[LATENT_SURVIVAL_PRIMARY_Q_EXIT];
            derivative[row_idx] = -wi * primary_gradient[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT];
            // Interval upper-bound (`R`) channel. `q_right` shares the time-block
            // coefficients but carries its OWN baseline-θ η-offset evaluated at
            // `R` (`o_R(θ)`), so the profile-NLL θ-gradient must include it.
            // `∂(log-likelihood)/∂q_right` is exactly 0 for non-interval rows
            // (the `Q_RIGHT` channel is inert there), so this is 0 except on
            // interval-censored rows.
            right[row_idx] = -wi * primary_gradient[LATENT_SURVIVAL_PRIMARY_Q_RIGHT];
        }
        Ok(crate::families::survival::OffsetChannelResiduals {
            exit,
            entry,
            derivative,
            right,
        })
    }

    /// Block-diagonal-only pullback: writes only time-time, mean-mean, and
    /// log_sigma-log_sigma rowwise contributions into per-block targets.
    /// Used by `evaluate()` to populate per-block working sets without ever
    /// materializing the cross blocks the inner solver does not consume.
    fn add_pullback_primary_block_diagonals(
        &self,
        row: usize,
        primary_hessian: &Array2<f64>,
        time_target: &mut Array2<f64>,
        mean_target: &mut Array2<f64>,
        log_sigma_target: Option<&mut Array2<f64>>,
    ) -> Result<(), String> {
        let h = primary_hessian;
        // Time block: 4 squared rows (entry/exit/qdot/right) + 6 symmetric
        // crosses. The interval right-boundary functional `q_right` shares the
        // time-block coefficients, so it accumulates into the same time target.
        dense_outer_accumulate(
            time_target,
            h[[
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
            ]],
            self.x_time_entry.row(row),
        );
        dense_outer_accumulate(
            time_target,
            h[[
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
            ]],
            self.x_time_exit.row(row),
        );
        dense_outer_accumulate(
            time_target,
            h[[
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
            ]],
            self.x_time_derivative_exit.row(row),
        );
        dense_outer_accumulate(
            time_target,
            h[[
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
            ]],
            self.x_time_right.row(row),
        );
        for (a, b, lhs, rhs) in [
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                &self.x_time_entry,
                &self.x_time_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                &self.x_time_entry,
                &self.x_time_derivative_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                &self.x_time_exit,
                &self.x_time_derivative_exit,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                &self.x_time_entry,
                &self.x_time_right,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                &self.x_time_exit,
                &self.x_time_right,
            ),
            (
                LATENT_SURVIVAL_PRIMARY_QDOT_EXIT,
                LATENT_SURVIVAL_PRIMARY_Q_RIGHT,
                &self.x_time_derivative_exit,
                &self.x_time_right,
            ),
        ] {
            let weight = h[[a, b]];
            if weight == 0.0 {
                continue;
            }
            dense_symmetric_cross_accumulate(time_target, weight, lhs.row(row), rhs.row(row));
        }
        // Mean block.
        let mean_weight = h[[LATENT_SURVIVAL_PRIMARY_MU, LATENT_SURVIVAL_PRIMARY_MU]];
        self.x_mean
            .syr_row_into_view(row, mean_weight, mean_target.view_mut())
            .map_err(|error| {
                format!(
                    "latent survival mean block-diagonal pullback dimension mismatch: row={row}, mean_target_dim={:?}, x_mean_cols={}, error={error}",
                    mean_target.dim(),
                    self.x_mean.ncols()
                )
            })?;
        // Log-σ block (scalar).
        if let Some(target) = log_sigma_target {
            target[[0, 0]] += h[[
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
                LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            ]];
        }
        Ok(())
    }

    /// Block-diagonal evaluator used by `evaluate()`. Returns the per-row
    /// log-likelihood, the joint gradient (sliced into block gradients by
    /// the caller), and the three per-block diagonal Hessians without ever
    /// materializing the full joint matrix.
    fn evaluate_exact_newton_block_diagonals(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<
        (
            f64,
            Array1<f64>,
            Array2<f64>,
            Array2<f64>,
            Option<Array2<f64>>,
        ),
        String,
    > {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let mut ll = 0.0;
        let mut gradient = Array1::<f64>::zeros(slices.total);
        let p_time = slices.time.len();
        let p_mean = slices.mean.len();
        let mut hess_time = Array2::<f64>::zeros((p_time, p_time));
        let mut hess_mean = Array2::<f64>::zeros((p_mean, p_mean));
        let mut hess_log_sigma = if include_log_sigma {
            Some(Array2::<f64>::zeros((1, 1)))
        } else {
            None
        };
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row = self.build_row_at(
                row_idx,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                q_right[row_idx],
            )?;
            let (row_ll, primary_gradient, primary_hessian) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                    mu[row_idx],
                    sigma,
                    include_log_sigma,
                )?;
            ll += wi * row_ll;
            self.add_pullback_primary_gradient(
                &mut gradient,
                row_idx,
                &slices,
                &primary_gradient,
                wi,
            )?;
            self.add_pullback_primary_block_diagonals(
                row_idx,
                &(wi * primary_hessian),
                &mut hess_time,
                &mut hess_mean,
                hess_log_sigma.as_mut(),
            )?;
        }
        Ok((ll, gradient, hess_time, hess_mean, hess_log_sigma))
    }

    fn evaluate_exact_newton_joint_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        let acc = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || LatentSurvivalJointDenseAccum {
                ll: 0.0,
                gradient: Array1::<f64>::zeros(total),
                hessian: Array2::<f64>::zeros((total, total)),
            },
            |row_idx, acc| {
                let wi = self.weights[row_idx];
                if wi <= MIN_WEIGHT {
                    return Ok(());
                }
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let (row_ll, primary_gradient, primary_hessian) =
                    latent_survival_row_primary_gradient_hessian(
                        &self.quadctx,
                        &row,
                        q_entry[row_idx],
                        q_exit[row_idx],
                        qdot_exit[row_idx],
                        q_right[row_idx],
                        mu[row_idx],
                        sigma,
                        include_log_sigma,
                    )?;
                acc.ll += wi * row_ll;
                self.add_pullback_primary_gradient(
                    &mut acc.gradient,
                    row_idx,
                    &slices,
                    &primary_gradient,
                    wi,
                )?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &(wi * primary_hessian),
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.ll += chunk_acc.ll;
                total_acc.gradient += &chunk_acc.gradient;
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        Ok((acc.ll, acc.gradient, acc.hessian))
    }

    fn exact_newton_joint_hessian_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "latent survival joint dH direction length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        let acc = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || LatentSurvivalDenseHessianAccum {
                hessian: Array2::<f64>::zeros((total, total)),
            },
            |row_idx, acc| {
                let wi = self.weights[row_idx];
                if wi <= MIN_WEIGHT {
                    return Ok(());
                }
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let direction = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
                let third = latent_survival_row_primary_third_contracted(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                    mu[row_idx],
                    sigma,
                    &direction,
                    include_log_sigma,
                )?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &(wi * third),
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        Ok(acc.hessian)
    }

    fn exact_newton_joint_hessian_second_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        if d_beta_u_flat.len() != slices.total || d_beta_v_flat.len() != slices.total {
            return Err(format!(
                "latent survival joint d2H direction length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                slices.total
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        let acc = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || LatentSurvivalDenseHessianAccum {
                hessian: Array2::<f64>::zeros((total, total)),
            },
            |row_idx, acc| {
                let wi = self.weights[row_idx];
                if wi <= MIN_WEIGHT {
                    return Ok(());
                }
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let direction_u =
                    self.row_primary_direction_from_flat(row_idx, &slices, d_beta_u_flat);
                let direction_v =
                    self.row_primary_direction_from_flat(row_idx, &slices, d_beta_v_flat);
                let fourth = latent_survival_row_primary_fourth_contracted(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                    mu[row_idx],
                    sigma,
                    &direction_u,
                    &direction_v,
                    include_log_sigma,
                )?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &(wi * fourth),
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        Ok(acc.hessian)
    }
}

fn log_kernel_ratio(
    bundle: &crate::families::survival::lognormal_kernel::LogLognormalKernelBundle,
    num: usize,
    den: usize,
) -> f64 {
    let delta = bundle.get(num) - bundle.get(den);
    if delta.is_finite() {
        delta.exp()
    } else if delta > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

fn logk_q_derivatives(
    quadctx: &QuadratureContext,
    k: usize,
    mass: f64,
    mu: f64,
    sigma: f64,
) -> Result<(f64, f64, IntegratedExpectationMode), LatentSurvivalError> {
    if mass <= 0.0 {
        return Ok((0.0, 0.0, IntegratedExpectationMode::ExactClosedForm));
    }
    let bundle = log_kernel_bundle(quadctx, mass, mu, sigma, k + 2).map_err(|e| {
        LatentSurvivalError::NumericalFailure {
            reason: format!("latent survival kernel evaluation failed: {e}"),
        }
    })?;
    let r1 = log_kernel_ratio(&bundle, k + 1, k);
    let r2 = log_kernel_ratio(&bundle, k + 2, k);
    let d1 = -mass * r1;
    let d2 = d1 + mass * mass * (r2 - r1 * r1);
    Ok((d1, d2, bundle.mode))
}

fn latent_survival_time_jet(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    qdot_exit: f64,
    mu: f64,
    sigma: f64,
) -> Result<LatentSurvivalTimeJet, LatentSurvivalError> {
    let (entry_d1, entry_d2, _) = logk_q_derivatives(quadctx, 0, row.mass_entry, mu, sigma)?;
    match row.event_type {
        LatentSurvivalEventType::RightCensored => {
            let (exit_d1, exit_d2, _) = logk_q_derivatives(quadctx, 0, row.mass_exit, mu, sigma)?;
            Ok(LatentSurvivalTimeJet {
                grad_entry: -entry_d1,
                grad_exit: exit_d1,
                neg_hess_entry: entry_d2,
                neg_hess_exit: -exit_d2,
            })
        }
        LatentSurvivalEventType::ExactEvent => {
            if !(qdot_exit.is_finite() && qdot_exit > 0.0) {
                return Err(LatentSurvivalError::NumericalFailure {
                    reason: format!(
                        "latent survival requires positive finite baseline hazard derivative, got {qdot_exit}"
                    ),
                });
            }
            if row.hazard_unloaded > 0.0 {
                let bundle =
                    log_kernel_bundle(quadctx, row.mass_exit, mu, sigma, 3).map_err(|e| {
                        LatentSurvivalError::NumericalFailure {
                            reason: format!("latent survival kernel evaluation failed: {e}"),
                        }
                    })?;
                let (unloaded_d1, unloaded_d2, _) =
                    logk_q_derivatives(quadctx, 0, row.mass_exit, mu, sigma)?;
                let (loaded_log_d1, loaded_d2, _) =
                    logk_q_derivatives(quadctx, 1, row.mass_exit, mu, sigma)?;
                let loaded_d1 = 1.0 + loaded_log_d1;
                let log_loaded = row.hazard_loaded.ln() + bundle.get(1);
                let log_unloaded = row.hazard_unloaded.ln() + bundle.get(0);
                let shift = log_loaded.max(log_unloaded);
                let loaded_weight = (log_loaded - shift).exp();
                let unloaded_weight = (log_unloaded - shift).exp();
                let normalizer = loaded_weight + unloaded_weight;
                if !(normalizer.is_finite() && normalizer > 0.0) {
                    return Err(LatentSurvivalError::NumericalFailure {
                        reason: "latent survival exact-event numerator became non-finite under loaded/unloaded hazard decomposition"
                            .to_string(),
                    });
                }
                let w_loaded = loaded_weight / normalizer;
                let w_unloaded = unloaded_weight / normalizer;
                let grad_exit = w_loaded * loaded_d1 + w_unloaded * unloaded_d1;
                let d2_exit = w_loaded * (loaded_d2 + loaded_d1 * loaded_d1)
                    + w_unloaded * (unloaded_d2 + unloaded_d1 * unloaded_d1)
                    - grad_exit * grad_exit;
                Ok(LatentSurvivalTimeJet {
                    grad_entry: -entry_d1,
                    grad_exit,
                    neg_hess_entry: entry_d2,
                    neg_hess_exit: -d2_exit,
                })
            } else {
                let (exit_d1, exit_d2, _) =
                    logk_q_derivatives(quadctx, 1, row.mass_exit, mu, sigma)?;
                Ok(LatentSurvivalTimeJet {
                    grad_entry: -entry_d1,
                    grad_exit: 1.0 + exit_d1,
                    neg_hess_entry: entry_d2,
                    neg_hess_exit: -exit_d2,
                })
            }
        }
        LatentSurvivalEventType::IntervalCensored => {
            Err(LatentSurvivalError::UnsupportedConfiguration {
                reason:
                    "latent survival dynamic time derivatives do not implement interval censoring"
                        .to_string(),
            })
        }
    }
}

fn dense_outer_accumulate<S>(
    target: &mut ndarray::ArrayBase<S, ndarray::Ix2>,
    weight: f64,
    x: ArrayView1<'_, f64>,
) where
    S: ndarray::DataMut<Elem = f64>,
{
    for a in 0..x.len() {
        let xa = x[a];
        if xa == 0.0 {
            continue;
        }
        for b in 0..x.len() {
            let xb = x[b];
            if xb == 0.0 {
                continue;
            }
            target[[a, b]] += weight * xa * xb;
        }
    }
}

fn dense_symmetric_cross_accumulate<S>(
    target: &mut ndarray::ArrayBase<S, ndarray::Ix2>,
    weight: f64,
    x: ArrayView1<'_, f64>,
    y: ArrayView1<'_, f64>,
) where
    S: ndarray::DataMut<Elem = f64>,
{
    for a in 0..x.len() {
        let xa = x[a];
        let ya = y[a];
        if xa == 0.0 && ya == 0.0 {
            continue;
        }
        for b in 0..x.len() {
            let xb = x[b];
            let yb = y[b];
            let contribution = xa * yb + ya * xb;
            if contribution == 0.0 {
                continue;
            }
            target[[a, b]] += weight * contribution;
        }
    }
}

fn build_latent_survival_row(
    row_index: usize,
    hazard_loading: HazardLoading,
    event_type: LatentSurvivalEventType,
    q_entry: f64,
    q_exit: f64,
    qdot_exit: f64,
    q_right: f64,
    unloaded_mass_entry: f64,
    unloaded_mass_exit: f64,
    unloaded_mass_right: f64,
    unloaded_hazard_exit: f64,
) -> Result<LatentSurvivalRow, LatentSurvivalError> {
    if !(q_entry.is_finite() && q_exit.is_finite()) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent survival requires finite q_entry and q_exit, got q_entry={q_entry}, q_exit={q_exit}"
            ),
        });
    }
    if q_exit < q_entry {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent survival requires q_exit >= q_entry so cumulative mass is monotone, got q_entry={q_entry}, q_exit={q_exit}"
            ),
        });
    }
    if !(unloaded_mass_entry.is_finite()
        && unloaded_mass_exit.is_finite()
        && unloaded_hazard_exit.is_finite())
    {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!(
                "latent survival requires finite unloaded components, got entry_mass={unloaded_mass_entry}, exit_mass={unloaded_mass_exit}, exit_hazard={unloaded_hazard_exit}"
            ),
        });
    }
    if unloaded_mass_entry < 0.0
        || unloaded_mass_exit < unloaded_mass_entry
        || unloaded_hazard_exit < 0.0
    {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!(
                "latent survival requires unloaded masses/hazard to be non-negative and monotone, got entry_mass={unloaded_mass_entry}, exit_mass={unloaded_mass_exit}, exit_hazard={unloaded_hazard_exit}"
            ),
        });
    }
    let mass_entry = q_entry.exp();
    let mass_exit = q_exit.exp();
    let row = match event_type {
        LatentSurvivalEventType::RightCensored => {
            validate_unloaded_components_for_loading(
                "latent-survival",
                row_index,
                hazard_loading,
                unloaded_mass_entry,
                unloaded_mass_exit,
                Some(unloaded_hazard_exit),
            )?;
            LatentSurvivalRow::right_censored(
                mass_entry,
                mass_exit,
                unloaded_mass_entry,
                unloaded_mass_exit,
            )
        }
        LatentSurvivalEventType::ExactEvent => {
            validate_unloaded_components_for_loading(
                "latent-survival",
                row_index,
                hazard_loading,
                unloaded_mass_entry,
                unloaded_mass_exit,
                Some(unloaded_hazard_exit),
            )?;
            LatentSurvivalRow::exact_event(
                mass_entry,
                mass_exit,
                unloaded_mass_entry,
                unloaded_mass_exit,
                mass_exit
                    * if qdot_exit.is_finite() && qdot_exit > 0.0 {
                        qdot_exit
                    } else {
                        return Err(LatentSurvivalError::NumericalFailure {
                            reason: format!(
                                "latent survival exact event requires positive finite baseline hazard derivative, got {qdot_exit}"
                            ),
                        });
                    },
                unloaded_hazard_exit,
            )
        }
        LatentSurvivalEventType::IntervalCensored => {
            // Interval `(L, R]`: `q_exit` carries the LEFT boundary transform
            // `log B(L)` (so `mass_left = exp(q_exit)`) and `q_right` the RIGHT
            // boundary `log B(R)`. The likelihood is the survival-mass
            // difference `log[S(L) − S(R)]`, requiring `B(L) ≤ B(R)` i.e.
            // `q_exit ≤ q_right`. No event hazard participates, so the unloaded
            // exit hazard must be the full-loading zero (validated below via the
            // interval-specific unloaded check at the left/right boundaries).
            if !q_right.is_finite() {
                return Err(LatentSurvivalError::NumericalFailure {
                    reason: format!(
                        "latent survival interval row {} requires a finite q_right, got {q_right}",
                        row_index + 1
                    ),
                });
            }
            if q_right < q_exit {
                return Err(LatentSurvivalError::NumericalFailure {
                    reason: format!(
                        "latent survival interval row {} requires q_right >= q_exit (R >= L) so the \
                         survival-mass difference is non-negative, got q_left={q_exit}, q_right={q_right}",
                        row_index + 1
                    ),
                });
            }
            if !(unloaded_mass_right.is_finite()) || unloaded_mass_right < unloaded_mass_exit {
                return Err(LatentSurvivalError::InvalidDataset {
                    reason: format!(
                        "latent survival interval row {} requires a finite unloaded right mass >= unloaded left mass, got left={unloaded_mass_exit}, right={unloaded_mass_right}",
                        row_index + 1
                    ),
                });
            }
            // Interval rows carry no exit-event hazard; the loaded/unloaded
            // contract is validated by `LatentSurvivalRow::validate` (entry <=
            // left <= right monotonicity on both loaded and unloaded masses).
            let mass_right = q_right.exp();
            LatentSurvivalRow::interval_censored(
                mass_entry,
                mass_exit,
                mass_right,
                unloaded_mass_entry,
                unloaded_mass_exit,
                unloaded_mass_right,
            )
        }
    };
    row.validate()
        .map_err(|e| LatentSurvivalError::InvalidDataset {
            reason: e.to_string(),
        })?;
    Ok(row)
}

#[derive(Clone, Copy)]
struct BinaryFromLogSurvival {
    log_lik: f64,
    /// dℓ/ds where s = log_survival and ℓ = log_lik. For event=1 this is
    /// ℓ' = -S/(1-S); for event=0 this is ℓ' = 1 (because ℓ ≡ s).
    grad_scale: f64,
    /// Coefficient applied to `survival_jet.neg_hessian` (which equals
    /// -d²s/dβ²) when assembling the negative Hessian of `wi * log_lik`
    /// against β. The Newton accumulator computes
    ///     neg_Hess(log_lik) = grad_scale * neg_hessian + outer_scale * score²
    /// so by the chain rule this MUST equal `grad_scale` (= ℓ'). Keeping
    /// the two fields separate is purely for readability at call sites;
    /// the `assert!` in [`binary_log_survival_scales`] enforces the
    /// equality.
    neg_hess_scale: f64,
    /// -ℓ''(s). For event=1 this is +S/(1-S)²; for event=0 it is 0.
    outer_scale: f64,
    /// ℓ''(s) — derivative of `grad_scale` w.r.t. s.
    grad_scale_prime: f64,
    /// ℓ'''(s) — second derivative of `grad_scale` w.r.t. s.
    grad_scale_second: f64,
    /// -ℓ'''(s) — derivative of `outer_scale` w.r.t. s.
    outer_scale_prime: f64,
    /// -ℓ''''(s) — second derivative of `outer_scale` w.r.t. s.
    outer_scale_second: f64,
}

/// Analytic source of truth for the directional derivatives of
/// ℓ(s) = log(1 - exp(s)) at s = `log_survival`. Returns
/// `(ℓ, ℓ', ℓ'', ℓ''', ℓ'''')`. All consumer scales (`grad_scale`,
/// `neg_hess_scale`, `outer_scale`, and their two derivatives each)
/// are derived from this single function so the sign/algebra cannot
/// drift between sites.
#[inline]
fn binary_log_survival_scales(survival: f64, event_prob: f64) -> (f64, f64, f64, f64, f64) {
    // ℓ(s)   = log(1 - exp(s)) = log(event_prob)
    // dS/ds  = S,    dP/ds = -S        (S=survival, P=event_prob)
    // ℓ'(s)  = -S/P
    // ℓ''(s) = d/ds[-S/P] = -S/P²        (since P + S = 1)
    // ℓ'''(s) = d/ds[-S/P²] = -S(1 + S)/P³
    // ℓ''''(s) = d/ds[-S(1+S)/P³]
    //          = -S/P³ - 3S²/P³ - 6S²(1+S)/P⁴ - ... ; expanded form below.
    let log_lik = event_prob.ln();
    let p = event_prob;
    let p2 = p * p;
    let p3 = p2 * p;
    let p4 = p3 * p;
    let s = survival;
    let s2 = s * s;
    let s3 = s2 * s;
    let ell_prime = -s / p;
    let ell_pp = -s / p2;
    let ell_ppp = -s * (1.0 + s) / p3;
    // ℓ''''(s) = -S·(1 + 4S + S²) / P⁴ - 3·S²·(1+S)/P⁴? Use the equivalent
    // expansion that matches the prior closed form:
    //   d/ds[-S(1+S)/P³] = -(S + 2S²)/P³ - 3·S·(1+S)·S/P⁴
    //                    = -(S + 2S²)/P³ - 3S²(1+S)/P⁴
    // Combining over P⁴: -(S + 2S²)·P/P⁴ - 3S²(1+S)/P⁴
    //                  = -[S·P + 2S²·P + 3S² + 3S³] / P⁴
    // With P = 1 - S: S·P = S - S²; 2S²·P = 2S² - 2S³.
    //   numerator = -[S - S² + 2S² - 2S³ + 3S² + 3S³] = -[S + 4S² + S³].
    // So ℓ''''(s) = -(S + 4S² + S³) / P⁴.
    let ell_pppp = -(s + 4.0 * s2 + s3) / p4;
    (log_lik, ell_prime, ell_pp, ell_ppp, ell_pppp)
}

fn binary_from_log_survival(
    log_survival: f64,
    event: u8,
) -> Result<BinaryFromLogSurvival, LatentSurvivalError> {
    if event == 0 {
        // ℓ(s) = s ⇒ ℓ' = 1, ℓ'' = ℓ''' = ℓ'''' = 0.
        return Ok(BinaryFromLogSurvival {
            log_lik: log_survival,
            grad_scale: 1.0,
            neg_hess_scale: 1.0,
            outer_scale: 0.0,
            grad_scale_prime: 0.0,
            grad_scale_second: 0.0,
            outer_scale_prime: 0.0,
            outer_scale_second: 0.0,
        });
    }
    if event != 1 {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!("latent-binary requires event targets in {{0,1}}, got {event}"),
        });
    }
    // Cap log S(t) strictly below zero so the event probability
    // `1 - exp(log S)` stays strictly positive even when the survival
    // probability rounds to exactly 1 (log S == 0): a zero event probability
    // would make the binary log-likelihood `log(event_prob)` diverge. The cap
    // is at the f64 resolution near 1.0, so it never perturbs a genuinely
    // informative survival value.
    const MAX_LOG_SURVIVAL: f64 = -1e-15;
    let log_survival = log_survival.min(MAX_LOG_SURVIVAL);
    let survival = log_survival.exp();
    let event_prob = 1.0 - survival;
    if !(event_prob.is_finite() && event_prob > 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent-binary encountered non-positive event probability from log survival {log_survival}"
            ),
        });
    }
    let (log_lik, ell_prime, ell_pp, ell_ppp, ell_pppp) =
        binary_log_survival_scales(survival, event_prob);
    let grad_scale = ell_prime;
    let neg_hess_scale = ell_prime; // coefficient on (-d²s/dβ²); equals ℓ'.
    let outer_scale = -ell_pp;
    let grad_scale_prime = ell_pp;
    let grad_scale_second = ell_ppp;
    let outer_scale_prime = -ell_ppp;
    let outer_scale_second = -ell_pppp;
    // The Newton accumulator at the call sites computes
    //     neg_Hess(log_lik) = neg_hess_scale * (-d²s/dβ²) + outer_scale * (ds/dβ)²
    // For this identity to hold by the chain rule, the coefficient on the
    // neg_hessian term must equal ℓ' (== grad_scale). Document the invariant.
    assert!(
        (grad_scale - neg_hess_scale).abs() <= 1e-15 * grad_scale.abs().max(1.0),
        "binary_from_log_survival invariant: neg_hess_scale ({neg_hess_scale}) must equal grad_scale ({grad_scale}) so that grad_scale and the coefficient on neg_hessian share sign"
    );
    assert!(
        outer_scale >= 0.0 || !outer_scale.is_finite(),
        "binary_from_log_survival invariant: outer_scale (= -ℓ'') must be non-negative for event=1; got {outer_scale}"
    );
    Ok(BinaryFromLogSurvival {
        log_lik,
        grad_scale,
        neg_hess_scale,
        outer_scale,
        grad_scale_prime,
        grad_scale_second,
        outer_scale_prime,
        outer_scale_second,
    })
}

impl LatentBinaryFamily {
    /// Assemble the per-row [`LatentSurvivalRow`] for a row treated as a pure
    /// right-censored survival contribution (exit time is the censoring
    /// boundary, unit exit-hazard derivative, no right / post-exit unloaded
    /// mass). Shared by every per-row binary-from-survival pullback reduction;
    /// behavior is identical to the previously inlined `RightCensored` call.
    fn build_right_censored_row_at(
        &self,
        row_idx: usize,
        q_entry: f64,
        q_exit: f64,
    ) -> Result<LatentSurvivalRow, LatentSurvivalError> {
        build_latent_survival_row(
            row_idx,
            self.hazard_loading,
            LatentSurvivalEventType::RightCensored,
            q_entry,
            q_exit,
            1.0,
            q_exit,
            self.unloaded_mass_entry[row_idx],
            self.unloaded_mass_exit[row_idx],
            0.0,
            0.0,
        )
    }

    fn joint_slices(&self) -> LatentSurvivalJointSlices {
        let p_time = self.x_time_exit.ncols();
        let p_mean = self.x_mean.ncols();
        LatentSurvivalJointSlices {
            time: 0..p_time,
            mean: p_time..p_time + p_mean,
            log_sigma: None,
            total: p_time + p_mean,
        }
    }

    fn row_primary_direction_from_flat(
        &self,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        d_beta_flat: &Array1<f64>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        let d_time = d_beta_flat.slice(s![slices.time.clone()]);
        out[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] = self.x_time_entry.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_Q_EXIT] = self.x_time_exit.row(row).dot(&d_time);
        out[LATENT_SURVIVAL_PRIMARY_MU] = self
            .x_mean
            .dot_row_view(row, d_beta_flat.slice(s![slices.mean.clone()]));
        out
    }

    fn add_pullback_primary_gradient(
        &self,
        target: &mut Array1<f64>,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        primary_gradient: &Array1<f64>,
        weight: f64,
    ) {
        for (primary_idx, time_vec) in [
            (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
            (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
        ] {
            let scale = weight * primary_gradient[primary_idx];
            if scale == 0.0 {
                continue;
            }
            for i in 0..time_vec.len() {
                let xi = time_vec[i];
                if xi != 0.0 {
                    target[slices.time.start + i] += scale * xi;
                }
            }
        }

        let mean_scale = weight * primary_gradient[LATENT_SURVIVAL_PRIMARY_MU];
        if mean_scale != 0.0 {
            self.x_mean
                .axpy_row_into(
                    row,
                    mean_scale,
                    &mut target.slice_mut(s![slices.mean.clone()]),
                )
                // SAFETY: `slices.mean` sized at construction to match
                // `x_mean.ncols()`; an error means caller-side shape drift,
                // an invariant violation. A swallowed sentinel would silently
                // corrupt the joint gradient, so fail loudly instead.
                .unwrap_or_else(|error| {
                    panic!(
                        "latent binary mean gradient pullback dimension mismatch: row={row}, mean_slice={:?}, target_len={}, x_mean_cols={}, error={error}",
                        slices.mean,
                        target.len(),
                        self.x_mean.ncols()
                    )
                });
        }
    }

    fn add_pullback_primary_hessian(
        &self,
        target: &mut Array2<f64>,
        row: usize,
        slices: &LatentSurvivalJointSlices,
        primary_hessian: &Array2<f64>,
    ) {
        {
            let time_target = &mut target.slice_mut(s![slices.time.clone(), slices.time.clone()]);
            dense_outer_accumulate(
                time_target,
                primary_hessian[[
                    LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                    LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                ]],
                self.x_time_entry.row(row),
            );
            dense_outer_accumulate(
                time_target,
                primary_hessian[[
                    LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                    LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                ]],
                self.x_time_exit.row(row),
            );
            dense_symmetric_cross_accumulate(
                time_target,
                primary_hessian[[
                    LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
                    LATENT_SURVIVAL_PRIMARY_Q_EXIT,
                ]],
                self.x_time_entry.row(row),
                self.x_time_exit.row(row),
            );
        }

        let mean_weight = primary_hessian[[LATENT_SURVIVAL_PRIMARY_MU, LATENT_SURVIVAL_PRIMARY_MU]];
        self.x_mean
            .syr_row_into_view(
                row,
                mean_weight,
                target.slice_mut(s![slices.mean.clone(), slices.mean.clone()]),
            )
            .unwrap_or_else(|error| {
                // SAFETY: `slices.mean` × `slices.mean` slab sized at
                // construction to `x_mean.ncols()` × `x_mean.ncols()`;
                // an error here is caller-side shape drift, an invariant
                // violation. A swallowed sentinel would silently corrupt the
                // joint Hessian, so fail loudly instead.
                panic!(
                    "latent binary mean Hessian pullback dimension mismatch: row={row}, mean_slice={:?}, target_dim={:?}, x_mean_cols={}, error={error}",
                    slices.mean,
                    target.dim(),
                    self.x_mean.ncols()
                )
            });

        let mean_row = self
            .x_mean
            .try_row_chunk(row..row + 1)
            .unwrap_or_else(|error| {
                // SAFETY: row index comes from the enclosing `0..n` loop
                // bound by `self.x_mean.nrows()`, so `row..row+1` is
                // always a valid single-row chunk.
                panic!(
                    "latent binary mean pullback row chunk failed: row={row}, x_mean_rows={}, x_mean_cols={}, error={error}",
                    self.x_mean.nrows(),
                    self.x_mean.ncols()
                )
            });
        let mean_vec = mean_row.row(0);
        for (primary_idx, time_vec) in [
            (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
            (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
        ] {
            let weight = primary_hessian[[primary_idx, LATENT_SURVIVAL_PRIMARY_MU]];
            if weight == 0.0 {
                continue;
            }
            for i in 0..time_vec.len() {
                let xi = time_vec[i];
                if xi == 0.0 {
                    continue;
                }
                for j in 0..mean_vec.len() {
                    let xj = mean_vec[j];
                    if xj == 0.0 {
                        continue;
                    }
                    target[[slices.time.start + i, slices.mean.start + j]] += weight * xi * xj;
                    target[[slices.mean.start + j, slices.time.start + i]] += weight * xj * xi;
                }
            }
        }
    }

    fn evaluate_exact_newton_joint_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        let mut ll = 0.0;
        let mut gradient = Array1::<f64>::zeros(slices.total);
        let mut hessian = Array2::<f64>::zeros((slices.total, slices.total));
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, survival_hessian) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    1.0,
                    q_exit[row_idx],
                    mu[row_idx],
                    self.latent_sd,
                    false,
                )?;
            let binary = binary_from_log_survival(row_log_survival, self.event_target[row_idx])?;
            ll += wi * binary.log_lik;
            let primary_gradient = binary.grad_scale * &survival_gradient;
            let mut primary_hessian = binary.grad_scale * survival_hessian;
            for a in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                for b in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                    primary_hessian[[a, b]] +=
                        binary.outer_scale * survival_gradient[a] * survival_gradient[b];
                }
            }
            self.add_pullback_primary_gradient(
                &mut gradient,
                row_idx,
                &slices,
                &primary_gradient,
                wi,
            );
            self.add_pullback_primary_hessian(
                &mut hessian,
                row_idx,
                &slices,
                &(wi * primary_hessian),
            );
        }
        Ok((ll, gradient, hessian))
    }

    /// Per-row residuals of the unpenalized NLL with respect to the baseline
    /// time-block offsets `(entry, exit)`.
    ///
    /// The latent-binary deployment likelihood is a monotone scalar transform
    /// `ℓ_bin = b(log S_row)` of the latent-survival row log-survival, so by the
    /// chain rule `∂ℓ_bin/∂q_ch = b'(log S)·∂(log S)/∂q_ch = grad_scale·g_ch`,
    /// where `g_ch` are the `Q_ENTRY`/`Q_EXIT` components of the survival row
    /// primary gradient. The baseline θ enters only the additive entry/exit time
    /// offsets (`q̇_exit` is held at the constant deployment derivative `1`, so
    /// the derivative channel carries no baseline offset and its residual is 0).
    /// Sampleweight-scaled to match the [`OffsetChannelResiduals`] contract.
    pub fn offset_channel_residuals(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<crate::families::survival::OffsetChannelResiduals, String> {
        let n = self.event_target.len();
        if block_states.is_empty() {
            log::warn!(
                "LatentBinaryFamily::offset_channel_residuals: block_states is empty \
                 (degraded fit); returning zero offset residuals (n={n})"
            );
            return Ok(crate::families::survival::OffsetChannelResiduals {
                exit: Array1::<f64>::zeros(n),
                entry: Array1::<f64>::zeros(n),
                derivative: Array1::<f64>::zeros(n),
                right: Array1::<f64>::zeros(n),
            });
        }
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let mut entry = Array1::<f64>::zeros(n);
        let mut exit = Array1::<f64>::zeros(n);
        for row_idx in 0..n {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, _) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    1.0,
                    q_exit[row_idx],
                    mu[row_idx],
                    self.latent_sd,
                    false,
                )?;
            let binary = binary_from_log_survival(row_log_survival, self.event_target[row_idx])?;
            // ∂NLL/∂o_ch = −w · grad_scale · ∂(log S)/∂q_ch.
            entry[row_idx] =
                -wi * binary.grad_scale * survival_gradient[LATENT_SURVIVAL_PRIMARY_Q_ENTRY];
            exit[row_idx] =
                -wi * binary.grad_scale * survival_gradient[LATENT_SURVIVAL_PRIMARY_Q_EXIT];
        }
        Ok(crate::families::survival::OffsetChannelResiduals {
            exit,
            entry,
            derivative: Array1::<f64>::zeros(n),
            // Latent-binary deployment has no interval upper bound; the `R`
            // channel is structurally absent (every row is right-censored).
            right: Array1::<f64>::zeros(n),
        })
    }

    fn exact_newton_joint_hessian_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total {
            return Err(format!(
                "latent binary joint dH direction length mismatch: got {}, expected {}",
                d_beta_flat.len(),
                slices.total
            ));
        }
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, survival_hessian) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    1.0,
                    q_exit[row_idx],
                    mu[row_idx],
                    self.latent_sd,
                    false,
                )?;
            let binary = binary_from_log_survival(row_log_survival, self.event_target[row_idx])?;
            let direction = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
            let third = latent_survival_row_primary_third_contracted(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                1.0,
                q_exit[row_idx],
                mu[row_idx],
                self.latent_sd,
                &direction,
                false,
            )?;
            let g_u = -survival_hessian.dot(&direction);
            let t_u = survival_gradient.dot(&direction);
            let mut primary = binary.grad_scale * third;
            primary.scaled_add(binary.grad_scale_prime * t_u, &survival_hessian);
            for a in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                for b in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                    primary[[a, b]] += binary.outer_scale_prime
                        * t_u
                        * survival_gradient[a]
                        * survival_gradient[b]
                        + binary.outer_scale
                            * (g_u[a] * survival_gradient[b] + survival_gradient[a] * g_u[b]);
                }
            }
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &(wi * primary));
        }
        Ok(out)
    }

    fn exact_newton_joint_hessian_second_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        if d_beta_u_flat.len() != slices.total || d_beta_v_flat.len() != slices.total {
            return Err(format!(
                "latent binary joint d2H direction length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                slices.total
            ));
        }
        let mut out = Array2::<f64>::zeros((slices.total, slices.total));
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, survival_hessian) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    1.0,
                    q_exit[row_idx],
                    mu[row_idx],
                    self.latent_sd,
                    false,
                )?;
            let binary = binary_from_log_survival(row_log_survival, self.event_target[row_idx])?;
            let direction_u = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_u_flat);
            let direction_v = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_v_flat);
            let third_u = latent_survival_row_primary_third_contracted(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                1.0,
                q_exit[row_idx],
                mu[row_idx],
                self.latent_sd,
                &direction_u,
                false,
            )?;
            let third_v = latent_survival_row_primary_third_contracted(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                1.0,
                q_exit[row_idx],
                mu[row_idx],
                self.latent_sd,
                &direction_v,
                false,
            )?;
            let fourth = latent_survival_row_primary_fourth_contracted(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                1.0,
                q_exit[row_idx],
                mu[row_idx],
                self.latent_sd,
                &direction_u,
                &direction_v,
                false,
            )?;
            let g_u = -survival_hessian.dot(&direction_u);
            let g_v = -survival_hessian.dot(&direction_v);
            let g_uv = -third_v.dot(&direction_u);
            let t_u = survival_gradient.dot(&direction_u);
            let t_v = survival_gradient.dot(&direction_v);
            let l_uv = -direction_u.dot(&survival_hessian.dot(&direction_v));
            let c_u = binary.grad_scale_prime * t_u;
            let c_v = binary.grad_scale_prime * t_v;
            let c_uv = binary.grad_scale_second * t_u * t_v + binary.grad_scale_prime * l_uv;
            let o_u = binary.outer_scale_prime * t_u;
            let o_v = binary.outer_scale_prime * t_v;
            let o_uv = binary.outer_scale_second * t_u * t_v + binary.outer_scale_prime * l_uv;
            let mut primary = binary.grad_scale * fourth;
            primary.scaled_add(c_u, &third_v);
            primary.scaled_add(c_v, &third_u);
            primary.scaled_add(c_uv, &survival_hessian);
            for a in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                for b in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                    primary[[a, b]] += o_uv * survival_gradient[a] * survival_gradient[b]
                        + o_v * (g_u[a] * survival_gradient[b] + survival_gradient[a] * g_u[b])
                        + o_u * (g_v[a] * survival_gradient[b] + survival_gradient[a] * g_v[b])
                        + binary.outer_scale
                            * (g_uv[a] * survival_gradient[b]
                                + g_u[a] * g_v[b]
                                + g_v[a] * g_u[b]
                                + survival_gradient[a] * g_uv[b]);
                }
            }
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &(wi * primary));
        }
        Ok(out)
    }
}

/// Shared interface that both `LatentSurvivalFamily` and `LatentBinaryFamily`
/// expose to the joint Hessian workspace.
///
/// The two families produce the same `ExactNewtonJointHessianWorkspace`
/// shape — five of the six workspace methods are pure delegations to a
/// matching family method (dense evaluation, directional derivatives, and the
/// `slices` cache). The only family-specific piece is the per-row matvec body:
/// the survival family iterates over real (entry, exit, ḋ) triples and may
/// carry a log-σ block, while the binary family rewrites the same row kernel
/// through `binary_from_log_survival(·)` to recover the per-row binary
/// gradient/Hessian. That single difference is captured by `ws_matvec_into`;
/// every other method is shared by the generic `LatentHessianWorkspace<F>`
/// below.
trait LatentJointHessianFamily {
    fn ws_joint_slices(&self) -> LatentSurvivalJointSlices;

    fn ws_evaluate_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String>;

    fn ws_dh_directional(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String>;

    fn ws_dh_second_directional(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String>;

    /// Family-specific per-row Hessian matvec body, hoisted out of the
    /// workspace impl. Writes `out := H · v` (with `out.fill(0.0)` already
    /// performed by the caller) using the family's row kernel.
    fn ws_matvec_into(
        &self,
        slices: &LatentSurvivalJointSlices,
        block_states: &[ParameterBlockState],
        v: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<bool, String>;

    /// Family-name fragment used in the workspace's dimension-mismatch error
    /// message, so callers still see "latent survival …" / "latent binary …"
    /// after the workspace impl was unified.
    fn ws_label() -> &'static str;
}

impl LatentJointHessianFamily for LatentSurvivalFamily {
    fn ws_joint_slices(&self) -> LatentSurvivalJointSlices {
        self.joint_slices()
    }

    fn ws_evaluate_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        self.evaluate_exact_newton_joint_dense(block_states)
    }

    fn ws_dh_directional(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_dense(block_states, d_beta_flat)
    }

    fn ws_dh_second_directional(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_dense(
            block_states,
            d_beta_u,
            d_beta_v,
        )
    }

    fn ws_matvec_into(
        &self,
        slices: &LatentSurvivalJointSlices,
        block_states: &[ParameterBlockState],
        v: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<bool, String> {
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let include_log_sigma = slices.log_sigma.is_some();
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row = self.build_row_at(
                row_idx,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                q_right[row_idx],
            )?;
            let (_, _, primary_hessian) = latent_survival_row_primary_gradient_hessian(
                &self.quadctx,
                &row,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                q_right[row_idx],
                mu[row_idx],
                sigma,
                include_log_sigma,
            )?;
            let primary_dir = self.row_primary_direction_from_flat(row_idx, slices, v);
            let primary_hv = primary_hessian.dot(&primary_dir);
            self.add_pullback_primary_gradient(out, row_idx, slices, &primary_hv, wi)?;
        }
        Ok(true)
    }

    fn ws_label() -> &'static str {
        "survival"
    }
}

impl LatentJointHessianFamily for LatentBinaryFamily {
    fn ws_joint_slices(&self) -> LatentSurvivalJointSlices {
        self.joint_slices()
    }

    fn ws_evaluate_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        self.evaluate_exact_newton_joint_dense(block_states)
    }

    fn ws_dh_directional(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_directional_derivative_dense(block_states, d_beta_flat)
    }

    fn ws_dh_second_directional(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_dense(
            block_states,
            d_beta_u,
            d_beta_v,
        )
    }

    fn ws_matvec_into(
        &self,
        slices: &LatentSurvivalJointSlices,
        block_states: &[ParameterBlockState],
        v: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<bool, String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        for row_idx in 0..self.event_target.len() {
            let wi = self.weights[row_idx];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, survival_hessian) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    1.0,
                    q_exit[row_idx],
                    mu[row_idx],
                    self.latent_sd,
                    false,
                )?;
            let binary = binary_from_log_survival(row_log_survival, self.event_target[row_idx])?;
            let primary_dir = self.row_primary_direction_from_flat(row_idx, slices, v);
            let mut primary_hv = binary.grad_scale * survival_hessian.dot(&primary_dir);
            let outer_dot = survival_gradient.dot(&primary_dir);
            for a in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                primary_hv[a] += binary.outer_scale * survival_gradient[a] * outer_dot;
            }
            self.add_pullback_primary_gradient(out, row_idx, slices, &primary_hv, wi);
        }
        Ok(true)
    }

    fn ws_label() -> &'static str {
        "binary"
    }
}

/// Joint exact-Newton Hessian workspace shared by `LatentSurvivalFamily` and
/// `LatentBinaryFamily`. The two families plug into the workspace via
/// `LatentJointHessianFamily`; this struct holds the shared bookkeeping
/// (block states + cached slices) and routes every trait method either through
/// a thin family delegation or through the family's `ws_matvec_into` row
/// kernel.
struct LatentHessianWorkspace<F: LatentJointHessianFamily> {
    family: F,
    block_states: Vec<ParameterBlockState>,
    slices: LatentSurvivalJointSlices,
}

impl<F: LatentJointHessianFamily> LatentHessianWorkspace<F> {
    fn new(family: F, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = family.ws_joint_slices();
        Self {
            family,
            block_states,
            slices,
        }
    }
}

impl<F> ExactNewtonJointHessianWorkspace for LatentHessianWorkspace<F>
where
    F: LatentJointHessianFamily + Send + Sync + 'static,
{
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        self.family
            .ws_evaluate_dense(&self.block_states)
            .map(|(_, _, hessian)| Some(hessian))
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        let mut out = Array1::<f64>::zeros(self.slices.total);
        self.hessian_matvec_into(v, &mut out)?;
        Ok(Some(out))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        if v.len() != self.slices.total || out.len() != self.slices.total {
            return Err(format!(
                "latent {} Hessian matvec dimension mismatch: v={} out={} expected={}",
                F::ws_label(),
                v.len(),
                out.len(),
                self.slices.total
            ));
        }
        out.fill(0.0);
        self.family
            .ws_matvec_into(&self.slices, &self.block_states, v, out)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        let dense = self.family.ws_evaluate_dense(&self.block_states)?.2;
        Ok(Some(dense.diag().to_owned()))
    }

    fn directional_derivative(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .ws_dh_directional(&self.block_states, d_beta_flat)
            .map(Some)
    }

    fn second_directional_derivative(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.family
            .ws_dh_second_directional(&self.block_states, d_beta_u, d_beta_v)
            .map(Some)
    }
}

type LatentSurvivalHessianWorkspace = LatentHessianWorkspace<LatentSurvivalFamily>;
type LatentBinaryHessianWorkspace = LatentHessianWorkspace<LatentBinaryFamily>;

impl CustomFamily for LatentSurvivalFamily {
    // Latent survival fits keep the self-limiting Jeffreys/Firth curvature
    // active for their under-identification regime. The trait default flipped to
    // OFF in gam#1395 (flat-prior exact-Newton objective); opt back in here.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// Engage the inner self-vanishing Levenberg–Marquardt μ on a full-rank but
    /// indefinite / ill-conditioned penalized joint Hessian, mirroring the
    /// sibling [`SurvivalMarginalSlopeFamily`]. Interval-censored rows contribute
    /// `ℓ = log[S(L) − S(R)]`, the log of a DIFFERENCE of two survival kernels:
    /// unlike the log-concave exact-event / right-censored contributions, its
    /// per-row Hessian is legitimately INDEFINITE away from the optimum, so the
    /// coupled exact-joint penalized Hessian on the constrained (monotone-cone)
    /// time block can be full-rank (`nullity == 0`) yet indefinite or severely
    /// ill-conditioned at the cold-start seed. The constrained-QP path already
    /// REFLECTS negative-curvature modes to `|λ|` (a convex modified-Newton
    /// model), but with this gate OFF it adds NO diagonal floor on a full-rank
    /// ill-conditioned reflected model, so the trust-region Newton oscillates on
    /// the near-singular mode and stalls out the inner budget before any KKT
    /// snapshot is taken ("exited the joint Newton path before convergence — no
    /// math snapshot"). Arming the gate adds the SAME self-vanishing μ
    /// (∝ the projected KKT residual `‖∇ℓ − Sβ + ∇Φ‖` → 0 at the fixed point) the
    /// marginal-slope survival inner relies on, so the step is a well-damped
    /// modified-Newton descent that converges, while the converged β̂ is the
    /// EXACT unconditioned optimum (μ → 0 there) — zero REML/LAML bias, exact
    /// gradient unchanged.
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // `evaluate_exact_newton_joint_dense` builds a fully dense joint
        // Hessian over (Σ p_b)² across time, mean, and optional log-σ blocks
        // via per-row pullback of the latent-survival primary kernel.
        crate::custom_family::joint_coupled_coefficient_hessian_cost(
            self.event_target.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (ll, joint_gradient, hess_time, hess_mean, hess_log_sigma) =
            self.evaluate_exact_newton_block_diagonals(block_states)?;
        let block_ranges = self.joint_block_ranges();
        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![block_ranges[0].clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![block_ranges[1].clone()]).to_owned(),
                hessian: SymmetricMatrix::Dense(hess_mean),
            },
        ];
        if let (Some(range), Some(hessian)) = (block_ranges.get(2).cloned(), hess_log_sigma) {
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: joint_gradient.slice(s![range]).to_owned(),
                hessian: SymmetricMatrix::Dense(hessian),
            });
        }
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let latent_sd = self.latent_sd(block_states)?;
        let n = self.event_target.len();
        // Per-row latent-survival jet + log-lik contribution. Independent
        // across rows; sum via parallel reduce. `?` propagation happens
        // through a Result-collecting fold.
        let contributions: Result<Vec<f64>, String> = (0..n)
            .into_par_iter()
            .map(|i| -> Result<f64, String> {
                let wi = self.weights[i];
                if wi <= MIN_WEIGHT {
                    return Ok(0.0);
                }
                let row = self.build_row_at(i, q_entry[i], q_exit[i], qdot_exit[i], q_right[i])?;
                let jet = LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], latent_sd)
                    .map_err(|e| format!("LatentSurvivalFamily row {i}: {e}"))?;
                Ok(wi * jet.log_lik)
            })
            .collect();
        Ok(contributions?.into_iter().sum())
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx == Self::BLOCK_TIME {
            Ok(self.time_linear_constraints.clone())
        } else {
            Ok(None)
        }
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.evaluate_exact_newton_joint_dense(block_states)
            .map(|(_, _, hessian)| Some(hessian))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        Ok(Some(Arc::new(LatentSurvivalHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        ))))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        self.evaluate_exact_newton_joint_gradient_dense(block_states)
            .map(|(log_likelihood, gradient)| {
                Some(ExactNewtonJointGradientEvaluation {
                    log_likelihood,
                    gradient,
                })
            })
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_dense(block_states, d_beta_flat)
            .map(Some)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_dense(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
        .map(Some)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
}

impl CustomFamily for LatentBinaryFamily {
    // Latent binary fits have a separation regime; keep the self-limiting
    // Jeffreys/Firth curvature active. The trait default flipped to OFF in
    // gam#1395 (flat-prior exact-Newton objective); opt back in here.
    fn joint_jeffreys_term_required(&self) -> bool {
        true
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }

    /// Same self-vanishing Levenberg–Marquardt gate as
    /// [`LatentSurvivalFamily`]: the latent-binary deployment shares the
    /// constrained (monotone-cone) coupled time block, so a full-rank but
    /// ill-conditioned penalized joint Hessian at the cold-start seed must get
    /// the self-vanishing μ floor rather than oscillating the constrained-QP
    /// trust region into a snapshot-less stall. μ → 0 at the fixed point, so the
    /// converged β̂ is exact (no REML/LAML bias).
    fn levenberg_on_ill_conditioning(&self) -> bool {
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        crate::custom_family::joint_coupled_coefficient_hessian_cost(
            self.event_target.len() as u64,
            specs,
        )
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let n = self.event_target.len();
        let p_time = self.x_time_exit.ncols();
        let p_mean = self.x_mean.ncols();

        let mut ll = 0.0;
        let mut grad_time = Array1::<f64>::zeros(p_time);
        let mut hess_time = Array2::<f64>::zeros((p_time, p_time));
        let mut grad_mean = Array1::<f64>::zeros(p_mean);
        let mut hess_mean = Array2::<f64>::zeros((p_mean, p_mean));
        // Reusable 1-row buffer for x_mean so we avoid allocating a fresh
        // Array2<f64> on every iteration via try_row_chunk(i..i+1).
        let mut mean_row_buf = Array2::<f64>::zeros((1, p_mean));

        for i in 0..n {
            let wi = self.weights[i];
            if wi <= MIN_WEIGHT {
                continue;
            }
            if !(q_entry[i].is_finite() && q_exit[i].is_finite() && mu[i].is_finite()) {
                return Err(format!(
                    "latent-binary row {i} contains non-finite predictors: q_entry={}, q_exit={}, mu={}",
                    q_entry[i], q_exit[i], mu[i]
                ));
            }
            let row = self.build_right_censored_row_at(i, q_entry[i], q_exit[i])?;
            let survival_jet =
                LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], self.latent_sd)
                    .map_err(|e| format!("LatentBinaryFamily row {i}: {e}"))?;
            let binary = binary_from_log_survival(survival_jet.log_lik, self.event_target[i])?;
            ll += wi * binary.log_lik;

            self.x_mean
                .row_chunk_into(i..i + 1, mean_row_buf.view_mut())
                .map_err(|e| format!("LatentBinaryFamily row {i} mean row_chunk: {e}"))?;
            let mean_vec = mean_row_buf.row(0);
            let mean_grad_scale = wi * binary.grad_scale * survival_jet.score;
            for j in 0..p_mean {
                grad_mean[j] += mean_grad_scale * mean_vec[j];
            }
            let mean_neg_hess = wi
                * (binary.neg_hess_scale * survival_jet.neg_hessian
                    + binary.outer_scale * survival_jet.score * survival_jet.score);
            dense_outer_accumulate(&mut hess_mean, mean_neg_hess, mean_vec);

            let time_jet =
                latent_survival_time_jet(&self.quadctx, &row, 0.0, mu[i], self.latent_sd)?;
            let t_entry = self.x_time_entry.row(i);
            let t_exit = self.x_time_exit.row(i);
            for j in 0..p_time {
                grad_time[j] += wi
                    * binary.grad_scale
                    * (time_jet.grad_entry * t_entry[j] + time_jet.grad_exit * t_exit[j]);
            }
            dense_outer_accumulate(
                &mut hess_time,
                wi * binary.neg_hess_scale * time_jet.neg_hess_entry,
                t_entry,
            );
            dense_outer_accumulate(
                &mut hess_time,
                wi * binary.neg_hess_scale * time_jet.neg_hess_exit,
                t_exit,
            );
            if binary.outer_scale != 0.0 {
                dense_outer_accumulate(
                    &mut hess_time,
                    wi * binary.outer_scale * time_jet.grad_entry * time_jet.grad_entry,
                    t_entry,
                );
                dense_outer_accumulate(
                    &mut hess_time,
                    wi * binary.outer_scale * time_jet.grad_exit * time_jet.grad_exit,
                    t_exit,
                );
                dense_symmetric_cross_accumulate(
                    &mut hess_time,
                    wi * binary.outer_scale * time_jet.grad_entry * time_jet.grad_exit,
                    t_entry,
                    t_exit,
                );
            }
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad_time,
                    hessian: SymmetricMatrix::Dense(hess_time),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad_mean,
                    hessian: SymmetricMatrix::Dense(hess_mean),
                },
            ],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let mut ll = 0.0;
        for i in 0..self.event_target.len() {
            let wi = self.weights[i];
            if wi <= MIN_WEIGHT {
                continue;
            }
            let row = self.build_right_censored_row_at(i, q_entry[i], q_exit[i])?;
            let survival_jet =
                LatentSurvivalRowJet::evaluate(&self.quadctx, &row, mu[i], self.latent_sd)
                    .map_err(|e| format!("LatentBinaryFamily row {i}: {e}"))?;
            ll +=
                wi * binary_from_log_survival(survival_jet.log_lik, self.event_target[i])?.log_lik;
        }
        Ok(ll)
    }

    fn block_linear_constraints(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx == Self::BLOCK_TIME {
            Ok(self.time_linear_constraints.clone())
        } else {
            Ok(None)
        }
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        self.evaluate_exact_newton_joint_dense(block_states)
            .map(|(_, _, hessian)| Some(hessian))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        Ok(Some(Arc::new(LatentBinaryHessianWorkspace::new(
            self.clone(),
            block_states.to_vec(),
        ))))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        block_specs: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        self.evaluate_exact_newton_joint_dense(block_states)
            .map(|(log_likelihood, gradient, _)| {
                Some(ExactNewtonJointGradientEvaluation {
                    log_likelihood,
                    gradient,
                })
            })
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_dense(block_states, d_beta_flat)
            .map(Some)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_second_directional_derivative_dense(
            block_states,
            d_beta_u_flat,
            d_beta_v_flat,
        )
        .map(Some)
    }

    fn requires_joint_outer_hyper_path(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::custom_family::BlockWorkingSet;
    use crate::matrix::DenseDesignMatrix;
    use ndarray::array;

    fn learnable_sigma_test_family() -> LatentSurvivalFamily {
        LatentSurvivalFamily {
            event_target: array![1u8, 0u8],
            weights: array![1.0, 0.7],
            latent_sd_fixed: None,
            hazard_loading: HazardLoading::LoadedVsUnloaded,
            unloaded_mass_entry: array![0.02, 0.03],
            unloaded_mass_exit: array![0.05, 0.08],
            unloaded_hazard_exit: array![0.04, 0.0],
            x_time_entry: array![[1.0, -0.2], [0.4, 0.7]],
            x_time_exit: array![[1.3, 0.1], [0.9, 1.0]],
            x_time_derivative_exit: array![[0.8, 0.4], [0.6, 0.5]],
            x_time_right: array![[1.3, 0.1], [0.9, 1.0]],
            time_offset_right: Array1::zeros(2),
            unloaded_mass_right: Array1::zeros(2),
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0, -0.3], [0.2, 0.9]])),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        }
    }

    fn learnable_sigma_test_joint_beta() -> Array1<f64> {
        array![0.15, 0.25, 0.1, -0.15, 0.35_f64.ln()]
    }

    fn survival_stress_test_family(n: usize) -> LatentSurvivalFamily {
        LatentSurvivalFamily {
            event_target: Array1::from_iter((0..n).map(|i| if i % 3 == 0 { 1u8 } else { 0u8 })),
            weights: Array1::from_iter((0..n).map(|i| 0.55 + 0.03 * ((i % 7) as f64))),
            latent_sd_fixed: None,
            hazard_loading: HazardLoading::LoadedVsUnloaded,
            unloaded_mass_entry: Array1::from_iter(
                (0..n).map(|i| 0.015 + 0.0015 * ((i % 11) as f64)),
            ),
            unloaded_mass_exit: Array1::from_iter((0..n).map(|i| 0.06 + 0.002 * ((i % 13) as f64))),
            unloaded_hazard_exit: Array1::from_iter((0..n).map(|i| {
                if i % 4 == 0 {
                    0.018 + 0.001 * ((i % 5) as f64)
                } else {
                    0.0
                }
            })),
            x_time_entry: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.2 + 0.03 * ((i + 2 * j) % 9) as f64 - if j == 1 { 0.12 } else { 0.0 }
            }),
            x_time_exit: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.35 + 0.025 * ((2 * i + j) % 10) as f64 - if j == 2 { 0.08 } else { 0.0 }
            }),
            x_time_derivative_exit: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.45 + 0.015 * ((i + 3 * j) % 8) as f64
            }),
            x_time_right: Array2::from_shape_fn((n, 4), |(i, j)| {
                0.35 + 0.025 * ((2 * i + j) % 10) as f64 - if j == 2 { 0.08 } else { 0.0 }
            }),
            time_offset_right: Array1::zeros(n),
            unloaded_mass_right: Array1::zeros(n),
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_shape_fn(
                (n, 3),
                |(i, j)| 0.1 + 0.04 * ((3 * i + j) % 7) as f64 - if j == 0 { 0.18 } else { 0.0 },
            ))),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        }
    }

    fn survival_stress_test_joint_beta() -> Array1<f64> {
        array![0.18, 0.11, 0.07, 0.13, -0.09, 0.05, 0.12, 0.42_f64.ln()]
    }

    fn latent_survival_states_from_joint_beta(
        family: &LatentSurvivalFamily,
        joint_beta: &Array1<f64>,
    ) -> Vec<ParameterBlockState> {
        let slices = family.joint_slices();
        let n = family.event_target.len();
        let beta_time = joint_beta.slice(s![slices.time.clone()]).to_owned();
        let beta_mean = joint_beta.slice(s![slices.mean.clone()]).to_owned();

        let mut eta_time = Array1::<f64>::zeros(3 * n);
        eta_time
            .slice_mut(s![0..n])
            .assign(&crate::faer_ndarray::fast_av(
                &family.x_time_entry,
                &beta_time,
            ));
        eta_time
            .slice_mut(s![n..2 * n])
            .assign(&crate::faer_ndarray::fast_av(
                &family.x_time_exit,
                &beta_time,
            ));
        eta_time
            .slice_mut(s![2 * n..3 * n])
            .assign(&crate::faer_ndarray::fast_av(
                &family.x_time_derivative_exit,
                &beta_time,
            ));

        let mut states = vec![
            ParameterBlockState {
                beta: beta_time,
                eta: eta_time,
            },
            ParameterBlockState {
                beta: beta_mean.clone(),
                eta: family.x_mean.dot(&beta_mean),
            },
        ];
        if let Some(log_sigma) = slices.log_sigma {
            let beta_log_sigma = array![joint_beta[log_sigma.start]];
            states.push(ParameterBlockState {
                beta: beta_log_sigma.clone(),
                eta: beta_log_sigma,
            });
        }
        states
    }

    fn max_relative_array1(left: &Array1<f64>, right: &Array1<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| (l - r).abs() / l.abs().max(r.abs()).max(1e-12))
            .fold(0.0_f64, f64::max)
    }

    fn max_relative_array2(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        left.iter()
            .zip(right.iter())
            .map(|(l, r)| (l - r).abs() / l.abs().max(r.abs()).max(1e-12))
            .fold(0.0_f64, f64::max)
    }

    fn frobenius_relative_array2(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
        let mut diff2 = 0.0_f64;
        let mut scale2 = 0.0_f64;
        for (l, r) in left.iter().zip(right.iter()) {
            let d = l - r;
            diff2 += d * d;
            scale2 += l * l + r * r;
        }
        diff2.sqrt() / scale2.sqrt().max(1e-12)
    }

    fn latent_survival_row_loglik_from_primary(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        primary: &Array1<f64>,
    ) -> f64 {
        let q_entry = primary[LATENT_SURVIVAL_PRIMARY_Q_ENTRY];
        let q_exit = primary[LATENT_SURVIVAL_PRIMARY_Q_EXIT];
        let qdot_exit = primary[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT];
        let q_right = primary[LATENT_SURVIVAL_PRIMARY_Q_RIGHT];
        let mu = primary[LATENT_SURVIVAL_PRIMARY_MU];
        let sigma = primary[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA].exp();
        latent_survival_row_primary_gradient_hessian(
            quadctx, row, q_entry, q_exit, qdot_exit, q_right, mu, sigma, true,
        )
        .expect("row primary evaluation")
        .0
    }

    fn latent_test_specs(n: usize, block_dims: &[(&str, usize)]) -> Vec<ParameterBlockSpec> {
        block_dims
            .iter()
            .map(|(name, p)| ParameterBlockSpec {
                name: (*name).to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((n, *p)))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            })
            .collect()
    }

    fn fixed_sigma_binary_test_family() -> LatentBinaryFamily {
        LatentBinaryFamily {
            event_target: array![1u8, 0u8],
            weights: array![1.0, 0.7],
            latent_sd: 0.35,
            hazard_loading: HazardLoading::LoadedVsUnloaded,
            unloaded_mass_entry: array![0.02, 0.03],
            unloaded_mass_exit: array![0.05, 0.08],
            x_time_entry: array![[1.0, -0.2], [0.4, 0.7]],
            x_time_exit: array![[1.3, 0.1], [0.9, 1.0]],
            x_mean: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0, -0.3], [0.2, 0.9]])),
            time_linear_constraints: None,
            quadctx: Arc::new(QuadratureContext::new()),
        }
    }

    fn latent_binary_states_from_joint_beta(
        family: &LatentBinaryFamily,
        joint_beta: &Array1<f64>,
    ) -> Vec<ParameterBlockState> {
        let slices = family.joint_slices();
        let n = family.event_target.len();
        let beta_time = joint_beta.slice(s![slices.time.clone()]).to_owned();
        let beta_mean = joint_beta.slice(s![slices.mean.clone()]).to_owned();

        let mut eta_time = Array1::<f64>::zeros(3 * n);
        eta_time
            .slice_mut(s![0..n])
            .assign(&crate::faer_ndarray::fast_av(
                &family.x_time_entry,
                &beta_time,
            ));
        eta_time
            .slice_mut(s![n..2 * n])
            .assign(&crate::faer_ndarray::fast_av(
                &family.x_time_exit,
                &beta_time,
            ));

        vec![
            ParameterBlockState {
                beta: beta_time,
                eta: eta_time,
            },
            ParameterBlockState {
                beta: beta_mean.clone(),
                eta: family.x_mean.dot(&beta_mean),
            },
        ]
    }

    // --- shared latent-interval validation engine: parity / contract tests ---

    use crate::families::survival::location_scale::{TimeBlockInput, TimeBlockMonotonicity};

    /// Minimal, structurally valid `TimeBlockInput` for `n` rows and `p_time`
    /// columns, used to exercise the shared validation driver without standing
    /// up a full term-collection design.
    fn validation_time_block(n: usize, p_time: usize) -> TimeBlockInput {
        let design = |fill: f64| {
            DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem(
                (n, p_time),
                fill,
            )))
        };
        TimeBlockInput {
            design_entry: design(0.1),
            design_exit: design(0.2),
            design_derivative_exit: design(0.3),
            offset_entry: Array1::zeros(n),
            offset_exit: Array1::zeros(n),
            derivative_offset_exit: Array1::zeros(n),
            time_monotonicity: TimeBlockMonotonicity::EnforcedByCoordinateCone,
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: None,
            initial_beta: None,
        }
    }

    fn empty_meanspec() -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: Vec::new(),
            random_effect_terms: Vec::new(),
            smooth_terms: Vec::new(),
        }
    }

    /// A valid two-row latent-survival term spec (one exact event under loaded
    /// hazard, one right-censored row).
    fn valid_survival_spec(n: usize, p_time: usize) -> LatentSurvivalTermSpec {
        LatentSurvivalTermSpec {
            age_entry: Array1::zeros(n),
            age_exit: Array1::from_elem(n, 1.0),
            event_target: Array1::from_shape_fn(n, |i| (i % 2) as u8),
            weights: Array1::from_elem(n, 1.0),
            derivative_guard: 0.0,
            time_block: validation_time_block(n, p_time),
            time_design_right: None,
            time_offset_right: None,
            unloaded_mass_entry: Array1::from_elem(n, 0.01),
            unloaded_mass_exit: Array1::from_elem(n, 0.05),
            unloaded_mass_right: Array1::zeros(0),
            unloaded_hazard_exit: Array1::from_elem(n, 0.02),
            meanspec: empty_meanspec(),
            mean_offset: Array1::zeros(n),
        }
    }

    /// A valid latent-binary term spec mirroring `valid_survival_spec` but
    /// without the per-row unloaded hazard.
    fn valid_binary_spec(n: usize, p_time: usize) -> LatentBinaryTermSpec {
        LatentBinaryTermSpec {
            age_entry: Array1::zeros(n),
            age_exit: Array1::from_elem(n, 1.0),
            event_target: Array1::from_shape_fn(n, |i| (i % 2) as u8),
            weights: Array1::from_elem(n, 1.0),
            derivative_guard: 0.0,
            time_block: validation_time_block(n, p_time),
            unloaded_mass_entry: Array1::from_elem(n, 0.01),
            unloaded_mass_exit: Array1::from_elem(n, 0.05),
            meanspec: empty_meanspec(),
            mean_offset: Array1::zeros(n),
        }
    }

    fn loaded_frailty() -> FrailtySpec {
        FrailtySpec::HazardMultiplier {
            sigma_fixed: Some(0.3),
            loading: HazardLoading::LoadedVsUnloaded,
        }
    }

    /// Both adapters route through the shared `validate_latent_interval_inputs`
    /// engine, but each must still emit its own context prefix and (for the
    /// size-mismatch / unloaded-decomposition diagnostics) the hazard-aware vs
    /// mass-only message variant. This pins the byte-for-byte contract the
    /// unification had to preserve, the property the issue's "old vs new
    /// validation errors" parity test guards.
    #[test]
    fn latent_interval_validation_parity_across_models() {
        let n = 2;
        let p_time = 2;
        let data = Array2::<f64>::zeros((n, 3));

        // 1. A clean spec validates and round-trips the resolved sigma.
        //    Survival keeps the (possibly learnable) Option; binary unwraps to
        //    the fixed scalar.
        let surv_sigma = validate_latent_survival_inputs(
            data.view(),
            &valid_survival_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect("valid survival spec must validate");
        assert_eq!(surv_sigma, Some(0.3));
        let bin_sigma = validate_latent_binary_inputs(
            data.view(),
            &valid_binary_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect("valid binary spec must validate");
        assert_eq!(bin_sigma, 0.3);

        // 2. Empty data: shared driver, per-model context prefix.
        let empty = Array2::<f64>::zeros((0, 3));
        let surv_empty = validate_latent_survival_inputs(
            empty.view(),
            &valid_survival_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect_err("empty data must be rejected");
        assert_eq!(
            surv_empty.to_string(),
            "latent-survival requires a non-empty dataset"
        );
        let bin_empty = validate_latent_binary_inputs(
            empty.view(),
            &valid_binary_spec(n, p_time),
            &loaded_frailty(),
        )
        .expect_err("empty data must be rejected");
        assert_eq!(
            bin_empty.to_string(),
            "latent-binary requires a non-empty dataset"
        );

        // 3. Size mismatch: survival's message carries `unloaded_hazard=`,
        //    binary's does not. This is the one shape that distinguishes the
        //    two row views feeding the shared driver.
        let mut surv_bad = valid_survival_spec(n, p_time);
        surv_bad.weights = Array1::from_elem(n + 1, 1.0);
        let surv_size = validate_latent_survival_inputs(data.view(), &surv_bad, &loaded_frailty())
            .expect_err("size mismatch must be rejected");
        let surv_msg = surv_size.to_string();
        assert!(
            surv_msg.starts_with("latent-survival size mismatch")
                && surv_msg.contains("unloaded_hazard="),
            "survival size-mismatch message must include unloaded_hazard: {surv_msg}"
        );
        let mut bin_bad = valid_binary_spec(n, p_time);
        bin_bad.weights = Array1::from_elem(n + 1, 1.0);
        let bin_size = validate_latent_binary_inputs(data.view(), &bin_bad, &loaded_frailty())
            .expect_err("size mismatch must be rejected");
        let bin_msg = bin_size.to_string();
        assert!(
            bin_msg.starts_with("latent-binary size mismatch")
                && !bin_msg.contains("unloaded_hazard"),
            "binary size-mismatch message must omit unloaded_hazard: {bin_msg}"
        );

        // 4. Invalid unloaded decomposition: survival reports `exit_hazard=`,
        //    binary reports only the two masses.
        let mut surv_neg_hazard = valid_survival_spec(n, p_time);
        surv_neg_hazard.unloaded_hazard_exit[0] = -1.0;
        let surv_decomp =
            validate_latent_survival_inputs(data.view(), &surv_neg_hazard, &loaded_frailty())
                .expect_err("negative unloaded hazard must be rejected");
        assert_eq!(
            surv_decomp.to_string(),
            "latent-survival row 1 has invalid unloaded hazard decomposition: entry_mass=0.01, exit_mass=0.05, exit_hazard=-1"
        );
        let mut bin_bad_mass = valid_binary_spec(n, p_time);
        bin_bad_mass.unloaded_mass_exit[0] = 0.0; // exit < entry
        let bin_decomp =
            validate_latent_binary_inputs(data.view(), &bin_bad_mass, &loaded_frailty())
                .expect_err("non-monotone unloaded mass must be rejected");
        assert_eq!(
            bin_decomp.to_string(),
            "latent-binary row 1 has invalid unloaded mass decomposition: entry_mass=0.01, exit_mass=0"
        );

        // 5. Per-row interval/event/weight diagnostics share one engine, so an
        //    identical invalid input yields identical (modulo prefix) text.
        let mut surv_event = valid_survival_spec(n, p_time);
        surv_event.event_target[1] = 7;
        let surv_event_err =
            validate_latent_survival_inputs(data.view(), &surv_event, &loaded_frailty())
                .expect_err("invalid event target must be rejected");
        assert_eq!(
            surv_event_err.to_string(),
            "latent-survival row 2 has invalid event target 7; expected 0 or 1"
        );
        let mut bin_event = valid_binary_spec(n, p_time);
        bin_event.event_target[1] = 7;
        let bin_event_err =
            validate_latent_binary_inputs(data.view(), &bin_event, &loaded_frailty())
                .expect_err("invalid event target must be rejected");
        assert_eq!(
            bin_event_err.to_string(),
            "latent-binary row 2 has invalid event target 7; expected 0 or 1"
        );

        // 6. Frailty policy divergence: survival accepts a learnable scale
        //    (`sigma_fixed = None` ⇒ `Ok(None)`), binary rejects it.
        let learnable = FrailtySpec::HazardMultiplier {
            sigma_fixed: None,
            loading: HazardLoading::LoadedVsUnloaded,
        };
        let surv_learnable = validate_latent_survival_inputs(
            data.view(),
            &valid_survival_spec(n, p_time),
            &learnable,
        )
        .expect("survival accepts a learnable latent scale");
        assert_eq!(surv_learnable, None);
        let bin_learnable =
            validate_latent_binary_inputs(data.view(), &valid_binary_spec(n, p_time), &learnable)
                .expect_err("binary requires a fixed latent scale");
        assert_eq!(
            bin_learnable.to_string(),
            "latent-binary currently requires a fixed hazard-multiplier sigma"
        );

        // 7. The time-block shape check is owned by the shared driver: a
        //    column-count mismatch is reported with the per-model prefix.
        let mut surv_time_bad = valid_survival_spec(n, p_time);
        surv_time_bad.time_block.design_entry = DesignMatrix::Dense(DenseDesignMatrix::from(
            Array2::from_elem((n, p_time + 1), 0.1),
        ));
        let surv_time_err =
            validate_latent_survival_inputs(data.view(), &surv_time_bad, &loaded_frailty())
                .expect_err("time block column mismatch must be rejected");
        assert!(
            surv_time_err
                .to_string()
                .starts_with("latent-survival time block column mismatch"),
            "unexpected survival time-block message: {surv_time_err}"
        );
    }

    #[test]
    fn latent_survival_coefficient_cost_uses_joint_coupled_formula() {
        // `evaluate_exact_newton_joint_dense` builds a fully dense joint
        // Hessian over (Σ p_b)² across the time, mean, and log-σ blocks via
        // per-row pullback of the latent-survival primary kernel. The override
        // must reflect that joint coupling rather than the block-diagonal
        // default.
        let family = learnable_sigma_test_family();
        let n = family.event_target.len() as u64;
        let p_time = 2u64;
        let p_mean = 2u64;
        let p_log_sigma = 1u64;
        let specs = vec![
            ParameterBlockSpec {
                name: "time".to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((
                    n as usize,
                    p_time as usize,
                )))),
                offset: Array1::zeros(n as usize),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
            ParameterBlockSpec {
                name: "mean".to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((
                    n as usize,
                    p_mean as usize,
                )))),
                offset: Array1::zeros(n as usize),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((
                    n as usize,
                    p_log_sigma as usize,
                )))),
                offset: Array1::zeros(n as usize),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: None,
                gauge_priority: 100,
                jacobian_callback: None,
                stacked_design: None,
                stacked_offset: None,
            },
        ];
        let p_total = p_time + p_mean + p_log_sigma;
        let expected_joint = n * p_total * p_total;
        let expected_block_diag =
            n * (p_time * p_time + p_mean * p_mean + p_log_sigma * p_log_sigma);
        assert_eq!(family.coefficient_hessian_cost(&specs), expected_joint);
        // Cross-block fill (time–mean, time–log_sigma, mean–log_sigma) makes
        // the joint cost strictly larger than the block-diagonal default.
        assert!(expected_joint > expected_block_diag);
    }

    #[test]
    fn latent_family_planner_keeps_outer_hessian_at_large_n() {
        use crate::families::custom_family::custom_family_outer_derivatives;
        use crate::solver::rho_optimizer::{DeclaredHessianForm, Derivative};

        let options = BlockwiseFitOptions::default();
        let large_n = 50_001;

        let survival = learnable_sigma_test_family();
        let survival_specs =
            latent_test_specs(large_n, &[("time", 2), ("mean", 2), ("log_sigma", 1)]);
        let (surv_grad, surv_hess) =
            custom_family_outer_derivatives(&survival, &survival_specs, &options);
        assert_eq!(surv_grad, Derivative::Analytic);
        assert_eq!(surv_hess, DeclaredHessianForm::Either);

        let binary = fixed_sigma_binary_test_family();
        let binary_specs = latent_test_specs(large_n, &[("time", 2), ("mean", 2)]);
        let (bin_grad, bin_hess) =
            custom_family_outer_derivatives(&binary, &binary_specs, &options);
        assert_eq!(bin_grad, Derivative::Analytic);
        assert_eq!(bin_hess, DeclaredHessianForm::Either);
    }

    #[test]
    fn latent_families_arm_self_vanishing_levenberg_on_ill_conditioning() {
        // Regression guard for #1108. The interval-censored row contribution
        // `ℓ = log[S(L) − S(R)]` is the log of a DIFFERENCE of survival kernels and
        // is legitimately NON-concave (indefinite per-row Hessian) away from the
        // optimum; on the constrained (monotone-cone) coupled time block this can
        // make the penalized joint Hessian full-rank yet indefinite / severely
        // ill-conditioned at the cold-start seed. The coupled exact-joint inner
        // solver only adds the self-vanishing Levenberg–Marquardt diagonal floor
        // (the cure for a full-rank ill-conditioned reflected QP that otherwise
        // oscillates the trust region into a snapshot-less stall) when the family
        // opts in via `levenberg_on_ill_conditioning()`. Both latent families MUST
        // keep this armed (the default is `false`, which leaves the interval inner
        // solve diverging with "exited the joint Newton path before convergence").
        assert!(
            learnable_sigma_test_family().levenberg_on_ill_conditioning(),
            "LatentSurvivalFamily must arm the self-vanishing Levenberg floor so the \
             indefinite interval-censored joint Hessian converges (see #1108)"
        );
        assert!(
            fixed_sigma_binary_test_family().levenberg_on_ill_conditioning(),
            "LatentBinaryFamily must arm the self-vanishing Levenberg floor on its \
             constrained coupled time block (see #1108)"
        );
    }

    #[test]
    fn latent_binary_exact_joint_hessian_and_workspace_matvec_match_fd() {
        let family = fixed_sigma_binary_test_family();
        let beta = array![0.15, 0.25, 0.1, -0.15];
        let states = latent_binary_states_from_joint_beta(&family, &beta);
        let h = 1e-6;

        let analytic_hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("analytic latent binary joint hessian evaluation")
            .expect("latent binary should expose exact joint hessian");

        for j in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[j] += h;
            let gradient_plus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_binary_states_from_joint_beta(&family, &beta_plus),
                    &[],
                )
                .expect("joint gradient plus")
                .expect("joint gradient should exist")
                .gradient;

            let mut beta_minus = beta.clone();
            beta_minus[j] -= h;
            let gradient_minus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_binary_states_from_joint_beta(&family, &beta_minus),
                    &[],
                )
                .expect("joint gradient minus")
                .expect("joint gradient should exist")
                .gradient;

            let fd_column = -((&gradient_plus - &gradient_minus) / (2.0 * h));
            let analytic_column = analytic_hessian.column(j).to_owned();
            let rel = max_relative_array1(&analytic_column, &fd_column);
            assert!(
                rel < 5e-4,
                "latent binary joint Hessian column {j} mismatch: rel={rel}, analytic={analytic_column:?}, fd={fd_column:?}"
            );
        }

        let workspace = family
            .exact_newton_joint_hessian_workspace(&states, &[])
            .expect("latent binary hessian workspace")
            .expect("workspace should exist");
        let direction = array![0.4, -0.2, 0.3, 0.1];
        let hv = workspace
            .hessian_matvec(&direction)
            .expect("workspace matvec")
            .expect("workspace should support matvec");
        let dense_hv = analytic_hessian.dot(&direction);
        assert!(
            max_relative_array1(&hv, &dense_hv) < 1e-12,
            "latent binary workspace HVP mismatch: hv={hv:?}, dense={dense_hv:?}"
        );

        let dh = workspace
            .directional_derivative(&direction)
            .expect("workspace dH")
            .expect("workspace should support dH");
        let fd_step = 1e-5;
        let h_plus = family
            .exact_newton_joint_hessian(&latent_binary_states_from_joint_beta(
                &family,
                &(beta.clone() + &(fd_step * &direction)),
            ))
            .expect("hessian plus")
            .expect("hessian plus should exist");
        let h_minus = family
            .exact_newton_joint_hessian(&latent_binary_states_from_joint_beta(
                &family,
                &(beta - &(fd_step * &direction)),
            ))
            .expect("hessian minus")
            .expect("hessian minus should exist");
        let fd_dh = (&h_plus - &h_minus) / (2.0 * fd_step);
        assert!(
            max_relative_array2(&dh, &fd_dh) < 2e-4,
            "latent binary workspace dH mismatch: dh={dh:?}, fd={fd_dh:?}"
        );
    }

    #[test]
    fn latent_survival_learnable_sigma_block_matches_family_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let slices = family.joint_slices();
        let sigma_idx = slices
            .log_sigma
            .as_ref()
            .expect("learnable sigma test family should expose log_sigma")
            .start;
        let h = 2e-4;

        let eval = family
            .evaluate(&states)
            .expect("learnable latent survival evaluation");
        let joint_gradient = family
            .exact_newton_joint_gradient_evaluation(&states, &[])
            .expect("joint gradient evaluation")
            .expect("joint gradient should exist")
            .gradient;
        let joint_hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("joint hessian evaluation")
            .expect("joint hessian should exist");
        assert_eq!(eval.blockworking_sets.len(), 3);

        let (block_grad, block_neg_hess) =
            match &eval.blockworking_sets[LatentSurvivalFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, hessian } => {
                    let neg_hess = match hessian {
                        SymmetricMatrix::Dense(mat) => mat[[0, 0]],
                        _ => panic!("log_sigma block should use a dense exact-Newton Hessian"),
                    };
                    (gradient[0], neg_hess)
                }
                _ => panic!("log_sigma block should use ExactNewton"),
            };

        assert!((block_grad - joint_gradient[sigma_idx]).abs() < 1e-12);
        assert!((block_neg_hess - joint_hessian[[sigma_idx, sigma_idx]]).abs() < 1e-12);

        let mut beta_plus = beta.clone();
        beta_plus[sigma_idx] += h;
        let ll_plus = family
            .log_likelihood_only(&latent_survival_states_from_joint_beta(&family, &beta_plus))
            .expect("ll plus");
        let ll_0 = family.log_likelihood_only(&states).expect("ll base");
        let mut beta_minus = beta.clone();
        beta_minus[sigma_idx] -= h;
        let ll_minus = family
            .log_likelihood_only(&latent_survival_states_from_joint_beta(
                &family,
                &beta_minus,
            ))
            .expect("ll minus");

        let fd_grad = (ll_plus - ll_minus) / (2.0 * h);
        let fd_neg_hess = -(ll_plus - 2.0 * ll_0 + ll_minus) / (h * h);
        assert!(
            (joint_gradient[sigma_idx] - fd_grad).abs()
                / joint_gradient[sigma_idx]
                    .abs()
                    .max(fd_grad.abs())
                    .max(1e-12)
                < 2e-3,
            "family log_sigma grad={}, fd={fd_grad}",
            joint_gradient[sigma_idx]
        );
        assert!(
            (joint_hessian[[sigma_idx, sigma_idx]] - fd_neg_hess).abs()
                / joint_hessian[[sigma_idx, sigma_idx]]
                    .abs()
                    .max(fd_neg_hess.abs())
                    .max(1e-10)
                < 2e-2,
            "family log_sigma neg_hess={}, fd={fd_neg_hess}",
            joint_hessian[[sigma_idx, sigma_idx]]
        );
    }

    #[test]
    fn latent_survival_exact_joint_hessian_matches_gradient_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let h = 1e-6;

        let analytic_hessian = family
            .exact_newton_joint_hessian(&states)
            .expect("analytic joint hessian evaluation")
            .expect("latent survival should expose exact joint hessian");

        for j in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[j] += h;
            let gradient_plus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_survival_states_from_joint_beta(&family, &beta_plus),
                    &[],
                )
                .expect("joint gradient plus")
                .expect("joint gradient should exist")
                .gradient;

            let mut beta_minus = beta.clone();
            beta_minus[j] -= h;
            let gradient_minus = family
                .exact_newton_joint_gradient_evaluation(
                    &latent_survival_states_from_joint_beta(&family, &beta_minus),
                    &[],
                )
                .expect("joint gradient minus")
                .expect("joint gradient should exist")
                .gradient;

            let fd_column = (&gradient_plus - &gradient_minus) / (2.0 * h);
            let analytic_column = analytic_hessian.column(j).to_owned();
            let rel = max_relative_array1(&analytic_column, &(-fd_column));
            assert!(
                rel < 5e-4,
                "joint Hessian column {j} mismatch: rel={rel}, analytic={analytic_column:?}, fd={:?}",
                -((&gradient_plus - &gradient_minus) / (2.0 * h))
            );
        }
    }

    /// FD check for `LatentSurvivalFamily::offset_channel_residuals`: each
    /// channel residual sums to `∂(−ℓ)/∂o_ch` for a uniform additive offset on
    /// that time channel (the baseline-θ enters only through these offsets).
    /// `o_ch` shifts `eta_time[ch-slice]` uniformly, so `Σ_i r^ch_i` is exactly
    /// the directional derivative of `−ℓ` along a constant offset on channel ch.
    /// This validates the envelope-theorem latent baseline-θ gradient primitive.
    #[test]
    fn latent_survival_offset_channel_residuals_match_finite_difference() {
        let family = survival_stress_test_family(24);
        let beta = survival_stress_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let n = family.event_target.len();

        let residuals = family
            .offset_channel_residuals(&states)
            .expect("offset channel residuals");
        let sum_entry: f64 = residuals.entry.sum();
        let sum_exit: f64 = residuals.exit.sum();
        let sum_deriv: f64 = residuals.derivative.sum();

        // `−ℓ` after shifting one time channel's eta by a constant δ.
        let neg_ll_with_offset = |channel: usize, delta: f64| -> f64 {
            let mut shifted = states.clone();
            let slice = match channel {
                0 => s![0..n],
                1 => s![n..2 * n],
                2 => s![2 * n..3 * n],
                _ => unreachable!(),
            };
            shifted[LatentSurvivalFamily::BLOCK_TIME]
                .eta
                .slice_mut(slice)
                .mapv_inplace(|v| v + delta);
            let (ll, _) = family
                .evaluate_exact_newton_joint_gradient_dense(&shifted)
                .expect("shifted joint gradient evaluation");
            -ll
        };

        let h = 1e-6;
        let fd_entry = (neg_ll_with_offset(0, h) - neg_ll_with_offset(0, -h)) / (2.0 * h);
        let fd_exit = (neg_ll_with_offset(1, h) - neg_ll_with_offset(1, -h)) / (2.0 * h);
        let fd_deriv = (neg_ll_with_offset(2, h) - neg_ll_with_offset(2, -h)) / (2.0 * h);

        assert!(
            (sum_entry - fd_entry).abs() <= 1e-5 * fd_entry.abs().max(1.0),
            "entry-channel residual sum mismatch: analytic={sum_entry}, fd={fd_entry}"
        );
        assert!(
            (sum_exit - fd_exit).abs() <= 1e-5 * fd_exit.abs().max(1.0),
            "exit-channel residual sum mismatch: analytic={sum_exit}, fd={fd_exit}"
        );
        assert!(
            (sum_deriv - fd_deriv).abs() <= 1e-5 * fd_deriv.abs().max(1.0),
            "derivative-channel residual sum mismatch: analytic={sum_deriv}, fd={fd_deriv}"
        );
    }

    #[test]
    fn latent_survival_exact_joint_parallel_stress_is_repeatable() {
        let family = survival_stress_test_family(96);
        let beta = survival_stress_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let direction_u = array![0.03, -0.02, 0.01, 0.04, -0.015, 0.025, -0.005, 0.02];
        let direction_v = array![-0.01, 0.035, -0.025, 0.015, 0.02, -0.01, 0.03, -0.015];

        let (ll_a, grad_a) = family
            .evaluate_exact_newton_joint_gradient_dense(&states)
            .expect("stress joint gradient evaluation");
        let (ll_b, grad_b) = family
            .evaluate_exact_newton_joint_gradient_dense(&states)
            .expect("repeat stress joint gradient evaluation");
        assert_eq!(ll_a.to_bits(), ll_b.to_bits());
        assert_eq!(grad_a, grad_b);

        let (joint_ll_a, joint_grad_a, hess_a) = family
            .evaluate_exact_newton_joint_dense(&states)
            .expect("stress joint dense evaluation");
        let (joint_ll_b, joint_grad_b, hess_b) = family
            .evaluate_exact_newton_joint_dense(&states)
            .expect("repeat stress joint dense evaluation");
        assert_eq!(joint_ll_a.to_bits(), joint_ll_b.to_bits());
        assert_eq!(joint_grad_a, joint_grad_b);
        assert_eq!(hess_a, hess_b);
        assert!(hess_a.iter().all(|value| value.is_finite()));
        assert!(max_relative_array2(&hess_a, &hess_a.t().to_owned()) < 1e-12);

        let dh_a = family
            .exact_newton_joint_hessian_directional_derivative_dense(&states, &direction_u)
            .expect("stress joint dH evaluation");
        let dh_b = family
            .exact_newton_joint_hessian_directional_derivative_dense(&states, &direction_u)
            .expect("repeat stress joint dH evaluation");
        assert_eq!(dh_a, dh_b);
        assert!(dh_a.iter().all(|value| value.is_finite()));
        assert!(max_relative_array2(&dh_a, &dh_a.t().to_owned()) < 1e-12);

        let d2h_a = family
            .exact_newton_joint_hessian_second_directional_derivative_dense(
                &states,
                &direction_u,
                &direction_v,
            )
            .expect("stress joint d2H evaluation");
        let d2h_b = family
            .exact_newton_joint_hessian_second_directional_derivative_dense(
                &states,
                &direction_u,
                &direction_v,
            )
            .expect("repeat stress joint d2H evaluation");
        assert_eq!(d2h_a, d2h_b);
        assert!(d2h_a.iter().all(|value| value.is_finite()));
        assert!(max_relative_array2(&d2h_a, &d2h_a.t().to_owned()) < 1e-12);
    }

    #[test]
    fn latent_survival_exact_joint_dh_matches_hessian_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let h = 2e-4;
        let direction = array![0.07, -0.03, 0.05, 0.02, -0.04];

        let analytic = family
            .exact_newton_joint_hessian_directional_derivative(&states, &direction)
            .expect("analytic joint dH evaluation")
            .expect("latent survival should expose exact joint dH");

        let hessian_plus = family
            .exact_newton_joint_hessian(&latent_survival_states_from_joint_beta(
                &family,
                &(beta.clone() + h * &direction),
            ))
            .expect("joint hessian plus")
            .expect("joint hessian should exist");
        let hessian_minus = family
            .exact_newton_joint_hessian(&latent_survival_states_from_joint_beta(
                &family,
                &(beta.clone() - h * &direction),
            ))
            .expect("joint hessian minus")
            .expect("joint hessian should exist");

        let fd = (&hessian_plus - &hessian_minus) / (2.0 * h);
        let rel = frobenius_relative_array2(&analytic, &fd);
        assert!(rel < 2e-3, "joint dH mismatch: rel={rel}");
    }

    #[test]
    fn latent_survival_exact_joint_d2h_matches_directional_fd() {
        let family = learnable_sigma_test_family();
        let beta = learnable_sigma_test_joint_beta();
        let states = latent_survival_states_from_joint_beta(&family, &beta);
        let h = 5e-4;
        let direction_u = array![0.07, -0.03, 0.05, 0.02, -0.04];
        let direction_v = array![-0.02, 0.06, -0.01, 0.03, 0.05];

        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &direction_u,
                &direction_v,
            )
            .expect("analytic joint d2H evaluation")
            .expect("latent survival should expose exact joint d2H");
        let swapped = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                &states,
                &direction_v,
                &direction_u,
            )
            .expect("swapped analytic joint d2H evaluation")
            .expect("latent survival should expose exact joint d2H");
        let symmetry_rel = max_relative_array2(&analytic, &swapped);
        assert!(
            symmetry_rel < 1e-10,
            "joint d2H should be symmetric in directions, got rel={symmetry_rel}"
        );

        let dh_plus = family
            .exact_newton_joint_hessian_directional_derivative(
                &latent_survival_states_from_joint_beta(
                    &family,
                    &(beta.clone() + h * &direction_v),
                ),
                &direction_u,
            )
            .expect("joint dH plus")
            .expect("joint dH should exist");
        let dh_minus = family
            .exact_newton_joint_hessian_directional_derivative(
                &latent_survival_states_from_joint_beta(
                    &family,
                    &(beta.clone() - h * &direction_v),
                ),
                &direction_u,
            )
            .expect("joint dH minus")
            .expect("joint dH should exist");

        let fd = (&dh_plus - &dh_minus) / (2.0 * h);
        let rel = frobenius_relative_array2(&analytic, &fd);
        assert!(rel < 2.5e-2, "joint d2H mismatch: rel={rel}");
    }

    #[test]
    fn latent_survival_row_primary_derivatives_match_fd() {
        let quadctx = QuadratureContext::new();
        let row = LatentSurvivalRow::exact_event(0.35, 1.4, 0.1, 0.45, 0.8, 0.12);
        // [q_entry, q_exit, qdot_exit, q_right, mu, log_sigma]. This is an
        // exact-event row, so the `q_right` channel is inert (the likelihood
        // does not depend on it); the FD loop below confirms its gradient/Hessian
        // entries are zero.
        let primary = array![
            0.35f64.ln(),
            1.4f64.ln(),
            0.8,
            1.6f64.ln(),
            -0.2,
            0.4f64.ln()
        ];
        let sigma = primary[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA].exp();
        let h_grad = 1e-6;
        let h_hess = 2e-4;

        let (_, gradient, neg_hessian) = latent_survival_row_primary_gradient_hessian(
            &quadctx,
            &row,
            primary[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
            primary[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
            primary[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
            primary[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
            primary[LATENT_SURVIVAL_PRIMARY_MU],
            sigma,
            true,
        )
        .expect("analytic row primary gradient/hessian");

        for j in 0..LATENT_SURVIVAL_PRIMARY_DIM {
            let mut plus = primary.clone();
            plus[j] += h_grad;
            let mut minus = primary.clone();
            minus[j] -= h_grad;
            let fd_grad = (latent_survival_row_loglik_from_primary(&quadctx, &row, &plus)
                - latent_survival_row_loglik_from_primary(&quadctx, &row, &minus))
                / (2.0 * h_grad);
            let rel_grad =
                (gradient[j] - fd_grad).abs() / gradient[j].abs().max(fd_grad.abs()).max(1e-12);
            assert!(
                rel_grad < 2e-4,
                "row primary grad[{j}] mismatch: analytic={}, fd={fd_grad}, rel={rel_grad}",
                gradient[j]
            );

            for k in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                let mut pp = primary.clone();
                pp[j] += h_hess;
                pp[k] += h_hess;
                let mut pm = primary.clone();
                pm[j] += h_hess;
                pm[k] -= h_hess;
                let mut mp = primary.clone();
                mp[j] -= h_hess;
                mp[k] += h_hess;
                let mut mm = primary.clone();
                mm[j] -= h_hess;
                mm[k] -= h_hess;
                let fd_neg_hess = -(latent_survival_row_loglik_from_primary(&quadctx, &row, &pp)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &pm)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &mp)
                    + latent_survival_row_loglik_from_primary(&quadctx, &row, &mm))
                    / (4.0 * h_hess * h_hess);
                let analytic = neg_hessian[[j, k]];
                let abs_err = (analytic - fd_neg_hess).abs();
                let rel = abs_err / analytic.abs().max(fd_neg_hess.abs()).max(1e-10);
                assert!(
                    abs_err < 2e-5 || rel < 2e-3,
                    "row primary neg_hess[{j},{k}] mismatch: analytic={analytic}, fd={fd_neg_hess}, abs_err={abs_err}, rel={rel}"
                );
            }
        }
    }

    #[test]
    fn latent_survival_interval_row_primary_derivatives_match_fd() {
        // Interval-censored row jet `ℓ = log[S(L) − S(R)] − log S(entry)`. The
        // dynamic two-state numerator differentiates BOTH boundary masses
        // `M_L = exp(q_exit)` (left, `q_exit`) and `M_R = exp(q_right)` (right,
        // `q_right`) independently — channels that the static
        // `LatentSurvivalRowJet::interval_censored` (μ-only) never exercises. This
        // FD-verifies the gradient AND neg-Hessian of the interval contribution
        // w.r.t. ALL six primary coordinates (q_entry, q_exit/L, qdot_exit,
        // q_right/R, mu, log_sigma) on a WELL-POSED bracket where `S(L) − S(R)` is
        // comfortably positive (M_L = e^{−0.4} ≈ 0.67 well below M_R = e^{0.5} ≈
        // 1.65, so the survival-mass difference is large and the log-of-a-
        // difference curvature is well-conditioned).
        let quadctx = QuadratureContext::new();
        // Bracket masses: entry < L < R with comfortable gaps.
        let q_entry = -1.2_f64; // M_entry = e^{−1.2} ≈ 0.30
        let q_exit = -0.4_f64; // L: M_L = e^{−0.4} ≈ 0.67
        let q_right = 0.5_f64; // R: M_R = e^{0.5} ≈ 1.65 (> M_L)
        let mu = -0.15_f64;
        let log_sigma = 0.3_f64; // σ ≈ 1.35
        // Small, monotone unloaded masses (entry ≤ left ≤ right); qdot is inert
        // for the interval contribution.
        let row = LatentSurvivalRow::interval_censored(
            q_entry.exp(), // mass_entry (consistency only; jet reads q's)
            q_exit.exp(),  // mass_left
            q_right.exp(), // mass_right
            0.01,          // mass_unloaded_entry
            0.02,          // mass_unloaded_left
            0.05,          // mass_unloaded_right
        );
        assert!(matches!(
            row.event_type,
            LatentSurvivalEventType::IntervalCensored
        ));

        // [q_entry, q_exit/L, qdot_exit, q_right/R, mu, log_sigma]. qdot_exit is
        // inert for interval rows (no hazard-derivative channel); the FD loop
        // confirms its gradient/Hessian entries are 0.
        let primary = array![q_entry, q_exit, 0.7, q_right, mu, log_sigma];
        let sigma = primary[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA].exp();
        let h_grad = 1e-6;
        let h_hess = 2e-4;

        let (_, gradient, neg_hessian) = latent_survival_row_primary_gradient_hessian(
            &quadctx,
            &row,
            primary[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
            primary[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
            primary[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
            primary[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
            primary[LATENT_SURVIVAL_PRIMARY_MU],
            sigma,
            true,
        )
        .expect("analytic interval row primary gradient/hessian");

        // The interval contribution must be a positive survival-mass difference
        // at this bracket, so the value channel is finite.
        let value = latent_survival_row_loglik_from_primary(&quadctx, &row, &primary);
        assert!(
            value.is_finite(),
            "interval row log-likelihood must be finite on a well-posed bracket, got {value}"
        );

        for j in 0..LATENT_SURVIVAL_PRIMARY_DIM {
            let mut plus = primary.clone();
            plus[j] += h_grad;
            let mut minus = primary.clone();
            minus[j] -= h_grad;
            let fd_grad = (latent_survival_row_loglik_from_primary(&quadctx, &row, &plus)
                - latent_survival_row_loglik_from_primary(&quadctx, &row, &minus))
                / (2.0 * h_grad);
            let rel_grad =
                (gradient[j] - fd_grad).abs() / gradient[j].abs().max(fd_grad.abs()).max(1e-12);
            assert!(
                rel_grad < 2e-4,
                "interval row primary grad[{j}] mismatch: analytic={}, fd={fd_grad}, rel={rel_grad}",
                gradient[j]
            );

            for k in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                let mut pp = primary.clone();
                pp[j] += h_hess;
                pp[k] += h_hess;
                let mut pm = primary.clone();
                pm[j] += h_hess;
                pm[k] -= h_hess;
                let mut mp = primary.clone();
                mp[j] -= h_hess;
                mp[k] += h_hess;
                let mut mm = primary.clone();
                mm[j] -= h_hess;
                mm[k] -= h_hess;
                let fd_neg_hess = -(latent_survival_row_loglik_from_primary(&quadctx, &row, &pp)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &pm)
                    - latent_survival_row_loglik_from_primary(&quadctx, &row, &mp)
                    + latent_survival_row_loglik_from_primary(&quadctx, &row, &mm))
                    / (4.0 * h_hess * h_hess);
                let analytic = neg_hessian[[j, k]];
                let abs_err = (analytic - fd_neg_hess).abs();
                let rel = abs_err / analytic.abs().max(fd_neg_hess.abs()).max(1e-10);
                assert!(
                    abs_err < 5e-5 || rel < 3e-3,
                    "interval row primary neg_hess[{j},{k}] mismatch: analytic={analytic}, fd={fd_neg_hess}, abs_err={abs_err}, rel={rel}"
                );
            }
        }
    }
}
