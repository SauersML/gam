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

use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, ConstraintSet, CustomFamily,
    ExactNewtonJointGradientEvaluation, ExactNewtonJointHessianWorkspace, FamilyEvaluation,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, fit_custom_family,
    fit_custom_family_fixed_log_lambdas,
};
use crate::fit_orchestration::drivers::freeze_term_collection_from_design;
use crate::gamlss::{FamilyMetadata, ParameterLink};
use crate::model_types::UnifiedFitResult;
use crate::probability::{
    exact_binary64_sum_sign, log1mexp_positive, signed_log_sum_exp,
};
use crate::quadrature::{IntegratedExpectationMode, QuadratureContext};
use crate::sigma_link::{exp_sigma_eta_for_sigma_scalar, exp_sigma_from_eta_scalar};
use crate::survival::latent::interval::{
    LatentFrailtyResolution, LatentIntervalModel, LatentIntervalRowView,
    validate_latent_interval_inputs,
};
use crate::survival::location_scale::{
    TimeBlockInput, project_onto_linear_constraints, structural_time_coefficient_constraints,
};
use crate::survival::lognormal_kernel::{
    FrailtyScale, FrailtySpec, HazardLoading, LatentSurvivalEventType, LatentSurvivalRow,
    LatentSurvivalRowJet, LogLognormalKernelBundle, log_kernel_bundle,
};
use gam_linalg::matrix::{DenseDesignMatrix, DesignMatrix, LinearOperator, SymmetricMatrix};
use gam_math::jet_scalar::{JetScalar, OneSeed, Order2, TwoSeed};
// `value`/`compose_unary`/… now live on the shared `JetField` base (JetScalar: JetField);
// the concrete `row_jet.base.value()` reads below need it in scope.
use gam_math::nested_dual::JetField;
use gam_solve::pirls::LinearInequalityConstraints;
use gam_terms::smooth::{TermCollectionDesign, TermCollectionSpec, build_term_collection_design};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use smallvec::SmallVec;
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
    /// A derivative could not be certified as the unique binary64 rounding of
    /// the exact cumulant over its rounded finite moment inputs.
    DerivativeAccuracyUnresolved { reason: String },
    /// The requested combination of time-block structure or event type is
    /// not implemented (non-structural monotonicity, interval-censored rows
    /// on the dynamic-derivative path).
    UnsupportedConfiguration { reason: String },
}

impl_reason_error_boilerplate! {
    LatentSurvivalError {
        InvalidFrailty,
        InvalidDataset,
        BlockMismatch,
        NumericalFailure,
        DerivativeAccuracyUnresolved,
        UnsupportedConfiguration,
    }
}

impl From<crate::block_layout::block_count::BlockCountMismatch> for LatentSurvivalError {
    fn from(err: crate::block_layout::block_count::BlockCountMismatch) -> LatentSurvivalError {
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

/// Whole-input proof that likelihood row weights are finite and non-negative.
///
/// Construct this before evaluating any response or predictor row. Once
/// constructed, `weight == 0` is the only dormant-row case; every positive
/// value, including subnormals, remains part of the likelihood.
#[derive(Clone, Copy)]
struct ValidatedLikelihoodWeights<'a> {
    values: &'a Array1<f64>,
}

impl<'a> ValidatedLikelihoodWeights<'a> {
    fn new(values: &'a Array1<f64>, context: &str) -> Result<Self, LatentSurvivalError> {
        if let Some((row, &weight)) = values
            .iter()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || **weight < 0.0)
        {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: format!(
                    "{context} row {} has invalid likelihood weight {weight:?}; expected finite weight >= 0",
                    row + 1
                ),
            });
        }
        Ok(Self { values })
    }

    #[inline]
    fn at(self, row: usize) -> f64 {
        self.values[row]
    }
}

/// Multiply one already-finite row quantity by a validated positive weight.
/// Overflow and non-zero underflow are explicit errors rather than silently
/// changing the row's contribution. Exact zero row quantities remain zero.
fn checked_weighted_row_value(
    weight: f64,
    value: f64,
    row: usize,
    quantity: &str,
) -> Result<f64, String> {
    assert!(weight.is_finite() && weight > 0.0);
    if !value.is_finite() {
        return Err(format!(
            "latent likelihood row {} has non-finite unweighted {quantity}: {value:?}",
            row + 1
        ));
    }
    let weighted = weight * value;
    if !weighted.is_finite() {
        return Err(format!(
            "latent likelihood row {} weighted {quantity} is not representable: {weight:?} * {value:?}",
            row + 1
        ));
    }
    if value != 0.0 && weighted == 0.0 {
        return Err(format!(
            "latent likelihood row {} weighted {quantity} underflowed and is not representable: {weight:?} * {value:?}",
            row + 1
        ));
    }
    Ok(weighted)
}

fn checked_weighted_row_matrix(
    weight: f64,
    values: &Array2<f64>,
    row: usize,
    quantity: &str,
) -> Result<Array2<f64>, String> {
    let mut weighted = Array2::<f64>::zeros(values.dim());
    for ((left, right), &value) in values.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "latent likelihood row {} has non-finite unweighted {quantity}[{left},{right}]: {value:?}",
                row + 1
            ));
        }
        let product = weight * value;
        if !product.is_finite() || (value != 0.0 && product == 0.0) {
            return Err(format!(
                "latent likelihood row {} weighted {quantity}[{left},{right}] is not representable: {weight:?} * {value:?}",
                row + 1
            ));
        }
        weighted[[left, right]] = product;
    }
    Ok(weighted)
}

fn require_finite_likelihood_scalar(value: f64, quantity: &str) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!(
            "latent likelihood accumulated {quantity} is not representable: {value:?}"
        ))
    }
}

fn require_finite_likelihood_vector(values: &Array1<f64>, quantity: &str) -> Result<(), String> {
    if let Some((index, &value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "latent likelihood accumulated {quantity}[{index}] is not representable: {value:?}"
        ));
    }
    Ok(())
}

fn require_finite_likelihood_matrix(values: &Array2<f64>, quantity: &str) -> Result<(), String> {
    if let Some(((row, col), &value)) = values.indexed_iter().find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "latent likelihood accumulated {quantity}[{row},{col}] is not representable: {value:?}"
        ));
    }
    Ok(())
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
    pub baseline_offset_residuals: crate::survival::OffsetChannelResiduals,
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
    pub baseline_offset_residuals: crate::survival::OffsetChannelResiduals,
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
        crate::block_layout::block_count::validate_block_count::<LatentSurvivalError>(
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
        crate::block_layout::block_count::validate_block_count::<LatentSurvivalError>(
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
    frailty
        .validate()
        .map_err(|err| LatentSurvivalError::InvalidFrailty {
            reason: err.to_string(),
        })?;
    match frailty {
        FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Fixed { sigma },
            loading,
        } => Ok((*sigma, *loading)),
        FrailtySpec::HazardMultiplier {
            scale: FrailtyScale::Learned { .. },
            ..
        } => Err(LatentSurvivalError::InvalidFrailty {
            reason: format!("{context} requires a fixed hazard-multiplier sigma"),
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
    mut spec: LatentSurvivalTermSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentSurvivalTermFitResult, String> {
    let frailty_scale = validate_latent_survival_inputs(data, &spec, &frailty)?;
    // Cover the monotone I-spline baseline's unpenalized affine null direction
    // with a REML-selected function-space shrinkage ridge, so the interval
    // warm-start surrogates (whose likelihood has no curvature along that
    // direction) have a unique MAP instead of refusing. No-op on blocks whose
    // penalties already span their column space. See the installer's doc.
    install_latent_time_nullspace_shrinkage_penalty(&mut spec.time_block)?;
    let (latent_sd, learned_initial_sigma) = match frailty_scale {
        FrailtyScale::Fixed { sigma } => (Some(sigma), None),
        FrailtyScale::Learned { initial_sigma } => (None, Some(initial_sigma)),
    };
    let hazard_loading = latent_hazard_loading(&frailty, "latent-survival")?;
    let mean_design =
        build_term_collection_design(data, &spec.meanspec).map_err(|e| e.to_string())?;
    let mean_offset = mean_design
        .compose_offset(spec.mean_offset.view(), "latent-survival mean block")
        .map_err(|e| e.to_string())?;
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
        build_mean_blockspec(&mean_design, mean_offset),
    ];
    if let Some(initial_sigma) = learned_initial_sigma {
        blocks.push(build_log_sigma_blockspec(
            initial_sigma,
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
        let censored_warm_event_target = spec.event_target.mapv(|code| {
            if code == LATENT_SURVIVAL_EVENT_INTERVAL {
                0u8
            } else {
                code
            }
        });
        let mut warm_family = family.clone();
        warm_family.event_target = censored_warm_event_target;
        // Right-censored-at-L ignores the interval upper bound `R`, so the
        // (unused) `q_right` channel cannot drift the fit; leaving the right
        // design/mass in place is harmless (no interval row remains to read it).
        // Fixed-λ surrogate: no outer smoothing loop runs here, and the inner
        // solve is convergence-gated inside `fit_custom_family_fixed_log_lambdas`,
        // so the assembled surrogate fit is (vacuously) outer-converged.
        let warm_fit_result = fit_custom_family_fixed_log_lambdas(
            &warm_family,
            &blocks,
            options,
            None,
        );
        let warm_fit = match warm_fit_result {
            Ok(fit) => fit,
            Err(censored_error) => {
                let has_finite_event_in_censored_surrogate =
                    warm_family.event_target.iter().any(|&code| code != 0);
                if has_finite_event_in_censored_surrogate {
                    return Err(format!(
                        "latent interval warm start: right-censored-at-L surrogate fit failed \
                         (so the interval fit cannot be safely warm-started; this surrogate is \
                         log-concave and should converge — investigate the surrogate, not the \
                         interval kernel): {censored_error}"
                    ));
                }

                // When every observed row is interval-censored, the
                // right-censored-at-L surrogate contains no failures at all.
                // Its likelihood is maximized only on the zero-hazard boundary
                // (β_time -> -∞), so the fixed-λ Newton solve is correctly
                // allowed to refuse it even though the objective is concave.
                // Use the finite lower-endpoint event surrogate solely to obtain
                // an interior β/σ seed for the exact interval likelihood below;
                // no fitted surrogate likelihood or derivative is reused.
                let lower_event_warm_target = spec.event_target.mapv(|code| {
                    if code == LATENT_SURVIVAL_EVENT_INTERVAL {
                        1u8
                    } else {
                        code
                    }
                });
                let mut event_warm_family = family.clone();
                event_warm_family.event_target = lower_event_warm_target;
                fit_custom_family_fixed_log_lambdas(
                    &event_warm_family,
                    &blocks,
                    options,
                    None,
                )
                .map_err(|event_error| {
                    format!(
                        "latent interval warm start failed: the right-censored-at-L surrogate \
                         has no finite failures and refused its boundary optimum ({censored_error}); \
                         the finite lower-endpoint event surrogate also failed ({event_error})"
                    )
                })?
            }
        };
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
    let mean_offset = mean_design
        .compose_offset(spec.mean_offset.view(), "latent-binary mean block")
        .map_err(|e| e.to_string())?;
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
        build_mean_blockspec(&mean_design, mean_offset),
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
/// Survival permits [`FrailtyScale::Learned`] and carries the
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
        frailty
            .validate()
            .map_err(|err| LatentSurvivalError::InvalidFrailty {
                reason: err.to_string(),
            })?;
        match frailty {
            FrailtySpec::HazardMultiplier {
                scale,
                loading,
            } => Ok(LatentFrailtyResolution {
                scale: *scale,
                loading: *loading,
            }),
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
) -> Result<FrailtyScale, LatentSurvivalError> {
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
            scale: FrailtyScale::Fixed { sigma },
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
    match validate_latent_interval_inputs::<LatentBinaryModel>(data, &row)? {
        FrailtyScale::Fixed { sigma } => Ok(sigma),
        FrailtyScale::Learned { .. } => Err(LatentSurvivalError::InvalidFrailty {
            reason: "latent-binary requires a fixed latent sigma".to_string(),
        }),
    }
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
    let stacked_offset = gam_linalg::utils::stack_offsets(&[
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

/// Install the Marra & Wood function-space null-space shrinkage penalty (the
/// "double penalty") on the latent-survival time block, so the monotone I-spline
/// baseline's unpenalized affine null direction carries its own REML-selected
/// curvature instead of being left flat.
///
/// The I-spline value-space penalty `S_I = Lᵀ S_B[1:,1:] L` (see the `ISpline`
/// arm of [`crate::survival::construction::build_survival_time_basis`])
/// deliberately leaves the affine trend `d(log Λ)/d(log t)` in its 1-D null space
/// (`constant γ ↦ affine log Λ ↦ D₂ = 0`, gam#1076): on an event-rich fit the
/// likelihood identifies that trend, so penalizing it would only bias the
/// baseline. But the interval-censored warm-start evaluates the same direction
/// where the surrogate likelihood has no curvature there — the
/// right-censored-at-`L` surrogate has no finite failures at all, and the
/// finite-lower-endpoint surrogate's grid-tied events leave `Jᵀ W J` near-flat
/// along the affine mode — so the MAP is non-unique and the fit refuses
/// (`gam_identifiability::check_map_uniqueness`: the affine null direction of
/// `Jᵀ W J` carries `nᵀ S n < tol`, dominant block `time_transform`).
///
/// This is the treatment the survival marginal-slope time block already installs
/// (`install_time_nullspace_shrinkage_penalty`): the shared
/// [`gam_terms::basis::function_space_nullspace_shrinkage`] builds the
/// function-metric ridge `G Z (ZᵀGZ)⁻¹ ZᵀG` (`Z` spanning the primary penalty's
/// null space, `G` the endpoint-averaged basis Gram), whose range is exactly that
/// null direction, so `nᵀ S n > 0` there. It is a *second* REML coordinate:
/// where the affine trend is identified by the data REML drives its λ toward zero
/// (no bias — the gam#1076 behaviour is preserved), and where it is not (the
/// interval warm-start) the ridge supplies the curvature that makes the MAP
/// unique. Returns `Ok(false)` when the block carries no penalty with a null
/// space (nothing to cover).
fn install_latent_time_nullspace_shrinkage_penalty(
    time_block: &mut TimeBlockInput,
) -> Result<bool, String> {
    let p = time_block.design_exit.ncols();
    if p == 0 || time_block.penalties.is_empty() {
        return Ok(false);
    }
    if time_block.nullspace_dims.len() != time_block.penalties.len() {
        return Err(format!(
            "latent-survival time_block nullspace_dims length {} does not match penalties {}",
            time_block.nullspace_dims.len(),
            time_block.penalties.len(),
        ));
    }

    // Aggregate the existing wiggliness penalties, each normalized by its own
    // max-abs scale so the shared null space (not a scale-weighted average) is
    // what the ridge covers. Mirrors the marginal-slope installer.
    let mut aggregate = Array2::<f64>::zeros((p, p));
    for (idx, penalty) in time_block.penalties.iter().enumerate() {
        if penalty.nrows() != p || penalty.ncols() != p {
            return Err(format!(
                "latent-survival time_block penalty {idx} must be {p}x{p}, got {}x{}",
                penalty.nrows(),
                penalty.ncols(),
            ));
        }
        let scale = penalty
            .iter()
            .try_fold(0.0_f64, |acc, &value| {
                value.is_finite().then_some(acc.max(value.abs()))
            })
            .ok_or_else(|| {
                format!("latent-survival time_block penalty {idx} contains non-finite values")
            })?;
        if scale > 0.0 {
            ndarray::Zip::from(&mut aggregate)
                .and(penalty)
                .for_each(|agg, &value| *agg += value / scale);
        }
    }

    // Endpoint-averaged function metric: entry and exit are the two value
    // channels through which the baseline enters the survival likelihood, so
    // averaging their Grams makes the ridge invariant to whole-sample
    // replication and covariant under any coefficient-chart change
    // (`G -> MᵀGM`), exactly like the marginal-slope time-block ridge.
    if time_block.design_entry.ncols() != p {
        return Err(format!(
            "latent-survival time_block entry design has {} columns, expected {p}",
            time_block.design_entry.ncols(),
        ));
    }
    let entry_mass = time_block.design_entry.nrows();
    let exit_mass = time_block.design_exit.nrows();
    let total_mass = entry_mass.saturating_add(exit_mass);
    if total_mass == 0 {
        return Err(
            "latent-survival time_block cannot define a function metric from zero endpoint rows"
                .to_string(),
        );
    }
    let entry_gram = time_block
        .design_entry
        .diag_xtw_x(&Array1::ones(entry_mass))
        .map_err(|err| format!("latent-survival time_block entry function Gram: {err}"))?;
    let exit_gram = time_block
        .design_exit
        .diag_xtw_x(&Array1::ones(exit_mass))
        .map_err(|err| format!("latent-survival time_block exit function Gram: {err}"))?;
    let function_gram = (entry_gram + exit_gram).mapv(|value| value / total_mass as f64);

    let Some(shrinkage) =
        gam_terms::basis::function_space_nullspace_shrinkage(&aggregate, &function_gram)
            .map_err(|err| format!("latent-survival time_block nullspace shrinkage: {err}"))?
    else {
        return Ok(false);
    };
    if shrinkage.nrows() != p || shrinkage.ncols() != p {
        return Err(format!(
            "latent-survival time_block nullspace shrinkage penalty must be {p}x{p}, got {}x{}",
            shrinkage.nrows(),
            shrinkage.ncols(),
        ));
    }
    time_block.penalties.push(shrinkage);
    time_block.nullspace_dims.push(0);
    // Keep the seed ρ vector consistent with the widened penalty list. The
    // latent block builder reads `initial_log_lambdas` directly (unlike the
    // marginal-slope path, which rebuilds its seed from the penalty list), so a
    // `Some` seed must gain a coordinate for the new null-space penalty; seed it
    // at the same smoothing scale as the existing time penalties.
    if let Some(seeds) = time_block.initial_log_lambdas.as_mut() {
        let seed = seeds.iter().copied().last().unwrap_or(0.0);
        let mut widened = seeds.to_vec();
        widened.push(seed);
        *seeds = Array1::from_vec(widened);
    }
    Ok(true)
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

/// Certified derivative tower of `f(x) = log(1 - exp(x))` for `x < 0`.
///
/// The value uses `log1mexp`; derivative magnitudes are assembled in log space:
///
/// ```text
/// |f'|    = r / s
/// |f''|   = r / s²
/// |f'''|  = r(1 + r) / s³
/// |f''''| = r(1 + 4r + r²) / s⁴,
/// r = exp(x), s = 1 - r.
/// ```
///
/// This never forms `1/s^k`. If a true derivative magnitude cannot be
/// represented by `f64`, the routine returns a typed numerical refusal instead
/// of publishing an infinite jet. Only the derivative order consumed by the
/// selected jet backend is certified. There is no clamp or magnitude cutoff.
fn latent_unary_derivatives_log1mexp_negative(
    x: f64,
    derivative_order: usize,
    context: &str,
) -> Result<[f64; 5], LatentSurvivalError> {
    assert!(derivative_order <= 4);
    if !(x.is_finite() && x < 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!("{context} requires a finite negative log-boundary gap, got {x:?}"),
        });
    }
    let value = log1mexp_positive(-x);
    let exp_x = x.exp();
    let log_derivative_magnitudes = [
        x - value,
        x - 2.0 * value,
        x + exp_x.ln_1p() - 3.0 * value,
        x + (exp_x * (4.0 + exp_x)).ln_1p() - 4.0 * value,
    ];
    let mut derivatives = [value, 0.0, 0.0, 0.0, 0.0];
    for (offset, log_magnitude) in log_derivative_magnitudes
        .into_iter()
        .take(derivative_order)
        .enumerate()
    {
        let order = offset + 1;
        let magnitude = log_magnitude.exp();
        if !magnitude.is_finite() {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!(
                    "{context} derivative order {order} is not representable at \
                     log-boundary gap {x:?} (log magnitude {log_magnitude:?})"
                ),
            });
        }
        derivatives[order] = -magnitude;
    }
    Ok(derivatives)
}

/// Stable jet for `log(exp(log_left + c_left) - exp(log_right + c_right))`.
///
/// Positivity implies the log-domain gap
/// `delta = (log_right + c_right) - (log_left + c_left)` is negative, so
///
/// ```text
/// log(A - B) = log(A) + log(1 - exp(delta)).
/// ```
///
/// Absolute mass never leaves log space. The caller supplies a complete
/// finiteness predicate for its concrete jet representation so an `Ok` result
/// certifies every carried channel, including contracted third/fourth parts.
fn latent_survival_positive_log_difference_jet<J: JetField>(
    log_left: &J,
    log_coefficient_left: f64,
    log_right: &J,
    log_coefficient_right: f64,
    derivative_order: usize,
    context: &str,
    all_channels_finite: impl Fn(&J) -> bool,
) -> Result<J, LatentSurvivalError> {
    let weighted_left = log_left.add(&log_left.constant_like(log_coefficient_left));
    let weighted_right = log_right.add(&log_right.constant_like(log_coefficient_right));
    let delta = weighted_right.sub(&weighted_left);
    let delta_value = delta.value();
    if !(delta_value.is_finite() && delta_value < 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} must be a positive survival-mass difference: \
                 log(c_L*K0(M_L))={:?}, log(c_R*K0(M_R))={:?}; \
                 require M_L < M_R (i.e. L < R)",
                weighted_left.value(),
                weighted_right.value(),
            ),
        });
    }
    let derivatives =
        latent_unary_derivatives_log1mexp_negative(delta_value, derivative_order, context)?;
    let out = weighted_left.add(&delta.compose_unary(derivatives));
    if !all_channels_finite(&out) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} derivative jet is not representable at log-boundary gap {delta_value:?}"
            ),
        });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct LatentKernelPrimaryTerm {
    coeff: f64,
    q_exp: usize,
    qdot_power: usize,
    tau_exp: usize,
    k: usize,
}

/// One signed magnitude kept in logarithmic coordinates.
///
/// The latent-kernel recurrence naturally produces derivatives as signed
/// log-sums.  Keeping that representation through the moment-to-cumulant
/// conversion is essential: materialising `S_ab / S` and `S_a / S` separately
/// before computing `S_ab / S - (S_a / S)(S_b / S)` rounds both large moments
/// and destroys the small curvature left by their cancellation.
#[derive(Clone, Copy, Debug)]
struct LatentSignedLog {
    log_abs: f64,
    sign: f64,
}

impl LatentSignedLog {
    const ZERO: Self = Self {
        log_abs: f64::NEG_INFINITY,
        sign: 0.0,
    };
    const ONE: Self = Self {
        log_abs: 0.0,
        sign: 1.0,
    };
}

// A pointed cumulant over `r` slots consists of one leading moment and every
// proper pointed block of sizes `1..r-1`. Multiplication by the complementary
// rounded moment can at most double an exact expansion's length:
//
// C₁ = 1
// C₂ = 1 + 1·2C₁ = 3
// C₃ = 1 + 1·2C₁ + 2·2C₂ = 15
// C₄ = 1 + 1·2C₁ + 3·2C₂ + 3·2C₃ = 111.
//
// Certifying the final rounding subtracts one candidate binary64, requiring
// exactly one additional component. These are structural support bounds, not
// tunable numerical capacities.
const LATENT_EXACT_EXPANSION_ORDER1: usize = 1;
const LATENT_EXACT_EXPANSION_ORDER2: usize = 1 + 2 * LATENT_EXACT_EXPANSION_ORDER1;
const LATENT_EXACT_EXPANSION_ORDER3: usize =
    1 + 2 * LATENT_EXACT_EXPANSION_ORDER1 + 2 * 2 * LATENT_EXACT_EXPANSION_ORDER2;
const LATENT_EXACT_EXPANSION_ORDER4: usize = 1
    + 2 * LATENT_EXACT_EXPANSION_ORDER1
    + 3 * 2 * LATENT_EXACT_EXPANSION_ORDER2
    + 3 * 2 * LATENT_EXACT_EXPANSION_ORDER3;
const LATENT_EXACT_EXPANSION_CAPACITY: usize = LATENT_EXACT_EXPANSION_ORDER4 + 1;

/// The one refusal that replaces the pre-gam#2714 product refusal: not "a
/// monomial underflowed" (which is a fact about binary64 and says nothing about
/// the derivative) but "what binary64 could not carry is big enough to decide
/// the answer", which is the statement a certification is entitled to make.
const LATENT_UNREPRESENTABLE_MASS_REACHES_CELL: &str =
    "the product mass binary64 cannot represent reaches the cumulant's rounding cell";
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER1 == 1);
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER2 == 3);
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER3 == 15);
const _: () = assert!(LATENT_EXACT_EXPANSION_ORDER4 == 111);
const _: () = assert!(LATENT_EXACT_EXPANSION_CAPACITY == 112);

/// Fixed, increasing-magnitude floating-point expansion.
///
/// Each component is an exact binary64 and their real sum is the represented
/// value TO WITHIN [`Self::unrepresentable_mass`]. `TwoSum` and FMA
/// `TwoProduct` retain every arithmetic residual the format can hold, so the
/// pointed cumulant polynomial is evaluated exactly over its rounded binary64
/// moments wherever binary64 is capable of it. The sole rounding occurs in
/// [`Self::certified_round`], which proves its cell against the components AND
/// that mass.
#[derive(Clone, Copy)]
struct LatentExactExpansion {
    components: [f64; LATENT_EXACT_EXPANSION_CAPACITY],
    len: usize,
    /// Outward upper bound on `|exact value − Σ components|` (gam#2714).
    ///
    /// `TwoSum` is exact for every finite pair, and `TwoProduct` is exact
    /// whenever the product stays at or above `2^-970`, so this is `0.0` on
    /// every expansion the recurrence could already build. It becomes nonzero
    /// exactly where an FMA residual needs bits under `2^-1074` — see
    /// [`Self::two_product`] for the bound and its proof — and it is the reason
    /// a monomial binary64 cannot carry no longer refuses a derivative binary64
    /// can round perfectly well.
    unrepresentable_mass: f64,
}

#[derive(Clone, Copy, Debug)]
struct LatentCertifiedCumulant {
    /// Unique nearest binary64 rounding of the exact recurrence.
    value: f64,
    /// Outward upper bound on the sum of absolute expanded monomials.
    ///
    /// This is the numerator of the componentwise condition number and lets an
    /// independent floating implementation derive its own forward-error band
    /// without comparing against production or fitting a tolerance.
    absolute_term_mass: f64,
}

impl LatentCertifiedCumulant {
    const ZERO: Self = Self {
        value: 0.0,
        absolute_term_mass: 0.0,
    };
}

impl LatentExactExpansion {
    const ZERO: Self = Self {
        components: [0.0; LATENT_EXACT_EXPANSION_CAPACITY],
        len: 0,
        unrepresentable_mass: 0.0,
    };

    /// One subnormal ulp — the outward bound on a single FMA product residual
    /// that binary64 cannot represent (gam#2714).
    ///
    /// In the guarded band `|p| = |fl(a·b)| < 2^-970` the exact residual obeys
    /// `|r| = |a·b − p| ≤ ulp(p)/2 ≤ 2^-1023`, so `r` lies inside the subnormal
    /// range where the grid spacing is exactly `2^-1074`. `fma(a, b, −p)`
    /// returns `r` correctly rounded onto that grid, so its error is at most a
    /// half spacing, `2^-1075`. That half is not itself representable, so the
    /// bound carried here is the full spacing — strictly conservative, and the
    /// smallest positive binary64 there is. The completely-underflowed case
    /// `p = 0` is the same statement with `|a·b| ≤ 2^-1075`.
    const SUBNORMAL_ULP: f64 = f64::from_bits(1);

    /// Round a nonnegative bound outward, so an accumulated bound stays a
    /// bound under binary64 addition and multiplication.
    ///
    /// Exactly zero is returned unchanged: an expansion that has lost nothing
    /// must not acquire a bound merely by being added or scaled, or the "no
    /// underflow ⇒ byte-identical" property would not hold.
    #[inline]
    fn outward_bound(value: f64) -> Result<f64, &'static str> {
        if !value.is_finite() || value < 0.0 {
            return Err("an unrepresentable-mass bound left the finite nonnegative range");
        }
        if value == 0.0 {
            return Ok(0.0);
        }
        Ok(Self::next_up(value))
    }

    fn scalar(value: f64) -> Self {
        if value == 0.0 {
            Self::ZERO
        } else {
            let mut out = Self::ZERO;
            out.components[0] = value;
            out.len = 1;
            out
        }
    }

    #[inline]
    fn component(&self, index: usize) -> f64 {
        self.components[index]
    }

    fn push_nonzero(&mut self, value: f64) -> Result<(), &'static str> {
        if value == 0.0 {
            return Ok(());
        }
        if !value.is_finite() {
            return Err("an exact-expansion component became non-finite");
        }
        if self.len == LATENT_EXACT_EXPANSION_CAPACITY {
            return Err("the structurally bounded exact expansion exhausted its capacity");
        }
        self.components[self.len] = value;
        self.len += 1;
        Ok(())
    }

    /// Error-free `a + b = sum + error`.
    #[inline]
    fn two_sum(a: f64, b: f64) -> Result<(f64, f64), &'static str> {
        let sum = a + b;
        if !sum.is_finite() {
            return Err("an exact-expansion addition overflowed");
        }
        let virtual_b = sum - a;
        let virtual_a = sum - virtual_b;
        let error = (a - virtual_a) + (b - virtual_b);
        Ok((sum, error))
    }

    /// Finite product through one fused residual, with an outward bound on the
    /// part of the exact product binary64 cannot carry.
    ///
    /// An FMA product residual is EXACT provided neither the product nor its
    /// residual underflows: a product at least `2^-970` has an exact-product
    /// least-significant bit no smaller than `2^-1074` for binary64 operands,
    /// so `mul_add(a, b, -product)` IS `a·b − product` with no rounding, and
    /// the returned bound is `0.0`.
    ///
    /// Below that proved range the residual can need bits the format does not
    /// have. **That is a fact about binary64, not about the derivative**
    /// (gam#2714). This used to `Err` there, which threw away an entire
    /// certified cumulant over one monomial — and the monomials that land there
    /// are, by construction, the ones too small to reach the answer: two
    /// perfectly ordinary normal moments at `1e-152` multiply to `1e-304`, 252
    /// orders below the half-ulp of a leading moment of `1`. Measured on the
    /// #2714 witness, five of seven outer seeds of the veteran latent-frailty
    /// fit died on exactly that. It is the same shape as the tie-to-even
    /// refusal fixed for gam#2538 in [`Self::certified_round`]: the
    /// certification refusing precisely the inputs it can decide.
    ///
    /// So the residual is KEPT — it is still the correctly rounded value of
    /// `a·b − product`, which is the best binary64 has — and the mass it may be
    /// off by is carried outward in [`Self::unrepresentable_mass`] and proved
    /// against the rounding cell at the end. See [`Self::SUBNORMAL_ULP`] for
    /// the bound. Nothing is loosened: an expansion whose products all clear
    /// `2^-970` carries a bound of exactly zero and certifies bit-identically.
    #[inline]
    fn two_product(a: f64, b: f64) -> Result<(f64, f64, f64), &'static str> {
        if a == 0.0 || b == 0.0 {
            return Ok((0.0, 0.0, 0.0));
        }
        let product = a * b;
        if !product.is_finite() {
            return Err("an exact-expansion product overflowed");
        }
        let error = a.mul_add(b, -product);
        if !error.is_finite() {
            return Err("an exact-expansion product residual became non-finite");
        }
        let unrepresentable = if product.abs() < f64::MIN_POSITIVE / f64::EPSILON {
            Self::SUBNORMAL_ULP
        } else {
            0.0
        };
        Ok((product, error, unrepresentable))
    }

    /// Error-free sum of two increasing-magnitude expansions.
    ///
    /// `TwoSum` is exact for every finite pair, so the sum adds no
    /// unrepresentable mass of its own; the two operands' bounds simply add.
    fn add(self, other: Self) -> Result<Self, &'static str> {
        let carried = Self::outward_bound(self.unrepresentable_mass + other.unrepresentable_mass)?;
        if self.len == 0 {
            return Ok(Self {
                unrepresentable_mass: carried,
                ..other
            });
        }
        if other.len == 0 {
            return Ok(Self {
                unrepresentable_mass: carried,
                ..self
            });
        }
        if self.len + other.len > LATENT_EXACT_EXPANSION_CAPACITY {
            return Err("an exact-expansion sum exceeded its structural support bound");
        }

        let mut left = 0usize;
        let mut right = 0usize;
        let take_left = |left_value: f64, right_value: f64| {
            left_value.abs() <= right_value.abs()
        };
        let mut q = if take_left(self.component(left), other.component(right)) {
            let value = self.component(left);
            left += 1;
            value
        } else {
            let value = other.component(right);
            right += 1;
            value
        };
        let mut out = Self::ZERO;

        while left < self.len || right < other.len {
            let next = if right == other.len
                || (left < self.len
                    && take_left(self.component(left), other.component(right)))
            {
                let value = self.component(left);
                left += 1;
                value
            } else {
                let value = other.component(right);
                right += 1;
                value
            };
            let (sum, error) = Self::two_sum(q, next)?;
            out.push_nonzero(error)?;
            q = sum;
        }
        out.push_nonzero(q)?;
        out.unrepresentable_mass = carried;
        Ok(out)
    }

    /// Multiplication by one rounded binary64 moment, exact wherever binary64
    /// can be (see [`Self::two_product`]).
    ///
    /// The incoming bound scales with the multiplier — `|exact − Σc| ≤ m` gives
    /// `|s·exact − Σ(s·c)| ≤ |s|·m` — and every product residual the format
    /// cannot hold adds its own [`Self::SUBNORMAL_ULP`] on top.
    fn scale(self, scalar: f64) -> Result<Self, &'static str> {
        if self.len == 0 || scalar == 0.0 {
            return Ok(Self::ZERO);
        }
        if self.len * 2 > LATENT_EXACT_EXPANSION_CAPACITY {
            return Err("a scaled exact expansion exceeded its structural support bound");
        }

        let mut unrepresentable =
            Self::outward_bound(self.unrepresentable_mass * scalar.abs())?;
        let (mut q, first_error, first_unrepresentable) =
            Self::two_product(self.component(0), scalar)?;
        unrepresentable = Self::outward_bound(unrepresentable + first_unrepresentable)?;
        let mut out = Self::ZERO;
        out.push_nonzero(first_error)?;
        for index in 1..self.len {
            let (product, product_error, product_unrepresentable) =
                Self::two_product(self.component(index), scalar)?;
            unrepresentable = Self::outward_bound(unrepresentable + product_unrepresentable)?;
            let (sum, sum_error) = Self::two_sum(q, product_error)?;
            out.push_nonzero(sum_error)?;
            // The exact scaling recurrence guarantees `|product| >= |sum|`;
            // generic TwoSum keeps the identity valid without relying on that
            // ordering as a runtime premise.
            let (next_q, product_sum_error) = Self::two_sum(product, sum)?;
            out.push_nonzero(product_sum_error)?;
            q = next_q;
        }
        out.push_nonzero(q)?;
        out.unrepresentable_mass = unrepresentable;
        Ok(out)
    }

    #[inline]
    fn next_up(value: f64) -> f64 {
        if value.is_nan() || value == f64::INFINITY {
            value
        } else if value == 0.0 {
            f64::from_bits(1)
        } else if value > 0.0 {
            f64::from_bits(value.to_bits() + 1)
        } else {
            f64::from_bits(value.to_bits() - 1)
        }
    }

    #[inline]
    fn next_down(value: f64) -> f64 {
        -Self::next_up(-value)
    }

    /// Sign of the exact expansion after adding one binary64 scalar.
    ///
    /// Every finite binary64 is an integer multiple of 2^-1074. Accumulating
    /// positive and negative significands into separate fixed unsigned integers
    /// and comparing those integers is therefore an exact, order-independent
    /// sign decision. It does not assume that a preceding error-free transform
    /// happened to retain a particular nonoverlap strength.
    fn exact_sign_after_adding_scalar(self, scalar: f64) -> Result<f64, &'static str> {
        self.exact_sign_after_adding_scalars(scalar, 0.0)
    }

    /// As [`Self::exact_sign_after_adding_scalar`], with two addends.
    ///
    /// The second is the unrepresentable-mass offset that turns a statement
    /// about `Σ components` into a statement about the whole interval the exact
    /// value is known to lie in (gam#2714). Both go through the same exact
    /// fixed-point accumulator, so nothing about the decision becomes a
    /// tolerance comparison.
    fn exact_sign_after_adding_scalars(
        self,
        first: f64,
        second: f64,
    ) -> Result<f64, &'static str> {
        let ordering = exact_binary64_sum_sign(
            self.components[..self.len]
                .iter()
                .copied()
                .chain([first, second]),
        )
        .map_err(|_| "the shared exact-sign accumulator rejected its structural finite input")?;
        Ok(match ordering {
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 0.0,
            std::cmp::Ordering::Greater => 1.0,
        })
    }

    /// Half the distance from `value` to whichever of its two binary64
    /// neighbours is nearer — the radius of its rounding cell.
    #[inline]
    fn rounding_cell_radius(value: f64) -> f64 {
        let up = Self::next_up(value) - value;
        let down = value - Self::next_down(value);
        0.5 * up.min(down)
    }

    /// `candidate` is the answer iff the whole interval the exact value is
    /// known to lie in — `Σ components ± mass` — sits inside its rounding cell.
    ///
    /// Used where the components sum to `candidate` EXACTLY, so the only thing
    /// separating the two is the mass binary64 could not carry (gam#2714).
    fn certify_exact_sum_against_mass(candidate: f64, mass: f64) -> Result<f64, &'static str> {
        if mass == 0.0 {
            return Ok(candidate);
        }
        if mass < Self::rounding_cell_radius(candidate) {
            return Ok(candidate);
        }
        Err(LATENT_UNREPRESENTABLE_MASS_REACHES_CELL)
    }

    /// Unique nearest-even binary64 rounding, proved against both adjacent
    /// midpoint boundaries and against the mass binary64 could not carry.
    ///
    /// The starting point is the accumulated sum of the components, which is
    /// only a neighbourhood of the exact value. Where it is not the nearest
    /// binary64 the boundary comparison says so exactly, and the walk below
    /// steps one ulp onto the neighbour that comparison names, so the returned
    /// value is the correctly rounded one regardless of how the accumulation
    /// happened to round.
    ///
    /// gam#2714 adds the second half of the proof. `Σ components` is the exact
    /// value only to within [`Self::unrepresentable_mass`], so every boundary
    /// comparison is made at the END of that interval that is nearest the
    /// boundary. `mass == 0` — every expansion whose products all cleared
    /// `2^-970`, which is every expansion this routine could previously be
    /// handed at all — reduces each comparison to the one made before, so the
    /// returned value is bit-identical there.
    fn certified_round(self) -> Result<f64, &'static str> {
        let mass = self.unrepresentable_mass;
        if !(mass.is_finite() && mass >= 0.0) {
            return Err("an unrepresentable-mass bound left the finite nonnegative range");
        }
        if self.len == 0 {
            return Self::certify_exact_sum_against_mass(0.0, mass);
        }
        let mut candidate = 0.0_f64;
        for index in 0..self.len {
            candidate += self.component(index);
        }
        if !candidate.is_finite() {
            return Err("the exact cumulant is outside the finite binary64 range");
        }

        // The accumulation above rounds once per component, so `candidate` is a
        // NEIGHBOURHOOD of the exact value, not provably its nearest binary64.
        // When it is not, the certification below can say so exactly -- the
        // residual expansion and the sign accumulator are both exact -- and the
        // honest response is to MOVE to the neighbour it names rather than to
        // refuse. A step is taken only when the exact value lies strictly
        // beyond the midpoint separating `candidate` from `adjacent`, which is
        // precisely the statement that `adjacent` is strictly nearer, so the
        // walk contracts by one ulp per step, never reverses direction, and is
        // bounded by the (finite) ulp distance to the exact value. The step cap
        // is a runaway backstop, not the decision: it is the same structural
        // support bound that bounds the number of roundings the accumulation
        // above could have committed, since each component contributes at most
        // one.
        let mut walked = 0usize;
        loop {
            let residual = self.add(Self::scalar(-candidate))?;
            if residual.len == 0 {
                return Self::certify_exact_sum_against_mass(candidate, mass);
            }
            let residual_sign = residual.exact_sign_after_adding_scalar(0.0)?;
            if residual_sign == 0.0 {
                // Nonzero components summing to exactly zero: `candidate` IS the
                // components' sum, and only the unrepresentable mass separates
                // it from the exact value.
                return Self::certify_exact_sum_against_mass(candidate, mass);
            }
            let adjacent = if residual_sign > 0.0 {
                Self::next_up(candidate)
            } else {
                Self::next_down(candidate)
            };
            if !adjacent.is_finite() {
                return Err("the exact cumulant is outside the finite binary64 range");
            }
            let midpoint_distance = 0.5 * (adjacent - candidate).abs();
            if midpoint_distance == 0.0 {
                return Err("the exact cumulant reached an unrepresentable subnormal midpoint");
            }
            let boundary_offset = -residual_sign * midpoint_distance;
            // The components' sum is inside `candidate`'s cell; the exact value
            // is inside it too only if the mass binary64 could not carry fits
            // in what is left (gam#2714). `mass == 0` makes the widened
            // comparison the same exact-sign call as the plain one.
            let cell_holds_the_mass = move |residual: Self| -> Result<f64, &'static str> {
                let widened = residual
                    .exact_sign_after_adding_scalars(boundary_offset, residual_sign * mass)?;
                if widened == -residual_sign {
                    Ok(candidate)
                } else {
                    Err(LATENT_UNREPRESENTABLE_MASS_REACHES_CELL)
                }
            };
            match residual
                .exact_sign_after_adding_scalar(boundary_offset)?
                .partial_cmp(&0.0)
            {
                Some(std::cmp::Ordering::Less) if residual_sign > 0.0 => {
                    return cell_holds_the_mass(residual);
                }
                Some(std::cmp::Ordering::Greater) if residual_sign < 0.0 => {
                    return cell_holds_the_mass(residual);
                }
                Some(std::cmp::Ordering::Equal) if mass != 0.0 => {
                    // The components land exactly on the midpoint, so the
                    // exact value straddles it by the mass. A tie is a fact
                    // about a value that is KNOWN; this one is not.
                    return Err(LATENT_UNREPRESENTABLE_MASS_REACHES_CELL);
                }
                Some(std::cmp::Ordering::Equal) => {
                    // An exact midpoint is not an unroundable value. IEEE-754's
                    // default mode -- round to nearest, ties to EVEN -- makes it
                    // unique, and it is the mode every binary64 operation
                    // downstream of this one already uses. This function's own doc
                    // comment names that rule ("Unique nearest-even binary64
                    // rounding"), so refusing here contradicted the contract and
                    // was stricter than the arithmetic the result feeds.
                    //
                    // It is not a measure-zero case in practice. Measured on
                    // gam#2538 at current main: the latent-survival frailty fit
                    // rejects ALL SEVEN outer seeds in 0.49 s with
                    // `latent survival numerator derivative mask 0b0011 has no
                    // certified binary64 value`, because at the beta = 0 outer
                    // seed the cumulant is a small dyadic rational and lands
                    // exactly on a midpoint. The certification refused precisely
                    // the inputs it can round exactly.
                    //
                    // `candidate` and `adjacent` are bit-adjacent -- `next_up` /
                    // `next_down` are `to_bits() +/- 1` -- so exactly one of the
                    // two has an even significand, and the low bit of the pattern
                    // is that significand's LSB. Selecting the even one IS the
                    // tie-to-even neighbour; no tolerance and no choice enter.
                    return Ok(if candidate.to_bits() % 2 == 0 {
                        candidate
                    } else {
                        adjacent
                    });
                }
                _ => {
                    // The exact value lies AT OR BEYOND the far side of the
                    // midpoint, so `adjacent` is strictly nearer than `candidate`:
                    // the accumulated sum simply missed the nearest binary64. Step
                    // onto the neighbour the exact comparison named and certify
                    // again. Nothing here is a tolerance -- the step is decided by
                    // the same exact sign accumulator that decides the return.
                    if walked == LATENT_EXACT_EXPANSION_CAPACITY {
                        return Err(
                            "the exact cumulant stayed outside its rounding cell after a full \
                             structural walk",
                        );
                    }
                    walked += 1;
                    candidate = adjacent;
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct LatentSignedLogOrder2<const K: usize> {
    v: LatentSignedLog,
    g: [LatentSignedLog; K],
    h: [[LatentSignedLog; K]; K],
}

impl<const K: usize> LatentSignedLogOrder2<K> {
    const fn zero() -> Self {
        Self {
            v: LatentSignedLog::ZERO,
            g: [LatentSignedLog::ZERO; K],
            h: [[LatentSignedLog::ZERO; K]; K],
        }
    }
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

/// Complete primary-coordinate state for one latent-survival row.
///
/// Keeping these coupled channels together prevents boundary reordering and
/// cross-row mean/scale mismatches when selecting a derivative backend.
/// Public because [`latent_survival_log_sigma_curvature_certified`] takes it: a
/// certificate a caller cannot invoke is not an export. Widening this was the API
/// decision that function deferred, and the compiler was right to force it —
/// `pub(crate)` made the whole certificate dead code, which is the build saying
/// "this has no consumer" rather than a lint to route around (#2566).
#[derive(Clone, Copy, Debug)]
pub struct LatentSurvivalPrimaryPoint {
    pub q_entry: f64,
    pub q_exit: f64,
    pub qdot_exit: f64,
    pub q_right: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl LatentSurvivalPrimaryPoint {
    /// Log-scale factor used only by derivatives along the learnable-sigma
    /// coordinate. Fixed zero frailty has no such coordinate, so its factor is
    /// the neutral finite placeholder; every other invalid scale remains NaN
    /// or infinite and is rejected by the kernel's numerical checks.
    #[inline]
    fn log_sigma_factor(self) -> f64 {
        if self.sigma == 0.0 {
            0.0
        } else {
            self.sigma.ln()
        }
    }
}

#[cfg(test)]
mod tests_kernel_recurrence {
    use super::*;
    use std::collections::BTreeMap;

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

    pub(super) fn latent_kernel_differentiate_terms(
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
}

// Fourth-order latent-kernel recurrences have a small finite support. Keeping
// the sorted support inline removes the BTreeMap node allocation and output-Vec
// allocation that the pre-cutover directional oracle intentionally retains.
// The all-event/K=5/K=6 oracle below asserts this capacity never spills.
const LATENT_TERM_INLINE_CAPACITY: usize = 64;
type LatentTermBuffer = SmallVec<[LatentKernelPrimaryTerm; LATENT_TERM_INLINE_CAPACITY]>;

/// Highest kernel rung, `σ`-power, and `qdot`-power the `∂_a` basis handles.
///
/// These are structural bounds, not tolerances: the rung bound keeps every
/// falling-factorial coefficient an exact f64 integer (`|s(12,1)| = 11! =
/// 39916800`), and the other two size the accumulation table. A term list that
/// exceeds any of them routes to the rung basis instead, which is always
/// available.
const LATENT_A_BASIS_MAX_RUNG: usize = 12;
const LATENT_A_BASIS_MAX_TAU_EXP: usize = 12;
const LATENT_A_BASIS_MAX_QDOT_POWER: usize = 4;

/// Signed Stirling numbers of the first kind: the coefficients of the falling
/// factorial `(x)_k = x(x−1)···(x−k+1) = Σ_j s(k,j) x^j`.
///
/// These are the weights that re-express a kernel rung in the `∂_a^j K_0` basis,
/// because `m^k K_k = (−1)^k (∂_a)_k K_0` (#2610). Built by the standard
/// recurrence `(x)_{k+1} = (x)_k · (x − k)` in integers so every entry converts
/// to f64 exactly.
const fn latent_falling_factorial_table()
-> [[i64; LATENT_A_BASIS_MAX_RUNG + 1]; LATENT_A_BASIS_MAX_RUNG + 1] {
    let mut table = [[0_i64; LATENT_A_BASIS_MAX_RUNG + 1]; LATENT_A_BASIS_MAX_RUNG + 1];
    table[0][0] = 1;
    let mut rung = 0usize;
    while rung < LATENT_A_BASIS_MAX_RUNG {
        let mut power = 0usize;
        while power <= rung {
            let value = table[rung][power];
            if value != 0 {
                table[rung + 1][power + 1] += value;
                table[rung + 1][power] -= (rung as i64) * value;
            }
            power += 1;
        }
        rung += 1;
    }
    table
}

const LATENT_FALLING_FACTORIAL: [[i64; LATENT_A_BASIS_MAX_RUNG + 1];
    LATENT_A_BASIS_MAX_RUNG + 1] = latent_falling_factorial_table();

/// Evaluate one latent kernel term list as a signed log magnitude.
///
/// Shared by the packed order-two production path and the multi-direction
/// oracle so the two cannot drift in either the basis they use or the
/// accumulation order they use it in.
fn latent_kernel_evaluate_terms(
    bundle: &LogLognormalKernelBundle,
    state: LatentKernelPrimaryState,
    terms: &[LatentKernelPrimaryTerm],
    context: &str,
) -> Result<(f64, f64), LatentSurvivalError> {
    let needs_qdot = terms
        .iter()
        .any(|term| term.coeff != 0.0 && term.qdot_power > 0);
    if needs_qdot && !(state.qdot.is_finite() && state.qdot > 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} requires positive finite qdot for exact-event directional terms, got {}",
                state.qdot
            ),
        });
    }
    let log_qdot = if needs_qdot { state.qdot.ln() } else { 0.0 };
    if let Some(sum) = latent_kernel_evaluate_terms_in_a_basis(bundle, state, terms, log_qdot) {
        return Ok(sum);
    }
    let mut log_mags = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    let mut signs = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        log_mags.push(
            term.coeff.abs().ln()
                + term.q_exp as f64 * state.q
                + term.tau_exp as f64 * state.log_sigma_factor
                + term.qdot_power as f64 * log_qdot
                + bundle.get(term.k),
        );
        signs.push(term.coeff.signum());
    }
    if log_mags.is_empty() {
        return Ok((f64::NEG_INFINITY, 0.0));
    }
    Ok(signed_log_sum_exp(&log_mags, &signs))
}

/// The same term list evaluated in the `∂_a^j K_0` basis, or `None` when that
/// basis is unavailable for this bundle or this term list (#2610).
///
/// The rung basis and this one are related by an exact integer change of basis,
/// so this is not an approximation — it is the same sum with the cancellation
/// performed on the COEFFICIENTS instead of on the values. That is the whole
/// repair. `∂_{log σ} K_0` reaches the rung basis as `σ²(−mK_1 + m²K_2)`, whose
/// two summands agree to `~1/(120σ²)` of their own size; in this basis the
/// identical quantity is `σ² ∂_a² K_0`, one entry, because the `∂_a` coefficient
/// cancels ANALYTICALLY when like powers are collected. Past `log σ ≈ 5.4` that
/// is the difference between a curvature and its roundoff.
///
/// Requires `q_exp == k` on every term. That is not a restriction in practice:
/// the differentiation rules shift `q_exp` and `k` in lockstep, and every base
/// term the row expression builds starts on the diagonal, so the whole
/// derivative tree stays there. A term off the diagonal routes to the rung
/// basis rather than being handled approximately.
fn latent_kernel_evaluate_terms_in_a_basis(
    bundle: &LogLognormalKernelBundle,
    state: LatentKernelPrimaryState,
    terms: &[LatentKernelPrimaryTerm],
    log_qdot: f64,
) -> Option<(f64, f64)> {
    let tower = bundle.log_scaled_a_derivatives.as_ref()?;
    let mut max_rung = 0usize;
    let mut max_tau_exp = 0usize;
    let mut max_qdot_power = 0usize;
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        if term.q_exp != term.k
            || term.k >= tower.len()
            || term.k > LATENT_A_BASIS_MAX_RUNG
            || term.tau_exp > LATENT_A_BASIS_MAX_TAU_EXP
            || term.qdot_power > LATENT_A_BASIS_MAX_QDOT_POWER
        {
            return None;
        }
        max_rung = max_rung.max(term.k);
        max_tau_exp = max_tau_exp.max(term.tau_exp);
        max_qdot_power = max_qdot_power.max(term.qdot_power);
    }
    // One accumulator per `(qdot_power, tau_exp, j)` monomial. Terms sharing a
    // cell share every factor except the integer coefficient, so collecting them
    // here is where the analytic cancellation happens — exactly, in integers
    // scaled by one common power of σ.
    let rung_stride = max_rung + 1;
    let tau_stride = max_tau_exp + 1;
    let mut coefficients =
        SmallVec::<[f64; 256]>::from_elem(0.0, (max_qdot_power + 1) * tau_stride * rung_stride);
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        let rung_parity = if term.k % 2 == 0 { 1.0 } else { -1.0 };
        let cell = (term.qdot_power * tau_stride + term.tau_exp) * rung_stride;
        for power in 0..=term.k {
            let stirling = LATENT_FALLING_FACTORIAL[term.k][power];
            if stirling != 0 {
                coefficients[cell + power] += rung_parity * term.coeff * stirling as f64;
            }
        }
    }
    let mut log_mags = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    let mut signs = SmallVec::<[f64; LATENT_TERM_INLINE_CAPACITY]>::new();
    for qdot_power in 0..=max_qdot_power {
        for tau_exp in 0..=max_tau_exp {
            for power in 0..=max_rung {
                let coefficient =
                    coefficients[(qdot_power * tau_stride + tau_exp) * rung_stride + power];
                let entry = tower[power];
                if coefficient == 0.0 || entry.sign == 0.0 {
                    continue;
                }
                // `tower` holds `σ^j ∂_a^j K_0`, so the stored `σ^j` is divided
                // back out alongside the term's own `σ^{tau_exp}`.
                log_mags.push(
                    coefficient.abs().ln()
                        + (tau_exp as f64 - power as f64) * state.log_sigma_factor
                        + qdot_power as f64 * log_qdot
                        + entry.log_abs,
                );
                signs.push(coefficient.signum() * entry.sign);
            }
        }
    }
    if log_mags.is_empty() {
        return Some((f64::NEG_INFINITY, 0.0));
    }
    Some(signed_log_sum_exp(&log_mags, &signs))
}

#[inline]
fn latent_kernel_accumulate_term_inline(
    terms: &mut LatentTermBuffer,
    term: LatentKernelPrimaryTerm,
    scale: f64,
) {
    if scale == 0.0 || term.coeff == 0.0 {
        return;
    }
    let contribution = scale * term.coeff;
    if let Some(existing) = terms.iter_mut().find(|existing| {
        existing.q_exp == term.q_exp
            && existing.qdot_power == term.qdot_power
            && existing.tau_exp == term.tau_exp
            && existing.k == term.k
    }) {
        existing.coeff += contribution;
    } else {
        terms.push(LatentKernelPrimaryTerm {
            coeff: contribution,
            ..term
        });
    }
}

fn latent_kernel_differentiate_terms_inline(
    terms: &[LatentKernelPrimaryTerm],
    dir: LatentKernelPrimaryDirection,
) -> LatentTermBuffer {
    let mut out = LatentTermBuffer::new();
    for term in terms {
        if dir.dq != 0.0 {
            if term.q_exp > 0 {
                latent_kernel_accumulate_term_inline(&mut out, *term, dir.dq * term.q_exp as f64);
            }
            latent_kernel_accumulate_term_inline(
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
                latent_kernel_accumulate_term_inline(&mut out, *term, dir.dmu * term.k as f64);
            }
            latent_kernel_accumulate_term_inline(
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
                latent_kernel_accumulate_term_inline(
                    &mut out,
                    *term,
                    dir.dtau * term.tau_exp as f64,
                );
            }
            let kf = term.k as f64;
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    tau_exp: term.tau_exp + 2,
                    ..*term
                },
                dir.dtau * kf * kf,
            );
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    q_exp: term.q_exp + 1,
                    tau_exp: term.tau_exp + 2,
                    k: term.k + 1,
                    ..*term
                },
                -dir.dtau * (2.0 * kf + 1.0),
            );
            latent_kernel_accumulate_term_inline(
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
            latent_kernel_accumulate_term_inline(
                &mut out,
                LatentKernelPrimaryTerm {
                    qdot_power: term.qdot_power - 1,
                    ..*term
                },
                dir.dqd * term.qdot_power as f64,
            );
        }
    }
    out.retain(|term| term.coeff != 0.0);
    out.sort_unstable_by_key(|term| (term.q_exp, term.qdot_power, term.tau_exp, term.k));
    out
}

fn latent_kernel_term_sequence_inline(
    base_terms: &[LatentKernelPrimaryTerm],
    axes: &[LatentKernelPrimaryDirection],
    suffix: &[LatentKernelPrimaryDirection],
) -> LatentTermBuffer {
    let mut terms = LatentTermBuffer::from_slice(base_terms);
    terms.retain(|term| term.coeff != 0.0);
    // The canonical subset-cache recurrence strips the least-significant
    // selected slot and applies it after recursively building the remaining
    // mask. Its deterministic floating-point order is therefore highest slot
    // to lowest slot. Preserve that order here so the allocation-free packed
    // path and the independent MultiDir layout accumulate identical analytic
    // coefficients, including cancellation-heavy tail derivatives.
    for direction in axes.iter().chain(suffix.iter()).rev() {
        terms = latent_kernel_differentiate_terms_inline(&terms, *direction);
    }
    terms
}

#[cfg(test)]
mod tests_multidir_kernel {
    /// Derivatives of `log(x)` through fourth order at the only point needed by
    /// normalized kernel sums: the literal `x = 1`.
    ///
    /// Keeping the point in the function name and removing the free argument
    /// makes the representability contract structural. No caller can
    /// accidentally feed a small positive linear-domain mass into the
    /// reciprocal powers.
    #[inline]
    fn latent_unary_derivatives_log_at_one() -> [f64; 5] {
        [0.0, 1.0, -1.0, 2.0, -6.0]
    }

    use super::tests_kernel_recurrence::latent_kernel_differentiate_terms;
    use super::*;
    use gam_math::jet_partitions::MultiDirJet as LatentMultiDirJet;

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

    pub(super) fn latent_kernel_sum_log_jet(
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
        let bundle = log_kernel_bundle(quadctx, state.q.exp(), state.mu, state.sigma, max_k)
            .map_err(|e| LatentSurvivalError::NumericalFailure {
                reason: format!("{context} kernel evaluation failed: {e}"),
            })?;

        let evaluate_terms = |terms: &[LatentKernelPrimaryTerm]| {
            latent_kernel_evaluate_terms(&bundle, state, terms, context)
        };

        let (base_log_sum, base_sign) = evaluate_terms(&term_lists[0])?;
        if !(base_log_sum.is_finite() && base_sign > 0.0) {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!("{context} produced a non-positive signed kernel sum"),
            });
        }

        let mut normalized = LatentMultiDirJet::constant(directions.len(), 1.0);
        let mut signed_moments = [LatentSignedLog::ZERO; 16];
        signed_moments[0] = LatentSignedLog::ONE;
        for mask in 1..term_lists.len() {
            let (log_abs, sign) = evaluate_terms(&term_lists[mask])?;
            let moment =
                latent_signed_log_normalized(log_abs, sign, base_log_sum, context)?;
            signed_moments[mask] = moment;
            normalized.coeffs[mask] = latent_signed_log_materialize(moment, context)?;
        }

        let mut out = normalized.compose_unary(latent_unary_derivatives_log_at_one());
        out.coeffs[0] += base_log_sum;
        if term_lists.len() == 1 {
            return Ok(out);
        }

        // Independently grade the pre-cutover floating oracle against the exact
        // rounded-moment polynomial. `MultiDirJet::compose_unary` supports at
        // most four live slots here. Its K=4 pointed schedule walks 34 Dot2
        // terms (10 rounded operations each), writes 17 compensated power
        // outputs (5 operations each), and combines four powers in at most 32
        // operations: 457 total. Smaller orders are strict subsets, so this is
        // a structural uniform bound rather than an empirical multiplier.
        const MULTIDIR_DOT2_TERMS_K4: usize = 34;
        const MULTIDIR_DOT2_OPERATIONS: usize = 10;
        const MULTIDIR_POWER_OUTPUTS_K4: usize = 17;
        const MULTIDIR_POWER_OUTPUT_OPERATIONS: usize = 5;
        const MULTIDIR_COMBINE_OPERATIONS: usize = 32;
        const MULTIDIR_MAX_OPERATIONS: usize =
            MULTIDIR_DOT2_TERMS_K4 * MULTIDIR_DOT2_OPERATIONS
                + MULTIDIR_POWER_OUTPUTS_K4 * MULTIDIR_POWER_OUTPUT_OPERATIONS
                + MULTIDIR_COMBINE_OPERATIONS;
        const _: () = assert!(MULTIDIR_MAX_OPERATIONS == 457);
        let operation_roundoff = MULTIDIR_MAX_OPERATIONS as f64 * f64::EPSILON;
        let gamma = LatentExactExpansion::next_up(
            operation_roundoff / (1.0 - operation_roundoff),
        );
        let exact = latent_certified_cumulants(
            signed_moments,
            term_lists.len() - 1,
            context,
        )?;
        for mask in 1..term_lists.len() {
            let relative_roundoff =
                LatentExactExpansion::next_up(gamma * exact[mask].absolute_term_mass);
            let gradual_underflow =
                MULTIDIR_MAX_OPERATIONS as f64 * f64::from_bits(1);
            let arithmetic_bound =
                LatentExactExpansion::next_up(relative_roundoff + gradual_underflow);
            let error =
                LatentExactExpansion::next_up((out.coeffs[mask] - exact[mask].value).abs());
            assert!(
                error <= arithmetic_bound,
                "{context}: MultiDir mask {mask:#06b} escaped its derived forward-error \
                 certificate: got={:.17e}, exact-rounded={:.17e}, error={error:.17e}, \
                 bound={arithmetic_bound:.17e}, absolute-term-mass={:.17e}, operations={MULTIDIR_MAX_OPERATIONS}",
                out.coeffs[mask],
                exact[mask].value,
                exact[mask].absolute_term_mass,
            );
        }
        Ok(out)
    }
}

fn latent_signed_log_checked(
    log_abs: f64,
    sign: f64,
    context: &str,
    quantity: &str,
) -> Result<LatentSignedLog, LatentSurvivalError> {
    if log_abs == f64::NEG_INFINITY && sign == 0.0 {
        return Ok(LatentSignedLog::ZERO);
    }
    if log_abs.is_finite() && (sign == -1.0 || sign == 1.0) {
        return Ok(LatentSignedLog { log_abs, sign });
    }
    Err(LatentSurvivalError::NumericalFailure {
        reason: format!(
            "{context} produced an invalid signed-log {quantity}: log_abs={log_abs}, sign={sign}"
        ),
    })
}

fn latent_signed_log_normalized(
    log_abs: f64,
    sign: f64,
    base_log_sum: f64,
    context: &str,
) -> Result<LatentSignedLog, LatentSurvivalError> {
    let value = latent_signed_log_checked(log_abs, sign, context, "kernel derivative")?;
    if value.sign == 0.0 {
        Ok(value)
    } else {
        latent_signed_log_checked(
            value.log_abs - base_log_sum,
            value.sign,
            context,
            "normalised kernel derivative",
        )
    }
}

fn latent_signed_log_materialize(
    value: LatentSignedLog,
    context: &str,
) -> Result<f64, LatentSurvivalError> {
    let value =
        latent_signed_log_checked(value.log_abs, value.sign, context, "log-sum derivative")?;
    if value.sign == 0.0 {
        return Ok(0.0);
    }
    let materialized = value.sign * value.log_abs.exp();
    if materialized.is_finite() {
        Ok(materialized)
    } else {
        Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} log-sum derivative is outside the finite f64 range: \
                 log_abs={}, sign={}",
                value.log_abs, value.sign
            ),
        })
    }
}

/// Convert normalized signed-log moments into certified derivatives of `log(S)`.
///
/// For a non-empty slot set `A`, normalized moments and log derivatives obey
///
/// `m(A) = Σ_{B ⊆ A, pivot ∈ B} κ(B) m(A \\ B)`,
///
/// where `m(A) = S_A / S`, `κ(A) = ∂_A log(S)`, and the distinguished pivot is
/// the least-significant slot. Isolating `B = A` gives a pointed cumulant
/// recurrence.
///
/// The contract is deliberately about the inputs an actual binary64 consumer
/// has: each finite signed-log moment is rounded once to binary64, then the
/// exact-real cumulant polynomial over those rounded moments is evaluated by a
/// fixed error-free expansion. Publication succeeds only when the expansion
/// proves a unique nearest binary64 rounding. A moment lost to underflow,
/// overflow, an unrepresentable intermediate product, or an ambiguous final
/// rounding produces [`LatentSurvivalError::DerivativeAccuracyUnresolved`]
/// instead of an approximate derivative.
///
/// The four-slot table covers the `(a,b,u,v)` layouts used by Order2, OneSeed,
/// and TwoSeed.
fn latent_certified_cumulants(
    moments: [LatentSignedLog; 16],
    target_mask: usize,
    context: &str,
) -> Result<[LatentCertifiedCumulant; 16], LatentSurvivalError> {
    assert!(target_mask > 0 && target_mask < 16);

    let unresolved = |mask: usize, reason: &str| {
        LatentSurvivalError::DerivativeAccuracyUnresolved {
            reason: format!(
                "{context} derivative mask {mask:#06b} has no certified binary64 value: {reason}"
            ),
        }
    };
    let mut rounded_moments = [0.0_f64; 16];
    for mask in 0usize..16 {
        if mask & !target_mask != 0 {
            continue;
        }
        let moment = latent_signed_log_checked(
            moments[mask].log_abs,
            moments[mask].sign,
            context,
            "normalised cumulant moment",
        )?;
        if moment.sign == 0.0 {
            continue;
        }
        let magnitude = moment.log_abs.exp();
        if !magnitude.is_finite() {
            return Err(unresolved(
                mask,
                "the rounded moment magnitude overflowed",
            ));
        }
        if magnitude == 0.0 {
            return Err(unresolved(
                mask,
                "the rounded moment magnitude underflowed to zero",
            ));
        }
        rounded_moments[mask] = moment.sign * magnitude;
    }

    // The moments' own dynamic range is what every refusal below is actually
    // about, and it is the one number a caller cannot recover from the message
    // afterwards — recovering it has twice cost this lane a three-quarter-hour
    // fit (gam#2714). Report it with the refusal, not instead of it.
    let moment_span = || -> String {
        let mut lowest = f64::INFINITY;
        let mut highest = 0.0_f64;
        for mask in 0usize..16 {
            let magnitude = rounded_moments[mask].abs();
            if magnitude > 0.0 {
                lowest = lowest.min(magnitude);
                highest = highest.max(magnitude);
            }
        }
        if highest == 0.0 {
            return "no nonzero rounded moment".to_string();
        }
        format!("rounded moment magnitudes span [{lowest:.6e}, {highest:.6e}]")
    };
    let unresolved_with_span = |mask: usize, reason: &str| {
        unresolved(mask, &format!("{reason} ({})", moment_span()))
    };

    let mut exact_cumulants = [LatentExactExpansion::ZERO; 16];
    let mut certificates = [LatentCertifiedCumulant::ZERO; 16];
    for mask in 1usize..16 {
        if mask & !target_mask != 0 {
            continue;
        }
        if mask.is_power_of_two() {
            exact_cumulants[mask] = LatentExactExpansion::scalar(rounded_moments[mask]);
            certificates[mask] = LatentCertifiedCumulant {
                value: rounded_moments[mask],
                absolute_term_mass: rounded_moments[mask].abs(),
            };
            continue;
        }
        let pivot = 1usize << mask.trailing_zeros();
        let mut cumulant = LatentExactExpansion::scalar(rounded_moments[mask]);
        let mut absolute_term_mass = rounded_moments[mask].abs();
        let mut block = (mask - 1) & mask;
        while block != 0 {
            if block & pivot != 0 {
                let complement = mask ^ block;
                let subtract = exact_cumulants[block]
                    .scale(-rounded_moments[complement])
                    .and_then(|term| cumulant.add(term))
                    .map_err(|reason| unresolved_with_span(mask, reason))?;
                cumulant = subtract;
                let term_mass = LatentExactExpansion::next_up(
                    certificates[block].absolute_term_mass
                        * rounded_moments[complement].abs(),
                );
                if !term_mass.is_finite() {
                    return Err(unresolved(
                        mask,
                        "the conditioning mass overflowed binary64",
                    ));
                }
                absolute_term_mass =
                    LatentExactExpansion::next_up(absolute_term_mass + term_mass);
                if !absolute_term_mass.is_finite() {
                    return Err(unresolved(
                        mask,
                        "the accumulated conditioning mass overflowed binary64",
                    ));
                }
            }
            block = (block - 1) & mask;
        }
        let value = cumulant
            .certified_round()
            .map_err(|reason| unresolved_with_span(mask, reason))?;
        exact_cumulants[mask] = cumulant;
        certificates[mask] = LatentCertifiedCumulant {
            value,
            absolute_term_mass,
        };
    }
    Ok(certificates)
}

/// A one-pass analytic lift of a latent kernel sum into an order-specific jet.
///
/// `suffixes` describes the nilpotent parts carried by the requested scalar:
/// `[]` for the ordinary order-two base, `[u]` for the `OneSeed` epsilon part,
/// and `[u]`, `[v]`, `[u,v]` for the three non-base `TwoSeed` parts.  For each
/// part we differentiate the SAME kernel-term program in the canonical
/// highest-slot-to-lowest-slot order used by the pre-cutover `MultiDirJet`
/// subset cache. Every requested raw derivative is therefore assembled by the
/// same recurrence, accumulation order, and signed-log reduction as its oracle.
/// The expensive quadrature bundle is then evaluated ONCE at the maximum `k`
/// required by the complete output instead of once per Hessian cell.
/// The kernel rung ceiling one lift needs.
///
/// Every emitted part carries a full order-two base, so the largest requested
/// recurrence is the base list's own highest rung plus two primary
/// differentiations plus the largest nilpotent suffix. This exact support bound
/// is what lets the one shared bundle be built before any individual derivative
/// term list is constructed.
///
/// Extracted so the value-only lift and the order-two lift cannot drift: the
/// bundle's `max_k` decides which rungs exist AND (through
/// `log_scaled_a_derivative_tower`) how long the `∂_a` tower is, so two lifts
/// that computed it differently would evaluate the same base term list on
/// different data (#2714).
fn latent_kernel_sum_max_k<const K: usize>(
    base_terms: &[LatentKernelPrimaryTerm],
    primary_directions: &[LatentKernelPrimaryDirection; K],
    suffixes: &[&[LatentKernelPrimaryDirection]],
) -> usize {
    let base_max_k = base_terms.iter().map(|term| term.k).max().unwrap_or(0);
    let k_increment = |direction: &LatentKernelPrimaryDirection| {
        if direction.dtau != 0.0 {
            2
        } else if direction.dq != 0.0 || direction.dmu != 0.0 {
            1
        } else {
            0
        }
    };
    let max_primary_increment = primary_directions
        .iter()
        .map(&k_increment)
        .max()
        .unwrap_or(0);
    let max_suffix_increment = suffixes
        .iter()
        .map(|suffix| suffix.iter().map(&k_increment).sum::<usize>())
        .max()
        .unwrap_or(0);
    base_max_k + 2 * max_primary_increment + max_suffix_increment
}

/// The shared prologue of every lift: the ONE kernel bundle, and the base term
/// list's signed-log sum on it.
///
/// This is the row's log-likelihood contribution before any normalisation —
/// `latent_kernel_sum_order2_parts` publishes it verbatim as the value channel
/// (`out[0].0.v = base_log_sum`). Sharing it with the value-only lift is what
/// makes `log_likelihood_only` and the joint gradient evaluate ONE function of
/// `β` rather than two implementations of one formula (#2714).
fn latent_kernel_sum_base<const K: usize>(
    quadctx: &QuadratureContext,
    base_terms: &[LatentKernelPrimaryTerm],
    state: LatentKernelPrimaryState,
    primary_directions: &[LatentKernelPrimaryDirection; K],
    suffixes: &[&[LatentKernelPrimaryDirection]],
    context: &str,
) -> Result<(LogLognormalKernelBundle, f64), LatentSurvivalError> {
    let max_k = latent_kernel_sum_max_k(base_terms, primary_directions, suffixes);
    let bundle =
        log_kernel_bundle(quadctx, state.q.exp(), state.mu, state.sigma, max_k).map_err(|e| {
            LatentSurvivalError::NumericalFailure {
                reason: format!("{context} kernel evaluation failed: {e}"),
            }
        })?;
    let (base_log_sum, base_sign) =
        latent_kernel_evaluate_terms(&bundle, state, base_terms, context)?;
    if !(base_log_sum.is_finite() && base_sign > 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!("{context} produced a non-positive signed kernel sum"),
        });
    }
    Ok((bundle, base_log_sum))
}

fn latent_kernel_sum_order2_parts<const K: usize>(
    quadctx: &QuadratureContext,
    base_terms: &[LatentKernelPrimaryTerm],
    state: LatentKernelPrimaryState,
    primary_directions: &[LatentKernelPrimaryDirection; K],
    suffixes: &[&[LatentKernelPrimaryDirection]],
    context: &str,
) -> Result<[Order2<K>; 4], LatentSurvivalError> {
    assert!(
        !suffixes.is_empty() && suffixes.len() <= 4,
        "latent kernel lift supports one to four order-two parts"
    );
    let (bundle, base_log_sum) = latent_kernel_sum_base(
        quadctx,
        base_terms,
        state,
        primary_directions,
        suffixes,
        context,
    )?;

    let evaluate_terms = |terms: &[LatentKernelPrimaryTerm]| {
        latent_kernel_evaluate_terms(&bundle, state, terms, context)
    };

    let normalized = |axes: &[LatentKernelPrimaryDirection],
                      suffix: &[LatentKernelPrimaryDirection]|
     -> Result<LatentSignedLog, LatentSurvivalError> {
        let is_zero = |direction: &LatentKernelPrimaryDirection| {
            direction.dq == 0.0
                && direction.dqd == 0.0
                && direction.dmu == 0.0
                && direction.dtau == 0.0
        };
        if axes.iter().chain(suffix.iter()).any(is_zero) {
            return Ok(LatentSignedLog::ZERO);
        }
        let terms = latent_kernel_term_sequence_inline(base_terms, axes, suffix);
        assert!(
            !terms.spilled(),
            "latent derivative support exceeded the inline allocation-free capacity: {} > {}",
            terms.len(),
            LATENT_TERM_INLINE_CAPACITY
        );
        let (log_abs, sign) = evaluate_terms(&terms)?;
        latent_signed_log_normalized(log_abs, sign, base_log_sum, context)
    };

    let mut parts = [LatentSignedLogOrder2::<K>::zero(); 4];
    for (part, suffix) in suffixes.iter().enumerate() {
        let value = if part == 0 {
            // The base is the kernel divided by itself.
            LatentSignedLog::ONE
        } else {
            normalized(&[], suffix)?
        };
        let mut tower = LatentSignedLogOrder2::<K>::zero();
        tower.v = value;
        for a in 0..K {
            tower.g[a] = normalized(&[primary_directions[a]], suffix)?;
        }
        for a in 0..K {
            for b in a..K {
                let derivative =
                    normalized(&[primary_directions[a], primary_directions[b]], suffix)?;
                tower.h[a][b] = derivative;
                tower.h[b][a] = derivative;
            }
        }
        parts[part] = tower;
    }
    latent_kernel_signed_log_parts(
        base_log_sum,
        parts,
        suffixes.len(),
        context,
    )
}

/// Convert normalized signed-log kernel moments into derivatives of the log sum.
///
/// `normalized_parts` is the single analytic recurrence's compact moment
/// layout: the base carries `(1, S_a/S, S_ab/S)`, the one-seed parts carry
/// `(S_u/S, S_au/S, S_abu/S)`, and the two-seed cross part carries `(S_uv/S,
/// S_auv/S, S_abuv/S)`. For each requested output channel those moments become
/// a four-slot `(a,b,u,v)` table for [`latent_certified_cumulants`]. Each
/// derivative is published only after its exact expansion has certified the
/// unique rounded result.
fn latent_kernel_signed_log_parts<const K: usize>(
    base_log_sum: f64,
    normalized_parts: [LatentSignedLogOrder2<K>; 4],
    part_count: usize,
    context: &str,
) -> Result<[Order2<K>; 4], LatentSurvivalError> {
    assert!(matches!(part_count, 1 | 2 | 4));
    let compose_log = |moments: [LatentSignedLog; 16], target_mask: usize| {
        latent_certified_cumulants(moments, target_mask, context)
    };
    let moments_for = |a: usize, b: usize| {
        let base = &normalized_parts[0];
        let u = &normalized_parts[1];
        let v = &normalized_parts[2];
        let uv = &normalized_parts[3];
        [
            LatentSignedLog::ONE,
            base.g[a],
            base.g[b],
            base.h[a][b],
            u.v,
            u.g[a],
            u.g[b],
            u.h[a][b],
            v.v,
            v.g[a],
            v.g[b],
            v.h[a][b],
            uv.v,
            uv.g[a],
            uv.g[b],
            uv.h[a][b],
        ]
    };

    let mut out = [Order2::<K>::constant(0.0); 4];
    out[0].0.v = base_log_sum;
    if part_count >= 2 {
        out[1].0.v = latent_signed_log_materialize(normalized_parts[1].v, context)?;
    }
    if part_count == 4 {
        out[2].0.v = latent_signed_log_materialize(normalized_parts[2].v, context)?;
        let composed = compose_log(moments_for(0, 0), 0b1100)?;
        out[3].0.v = composed[0b1100].value;
    }

    let (gradient_mask, hessian_mask) = match part_count {
        1 => (0b0001, 0b0011),
        2 => (0b0101, 0b0111),
        4 => (0b1101, 0b1111),
        other => {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!(
                    "{context} composed a latent moment jet over {other} parts; \
                     only 1, 2, or 4 are constructible"
                ),
            })
        }
    };
    for a in 0..K {
        let composed = compose_log(moments_for(a, a), gradient_mask)?;
        out[0].0.g[a] = composed[0b0001].value;
        if part_count >= 2 {
            out[1].0.g[a] = composed[0b0101].value;
        }
        if part_count == 4 {
            out[2].0.g[a] = composed[0b1001].value;
            out[3].0.g[a] = composed[0b1101].value;
        }
        for b in a..K {
            let composed = compose_log(moments_for(a, b), hessian_mask)?;
            out[0].0.h[a][b] = composed[0b0011].value;
            if part_count >= 2 {
                out[1].0.h[a][b] = composed[0b0111].value;
            }
            if part_count == 4 {
                out[2].0.h[a][b] = composed[0b1011].value;
                out[3].0.h[a][b] = composed[0b1111].value;
            }
            for part in 0..part_count {
                out[part].0.h[b][a] = out[part].0.h[a][b];
            }
        }
    }
    Ok(out)
}

#[inline]
fn latent_kernel_direction_linear_combination<const K: usize>(
    primary_directions: &[LatentKernelPrimaryDirection; K],
    coefficients: &[f64; K],
) -> LatentKernelPrimaryDirection {
    let mut out = LatentKernelPrimaryDirection {
        dq: 0.0,
        dqd: 0.0,
        dmu: 0.0,
        dtau: 0.0,
    };
    for a in 0..K {
        out.dq += coefficients[a] * primary_directions[a].dq;
        out.dqd += coefficients[a] * primary_directions[a].dqd;
        out.dmu += coefficients[a] * primary_directions[a].dmu;
        out.dtau += coefficients[a] * primary_directions[a].dtau;
    }
    out
}

fn latent_order2_all_finite<const K: usize>(jet: &Order2<K>) -> bool {
    jet.value().is_finite()
        && jet.g().iter().all(|value| value.is_finite())
        && jet
            .h()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
}

/// Backend seam for the single latent-survival row expression.  Only the
/// analytic multivariate kernel primitive differs by requested channel; all
/// numerator/denominator/event algebra below is instantiated unchanged.
trait LatentPrimaryJetBackend<const K: usize> {
    type Jet: JetScalar<K>;

    fn derivative_order(&self) -> usize;
    fn all_channels_finite(&self, jet: &Self::Jet) -> bool;

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError>;
}

#[derive(Clone, Copy)]
struct LatentOrder2Backend;

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentOrder2Backend {
    type Jet = Order2<K>;

    fn derivative_order(&self) -> usize {
        2
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(jet)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        let suffixes: [&[LatentKernelPrimaryDirection]; 1] = [&[]];
        let parts = latent_kernel_sum_order2_parts(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(parts[0])
    }
}

/// Value-only backend: the SAME row expression, evaluated for its value channel
/// and nothing else (#2714).
///
/// This is what `log_likelihood_only` — the scalar the joint-Newton trust region
/// evaluates at the TRIAL β — goes through, so that the accept test measures the
/// value of the function whose gradient the step was built from, rather than a
/// second implementation of the same formula. The trust ratio's numerator
/// differences `old_objective` (built from the gradient hook's log-likelihood)
/// against `trial_objective` (built from this one); a gap between them is a
/// constant of the backtracking ladder that shrinking the radius cannot remove.
///
/// **`K` is deliberately the same `K` the derivative backends use** even though
/// no derivative is produced. [`latent_kernel_sum_max_k`] sizes the kernel
/// bundle from the primary directions, and the bundle's `max_k` decides both
/// which rungs exist and how long the `∂_a` tower is — so a backend that dropped
/// the directions would build a SHORTER bundle and could be routed to a
/// different basis for the same term list. Same `K` ⇒ same bundle ⇒ same basis ⇒
/// the value is bit-identical, which is the entire point.
///
/// `derivative_order = 0` because no derivative channel is filled: it reaches
/// only `latent_survival_positive_log_difference_jet`, where it means the
/// interval branch validates the unary composition to order 0. A value-only
/// evaluation must not refuse because some derivative of the log-gap is
/// unrepresentable.
#[derive(Clone, Copy)]
struct LatentValueBackend;

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentValueBackend {
    type Jet = Order2<K>;

    fn derivative_order(&self) -> usize {
        0
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(jet)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        // The same `suffixes` the order-two backend passes, so
        // `latent_kernel_sum_max_k` returns the same ceiling.
        let suffixes: [&[LatentKernelPrimaryDirection]; 1] = [&[]];
        let (_, base_log_sum) = latent_kernel_sum_base(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(Order2::<K>::constant(base_log_sum))
    }
}

#[derive(Clone, Copy)]
struct LatentOneSeedBackend<const K: usize> {
    direction: [f64; K],
}

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentOneSeedBackend<K> {
    type Jet = OneSeed<K>;

    fn derivative_order(&self) -> usize {
        3
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(&jet.base) && latent_order2_all_finite(&jet.eps)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        let seed = latent_kernel_direction_linear_combination(primary_directions, &self.direction);
        let seed_suffix = [seed];
        let suffixes: [&[LatentKernelPrimaryDirection]; 2] = [&[], &seed_suffix];
        let parts = latent_kernel_sum_order2_parts(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(OneSeed {
            base: parts[0],
            eps: parts[1],
        })
    }
}

#[derive(Clone, Copy)]
struct LatentTwoSeedBackend<const K: usize> {
    direction_u: [f64; K],
    direction_v: [f64; K],
}

impl<const K: usize> LatentPrimaryJetBackend<K> for LatentTwoSeedBackend<K> {
    type Jet = TwoSeed<K>;

    fn derivative_order(&self) -> usize {
        4
    }

    fn all_channels_finite(&self, jet: &Self::Jet) -> bool {
        latent_order2_all_finite(&jet.base)
            && latent_order2_all_finite(&jet.eps)
            && latent_order2_all_finite(&jet.del)
            && latent_order2_all_finite(&jet.eps_del)
    }

    fn kernel_sum_log(
        &self,
        quadctx: &QuadratureContext,
        base_terms: &[LatentKernelPrimaryTerm],
        state: LatentKernelPrimaryState,
        primary_directions: &[LatentKernelPrimaryDirection; K],
        context: &str,
    ) -> Result<Self::Jet, LatentSurvivalError> {
        let seed_u =
            latent_kernel_direction_linear_combination(primary_directions, &self.direction_u);
        let seed_v =
            latent_kernel_direction_linear_combination(primary_directions, &self.direction_v);
        let suffix_u = [seed_u];
        let suffix_v = [seed_v];
        let suffix_uv = [seed_u, seed_v];
        let suffixes: [&[LatentKernelPrimaryDirection]; 4] =
            [&[], &suffix_u, &suffix_v, &suffix_uv];
        let parts = latent_kernel_sum_order2_parts(
            quadctx,
            base_terms,
            state,
            primary_directions,
            &suffixes,
            context,
        )?;
        Ok(TwoSeed {
            base: parts[0],
            eps: parts[1],
            del: parts[2],
            eps_del: parts[3],
        })
    }
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

#[cfg(test)]
mod tests_multidir_row {
    use super::tests_multidir_kernel::latent_kernel_sum_log_jet;
    use super::*;
    use gam_math::jet_partitions::MultiDirJet as LatentMultiDirJet;

    pub(super) fn latent_survival_row_primary_log_jet_multidir_reference(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        point: LatentSurvivalPrimaryPoint,
        directions: &[LatentSurvivalPrimaryDirection],
    ) -> Result<LatentMultiDirJet, String> {
        let LatentSurvivalPrimaryPoint {
            q_entry,
            q_exit,
            qdot_exit,
            mu,
            sigma,
            ..
        } = point;
        let log_sigma_factor = point.log_sigma_factor();
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
            LatentSurvivalEventType::IntervalCensored => {
                latent_survival_interval_numerator_log_jet_multidir_reference(
                    quadctx, row, point, directions,
                )?
            }
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
    /// The two boundary kernels use distinct states and direction maps, but their
    /// difference stays in log space:
    ///
    /// ```text
    /// log(c_L K_L - c_R K_R)
    ///   = log(c_L K_L) + log1mexp(log(c_R K_R) - log(c_L K_L)).
    /// ```
    ///
    /// The same certified unary stack as production is composed over the
    /// multi-direction oracle, so absolute tail mass cannot underflow and the
    /// reference cannot silently publish a non-finite higher coefficient.
    fn latent_survival_interval_numerator_log_jet_multidir_reference(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        point: LatentSurvivalPrimaryPoint,
        directions: &[LatentSurvivalPrimaryDirection],
    ) -> Result<LatentMultiDirJet, String> {
        let LatentSurvivalPrimaryPoint {
            q_exit,
            q_right,
            mu,
            sigma,
            ..
        } = point;
        let log_sigma_factor = point.log_sigma_factor();
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

        // `MultiDirJet` is the runtime-width test oracle rather than a
        // `JetField`, so spell out the same log-domain identity while sharing
        // the certified unary derivative stack with production.
        let weighted_left = log_left.add(&LatentMultiDirJet::constant(
            directions.len(),
            -row.mass_unloaded_left,
        ));
        let weighted_right = log_right.add(&LatentMultiDirJet::constant(
            directions.len(),
            -row.mass_unloaded_right,
        ));
        let delta = weighted_right.sub(&weighted_left);
        let delta_value = delta.coeff(0);
        if !(delta_value.is_finite() && delta_value < 0.0) {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!(
                    "latent survival interval numerator must be a positive \
                     survival-mass difference: log(c_L*K0(M_L))={:?}, \
                     log(c_R*K0(M_R))={:?}; require M_L < M_R (i.e. L < R)",
                    weighted_left.coeff(0),
                    weighted_right.coeff(0),
                ),
            }
            .to_string());
        }
        let derivatives = latent_unary_derivatives_log1mexp_negative(
            delta_value,
            directions.len(),
            "latent survival interval numerator",
        )
        .map_err(|error| error.to_string())?;
        let out = weighted_left.add(&delta.compose_unary(derivatives));
        if !out.coeffs.iter().all(|value| value.is_finite()) {
            return Err(LatentSurvivalError::NumericalFailure {
                reason: format!(
                    "latent survival interval numerator derivative jet is not \
                     representable at log-boundary gap {delta_value:?}"
                ),
            }
            .to_string());
        }
        Ok(out)
    }
}

/// The single latent-survival row program, instantiated at an order-two,
/// one-seed, or two-seed backend.  Event dispatch and the interval
/// log-difference are intentionally expressed once here; a backend changes only
/// the derivative layout used to lift each analytic kernel-sum primitive.
fn latent_survival_row_primary_jet<const K: usize, B: LatentPrimaryJetBackend<K>>(
    backend: &B,
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
) -> Result<B::Jet, LatentSurvivalError> {
    let LatentSurvivalPrimaryPoint {
        q_entry,
        q_exit,
        qdot_exit,
        mu,
        sigma,
        ..
    } = point;
    let log_sigma_factor = point.log_sigma_factor();
    let entry_state = LatentKernelPrimaryState {
        q: q_entry,
        qdot: 1.0,
        mu,
        sigma,
        log_sigma_factor,
    };
    let entry_directions: [LatentKernelPrimaryDirection; K] = std::array::from_fn(|a| {
        latent_survival_map_entry_direction(latent_survival_basis_direction(a))
    });
    let denominator = backend
        .kernel_sum_log(
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

    let numerator = match row.event_type {
        LatentSurvivalEventType::RightCensored => {
            let exit_state = LatentKernelPrimaryState {
                q: q_exit,
                qdot: qdot_exit,
                mu,
                sigma,
                log_sigma_factor,
            };
            let exit_directions: [LatentKernelPrimaryDirection; K] = std::array::from_fn(|a| {
                latent_survival_map_exit_direction(
                    latent_survival_basis_direction(a),
                    row.event_type,
                )
            });
            backend
                .kernel_sum_log(
                    quadctx,
                    &[LatentKernelPrimaryTerm {
                        coeff: 1.0,
                        q_exp: 0,
                        qdot_power: 0,
                        tau_exp: 0,
                        k: 0,
                    }],
                    exit_state,
                    &exit_directions,
                    "latent survival numerator",
                )?
        }
        LatentSurvivalEventType::ExactEvent => {
            let exit_state = LatentKernelPrimaryState {
                q: q_exit,
                qdot: qdot_exit,
                mu,
                sigma,
                log_sigma_factor,
            };
            let exit_directions: [LatentKernelPrimaryDirection; K] = std::array::from_fn(|a| {
                latent_survival_map_exit_direction(
                    latent_survival_basis_direction(a),
                    LatentSurvivalEventType::ExactEvent,
                )
            });
            // A zero unloaded-hazard term remains in this stack array but the
            // signed-log evaluator and derivative recurrences discard it.
            let numerator_terms = [
                LatentKernelPrimaryTerm {
                    coeff: row.hazard_unloaded,
                    q_exp: 0,
                    qdot_power: 0,
                    tau_exp: 0,
                    k: 0,
                },
                LatentKernelPrimaryTerm {
                    coeff: 1.0,
                    q_exp: 1,
                    qdot_power: 1,
                    tau_exp: 0,
                    k: 1,
                },
            ];
            backend
                .kernel_sum_log(
                    quadctx,
                    &numerator_terms,
                    exit_state,
                    &exit_directions,
                    "latent survival numerator",
                )?
        }
        LatentSurvivalEventType::IntervalCensored => {
            latent_survival_interval_numerator_jet(backend, quadctx, row, point)?
        }
    };

    let unloaded_offset = match row.event_type {
        LatentSurvivalEventType::IntervalCensored => row.mass_unloaded_entry,
        _ => -row.mass_unloaded_exit + row.mass_unloaded_entry,
    };
    Ok(numerator
        .sub(&denominator)
        .add(&B::Jet::constant(unloaded_offset)))
}

fn latent_survival_interval_numerator_jet<const K: usize, B: LatentPrimaryJetBackend<K>>(
    backend: &B,
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
) -> Result<B::Jet, LatentSurvivalError> {
    let LatentSurvivalPrimaryPoint {
        q_exit,
        q_right,
        mu,
        sigma,
        ..
    } = point;
    let log_sigma_factor = point.log_sigma_factor();
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
    let left_directions: [LatentKernelPrimaryDirection; K] = std::array::from_fn(|a| {
        latent_survival_map_left_direction(latent_survival_basis_direction(a))
    });
    let right_directions: [LatentKernelPrimaryDirection; K] = std::array::from_fn(|a| {
        latent_survival_map_right_direction(latent_survival_basis_direction(a))
    });
    let log_left = backend
        .kernel_sum_log(
            quadctx,
            &single_k0,
            left_state,
            &left_directions,
            "latent survival interval left boundary",
        )?;
    let log_right = backend
        .kernel_sum_log(
            quadctx,
            &single_k0,
            right_state,
            &right_directions,
            "latent survival interval right boundary",
        )?;

    latent_survival_positive_log_difference_jet(
        &log_left,
        -row.mass_unloaded_left,
        &log_right,
        -row.mass_unloaded_right,
        backend.derivative_order(),
        "latent survival interval numerator",
        |jet| backend.all_channels_finite(jet),
    )
}

/// # Accuracy of the `log σ` curvature (#2566)
///
/// The value and gradient channels are trustworthy across the whole σ range. The
/// **second-derivative** channel is not, and the boundary has been measured
/// rather than estimated.
///
/// A 0.05-step sweep of `log σ` on the #2566 fixture
/// (`zz_measure_2566_curvature_fine_sweep`) shows `value` and `gradient` smooth
/// and monotone to every printed digit while the returned curvature degrades
/// progressively — relative step-to-step jumps run `0.05, 0.07, 0.19, 0.47, 0.84,
/// 2.89, 9.78` — and past `log σ ≈ 5.45` it **changes sign between adjacent
/// samples**, repeatedly.
///
/// That is catastrophic cancellation in forming the second cumulant, not a
/// routing switch: a branch gives a consistently wrong value on one side of a
/// threshold, whereas this oscillates. Three consequences the measurements
/// already settled, so nobody has to re-derive them:
///
/// * more quadrature resolution MOVES the crossing and cannot remove it — going
///   513 → 1025 nodes bought 335× at `log σ = 5` and left `log σ ≥ 6` untouched,
///   because the cancelled quantity keeps shrinking while the roundoff floor does
///   not;
/// * `max_k` is not the mechanism (the bundle mode is constant in `k` over
///   `0..8`), and `IntegratedExpectationMode` cannot flag it — the mode is
///   `ControlledAsymptotic` at both the healthy `log σ = 4` and the inverted
///   `log σ = 7`;
/// * past the crossing the curvature stops tracking its inputs at all: a node
///   change that moved the gradient channel 34% left the Hessian unmoved.
///
/// **Usable to `log σ ≈ 5.4` on this fixture.** Beyond it there is no correct
/// value to return, so a consumer needing a definite Hessian must refuse rather
/// than scale its tolerance. The discriminator to refuse ON already exists — the
/// gate's Richardson construction reads `0.853` at `log σ = 4` against `109.765`
/// at `log σ = 6` — and exporting it is #2566's remaining work. A genuine repair
/// needs the cumulant formed without the cancelling difference, which is a
/// reformulation rather than a tolerance.
fn latent_survival_row_primary_gradient_hessian(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    include_log_sigma: bool,
) -> Result<(f64, Array1<f64>, Array2<f64>), LatentSurvivalError> {
    if include_log_sigma {
        let out = latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &LatentOrder2Backend,
            quadctx,
            row,
            point,
        )?;
        let out_gradient = out.g();
        let hessian = out.h();
        Ok((
            out.value(),
            Array1::from_shape_fn(LATENT_SURVIVAL_PRIMARY_DIM, |a| out_gradient[a]),
            Array2::from_shape_fn(
                (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
                |(a, b)| -hessian[a][b],
            ),
        ))
    } else {
        let out = latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
            &LatentOrder2Backend,
            quadctx,
            row,
            point,
        )?;
        let out_gradient = out.g();
        let out_hessian = out.h();
        Ok((
            out.value(),
            Array1::from_shape_fn(LATENT_SURVIVAL_PRIMARY_DIM, |a| {
                if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA {
                    out_gradient[a]
                } else {
                    0.0
                }
            }),
            Array2::from_shape_fn(
                (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
                |(a, b)| {
                    if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                        && b < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                    {
                        -out_hessian[a][b]
                    } else {
                        0.0
                    }
                },
            ),
        ))
    }
}

/// The row log-likelihood, on the SAME surface as
/// [`latent_survival_row_primary_gradient_hessian`]'s value channel — and, on a
/// bundle whose basis does not depend on the derivative request, bit-identical
/// to it (#2714).
///
/// The `include_log_sigma` switch selects the same `K` the gradient/Hessian
/// entry point selects, because `K` is what sizes the kernel bundle.
fn latent_survival_row_primary_value(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    include_log_sigma: bool,
) -> Result<f64, LatentSurvivalError> {
    if include_log_sigma {
        latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &LatentValueBackend,
            quadctx,
            row,
            point,
        )
        .map(|jet| jet.value())
    } else {
        latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
            &LatentValueBackend,
            quadctx,
            row,
            point,
        )
        .map(|jet| jet.value())
    }
}

fn latent_survival_row_primary_one_seed_fixed_sigma(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction: &Array1<f64>,
) -> Result<OneSeed<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA>, LatentSurvivalError> {
    let backend = LatentOneSeedBackend {
        direction: std::array::from_fn(|a| direction[a]),
    };
    latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
        &backend, quadctx, row, point,
    )
}

fn latent_survival_row_primary_two_seed_fixed_sigma(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
) -> Result<TwoSeed<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA>, LatentSurvivalError> {
    let backend = LatentTwoSeedBackend {
        direction_u: std::array::from_fn(|a| direction_u[a]),
        direction_v: std::array::from_fn(|a| direction_v[a]),
    };
    latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
        &backend, quadctx, row, point,
    )
}

fn latent_survival_row_primary_third_contracted(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, LatentSurvivalError> {
    if include_log_sigma {
        let backend = LatentOneSeedBackend {
            direction: std::array::from_fn(|a| direction[a]),
        };
        let out = latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &backend, quadctx, row, point,
        )?;
        let third = out.contracted_third();
        Ok(Array2::from_shape_fn(
            (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
            |(a, b)| -third[a][b],
        ))
    } else {
        let out = latent_survival_row_primary_one_seed_fixed_sigma(quadctx, row, point, direction)?;
        let third = out.contracted_third();
        Ok(Array2::from_shape_fn(
            (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
            |(a, b)| {
                if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA && b < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA {
                    -third[a][b]
                } else {
                    0.0
                }
            },
        ))
    }
}

fn latent_survival_row_primary_fourth_contracted(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, LatentSurvivalError> {
    if include_log_sigma {
        let backend = LatentTwoSeedBackend {
            direction_u: std::array::from_fn(|a| direction_u[a]),
            direction_v: std::array::from_fn(|a| direction_v[a]),
        };
        let out = latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &backend, quadctx, row, point,
        )?;
        let fourth = out.contracted_fourth();
        Ok(Array2::from_shape_fn(
            (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
            |(a, b)| -fourth[a][b],
        ))
    } else {
        let out = latent_survival_row_primary_two_seed_fixed_sigma(
            quadctx,
            row,
            point,
            direction_u,
            direction_v,
        )?;
        let fourth = out.contracted_fourth();
        Ok(Array2::from_shape_fn(
            (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
            |(a, b)| {
                if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA && b < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA {
                    -fourth[a][b]
                } else {
                    0.0
                }
            },
        ))
    }
}

#[cfg(test)]
mod tests_multidir_channels {
    use super::tests_multidir_row::latent_survival_row_primary_log_jet_multidir_reference;
    use super::*;

    pub(super) fn latent_survival_row_primary_gradient_hessian_multidir_reference(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        point: LatentSurvivalPrimaryPoint,
        include_log_sigma: bool,
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let mut gradient = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        let mut neg_hessian =
            Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
        let active_primary = if include_log_sigma {
            LATENT_SURVIVAL_PRIMARY_DIM
        } else {
            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
        };
        let log_lik =
            latent_survival_row_primary_log_jet_multidir_reference(quadctx, row, point, &[])?
                .coeff(0);
        for a in 0..active_primary {
            let dir_a = latent_survival_basis_direction(a);
            gradient[a] = latent_survival_row_primary_log_jet_multidir_reference(
                quadctx,
                row,
                point,
                &[dir_a],
            )?
            .coeff(1);
            for b in a..active_primary {
                let coeff = latent_survival_row_primary_log_jet_multidir_reference(
                    quadctx,
                    row,
                    point,
                    &[dir_a, latent_survival_basis_direction(b)],
                )?
                .coeff(3);
                neg_hessian[[a, b]] = -coeff;
                neg_hessian[[b, a]] = -coeff;
            }
        }
        Ok((log_lik, gradient, neg_hessian))
    }

    pub(super) fn latent_survival_row_primary_third_contracted_multidir_reference(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        point: LatentSurvivalPrimaryPoint,
        direction: &Array1<f64>,
        include_log_sigma: bool,
    ) -> Result<Array2<f64>, String> {
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
        let mut out =
            Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
        for a in 0..active_primary {
            let dir_a = latent_survival_basis_direction(a);
            for b in a..active_primary {
                let coeff = latent_survival_row_primary_log_jet_multidir_reference(
                    quadctx,
                    row,
                    point,
                    &[dir_a, latent_survival_basis_direction(b), dir],
                )?
                .coeff(7);
                out[[a, b]] = -coeff;
                out[[b, a]] = -coeff;
            }
        }
        Ok(out)
    }

    pub(super) fn latent_survival_row_primary_fourth_contracted_multidir_reference(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        point: LatentSurvivalPrimaryPoint,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        include_log_sigma: bool,
    ) -> Result<Array2<f64>, String> {
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
        let mut out =
            Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
        for a in 0..active_primary {
            let dir_a = latent_survival_basis_direction(a);
            for b in a..active_primary {
                let coeff = latent_survival_row_primary_log_jet_multidir_reference(
                    quadctx,
                    row,
                    point,
                    &[dir_a, latent_survival_basis_direction(b), dir_u, dir_v],
                )?
                .coeff(15);
                out[[a, b]] = -coeff;
                out[[b, a]] = -coeff;
            }
        }
        Ok(out)
    }
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
    ll: CompensatedRowSum,
    gradient: Array1<f64>,
}

#[derive(Clone)]
struct LatentSurvivalJointDenseAccum {
    ll: CompensatedRowSum,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
}

#[derive(Clone)]
struct LatentSurvivalDenseHessianAccum {
    hessian: Array2<f64>,
}

#[derive(Clone, Copy, Default)]
struct CompensatedRowSum {
    sum: f64,
    correction: f64,
}

impl CompensatedRowSum {
    #[inline]
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    #[inline]
    fn value(self) -> f64 {
        self.sum + self.correction
    }
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
            let scale = checked_weighted_row_value(
                weight,
                primary_gradient[primary_idx],
                row,
                "primary gradient",
            )?;
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

        let mean_scale = checked_weighted_row_value(
            weight,
            primary_gradient[LATENT_SURVIVAL_PRIMARY_MU],
            row,
            "mean gradient",
        )?;
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
            target[log_sigma.start] += checked_weighted_row_value(
                weight,
                primary_gradient[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA],
                row,
                "log-sigma gradient",
            )?;
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
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        let acc = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || LatentSurvivalJointGradientAccum {
                ll: CompensatedRowSum::default(),
                gradient: Array1::<f64>::zeros(total),
            },
            |row_idx, acc| {
                let wi = weights.at(row_idx);
                if wi == 0.0 {
                    return Ok(());
                }
                let row = self.build_row_at(
                    row_idx,
                    q_entry[row_idx],
                    q_exit[row_idx],
                    qdot_exit[row_idx],
                    q_right[row_idx],
                )?;
                let point = LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[row_idx],
                    q_exit: q_exit[row_idx],
                    qdot_exit: qdot_exit[row_idx],
                    q_right: q_right[row_idx],
                    mu: mu[row_idx],
                    sigma,
                };
                let (row_ll, primary_gradient, _) = latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    point,
                    include_log_sigma,
                )?;
                acc.ll.add(checked_weighted_row_value(
                    wi,
                    row_ll,
                    row_idx,
                    "log likelihood",
                )?);
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
                total_acc.ll.add(chunk_acc.ll.value());
                total_acc.gradient += &chunk_acc.gradient;
            },
        )?;
        let ll = require_finite_likelihood_scalar(acc.ll.value(), "log likelihood")?;
        require_finite_likelihood_vector(&acc.gradient, "gradient")?;
        Ok((ll, acc.gradient))
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
    /// `WorkingModelSurvival::offset_channel_residuals`. Because
    /// `∂q_ch/∂o_ch = 1`, the residual `∂NLL/∂o_ch_i` equals
    /// `−∂(log-likelihood)/∂q_ch_i`, and the per-row primary log-likelihood
    /// gradient over `(q_entry, q_exit, q̇_exit)` is precisely the
    /// `Q_ENTRY`/`Q_EXIT`/`QDOT_EXIT` components returned by
    /// `latent_survival_row_primary_gradient_hessian`. Sampleweight-scaled to
    /// match the `OffsetChannelResiduals` contract consumed by
    /// `baseline_chain_rule_gradient`.
    ///
    /// At the converged (constrained) β̂ the envelope theorem makes this the
    /// exact θ-gradient of the profile penalized NLL `0.5·deviance + 0.5·βᵀSβ`.
    /// The interval upper-bound `q_right = x_time_right·β_time + o_R(θ)` channel
    /// DOES carry its own baseline-θ offset `o_R(θ)` (the time basis evaluated at
    /// the bracket upper bound `R`), distinct from the exit offset at `L`, so its
    /// residual `−∂(log-likelihood)/∂q_right` is returned in the dedicated
    /// `OffsetChannelResiduals::right` channel; it is exactly 0 on every
    /// non-interval row (the `Q_RIGHT` primary channel is inert there) and the
    /// baseline-θ chain rule contracts it against the `age_right`-evaluated
    /// η-partial.
    pub fn offset_channel_residuals(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<crate::survival::OffsetChannelResiduals, LatentSurvivalError> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")?;
        let n = self.event_target.len();
        // `split_time_eta` validates the complete block slate before indexing
        // it and returns `LatentSurvivalError::BlockMismatch` when fitted state
        // is missing. There is deliberately no zero-residual fallback: zeros
        // would manufacture a stationary outer baseline gradient.
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let include_log_sigma = self.joint_slices().log_sigma.is_some();
        let mut entry = Array1::<f64>::zeros(n);
        let mut exit = Array1::<f64>::zeros(n);
        let mut derivative = Array1::<f64>::zeros(n);
        let mut right = Array1::<f64>::zeros(n);
        for row_idx in 0..n {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row = self.build_row_at(
                row_idx,
                q_entry[row_idx],
                q_exit[row_idx],
                qdot_exit[row_idx],
                q_right[row_idx],
            )?;
            let point = LatentSurvivalPrimaryPoint {
                q_entry: q_entry[row_idx],
                q_exit: q_exit[row_idx],
                qdot_exit: qdot_exit[row_idx],
                q_right: q_right[row_idx],
                mu: mu[row_idx],
                sigma,
            };
            let (_, primary_gradient, _) = latent_survival_row_primary_gradient_hessian(
                &self.quadctx,
                &row,
                point,
                include_log_sigma,
            )?;
            // ∂NLL/∂o_ch = −w · ∂(log-likelihood)/∂q_ch.
            entry[row_idx] = -checked_weighted_row_value(
                wi,
                primary_gradient[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
                row_idx,
                "entry-offset score",
            )
            .map_err(|reason| LatentSurvivalError::NumericalFailure { reason })?;
            exit[row_idx] = -checked_weighted_row_value(
                wi,
                primary_gradient[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
                row_idx,
                "exit-offset score",
            )
            .map_err(|reason| LatentSurvivalError::NumericalFailure { reason })?;
            derivative[row_idx] = -checked_weighted_row_value(
                wi,
                primary_gradient[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT],
                row_idx,
                "derivative-offset score",
            )
            .map_err(|reason| LatentSurvivalError::NumericalFailure { reason })?;
            // Interval upper-bound (`R`) channel. `q_right` shares the time-block
            // coefficients but carries its OWN baseline-θ η-offset evaluated at
            // `R` (`o_R(θ)`), so the profile-NLL θ-gradient must include it.
            // `∂(log-likelihood)/∂q_right` is exactly 0 for non-interval rows
            // (the `Q_RIGHT` channel is inert there), so this is 0 except on
            // interval-censored rows.
            right[row_idx] = -checked_weighted_row_value(
                wi,
                primary_gradient[LATENT_SURVIVAL_PRIMARY_Q_RIGHT],
                row_idx,
                "right-offset score",
            )
            .map_err(|reason| LatentSurvivalError::NumericalFailure { reason })?;
        }
        Ok(crate::survival::OffsetChannelResiduals {
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
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let mut ll = CompensatedRowSum::default();
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
            let wi = weights.at(row_idx);
            if wi == 0.0 {
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
                    LatentSurvivalPrimaryPoint {
                        q_entry: q_entry[row_idx],
                        q_exit: q_exit[row_idx],
                        qdot_exit: qdot_exit[row_idx],
                        q_right: q_right[row_idx],
                        mu: mu[row_idx],
                        sigma,
                    },
                    include_log_sigma,
                )?;
            ll.add(checked_weighted_row_value(
                wi,
                row_ll,
                row_idx,
                "log likelihood",
            )?);
            self.add_pullback_primary_gradient(
                &mut gradient,
                row_idx,
                &slices,
                &primary_gradient,
                wi,
            )?;
            let weighted_primary_hessian =
                checked_weighted_row_matrix(wi, &primary_hessian, row_idx, "primary Hessian")?;
            self.add_pullback_primary_block_diagonals(
                row_idx,
                &weighted_primary_hessian,
                &mut hess_time,
                &mut hess_mean,
                hess_log_sigma.as_mut(),
            )?;
        }
        let ll = require_finite_likelihood_scalar(ll.value(), "log likelihood")?;
        require_finite_likelihood_vector(&gradient, "gradient")?;
        require_finite_likelihood_matrix(&hess_time, "time Hessian")?;
        require_finite_likelihood_matrix(&hess_mean, "mean Hessian")?;
        if let Some(hessian) = hess_log_sigma.as_ref() {
            require_finite_likelihood_matrix(hessian, "log-sigma Hessian")?;
        }
        Ok((ll, gradient, hess_time, hess_mean, hess_log_sigma))
    }

    fn evaluate_exact_newton_joint_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        let acc = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || LatentSurvivalJointDenseAccum {
                ll: CompensatedRowSum::default(),
                gradient: Array1::<f64>::zeros(total),
                hessian: Array2::<f64>::zeros((total, total)),
            },
            |row_idx, acc| {
                let wi = weights.at(row_idx);
                if wi == 0.0 {
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
                        LatentSurvivalPrimaryPoint {
                            q_entry: q_entry[row_idx],
                            q_exit: q_exit[row_idx],
                            qdot_exit: qdot_exit[row_idx],
                            q_right: q_right[row_idx],
                            mu: mu[row_idx],
                            sigma,
                        },
                        include_log_sigma,
                    )?;
                acc.ll.add(checked_weighted_row_value(
                    wi,
                    row_ll,
                    row_idx,
                    "log likelihood",
                )?);
                self.add_pullback_primary_gradient(
                    &mut acc.gradient,
                    row_idx,
                    &slices,
                    &primary_gradient,
                    wi,
                )?;
                let weighted_primary_hessian =
                    checked_weighted_row_matrix(wi, &primary_hessian, row_idx, "primary Hessian")?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &weighted_primary_hessian,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.ll.add(chunk_acc.ll.value());
                total_acc.gradient += &chunk_acc.gradient;
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        let ll = require_finite_likelihood_scalar(acc.ll.value(), "log likelihood")?;
        require_finite_likelihood_vector(&acc.gradient, "gradient")?;
        require_finite_likelihood_matrix(&acc.hessian, "Hessian")?;
        Ok((ll, acc.gradient, acc.hessian))
    }

    fn exact_newton_joint_hessian_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
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
                let wi = weights.at(row_idx);
                if wi == 0.0 {
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
                    LatentSurvivalPrimaryPoint {
                        q_entry: q_entry[row_idx],
                        q_exit: q_exit[row_idx],
                        qdot_exit: qdot_exit[row_idx],
                        q_right: q_right[row_idx],
                        mu: mu[row_idx],
                        sigma,
                    },
                    &direction,
                    include_log_sigma,
                )?;
                let weighted_third =
                    checked_weighted_row_matrix(wi, &third, row_idx, "contracted third")?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &weighted_third,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        require_finite_likelihood_matrix(&acc.hessian, "directional Hessian derivative")?;
        Ok(acc.hessian)
    }

    fn exact_newton_joint_hessian_second_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
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
                let wi = weights.at(row_idx);
                if wi == 0.0 {
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
                    LatentSurvivalPrimaryPoint {
                        q_entry: q_entry[row_idx],
                        q_exit: q_exit[row_idx],
                        qdot_exit: qdot_exit[row_idx],
                        q_right: q_right[row_idx],
                        mu: mu[row_idx],
                        sigma,
                    },
                    &direction_u,
                    &direction_v,
                    include_log_sigma,
                )?;
                let weighted_fourth =
                    checked_weighted_row_matrix(wi, &fourth, row_idx, "contracted fourth")?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &weighted_fourth,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        require_finite_likelihood_matrix(&acc.hessian, "second directional Hessian derivative")?;
        Ok(acc.hessian)
    }
}

fn log_kernel_ratio(
    bundle: &crate::survival::lognormal_kernel::LogLognormalKernelBundle,
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
    // #2610 -- \ is a CANCELLING difference. Past     // the two terms agree to \ of their own size, so the
    // subtraction keeps only the bits they do not share and the result changes
    // SIGN between adjacent 0.05 steps of \. No working precision
    // repairs it: 513 -> 1025 quadrature nodes bought 335x at \ and
    // nothing at \, because the cancelled quantity keeps shrinking
    // while the roundoff floor does not (#2566).
    //
    // \ forms the same quantity without the subtraction:
    // it factors the common ratio out into an \ and takes the second
    // difference of the ANALYTIC prefix in closed form -- exactly \ for
    // every \ and \ -- so only the slowly-varying Laplace half is
    // differenced numerically. It was written for this defect and had NO
    // production caller; this is that caller.
    //
    // It refuses rather than degrade when a rung is missing or an input is
    // non-finite, and the difference form is then the only route left. Taking
    // that route silently is what let this defect hide, so the fallback says so.
    let second_cumulant = match bundle.second_cumulant_ratio(k, sigma) {
        Some(value) => value,
        None => {
            log::warn!(
                "[#2610] cancellation-free second cumulant unavailable at k={k} \
                 (sigma={sigma:.6e}, mass={mass:.6e}); falling back to the \
                 differencing form, whose relative error grows without bound past \
                 log sigma ~ 5.4"
            );
            r2 - r1 * r1
        }
    };
    let d2 = d1 + mass * mass * second_cumulant;
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

#[derive(Clone, Copy, Debug)]
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
    /// the `assert!` in [`binary_from_log_survival`] enforces the
    /// equality.
    neg_hess_scale: f64,
    /// -ℓ''(s). For event=1 this is +S/(1-S)²; for event=0 it is 0.
    outer_scale: f64,
}

/// Exact binary log likelihood from a row log-survival `s`.
///
/// This value-only path deliberately does not evaluate derivatives: near
/// `s = 0`, `log(1-exp(s))` can remain finite after one or more derivatives
/// cease to be representable. A likelihood-only caller must not fail because
/// of an output it did not request.
fn binary_log_likelihood_from_log_survival(
    log_survival: f64,
    event: u8,
) -> Result<f64, LatentSurvivalError> {
    match event {
        0 => {
            if !log_survival.is_finite() || log_survival > 0.0 {
                return Err(LatentSurvivalError::NumericalFailure {
                    reason: format!(
                        "latent-binary requires finite log survival <= 0 for a censored row, got {log_survival:?}"
                    ),
                });
            }
            Ok(log_survival)
        }
        1 => {
            if !log_survival.is_finite() || log_survival >= 0.0 {
                return Err(LatentSurvivalError::NumericalFailure {
                    reason: format!(
                        "latent-binary requires finite log survival < 0 for an observed event, got {log_survival:?}"
                    ),
                });
            }
            let event_prob = -log_survival.exp_m1();
            if !(event_prob.is_finite() && event_prob > 0.0) {
                return Err(LatentSurvivalError::NumericalFailure {
                    reason: format!(
                        "latent-binary event probability is not representable from log survival {log_survival:?}"
                    ),
                });
            }
            Ok(event_prob.ln())
        }
        _ => Err(LatentSurvivalError::InvalidDataset {
            reason: format!("latent-binary requires event targets in {{0,1}}, got {event}"),
        }),
    }
}

/// Value and first log-survival derivative for the binary row transform.
fn binary_from_log_survival_through_first(
    log_survival: f64,
    event: u8,
) -> Result<(f64, f64), LatentSurvivalError> {
    let log_lik = binary_log_likelihood_from_log_survival(log_survival, event)?;
    if event == 0 {
        return Ok((log_lik, 1.0));
    }
    let odds = (log_survival - log_lik).exp();
    if !odds.is_finite() {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent-binary log-survival derivative order 1 is not representable at {log_survival:?}: {odds:?}"
            ),
        });
    }
    Ok((log_lik, -odds))
}

/// Analytic source of truth for derivatives of
/// `ell(s) = log(1 - exp(s))`, evaluated directly in the log-survival
/// coordinate `s < 0`.
///
/// `P = -expm1(s)` avoids cancellation when survival is near one. Writing the
/// derivative algebra in terms of the odds `r = exp(s) / P` avoids the `P²`
/// intermediate that previously underflowed before a finite ratio could be
/// formed. This base routine computes only through order two; third/fourth
/// derivatives are validated lazily by the directional-Hessian paths that
/// consume them, so an unrepresentable unused fourth derivative cannot reject
/// an otherwise representable likelihood, gradient, and Hessian.
fn binary_log_survival_scales(log_survival: f64) -> Result<(f64, f64, f64), LatentSurvivalError> {
    let (log_lik, ell_prime) = binary_from_log_survival_through_first(log_survival, 1)?;
    let odds = -ell_prime;
    let one_plus_odds = 1.0 + odds;
    let ell_pp = -odds * one_plus_odds;
    let scales = [log_lik, ell_prime, ell_pp];
    if let Some((order, value)) = scales
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent-binary log-survival derivative order {order} is not representable at {log_survival:?}: {value:?}"
            ),
        });
    }
    Ok((log_lik, ell_prime, ell_pp))
}

fn binary_from_log_survival(
    log_survival: f64,
    event: u8,
) -> Result<BinaryFromLogSurvival, LatentSurvivalError> {
    if event == 0 {
        // ℓ(s) = s ⇒ ℓ' = 1, ℓ'' = ℓ''' = ℓ'''' = 0.
        return Ok(BinaryFromLogSurvival {
            log_lik: binary_log_likelihood_from_log_survival(log_survival, event)?,
            grad_scale: 1.0,
            neg_hess_scale: 1.0,
            outer_scale: 0.0,
        });
    }
    if event != 1 {
        return Err(LatentSurvivalError::InvalidDataset {
            reason: format!("latent-binary requires event targets in {{0,1}}, got {event}"),
        });
    }
    let (log_lik, ell_prime, ell_pp) = binary_log_survival_scales(log_survival)?;
    let grad_scale = ell_prime;
    let neg_hess_scale = ell_prime; // coefficient on (-d²s/dβ²); equals ℓ'.
    let outer_scale = -ell_pp;
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
    })
}

/// Binary log-survival chain rule through third order. The extra scalar is
/// `d outer_scale / ds = -ℓ'''(s)`. The derivative of `grad_scale` is already
/// available exactly as `-base.outer_scale = ℓ''(s)`.
fn binary_from_log_survival_through_third(
    log_survival: f64,
    event: u8,
) -> Result<(BinaryFromLogSurvival, f64), LatentSurvivalError> {
    let base = binary_from_log_survival(log_survival, event)?;
    if event == 0 {
        return Ok((base, 0.0));
    }
    let odds = -base.grad_scale;
    let ell_pp = -base.outer_scale;
    let ell_ppp = ell_pp * (1.0 + 2.0 * odds);
    if !ell_ppp.is_finite() {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent-binary log-survival derivative order 3 is not representable at {log_survival:?}: {ell_ppp:?}"
            ),
        });
    }
    Ok((base, -ell_ppp))
}

/// Binary log-survival chain rule through fourth order. Returns
/// `(base, -ℓ''', -ℓ'''')`, the first and second derivatives of
/// `base.outer_scale` with respect to log survival.
fn binary_from_log_survival_through_fourth(
    log_survival: f64,
    event: u8,
) -> Result<(BinaryFromLogSurvival, f64, f64), LatentSurvivalError> {
    let (base, outer_scale_prime) = binary_from_log_survival_through_third(log_survival, event)?;
    if event == 0 {
        return Ok((base, 0.0, 0.0));
    }
    let odds = -base.grad_scale;
    let ell_pp = -base.outer_scale;
    let ell_pppp = ell_pp * (1.0 + 6.0 * odds + 6.0 * odds * odds);
    if !ell_pppp.is_finite() {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent-binary log-survival derivative order 4 is not representable at {log_survival:?}: {ell_pppp:?}"
            ),
        });
    }
    Ok((base, outer_scale_prime, -ell_pppp))
}

/// Fitted frailty-scale coordinate used by exact saved latent-survival ALO.
///
/// A fixed scale is likelihood metadata and therefore contributes no fitted
/// coordinate. A learned scale is represented by the exact raw `log_sigma`
/// coefficient consumed by the fitter; replay evaluates `sigma = exp(eta)`
/// inside the same primary row program.
#[derive(Clone, Copy, Debug)]
pub enum LatentSurvivalAloSigma {
    Fixed(f64),
    LearnedLogScale(f64),
}

/// One saved latent-survival row in the fitter's affine primary coordinates.
pub struct LatentSurvivalAloRowInput<'a> {
    pub quadrature: &'a QuadratureContext,
    pub hazard_loading: HazardLoading,
    pub event_code: u8,
    pub prior_weight: f64,
    pub q_entry: f64,
    pub q_exit: f64,
    pub qdot_exit: f64,
    pub q_right: f64,
    pub mu: f64,
    pub sigma: LatentSurvivalAloSigma,
    pub unloaded_mass_entry: f64,
    pub unloaded_mass_exit: f64,
    pub unloaded_mass_right: f64,
    pub unloaded_hazard_exit: f64,
}

/// One saved latent-binary row in its three live affine coordinates
/// `[q_entry, q_exit, mu]`.
pub struct LatentBinaryAloRowInput<'a> {
    pub quadrature: &'a QuadratureContext,
    pub hazard_loading: HazardLoading,
    pub event: u8,
    pub prior_weight: f64,
    pub q_entry: f64,
    pub q_exit: f64,
    pub mu: f64,
    pub sigma: f64,
    pub unloaded_mass_entry: f64,
    pub unloaded_mass_exit: f64,
}

/// Exact NLL row geometry returned by the saved latent-window replay seam.
pub struct LatentWindowAloRowGeometry {
    pub nll_score: Array1<f64>,
    pub observed_hessian: Array2<f64>,
    pub coordinate_values: Array1<f64>,
}

fn validate_saved_alo_weight(weight: f64, context: &str) -> Result<(), String> {
    if weight.is_finite() && weight >= 0.0 {
        Ok(())
    } else {
        Err(format!(
            "{context} prior weight must be finite and non-negative, got {weight}"
        ))
    }
}

fn checked_saved_alo_scale_vector(
    values: Array1<f64>,
    scale: f64,
    context: &str,
) -> Result<Array1<f64>, String> {
    let mut out = Array1::<f64>::zeros(values.len());
    for (axis, value) in values.into_iter().enumerate() {
        let product = scale * value;
        if !product.is_finite() || (scale != 0.0 && value != 0.0 && product == 0.0) {
            return Err(format!(
                "{context}[{axis}] is not representable: {scale:?} * {value:?}"
            ));
        }
        out[axis] = product;
    }
    Ok(out)
}

fn checked_saved_alo_scale_matrix(
    values: Array2<f64>,
    scale: f64,
    context: &str,
) -> Result<Array2<f64>, String> {
    let mut out = Array2::<f64>::zeros(values.dim());
    for ((row, column), value) in values.indexed_iter() {
        let product = scale * value;
        if !product.is_finite() || (scale != 0.0 && *value != 0.0 && product == 0.0) {
            return Err(format!(
                "{context}[{row},{column}] is not representable: {scale:?} * {value:?}"
            ));
        }
        out[[row, column]] = product;
    }
    Ok(out)
}

/// Replay one saved latent-survival likelihood row through the exact fitting
/// program.
///
/// Coordinates are `[q_entry, q_exit, qdot_exit, q_right, mu]` followed by
/// `log_sigma` only when that scale was learned. The primary authority returns
/// log-likelihood score and negative log-likelihood Hessian; this boundary
/// applies the row weight and flips only the score to the NLL convention.
pub fn latent_survival_alo_row_geometry(
    input: LatentSurvivalAloRowInput<'_>,
) -> Result<LatentWindowAloRowGeometry, String> {
    validate_saved_alo_weight(input.prior_weight, "latent-survival ALO")?;
    let mut coordinate_values = vec![
        input.q_entry,
        input.q_exit,
        input.qdot_exit,
        input.q_right,
        input.mu,
    ];
    let (sigma, include_log_sigma) = match input.sigma {
        LatentSurvivalAloSigma::Fixed(sigma) => (sigma, false),
        LatentSurvivalAloSigma::LearnedLogScale(log_sigma) => {
            coordinate_values.push(log_sigma);
            (log_sigma.exp(), true)
        }
    };
    let coordinate_values = Array1::from_vec(coordinate_values);
    let dimension = coordinate_values.len();
    if input.prior_weight == 0.0 {
        return Ok(LatentWindowAloRowGeometry {
            nll_score: Array1::zeros(dimension),
            observed_hessian: Array2::zeros((dimension, dimension)),
            coordinate_values,
        });
    }
    if !matches!(input.event_code, 0 | 1 | LATENT_SURVIVAL_EVENT_INTERVAL) {
        return Err(format!(
            "latent-survival ALO event code must be 0, 1, or the interval sentinel {LATENT_SURVIVAL_EVENT_INTERVAL}, got {}",
            input.event_code
        ));
    }
    if !sigma.is_finite()
        || sigma < 0.0
        || (include_log_sigma && (sigma == 0.0 || !coordinate_values[5].is_finite()))
    {
        return Err(format!(
            "latent-survival ALO frailty scale is invalid: sigma={sigma:?}, learned={include_log_sigma}"
        ));
    }
    if coordinate_values.iter().any(|value| !value.is_finite()) {
        return Err("latent-survival ALO affine coordinates must be finite".to_string());
    }
    let event_type = latent_survival_event_type_for(input.event_code);
    let row = build_latent_survival_row(
        0,
        input.hazard_loading,
        event_type,
        input.q_entry,
        input.q_exit,
        input.qdot_exit,
        input.q_right,
        input.unloaded_mass_entry,
        input.unloaded_mass_exit,
        input.unloaded_mass_right,
        input.unloaded_hazard_exit,
    )
    .map_err(String::from)?;
    let (_, log_likelihood_score, negative_log_likelihood_hessian) =
        latent_survival_row_primary_gradient_hessian(
            input.quadrature,
            &row,
            LatentSurvivalPrimaryPoint {
                q_entry: input.q_entry,
                q_exit: input.q_exit,
                qdot_exit: input.qdot_exit,
                q_right: input.q_right,
                mu: input.mu,
                sigma,
            },
            include_log_sigma,
        )?;
    let nll_score = checked_saved_alo_scale_vector(
        log_likelihood_score.slice(s![0..dimension]).to_owned(),
        -input.prior_weight,
        "latent-survival ALO NLL score",
    )?;
    let observed_hessian = checked_saved_alo_scale_matrix(
        negative_log_likelihood_hessian
            .slice(s![0..dimension, 0..dimension])
            .to_owned(),
        input.prior_weight,
        "latent-survival ALO observed Hessian",
    )?;
    Ok(LatentWindowAloRowGeometry {
        nll_score,
        observed_hessian,
        coordinate_values,
    })
}

/// Replay one saved latent-binary row through the exact right-censored latent
/// survival authority and the fitter's analytic binary-from-log-survival
/// chain. `W` is the observed NLL Hessian; the score outer product remains a
/// separate downstream ALO covariance channel.
pub fn latent_binary_alo_row_geometry(
    input: LatentBinaryAloRowInput<'_>,
) -> Result<LatentWindowAloRowGeometry, String> {
    validate_saved_alo_weight(input.prior_weight, "latent-binary ALO")?;
    let coordinate_values = Array1::from_vec(vec![input.q_entry, input.q_exit, input.mu]);
    const DIMENSION: usize = 3;
    if input.prior_weight == 0.0 {
        return Ok(LatentWindowAloRowGeometry {
            nll_score: Array1::zeros(DIMENSION),
            observed_hessian: Array2::zeros((DIMENSION, DIMENSION)),
            coordinate_values,
        });
    }
    if input.event > 1 {
        return Err(format!(
            "latent-binary ALO event must be 0 or 1, got {}",
            input.event
        ));
    }
    if !input.sigma.is_finite() || input.sigma < 0.0 {
        return Err(format!(
            "latent-binary ALO frailty sigma must be finite and non-negative, got {:?}",
            input.sigma
        ));
    }
    if coordinate_values.iter().any(|value| !value.is_finite()) {
        return Err("latent-binary ALO affine coordinates must be finite".to_string());
    }
    let row = build_latent_survival_row(
        0,
        input.hazard_loading,
        LatentSurvivalEventType::RightCensored,
        input.q_entry,
        input.q_exit,
        1.0,
        input.q_exit,
        input.unloaded_mass_entry,
        input.unloaded_mass_exit,
        0.0,
        0.0,
    )
    .map_err(String::from)?;
    let (log_survival, survival_score, survival_negative_hessian) =
        latent_survival_row_primary_gradient_hessian(
            input.quadrature,
            &row,
            LatentSurvivalPrimaryPoint {
                q_entry: input.q_entry,
                q_exit: input.q_exit,
                qdot_exit: 1.0,
                q_right: input.q_exit,
                mu: input.mu,
                sigma: input.sigma,
            },
            false,
        )?;
    let binary = binary_from_log_survival(log_survival, input.event).map_err(String::from)?;
    let primary_indices = [
        LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
        LATENT_SURVIVAL_PRIMARY_Q_EXIT,
        LATENT_SURVIVAL_PRIMARY_MU,
    ];
    let binary_log_likelihood_score = Array1::from_shape_fn(DIMENSION, |axis| {
        binary.grad_scale * survival_score[primary_indices[axis]]
    });
    let binary_negative_log_likelihood_hessian =
        Array2::from_shape_fn((DIMENSION, DIMENSION), |(left, right)| {
            let source_left = primary_indices[left];
            let source_right = primary_indices[right];
            binary.neg_hess_scale * survival_negative_hessian[[source_left, source_right]]
                + binary.outer_scale * survival_score[source_left] * survival_score[source_right]
        });
    let nll_score = checked_saved_alo_scale_vector(
        binary_log_likelihood_score,
        -input.prior_weight,
        "latent-binary ALO NLL score",
    )?;
    let observed_hessian = checked_saved_alo_scale_matrix(
        binary_negative_log_likelihood_hessian,
        input.prior_weight,
        "latent-binary ALO observed Hessian",
    )?;
    Ok(LatentWindowAloRowGeometry {
        nll_score,
        observed_hessian,
        coordinate_values,
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
    ) -> Result<(), String> {
        for (primary_idx, time_vec) in [
            (LATENT_SURVIVAL_PRIMARY_Q_ENTRY, self.x_time_entry.row(row)),
            (LATENT_SURVIVAL_PRIMARY_Q_EXIT, self.x_time_exit.row(row)),
        ] {
            let scale = checked_weighted_row_value(
                weight,
                primary_gradient[primary_idx],
                row,
                "binary primary gradient",
            )?;
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

        let mean_scale = checked_weighted_row_value(
            weight,
            primary_gradient[LATENT_SURVIVAL_PRIMARY_MU],
            row,
            "binary mean gradient",
        )?;
        if mean_scale != 0.0 {
            self.x_mean
                .axpy_row_into(
                    row,
                    mean_scale,
                    &mut target.slice_mut(s![slices.mean.clone()]),
                )
                .map_err(|error| {
                    format!(
                        "latent binary mean gradient pullback dimension mismatch: row={row}, mean_slice={:?}, target_len={}, x_mean_cols={}, error={error}",
                        slices.mean,
                        target.len(),
                        self.x_mean.ncols()
                    )
                })?;
        }
        Ok(())
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
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        let mut ll = CompensatedRowSum::default();
        let mut gradient = Array1::<f64>::zeros(slices.total);
        let mut hessian = Array2::<f64>::zeros((slices.total, slices.total));
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, survival_hessian) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    LatentSurvivalPrimaryPoint {
                        q_entry: q_entry[row_idx],
                        q_exit: q_exit[row_idx],
                        qdot_exit: 1.0,
                        q_right: q_exit[row_idx],
                        mu: mu[row_idx],
                        sigma: self.latent_sd,
                    },
                    false,
                )?;
            let binary = binary_from_log_survival(row_log_survival, self.event_target[row_idx])?;
            ll.add(checked_weighted_row_value(
                wi,
                binary.log_lik,
                row_idx,
                "binary log likelihood",
            )?);
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
            )?;
            let weighted_primary_hessian = checked_weighted_row_matrix(
                wi,
                &primary_hessian,
                row_idx,
                "binary primary Hessian",
            )?;
            self.add_pullback_primary_hessian(
                &mut hessian,
                row_idx,
                &slices,
                &weighted_primary_hessian,
            );
        }
        let ll = require_finite_likelihood_scalar(ll.value(), "binary log likelihood")?;
        require_finite_likelihood_vector(&gradient, "binary gradient")?;
        require_finite_likelihood_matrix(&hessian, "binary Hessian")?;
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
    /// Sampleweight-scaled to match the `OffsetChannelResiduals` contract.
    pub fn offset_channel_residuals(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<crate::survival::OffsetChannelResiduals, LatentSurvivalError> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")?;
        let n = self.event_target.len();
        // `split_time_eta` returns a typed block-count error before indexing.
        // Missing state is never translated into zero residuals, because that
        // would falsely certify the enclosing baseline optimization.
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let mut entry = Array1::<f64>::zeros(n);
        let mut exit = Array1::<f64>::zeros(n);
        for row_idx in 0..n {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, _) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    LatentSurvivalPrimaryPoint {
                        q_entry: q_entry[row_idx],
                        q_exit: q_exit[row_idx],
                        qdot_exit: 1.0,
                        q_right: q_exit[row_idx],
                        mu: mu[row_idx],
                        sigma: self.latent_sd,
                    },
                    false,
                )?;
            let (_, grad_scale) = binary_from_log_survival_through_first(
                row_log_survival,
                self.event_target[row_idx],
            )?;
            // ∂NLL/∂o_ch = −w · grad_scale · ∂(log S)/∂q_ch.
            entry[row_idx] = -checked_weighted_row_value(
                wi,
                grad_scale * survival_gradient[LATENT_SURVIVAL_PRIMARY_Q_ENTRY],
                row_idx,
                "binary entry-offset score",
            )
            .map_err(|reason| LatentSurvivalError::NumericalFailure { reason })?;
            exit[row_idx] = -checked_weighted_row_value(
                wi,
                grad_scale * survival_gradient[LATENT_SURVIVAL_PRIMARY_Q_EXIT],
                row_idx,
                "binary exit-offset score",
            )
            .map_err(|reason| LatentSurvivalError::NumericalFailure { reason })?;
        }
        Ok(crate::survival::OffsetChannelResiduals {
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
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
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
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let direction = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
            // OneSeed already carries the ordinary value/gradient/Hessian in
            // its base part.  Reuse those channels for the binary outer chain
            // instead of running a separate Order2 row first.
            let row_jet = latent_survival_row_primary_one_seed_fixed_sigma(
                &self.quadctx,
                &row,
                LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[row_idx],
                    q_exit: q_exit[row_idx],
                    qdot_exit: 1.0,
                    q_right: q_exit[row_idx],
                    mu: mu[row_idx],
                    sigma: self.latent_sd,
                },
                &direction,
            )?;
            let (binary, outer_scale_prime) = binary_from_log_survival_through_third(
                row_jet.base.value(),
                self.event_target[row_idx],
            )?;
            let base_gradient = row_jet.base.g();
            let base_hessian = row_jet.base.h();
            let contracted_third = row_jet.contracted_third();
            let survival_gradient = Array1::from_shape_fn(LATENT_SURVIVAL_PRIMARY_DIM, |a| {
                if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA {
                    base_gradient[a]
                } else {
                    0.0
                }
            });
            let survival_hessian = Array2::from_shape_fn(
                (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
                |(a, b)| {
                    if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                        && b < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                    {
                        -base_hessian[a][b]
                    } else {
                        0.0
                    }
                },
            );
            let third = Array2::from_shape_fn(
                (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
                |(a, b)| {
                    if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                        && b < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                    {
                        -contracted_third[a][b]
                    } else {
                        0.0
                    }
                },
            );
            let g_u = -survival_hessian.dot(&direction);
            let t_u = survival_gradient.dot(&direction);
            let mut primary = binary.grad_scale * third;
            primary.scaled_add(-binary.outer_scale * t_u, &survival_hessian);
            for a in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                for b in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                    primary[[a, b]] +=
                        outer_scale_prime * t_u * survival_gradient[a] * survival_gradient[b]
                            + binary.outer_scale
                                * (g_u[a] * survival_gradient[b] + survival_gradient[a] * g_u[b]);
                }
            }
            let weighted_primary =
                checked_weighted_row_matrix(wi, &primary, row_idx, "binary contracted third")?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &weighted_primary);
        }
        require_finite_likelihood_matrix(&out, "binary directional Hessian derivative")?;
        Ok(out)
    }

    fn exact_newton_joint_hessian_second_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
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
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let direction_u = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_u_flat);
            let direction_v = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_v_flat);
            // One TwoSeed row contains the base VGH, both one-seed Hessians,
            // and the mixed two-seed Hessian.  The previous composition ran
            // four complete rows (Order2 + OneSeed(u) + OneSeed(v) +
            // TwoSeed(u,v)) to recover these same channels.
            let row_jet = latent_survival_row_primary_two_seed_fixed_sigma(
                &self.quadctx,
                &row,
                LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[row_idx],
                    q_exit: q_exit[row_idx],
                    qdot_exit: 1.0,
                    q_right: q_exit[row_idx],
                    mu: mu[row_idx],
                    sigma: self.latent_sd,
                },
                &direction_u,
                &direction_v,
            )?;
            let (binary, outer_scale_prime, outer_scale_second) =
                binary_from_log_survival_through_fourth(
                    row_jet.base.value(),
                    self.event_target[row_idx],
                )?;
            let base_gradient = row_jet.base.g();
            let base_hessian = row_jet.base.h();
            let contracted_third_u = row_jet.eps.h();
            let contracted_third_v = row_jet.del.h();
            let contracted_fourth = row_jet.contracted_fourth();
            let survival_gradient = Array1::from_shape_fn(LATENT_SURVIVAL_PRIMARY_DIM, |a| {
                if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA {
                    base_gradient[a]
                } else {
                    0.0
                }
            });
            let pad_matrix =
                |matrix: &[[f64; LATENT_SURVIVAL_PRIMARY_LOG_SIGMA];
                      LATENT_SURVIVAL_PRIMARY_LOG_SIGMA]| {
                    Array2::from_shape_fn(
                        (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
                        |(a, b)| {
                            if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                                && b < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
                            {
                                -matrix[a][b]
                            } else {
                                0.0
                            }
                        },
                    )
                };
            let survival_hessian = pad_matrix(&base_hessian);
            let third_u = pad_matrix(&contracted_third_u);
            let third_v = pad_matrix(&contracted_third_v);
            let fourth = pad_matrix(&contracted_fourth);
            let g_u = -survival_hessian.dot(&direction_u);
            let g_v = -survival_hessian.dot(&direction_v);
            let g_uv = -third_v.dot(&direction_u);
            let t_u = survival_gradient.dot(&direction_u);
            let t_v = survival_gradient.dot(&direction_v);
            let l_uv = -direction_u.dot(&survival_hessian.dot(&direction_v));
            let grad_scale_prime = -binary.outer_scale;
            let grad_scale_second = -outer_scale_prime;
            let c_u = grad_scale_prime * t_u;
            let c_v = grad_scale_prime * t_v;
            let c_uv = grad_scale_second * t_u * t_v + grad_scale_prime * l_uv;
            let o_u = outer_scale_prime * t_u;
            let o_v = outer_scale_prime * t_v;
            let o_uv = outer_scale_second * t_u * t_v + outer_scale_prime * l_uv;
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
            let weighted_primary =
                checked_weighted_row_matrix(wi, &primary, row_idx, "binary contracted fourth")?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &weighted_primary);
        }
        require_finite_likelihood_matrix(&out, "binary second directional Hessian derivative")?;
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
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let include_log_sigma = slices.log_sigma.is_some();
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
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
                LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[row_idx],
                    q_exit: q_exit[row_idx],
                    qdot_exit: qdot_exit[row_idx],
                    q_right: q_right[row_idx],
                    mu: mu[row_idx],
                    sigma,
                },
                include_log_sigma,
            )?;
            let primary_dir = self.row_primary_direction_from_flat(row_idx, slices, v);
            let primary_hv = primary_hessian.dot(&primary_dir);
            self.add_pullback_primary_gradient(out, row_idx, slices, &primary_hv, wi)?;
        }
        require_finite_likelihood_vector(out, "Hessian matvec")?;
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
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (row_log_survival, survival_gradient, survival_hessian) =
                latent_survival_row_primary_gradient_hessian(
                    &self.quadctx,
                    &row,
                    LatentSurvivalPrimaryPoint {
                        q_entry: q_entry[row_idx],
                        q_exit: q_exit[row_idx],
                        qdot_exit: 1.0,
                        q_right: q_exit[row_idx],
                        mu: mu[row_idx],
                        sigma: self.latent_sd,
                    },
                    false,
                )?;
            let binary = binary_from_log_survival(row_log_survival, self.event_target[row_idx])?;
            let primary_dir = self.row_primary_direction_from_flat(row_idx, slices, v);
            let mut primary_hv = binary.grad_scale * survival_hessian.dot(&primary_dir);
            let outer_dot = survival_gradient.dot(&primary_dir);
            for a in 0..LATENT_SURVIVAL_PRIMARY_DIM {
                primary_hv[a] += binary.outer_scale * survival_gradient[a] * outer_dot;
            }
            self.add_pullback_primary_gradient(out, row_idx, slices, &primary_hv, wi)?;
        }
        require_finite_likelihood_vector(out, "binary Hessian matvec")?;
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
    fn warm_up_outer_caches_for_mode(
        &self,
        eval_mode: gam_problem::EvalMode,
    ) -> Result<(), String> {
        match eval_mode {
            gam_problem::EvalMode::ValueOnly
            | gam_problem::EvalMode::ValueAndGradient
            | gam_problem::EvalMode::ValueGradientHessian => Ok(()),
        }
    }

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

/// `CustomFamily` for both latent families. Lexically split out (#2601)
/// when this file hit the 10,000-line ceiling; see `survival/custom_family.rs`.
mod custom_family;

/// The #2566 `log sigma` curvature certificate and its independent authority.
/// Split out so the tracked scanner exemption covers the certificate machinery
/// rather than this whole file, which is the fit math and must stay covered.
mod log_sigma_curvature_certificate;

pub use log_sigma_curvature_certificate::CertifiedLogSigmaCurvature;

#[cfg(test)]
mod tests;
