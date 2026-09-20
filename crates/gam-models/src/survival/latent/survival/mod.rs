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
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix, fit_custom_family_arming_on_evidence,
    fit_custom_family_fixed_log_lambdas,
};
use crate::fit_orchestration::drivers::freeze_term_collection_from_design;
use crate::fit_orchestration::FitFailure;
use crate::gamlss::{FamilyMetadata, ParameterLink};
use crate::model_types::UnifiedFitResult;
use crate::probability::{
    exact_binary64_sum_sign, log1mexp_positive, signed_log_sum_exp,
};
use crate::quadrature::{IntegratedExpectationMode, QuadratureContext};
use crate::sigma_link::{exp_sigma_eta_for_sigma_scalar, exp_sigma_from_eta_scalar};
use crate::survival::construction::SurvivalBaselineConfig;
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

impl LatentSurvivalError {
    /// The fixed category of this refusal, so a fit that stops on it raises the
    /// class that names what failed (#2937).
    #[must_use]
    pub fn failure_category(&self) -> gam_problem::FailureCategory {
        use gam_problem::FailureCategory;
        match self {
            Self::InvalidFrailty { .. }
            | Self::InvalidDataset { .. }
            | Self::UnsupportedConfiguration { .. } => FailureCategory::Input,
            // The fit's own block states and etas disagreeing in length.
            Self::BlockMismatch { .. } => FailureCategory::Invariant,
            Self::NumericalFailure { .. } | Self::DerivativeAccuracyUnresolved { .. } => {
                FailureCategory::Numerical
            }
        }
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
    /// The interval upper bounds `R` the right offsets were realized at, so the
    /// baseline chart moves them with θ. `None` exactly when `time_offset_right`
    /// is `None`.
    pub age_right: Option<Array1<f64>>,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    /// Unloaded (background) cumulative mass at the interval upper bound `R`.
    /// Length-`n`; entries for non-interval rows are ignored. Empty/`None`
    /// folds to zero (full-loading interval rows).
    pub unloaded_mass_right: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
    pub meanspec: TermCollectionSpec,
    pub mean_offset: Array1<f64>,
    /// The parametric baseline the time block's offsets were realized from.
    /// The fit result carries it to every consumer that rebuilds those offsets
    /// at new ages, so the saved model cannot disagree with the fit (#2714).
    pub baseline_config: SurvivalBaselineConfig,
}

pub struct LatentSurvivalTermFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub latent_sd: f64,
    /// The baseline whose offsets this fit's time coefficients were estimated
    /// against; the saved model persists exactly this configuration.
    pub baseline_config: SurvivalBaselineConfig,
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
    /// The parametric baseline the time block's offsets were realized from
    /// (see [`LatentSurvivalTermSpec::baseline_config`]).
    pub baseline_config: SurvivalBaselineConfig,
}

pub struct LatentBinaryTermFitResult {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    /// The baseline whose offsets this fit's time coefficients were estimated
    /// against; the saved model persists exactly this configuration.
    pub baseline_config: SurvivalBaselineConfig,
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
    /// The baseline chart point this family's time offsets were realized at, with
    /// their per-row θ partials: the family-owned hyper axes it serves. `None`
    /// when the baseline carries no outer coordinates (#2714).
    pub(crate) baseline_theta_rows:
        Option<Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>>,
    /// Whether this member's Jeffreys/Firth prior is armed. A fit arms it only
    /// on the unarmed fit's own evidence, through
    /// `fit_custom_family_arming_on_evidence` (#979).
    pub(crate) jeffreys_armed: bool,
}

#[derive(Clone)]
pub(crate) struct LatentBinaryFamily {
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
    /// The baseline chart point this family's time offsets were realized at (see
    /// [`LatentSurvivalFamily::baseline_theta_rows`]).
    pub(crate) baseline_theta_rows:
        Option<Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>>,
    /// Whether this member's Jeffreys/Firth prior is armed. A fit arms it only
    /// on the unarmed fit's own evidence, through
    /// `fit_custom_family_arming_on_evidence` (#979).
    pub(crate) jeffreys_armed: bool,
}

impl LatentSurvivalFamily {
    pub(crate) const BLOCK_TIME: usize = 0;
    pub(crate) const BLOCK_MEAN: usize = 1;
    pub const BLOCK_LOG_SIGMA: usize = 2;

    pub fn parameter_names() -> &'static [&'static str] {
        &["time_transform", "mean"]
    }

    pub(crate) fn parameter_links() -> &'static [ParameterLink] {
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
    pub(crate) const BLOCK_TIME: usize = 0;
    pub(crate) const BLOCK_MEAN: usize = 1;

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

pub(crate) fn fit_latent_survival_terms(
    data: ArrayView2<'_, f64>,
    mut spec: LatentSurvivalTermSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentSurvivalTermFitResult, FitFailure> {
    let frailty_scale = validate_latent_survival_inputs(data, &spec, &frailty)?;
    // Cover the monotone I-spline baseline's unpenalized affine null direction
    // with a REML-selected function-space shrinkage ridge, so the interval
    // warm-start surrogates (whose likelihood has no curvature along that
    // direction) have a unique MAP instead of refusing. No-op on blocks whose
    // penalties already span their column space. See the installer's doc.
    install_latent_time_nullspace_shrinkage_penalty(&mut spec.time_block)
        .map_err(FitFailure::invariant)?;
    let (latent_sd, learned_initial_sigma) = match frailty_scale {
        FrailtyScale::Fixed { sigma } => (Some(sigma), None),
        FrailtyScale::Learned { initial_sigma } => (None, Some(initial_sigma)),
    };
    // The frailty spec is the caller's (#2937).
    let hazard_loading =
        latent_hazard_loading(&frailty, "latent-survival").map_err(FitFailure::input)?;
    let mean_design = build_term_collection_design(data, &spec.meanspec)?;
    let mean_offset =
        mean_design.compose_offset(spec.mean_offset.view(), "latent-survival mean block")?;
    let resolvedspec = freeze_term_collection_from_design(&spec.meanspec, &mean_design)?;
    let time_prepared = prepare_latent_time_block(
        &spec.time_block,
        spec.time_design_right.as_ref(),
        spec.derivative_guard,
    )?;

    let n = spec.event_target.len();
    let time_offset_right = match spec.time_offset_right.as_ref() {
        Some(offset) => {
            if offset.len() != n {
                return Err(FitFailure::input(format!(
                    "latent survival interval right time offset must have length {n}, got {}",
                    offset.len()
                )));
            }
            offset.clone()
        }
        None => Array1::zeros(n),
    };
    let unloaded_mass_right = if spec.unloaded_mass_right.is_empty() {
        Array1::zeros(n)
    } else {
        if spec.unloaded_mass_right.len() != n {
            return Err(FitFailure::input(format!(
                "latent survival interval right unloaded mass must have length {n}, got {}",
                spec.unloaded_mass_right.len()
            )));
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
        baseline_theta_rows: None,
        jeffreys_armed: true,
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
                    // The surrogate's solver error, carried whole (#2937).
                    return Err(FitFailure::from(censored_error).context(
                        "latent interval warm start: right-censored-at-L surrogate fit failed \
                         (so the interval fit cannot be safely warm-started; this surrogate is \
                         log-concave and should converge — investigate the surrogate, not the \
                         interval kernel)",
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
                    FitFailure::from(event_error).context(format!(
                        "latent interval warm start failed: the right-censored-at-L surrogate \
                         has no finite failures and refused its boundary optimum ({censored_error}); \
                         the finite lower-endpoint event surrogate also failed"
                    ))
                })?
            }
        };
        let warm_beta_usable = warm_fit
            .block_states
            .iter()
            .any(|s| s.beta.iter().all(|v| v.is_finite()) && s.beta.iter().any(|&v| v != 0.0));
        if !warm_beta_usable {
            return Err(FitFailure::numerical(
                "latent interval warm start: right-censored-at-L surrogate returned a \
                 degenerate (non-finite or all-zero) β across every block; the warm start \
                 cannot seed the interval fit. This indicates the surrogate's time-block \
                 design is rank-deficient or the inner solve stalled at the seed — \
                 investigate the surrogate before retrying the interval fit.",
            ));
        }
        for (block, state) in blocks.iter_mut().zip(warm_fit.block_states.iter()) {
            if state.beta.iter().all(|v| v.is_finite()) {
                block.initial_beta = Some(state.beta.clone());
            }
        }
    }
    // The baseline chart is a set of family-owned outer coordinates selected
    // together with ρ (#2714): a fully loaded hazard moves only the time offsets,
    // and a loaded/unloaded split also moves its background components.
    let chart = crate::survival::construction::LatentSurvivalFrozenOffsetChart::new(
        &spec.age_entry,
        &spec.age_exit,
        spec.age_right.as_ref(),
        &spec.baseline_config,
        hazard_loading,
        &spec.time_block.offset_entry,
        &spec.time_block.offset_exit,
        &spec.time_block.derivative_offset_exit,
        &family.time_offset_right,
    )
    .map_err(FitFailure::invariant)?;
    let (fit, family, baseline_config) = match chart {
        Some(chart) => fit_latent_baseline_axes(
            data,
            &family,
            &blocks,
            &spec.meanspec,
            &spec.time_block,
            spec.derivative_guard,
            &chart,
            options,
        )?,
        None => {
            let fit = fit_custom_family_arming_on_evidence(&family, &blocks, options)?;
            (fit, family, spec.baseline_config.clone())
        }
    };
    let latent_sd = family.latent_sd(&fit.block_states)?;
    Ok(LatentSurvivalTermFitResult {
        fit,
        design: mean_design,
        resolvedspec,
        latent_sd,
        baseline_config,
    })
}

/// A latent family whose time offsets follow a nonlinear baseline chart (#2714).
trait LatentBaselineChartFamily: CustomFamily + Clone + Send + Sync + 'static {
    /// The time block's index among the family's parameter blocks.
    const TIME_BLOCK: usize;

    /// This family realized at `geometry`, with the derivative-guard constraints
    /// the moved `o_D` realizes.
    fn at_chart_point(
        &self,
        geometry: Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>,
        time_linear_constraints: Option<LinearInequalityConstraints>,
    ) -> Self;

    /// The chart point this family was realized at.
    fn chart_point(
        &self,
    ) -> Option<&Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>>;
}

impl LatentBaselineChartFamily for LatentSurvivalFamily {
    const TIME_BLOCK: usize = Self::BLOCK_TIME;

    fn at_chart_point(
        &self,
        geometry: Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>,
        time_linear_constraints: Option<LinearInequalityConstraints>,
    ) -> Self {
        let mut family = self.clone();
        family.time_linear_constraints = time_linear_constraints;
        family.time_offset_right = geometry.offset_right.clone();
        if let Some(unloaded) = geometry.unloaded.as_ref() {
            family.unloaded_mass_entry = unloaded.mass_entry.clone();
            family.unloaded_mass_exit = unloaded.mass_exit.clone();
            family.unloaded_hazard_exit = unloaded.hazard_exit.clone();
            family.unloaded_mass_right = unloaded.mass_right.clone();
        }
        family.baseline_theta_rows = Some(geometry);
        family
    }

    fn chart_point(
        &self,
    ) -> Option<&Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>> {
        self.baseline_theta_rows.as_ref()
    }
}

impl LatentBaselineChartFamily for LatentBinaryFamily {
    const TIME_BLOCK: usize = Self::BLOCK_TIME;

    /// The binary deployment reads no interval bound and no exit hazard, so the
    /// constraints, a split's background masses and the chart point move.
    fn at_chart_point(
        &self,
        geometry: Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>,
        time_linear_constraints: Option<LinearInequalityConstraints>,
    ) -> Self {
        let mut family = self.clone();
        family.time_linear_constraints = time_linear_constraints;
        if let Some(unloaded) = geometry.unloaded.as_ref() {
            family.unloaded_mass_entry = unloaded.mass_entry.clone();
            family.unloaded_mass_exit = unloaded.mass_exit.clone();
        }
        family.baseline_theta_rows = Some(geometry);
        family
    }

    fn chart_point(
        &self,
    ) -> Option<&Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>> {
        self.baseline_theta_rows.as_ref()
    }
}

/// Fit a latent model with its nonlinear baseline chart as family-owned outer
/// coordinates of the one LAML problem (#2714).
///
/// The workflow used to search θ in a nested BFGS whose every probe was a complete
/// REML fit, against the profile-NLL envelope gradient, which omits the Jeffreys
/// term's mode response and the response of ρ̂ to θ. Here ρ and θ are selected
/// together on the exact criterion. A θ moves the additive time offsets, the
/// derivative-guard constraints built from the moved `o_D` and, for a
/// loaded/unloaded split, the background components; no design, knot or penalty
/// moves. Working precision is the chart's only domain: a θ whose offsets
/// leave the likelihood's domain is refused at evaluation.
fn fit_latent_baseline_axes<F: LatentBaselineChartFamily + crate::custom_family::JeffreysArming>(
    data: ArrayView2<'_, f64>,
    seed_family: &F,
    seed_blocks: &[ParameterBlockSpec],
    meanspec: &TermCollectionSpec,
    time_block_input: &TimeBlockInput,
    derivative_guard: f64,
    chart: &crate::survival::construction::LatentSurvivalFrozenOffsetChart,
    options: &BlockwiseFitOptions,
) -> Result<(UnifiedFitResult, F, SurvivalBaselineConfig), FitFailure> {
    use crate::fit_orchestration::drivers::{
        ExactJointEfsEvaluation, ExactJointEvaluation, ExactJointHyperSetup, SpatialFitProvenance,
        optimize_spatial_length_scale_exact_joint_typed,
    };
    let penalty_counts: Vec<usize> = seed_blocks
        .iter()
        .map(|block| block.penalties.len())
        .collect();
    let rho_dim: usize = penalty_counts.iter().sum();
    let rho0 = Array1::from_iter(
        seed_blocks
            .iter()
            .flat_map(|block| block.initial_log_lambdas.iter().copied()),
    );
    let theta0 = chart.initial_theta().clone();
    let axis_count = theta0.len();
    let (precision_lower, precision_upper) = gam_solve::estimate::rho_domain::precision_box();
    let no_kappa =
        || gam_terms::smooth::SpatialLogKappaCoords::new_with_dims(Array1::zeros(0), Vec::new());
    // ρ is searched on the #2812 domain `fit_custom_family` gives these same seed
    // blocks, over every block that owns a ρ coordinate (#2902 item 15).
    let (rho_lower, rho_upper) =
        crate::fit_orchestration::drivers::realized_blocks_rho_domain(seed_blocks, options, rho_dim)?;
    let setup = ExactJointHyperSetup::new(
        rho0,
        rho_lower,
        rho_upper,
        no_kappa(),
        no_kappa(),
        no_kappa(),
    )
    .with_auxiliary(
            theta0,
            Array1::from_elem(axis_count, precision_lower),
            Array1::from_elem(axis_count, precision_upper),
        );

    let realize = |theta: &Array1<f64>| -> Result<(F, Vec<ParameterBlockSpec>), String> {
        if theta.len() != rho_dim + axis_count {
            return Err(format!(
                "latent outer point has {} coordinates, expected {rho_dim} smoothing and {axis_count} baseline",
                theta.len()
            ));
        }
        let geometry = chart.evaluate(&theta.slice(s![rho_dim..]).to_owned())?;
        let time_linear_constraints = structural_time_coefficient_constraints(
            &time_block_input.design_derivative_exit,
            &geometry.derivative_offset_exit,
            derivative_guard,
        )?;
        let mut blocks = seed_blocks.to_vec();
        let mut cursor = 0usize;
        for (block, &count) in blocks.iter_mut().zip(penalty_counts.iter()) {
            block.initial_log_lambdas = theta.slice(s![cursor..cursor + count]).to_owned();
            cursor += count;
        }
        let time_block = &mut blocks[F::TIME_BLOCK];
        time_block.offset = geometry.offset_exit.clone();
        time_block.stacked_offset = Some(gam_linalg::utils::stack_offsets(&[
            &geometry.offset_entry,
            &geometry.offset_exit,
            &geometry.derivative_offset_exit,
        ]));
        let family = seed_family.at_chart_point(Arc::new(geometry), time_linear_constraints);
        Ok((family, blocks))
    };
    let check_designs = |specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
        if specs.len() == 1 && designs.len() == 1 {
            Ok(())
        } else {
            Err(format!(
                "latent outer driver handed {} term specs and {} designs for its one mean term collection",
                specs.len(),
                designs.len()
            ))
        }
    };
    let family_hyper_layout = |blocks: &[ParameterBlockSpec], theta: &Array1<f64>| {
        crate::custom_family::CustomFamilyHyperLayout::new(
            vec![Vec::new(); blocks.len()],
            (0..axis_count).collect(),
            theta.slice(s![rho_dim..]).to_owned(),
        )
    };
    // The coefficient mode every evaluation starts from is the certified mode of
    // the accepted outer iterate, never the last trial's (#2714). This used to
    // be a slot every successful evaluation overwrote, including a BFGS trial
    // the line search then rejected. Job 657828 measured the cost on
    // `latent_loaded_vs_unloaded_fit_selects_its_background_with_rho_2714`: seed
    // 0's first trial at shape 0.70 solved at a higher cost, and every
    // backtracking probe toward shape 0.01 then started from that trial's mode.
    // The profiled criterion depended on probe order, the way #2668 and #2765
    // measured on the other exact-joint drivers.
    let walk_signals = crate::exact_mode_branch::OuterWalkSignals::default();
    let exact_mode_branch = std::cell::RefCell::new(
        crate::exact_mode_branch::ExactCoefficientModeBranch::new(walk_signals.clone()),
    );
    // Outer ρ-cache β-seed staging slot: promoted once the per-block widths of the
    // realized blocks are known (the survival location-scale contract).
    let pending_beta_seed = std::cell::RefCell::new(None::<Array1<f64>>);
    let promote_pending_seed = |blocks: &[ParameterBlockSpec]| {
        if let Some(beta_seed) = pending_beta_seed.borrow_mut().take() {
            let widths: Vec<usize> = blocks.iter().map(|block| block.design.ncols()).collect();
            match crate::custom_family::CustomFamilyWarmStart::from_cached_beta(&widths, &beta_seed)
            {
                Ok(warm_start) => {
                    if !exact_mode_branch.borrow_mut().install_seed(warm_start) {
                        log::trace!(
                            "[latent] ignored a late outer ρ-cache β seed: an accepted outer iterate already owns the coefficient-mode anchor"
                        );
                    }
                }
                Err(error) => {
                    log::debug!(
                        "[latent] outer ρ-cache β warm start rejected: {error}; the next solve starts cold"
                    );
                }
            }
        }
    };
    let outer_policy = seed_family.outer_derivative_policy(seed_blocks, options);
    // The chart axes are family-owned hyper axes. The evaluator's exact outer
    // Hessian reads their coefficient drift through an owned exact-ψ workspace
    // (`build_psi_drift_deriv_callback`), which neither latent family serves, so a
    // Hessian request refuses every trial point. The route searches first-order
    // until the chart axes have that workspace (#2677).
    let analytic_outer_hessian_available = false;
    let kappa_options = gam_terms::smooth::SpatialLengthScaleOptimizationOptions {
        enabled: false,
        ..Default::default()
    };
    let solved = optimize_spatial_length_scale_exact_joint_typed(
        data,
        std::slice::from_ref(meanspec),
        &[Vec::new()],
        &kappa_options,
        &setup,
        true,
        analytic_outer_hessian_available,
        true,
        Some(walk_signals),
        outer_policy,
        // The final fit: the solver's error is carried whole (#2937). The family
        // and blocks are realized at a theta the driver produced on the chart
        // built above, so failing to realize them is an engine defect.
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], provenance| {
            check_designs(specs, designs).map_err(FitFailure::invariant)?;
            let (family, blocks) = realize(theta).map_err(FitFailure::invariant)?;
            let fit = match provenance {
                SpatialFitProvenance::NoOuterOptimization => {
                    fit_custom_family_arming_on_evidence(&family, &blocks, options)?
                }
                SpatialFitProvenance::Certified { outer, mode } => {
                    let exact_options = crate::outer_subsample::exact_outer_options(options);
                    crate::custom_family::fit_custom_family_fixed_log_lambdas_from_owned_mode(
                        &family,
                        &blocks,
                        &exact_options,
                        mode,
                        theta,
                        outer,
                    )?
                }
            };
            Ok((fit, family))
        },
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         _| {
            check_designs(specs, designs)?;
            let (family, blocks) = realize(theta)?;
            promote_pending_seed(&blocks);
            let rho = theta.slice(s![..rho_dim]).to_owned();
            let hyper_layout = family_hyper_layout(&blocks, theta)?;
            // No exact outer Hessian along the chart axes (see above): ask for the gradient.
            let eval_mode = match eval_mode {
                gam_problem::EvalMode::ValueGradientHessian => {
                    gam_problem::EvalMode::ValueAndGradient
                }
                other => other,
            };
            let eval_options = crate::outer_subsample::exact_outer_options(options);
            let (first_iterate, candidates) =
                exact_mode_branch
                    .borrow_mut()
                    .candidates(eval_mode, theta, &rho);
            if first_iterate {
                log::debug!(
                    "[latent] first derivative-bearing outer evaluation: its certified mode becomes the coefficient-mode anchor every later probe starts from"
                );
            }
            let warm_start = candidates.into_iter().next().flatten();
            let owned = crate::custom_family::evaluate_custom_family_joint_hyper_owned(
                &family,
                &blocks,
                &eval_options,
                &rho,
                &hyper_layout,
                warm_start.as_ref(),
                eval_mode,
            )
            ?;
            exact_mode_branch.borrow_mut().record_value(
                eval_mode,
                theta,
                owned.result.warm_start.clone(),
                owned.result.inner_converged,
            );
            // An unconverged inner state is neither a fit nor a seed (#2902).
            if !owned.result.inner_converged {
                return Err("latent exact joint inner solve did not converge".to_string().into());
            }
            Ok(ExactJointEvaluation {
                objective: owned.result.objective,
                gradient: owned.result.gradient,
                hessian: owned.result.outer_hessian,
                mode: owned.mode,
            })
        },
        // The latent route disables the fixed-point optimizer (the call above
        // passes `disable_fixed_point = true`), so an EFS evaluation is a contract
        // violation. It is refused here rather than served from a second seed
        // policy.
        |_, _: &[TermCollectionSpec], _: &[TermCollectionDesign]| {
            Err::<ExactJointEfsEvaluation<crate::custom_family::CustomFamilyOwnedMode>, _>(
                "latent survival EFS callback invoked even though fixed-point optimization is disabled for this exact joint route"
                    .to_string()
                    .into(),
            )
        },
        crate::marginal_slope_shared::make_beta_seed_validator(&pending_beta_seed),
    )?;
    let (fit, family) = solved.fit;
    let baseline_config = family
        .chart_point()
        .map(|geometry| geometry.baseline_config.clone())
        .ok_or_else(|| FitFailure::invariant("latent baseline-axes fit lost its chart point"))?;
    Ok((fit, family, baseline_config))
}

pub(crate) fn fit_latent_binary_terms(
    data: ArrayView2<'_, f64>,
    spec: LatentBinaryTermSpec,
    frailty: FrailtySpec,
    options: &BlockwiseFitOptions,
) -> Result<LatentBinaryTermFitResult, FitFailure> {
    let latent_sd = validate_latent_binary_inputs(data, &spec, &frailty)?;
    // The frailty spec is the caller's (#2937).
    let (_, hazard_loading) =
        fixed_latent_hazard_frailty(&frailty, "latent-binary").map_err(FitFailure::input)?;
    let mean_design = build_term_collection_design(data, &spec.meanspec)?;
    let mean_offset =
        mean_design.compose_offset(spec.mean_offset.view(), "latent-binary mean block")?;
    let resolvedspec = freeze_term_collection_from_design(&spec.meanspec, &mean_design)?;
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
        baseline_theta_rows: None,
        jeffreys_armed: true,
    };

    let blocks = vec![
        build_time_blockspec(&time_prepared, &spec.time_block),
        build_mean_blockspec(&mean_design, mean_offset),
    ];
    // The binary deployment selects its baseline chart together with ρ, as the
    // survival family does (#2714).
    let chart = crate::survival::construction::LatentSurvivalFrozenOffsetChart::new(
        &spec.age_entry,
        &spec.age_exit,
        None,
        &spec.baseline_config,
        hazard_loading,
        &spec.time_block.offset_entry,
        &spec.time_block.offset_exit,
        &spec.time_block.derivative_offset_exit,
        &Array1::zeros(spec.event_target.len()),
    )
    .map_err(FitFailure::invariant)?;
    let (fit, baseline_config) = match chart {
        Some(chart) => {
            let solved = fit_latent_baseline_axes(
                data,
                &family,
                &blocks,
                &spec.meanspec,
                &spec.time_block,
                spec.derivative_guard,
                &chart,
                options,
            )?;
            (solved.0, solved.2)
        }
        None => {
            let fit = fit_custom_family_arming_on_evidence(&family, &blocks, options)?;
            (fit, spec.baseline_config.clone())
        }
    };
    Ok(LatentBinaryTermFitResult {
        fit,
        design: mean_design,
        resolvedspec,
        baseline_config,
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
/// The shared [`gam_terms::basis::function_space_nullspace_shrinkage`] builds the
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

// [#780 line-count gate] The row kernel's jet machinery lives in the sibling
// `row_kernel_jets.rs`, inlined here so it keeps the SAME module scope.
include!("row_kernel_jets.rs");

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
        .row_sub(&denominator)
        .row_add(&B::Jet::row_constant(unloaded_offset)))
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
/// than scale its tolerance. No discriminator to refuse on exists today: the
/// independent Richardson authority that measured this boundary (it read `0.853`
/// at `log σ = 4` against `109.765` at `log σ = 6`) was removed as unconsumed
/// (d484a091a), together with the certificate type it filled. A genuine repair
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

/// One one-seed lift along a primary direction, read as the row's channels
/// `(∇ℓ, −∇²ℓ, −∇³ℓ[u])` and masked to the live primaries (#2714). A
/// baseline-chart axis moves only the offsets, so its row direction is a primary
/// vector and one lift serves its value, score and information derivative.
fn latent_survival_row_primary_one_seed_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<(Array1<f64>, Array2<f64>, Array2<f64>), LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    if include_log_sigma {
        let backend = LatentOneSeedBackend {
            direction: std::array::from_fn(|a| direction[a]),
        };
        let out = latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &backend, quadctx, row, point,
        )?;
        let gradient = out.base.g();
        let hessian = out.base.h();
        let third = out.contracted_third();
        Ok((
            Array1::from_shape_fn(dim, |a| gradient[a]),
            Array2::from_shape_fn((dim, dim), |(a, b)| -hessian[a][b]),
            Array2::from_shape_fn((dim, dim), |(a, b)| -third[a][b]),
        ))
    } else {
        let out = latent_survival_row_primary_one_seed_fixed_sigma(quadctx, row, point, direction)?;
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        let gradient = out.base.g();
        let hessian = out.base.h();
        let third = out.contracted_third();
        Ok((
            Array1::from_shape_fn(dim, |a| if live(a) { gradient[a] } else { 0.0 }),
            Array2::from_shape_fn((dim, dim), |(a, b)| {
                if live(a) && live(b) {
                    -hessian[a][b]
                } else {
                    0.0
                }
            }),
            Array2::from_shape_fn((dim, dim), |(a, b)| {
                if live(a) && live(b) {
                    -third[a][b]
                } else {
                    0.0
                }
            }),
        ))
    }
}

/// `u ↦ eᵘ/(1 − eᵘ)` with its first four derivatives at a negative log-survival
/// gap `u` (#2714).
///
/// With `ω = eᵘ/(1 − eᵘ)`: `ω′ = ω(1 + ω)`, `ω″ = ω′(1 + 2ω)`,
/// `ω‴ = 2ω′² + (1 + 2ω)ω″` and `ω⁗ = 6ω′ω″ + (1 + 2ω)ω‴`. Every term is a
/// product of positive factors, so no channel cancels.
fn latent_unary_derivatives_survival_odds(
    u: f64,
    context: &str,
) -> Result<[f64; 5], LatentSurvivalError> {
    if !(u.is_finite() && u < 0.0) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!("{context} requires a finite negative log-survival gap, got {u:?}"),
        });
    }
    let odds = u.exp() / -u.exp_m1();
    let spread = 1.0 + 2.0 * odds;
    let first = odds * (1.0 + odds);
    let second = first * spread;
    let third = 2.0 * first * first + spread * second;
    let fourth = 6.0 * first * second + spread * third;
    let stack = [odds, first, second, third, fourth];
    if !stack.iter().all(|value| value.is_finite()) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} odds derivatives are not representable at log-survival gap {u:?}: {stack:?}"
            ),
        });
    }
    Ok(stack)
}

/// A loaded/unloaded row's log-likelihood derivative along the background scale
/// `ln m`, as a jet over the row primaries (#2714).
///
/// Every background component is linear in the Makeham rate `m`, so its
/// `∂/∂ln m` is itself, and the row's derivative closes on the kernel jets its
/// likelihood builds:
///
/// ```text
///   right-censored:  φ = M_U(a_in) − M_U(a_out),
///   exact event:     φ = M_U(a_in) − M_U(a_out) + h_U·K₀ / (h_U·K₀ + q̇·e^q·K₁),
///   interval:        φ = M_U(a_in) − M_U(L) + (M_U(R) − M_U(L))·ω(b − a),
/// ```
///
/// with `a = log K₀(L) − M_U(L)`, `b = log K₀(R) − M_U(R)` and
/// `ω(u) = eᵘ/(1 − eᵘ)`. The jet's gradient and Hessian are `∂_{ln m}` of the
/// row's.
fn latent_survival_row_unloaded_scale_jet<
    const K: usize,
    B: LatentPrimaryJetBackend<K, Jet: JetScalar<K>>,
>(
    backend: &B,
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
) -> Result<B::Jet, LatentSurvivalError> {
    latent_survival_row_background_scale_jet(
        backend,
        quadctx,
        row,
        point,
        LatentBackgroundScaleOrder::First,
    )
}

/// The background-scale derivative of order `order` of one latent survival row
/// (#2677): `φ = ∂ℓ/∂ln m` ([`latent_survival_row_unloaded_scale_jet`]) or
/// `∂φ/∂ln m = ∂²ℓ/∂(ln m)²`. Every background component is linear in `m`, so
/// `∂_{ln m}` maps each component to itself:
///
/// ```text
///   right-censored:  ∂φ = M_U(a_in) − M_U(a_out),
///   exact event:     ∂φ = M_U(a_in) − M_U(a_out) + s·(1 − s),
///   interval:        ∂φ = M_U(a_in) − M_U(L) + Δ·ω(u) − Δ²·ω′(u),
/// ```
///
/// with `s` the background share, `u = b − a` and `Δ = M_U(R) − M_U(L)`, because
/// `∂_{ln m} log s = 1 − s` and `∂_{ln m} u = −Δ`.
fn latent_survival_row_background_scale_jet<
    const K: usize,
    B: LatentPrimaryJetBackend<K, Jet: JetScalar<K>>,
>(
    backend: &B,
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    order: LatentBackgroundScaleOrder,
) -> Result<B::Jet, LatentSurvivalError> {
    let LatentSurvivalPrimaryPoint {
        q_exit,
        qdot_exit,
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
    let out = match row.event_type {
        LatentSurvivalEventType::RightCensored => {
            B::Jet::constant(row.mass_unloaded_entry - row.mass_unloaded_exit)
        }
        LatentSurvivalEventType::ExactEvent => {
            if !(row.hazard_unloaded.is_finite() && row.hazard_unloaded > 0.0) {
                return Err(LatentSurvivalError::NumericalFailure {
                    reason: format!(
                        "latent survival background share requires a positive background hazard, got {:?}",
                        row.hazard_unloaded
                    ),
                });
            }
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
            let numerator = backend.kernel_sum_log(
                quadctx,
                &[
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
                ],
                exit_state,
                &exit_directions,
                "latent survival numerator",
            )?;
            let background = backend.kernel_sum_log(
                quadctx,
                &single_k0,
                exit_state,
                &exit_directions,
                "latent survival background share",
            )?;
            let log_share = background
                .sub(&numerator)
                .add(&B::Jet::constant(row.hazard_unloaded.ln()));
            let share = log_share.value().exp();
            let background_share = match order {
                LatentBackgroundScaleOrder::First => log_share.compose_unary([share; 5]),
                // `∂_{ln m} s = s·(1 − s) = eˣ − e²ˣ` at `x = log s`.
                LatentBackgroundScaleOrder::Second => {
                    log_share.compose_unary(std::array::from_fn(|k| {
                        share - f64::from(1u32 << k) * share * share
                    }))
                }
            };
            background_share.add(&B::Jet::constant(
                row.mass_unloaded_entry - row.mass_unloaded_exit,
            ))
        }
        LatentSurvivalEventType::IntervalCensored => {
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
            let log_left = backend.kernel_sum_log(
                quadctx,
                &single_k0,
                left_state,
                &left_directions,
                "latent survival interval left boundary",
            )?;
            let log_right = backend.kernel_sum_log(
                quadctx,
                &single_k0,
                right_state,
                &right_directions,
                "latent survival interval right boundary",
            )?;
            let gap = log_right
                .add(&B::Jet::constant(-row.mass_unloaded_right))
                .sub(&log_left.add(&B::Jet::constant(-row.mass_unloaded_left)));
            let odds = latent_unary_derivatives_survival_odds(
                gap.value(),
                "latent survival interval background share",
            )?;
            let background_mass = row.mass_unloaded_right - row.mass_unloaded_left;
            let share = match order {
                LatentBackgroundScaleOrder::First => gap.compose_unary(odds).scale(background_mass),
                LatentBackgroundScaleOrder::Second => {
                    gap.compose_unary(latent_scaled_shifted_odds_stack(
                        gap.value(),
                        odds,
                        background_mass,
                        -background_mass,
                        "latent survival interval background share",
                    )?)
                }
            };
            share.add(&B::Jet::constant(
                row.mass_unloaded_entry - row.mass_unloaded_left,
            ))
        }
    };
    if !backend.all_channels_finite(&out) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "latent survival background share is not finite on a {:?} row",
                row.event_type
            ),
        });
    }
    Ok(out)
}

/// Which background-scale derivative a row jet reads (#2677).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LatentBackgroundScaleOrder {
    /// `φ = ∂ℓ/∂ln m`.
    First,
    /// `∂φ/∂ln m = ∂²ℓ/∂(ln m)²`.
    Second,
}

/// `∂_{ln m}` of `scale·ω(u)` when `scale` is linear in `m` and `u` moves at
/// `rate` along `ln m` (#2677): `scale·ω⁽ᵏ⁾(u) + scale·rate·ω⁽ᵏ⁺¹⁾(u)` for
/// `k = 0..4`, from the stack of [`latent_unary_derivatives_survival_odds`] and
/// `ω⁽⁵⁾ = (1 + 2ω)·ω⁗ + 8·ω′·ω‴ + 6·ω″²`.
fn latent_scaled_shifted_odds_stack(
    u: f64,
    odds: [f64; 5],
    scale: f64,
    rate: f64,
    context: &str,
) -> Result<[f64; 5], LatentSurvivalError> {
    let [omega, first, second, third, fourth] = odds;
    let fifth = (1.0 + 2.0 * omega) * fourth + 8.0 * first * third + 6.0 * second * second;
    let derivatives = [omega, first, second, third, fourth, fifth];
    let stack: [f64; 5] =
        std::array::from_fn(|k| scale * (derivatives[k] + rate * derivatives[k + 1]));
    if !stack.iter().all(|value| value.is_finite()) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!(
                "{context} background-scale odds derivatives are not representable at log-survival gap {u:?}: {stack:?}"
            ),
        });
    }
    Ok(stack)
}

/// `(φ, ∇φ, ∇²φ)` of [`latent_survival_row_unloaded_scale_jet`] from one order-2
/// lift, masked to the live primaries (#2714).
fn latent_survival_row_unloaded_scale_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    include_log_sigma: bool,
) -> Result<(f64, Array1<f64>, Array2<f64>), LatentSurvivalError> {
    latent_survival_row_background_scale_channels(
        quadctx,
        row,
        point,
        include_log_sigma,
        LatentBackgroundScaleOrder::First,
    )
}

/// `(∂φ/∂ln m, ∇∂φ/∂ln m, ∇²∂φ/∂ln m)` of
/// [`latent_survival_row_background_scale_jet`] from one order-2 lift, masked to
/// the live primaries (#2677).
fn latent_survival_row_unloaded_scale_second_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    include_log_sigma: bool,
) -> Result<(f64, Array1<f64>, Array2<f64>), LatentSurvivalError> {
    latent_survival_row_background_scale_channels(
        quadctx,
        row,
        point,
        include_log_sigma,
        LatentBackgroundScaleOrder::Second,
    )
}

/// The background-scale channels of order `order` of one latent survival row from
/// one order-2 lift, masked to the live primaries (#2677).
fn latent_survival_row_background_scale_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    include_log_sigma: bool,
    order: LatentBackgroundScaleOrder,
) -> Result<(f64, Array1<f64>, Array2<f64>), LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    if include_log_sigma {
        let jet = latent_survival_row_background_scale_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &LatentOrder2Backend,
            quadctx,
            row,
            point,
            order,
        )?;
        let gradient = jet.g();
        let hessian = jet.h();
        Ok((
            jet.value(),
            Array1::from_shape_fn(dim, |a| gradient[a]),
            Array2::from_shape_fn((dim, dim), |(a, b)| hessian[a][b]),
        ))
    } else {
        let jet = latent_survival_row_background_scale_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
            &LatentOrder2Backend,
            quadctx,
            row,
            point,
            order,
        )?;
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        let gradient = jet.g();
        let hessian = jet.h();
        Ok((
            jet.value(),
            Array1::from_shape_fn(dim, |a| if live(a) { gradient[a] } else { 0.0 }),
            Array2::from_shape_fn((dim, dim), |(a, b)| {
                if live(a) && live(b) {
                    hessian[a][b]
                } else {
                    0.0
                }
            }),
        ))
    }
}

/// `∇³φ[u]` of [`latent_survival_row_unloaded_scale_jet`] from one one-seed lift,
/// masked to the live primaries (#2714).
fn latent_survival_row_unloaded_scale_third(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    if include_log_sigma {
        let backend = LatentOneSeedBackend {
            direction: std::array::from_fn(|a| direction[a]),
        };
        let third = latent_survival_row_unloaded_scale_jet::<LATENT_SURVIVAL_PRIMARY_DIM, _>(
            &backend, quadctx, row, point,
        )?
        .contracted_third();
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| third[a][b]))
    } else {
        let backend = LatentOneSeedBackend {
            direction: std::array::from_fn(|a| direction[a]),
        };
        let third = latent_survival_row_unloaded_scale_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
            &backend, quadctx, row, point,
        )?
        .contracted_third();
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
            if live(a) && live(b) {
                third[a][b]
            } else {
                0.0
            }
        }))
    }
}

/// A latent-binary row's log-likelihood derivative along the background scale
/// `ln m`, as a jet over the fixed-σ primaries (#2714).
///
/// The row's log survival `s` moves by the β-free background shift
/// `c = M_U(a_in) − M_U(a_out)`. A survivor's log-likelihood is `s`, so `φ = c`;
/// an event's is `log(1 − eˢ)`, so `φ = −c·ω(s)` with `ω(s) = eˢ/(1 − eˢ)`.
fn latent_binary_row_unloaded_scale_jet<
    B: LatentPrimaryJetBackend<
            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            Jet: JetScalar<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA>,
        >,
>(
    backend: &B,
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
) -> Result<B::Jet, LatentSurvivalError> {
    latent_binary_row_background_scale_jet(
        backend,
        quadctx,
        row,
        point,
        event,
        LatentBackgroundScaleOrder::First,
    )
}

/// The background-scale derivative of order `order` of one latent-binary row
/// (#2677): `φ = ∂ℓ/∂ln m` ([`latent_binary_row_unloaded_scale_jet`]) or
/// `∂φ/∂ln m`. The shift `c` is linear in `m` and moves the log survival at rate
/// `c`, so a survivor's `∂φ/∂ln m = c` and an event's is `−c·ω(s) − c²·ω′(s)`.
fn latent_binary_row_background_scale_jet<
    B: LatentPrimaryJetBackend<
            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA,
            Jet: JetScalar<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA>,
        >,
>(
    backend: &B,
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    order: LatentBackgroundScaleOrder,
) -> Result<B::Jet, LatentSurvivalError> {
    let shift = row.mass_unloaded_entry - row.mass_unloaded_exit;
    if event == 0 {
        return Ok(B::Jet::constant(shift));
    }
    let log_survival = latent_survival_row_primary_jet::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA, _>(
        backend, quadctx, row, point,
    )?;
    let odds = latent_unary_derivatives_survival_odds(
        log_survival.value(),
        "latent binary background share",
    )?;
    let out = match order {
        LatentBackgroundScaleOrder::First => log_survival.compose_unary(odds).scale(-shift),
        LatentBackgroundScaleOrder::Second => {
            log_survival.compose_unary(latent_scaled_shifted_odds_stack(
                log_survival.value(),
                odds,
                -shift,
                shift,
                "latent binary background share",
            )?)
        }
    };
    if !backend.all_channels_finite(&out) {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: "latent binary background share is not finite on an event row".to_string(),
        });
    }
    Ok(out)
}

/// `(φ, ∇φ, ∇²φ)` of [`latent_binary_row_unloaded_scale_jet`] from one order-2
/// lift, masked to the live primaries (#2714).
fn latent_binary_row_unloaded_scale_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
) -> Result<(f64, Array1<f64>, Array2<f64>), LatentSurvivalError> {
    latent_binary_row_background_scale_channels(
        quadctx,
        row,
        point,
        event,
        LatentBackgroundScaleOrder::First,
    )
}

/// `(∂φ/∂ln m, ∇∂φ/∂ln m, ∇²∂φ/∂ln m)` of
/// [`latent_binary_row_background_scale_jet`] from one order-2 lift, masked to
/// the live primaries (#2677).
fn latent_binary_row_unloaded_scale_second_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
) -> Result<(f64, Array1<f64>, Array2<f64>), LatentSurvivalError> {
    latent_binary_row_background_scale_channels(
        quadctx,
        row,
        point,
        event,
        LatentBackgroundScaleOrder::Second,
    )
}

/// The background-scale channels of order `order` of one latent-binary row from
/// one order-2 lift, masked to the live primaries (#2677).
fn latent_binary_row_background_scale_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    order: LatentBackgroundScaleOrder,
) -> Result<(f64, Array1<f64>, Array2<f64>), LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    let jet = latent_binary_row_background_scale_jet(
        &LatentOrder2Backend,
        quadctx,
        row,
        point,
        event,
        order,
    )?;
    let gradient = jet.g();
    let hessian = jet.h();
    Ok((
        jet.value(),
        Array1::from_shape_fn(dim, |a| if live(a) { gradient[a] } else { 0.0 }),
        Array2::from_shape_fn((dim, dim), |(a, b)| {
            if live(a) && live(b) {
                hessian[a][b]
            } else {
                0.0
            }
        }),
    ))
}

/// `∇³φ[u]` of [`latent_binary_row_unloaded_scale_jet`] from one one-seed lift,
/// masked to the live primaries (#2714).
fn latent_binary_row_unloaded_scale_third(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction: &Array1<f64>,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    let backend = LatentOneSeedBackend {
        direction: std::array::from_fn(|a| direction[a]),
    };
    let third =
        latent_binary_row_unloaded_scale_jet(&backend, quadctx, row, point, event)?.contracted_third();
    Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
        if live(a) && live(b) {
            third[a][b]
        } else {
            0.0
        }
    }))
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

/// `−∂_u ∂_v ∂_w ∇²ℓ` of one latent survival row from one three-seed lift,
/// masked to the live primaries (#2677): the row kernel of the third information
/// derivative `D³H[u, v, e_a]`.
fn latent_survival_row_primary_fifth_contracted(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
    direction_w: &Array1<f64>,
    include_log_sigma: bool,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    if include_log_sigma {
        let backend = LatentThreeSeedBackend::<LATENT_SURVIVAL_PRIMARY_DIM> {
            direction_u: std::array::from_fn(|a| direction_u[a]),
            direction_v: std::array::from_fn(|a| direction_v[a]),
            direction_w: std::array::from_fn(|a| direction_w[a]),
        };
        let fifth = *latent_survival_row_primary_jet(&backend, quadctx, row, point)?.parts[7].h();
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| -fifth[a][b]))
    } else {
        let backend = LatentThreeSeedBackend::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA> {
            direction_u: std::array::from_fn(|a| direction_u[a]),
            direction_v: std::array::from_fn(|a| direction_v[a]),
            direction_w: std::array::from_fn(|a| direction_w[a]),
        };
        let fifth = *latent_survival_row_primary_jet(&backend, quadctx, row, point)?.parts[7].h();
        let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
        Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
            if live(a) && live(b) {
                -fifth[a][b]
            } else {
                0.0
            }
        }))
    }
}

/// `−∂_u ∂_v ∂_w ∇²ℓ_bin` of a latent-binary row from one three-seed fixed-σ lift
/// (#2677). A survivor's log-likelihood is its log survival `s`; an event's is
/// `log(1 − eˢ)`, composed over the same lift.
fn latent_binary_row_contracted_fifth(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
    direction_w: &Array1<f64>,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let backend = LatentThreeSeedBackend::<LATENT_SURVIVAL_PRIMARY_LOG_SIGMA> {
        direction_u: std::array::from_fn(|a| direction_u[a]),
        direction_v: std::array::from_fn(|a| direction_v[a]),
        direction_w: std::array::from_fn(|a| direction_w[a]),
    };
    let log_survival = latent_survival_row_primary_jet(&backend, quadctx, row, point)?;
    let log_likelihood = match event {
        0 => log_survival,
        1 => log_survival
            .row_compose_log1mexp_negative(backend.derivative_order(), "latent binary event")?,
        other => {
            return Err(LatentSurvivalError::InvalidDataset {
                reason: format!("latent-binary requires event targets in {{0,1}}, got {other}"),
            });
        }
    };
    if !log_likelihood.all_channels_finite() {
        return Err(LatentSurvivalError::NumericalFailure {
            reason: format!("latent binary contracted fifth is not finite on an event={event} row"),
        });
    }
    let fifth = *log_likelihood.parts[7].h();
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    Ok(Array2::from_shape_fn((dim, dim), |(a, b)| {
        if live(a) && live(b) {
            -fifth[a][b]
        } else {
            0.0
        }
    }))
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
/// One row's lift request along a primary axis `e_γ`, for a joint-Hessian
/// derivative whose row kernel is linear in that seed (#2714, #2677).
struct LatentSurvivalPrimaryLift<'a> {
    row_idx: usize,
    row: &'a LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    primary: &'a Array1<f64>,
}

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

    /// Every canonical-axis first directional derivative of the joint Hessian,
    /// `Hdot[e_a]` for `a = 0..p`, with one row lift per primary instead of one
    /// full row pass per axis (#2714).
    ///
    /// The per-axis route, `exact_newton_joint_hessian_directional_derivative_dense`
    /// called `p` times, rebuilds every row's kernel bundles and runs a one-seed
    /// third-order lift along `X_i e_a` on each pass, and that sweep is what the
    /// joint-Newton Jeffreys term reads on every cycle. The lift is linear in its
    /// seed and `X_i e_a` only combines the row's primary axes, so each row lifts
    /// once along every primary `e_γ` its design touches, and every axis closes as
    ///
    /// ```text
    ///   Hdot[e_a]_i = X_iᵀ (Σ_γ (X_i e_a)_γ T_{i,γ}) X_i
    /// ```
    ///
    /// through the same chunked pullback reduction the per-axis route runs. The
    /// lifts are held per row (at most six `6×6` matrices) and the axes are
    /// reduced one at a time, so in-flight memory is `O(n + p²)` per axis on top
    /// of the `p³` result rather than a `p³` accumulator per chunk.
    fn exact_newton_joint_hessian_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        let include_log_sigma = self.joint_slices().log_sigma.is_some();
        self.joint_hessian_axes_from_primary_lifts(block_states, "contracted third", |lift| {
            latent_survival_row_primary_third_contracted(
                &self.quadctx,
                lift.row,
                lift.point,
                lift.primary,
                include_log_sigma,
            )
            .map_err(String::from)
        })
    }

    /// `{D³H[u, v, e_a]}` over every coefficient axis, the third information
    /// derivative an armed Jeffreys term's exact outer Hessian reads (#2677). The
    /// row kernel is linear in its third seed, so the axes close as the first
    /// derivative's do, from one three-seed lift along `(X_i u, X_i v, e_γ)` per
    /// touched primary.
    fn exact_newton_joint_hessian_third_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        let slices = self.joint_slices();
        if d_beta_u_flat.len() != slices.total || d_beta_v_flat.len() != slices.total {
            return Err(format!(
                "latent survival joint d3H direction length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                slices.total
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        self.joint_hessian_axes_from_primary_lifts(block_states, "contracted fifth", |lift| {
            let direction_u =
                self.row_primary_direction_from_flat(lift.row_idx, &slices, d_beta_u_flat);
            let direction_v =
                self.row_primary_direction_from_flat(lift.row_idx, &slices, d_beta_v_flat);
            latent_survival_row_primary_fifth_contracted(
                &self.quadctx,
                lift.row,
                lift.point,
                &direction_u,
                &direction_v,
                lift.primary,
                include_log_sigma,
            )
            .map_err(String::from)
        })
    }

    /// Every canonical axis `e_a` of a joint-Hessian derivative whose row kernel is
    /// linear in one primary seed, from one lift per row along every primary `e_γ`
    /// its design touches:
    ///
    /// ```text
    ///   D[e_a]_i = X_iᵀ (Σ_γ (X_i e_a)_γ F_{i,γ}) X_i
    /// ```
    ///
    /// `lift` returns the row kernel `F_{i,γ}`; `channel` names it in refusals.
    fn joint_hessian_axes_from_primary_lifts(
        &self,
        block_states: &[ParameterBlockState],
        channel: &str,
        lift: impl Fn(LatentSurvivalPrimaryLift<'_>) -> Result<Array2<f64>, String> + Sync,
    ) -> Result<Vec<Array2<f64>>, String> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let live_primaries = if include_log_sigma {
            LATENT_SURVIVAL_PRIMARY_DIM
        } else {
            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
        };
        let total = slices.total;
        let unit = |index: usize, len: usize| {
            let mut axis = Array1::<f64>::zeros(len);
            axis[index] = 1.0;
            axis
        };
        let row_lifts: Vec<Option<Vec<Option<Array2<f64>>>>> = (0..self.event_target.len())
            .into_par_iter()
            .map(|row_idx| -> Result<Option<Vec<Option<Array2<f64>>>>, String> {
                if weights.at(row_idx) == 0.0 {
                    return Ok(None);
                }
                // A primary is read by some axis exactly when its design row is
                // nonzero; the mean and log-sigma primaries are read whenever
                // their blocks exist.
                let mut touched = [false; LATENT_SURVIVAL_PRIMARY_DIM];
                touched[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] =
                    self.x_time_entry.row(row_idx).iter().any(|&value| value != 0.0);
                touched[LATENT_SURVIVAL_PRIMARY_Q_EXIT] =
                    self.x_time_exit.row(row_idx).iter().any(|&value| value != 0.0);
                touched[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT] = self
                    .x_time_derivative_exit
                    .row(row_idx)
                    .iter()
                    .any(|&value| value != 0.0);
                touched[LATENT_SURVIVAL_PRIMARY_Q_RIGHT] =
                    self.x_time_right.row(row_idx).iter().any(|&value| value != 0.0);
                touched[LATENT_SURVIVAL_PRIMARY_MU] = !slices.mean.is_empty();
                touched[LATENT_SURVIVAL_PRIMARY_LOG_SIGMA] = include_log_sigma;
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
                let lifts = (0..live_primaries)
                    .map(|gamma| -> Result<Option<Array2<f64>>, String> {
                        if !touched[gamma] {
                            return Ok(None);
                        }
                        lift(LatentSurvivalPrimaryLift {
                            row_idx,
                            row: &row,
                            point,
                            primary: &unit(gamma, LATENT_SURVIVAL_PRIMARY_DIM),
                        })
                        .map(Some)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(Some(lifts))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let axis_quantity = format!("{channel} Hessian axis derivative");
        let mut axes = Vec::with_capacity(total);
        for axis_index in 0..total {
            let axis = unit(axis_index, total);
            let acc = deterministic_latent_survival_row_reduction(
                self.event_target.len(),
                || LatentSurvivalDenseHessianAccum {
                    hessian: Array2::<f64>::zeros((total, total)),
                },
                |row_idx, acc| {
                    let Some(lifts) = row_lifts[row_idx].as_ref() else {
                        return Ok(());
                    };
                    let direction = self.row_primary_direction_from_flat(row_idx, &slices, &axis);
                    let mut combined = Array2::<f64>::zeros((
                        LATENT_SURVIVAL_PRIMARY_DIM,
                        LATENT_SURVIVAL_PRIMARY_DIM,
                    ));
                    for (&component, kernel) in direction.iter().zip(lifts.iter()) {
                        if component == 0.0 {
                            continue;
                        }
                        let kernel = kernel.as_ref().ok_or_else(|| {
                            format!(
                                "latent survival all-axes {channel}: row {row_idx} reads a primary \
                                 whose design row was classified as untouched"
                            )
                        })?;
                        combined.scaled_add(component, kernel);
                    }
                    let weighted = checked_weighted_row_matrix(
                        weights.at(row_idx),
                        &combined,
                        row_idx,
                        channel,
                    )?;
                    self.add_pullback_primary_hessian(
                        &mut acc.hessian,
                        row_idx,
                        &slices,
                        &weighted,
                    )?;
                    Ok(())
                },
                |total_acc, chunk_acc| {
                    total_acc.hessian += &chunk_acc.hessian;
                },
            )?;
            require_finite_likelihood_matrix(&acc.hessian, &axis_quantity)?;
            axes.push(acc.hessian);
        }
        Ok(axes)
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

    /// The row direction of baseline-chart axis `axis`. The chart moves only the
    /// four additive time offsets, so `∂q_i/∂θ_axis` is a primary vector with no
    /// mean or log-σ component (#2714).
    fn baseline_theta_row_direction(
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        row: usize,
        axis: usize,
    ) -> Array1<f64> {
        let mut direction = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        direction[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] = rows.offset_entry_theta[[row, axis]];
        direction[LATENT_SURVIVAL_PRIMARY_Q_EXIT] = rows.offset_exit_theta[[row, axis]];
        direction[LATENT_SURVIVAL_PRIMARY_QDOT_EXIT] =
            rows.derivative_offset_exit_theta[[row, axis]];
        direction[LATENT_SURVIVAL_PRIMARY_Q_RIGHT] = rows.offset_right_theta[[row, axis]];
        direction
    }

    /// The baseline-chart axis a hyper coordinate addresses (#2714).
    pub(crate) fn baseline_theta_family_axis(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_index: usize,
    ) -> Result<(Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>, usize), String>
    {
        latent_baseline_theta_family_axis(self.baseline_theta_rows.as_ref(), hyper_layout, psi_index)
    }

    /// Fixed-β first-order terms of baseline-chart axis `axis` (#2714):
    ///
    /// ```text
    ///   V_θ = −Σ_i w_i ∇ℓ_i·d_i,
    ///   g_θ = Σ_i w_i X_iᵀ (−∇²ℓ_i d_i),
    ///   H_θ = Σ_i w_i X_iᵀ (−∇³ℓ_i[d_i]) X_i,
    /// ```
    ///
    /// with `d_i = ∂q_i/∂θ` the row's offset direction. The Jeffreys information
    /// is the observed Hessian, so `H_θ` is also its explicit θ-derivative.
    fn baseline_theta_psi_terms_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
    ) -> Result<gam_problem::ExactNewtonJointPsiTerms, String> {
        if rows
            .unloaded
            .as_ref()
            .is_some_and(|unloaded| unloaded.axis == axis)
        {
            return self.unloaded_scale_psi_terms_dense(block_states);
        }
        struct BaselinePsiAccum {
            objective: CompensatedRowSum,
            score: Array1<f64>,
            hessian: Array2<f64>,
        }
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let include_log_sigma = slices.log_sigma.is_some();
        let total = slices.total;
        if axis >= rows.theta.len() || rows.offset_exit_theta.nrows() != self.event_target.len() {
            return Err(format!(
                "latent survival baseline psi axis {axis} is outside a {}-coordinate chart realized over {} rows for {} observations",
                rows.theta.len(),
                rows.offset_exit_theta.nrows(),
                self.event_target.len()
            ));
        }
        let acc = deterministic_latent_survival_row_reduction(
            self.event_target.len(),
            || BaselinePsiAccum {
                objective: CompensatedRowSum::default(),
                score: Array1::<f64>::zeros(total),
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
                let direction = Self::baseline_theta_row_direction(rows, row_idx, axis);
                let (gradient, neg_hessian, neg_third) =
                    latent_survival_row_primary_one_seed_channels(
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
                acc.objective.add(checked_weighted_row_value(
                    wi,
                    -gradient.dot(&direction),
                    row_idx,
                    "baseline psi objective",
                )?);
                self.add_pullback_primary_gradient(
                    &mut acc.score,
                    row_idx,
                    &slices,
                    &neg_hessian.dot(&direction),
                    wi,
                )?;
                let weighted_third =
                    checked_weighted_row_matrix(wi, &neg_third, row_idx, "baseline psi third")?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &weighted_third,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.objective.add(chunk_acc.objective.value());
                total_acc.score += &chunk_acc.score;
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        let objective_psi =
            require_finite_likelihood_scalar(acc.objective.value(), "baseline psi objective")?;
        require_finite_likelihood_vector(&acc.score, "baseline psi score")?;
        require_finite_likelihood_matrix(&acc.hessian, "baseline psi information derivative")?;
        Ok(gam_problem::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi: acc.score,
            hessian_psi: acc.hessian,
            hessian_psi_operator: None,
        })
    }

    /// `D_β H_θ[u] = Σ_i w_i X_iᵀ (−∇⁴ℓ_i[d_i, X_i u]) X_i` for baseline-chart
    /// axis `axis` (#2714): the mixed information derivative the explicit
    /// Jeffreys score and curvature read along every coefficient axis.
    fn baseline_theta_hessian_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if rows
            .unloaded
            .as_ref()
            .is_some_and(|unloaded| unloaded.axis == axis)
        {
            return self.unloaded_scale_hessian_directional_derivative_dense(
                block_states,
                d_beta_flat,
            );
        }
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total || axis >= rows.theta.len() {
            return Err(format!(
                "latent survival baseline psi-Hessian derivative: direction length {} against {} coefficients, axis {axis} of {}",
                d_beta_flat.len(),
                slices.total,
                rows.theta.len()
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
                let direction_theta = Self::baseline_theta_row_direction(rows, row_idx, axis);
                let direction_beta =
                    self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
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
                    &direction_theta,
                    &direction_beta,
                    include_log_sigma,
                )?;
                let weighted_fourth =
                    checked_weighted_row_matrix(wi, &fourth, row_idx, "baseline psi fourth")?;
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
        require_finite_likelihood_matrix(
            &acc.hessian,
            "baseline psi information second derivative",
        )?;
        Ok(acc.hessian)
    }

    /// Fixed-β first-order terms of the background-scale axis `ln m` of a
    /// loaded/unloaded chart (#2714). With `φ_i = ∂ℓ_i/∂ln m`
    /// ([`latent_survival_row_unloaded_scale_jet`]):
    ///
    /// ```text
    ///   V_θ = −Σ_i w_i φ_i,   g_θ = −Σ_i w_i X_iᵀ ∇φ_i,   H_θ = −Σ_i w_i X_iᵀ ∇²φ_i X_i.
    /// ```
    fn unloaded_scale_psi_terms_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<gam_problem::ExactNewtonJointPsiTerms, String> {
        struct UnloadedPsiAccum {
            objective: CompensatedRowSum,
            score: Array1<f64>,
            hessian: Array2<f64>,
        }
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
            || UnloadedPsiAccum {
                objective: CompensatedRowSum::default(),
                score: Array1::<f64>::zeros(total),
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
                let (value, gradient, hessian) = latent_survival_row_unloaded_scale_channels(
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
                acc.objective.add(checked_weighted_row_value(
                    wi,
                    -value,
                    row_idx,
                    "background-scale psi objective",
                )?);
                self.add_pullback_primary_gradient(
                    &mut acc.score,
                    row_idx,
                    &slices,
                    &(-&gradient),
                    wi,
                )?;
                let weighted_hessian = checked_weighted_row_matrix(
                    wi,
                    &(-&hessian),
                    row_idx,
                    "background-scale psi information",
                )?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &weighted_hessian,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.objective.add(chunk_acc.objective.value());
                total_acc.score += &chunk_acc.score;
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        let objective_psi = require_finite_likelihood_scalar(
            acc.objective.value(),
            "background-scale psi objective",
        )?;
        require_finite_likelihood_vector(&acc.score, "background-scale psi score")?;
        require_finite_likelihood_matrix(
            &acc.hessian,
            "background-scale psi information derivative",
        )?;
        Ok(gam_problem::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi: acc.score,
            hessian_psi: acc.hessian,
            hessian_psi_operator: None,
        })
    }

    /// `D_β H_θ[u] = −Σ_i w_i X_iᵀ ∇³φ_i[X_i u] X_i` for the background-scale axis
    /// `ln m` of a loaded/unloaded chart (#2714).
    fn unloaded_scale_hessian_directional_derivative_dense(
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
                "latent survival background-scale psi-Hessian derivative: direction length {} against {} coefficients",
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
                let direction_beta =
                    self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
                let third = latent_survival_row_unloaded_scale_third(
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
                    &direction_beta,
                    include_log_sigma,
                )?;
                let weighted_third = checked_weighted_row_matrix(
                    wi,
                    &(-&third),
                    row_idx,
                    "background-scale psi mixed information",
                )?;
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
        require_finite_likelihood_matrix(
            &acc.hessian,
            "background-scale psi information second derivative",
        )?;
        Ok(acc.hessian)
    }

    /// `∇²_β tr(W · H(β))` for a symmetric trace weight `W`, in ONE row pass
    /// (#2714).
    ///
    /// The exact Jeffreys completion otherwise contracts `W` against `p(p+1)/2`
    /// full-data passes `H''[e_a, e_b]`, and every pass rebuilds every row's
    /// kernel bundles. The contraction only needs each row's primary
    /// coordinates:
    ///
    /// ```text
    ///   Σ_cd W_cd H''[e_a, e_b]_cd = Σ_i w_i · x_aᵀ M_i x_b,
    ///   M_i = Σ_γδ Wp_i[γ, δ] · F_i(e_γ, e_δ),   Wp_i = X_i W X_iᵀ,
    /// ```
    ///
    /// where `F_i(u, v)` is the row's contracted fourth
    /// ([`latent_survival_row_primary_fourth_contracted`]) and `X_i` its primary
    /// Jacobian. `F_i` is symmetric in `(u, v)`, so the sum runs over the upper
    /// triangle with the off-diagonal weight doubled: at most 21 row jets per row,
    /// independent of `p`. Linear in `W` and never factorizes it, so a signed
    /// weight carrying gate or floor motion is admitted as the trait requires.
    fn jeffreys_information_contracted_trace_hessian_dense(
        &self,
        block_states: &[ParameterBlockState],
        weight: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-survival")
            .map_err(String::from)?;
        let (q_entry, q_exit, qdot_exit, mu) = self.split_time_eta(block_states)?;
        let q_right = self.time_q_right(block_states)?;
        let sigma = self.latent_sd(block_states)?;
        let slices = self.joint_slices();
        let total = slices.total;
        if weight.dim() != (total, total) {
            return Err(format!(
                "latent survival contracted trace Hessian: weight shape {:?}, expected ({total}, {total})",
                weight.dim()
            ));
        }
        let include_log_sigma = slices.log_sigma.is_some();
        let live_primaries = if include_log_sigma {
            LATENT_SURVIVAL_PRIMARY_DIM
        } else {
            LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
        };
        let primary_axis = |axis: usize| {
            let mut unit = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
            unit[axis] = 1.0;
            unit
        };
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
                let point = LatentSurvivalPrimaryPoint {
                    q_entry: q_entry[row_idx],
                    q_exit: q_exit[row_idx],
                    qdot_exit: qdot_exit[row_idx],
                    q_right: q_right[row_idx],
                    mu: mu[row_idx],
                    sigma,
                };
                // Wp[γ, δ] = x_γᵀ W x_δ with x_δ = X_iᵀ e_δ.
                let mut primary_weight =
                    Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
                for delta in 0..live_primaries {
                    let mut x_delta = Array1::<f64>::zeros(total);
                    self.add_pullback_primary_gradient(
                        &mut x_delta,
                        row_idx,
                        &slices,
                        &primary_axis(delta),
                        1.0,
                    )?;
                    let weighted_column = self.row_primary_direction_from_flat(
                        row_idx,
                        &slices,
                        &weight.dot(&x_delta),
                    );
                    for gamma in 0..live_primaries {
                        primary_weight[[gamma, delta]] = weighted_column[gamma];
                    }
                }
                let mut contracted =
                    Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
                for gamma in 0..live_primaries {
                    for delta in gamma..live_primaries {
                        let pair_weight = if gamma == delta {
                            primary_weight[[gamma, gamma]]
                        } else {
                            primary_weight[[gamma, delta]] + primary_weight[[delta, gamma]]
                        };
                        if pair_weight == 0.0 {
                            continue;
                        }
                        let fourth = latent_survival_row_primary_fourth_contracted(
                            &self.quadctx,
                            &row,
                            point,
                            &primary_axis(gamma),
                            &primary_axis(delta),
                            include_log_sigma,
                        )?;
                        contracted.scaled_add(pair_weight, &fourth);
                    }
                }
                let weighted_contracted = checked_weighted_row_matrix(
                    wi,
                    &contracted,
                    row_idx,
                    "contracted trace fourth",
                )?;
                self.add_pullback_primary_hessian(
                    &mut acc.hessian,
                    row_idx,
                    &slices,
                    &weighted_contracted,
                )?;
                Ok(())
            },
            |total_acc, chunk_acc| {
                total_acc.hessian += &chunk_acc.hessian;
            },
        )?;
        require_finite_likelihood_matrix(&acc.hessian, "contracted trace Hessian")?;
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
            log::debug!(
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

/// The baseline-chart axis a hyper coordinate addresses, after checking that the
/// manifest carries exactly the chart point the family was realized at (#2714).
fn latent_baseline_theta_family_axis(
    rows: Option<&Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>>,
    hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
    psi_index: usize,
) -> Result<(Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>, usize), String> {
    let rows = rows.ok_or_else(|| {
        "latent family was realized with no baseline chart, so it owns no hyper axes".to_string()
    })?;
    let manifest = hyper_layout
        .values()
        .slice(s![hyper_layout.design_axis_count()..]);
    if hyper_layout.family_axis_count() != rows.theta.len()
        || manifest.len() != rows.theta.len()
        || manifest
            .iter()
            .zip(rows.theta.iter())
            .any(|(declared, realized)| declared.to_bits() != realized.to_bits())
    {
        return Err(format!(
            "latent baseline chart was realized at {:?}, but the hyper manifest carries {:?}",
            rows.theta, manifest
        ));
    }
    match hyper_layout.axis(psi_index) {
        Some(crate::custom_family::CustomFamilyHyperAxis::Family { family_axis }) => {
            Ok((Arc::clone(rows), family_axis))
        }
        other => Err(format!(
            "a latent family owns only baseline-chart hyper axes; psi index {psi_index} resolves to {other:?}"
        )),
    }
}

/// A latent-binary row's NLL channels along one primary direction `u`, from one
/// one-seed fixed-σ lift (#2714): `(∇ℓ_bin, −∇²ℓ_bin, D_u(−∇²ℓ_bin))`.
///
/// The binary chain reads the survival value, gradient `g`, Hessian `H` and the
/// one-seed third contraction `T(u)`:
///
/// ```text
///   ∇ℓ_bin        = grad_scale·g,
///   −∇²ℓ_bin      = grad_scale·(−H) + outer_scale·g gᵀ,
///   D_u(−∇²ℓ_bin) = grad_scale·(−T(u)) − outer_scale·(g·u)·(−H) + outer_scale′·(g·u)·g gᵀ
///                   + outer_scale·((−H u) gᵀ + g (−H u)ᵀ),
/// ```
///
/// where `grad_scale`, `outer_scale` and `outer_scale′` are functions of the row's
/// value alone.
fn latent_binary_row_one_seed_channels(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction: &Array1<f64>,
) -> Result<(Array1<f64>, Array2<f64>, Array2<f64>), LatentSurvivalError> {
    let dim = LATENT_SURVIVAL_PRIMARY_DIM;
    let live = |a: usize| a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA;
    // OneSeed already carries the ordinary value/gradient/Hessian in its base
    // part; the binary outer chain reuses those channels instead of running a
    // separate Order2 row first.
    let row_jet = latent_survival_row_primary_one_seed_fixed_sigma(quadctx, row, point, direction)?;
    let (binary, outer_scale_prime) =
        binary_from_log_survival_through_third(row_jet.base.value(), event)?;
    let base_gradient = row_jet.base.g();
    let base_hessian = row_jet.base.h();
    let contracted_third = row_jet.contracted_third();
    let survival_gradient =
        Array1::from_shape_fn(dim, |a| if live(a) { base_gradient[a] } else { 0.0 });
    let survival_hessian = Array2::from_shape_fn((dim, dim), |(a, b)| {
        if live(a) && live(b) {
            -base_hessian[a][b]
        } else {
            0.0
        }
    });
    let third = Array2::from_shape_fn((dim, dim), |(a, b)| {
        if live(a) && live(b) {
            -contracted_third[a][b]
        } else {
            0.0
        }
    });
    let g_u = -survival_hessian.dot(direction);
    let t_u = survival_gradient.dot(direction);
    let mut neg_third = binary.grad_scale * third;
    neg_third.scaled_add(-binary.outer_scale * t_u, &survival_hessian);
    let mut neg_hessian = binary.grad_scale * &survival_hessian;
    for a in 0..dim {
        for b in 0..dim {
            neg_third[[a, b]] += outer_scale_prime * t_u * survival_gradient[a] * survival_gradient[b]
                + binary.outer_scale
                    * (g_u[a] * survival_gradient[b] + survival_gradient[a] * g_u[b]);
            neg_hessian[[a, b]] += binary.outer_scale * survival_gradient[a] * survival_gradient[b];
        }
    }
    let gradient = binary.grad_scale * &survival_gradient;
    Ok((gradient, neg_hessian, neg_third))
}

/// `D_u D_v(−∇²ℓ_bin)` of a latent-binary row, from one two-seed fixed-σ lift
/// (#2714). One TwoSeed row carries the base VGH, both one-seed Hessians and the
/// mixed two-seed Hessian, so the binary chain closes without the four complete
/// rows (Order2 + OneSeed(u) + OneSeed(v) + TwoSeed(u,v)) a composition would run.
fn latent_binary_row_contracted_fourth(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    point: LatentSurvivalPrimaryPoint,
    event: u8,
    direction_u: &Array1<f64>,
    direction_v: &Array1<f64>,
) -> Result<Array2<f64>, LatentSurvivalError> {
    let row_jet =
        latent_survival_row_primary_two_seed_fixed_sigma(quadctx, row, point, direction_u, direction_v)?;
    let (binary, outer_scale_prime, outer_scale_second) =
        binary_from_log_survival_through_fourth(row_jet.base.value(), event)?;
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
        |matrix: &[[f64; LATENT_SURVIVAL_PRIMARY_LOG_SIGMA]; LATENT_SURVIVAL_PRIMARY_LOG_SIGMA]| {
            Array2::from_shape_fn(
                (LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM),
                |(a, b)| {
                    if a < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA && b < LATENT_SURVIVAL_PRIMARY_LOG_SIGMA
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
    let g_u = -survival_hessian.dot(direction_u);
    let g_v = -survival_hessian.dot(direction_v);
    let g_uv = -third_v.dot(direction_u);
    let t_u = survival_gradient.dot(direction_u);
    let t_v = survival_gradient.dot(direction_v);
    let l_uv = -direction_u.dot(&survival_hessian.dot(direction_v));
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
    Ok(primary)
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
            let primary = latent_binary_row_one_seed_channels(
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
                self.event_target[row_idx],
                &direction,
            )?
            .2;
            let weighted_primary =
                checked_weighted_row_matrix(wi, &primary, row_idx, "binary contracted third")?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &weighted_primary);
        }
        require_finite_likelihood_matrix(&out, "binary directional Hessian derivative")?;
        Ok(out)
    }

    /// Every canonical-axis first directional derivative of the latent-binary joint
    /// Hessian from one row pass (#2714): the binary twin of the survival family's
    /// one-build lift.
    ///
    /// Per row the binary chain reads the survival value, gradient `g`, Hessian `H`
    /// and the one-seed third contraction `T(u)` along `u = X_i e_a`:
    ///
    /// ```text
    ///   grad_scale·T(u) − outer_scale·(g·u)·H + outer_scale′·(g·u)·g gᵀ
    ///     + outer_scale·((−H u) gᵀ + g (−H u)ᵀ),
    /// ```
    ///
    /// where `grad_scale`, `outer_scale` and `outer_scale′` are functions of the row's
    /// value alone. That is linear in `u`, and `X_i e_a` only combines the primaries a
    /// binary row reads (`q_entry`, `q_exit`, `μ`), so each row lifts once per primary
    /// and closes every axis. The per-axis sweep this replaces ran `p` serial row
    /// passes on every joint-Newton cycle that forms the Jeffreys term.
    fn exact_newton_joint_hessian_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        const BINARY_PRIMARIES: [usize; 3] = [
            LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
            LATENT_SURVIVAL_PRIMARY_Q_EXIT,
            LATENT_SURVIVAL_PRIMARY_MU,
        ];
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        let total = slices.total;
        let unit = |index: usize, len: usize| {
            let mut axis = Array1::<f64>::zeros(len);
            axis[index] = 1.0;
            axis
        };
        let coefficient_axes: Vec<Array1<f64>> = (0..total).map(|a| unit(a, total)).collect();
        let mut out = vec![Array2::<f64>::zeros((total, total)); total];
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let point = LatentSurvivalPrimaryPoint {
                q_entry: q_entry[row_idx],
                q_exit: q_exit[row_idx],
                qdot_exit: 1.0,
                q_right: q_exit[row_idx],
                mu: mu[row_idx],
                sigma: self.latent_sd,
            };
            let lifts = BINARY_PRIMARIES
                .iter()
                .map(|&gamma| {
                    latent_survival_row_primary_one_seed_fixed_sigma(
                        &self.quadctx,
                        &row,
                        point,
                        &unit(gamma, LATENT_SURVIVAL_PRIMARY_DIM),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            // The base channels do not depend on the seed: read them off the first lift.
            let base = &lifts[0].base;
            let (binary, outer_scale_prime) =
                binary_from_log_survival_through_third(base.value(), self.event_target[row_idx])?;
            let base_gradient = base.g();
            let base_hessian = base.h();
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
            let primary_thirds: Vec<Array2<f64>> = lifts
                .iter()
                .map(|lift| {
                    let contracted_third = lift.contracted_third();
                    Array2::from_shape_fn(
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
                    )
                })
                .collect();
            for (axis, target) in coefficient_axes.iter().zip(out.iter_mut()) {
                let direction = self.row_primary_direction_from_flat(row_idx, &slices, axis);
                let mut third =
                    Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
                for (&gamma, primary_third) in BINARY_PRIMARIES.iter().zip(primary_thirds.iter()) {
                    if direction[gamma] != 0.0 {
                        third.scaled_add(direction[gamma], primary_third);
                    }
                }
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
                self.add_pullback_primary_hessian(target, row_idx, &slices, &weighted_primary);
            }
        }
        for axis in &out {
            require_finite_likelihood_matrix(axis, "binary directional Hessian derivative")?;
        }
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
            let primary = latent_binary_row_contracted_fourth(
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
                self.event_target[row_idx],
                &direction_u,
                &direction_v,
            )?;
            let weighted_primary =
                checked_weighted_row_matrix(wi, &primary, row_idx, "binary contracted fourth")?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &weighted_primary);
        }
        require_finite_likelihood_matrix(&out, "binary second directional Hessian derivative")?;
        Ok(out)
    }

    /// `{D³H[u, v, e_a]}` over every coefficient axis (#2677): per row, one
    /// three-seed fixed-σ lift along `(X_i u, X_i v, e_γ)` for each binary primary,
    /// combined over the axes as the first derivative combines its lifts.
    fn exact_newton_joint_hessian_third_directional_derivative_all_axes_dense(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Vec<Array2<f64>>, String> {
        const BINARY_PRIMARIES: [usize; 3] = [
            LATENT_SURVIVAL_PRIMARY_Q_ENTRY,
            LATENT_SURVIVAL_PRIMARY_Q_EXIT,
            LATENT_SURVIVAL_PRIMARY_MU,
        ];
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        if d_beta_u_flat.len() != slices.total || d_beta_v_flat.len() != slices.total {
            return Err(format!(
                "latent binary joint d3H direction length mismatch: got {} and {}, expected {}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len(),
                slices.total
            ));
        }
        let total = slices.total;
        let unit = |index: usize, len: usize| {
            let mut axis = Array1::<f64>::zeros(len);
            axis[index] = 1.0;
            axis
        };
        let coefficient_axes: Vec<Array1<f64>> = (0..total).map(|a| unit(a, total)).collect();
        let mut out = vec![Array2::<f64>::zeros((total, total)); total];
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let point = LatentSurvivalPrimaryPoint {
                q_entry: q_entry[row_idx],
                q_exit: q_exit[row_idx],
                qdot_exit: 1.0,
                q_right: q_exit[row_idx],
                mu: mu[row_idx],
                sigma: self.latent_sd,
            };
            let direction_u = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_u_flat);
            let direction_v = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_v_flat);
            let primary_fifths = BINARY_PRIMARIES
                .iter()
                .map(|&gamma| {
                    latent_binary_row_contracted_fifth(
                        &self.quadctx,
                        &row,
                        point,
                        self.event_target[row_idx],
                        &direction_u,
                        &direction_v,
                        &unit(gamma, LATENT_SURVIVAL_PRIMARY_DIM),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (axis, target) in coefficient_axes.iter().zip(out.iter_mut()) {
                let direction = self.row_primary_direction_from_flat(row_idx, &slices, axis);
                let mut fifth =
                    Array2::<f64>::zeros((LATENT_SURVIVAL_PRIMARY_DIM, LATENT_SURVIVAL_PRIMARY_DIM));
                for (&gamma, primary_fifth) in BINARY_PRIMARIES.iter().zip(primary_fifths.iter()) {
                    if direction[gamma] != 0.0 {
                        fifth.scaled_add(direction[gamma], primary_fifth);
                    }
                }
                let weighted_fifth =
                    checked_weighted_row_matrix(wi, &fifth, row_idx, "binary contracted fifth")?;
                self.add_pullback_primary_hessian(target, row_idx, &slices, &weighted_fifth);
            }
        }
        for axis in &out {
            require_finite_likelihood_matrix(axis, "binary third directional Hessian derivative")?;
        }
        Ok(out)
    }

    /// The latent-binary row direction of baseline-chart axis `axis`. The
    /// deployment holds `q̇_exit = 1` and reads no interval bound, so only the entry
    /// and exit offsets move (#2714).
    fn baseline_theta_row_direction(
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        row: usize,
        axis: usize,
    ) -> Array1<f64> {
        let mut direction = Array1::<f64>::zeros(LATENT_SURVIVAL_PRIMARY_DIM);
        direction[LATENT_SURVIVAL_PRIMARY_Q_ENTRY] = rows.offset_entry_theta[[row, axis]];
        direction[LATENT_SURVIVAL_PRIMARY_Q_EXIT] = rows.offset_exit_theta[[row, axis]];
        direction
    }

    /// The baseline-chart axis a hyper coordinate addresses (#2714).
    pub(crate) fn baseline_theta_family_axis(
        &self,
        hyper_layout: &crate::custom_family::CustomFamilyHyperLayout,
        psi_index: usize,
    ) -> Result<(Arc<crate::survival::construction::LatentSurvivalOffsetGeometry>, usize), String>
    {
        latent_baseline_theta_family_axis(self.baseline_theta_rows.as_ref(), hyper_layout, psi_index)
    }

    /// Fixed-β first-order terms of baseline-chart axis `axis` for the binary
    /// deployment (#2714), the binary twin of
    /// [`LatentSurvivalFamily::baseline_theta_psi_terms_dense`]:
    /// `V_θ = −Σ w ∇ℓ_bin·d`, `g_θ = Σ w Xᵀ(−∇²ℓ_bin d)`, `H_θ = Σ w Xᵀ D_d(−∇²ℓ_bin) X`.
    fn baseline_theta_psi_terms_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
    ) -> Result<gam_problem::ExactNewtonJointPsiTerms, String> {
        if rows
            .unloaded
            .as_ref()
            .is_some_and(|unloaded| unloaded.axis == axis)
        {
            return self.unloaded_scale_psi_terms_dense(block_states);
        }
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        let total = slices.total;
        if axis >= rows.theta.len() || rows.offset_exit_theta.nrows() != self.event_target.len() {
            return Err(format!(
                "latent binary baseline psi axis {axis} is outside a {}-coordinate chart realized over {} rows for {} observations",
                rows.theta.len(),
                rows.offset_exit_theta.nrows(),
                self.event_target.len()
            ));
        }
        let mut objective = CompensatedRowSum::default();
        let mut score = Array1::<f64>::zeros(total);
        let mut hessian = Array2::<f64>::zeros((total, total));
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let direction = Self::baseline_theta_row_direction(rows, row_idx, axis);
            let (gradient, neg_hessian, neg_third) = latent_binary_row_one_seed_channels(
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
                self.event_target[row_idx],
                &direction,
            )?;
            objective.add(checked_weighted_row_value(
                wi,
                -gradient.dot(&direction),
                row_idx,
                "binary baseline psi objective",
            )?);
            self.add_pullback_primary_gradient(
                &mut score,
                row_idx,
                &slices,
                &neg_hessian.dot(&direction),
                wi,
            )?;
            let weighted_third =
                checked_weighted_row_matrix(wi, &neg_third, row_idx, "binary baseline psi third")?;
            self.add_pullback_primary_hessian(&mut hessian, row_idx, &slices, &weighted_third);
        }
        let objective_psi =
            require_finite_likelihood_scalar(objective.value(), "binary baseline psi objective")?;
        require_finite_likelihood_vector(&score, "binary baseline psi score")?;
        require_finite_likelihood_matrix(&hessian, "binary baseline psi information derivative")?;
        Ok(gam_problem::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi: score,
            hessian_psi: hessian,
            hessian_psi_operator: None,
        })
    }

    /// Fixed-β first-order terms of the background-scale axis `ln m` of the binary
    /// deployment of a loaded/unloaded chart (#2714), with `φ_i = ∂ℓ_i/∂ln m` from
    /// [`latent_binary_row_unloaded_scale_jet`]:
    ///
    /// ```text
    ///   V_θ = −Σ_i w_i φ_i,   g_θ = −Σ_i w_i X_iᵀ ∇φ_i,   H_θ = −Σ_i w_i X_iᵀ ∇²φ_i X_i.
    /// ```
    fn unloaded_scale_psi_terms_dense(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<gam_problem::ExactNewtonJointPsiTerms, String> {
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        let total = slices.total;
        let mut objective = CompensatedRowSum::default();
        let mut score = Array1::<f64>::zeros(total);
        let mut hessian = Array2::<f64>::zeros((total, total));
        for row_idx in 0..self.event_target.len() {
            let wi = weights.at(row_idx);
            if wi == 0.0 {
                continue;
            }
            let row =
                self.build_right_censored_row_at(row_idx, q_entry[row_idx], q_exit[row_idx])?;
            let (value, gradient, curvature) = latent_binary_row_unloaded_scale_channels(
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
                self.event_target[row_idx],
            )?;
            objective.add(checked_weighted_row_value(
                wi,
                -value,
                row_idx,
                "binary background-scale psi objective",
            )?);
            self.add_pullback_primary_gradient(&mut score, row_idx, &slices, &(-&gradient), wi)?;
            let weighted_curvature = checked_weighted_row_matrix(
                wi,
                &(-&curvature),
                row_idx,
                "binary background-scale psi information",
            )?;
            self.add_pullback_primary_hessian(&mut hessian, row_idx, &slices, &weighted_curvature);
        }
        let objective_psi = require_finite_likelihood_scalar(
            objective.value(),
            "binary background-scale psi objective",
        )?;
        require_finite_likelihood_vector(&score, "binary background-scale psi score")?;
        require_finite_likelihood_matrix(
            &hessian,
            "binary background-scale psi information derivative",
        )?;
        Ok(gam_problem::ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi: score,
            hessian_psi: hessian,
            hessian_psi_operator: None,
        })
    }

    /// `D_β H_θ[u] = −Σ_i w_i X_iᵀ ∇³φ_i[X_i u] X_i` for the background-scale axis
    /// `ln m` of the binary deployment of a loaded/unloaded chart (#2714).
    fn unloaded_scale_hessian_directional_derivative_dense(
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
                "latent binary background-scale psi-Hessian derivative: direction length {} against {} coefficients",
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
            let direction_beta = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
            let third = latent_binary_row_unloaded_scale_third(
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
                self.event_target[row_idx],
                &direction_beta,
            )?;
            let weighted_third = checked_weighted_row_matrix(
                wi,
                &(-&third),
                row_idx,
                "binary background-scale psi mixed information",
            )?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &weighted_third);
        }
        require_finite_likelihood_matrix(
            &out,
            "binary background-scale psi information second derivative",
        )?;
        Ok(out)
    }

    /// `D_β H_θ[u] = Σ w Xᵀ D_d D_{Xu}(−∇²ℓ_bin) X` for baseline-chart axis `axis`
    /// of the binary deployment (#2714).
    fn baseline_theta_hessian_directional_derivative_dense(
        &self,
        block_states: &[ParameterBlockState],
        rows: &crate::survival::construction::LatentSurvivalOffsetGeometry,
        axis: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        if rows
            .unloaded
            .as_ref()
            .is_some_and(|unloaded| unloaded.axis == axis)
        {
            return self.unloaded_scale_hessian_directional_derivative_dense(
                block_states,
                d_beta_flat,
            );
        }
        let weights = ValidatedLikelihoodWeights::new(&self.weights, "latent-binary")
            .map_err(String::from)?;
        let (q_entry, q_exit, mu) = self.split_time_eta(block_states)?;
        let slices = self.joint_slices();
        if d_beta_flat.len() != slices.total || axis >= rows.theta.len() {
            return Err(format!(
                "latent binary baseline psi-Hessian derivative: direction length {} against {} coefficients, axis {axis} of {}",
                d_beta_flat.len(),
                slices.total,
                rows.theta.len()
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
            let direction_theta = Self::baseline_theta_row_direction(rows, row_idx, axis);
            let direction_beta = self.row_primary_direction_from_flat(row_idx, &slices, d_beta_flat);
            let fourth = latent_binary_row_contracted_fourth(
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
                self.event_target[row_idx],
                &direction_theta,
                &direction_beta,
            )?;
            let weighted_fourth =
                checked_weighted_row_matrix(wi, &fourth, row_idx, "binary baseline psi fourth")?;
            self.add_pullback_primary_hessian(&mut out, row_idx, &slices, &weighted_fourth);
        }
        require_finite_likelihood_matrix(
            &out,
            "binary baseline psi information second derivative",
        )?;
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

    /// Every canonical-axis first Hessian derivative from one build of the
    /// family's row lifts (#2714).
    fn ws_dh_all_axes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String>;

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

    fn ws_dh_all_axes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_all_axes_dense(block_states)
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

    fn ws_dh_all_axes(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Vec<Array2<f64>>, String> {
        self.exact_newton_joint_hessian_directional_derivative_all_axes_dense(block_states)
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
    /// One dense joint Hessian for this workspace's fixed β. A joint-Newton
    /// cycle reads it as the solve source and again for the Jeffreys term
    /// (`hessian_dense_forced`), and each read was a full order-two row pass
    /// over the same immutable states (#2714). Failures are cached too: an
    /// assembly error at fixed β is deterministic.
    dense_hessian: std::sync::OnceLock<Result<Array2<f64>, String>>,
}

impl<F: LatentJointHessianFamily> LatentHessianWorkspace<F> {
    fn new(family: F, block_states: Vec<ParameterBlockState>) -> Self {
        let slices = family.ws_joint_slices();
        Self {
            family,
            block_states,
            slices,
            dense_hessian: std::sync::OnceLock::new(),
        }
    }

    fn cached_dense_hessian(&self) -> Result<Array2<f64>, String> {
        self.dense_hessian
            .get_or_init(|| {
                self.family
                    .ws_evaluate_dense(&self.block_states)
                    .map(|(_, _, hessian)| hessian)
            })
            .clone()
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
        self.cached_dense_hessian().map(Some)
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
        let dense = self.cached_dense_hessian()?;
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

    fn directional_derivative_all_axes(&self) -> Result<Option<Vec<Array2<f64>>>, String> {
        self.family.ws_dh_all_axes(&self.block_states).map(Some)
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
mod baseline_chart_pairs;
mod custom_family;

#[cfg(test)]
mod tests_rho_domain_2902;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_third_information_2677;

#[cfg(test)]
mod tests_failure_category_2937 {
    use super::LatentSurvivalError;
    use crate::fit_orchestration::FitFailure;
    use gam_problem::FailureCategory;

    /// #2937: a latent fit that stops on a `LatentSurvivalError` raises the class
    /// of its variant's category, with the refusal's own text.
    #[test]
    fn latent_survival_error_failure_categories_2937() {
        let reason = || "row 3".to_string();
        let cases = [
            (LatentSurvivalError::InvalidFrailty { reason: reason() }, FailureCategory::Input),
            (LatentSurvivalError::InvalidDataset { reason: reason() }, FailureCategory::Input),
            (
                LatentSurvivalError::UnsupportedConfiguration { reason: reason() },
                FailureCategory::Input,
            ),
            (LatentSurvivalError::BlockMismatch { reason: reason() }, FailureCategory::Invariant),
            (
                LatentSurvivalError::NumericalFailure { reason: reason() },
                FailureCategory::Numerical,
            ),
            (
                LatentSurvivalError::DerivativeAccuracyUnresolved { reason: reason() },
                FailureCategory::Numerical,
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.failure_category(), expected, "{error}");
            let failure = FitFailure::from(error.clone());
            assert_eq!(failure.category(), expected, "{failure}");
            assert_eq!(failure.to_string(), error.to_string());
        }
    }
}
