//! Survival model construction helpers.
//!
//! Types and functions for building survival model components:
//! - Baseline hazard targets (Weibull, Gompertz, Gompertz-Makeham)
//! - Time basis construction (I-spline on log-time)
//! - Baseline offset computation
//! - Time wiggle construction
//!
//! These are the building blocks a library consumer needs to construct
//! a `FitRequest::SurvivalLocationScale` without going through the CLI.

use crate::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
    BasisMetadata, BasisOptions, Dense, KnotSource, OneDimensionalBoundary, build_bspline_basis_1d,
    create_basis, evaluate_bspline_derivative_scalar,
};
use crate::families::survival::location_scale::{
    DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD, ResidualDistribution,
    SurvivalCovariateTermBlockTemplate,
};
use crate::families::survival::lognormal_kernel::HazardLoading;
use crate::families::survival::marginal_slope::DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD;
use crate::families::wiggle::{
    WiggleBlockConfig, append_selected_wiggle_penalty_orders, buildwiggle_block_input_from_seed,
    monotone_wiggle_basis_with_derivative_order, split_wiggle_penalty_orders,
};
use crate::inference::formula_dsl::LinkWiggleFormulaSpec;
use crate::matrix::{DenseDesignMatrix, DesignMatrix, SparseDesignMatrix, symmetrize_in_place};
use crate::probability::{normal_pdf, standard_normal_quantile};
use crate::types::{InverseLink, StandardLink};
use ndarray::{Array1, Array2, array, s};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Typed error
// ---------------------------------------------------------------------------

/// Structured failure surface for survival-model construction helpers
/// (`parse_*`, baseline-config builders, time-basis construction). Every
/// variant carries a free-form `reason: String` payload; `Display` emits
/// that payload verbatim, so converting to `String` via the `From` impl
/// produces text byte-equivalent to the pre-refactor `Err(format!(...))`
/// call sites that were the only producers in this module.
///
/// The public CLI-input parsers (`parse_survival_distribution`,
/// `parse_survival_likelihood_mode`, `parse_survival_baseline_config`)
/// keep their `Result<_, String>` signatures — string is the natural
/// failure type for free-form user input — and route through this enum
/// internally via `From<SurvivalConstructionError> for String`.
#[derive(Clone, Debug)]
pub enum SurvivalConstructionError {
    /// User-supplied configuration is malformed or out of range (knot
    /// counts, anchor offsets, derivative guards, ranks).
    InvalidConfig { reason: String },
    /// A required column or block of metadata is absent (e.g. saved
    /// survival ispline keep_cols, baseline target on a saved fit).
    MissingColumn { reason: String },
    /// Per-row / per-column shape disagreement (entry/exit lengths,
    /// penalty rank vs basis width, basis vs coefficient counts).
    IncompatibleDimensions { reason: String },
    /// Numeric / domain rejection: non-finite ratios, non-positive
    /// survival times, monotonicity violations, ispline-derivative
    /// underflow.
    DataValidationFailed { reason: String },
    /// Underlying basis / penalty builder rejected the construction
    /// request (invalid spline order, ispline keep_cols out of range,
    /// internal empty ispline time basis).
    BasisConstructionFailed { reason: String },
    /// User-named distribution / likelihood-mode / baseline target /
    /// time-basis kind is not one we recognise.
    UnsupportedDistribution { reason: String },
}

crate::impl_reason_error_boilerplate! {
    SurvivalConstructionError {
        InvalidConfig,
        MissingColumn,
        IncompatibleDimensions,
        DataValidationFailed,
        BasisConstructionFailed,
        UnsupportedDistribution,
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalBaselineTarget {
    /// No additional parametric target:
    /// eta_target(t) = 0, so regularized model defaults to linear log-cumulative
    /// hazard from the existing time basis.
    Linear,
    /// Parametric target: Weibull baseline.
    ///
    /// Transformation/cloglog survival uses `eta_target(t) = log(H0(t))`;
    /// marginal-slope probit survival uses `q(t) = -Phi^-1(exp(-H0(t)))`.
    Weibull,
    /// Parametric target: Gompertz baseline.
    ///
    /// Transformation/cloglog survival uses `eta_target(t) = log(H0(t))`;
    /// marginal-slope probit survival uses `q(t) = -Phi^-1(exp(-H0(t)))`.
    Gompertz,
    /// Parametric target: Gompertz-Makeham baseline.
    ///
    /// Transformation/cloglog survival uses `eta_target(t) = log(H0(t))`;
    /// marginal-slope probit survival uses `q(t) = -Phi^-1(exp(-H0(t)))`.
    GompertzMakeham,
}

#[derive(Clone, Debug)]
pub struct SurvivalBaselineConfig {
    pub target: SurvivalBaselineTarget,
    pub scale: Option<f64>,
    pub shape: Option<f64>,
    pub rate: Option<f64>,
    pub makeham: Option<f64>,
}

#[derive(Clone, Debug)]
pub enum SurvivalTimeBasisConfig {
    None,
    Linear,
    BSpline {
        degree: usize,
        knots: Array1<f64>,
        smooth_lambda: f64,
    },
    /// I-spline value rows on the `log(t)` axis with non-negative
    /// coefficients (`γ ≥ 0`) enforcing structural monotonicity of
    /// `q(t) = I_basis(log t) · γ`. This replaces the row-wise
    /// `D β + o ≥ guard` derivative-guard constraints the marginal-slope
    /// family previously relied on.
    ///
    /// The design builder lives below at `_build_time_block`'s
    /// `SurvivalTimeBasisConfig::ISpline` arm and exposes:
    ///
    /// * `x_entry_time` / `x_exit_time` — I-spline value rows on the
    ///   `log(t)` axis. Non-negative entries plus `γ ≥ 0` give a
    ///   monotone-non-decreasing `q(t)`, the structural property the
    ///   marginal-slope family needs.
    /// * `x_derivative_time` — right-cumulative B-spline-derivative on
    ///   `log(t)` scaled by `1/t`, again non-negative with `γ ≥ 0`, so
    ///   `q'(t) ≥ 0` pointwise. The `derivative_guard` constant is added
    ///   externally by [`add_survival_time_derivative_guard_offset`],
    ///   leaving the derivative guarantee `q'(t) ≥ guard` exact.
    /// * 2nd-difference penalty on the underlying degree-`(k+1)` B-spline
    ///   coefficients, filtered through `keep_cols` for identifiability.
    ///
    /// `TimeBlockInput::time_monotonicity` declares to the consuming
    /// family how monotonicity is enforced. The marginal-slope
    /// construction site sets it to
    /// [`crate::families::survival::location_scale::TimeBlockMonotonicity::StructuralISpline`]
    /// so the family skips row-wise `D β + o ≥ guard` constraint
    /// generation and treats `γ ≥ 0` as the sole derivative-guard
    /// mechanism. The universal `validate_time_qd1_feasible` safety net
    /// runs regardless.
    ///
    /// An earlier iteration proposed a separate C-spline antiderivative
    /// parameterization that put `q'(t)` in the I-spline space and `q(t)`
    /// in the integral-of-I-spline space. That was mathematically
    /// equivalent but a strictly worse fit for the codebase (extra basis
    /// degree, an extra antiderivative builder, an extra identifiability
    /// path, an extra penalty); it was removed in favor of the canonical
    /// I-spline-value path here.
    ISpline {
        degree: usize,
        knots: Array1<f64>,
        keep_cols: Vec<usize>,
        smooth_lambda: f64,
    },
}

/// Persistable snapshot of the time-basis state used by a survival fit.
///
/// Every survival family routes through [`SurvivalTimeBuildOutput`] during
/// the fit, but the FFI save path needs only the metadata — not the full
/// design matrices. This struct is the single source of truth that flows
/// from the workflow-level basis construction, through the family-specific
/// fit result, into the saved-model payload via
/// [`crate::inference::model::FittedModelPayload::apply_survival_time_basis`].
///
/// Threading this snapshot end-to-end eliminates the prior bug pattern
/// where each FFI builder had to reconstruct the metadata from
/// `fit_config` + the formula (silent drift risk; one builder forgetting
/// to do so caused the marginal-slope save→load break).
#[derive(Clone, Debug, PartialEq)]
pub struct SavedSurvivalTimeBasis {
    pub basisname: String,
    pub degree: Option<usize>,
    pub knots: Option<Vec<f64>>,
    pub keep_cols: Option<Vec<usize>>,
    pub smooth_lambda: Option<f64>,
    pub anchor: f64,
}

impl SavedSurvivalTimeBasis {
    /// Build a snapshot from the realised time-basis state and the entry
    /// anchor that was used during the fit.
    pub fn from_build(build: &SurvivalTimeBuildOutput, anchor: f64) -> Self {
        Self {
            basisname: build.basisname.clone(),
            degree: build.degree,
            knots: build.knots.clone(),
            keep_cols: build.keep_cols.clone(),
            smooth_lambda: build.smooth_lambda,
            anchor,
        }
    }
}

#[derive(Clone)]
pub struct SurvivalTimeBuildOutput {
    pub x_entry_time: DesignMatrix,
    pub x_exit_time: DesignMatrix,
    pub x_derivative_time: DesignMatrix,
    pub penalties: Vec<Array2<f64>>,
    /// Structural nullspace dimension of each penalty matrix.
    pub nullspace_dims: Vec<usize>,
    pub basisname: String,
    pub degree: Option<usize>,
    pub knots: Option<Vec<f64>>,
    pub keep_cols: Option<Vec<usize>>,
    pub smooth_lambda: Option<f64>,
}

pub const SURVIVAL_TIME_FLOOR: f64 = 1e-9;

/// Seed smoothing penalty `λ` used when a survival time basis is reconstructed
/// from a build (or saved model) that did not carry an explicit `smooth_lambda`.
/// This is only an initial value for the REML smoothing search, not a fixed
/// policy: a small positive seed keeps the baseline spline lightly regularized
/// at the start so the outer optimizer begins from a well-conditioned point and
/// then adapts `λ` to the data. Kept in one place so the b-spline and i-spline
/// reconstruction paths cannot drift apart.
const SURVIVAL_TIME_SMOOTH_LAMBDA_SEED: f64 = 1e-2;

/// Default initial Gompertz / Gompertz-Makeham shape parameter when the user
/// does not supply `--baseline-shape`. The Gompertz hazard is
/// `h(t) = rate · exp(shape · t)`; a near-zero shape seeds the baseline at an
/// almost-flat (exponential-like) hazard, letting the fit grow the
/// age-acceleration term from the data rather than committing to a strong
/// curvature up front. Shared by the parse and fit-seed paths so both start
/// from the same neutral shape.
const GOMPERTZ_DEFAULT_SHAPE_SEED: f64 = 0.01;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalLikelihoodMode {
    Transformation,
    Weibull,
    LocationScale,
    MarginalSlope,
    Latent,
    LatentBinary,
}

pub struct SurvivalTimeWiggleBuild {
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub knots: Array1<f64>,
    pub degree: usize,
    pub ncols: usize,
}

// ---------------------------------------------------------------------------
// Time normalization
// ---------------------------------------------------------------------------

pub fn normalize_survival_time_pair(
    entry_raw: f64,
    exit_raw: f64,
    row_index: usize,
) -> Result<(f64, f64), String> {
    if !entry_raw.is_finite() || !exit_raw.is_finite() {
        return Err(SurvivalConstructionError::DataValidationFailed {
            reason: format!("non-finite survival times at row {}", row_index + 1),
        }
        .into());
    }
    if entry_raw < 0.0 || exit_raw < 0.0 {
        return Err(SurvivalConstructionError::DataValidationFailed {
            reason: format!("negative survival times at row {}", row_index + 1),
        }
        .into());
    }

    let entry = entry_raw.max(SURVIVAL_TIME_FLOOR);
    let exit = exit_raw.max(entry + SURVIVAL_TIME_FLOOR);
    Ok((entry, exit))
}

// ---------------------------------------------------------------------------
// Basis monotonicity helpers
// ---------------------------------------------------------------------------

pub fn survival_basis_supports_structural_monotonicity(basisname: &str) -> bool {
    basisname.eq_ignore_ascii_case("ispline")
}

pub fn require_structural_survival_time_basis(
    basisname: &str,
    context: &str,
) -> Result<(), String> {
    if survival_basis_supports_structural_monotonicity(basisname) {
        return Ok(());
    }
    Err(SurvivalConstructionError::UnsupportedDistribution {
        reason: format!(
            "{context} requires a structural monotone survival time basis, but got '{basisname}'. \
Only `ispline` is accepted here because its basis functions enforce a monotone cumulative time effect by construction. \
`{basisname}` can fit non-monotone shapes, which can break survival semantics. \
Re-run with `--time-basis ispline`."
        ),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Baseline config parsing
// ---------------------------------------------------------------------------

pub fn parse_survival_baseline_config(
    target_raw: &str,
    scale: Option<f64>,
    shape: Option<f64>,
    rate: Option<f64>,
    makeham: Option<f64>,
) -> Result<SurvivalBaselineConfig, String> {
    let target = match target_raw.to_ascii_lowercase().as_str() {
        "linear" => SurvivalBaselineTarget::Linear,
        "weibull" => SurvivalBaselineTarget::Weibull,
        "gompertz" => SurvivalBaselineTarget::Gompertz,
        "gompertz-makeham" => SurvivalBaselineTarget::GompertzMakeham,
        other => {
            return Err(SurvivalConstructionError::UnsupportedDistribution {
                reason: format!(
                    "unsupported --baseline-target '{other}'; use linear|weibull|gompertz|gompertz-makeham"
                ),
            }
            .into());
        }
    };

    match target {
        SurvivalBaselineTarget::Linear => Ok(SurvivalBaselineConfig {
            target,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        }),
        SurvivalBaselineTarget::Weibull => {
            let scale = scale.ok_or_else(|| {
                "--baseline-target weibull requires --baseline-scale > 0".to_string()
            })?;
            let shape = shape.ok_or_else(|| {
                "--baseline-target weibull requires --baseline-shape > 0".to_string()
            })?;
            if !scale.is_finite() || scale <= 0.0 || !shape.is_finite() || shape <= 0.0 {
                return Err(
                    "weibull baseline requires finite positive --baseline-scale and --baseline-shape"
                        .to_string(),
                );
            }
            Ok(SurvivalBaselineConfig {
                target,
                scale: Some(scale),
                shape: Some(shape),
                rate: None,
                makeham: None,
            })
        }
        SurvivalBaselineTarget::Gompertz => {
            let rate = rate.unwrap_or(1.0);
            let shape = shape.unwrap_or(GOMPERTZ_DEFAULT_SHAPE_SEED);
            if !rate.is_finite() || rate <= 0.0 || !shape.is_finite() {
                return Err(
                    "gompertz baseline requires finite --baseline-shape and positive --baseline-rate"
                        .to_string(),
                );
            }
            Ok(SurvivalBaselineConfig {
                target,
                scale: None,
                shape: Some(shape),
                rate: Some(rate),
                makeham: None,
            })
        }
        SurvivalBaselineTarget::GompertzMakeham => {
            let rate = rate.unwrap_or(0.5);
            let shape = shape.unwrap_or(GOMPERTZ_DEFAULT_SHAPE_SEED);
            let makeham = makeham.unwrap_or(0.5);
            if !rate.is_finite()
                || rate <= 0.0
                || !shape.is_finite()
                || !makeham.is_finite()
                || makeham <= 0.0
            {
                return Err(
                    "gompertz-makeham baseline requires finite --baseline-shape, positive --baseline-rate, and positive --baseline-makeham"
                        .to_string(),
                );
            }
            Ok(SurvivalBaselineConfig {
                target,
                scale: None,
                shape: Some(shape),
                rate: Some(rate),
                makeham: Some(makeham),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Likelihood mode / distribution parsing
// ---------------------------------------------------------------------------

pub fn parse_survival_likelihood_mode(raw: &str) -> Result<SurvivalLikelihoodMode, String> {
    match raw.to_ascii_lowercase().as_str() {
        "transformation" => Ok(SurvivalLikelihoodMode::Transformation),
        "weibull" => Ok(SurvivalLikelihoodMode::Weibull),
        "location-scale" => Ok(SurvivalLikelihoodMode::LocationScale),
        "marginal-slope" => Ok(SurvivalLikelihoodMode::MarginalSlope),
        "latent" => Ok(SurvivalLikelihoodMode::Latent),
        "latent-binary" => Ok(SurvivalLikelihoodMode::LatentBinary),
        other => Err(SurvivalConstructionError::UnsupportedDistribution {
            reason: format!(
                "unsupported --survival-likelihood '{other}'; use transformation|weibull|location-scale|marginal-slope|latent|latent-binary"
            ),
        }
        .into()),
    }
}

pub const fn survival_likelihood_modename(mode: SurvivalLikelihoodMode) -> &'static str {
    match mode {
        SurvivalLikelihoodMode::Transformation => "transformation",
        SurvivalLikelihoodMode::Weibull => "weibull",
        SurvivalLikelihoodMode::LocationScale => "location-scale",
        SurvivalLikelihoodMode::MarginalSlope => "marginal-slope",
        SurvivalLikelihoodMode::Latent => "latent",
        SurvivalLikelihoodMode::LatentBinary => "latent-binary",
    }
}

pub fn parse_survival_distribution(raw: &str) -> Result<ResidualDistribution, String> {
    match raw.to_ascii_lowercase().as_str() {
        "gaussian" | "probit" => Ok(ResidualDistribution::Gaussian),
        "gumbel" | "cloglog" => Ok(ResidualDistribution::Gumbel),
        "logistic" | "logit" => Ok(ResidualDistribution::Logistic),
        other => Err(SurvivalConstructionError::UnsupportedDistribution {
            reason: format!(
                "unsupported survmodel(distribution='{other}'); accepted: gaussian / probit, gumbel / cloglog, logistic / logit"
            ),
        }
        .into()),
    }
}

pub const fn survival_baseline_targetname(target: SurvivalBaselineTarget) -> &'static str {
    match target {
        SurvivalBaselineTarget::Linear => "linear",
        SurvivalBaselineTarget::Weibull => "weibull",
        SurvivalBaselineTarget::Gompertz => "gompertz",
        SurvivalBaselineTarget::GompertzMakeham => "gompertz-makeham",
    }
}

pub fn positive_survival_time_seed(age_exit: &Array1<f64>) -> f64 {
    let sum = age_exit
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    let count = age_exit
        .iter()
        .filter(|value| value.is_finite() && **value > 0.0)
        .count()
        .max(1);
    (sum / count as f64).max(SURVIVAL_TIME_FLOOR)
}

pub fn initial_survival_baseline_config_for_fit(
    target_raw: &str,
    scale: Option<f64>,
    shape: Option<f64>,
    rate: Option<f64>,
    makeham: Option<f64>,
    age_exit: &Array1<f64>,
) -> Result<SurvivalBaselineConfig, String> {
    let target = match target_raw.trim().to_ascii_lowercase().as_str() {
        "linear" => SurvivalBaselineTarget::Linear,
        "weibull" => SurvivalBaselineTarget::Weibull,
        "gompertz" => SurvivalBaselineTarget::Gompertz,
        "gompertz-makeham" => SurvivalBaselineTarget::GompertzMakeham,
        other => {
            return Err(SurvivalConstructionError::UnsupportedDistribution {
                reason: format!(
                    "unsupported --baseline-target '{other}'; use linear|weibull|gompertz|gompertz-makeham"
                ),
            }
            .into());
        }
    };
    let time_scale_seed = positive_survival_time_seed(age_exit);
    let cfg = match target {
        SurvivalBaselineTarget::Linear => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        },
        SurvivalBaselineTarget::Weibull => SurvivalBaselineConfig {
            target,
            scale: Some(scale.unwrap_or(time_scale_seed)),
            shape: Some(shape.unwrap_or(1.0)),
            rate: None,
            makeham: None,
        },
        SurvivalBaselineTarget::Gompertz => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: Some(shape.unwrap_or(GOMPERTZ_DEFAULT_SHAPE_SEED)),
            rate: Some(rate.unwrap_or(1.0 / time_scale_seed)),
            makeham: None,
        },
        SurvivalBaselineTarget::GompertzMakeham => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: Some(shape.unwrap_or(GOMPERTZ_DEFAULT_SHAPE_SEED)),
            rate: Some(rate.unwrap_or(0.5 / time_scale_seed)),
            makeham: Some(makeham.unwrap_or(0.5 / time_scale_seed)),
        },
    };
    parse_survival_baseline_config(
        survival_baseline_targetname(cfg.target),
        cfg.scale,
        cfg.shape,
        cfg.rate,
        cfg.makeham,
    )
}

fn survival_baseline_theta_from_config(
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<Array1<f64>>, String> {
    Ok(match cfg.target {
        SurvivalBaselineTarget::Linear => None,
        SurvivalBaselineTarget::Weibull => Some(array![
            cfg.scale
                .ok_or_else(|| "missing weibull baseline scale".to_string())?
                .ln(),
            cfg.shape
                .ok_or_else(|| "missing weibull baseline shape".to_string())?
                .ln(),
        ]),
        SurvivalBaselineTarget::Gompertz => Some(array![
            cfg.rate
                .ok_or_else(|| "missing gompertz baseline rate".to_string())?
                .ln(),
            cfg.shape
                .ok_or_else(|| "missing gompertz baseline shape".to_string())?,
        ]),
        SurvivalBaselineTarget::GompertzMakeham => Some(array![
            cfg.rate
                .ok_or_else(|| "missing gompertz-makeham baseline rate".to_string())?
                .ln(),
            cfg.shape
                .ok_or_else(|| "missing gompertz-makeham baseline shape".to_string())?,
            cfg.makeham
                .ok_or_else(|| "missing gompertz-makeham baseline makeham".to_string())?
                .ln(),
        ]),
    })
}

fn survival_baseline_config_from_theta(
    target: SurvivalBaselineTarget,
    theta: &Array1<f64>,
) -> Result<SurvivalBaselineConfig, String> {
    let cfg = match target {
        SurvivalBaselineTarget::Linear => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        },
        SurvivalBaselineTarget::Weibull => {
            if theta.len() != 2 {
                return Err(SurvivalConstructionError::IncompatibleDimensions {
                    reason: format!(
                        "weibull baseline parameter dimension mismatch: expected 2, got {}",
                        theta.len()
                    ),
                }
                .into());
            }
            SurvivalBaselineConfig {
                target,
                scale: Some(theta[0].exp()),
                shape: Some(theta[1].exp()),
                rate: None,
                makeham: None,
            }
        }
        SurvivalBaselineTarget::Gompertz => {
            if theta.len() != 2 {
                return Err(SurvivalConstructionError::IncompatibleDimensions {
                    reason: format!(
                        "gompertz baseline parameter dimension mismatch: expected 2, got {}",
                        theta.len()
                    ),
                }
                .into());
            }
            SurvivalBaselineConfig {
                target,
                scale: None,
                shape: Some(theta[1]),
                rate: Some(theta[0].exp()),
                makeham: None,
            }
        }
        SurvivalBaselineTarget::GompertzMakeham => {
            if theta.len() != 3 {
                return Err(SurvivalConstructionError::IncompatibleDimensions {
                    reason: format!(
                        "gompertz-makeham baseline parameter dimension mismatch: expected 3, got {}",
                        theta.len()
                    ),
                }
                .into());
            }
            SurvivalBaselineConfig {
                target,
                scale: None,
                shape: Some(theta[1]),
                rate: Some(theta[0].exp()),
                makeham: Some(theta[2].exp()),
            }
        }
    };
    parse_survival_baseline_config(
        survival_baseline_targetname(cfg.target),
        cfg.scale,
        cfg.shape,
        cfg.rate,
        cfg.makeham,
    )
}

/// Derivative contract for the shared baseline-θ outer optimizer.
///
/// The two public baseline optimizers (`…_with_gradient_only`,
/// `…_with_gradient`) differ in exactly one axis: how much derivative
/// information the objective closure supplies, and therefore which curvature
/// declaration the `OuterProblem` must advertise. Every baseline-θ path now
/// supplies an exact analytic gradient (profile-NLL envelope gradient), so both
/// contracts route to a gradient-based solver. Everything else — θ↔config
/// conversion, the ±6 log-space box,
/// the single-seed config, the `run`/convergence/error-formatting boilerplate
/// — is identical, so it lives once in [`run_baseline_theta_optimizer`] and
/// this enum selects the per-contract `OuterProblem` configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BaselineDerivativeContract {
    /// Cost + analytic gradient, no analytic Hessian. Routes to BFGS, which
    /// builds its own quasi-Newton curvature from successive gradients.
    GradientOnly,
    /// Cost + analytic gradient + analytic Hessian. Routes to the primary
    /// second-order outer solver, which may use either the analytic Hessian or
    /// a BFGS approximation depending on the planner.
    GradientHessian,
}

impl BaselineDerivativeContract {
    /// Apply this contract's derivative declaration, solver class, tolerance,
    /// and iteration budget to a freshly-constructed `OuterProblem`. The
    /// bounds, initial ρ, and seed config are contract-independent and applied
    /// by [`run_baseline_theta_optimizer`].
    fn configure(
        self,
        problem: crate::solver::rho_optimizer::OuterProblem,
    ) -> crate::solver::rho_optimizer::OuterProblem {
        use crate::solver::rho_optimizer::{DeclaredHessianForm, Derivative};
        match self {
            // BFGS on a 2–3 dim problem with an exact gradient typically
            // converges in 5–10 outer evaluations.
            BaselineDerivativeContract::GradientOnly => problem
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Unavailable)
                .with_tolerance(1e-4)
                .with_max_iter(240),
            BaselineDerivativeContract::GradientHessian => problem
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Either)
                .with_tolerance(1e-4)
                .with_max_iter(240),
        }
    }
}

/// Shared engine behind the three public baseline-config optimizers.
///
/// Owns every step that is identical across the cost-only, gradient-only, and
/// gradient+Hessian contracts: config→θ seeding (with the linear/no-parameter
/// early return), the ±6 log-space box, the single-seed `OuterProblem`
/// skeleton, derivative-contract configuration, `build_objective` wiring,
/// `run`, the convergence check + error formatting, and θ→config. The only
/// contract-specific inputs are the already-wired `cost_fn`/`eval_fn` closures
/// (which embed the derivative shape and dimension validation) and the
/// `contract` selecting the `OuterProblem` derivative declaration.
fn run_baseline_theta_optimizer<Fc, Fe>(
    initial: &SurvivalBaselineConfig,
    context: &str,
    contract: BaselineDerivativeContract,
    cost_fn: Fc,
    eval_fn: Fe,
) -> Result<SurvivalBaselineConfig, String>
where
    Fc: FnMut(&mut (), &Array1<f64>) -> Result<f64, crate::model_types::EstimationError>,
    Fe: FnMut(
        &mut (),
        &Array1<f64>,
    ) -> Result<
        crate::solver::rho_optimizer::OuterEval,
        crate::model_types::EstimationError,
    >,
{
    use crate::solver::rho_optimizer::OuterProblem;
    let Some(seed) = survival_baseline_theta_from_config(initial)? else {
        return Ok(initial.clone());
    };
    let dim = seed.len();
    let target = initial.target;
    let lower = seed.mapv(|v| v - 6.0);
    let upper = seed.mapv(|v| v + 6.0);
    let problem = contract
        .configure(OuterProblem::new(dim))
        .with_bounds(lower, upper)
        .with_initial_rho(seed.clone())
        .with_seed_config(crate::seeding::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            num_auxiliary_trailing: dim,
            ..Default::default()
        });
    let mut obj = problem.build_objective(
        (),
        cost_fn,
        eval_fn,
        None::<fn(&mut ())>,
        None::<
            fn(
                &mut (),
                &Array1<f64>,
            ) -> Result<
                crate::solver::rho_optimizer::EfsEval,
                crate::model_types::EstimationError,
            >,
        >,
    );
    let result = problem
        .run(&mut obj, context)
        .map_err(|e| format!("{context} failed: {e}"))?;
    if !result.converged {
        return Err(SurvivalConstructionError::InvalidConfig {
            reason: format!(
                "{context} did not converge after {} iterations (final_objective={:.6e}, final_grad_norm={})",
                result.iterations,
                result.final_value,
                result.final_grad_norm_report(),
            ),
        }
        .into());
    }
    survival_baseline_config_from_theta(target, &result.rho)
}

/// Shared engine for the two derivative-carrying baseline-config optimizers.
///
/// Both `…_with_gradient_only` and `…_with_gradient` route an objective that
/// returns a fully-populated [`OuterEval`](crate::solver::rho_optimizer::OuterEval)
/// (cost + analytic gradient, optionally + analytic Hessian) for a given
/// config. Everything downstream of that — the `Rc<RefCell>` sharing that lets
/// the same user closure back both the `cost_fn` and `eval_fn`, the θ→config
/// conversion, and deriving the scalar `cost_fn` from the eval result — is
/// identical, so it lives here once. The contract-specific axis is only which
/// `HessianResult` the objective embeds, which the wrapper has already encoded
/// in the returned `OuterEval`, so this helper is contract-agnostic beyond the
/// `contract` it forwards to [`run_baseline_theta_optimizer`].
fn run_baseline_theta_optimizer_with_eval<F>(
    initial: &SurvivalBaselineConfig,
    context: &str,
    contract: BaselineDerivativeContract,
    objective: F,
) -> Result<SurvivalBaselineConfig, String>
where
    F: FnMut(&SurvivalBaselineConfig) -> Result<crate::solver::rho_optimizer::OuterEval, String>,
{
    let target = initial.target;
    let engine_context = context.to_string();
    let objective = std::rc::Rc::new(std::cell::RefCell::new(objective));
    let eval_at = move |obj: &std::rc::Rc<std::cell::RefCell<F>>,
                        theta: &Array1<f64>|
          -> Result<
        crate::solver::rho_optimizer::OuterEval,
        crate::model_types::EstimationError,
    > {
        let cfg = survival_baseline_config_from_theta(target, theta)
            .map_err(crate::model_types::EstimationError::InvalidInput)?;
        let eval =
            obj.borrow_mut()(&cfg).map_err(crate::model_types::EstimationError::InvalidInput)?;
        if eval.gradient.len() != theta.len() {
            return Err(crate::model_types::EstimationError::InvalidInput(format!(
                "{engine_context}: baseline gradient dimension mismatch: got {}, expected {}",
                eval.gradient.len(),
                theta.len()
            )));
        }
        if let crate::solver::rho_optimizer::HessianResult::Analytic(ref h) = eval.hessian {
            if h.nrows() != theta.len() || h.ncols() != theta.len() {
                return Err(crate::model_types::EstimationError::InvalidInput(format!(
                    "{engine_context}: baseline Hessian dimension mismatch: got {}x{}, expected {}x{}",
                    h.nrows(),
                    h.ncols(),
                    theta.len(),
                    theta.len()
                )));
            }
        }
        Ok(eval)
    };
    let cost_objective = std::rc::Rc::clone(&objective);
    let cost_eval = eval_at.clone();
    let cost_fn = move |_: &mut (), theta: &Array1<f64>| {
        cost_eval(&cost_objective, theta).map(|eval| eval.cost)
    };
    let eval_fn = move |_: &mut (), theta: &Array1<f64>| eval_at(&objective, theta);
    run_baseline_theta_optimizer(initial, context, contract, cost_fn, eval_fn)
}

/// Gradient-only outer baseline-config optimizer. Thin adapter over
/// [`run_baseline_theta_optimizer`] under the
/// [`BaselineDerivativeContract::GradientOnly`] contract, which advertises
/// `DeclaredHessianForm::Unavailable`, so the planner routes to BFGS and
/// builds its own quasi-Newton curvature from successive gradient
/// evaluations. Used by the survival location-scale path which has a
/// closed-form θ-gradient (`baseline_chain_rule_gradient` /
/// `marginal_slope_baseline_chain_rule_gradient`) but no native analytic
/// θ-Hessian; BFGS on a 2–3 dim problem with an exact gradient typically
/// converges in 5–10 outer evaluations.
pub fn optimize_survival_baseline_config_with_gradient_only<F>(
    initial: &SurvivalBaselineConfig,
    context: &str,
    mut objective: F,
) -> Result<SurvivalBaselineConfig, String>
where
    F: FnMut(&SurvivalBaselineConfig) -> Result<(f64, Array1<f64>), String>,
{
    use crate::solver::rho_optimizer::{HessianResult, OuterEval};
    run_baseline_theta_optimizer_with_eval(
        initial,
        context,
        BaselineDerivativeContract::GradientOnly,
        move |cfg| {
            let (cost, gradient) = objective(cfg)?;
            Ok(OuterEval {
                cost,
                gradient,
                hessian: HessianResult::Unavailable,
                inner_beta_hint: None,
            })
        },
    )
}

/// Gradient + Hessian outer baseline-config optimizer. Thin adapter over
/// [`run_baseline_theta_optimizer`] under the
/// [`BaselineDerivativeContract::GradientHessian`] contract, which advertises
/// an analytic θ-Hessian so the primary second-order outer solver can use it.
pub fn optimize_survival_baseline_config_with_gradient<F>(
    initial: &SurvivalBaselineConfig,
    context: &str,
    mut objective: F,
) -> Result<SurvivalBaselineConfig, String>
where
    F: FnMut(&SurvivalBaselineConfig) -> Result<(f64, Array1<f64>, Array2<f64>), String>,
{
    use crate::solver::rho_optimizer::{HessianResult, OuterEval};
    run_baseline_theta_optimizer_with_eval(
        initial,
        context,
        BaselineDerivativeContract::GradientHessian,
        move |cfg| {
            let (cost, gradient, hessian) = objective(cfg)?;
            Ok(OuterEval {
                cost,
                gradient,
                hessian: HessianResult::Analytic(hessian),
                inner_beta_hint: None,
            })
        },
    )
}

// ---------------------------------------------------------------------------
// Time basis config (library-friendly: takes primitives, not CLI args)
// ---------------------------------------------------------------------------

pub fn parse_survival_time_basis_config(
    time_basis: &str,
    time_degree: usize,
    time_num_internal_knots: usize,
    time_smooth_lambda: f64,
) -> Result<SurvivalTimeBasisConfig, String> {
    match time_basis.to_ascii_lowercase().as_str() {
        "none" => Ok(SurvivalTimeBasisConfig::None),
        "ispline" => {
            if time_degree < 1 {
                return Err(
                    "time-basis degree must be >= 1 for ispline time basis (CLI: --time-degree; Python: time_degree=)"
                        .to_string(),
                );
            }
            if time_num_internal_knots == 0 {
                return Err(
                    "time-basis must have > 0 internal knots for ispline time basis (CLI: --time-num-internal-knots; Python: time_num_internal_knots=)"
                        .to_string(),
                );
            }
            if !time_smooth_lambda.is_finite() || time_smooth_lambda < 0.0 {
                return Err(
                    "time-basis smoothing lambda must be finite and >= 0 (CLI: --time-smooth-lambda; Python: time_smooth_lambda=)"
                        .to_string(),
                );
            }
            Ok(SurvivalTimeBasisConfig::ISpline {
                degree: time_degree,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: time_smooth_lambda,
            })
        }
        "linear" | "bspline" => {
            // Forward to the shared structural-basis check so error text
            // stays consistent with every other call site. `linear` /
            // `bspline` are not structural, so this always returns Err;
            // we map a (currently impossible) `Ok` to an explicit error
            // string instead of `unreachable!`, keeping the match total
            // without relying on a never-executes claim.
            match require_structural_survival_time_basis(time_basis, "survival model configuration")
            {
                Err(e) => Err(e),
                Ok(()) => Err(format!(
                    "internal: structural-basis check accepted non-structural \
                     survival time basis '{time_basis}'"
                )),
            }
        }
        other => Err(format!(
            "unsupported --time-basis '{other}'; accepted values: ispline, none"
        )),
    }
}

// ---------------------------------------------------------------------------
// Time basis construction
// ---------------------------------------------------------------------------

pub fn build_survival_time_basis(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: SurvivalTimeBasisConfig,
    infer_knots_if_needed: Option<(usize, f64)>,
) -> Result<SurvivalTimeBuildOutput, String> {
    fn checked_log_survival_times(times: &Array1<f64>, label: &str) -> Result<Array1<f64>, String> {
        if let Some(row) = times.iter().position(|t| !t.is_finite()) {
            return Err(SurvivalConstructionError::DataValidationFailed {
                reason: format!(
                    "survival time basis requires finite {label} times (row {})",
                    row + 1
                ),
            }
            .into());
        }
        if let Some(row) = times.iter().position(|t| *t < 0.0) {
            return Err(SurvivalConstructionError::DataValidationFailed {
                reason: format!(
                    "survival time basis requires non-negative {label} times (row {})",
                    row + 1
                ),
            }
            .into());
        }
        Ok(times.mapv(|t| t.max(SURVIVAL_TIME_FLOOR).ln()))
    }

    let n = age_entry.len();
    if n != age_exit.len() {
        return Err(SurvivalConstructionError::IncompatibleDimensions {
            reason: "survival time basis requires matching entry/exit lengths".to_string(),
        }
        .into());
    }
    for i in 0..n {
        if age_exit[i] < age_entry[i] {
            return Err(format!(
                "survival time basis requires exit times >= entry times (row {})",
                i + 1
            ));
        }
    }
    let log_entry = checked_log_survival_times(age_entry, "entry")?;
    let log_exit = checked_log_survival_times(age_exit, "exit")?;

    fn survival_time_knot_input(log_entry: &Array1<f64>, log_exit: &Array1<f64>) -> Array1<f64> {
        let n = log_entry.len();
        let entry_range = log_entry
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let entry_degenerate = (entry_range.1 - entry_range.0).abs() < 1e-8;
        if entry_degenerate {
            log_exit.clone()
        } else {
            let mut combined = Array1::<f64>::zeros(2 * n);
            for i in 0..n {
                combined[i] = log_entry[i];
                combined[n + i] = log_exit[i];
            }
            combined
        }
    }

    /// Cap the requested monotone-baseline internal-knot count to what the
    /// observed time resolution can actually support.
    ///
    /// The survival location-scale baseline is a degree-`d` I-spline with
    /// `num_internal_knots + d` shape-varying columns. Its smoothing parameter
    /// is informed *only* by the distinct interior log-time points: with fewer
    /// distinct interior times than requested knots the baseline is
    /// rank-deficient, and the REML/LAML profile in the time smoothing
    /// parameter becomes a flat ridge — the exact-joint outer search then
    /// probes that ridge indefinitely (each inner constrained Newton burns its
    /// whole cycle budget without certifying convergence) and the fit never
    /// terminates. This is the survival analogue of the standard
    /// "df must not exceed the data resolution" guard (`mgcv` caps `k` at the
    /// number of unique covariate values; `flexsurv`/`rstpm2` use a handful of
    /// baseline knots): we never place more interior knots than there are
    /// distinct interior points, and we keep the total baseline dimension a
    /// bounded fraction of the sample so the smoothing profile stays curved.
    ///
    /// This clamp lives in the shared knot-inference routine so the fit and any
    /// independent rebuild of the time basis (e.g. a predictor reconstructing
    /// `design · β` at fresh covariates) resolve to the *same* knot vector from
    /// the same data — there is no raw/active dimension drift.
    fn data_capped_internal_knots(
        combined: &Array1<f64>,
        degree: usize,
        requested_internal_knots: usize,
    ) -> usize {
        if requested_internal_knots == 0 {
            return 0;
        }
        let mut sorted: Vec<f64> = combined.iter().copied().collect();
        sorted.sort_by(f64::total_cmp);
        let minval = sorted.first().copied().unwrap_or(0.0);
        let maxval = sorted.last().copied().unwrap_or(minval);
        if minval == maxval {
            // Degenerate (single distinct time): no interior structure to fit.
            return 1.min(requested_internal_knots);
        }
        let scale = (maxval - minval).abs().max(1.0);
        let tol = 1e-12 * scale;
        // Count distinct strictly-interior points (knots can only live strictly
        // between the data extremes).
        let mut distinct_interior = 0usize;
        let mut last: Option<f64> = None;
        for &x in &sorted {
            if x <= minval + tol || x >= maxval - tol {
                continue;
            }
            if last.is_some_and(|prev| (x - prev).abs() <= tol) {
                continue;
            }
            distinct_interior += 1;
            last = Some(x);
        }
        // Distinct-point ceiling: cannot place more interior knots than there
        // are distinct interior values.
        let mut cap = requested_internal_knots.min(distinct_interior.max(1));
        // Dimension-vs-resolution ceiling: keep the total baseline column count
        // `cap + degree` below ~1/4 of the distinct sample points so the
        // smoothing-parameter profile retains curvature (the data must be able
        // to identify the baseline shape, not just interpolate it). `n_distinct`
        // counts all distinct points (interior + the two extremes).
        let n_distinct = {
            let mut count = 0usize;
            let mut last: Option<f64> = None;
            for &x in &sorted {
                if last.is_some_and(|prev| (x - prev).abs() <= tol) {
                    continue;
                }
                count += 1;
                last = Some(x);
            }
            count
        };
        let dim_budget = n_distinct / 4;
        let dim_cap = dim_budget.saturating_sub(degree);
        cap = cap.min(dim_cap.max(1));
        cap.max(1)
    }

    fn infer_survival_time_knots(
        combined: &Array1<f64>,
        knot_degree: usize,
        validation_degree: usize,
        num_internal_knots: usize,
        basis_options: BasisOptions,
    ) -> Result<Array1<f64>, String> {
        // Identifiability/termination guard: never request more baseline
        // internal knots than the observed time resolution supports. See
        // `data_capped_internal_knots` for the full rationale (a flat smoothing
        // ridge on an over-parameterized baseline is what makes the survival
        // location-scale exact-joint outer search fail to terminate).
        let num_internal_knots =
            data_capped_internal_knots(combined, validation_degree, num_internal_knots);

        fn quantile_knot_inference_needs_uniform_fallback(
            combined: &Array1<f64>,
            num_internal_knots: usize,
        ) -> bool {
            if num_internal_knots == 0 || combined.is_empty() {
                return false;
            }

            let mut sorted: Vec<f64> = combined.iter().copied().collect();
            sorted.sort_by(f64::total_cmp);
            let minval = sorted[0];
            let maxval = *sorted.last().unwrap_or(&minval);
            if minval == maxval {
                return false;
            }

            let scale = (maxval - minval).abs().max(1.0);
            let tol = 1e-12 * scale;
            let mut support = Vec::with_capacity(sorted.len());
            let mut last: Option<f64> = None;
            for &x in &sorted {
                if x <= minval + tol || x >= maxval - tol {
                    continue;
                }
                if last.map(|prev| (x - prev).abs() <= tol).unwrap_or(false) {
                    continue;
                }
                support.push(x);
                last = Some(x);
            }
            if support.is_empty() {
                return true;
            }

            let n = support.len();
            let mut prev_q = minval;
            for j in 1..=num_internal_knots {
                let p = j as f64 / (num_internal_knots + 1) as f64;
                let pos = p * (n.saturating_sub(1) as f64);
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let frac = pos - lo as f64;
                let q = if lo == hi {
                    support[lo]
                } else {
                    support[lo] * (1.0 - frac) + support[hi] * frac
                }
                .clamp(minval, maxval);
                if q <= prev_q + tol || q >= maxval - tol {
                    return true;
                }
                prev_q = q;
            }

            false
        }

        let inferwith =
            |placement: crate::basis::BSplineKnotPlacement| -> Result<Array1<f64>, String> {
                let built = build_bspline_basis_1d(
                    combined.view(),
                    &BSplineBasisSpec {
                        degree: knot_degree,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Automatic {
                            num_internal_knots: Some(num_internal_knots),
                            placement,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::None,
                        boundary: OneDimensionalBoundary::Open,
                        boundary_conditions: BSplineBoundaryConditions::default(),
                    },
                )
                .map_err(|e| format!("failed to infer survival time knots: {e}"))?;
                let knots = match built.metadata {
                    BasisMetadata::BSpline1D { knots, .. } => knots,
                    _ => {
                        return Err(
                            "internal error: expected BSpline1D metadata for survival time basis"
                                .to_string(),
                        );
                    }
                };
                // `knot_degree` is the clamped B-spline degree used to size
                // the knot vector. `validation_degree` is the public basis
                // degree passed to the final evaluator. They differ for
                // I-splines because `create_basis(..., BasisOptions::i_spline())`
                // internally raises the public degree by one to its working
                // B-spline antiderivative degree. Validating with
                // `knot_degree` here would raise a second time and reject the
                // coherent knot vector we just inferred.
                create_basis::<Dense>(
                    combined.view(),
                    KnotSource::Provided(knots.view()),
                    validation_degree,
                    basis_options,
                )
                .map_err(|e| e.to_string())?;
                Ok(knots)
            };

        if quantile_knot_inference_needs_uniform_fallback(combined, num_internal_knots) {
            inferwith(crate::basis::BSplineKnotPlacement::Uniform)
        } else {
            inferwith(crate::basis::BSplineKnotPlacement::Quantile)
        }
    }

    match cfg {
        SurvivalTimeBasisConfig::None => Ok(SurvivalTimeBuildOutput {
            x_entry_time: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((n, 0)))),
            x_exit_time: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((n, 0)))),
            x_derivative_time: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((n, 0)))),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            basisname: "none".to_string(),
            degree: None,
            knots: None,
            keep_cols: None,
            smooth_lambda: None,
        }),
        SurvivalTimeBasisConfig::Linear => {
            let mut x_entry_time = Array2::<f64>::zeros((n, 2));
            let mut x_exit_time = Array2::<f64>::zeros((n, 2));
            let mut x_derivative_time = Array2::<f64>::zeros((n, 2));
            for i in 0..n {
                x_entry_time[[i, 0]] = 1.0;
                x_exit_time[[i, 0]] = 1.0;
                x_entry_time[[i, 1]] = log_entry[i];
                x_exit_time[[i, 1]] = log_exit[i];
                x_derivative_time[[i, 1]] = 1.0 / age_exit[i].max(SURVIVAL_TIME_FLOOR);
            }
            Ok(SurvivalTimeBuildOutput {
                x_entry_time: DesignMatrix::Dense(DenseDesignMatrix::from(x_entry_time)),
                x_exit_time: DesignMatrix::Dense(DenseDesignMatrix::from(x_exit_time)),
                x_derivative_time: DesignMatrix::Dense(DenseDesignMatrix::from(x_derivative_time)),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                basisname: "linear".to_string(),
                degree: None,
                knots: None,
                keep_cols: None,
                smooth_lambda: None,
            })
        }
        SurvivalTimeBasisConfig::BSpline {
            degree,
            knots,
            smooth_lambda,
        } => {
            let knotvec = if knots.is_empty() {
                let (num_internal_knots, _) = infer_knots_if_needed.ok_or_else(|| {
                    "internal error: bspline time basis requested without knot source".to_string()
                })?;
                let combined = survival_time_knot_input(&log_entry, &log_exit);
                infer_survival_time_knots(
                    &combined,
                    degree,
                    degree,
                    num_internal_knots,
                    BasisOptions::value(),
                )?
            } else {
                knots
            };

            let entry_basis = build_bspline_basis_1d(
                log_entry.view(),
                &BSplineBasisSpec {
                    degree,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Provided(knotvec.clone()),
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            )
            .map_err(|e| format!("failed to build bspline entry basis: {e}"))?;
            let exit_basis = build_bspline_basis_1d(
                log_exit.view(),
                &BSplineBasisSpec {
                    degree,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Provided(knotvec.clone()),
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            )
            .map_err(|e| format!("failed to build bspline exit basis: {e}"))?;

            let p_time = exit_basis.design.ncols();
            // Build derivative basis as sparse triplets — B-spline derivatives
            // have the same local support as the basis itself (at most degree+1
            // nonzeros per row), so building dense first wastes memory.
            let mut deriv_triplets = Vec::with_capacity(n * (degree + 1));
            let mut deriv_buf = vec![0.0_f64; p_time];
            for i in 0..n {
                deriv_buf.fill(0.0);
                evaluate_bspline_derivative_scalar(
                    log_exit[i],
                    knotvec.view(),
                    degree,
                    &mut deriv_buf,
                )
                .map_err(|e| format!("failed to evaluate bspline derivative: {e}"))?;
                let chain = 1.0 / age_exit[i].max(SURVIVAL_TIME_FLOOR);
                for j in 0..p_time {
                    let v = deriv_buf[j] * chain;
                    if v.abs() > 1e-15 {
                        deriv_triplets.push(faer::sparse::Triplet::new(i, j, v));
                    }
                }
            }
            let x_derivative_time =
                match faer::sparse::SparseColMat::try_new_from_triplets(n, p_time, &deriv_triplets)
                {
                    Ok(sparse) => DesignMatrix::Sparse(SparseDesignMatrix::new(sparse)),
                    Err(_) => {
                        // Fallback: build dense
                        let mut dense = Array2::<f64>::zeros((n, p_time));
                        for &faer::sparse::Triplet { row, col, val } in &deriv_triplets {
                            dense[[row, col]] = val;
                        }
                        DesignMatrix::Dense(DenseDesignMatrix::from(dense))
                    }
                };

            Ok(SurvivalTimeBuildOutput {
                x_entry_time: entry_basis.design,
                x_exit_time: exit_basis.design,
                x_derivative_time,
                nullspace_dims: entry_basis.nullspace_dims,
                penalties: entry_basis.penalties,
                basisname: "bspline".to_string(),
                degree: Some(degree),
                knots: Some(knotvec.to_vec()),
                keep_cols: None,
                smooth_lambda: Some(smooth_lambda),
            })
        }
        SurvivalTimeBasisConfig::ISpline {
            degree,
            knots,
            keep_cols,
            smooth_lambda,
        } => {
            let bspline_degree = degree
                .checked_add(1)
                .ok_or_else(|| "ispline degree overflow while building knot basis".to_string())?;
            let knotvec = if knots.is_empty() {
                let (num_internal_knots, _) = infer_knots_if_needed.ok_or_else(|| {
                    "internal error: ispline time basis requested without knot source".to_string()
                })?;
                let combined = survival_time_knot_input(&log_entry, &log_exit);
                infer_survival_time_knots(
                    &combined,
                    bspline_degree,
                    degree,
                    num_internal_knots,
                    BasisOptions::i_spline(),
                )?
            } else {
                knots
            };

            let (db_exit_arc, _) = create_basis::<Dense>(
                log_exit.view(),
                KnotSource::Provided(knotvec.view()),
                bspline_degree,
                BasisOptions::first_derivative(),
            )
            .map_err(|e| format!("failed to build ispline derivative basis: {e}"))?;

            // Build full-width I-spline bases inside a block scope so the
            // large Arc allocations are freed when the block ends.
            let (x_entry_time, x_exit_time, keep_cols, p_time, p_time_full) = {
                let (entry_arc, _) = create_basis::<Dense>(
                    log_entry.view(),
                    KnotSource::Provided(knotvec.view()),
                    degree,
                    BasisOptions::i_spline(),
                )
                .map_err(|e| format!("failed to build ispline entry basis: {e}"))?;
                let (exit_arc, _) = create_basis::<Dense>(
                    log_exit.view(),
                    KnotSource::Provided(knotvec.view()),
                    degree,
                    BasisOptions::i_spline(),
                )
                .map_err(|e| format!("failed to build ispline exit basis: {e}"))?;

                let x_entry_full = entry_arc.as_ref();
                let x_exit_full = exit_arc.as_ref();
                let p_time_full = x_exit_full.ncols();
                if p_time_full == 0 {
                    return Err(SurvivalConstructionError::BasisConstructionFailed {
                        reason: "internal error: empty ispline time basis".to_string(),
                    }
                    .into());
                }
                let db_exit = db_exit_arc.as_ref();
                if db_exit.ncols() != p_time_full + 1 {
                    return Err(
                        "internal error: ispline derivative basis width must exceed basis width by one"
                            .to_string(),
                    );
                }

                let keep_cols = if keep_cols.is_empty() {
                    let constant_tol = 1e-12_f64;
                    let mut inferred_keep_cols: Vec<usize> = Vec::new();
                    for j in 0..p_time_full {
                        let mut minv = f64::INFINITY;
                        let mut maxv = f64::NEG_INFINITY;
                        for i in 0..n {
                            let ve = x_exit_full[[i, j]];
                            let vs = x_entry_full[[i, j]];
                            minv = minv.min(ve.min(vs));
                            maxv = maxv.max(ve.max(vs));
                        }
                        if (maxv - minv) > constant_tol {
                            inferred_keep_cols.push(j);
                        }
                    }
                    inferred_keep_cols
                } else {
                    keep_cols
                };
                if keep_cols.is_empty() {
                    return Err(
                        "internal error: ispline basis has no shape-varying time columns"
                            .to_string(),
                    );
                }
                if keep_cols.iter().any(|&j| j >= p_time_full) {
                    return Err(SurvivalConstructionError::MissingColumn {
                        reason: "saved survival ispline keep_cols exceed basis width".to_string(),
                    }
                    .into());
                }

                let p_time = keep_cols.len();
                let x_entry_time = x_entry_full.select(ndarray::Axis(1), &keep_cols);
                let x_exit_time = x_exit_full.select(ndarray::Axis(1), &keep_cols);
                // entry_arc and exit_arc go out of scope here, freeing the
                // full-width bases before derivative computation below.
                (x_entry_time, x_exit_time, keep_cols, p_time, p_time_full)
            };
            let db_exit = db_exit_arc.as_ref();

            // Build I-spline derivative as sparse triplets.  The derivative
            // is a cumulative sum of B-spline derivatives and typically has
            // more nonzeros per row than a plain B-spline, but still much
            // fewer than p_time for modest bases.
            let mut deriv_triplets = Vec::with_capacity(n * p_time.min(16));
            let mut found_nonfinite: Option<(usize, usize)> = None;
            for i in 0..n {
                let mut running = 0.0_f64;
                let mut d_i_log_full = vec![0.0_f64; p_time_full];
                for j in (1..db_exit.ncols()).rev() {
                    let term = db_exit[[i, j]];
                    if term.is_finite() {
                        running += term;
                    }
                    d_i_log_full[j - 1] = running;
                }
                let chain = 1.0 / age_exit[i].max(SURVIVAL_TIME_FLOOR);
                for (j_new, &j_old) in keep_cols.iter().enumerate() {
                    let raw_v = d_i_log_full[j_old] * chain;
                    let v = if (-1e-12..0.0).contains(&raw_v) {
                        0.0
                    } else {
                        raw_v
                    };
                    if !v.is_finite() {
                        found_nonfinite = Some((i, j_new));
                    }
                    if v < -1e-12 {
                        return Err(format!(
                            "survival ispline derivative basis must stay non-negative at row {}, column {}; found {:.3e}",
                            i + 1,
                            j_new + 1,
                            v
                        ));
                    }
                    if v.abs() > 1e-15 {
                        deriv_triplets.push(faer::sparse::Triplet::new(i, j_new, v));
                    }
                }
            }
            if let Some((row, col)) = found_nonfinite {
                return Err(format!(
                    "survival ispline derivative basis produced non-finite value at row {}, column {}",
                    row + 1,
                    col + 1
                ));
            }
            let x_derivative_time =
                match faer::sparse::SparseColMat::try_new_from_triplets(n, p_time, &deriv_triplets)
                {
                    Ok(sparse) => DesignMatrix::Sparse(SparseDesignMatrix::new(sparse)),
                    Err(_) => {
                        let mut dense = Array2::<f64>::zeros((n, p_time));
                        for &faer::sparse::Triplet { row, col, val } in &deriv_triplets {
                            dense[[row, col]] = val;
                        }
                        DesignMatrix::Dense(DenseDesignMatrix::from(dense))
                    }
                };

            let penalty_basis = build_bspline_basis_1d(
                log_exit.view(),
                &BSplineBasisSpec {
                    degree: bspline_degree,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Provided(knotvec.clone()),
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            )
            .map_err(|e| format!("failed to build ispline smoothing penalty: {e}"))?;
            if penalty_basis.design.ncols() != p_time_full + 1 {
                return Err("internal error: ispline penalty dimension mismatch".to_string());
            }
            // I-spline curvature penalty in the *value* space of the baseline
            // log-cumulative-hazard, restricted to the retained (non-dropped)
            // coefficient block.
            //
            // The I-spline coefficient γ is the consecutive increment of the B-spline
            // value coefficients `c`: `c_0 = 0`, `c_k = Σ_{j<k} γ_j = (L γ)_k`, where
            // `L` is the `p_time × p_time` lower-triangular cumsum matrix. The
            // second-difference penalty on the B-spline values is `S_B = D₂ᵀD₂`
            // (the `penalty_basis.penalties` block). The correct curvature penalty
            // on γ is the **value-space congruence transform**
            //
            //   `S_I = Lᵀ S_B[1:,1:] L`,
            //
            // which satisfies `γᵀ S_I γ = (Lγ)ᵀ S_B[1:,1:] (Lγ)`.
            //
            // A constant γ (γ_k = γ₀ ∀k) maps to the linear value sequence
            // `c_k = k·γ₀`, which is annihilated by D₂: `D₂c = 0`. Therefore
            // `γᵀ S_I γ = 0` for constant γ, i.e. the **affine trend lies in the
            // penalty null space**. REML does not penalize the baseline slope
            // `d(log Λ)/d(log t)` or the overall level, so it correctly lets the
            // data determine these quantities without bias. The previous increment-
            // space form `S_B[1:,1:]` (applied directly to γ instead of Lγ) did NOT
            // have constant γ in its null space and therefore over-penalized affine
            // baselines, causing the fitted log-cumulative-hazard to lose its tail
            // slope to the penalty and fail quality tests (#1076).
            //
            // The value-space form has a 1-dimensional null space (span{(1,…,1)}),
            // declared via `nullspace_dims` so the REML generalized-logdet picks it
            // up. The penalized inner PIRLS is well-conditioned because the
            // likelihood Hessian H_lik has O(n_events) curvature along the affine
            // direction (the overall baseline level is identified by the data), and
            // the global stabilization ridge (ridge_lambda) provides an absolute
            // positive-definite floor.
            let mut penalties = Vec::<Array2<f64>>::new();
            for s_mat in &penalty_basis.penalties {
                if s_mat.nrows() != p_time_full + 1 || s_mat.ncols() != p_time_full + 1 {
                    continue;
                }
                // I-spline value-space penalty, computed in the CORRECT order
                // (gam#979). The B-spline value coefficients are the cumulative
                // sum of the I-spline increment coefficients, `c = L γ_full`, where
                // `L` is the FULL `p_time_full × p_time_full` LOWER-triangular
                // all-ones cumsum matrix (`L[i,j] = 1 iff j ≤ i`, so
                // `c_i = Σ_{j≤i} γ_j`). The value-space curvature penalty on the
                // full increment vector is the symmetric congruence
                //
                //   `S_I_full = Lᵀ · S_B[1:,1:] · L`,
                //
                // which is PSD because `S_B[1:,1:]` is a principal submatrix of the
                // PSD `S_B = D₂ᵀD₂` and congruence by any matrix preserves PSD.
                //
                // CRITICAL ORDERING (the gam#979 indefiniteness bug): the retained
                // columns `keep_cols` must be selected as a PRINCIPAL SUBMATRIX of
                // the FULL congruence `S_I_full` — i.e. congruence FIRST, selection
                // SECOND. The previous code selected `keep_cols` from `S_B[1:,1:]`
                // first and then applied a `p_time × p_time` cumsum to that
                // already-reduced block. Because the cumsum `L` couples every
                // increment, restricting the increment index set BEFORE the cumsum
                // does NOT commute with it: the reduced operator is a different,
                // generally INDEFINITE matrix (measured `s0_min_eval = −9.8e7`),
                // which makes `½γᵀS_Iγ` unbounded below and the penalized survival
                // NLL diverge (β drifts up the negative-eigenvalue mode, the inner
                // joint-Newton follows the unbounded objective, the outer REML never
                // terminates — the #979 hang). Doing the congruence on the full γ
                // and then taking the `keep_cols` principal submatrix restores the
                // PSD guarantee (a principal submatrix of a PSD matrix is PSD).
                let s_increment = s_mat.slice(s![1.., 1..]);
                if s_increment.nrows() != p_time_full || s_increment.ncols() != p_time_full {
                    return Err(format!(
                        "internal error: ispline penalty increment block must be {p_time_full}x{p_time_full}, got {}x{}",
                        s_increment.nrows(),
                        s_increment.ncols(),
                    ));
                }
                // Symmetrize the (already-symmetric) source with the shared
                // matrix utility. The survival builder's value-space
                // congruence is domain-specific; only the low-level symmetric
                // cleanup is common with the generic and SAE construction code.
                let mut s_full = s_increment.to_owned();
                symmetrize_in_place(&mut s_full);
                // S_mid = S_B[1:,1:] · L  (right-multiply by lower-triangular
                // cumsum): (S·L)[i,j] = Σ_k S[i,k]·L[k,j] = Σ_{k≥j} S[i,k]
                // because L[k,j] = 1 iff j ≤ k.
                let mut s_mid_full = Array2::<f64>::zeros((p_time_full, p_time_full));
                for i in 0..p_time_full {
                    for j in 0..p_time_full {
                        let mut v = 0.0;
                        for k in j..p_time_full {
                            v += s_full[[i, k]];
                        }
                        s_mid_full[[i, j]] = v;
                    }
                }
                // S_I_full = Lᵀ · S_mid = Lᵀ · S · L:
                // (Lᵀ·S_mid)[i,j] = Σ_k Lᵀ[i,k]·S_mid[k,j] = Σ_{k≥i} S_mid[k,j]
                // because Lᵀ[i,k] = L[k,i] = 1 iff i ≤ k.
                let mut s_full_congruent = Array2::<f64>::zeros((p_time_full, p_time_full));
                for i in 0..p_time_full {
                    for j in 0..p_time_full {
                        let mut v = 0.0;
                        for k in i..p_time_full {
                            v += s_mid_full[[k, j]];
                        }
                        s_full_congruent[[i, j]] = v;
                    }
                }
                // Principal submatrix on the retained (shape-varying) columns.
                let mut local = Array2::<f64>::zeros((p_time, p_time));
                for (i_new, &i_old) in keep_cols.iter().enumerate() {
                    for (j_new, &j_old) in keep_cols.iter().enumerate() {
                        // Symmetrize on the way out to absorb residual
                        // floating-point asymmetry.
                        local[[i_new, j_new]] = 0.5
                            * (s_full_congruent[[i_old, j_old]] + s_full_congruent[[j_old, i_old]]);
                    }
                }
                penalties.push(local);
            }

            // PSD contract (gam#979). The value-space congruence Lᵀ S_B[1:,1:] L,
            // restricted to a principal submatrix, is positive semidefinite by
            // construction. A negative eigenvalue here means the construction has
            // regressed to the increment-space / wrong-ordering form that made the
            // penalized survival NLL unbounded below (the #979 divergence). Verify
            // it here, at construction, so the defect can never silently reach the
            // inner solver again. The tolerance is the same relative scale the
            // nullspace detection below uses; a numerically tiny negative (round-off
            // on the genuine 1-D null direction) is allowed, a structural one is not.
            for (idx, s_mat) in penalties.iter().enumerate() {
                let p = s_mat.nrows();
                if p == 0 {
                    continue;
                }
                if let Ok((evals, _)) =
                    crate::faer_ndarray::FaerEigh::eigh(s_mat, faer::Side::Lower)
                {
                    let evals_slice: &[f64] = evals.as_slice().ok_or_else(|| {
                        "internal error: ispline penalty eigenvalues not contiguous".to_string()
                    })?;
                    let max_ev = evals_slice
                        .iter()
                        .copied()
                        .fold(0.0_f64, |a, b| a.max(b.abs()))
                        .max(1.0);
                    let min_ev = evals_slice.iter().copied().fold(f64::INFINITY, f64::min);
                    let neg_tol = -100.0 * (p as f64) * f64::EPSILON * max_ev;
                    if min_ev < neg_tol {
                        return Err(format!(
                            "internal error (gam#979): assembled ispline time-block penalty {idx} is \
                             indefinite (min eigenvalue {min_ev:.3e} < tol {neg_tol:.3e}, max |eig| \
                             {max_ev:.3e}); the value-space congruence Lᵀ S_B[1:,1:] L must be PSD"
                        ));
                    }
                }
            }

            // The value-space penalty S_I = L^T S_B[1:,1:] L has a 1-dimensional
            // null space (constant γ ↦ affine c ↦ D₂c = 0). Detect it spectrally
            // so the REML uses the generalized logdet over the penalized subspace.
            let nullspace_dims: Vec<usize> = penalties
                .iter()
                .map(|s_mat| {
                    let p = s_mat.nrows();
                    if p == 0 {
                        return 0;
                    }
                    match crate::faer_ndarray::FaerEigh::eigh(s_mat, faer::Side::Lower) {
                        Ok((evals, _)) => {
                            let evals_slice: &[f64] = evals.as_slice().unwrap();
                            let max_ev = evals_slice
                                .iter()
                                .copied()
                                .fold(0.0_f64, |a, b| a.max(b.abs()))
                                .max(1.0);
                            let threshold = 100.0 * (p as f64) * f64::EPSILON * max_ev;
                            evals_slice.iter().filter(|&&e| e <= threshold).count()
                        }
                        Err(_) => 0,
                    }
                })
                .collect();
            Ok(SurvivalTimeBuildOutput {
                x_entry_time: DesignMatrix::Dense(DenseDesignMatrix::from(x_entry_time)),
                x_exit_time: DesignMatrix::Dense(DenseDesignMatrix::from(x_exit_time)),
                x_derivative_time,
                penalties,
                nullspace_dims,
                basisname: "ispline".to_string(),
                degree: Some(degree),
                knots: Some(knotvec.to_vec()),
                keep_cols: Some(keep_cols),
                smooth_lambda: Some(smooth_lambda),
            })
        }
    }
}

pub fn resolved_survival_time_basis_config_from_build(
    basisname: &str,
    degree: Option<usize>,
    knots: Option<&Vec<f64>>,
    keep_cols: Option<&Vec<usize>>,
    smooth_lambda: Option<f64>,
) -> Result<SurvivalTimeBasisConfig, String> {
    match basisname {
        "none" => Ok(SurvivalTimeBasisConfig::None),
        "linear" => Ok(SurvivalTimeBasisConfig::Linear),
        "bspline" => Ok(SurvivalTimeBasisConfig::BSpline {
            degree: degree.ok_or_else(|| "survival bspline basis is missing degree".to_string())?,
            knots: Array1::from_vec(
                knots
                    .cloned()
                    .ok_or_else(|| "survival bspline basis is missing knots".to_string())?,
            ),
            smooth_lambda: smooth_lambda.unwrap_or(SURVIVAL_TIME_SMOOTH_LAMBDA_SEED),
        }),
        "ispline" => Ok(SurvivalTimeBasisConfig::ISpline {
            degree: degree.ok_or_else(|| "survival ispline basis is missing degree".to_string())?,
            knots: Array1::from_vec(
                knots
                    .cloned()
                    .ok_or_else(|| "survival ispline basis is missing knots".to_string())?,
            ),
            keep_cols: keep_cols
                .cloned()
                .ok_or_else(|| "survival ispline basis is missing keep_cols".to_string())?,
            smooth_lambda: smooth_lambda.unwrap_or(SURVIVAL_TIME_SMOOTH_LAMBDA_SEED),
        }),
        other => Err(format!("unsupported survival time basis '{other}'")),
    }
}

pub fn resolve_survival_time_anchor_value(
    age_entry: &Array1<f64>,
    time_anchor: Option<f64>,
) -> Result<f64, String> {
    if age_entry.is_empty() {
        return Err("survival time anchor requires non-empty entry times".to_string());
    }
    let anchor = match time_anchor {
        Some(t_anchor) => {
            if !t_anchor.is_finite() || t_anchor < 0.0 {
                return Err(format!(
                    "survival time anchor must be finite and non-negative, got {t_anchor}"
                ));
            }
            t_anchor
        }
        None => age_entry
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .ok_or_else(|| "failed to select survival time anchor".to_string())?,
    };
    Ok(anchor.max(SURVIVAL_TIME_FLOOR))
}

/// Marginal-slope centering anchor: a robust *interior* time on the **exit**
/// scale rather than the earliest entry age.
///
/// `center_survival_time_designs_at_anchor` subtracts the time-basis row at the
/// anchor from every entry/exit design row, so the anchor sets the origin of
/// the baseline-hazard I-spline's affine reparameterization. The
/// location-scale path anchors at the minimum entry age
/// ([`resolve_survival_time_anchor_value`]); for right-censored-only data that
/// minimum is ≈ the time origin, so centering is nearly a no-op.
///
/// Under **left truncation** the minimum entry age is a genuine positive
/// *left-tail* point, and centering there leaves the centered linear-trend
/// column `X(exit) − X(anchor)` large and one-signed across all rows (exit
/// times sit far to the right of the earliest entry). That column is the
/// unpenalized polynomial null space of the 2nd-difference time penalty, so the
/// inflated, one-signed column multiplies the marginal-slope time-block score
/// at the `γ = 0` monotone-cone seed up by hundreds — the constrained joint
/// Newton cannot certify KKT on it and REML rejects every seed (issue #751).
///
/// Centering instead at a robust interior location on the *exit* scale — the
/// **median exit age**, where the at-risk mass concentrates — keeps the
/// centered column small and two-signed (some exits below the median, some
/// above), so the exit-event likelihood pins the linear trend and the seed
/// score stays bounded. Re-centering is an exact affine reparameterization of
/// the baseline offset: the fitted `q(t)` and the REML objective are unchanged,
/// only the seed conditioning improves. The median is chosen (over the mean)
/// for robustness to the heavy right tail of survival times.
///
/// An explicit `--survival-time-anchor` is honored verbatim (same validation as
/// the location-scale path) so the user retains full control; the saved
/// `survival_time_anchor` scalar round-trips to predict unchanged.
pub fn resolve_survival_marginal_slope_time_anchor_value(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    time_anchor: Option<f64>,
) -> Result<f64, String> {
    if age_entry.is_empty() || age_exit.is_empty() {
        return Err(
            "survival marginal-slope time anchor requires non-empty entry/exit times".to_string(),
        );
    }
    let anchor = match time_anchor {
        Some(t_anchor) => {
            if !t_anchor.is_finite() || t_anchor < 0.0 {
                return Err(format!(
                    "survival time anchor must be finite and non-negative, got {t_anchor}"
                ));
            }
            t_anchor
        }
        None => {
            let mut sorted: Vec<f64> = age_exit.iter().copied().collect();
            sorted.sort_by(f64::total_cmp);
            let m = sorted.len();
            if m % 2 == 1 {
                sorted[m / 2]
            } else {
                0.5 * (sorted[m / 2 - 1] + sorted[m / 2])
            }
        }
    };
    Ok(anchor.max(SURVIVAL_TIME_FLOOR))
}

pub fn evaluate_survival_time_basis_row(
    age: f64,
    cfg: &SurvivalTimeBasisConfig,
) -> Result<Array1<f64>, String> {
    if !age.is_finite() || age < 0.0 {
        return Err(format!(
            "survival time basis row requires finite non-negative age, got {age}"
        ));
    }
    let age = age.max(SURVIVAL_TIME_FLOOR);
    let log_age = array![age.ln()];
    match cfg {
        SurvivalTimeBasisConfig::None => Ok(Array1::zeros(0)),
        SurvivalTimeBasisConfig::Linear => Ok(array![1.0, age.ln()]),
        SurvivalTimeBasisConfig::BSpline { degree, knots, .. } => {
            if knots.is_empty() {
                return Err(
                    "survival BSpline anchor evaluation requires resolved knot metadata"
                        .to_string(),
                );
            }
            let built = build_bspline_basis_1d(
                log_age.view(),
                &BSplineBasisSpec {
                    degree: *degree,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Provided(knots.clone()),
                    double_penalty: false,
                    identifiability: BSplineIdentifiability::None,
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            )
            .map_err(|e| format!("failed to evaluate survival bspline anchor row: {e}"))?;
            Ok(built.design.to_dense().row(0).to_owned())
        }
        SurvivalTimeBasisConfig::ISpline {
            degree,
            knots,
            keep_cols,
            ..
        } => {
            if knots.is_empty() {
                return Err(
                    "survival ISpline anchor evaluation requires resolved knot metadata"
                        .to_string(),
                );
            }
            let (basis_arc, _) = create_basis::<Dense>(
                log_age.view(),
                KnotSource::Provided(knots.view()),
                *degree,
                BasisOptions::i_spline(),
            )
            .map_err(|e| format!("failed to evaluate survival ispline anchor row: {e}"))?;
            let basis = basis_arc.as_ref();
            let row = basis.row(0);
            if keep_cols.is_empty() {
                return Ok(row.to_owned());
            }
            if keep_cols.iter().any(|&j| j >= row.len()) {
                return Err(SurvivalConstructionError::MissingColumn {
                    reason: "survival ISpline anchor keep_cols exceed basis width".to_string(),
                }
                .into());
            }
            Ok(Array1::from_iter(keep_cols.iter().map(|&j| row[j])))
        }
    }
}

pub fn center_survival_time_designs_at_anchor(
    design_entry: &mut DesignMatrix,
    design_exit: &mut DesignMatrix,
    anchor_row: &Array1<f64>,
) -> Result<(), String> {
    if design_entry.ncols() != anchor_row.len() || design_exit.ncols() != anchor_row.len() {
        return Err(format!(
            "survival time anchoring column mismatch: entry={}, exit={}, anchor={}",
            design_entry.ncols(),
            design_exit.ncols(),
            anchor_row.len()
        ));
    }
    // Centering destroys sparsity (every row gets a dense offset), so
    // materialize to dense.  This only runs once at construction time.
    fn center_dense(dm: &mut DesignMatrix, anchor: &Array1<f64>) {
        let mut dense = dm.to_dense();
        for mut row in dense.rows_mut() {
            row -= &anchor.view();
        }
        *dm = DesignMatrix::Dense(DenseDesignMatrix::from(dense));
    }
    center_dense(design_entry, anchor_row);
    center_dense(design_exit, anchor_row);
    Ok(())
}

// ---------------------------------------------------------------------------
// Baseline evaluation (Gompertz, Weibull, Gompertz-Makeham)
// ---------------------------------------------------------------------------

/// Partial derivatives of the baseline offsets `(eta_target, d_eta_target/dt)`
/// with respect to the θ-parameters in the same parameterization that
/// [`survival_baseline_theta_from_config`] / [`survival_baseline_config_from_theta`]
/// use:
///
/// - **Weibull**: θ = (log_scale, log_shape).  `eta = shape·(log t − log scale)`,
///   `o_D = shape/t`.
/// - **Gompertz**: θ = (log_rate, shape).  `eta = log H_G(t)` with
///   `H_G(t) = (rate/shape)·(exp(shape·t) − 1)`, `o_D = h_G(t)/H_G(t) =
///   shape·E/(E−1)` where `E = exp(shape·t)`.
/// - **Gompertz–Makeham**: θ = (log_rate, shape, log_makeham).
///   `eta = log H(t)` with `H(t) = makeham·t + H_G(t)`,
///   `o_D = (makeham + h_G(t)) / H(t)`.
///
/// Returns a flat `(d_eta/dθ_k, d_oD/dθ_k)` pair for each component of θ,
/// in the same order as `survival_baseline_theta_from_config`.  Linear has
/// no θ-parameters so returns `Ok(None)`.
///
/// The `eta`-channel derivatives are closed-form for every branch.  The
/// `o_D`-channel derivatives use the log-derivative identity
/// `∂o_D/∂θ = o_D · ∂log(o_D)/∂θ` which is more numerically stable near
/// the small-shape limit (shape·t → 0).  Near shape = 0 we fall back to
/// a third-order Taylor expansion with the same 1e-10 pivot that
/// `evaluate_survival_baseline` uses, keeping the value/derivative pair
/// continuous and agreement with the linear-hazard limit exact at shape=0.
pub fn baseline_offset_theta_partials(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<Vec<(f64, f64)>>, String> {
    let Some(params) = validated_baseline_params(age, cfg, "baseline derivative evaluation")?
    else {
        return Ok(None);
    };

    match params {
        ValidatedBaselineTarget::Weibull { scale, shape } => {
            // eta = shape·(log t − log scale)
            //     = shape·log t − shape·log scale
            // o_D = shape / t
            //
            // θ = (log_scale, log_shape):
            //   ∂eta/∂log_scale  = −shape          ∂o_D/∂log_scale = 0
            //   ∂eta/∂log_shape  = shape·(log t − log scale) = eta
            //   ∂o_D/∂log_shape  = shape / t = o_D
            let eta = shape * (age.ln() - scale.ln());
            let o_d = shape / age;
            let d_eta_d_log_scale = -shape;
            let d_od_d_log_scale = 0.0;
            let d_eta_d_log_shape = eta;
            let d_od_d_log_shape = o_d;
            Ok(Some(vec![
                (d_eta_d_log_scale, d_od_d_log_scale),
                (d_eta_d_log_shape, d_od_d_log_shape),
            ]))
        }
        ValidatedBaselineTarget::Gompertz { shape, .. } => {
            // θ = (log_rate, shape):
            //   Rate cancels in o_D = h/H for Gompertz, so ∂o_D/∂log_rate = 0
            //   and ∂eta/∂log_rate = 1. The shape channel uses
            //     ∂eta/∂shape   = −1/shape + t·E/(E−1)
            //     ∂log(o_D)/∂shape = 1/shape − t/(E−1)
            //     ∂o_D/∂shape  = o_D · ∂log(o_D)/∂shape
            //   Near shape=0 both numerators are 1/shape cancellations. Use
            //   Taylor expansions with the same 1e-10 pivot that
            //   gompertz_components uses in evaluate_survival_baseline.
            let (d_eta_d_shape, d_od_d_shape) = gompertz_shape_derivatives(age, shape);
            Ok(Some(vec![(1.0, 0.0), (d_eta_d_shape, d_od_d_shape)]))
        }
        ValidatedBaselineTarget::GompertzMakeham {
            rate,
            shape,
            makeham,
        } => {
            // H(t) = M·t + H_G(t),   H_G(t) = (rate/shape)·(E−1),  E = exp(shape·t)
            // h(t) = M + h_G(t),     h_G(t) = rate·E
            // o_D  = h/H
            //
            // θ = (log_rate, shape, log_makeham):
            //   ∂H/∂log_rate    = rate · ∂H/∂rate = H_G               (scales with rate)
            //   ∂H/∂shape       = H_G_shape                            (closed form below)
            //   ∂H/∂log_makeham = makeham · t                          (linear in makeham)
            //   ∂h/∂log_rate    = rate · ∂h/∂rate = h_G
            //   ∂h/∂shape       = h_G_shape = rate·t·E + 0              (= rate·t·E)
            //   ∂h/∂log_makeham = makeham
            //   ∂eta/∂θ = (∂H/∂θ) / H
            //   ∂o_D/∂θ = (∂h/∂θ − o_D · ∂H/∂θ) / H
            //           = (∂h/∂θ)/H − o_D · (∂H/∂θ)/H
            let (cum_g, inst_g) = gompertz_hazard_components(age, rate, shape);
            let cum_total = makeham * age + cum_g;
            if cum_total <= 0.0 || !cum_total.is_finite() {
                return Err(SurvivalConstructionError::DataValidationFailed {
                    reason: "gm baseline produced non-positive cumulative hazard".to_string(),
                }
                .into());
            }
            let inst_total = makeham + inst_g;
            let o_d = inst_total / cum_total;
            let inv_cum = 1.0 / cum_total;
            // Each channel: ∂cum/∂θ and ∂inst/∂θ → ∂eta/∂θ = ∂cum/∂θ / cum
            //                                       ∂o_D/∂θ = (∂inst/∂θ − o_D·∂cum/∂θ) / cum
            // log_rate channel: cum is linear in rate through H_G; ∂cum/∂rate = H_G/rate,
            //   so ∂cum/∂log_rate = H_G (= cum_g here). Similarly ∂inst/∂log_rate = h_G (= inst_g).
            let d_cum_dlr = cum_g;
            let d_inst_dlr = inst_g;
            let d_eta_dlr = d_cum_dlr * inv_cum;
            let d_od_dlr = (d_inst_dlr - o_d * d_cum_dlr) * inv_cum;
            // shape channel: only H_G and h_G have shape dependence.
            let (d_cum_dshape, d_inst_dshape) =
                gompertz_cumulative_shape_derivative(age, rate, shape);
            let d_eta_dshape = d_cum_dshape * inv_cum;
            let d_od_dshape = (d_inst_dshape - o_d * d_cum_dshape) * inv_cum;
            // log_makeham channel: cum contributes M·t, inst contributes M.
            //   ∂cum/∂log_makeham = makeham·t,  ∂inst/∂log_makeham = makeham.
            let d_cum_dlm = makeham * age;
            let d_inst_dlm = makeham;
            let d_eta_dlm = d_cum_dlm * inv_cum;
            let d_od_dlm = (d_inst_dlm - o_d * d_cum_dlm) * inv_cum;
            Ok(Some(vec![
                (d_eta_dlr, d_od_dlr),
                (d_eta_dshape, d_od_dshape),
                (d_eta_dlm, d_od_dlm),
            ]))
        }
    }
}

/// Shared chain-rule θ-gradient contraction for baseline offsets.
///
/// Both [`baseline_chain_rule_gradient`] (RP eta offsets) and
/// [`marginal_slope_baseline_chain_rule_gradient`] (probit q-offsets) reduce to
/// the same contraction of [`OffsetChannelResiduals`] against per-age baseline
/// θ-partials; only the `partials` provider differs. This engine owns the length
/// checks, the θ-dim probe, the parallel per-row reduction, the entry gating, and
/// the error handling. Each provider returns, per age, a length-`theta_dim` vector
/// of `(∂eta/∂θ_k, ∂(d eta/dt)/∂θ_k)` pairs (or `(∂q/∂θ_k, ∂(dq/dt)/∂θ_k)` for the
/// probit channel), and `None` when `cfg` has no θ-parameters (`Linear` target).
///
/// Contract (envelope theorem at converged β; the penalty has no θ dependence):
///
///   d[0.5·deviance + 0.5·βᵀS_λβ] / dθ_k
///     = Σᵢ r_X[i]·(∂o_X_i/∂θ_k) + r_D[i]·(∂o_D_i/∂θ_k) + r_E[i]·(∂o_E_i/∂θ_k)
///       + r_R[i]·(∂o_R_i/∂θ_k)
///
/// where `r_X = residuals.exit`, `r_D = residuals.derivative`, `r_E =
/// residuals.entry`, `r_R = residuals.right` (all sampleweight-scaled already).
/// Exit and derivative partials both come from the `age_exit[i]` evaluation;
/// the entry partial from `age_entry[i]`; the interval upper-bound (`R`)
/// η-partial from `age_right[i]`. Origin-entry rows have `r_E[i] == 0` exactly
/// and non-interval rows have `r_R[i] == 0` exactly, so those partials are
/// skipped for those rows (avoiding the `age > 0` precondition failure when an
/// inactive boundary age is 0 / a placeholder).
///
/// Returns `Ok(None)` when the provider reports no θ-parameters.
fn baseline_chain_rule_gradient_with_partials<F>(
    label: &'static str,
    age_entry: ndarray::ArrayView1<'_, f64>,
    age_exit: ndarray::ArrayView1<'_, f64>,
    age_right: ndarray::ArrayView1<'_, f64>,
    cfg: &SurvivalBaselineConfig,
    residuals: &crate::families::survival::OffsetChannelResiduals,
    partials: F,
) -> Result<Option<Array1<f64>>, String>
where
    F: Fn(f64, &SurvivalBaselineConfig) -> Result<Option<Vec<(f64, f64)>>, String> + Sync,
{
    let n = age_exit.len();
    if age_entry.len() != n
        || age_right.len() != n
        || residuals.exit.len() != n
        || residuals.entry.len() != n
        || residuals.derivative.len() != n
        || residuals.right.len() != n
    {
        return Err(format!(
            "{label}: length mismatch (age_entry={}, age_exit={}, age_right={}, r_exit={}, r_entry={}, r_deriv={}, r_right={})",
            age_entry.len(),
            n,
            age_right.len(),
            residuals.exit.len(),
            residuals.entry.len(),
            residuals.derivative.len(),
            residuals.right.len(),
        ));
    }
    // Probe θ-dim via any valid positive age. If the provider returns None the
    // config carries no θ-parameters (Linear target) and there is no θ-gradient.
    let probe_age = age_exit.iter().copied().find(|v| v.is_finite() && *v > 0.0);
    let theta_dim = match probe_age {
        Some(t) => match partials(t, cfg)? {
            None => return Ok(None),
            Some(v) => v.len(),
        },
        None => {
            return Err(format!("{label}: no valid positive age for dim probe"));
        }
    };
    // Per-row partial contractions are independent, but each row's
    // contribution is a `theta_dim`-vector of `O(theta_dim · partial_cost)`
    // flops — small enough that the rayon parallel reduction's split
    // overhead dominates for any plausible `theta_dim`, *and* the
    // non-associative IEEE-754 sum order across thread chunks made the
    // engine drift in the low-order bits from row to row. The serial
    // accumulator below mirrors the inline reference exactly (and remains
    // ~memory-bandwidth-bound at large-scale `n`), so the engine is now a
    // bit-for-bit replacement for the legacy path, not just a
    // floating-point-noise-equivalent one.
    let mut grad = Array1::<f64>::zeros(theta_dim);
    for i in 0..n {
        // Exit + derivative partials both come from the age_exit evaluation.
        let partials_exit = partials(age_exit[i], cfg)?
            .ok_or_else(|| format!("{label}: unexpected None from partials at exit"))?;
        if partials_exit.len() != theta_dim {
            return Err(format!(
                "{label}: theta_dim drifted ({} != {})",
                partials_exit.len(),
                theta_dim
            ));
        }
        let r_x = residuals.exit[i];
        let r_d = residuals.derivative[i];
        for k in 0..theta_dim {
            let (d_eta_dk, d_od_dk) = partials_exit[k];
            grad[k] += r_x * d_eta_dk + r_d * d_od_dk;
        }
        // Entry channel is nonzero only for rows with a positive entry
        // interval; for origin-entry rows age_entry may be 0 and calling
        // the provider would error. Gate on residual==0.
        let r_e = residuals.entry[i];
        if r_e != 0.0 {
            let partials_entry = partials(age_entry[i], cfg)?
                .ok_or_else(|| format!("{label}: unexpected None from partials at entry"))?;
            for k in 0..theta_dim {
                grad[k] += r_e * partials_entry[k].0;
            }
        }
        // Interval upper-bound (`R`) channel: `q_right = X_time(R)·β + o_R(θ)`
        // carries its own baseline-θ η-offset evaluated at `age_right[i]`. It is
        // an η-level offset with NO time-derivative channel (the interval
        // likelihood `log[S(L) − S(R)]` has no hazard-derivative term), so it
        // contracts against the η-partial `.0` only. Nonzero only for
        // interval-censored latent rows; for every other channel/model
        // `r_right[i] == 0` exactly, so the (possibly placeholder) `age_right[i]`
        // partial is never consulted.
        let r_r = residuals.right[i];
        if r_r != 0.0 {
            let partials_right = partials(age_right[i], cfg)?.ok_or_else(|| {
                format!("{label}: unexpected None from partials at right boundary")
            })?;
            if partials_right.len() != theta_dim {
                return Err(format!(
                    "{label}: theta_dim drifted at right boundary ({} != {})",
                    partials_right.len(),
                    theta_dim
                ));
            }
            for k in 0..theta_dim {
                grad[k] += r_r * partials_right[k].0;
            }
        }
    }
    Ok(Some(grad))
}

/// Contract `OffsetChannelResiduals` against `baseline_offset_theta_partials`
/// to produce the closed-form θ-gradient of the unpenalized NLL at converged β.
///
/// Derivation (envelope theorem on the penalized objective, β* minimizes the
/// same cost wrt β and the penalty has no θ dependence):
///
///   d[0.5·deviance + 0.5·βᵀS_λβ] / dθ_k
///     = d[NLL(β*; o(θ))] / dθ_k
///     = Σᵢ (∂NLL_i/∂o_X[i])·(∂o_X_i/∂θ_k)
///       + (∂NLL_i/∂o_E[i])·(∂o_E_i/∂θ_k)
///       + (∂NLL_i/∂o_D[i])·(∂o_D_i/∂θ_k)
///       + (∂NLL_i/∂o_R[i])·(∂o_R_i/∂θ_k)
///
/// The four `∂NLL_i/∂o_channel` terms are the `exit`, `entry`, `derivative`,
/// `right` fields of [`OffsetChannelResiduals`] (sampleweight-scaled already).
/// The `∂o/∂θ_k` terms come from [`baseline_offset_theta_partials`] per obs at
/// the appropriate age.
///
/// Per the RP offset convention:
///   o_E[i] = eta_target(age_entry[i])
///   o_X[i] = eta_target(age_exit[i])
///   o_D[i] = d/dt eta_target(t) |_{t=age_exit[i]}
///   o_R[i] = eta_target(age_right[i])   (interval upper bound `R`; η-level only)
///
/// so the exit and derivative partials are both evaluated at `age_exit[i]`,
/// the entry partial at `age_entry[i]`, and the interval-right η-partial at
/// `age_right[i]`. The origin-entry case (`entry_at_origin[i]`) has
/// `r_entry[i] = 0` exactly and every non-interval row has `r_right[i] = 0`
/// exactly, so we skip the `baseline_offset_theta_partials(age, ..)` call for
/// those rows (avoiding the `age > 0` precondition failure when an inactive
/// boundary age is 0 / a placeholder).
///
/// Returns `Ok(None)` when `cfg.target == Linear` (no θ-parameters).
pub fn baseline_chain_rule_gradient(
    age_entry: ndarray::ArrayView1<'_, f64>,
    age_exit: ndarray::ArrayView1<'_, f64>,
    age_right: ndarray::ArrayView1<'_, f64>,
    cfg: &SurvivalBaselineConfig,
    residuals: &crate::families::survival::OffsetChannelResiduals,
) -> Result<Option<Array1<f64>>, String> {
    baseline_chain_rule_gradient_with_partials(
        "baseline_chain_rule_gradient",
        age_entry,
        age_exit,
        age_right,
        cfg,
        residuals,
        baseline_offset_theta_partials,
    )
}

/// Chain-rule θ-gradient for marginal-slope probit baseline offsets.
///
/// This is the probit-survival counterpart of [`baseline_chain_rule_gradient`].
/// It contracts residuals against
/// [`marginal_slope_baseline_offset_theta_partials`], so the offset channels
/// are `(q_entry, q_exit, dq_exit/dt)` with `Phi(-q(t)) = exp(-H0(t))`.
pub fn marginal_slope_baseline_chain_rule_gradient(
    age_entry: ndarray::ArrayView1<'_, f64>,
    age_exit: ndarray::ArrayView1<'_, f64>,
    cfg: &SurvivalBaselineConfig,
    residuals: &crate::families::survival::OffsetChannelResiduals,
) -> Result<Option<Array1<f64>>, String> {
    // Marginal-slope has no interval upper-bound channel; `residuals.right` is
    // all-zero, so the right channel never contracts and `age_exit` serves as an
    // unconsulted placeholder for the (unused) `age_right` argument.
    baseline_chain_rule_gradient_with_partials(
        "marginal_slope_baseline_chain_rule_gradient",
        age_entry,
        age_exit,
        age_exit,
        cfg,
        residuals,
        marginal_slope_baseline_offset_theta_partials,
    )
}

/// Shared Gompertz hazard components `(H_G(t), h_G(t))`.
/// Mirrors the private helper in `evaluate_survival_baseline` with the
/// same 1e-10 small-shape pivot.
#[inline]
fn gompertz_hazard_components(age: f64, rate: f64, shape: f64) -> (f64, f64) {
    if shape.abs() < 1e-10 {
        // Taylor at shape=0: H_G(t) = rate·t·(1 + shape·t/2 + (shape·t)²/6),
        // h_G(t) = rate·(1 + shape·t + (shape·t)²/2).
        let x = shape * age;
        (
            rate * age * (1.0 + 0.5 * x + x * x / 6.0),
            rate * (1.0 + x + 0.5 * x * x),
        )
    } else {
        let shape_age = shape * age;
        let cumulative_hazard = (rate / shape) * shape_age.exp_m1();
        let instant_hazard = rate * shape_age.exp();
        (cumulative_hazard, instant_hazard)
    }
}

/// Partials of `(H_G(t), h_G(t))` with respect to the shape parameter.
///
/// H_G(t) = (rate/shape)·(E−1),  h_G(t) = rate·E,  E = exp(shape·t)
///
/// ∂H_G/∂shape  = −(rate/shape²)·(E−1) + (rate/shape)·t·E
///              = rate·[t·E/shape − (E−1)/shape²]
///              = rate·[t·E·shape − (E−1)] / shape²
/// ∂h_G/∂shape  = rate·t·E
///
/// Near shape=0 the first expression has a 1/shape² singularity that
/// cancels analytically. Using the series E−1 = Σₖ≥₁ (shape·t)ᵏ/k!:
///   t·E·shape − (E−1) = Σₖ≥₁ (shape·t)ᵏ·(k−1)/k!·shape⁰  [after simplification]
///                     = (shape·t)²/2 + 2(shape·t)³/6 + 3(shape·t)⁴/24 + ...
/// so ∂H_G/∂shape at shape→0 = rate·[t²/2 + shape·t³/3 + shape²·t⁴/8 + ...].
/// We use that Taylor expansion in the small-shape branch.
#[inline]
fn gompertz_cumulative_shape_derivative(age: f64, rate: f64, shape: f64) -> (f64, f64) {
    let x = shape * age;
    let dinstg_dshape = rate * age * x.exp();
    // The exact form rate·[t·E·shape − (E−1)]/shape² is a difference of two
    // O(1/shape) quantities whose leading terms cancel, so its accuracy is
    // governed by the dimensionless product x = shape·age, NOT by `shape`
    // alone. Pivoting on `shape < 1e-10` ignored `age`: for large ages a small
    // shape still yields a small x where the catastrophic cancellation has
    // already corrupted the difference. Pivot on x instead; the 3-term Taylor
    // (through O(x²)) is accurate to <1e-9 for |x| < 1e-4, and the exact branch
    // is clean above it.
    let dhg_dshape = if x.abs() < 1e-4 {
        let t = age;
        // Truncated to O(x³): t²/2 + x·t²/3 + x²·t²/8
        rate * t * t * (0.5 + x / 3.0 + x * x / 8.0)
    } else {
        // t·E·shape − (E−1) = t·e^x·shape − expm1(x)
        let e = x.exp();
        let em1 = x.exp_m1();
        let numerator = age * e * shape - em1;
        rate * numerator / (shape * shape)
    };
    (dhg_dshape, dinstg_dshape)
}

/// Partials `(∂eta/∂shape, ∂o_D/∂shape)` for the pure Gompertz baseline.
/// Pure Gompertz has rate cancelling in o_D, so there is no log_rate
/// contribution in o_D. The rate channel for eta is trivially 1; this
/// helper only covers the shape channel.
#[inline]
fn gompertz_shape_derivatives(age: f64, shape: f64) -> (f64, f64) {
    if shape.abs() < 1e-10 {
        // Closed-form limits from the series t·E/(E−1) = 1/x + 1/2 + x/12 + ...
        // with E = e^x, x = shape·t:
        //   ∂eta/∂shape  = −1/shape + t·E/(E−1)
        //                = t/2 + shape·t²/12 + O(shape²)
        //   o_D         = shape·E/(E−1)
        //                = 1/t + shape/2 + shape²·t/12 + O(shape³)
        //   ∂log(o_D)/∂shape = 1/shape − t/(E−1)
        //                = t/2 − shape·t²/12 + O(shape²)
        //   ∂o_D/∂shape = o_D · ∂log(o_D)/∂shape
        let t = age;
        let d_eta = 0.5 * t + shape * t * t / 12.0;
        let dlog_od = 0.5 * t - shape * t * t / 12.0;
        let o_d = 1.0 / t + 0.5 * shape + shape * shape * t / 12.0;
        (d_eta, o_d * dlog_od)
    } else {
        let x = shape * age;
        let e = x.exp();
        let em1 = x.exp_m1(); // E − 1 via expm1 for accuracy at small x
        let d_eta = -1.0 / shape + age * e / em1;
        // o_D = shape · E/(E−1); ∂log(o_D)/∂shape = 1/shape − t/(E−1)
        let o_d = shape * e / em1;
        let dlog_od = 1.0 / shape - age / em1;
        (d_eta, o_d * dlog_od)
    }
}

/// Per-target baseline parameters after the shared age guard and the per-target
/// required-field extraction + finiteness/positivity validation have passed.
///
/// This is the single source of truth for *which* config fields each baseline
/// target requires and *what* domain each must satisfy. Both the hazard-value
/// evaluator (`survival_cumulative_and_instant_hazard`) and the θ-partials
/// evaluator (`survival_hazard_theta_partials`) consume it and only differ in how
/// they assemble their (value vs derivative) outputs from these checked scalars.
#[derive(Clone, Copy, Debug)]
enum ValidatedBaselineTarget {
    Weibull { scale: f64, shape: f64 },
    Gompertz { rate: f64, shape: f64 },
    GompertzMakeham { rate: f64, shape: f64, makeham: f64 },
}

/// Shared prologue for the survival baseline hazard evaluators: validate the age,
/// then extract and domain-check the per-target parameters from `cfg`.
///
/// `Ok(None)` is the `Linear` target (no parametric baseline). `context` is woven
/// into the age-guard error so each caller keeps its specific phrasing.
fn validated_baseline_params(
    age: f64,
    cfg: &SurvivalBaselineConfig,
    context: &str,
) -> Result<Option<ValidatedBaselineTarget>, String> {
    if !age.is_finite() || age <= 0.0 {
        return Err(format!(
            "survival ages must be finite and positive for {context}"
        ));
    }

    match cfg.target {
        SurvivalBaselineTarget::Linear => Ok(None),
        SurvivalBaselineTarget::Weibull => {
            let scale = cfg
                .scale
                .ok_or_else(|| "weibull missing scale".to_string())?;
            let shape = cfg
                .shape
                .ok_or_else(|| "weibull missing shape".to_string())?;
            if !(scale.is_finite() && shape.is_finite() && scale > 0.0 && shape > 0.0) {
                return Err(SurvivalConstructionError::InvalidConfig {
                    reason: "weibull baseline requires finite positive scale and shape".to_string(),
                }
                .into());
            }
            Ok(Some(ValidatedBaselineTarget::Weibull { scale, shape }))
        }
        SurvivalBaselineTarget::Gompertz => {
            let rate = cfg
                .rate
                .ok_or_else(|| "gompertz missing rate".to_string())?;
            let shape = cfg
                .shape
                .ok_or_else(|| "gompertz missing shape".to_string())?;
            if !(rate.is_finite() && shape.is_finite() && rate > 0.0) {
                return Err(
                    "gompertz baseline requires finite positive rate and finite shape".to_string(),
                );
            }
            Ok(Some(ValidatedBaselineTarget::Gompertz { rate, shape }))
        }
        SurvivalBaselineTarget::GompertzMakeham => {
            let rate = cfg
                .rate
                .ok_or_else(|| "gompertz-makeham missing rate".to_string())?;
            let shape = cfg
                .shape
                .ok_or_else(|| "gompertz-makeham missing shape".to_string())?;
            let makeham = cfg
                .makeham
                .ok_or_else(|| "gompertz-makeham missing makeham".to_string())?;
            if !(rate.is_finite()
                && shape.is_finite()
                && makeham.is_finite()
                && rate > 0.0
                && makeham > 0.0)
            {
                return Err(
                    "gompertz-makeham baseline requires finite positive rate, makeham, and finite shape"
                        .to_string(),
                );
            }
            Ok(Some(ValidatedBaselineTarget::GompertzMakeham {
                rate,
                shape,
                makeham,
            }))
        }
    }
}

fn survival_hazard_theta_partials(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<Vec<(f64, f64)>>, String> {
    let Some(params) = validated_baseline_params(age, cfg, "baseline hazard partials")? else {
        return Ok(None);
    };

    match params {
        ValidatedBaselineTarget::Weibull { scale, shape } => {
            let log_time_ratio = age.ln() - scale.ln();
            let cumulative_hazard = (age / scale).powf(shape);
            let instant_hazard = shape * cumulative_hazard / age;
            let eta = shape * log_time_ratio;
            Ok(Some(vec![
                (-shape * cumulative_hazard, -shape * instant_hazard),
                (eta * cumulative_hazard, (1.0 + eta) * instant_hazard),
            ]))
        }
        ValidatedBaselineTarget::Gompertz { rate, shape } => {
            let (cumulative_hazard, instant_hazard) = gompertz_hazard_components(age, rate, shape);
            let (d_cum_dshape, d_inst_dshape) =
                gompertz_cumulative_shape_derivative(age, rate, shape);
            Ok(Some(vec![
                (cumulative_hazard, instant_hazard),
                (d_cum_dshape, d_inst_dshape),
            ]))
        }
        ValidatedBaselineTarget::GompertzMakeham {
            rate,
            shape,
            makeham,
        } => {
            let (cum_gompertz, inst_gompertz) = gompertz_hazard_components(age, rate, shape);
            let (d_cum_dshape, d_inst_dshape) =
                gompertz_cumulative_shape_derivative(age, rate, shape);
            Ok(Some(vec![
                (cum_gompertz, inst_gompertz),
                (d_cum_dshape, d_inst_dshape),
                (makeham * age, makeham),
            ]))
        }
    }
}

fn survival_cumulative_and_instant_hazard(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<(f64, f64)>, String> {
    let Some(params) = validated_baseline_params(age, cfg, "baseline hazard evaluation")? else {
        return Ok(None);
    };

    match params {
        ValidatedBaselineTarget::Weibull { scale, shape } => {
            let cumulative_hazard = (age / scale).powf(shape);
            let instant_hazard = shape * cumulative_hazard / age;
            Ok(Some((cumulative_hazard, instant_hazard)))
        }
        ValidatedBaselineTarget::Gompertz { rate, shape } => {
            let (cumulative_hazard, instant_hazard) = gompertz_hazard_components(age, rate, shape);
            Ok(Some((cumulative_hazard, instant_hazard)))
        }
        ValidatedBaselineTarget::GompertzMakeham {
            rate,
            shape,
            makeham,
        } => {
            let (h_gompertz, inst_gompertz) = gompertz_hazard_components(age, rate, shape);
            Ok(Some((makeham * age + h_gompertz, makeham + inst_gompertz)))
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MarginalSlopeBaselinePoint {
    instant_hazard: f64,
    q: f64,
    q_t: f64,
}

fn evaluate_marginal_slope_baseline_point(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<MarginalSlopeBaselinePoint>, String> {
    let Some((cumulative_hazard, instant_hazard)) =
        survival_cumulative_and_instant_hazard(age, cfg)?
    else {
        return Ok(None);
    };
    if !(cumulative_hazard.is_finite() && cumulative_hazard > 0.0) {
        return Err(format!(
            "{} marginal-slope baseline produced non-positive cumulative hazard",
            survival_baseline_targetname(cfg.target)
        ));
    }
    if !(instant_hazard.is_finite() && instant_hazard > 0.0) {
        return Err(format!(
            "{} marginal-slope baseline produced non-positive instant hazard",
            survival_baseline_targetname(cfg.target)
        ));
    }
    let survival = (-cumulative_hazard).exp();
    if !(survival.is_finite() && survival > 0.0 && survival < 1.0) {
        return Err(format!(
            "{} marginal-slope baseline survival must be strictly inside (0,1), got {survival}",
            survival_baseline_targetname(cfg.target)
        ));
    }
    let q = -standard_normal_quantile(survival).map_err(|e| {
        format!(
            "{} marginal-slope baseline failed to invert survival probability {survival}: {e}",
            survival_baseline_targetname(cfg.target)
        )
    })?;
    let phi_q = normal_pdf(q);
    if !(phi_q.is_finite() && phi_q > 0.0) {
        return Err(format!(
            "{} marginal-slope baseline produced non-positive probit density phi(q)={phi_q} at q={q}",
            survival_baseline_targetname(cfg.target)
        ));
    }
    Ok(Some(MarginalSlopeBaselinePoint {
        instant_hazard,
        q,
        q_t: instant_hazard * survival / phi_q,
    }))
}

/// Evaluate the parametric baseline target at a given age.
/// Returns `(eta_target(age), d eta_target / d age)` on the log-cumulative-hazard scale.
pub fn evaluate_survival_baseline(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<(f64, f64), String> {
    if !age.is_finite() || age < 0.0 {
        return Err(
            "survival ages must be finite and non-negative for baseline target evaluation"
                .to_string(),
        );
    }

    // At t = 0 every parametric cumulative-hazard target satisfies H(0) = 0
    // exactly (this is the defining property of a cumulative hazard:
    // S(0) = 1 ⇒ H(0) = -log S(0) = 0). The log-cumulative-hazard offset is
    // therefore eta(0) = log H(0) = -inf, and we report a zero log-derivative
    // since `exp(eta(0)) = H(0) = 0` is the only physically valid value.
    // Returning `Ok((-inf, 0.0))` keeps the baseline cumulative hazard exactly
    // zero at the origin; downstream callers that need to multiply this offset
    // into a linear predictor are responsible for handling the origin row via
    // the `entry_at_origin` / `exit_at_origin` gating already wired through the
    // engine.
    if age == 0.0 {
        return match cfg.target {
            SurvivalBaselineTarget::Linear => Ok((0.0, 0.0)),
            SurvivalBaselineTarget::Weibull
            | SurvivalBaselineTarget::Gompertz
            | SurvivalBaselineTarget::GompertzMakeham => Ok((f64::NEG_INFINITY, 0.0)),
        };
    }

    let Some(params) = validated_baseline_params(age, cfg, "baseline target evaluation")? else {
        return Ok((0.0, 0.0));
    };

    match params {
        ValidatedBaselineTarget::Weibull { scale, shape } => {
            let eta = shape * (age.ln() - scale.ln());
            let derivative = shape / age;
            Ok((eta, derivative))
        }
        ValidatedBaselineTarget::Gompertz { rate, shape } => {
            let (h, inst) = gompertz_hazard_components(age, rate, shape);
            if h <= 0.0 || !h.is_finite() {
                return Err(if shape.abs() < 1e-10 {
                    "invalid gompertz baseline at near-zero shape".to_string()
                } else {
                    "gompertz baseline produced non-positive cumulative hazard".to_string()
                });
            }
            let derivative = inst / h;
            Ok((h.ln(), derivative))
        }
        ValidatedBaselineTarget::GompertzMakeham {
            rate,
            shape,
            makeham,
        } => {
            let (h_gompertz, inst_gompertz) = gompertz_hazard_components(age, rate, shape);
            let h = makeham * age + h_gompertz;
            if h <= 0.0 || !h.is_finite() {
                return Err(
                    "gompertz-makeham baseline produced non-positive cumulative hazard".to_string(),
                );
            }
            let inst = makeham + inst_gompertz;
            let derivative = inst / h;
            Ok((h.ln(), derivative))
        }
    }
}

/// Evaluate the parametric baseline as the probit index whose marginal
/// survival is the true hazard survival `exp(-H0(t))`.
///
/// Returns `(q(age), dq / d age)` such that `Phi(-q(age)) = exp(-H0(age))`.
/// The derivative is `h0(t) * exp(-H0(t)) / phi(q(t))`.
pub fn evaluate_survival_marginal_slope_baseline(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<(f64, f64), String> {
    // Survival-curve origin. Every cumulative-hazard baseline satisfies
    // `H0(0) = 0` (`S0(0) = exp(-H0(0)) = 1`), so the probit index
    // `q(0) = -Phi^{-1}(S0(0)) = -Phi^{-1}(1) = -inf`: there is no *finite*
    // probit-survival offset at the origin. The survival surface anchors
    // `S(0) = 1` directly (see the `t <= 0` origin handling in the survival
    // predict paths), so the baseline contributes nothing here — report the
    // zero offset rather than aborting in the `age <= 0` hazard guard. This
    // mirrors `evaluate_survival_baseline`'s explicit `age == 0` branch on the
    // log-cumulative-hazard channel; without it the probit/marginal-slope
    // baseline path (location-scale + marginal-slope likelihoods) could not be
    // evaluated on a prediction grid whose first node is the origin (#1024).
    if age == 0.0 {
        return Ok((0.0, 0.0));
    }
    let Some(point) = evaluate_marginal_slope_baseline_point(age, cfg)? else {
        return Ok((0.0, 0.0));
    };
    Ok((point.q, point.q_t))
}

/// Partial derivatives of the true survival marginal-slope probit offsets
/// `(q(t), dq(t)/dt)` with respect to the baseline θ-parameters.
///
/// The returned channels match `survival_baseline_theta_from_config`.  For
/// Gompertz-Makeham, θ is `(log_rate, shape, log_makeham)`.  If
/// `S(t)=exp(-H(t))`, `q(t)=-Phi^-1(S(t))`, `A(t)=S(t)/phi(q(t))`, and
/// `h(t)=dH/dt`, then
///
///   dq/dθ      = A * dH/dθ
///   d(q')/dθ   = A * (dh/dθ + h * (q*A - 1) * dH/dθ)
///
/// which keeps the probit transform and the hazard baseline analytically tied.
pub fn marginal_slope_baseline_offset_theta_partials(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<Vec<(f64, f64)>>, String> {
    let Some(point) = evaluate_marginal_slope_baseline_point(age, cfg)? else {
        return Ok(None);
    };
    let hazard_partials = survival_hazard_theta_partials(age, cfg)?
        .ok_or_else(|| "unexpected missing hazard partials for nonlinear baseline".to_string())?;
    let a = point.q_t / point.instant_hazard;
    let a_log_derivative_factor = point.q * a - 1.0;
    Ok(Some(
        hazard_partials
            .into_iter()
            .map(|(d_h_cum, d_h_inst)| {
                (
                    a * d_h_cum,
                    a * (d_h_inst + point.instant_hazard * a_log_derivative_factor * d_h_cum),
                )
            })
            .collect(),
    ))
}

/// Contract marginal-slope offset residuals and channel curvatures into the
/// exact Hessian with respect to baseline θ-parameters.
pub fn marginal_slope_baseline_chain_rule_hessian(
    age_entry: ndarray::ArrayView1<'_, f64>,
    age_exit: ndarray::ArrayView1<'_, f64>,
    cfg: &SurvivalBaselineConfig,
    residuals: &crate::families::survival::OffsetChannelResiduals,
    curvatures: &crate::families::survival::OffsetChannelCurvatures,
) -> Result<Option<Array2<f64>>, String> {
    let n = age_exit.len();
    if age_entry.len() != n
        || residuals.exit.len() != n
        || residuals.entry.len() != n
        || residuals.derivative.len() != n
        || curvatures.rows.len() != n
    {
        return Err(format!(
            "marginal_slope_baseline_chain_rule_hessian: length mismatch (age_entry={}, age_exit={}, r_exit={}, r_entry={}, r_deriv={}, h_rows={})",
            age_entry.len(),
            n,
            residuals.exit.len(),
            residuals.entry.len(),
            residuals.derivative.len(),
            curvatures.rows.len(),
        ));
    }
    let probe_age = age_exit.iter().copied().find(|v| v.is_finite() && *v > 0.0);
    let dim = match probe_age {
        Some(t) => match marginal_slope_baseline_offset_theta_second_partials(t, cfg)? {
            None => return Ok(None),
            Some(parts) => parts.first.len(),
        },
        None => {
            return Err(
                "marginal_slope_baseline_chain_rule_hessian: no valid positive age for dim probe"
                    .to_string(),
            );
        }
    };
    // Per-row Hessian contractions are independent. Each row contributes a
    // dim×dim increment combining second partials (exit/entry channels) with
    // the curvature-weighted outer product of the (entry, exit, derivative)
    // first-partial Jacobians. Parallel try_fold/try_reduce accumulates them.
    let hessian = (0..n)
        .into_par_iter()
        .try_fold(
            || Array2::<f64>::zeros((dim, dim)),
            |mut acc, i| -> Result<Array2<f64>, String> {
                let exit_parts =
                    marginal_slope_baseline_offset_theta_second_partials(age_exit[i], cfg)?
                        .ok_or_else(|| {
                            "unexpected None from marginal-slope second partials at exit"
                                .to_string()
                        })?;
                if exit_parts.first.len() != dim {
                    return Err(
                        "marginal_slope_baseline_chain_rule_hessian: theta_dim drifted".to_string(),
                    );
                }
                let mut entry_parts = None;
                if residuals.entry[i] != 0.0 {
                    entry_parts = Some(
                        marginal_slope_baseline_offset_theta_second_partials(age_entry[i], cfg)?
                            .ok_or_else(|| {
                                "unexpected None from marginal-slope second partials at entry"
                                    .to_string()
                            })?,
                    );
                }
                for a in 0..dim {
                    for b in 0..dim {
                        let j_exit_a = exit_parts.first[a].0;
                        let j_exit_b = exit_parts.first[b].0;
                        let j_deriv_a = exit_parts.first[a].1;
                        let j_deriv_b = exit_parts.first[b].1;
                        let mut value = residuals.exit[i] * exit_parts.second[a][b].0
                            + residuals.derivative[i] * exit_parts.second[a][b].1;
                        if let Some(parts) = entry_parts.as_ref() {
                            value += residuals.entry[i] * parts.second[a][b].0;
                        }
                        let curv = curvatures.rows[i];
                        let j_entry_a = entry_parts.as_ref().map_or(0.0, |parts| parts.first[a].0);
                        let j_entry_b = entry_parts.as_ref().map_or(0.0, |parts| parts.first[b].0);
                        let ja = [j_entry_a, j_exit_a, j_deriv_a];
                        let jb = [j_entry_b, j_exit_b, j_deriv_b];
                        for u in 0..3 {
                            for v in 0..3 {
                                value += ja[u] * curv[u][v] * jb[v];
                            }
                        }
                        acc[[a, b]] += value;
                    }
                }
                Ok(acc)
            },
        )
        .try_reduce(|| Array2::<f64>::zeros((dim, dim)), |a, b| Ok(a + b))?;
    Ok(Some(hessian))
}

struct MarginalSlopeThetaSecondPartials {
    first: Vec<(f64, f64)>,
    second: Vec<Vec<(f64, f64)>>,
}

fn marginal_slope_baseline_offset_theta_second_partials(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<MarginalSlopeThetaSecondPartials>, String> {
    let Some(point) = evaluate_marginal_slope_baseline_point(age, cfg)? else {
        return Ok(None);
    };
    let Some((hazard, first, second)) = survival_hazard_theta_first_second(age, cfg)? else {
        return Ok(None);
    };
    let (cum_hazard, instant_hazard) = hazard;
    let survival = (-cum_hazard).exp();
    let a = survival / normal_pdf(point.q);
    let b = point.q * a - 1.0;
    let b_factor = a + point.q * b;
    let dim = first.len();
    let mut first_out = Vec::with_capacity(dim);
    let mut second_out = vec![vec![(0.0, 0.0); dim]; dim];
    for i in 0..dim {
        let (h_i, inst_i) = first[i];
        first_out.push((a * h_i, a * (inst_i + instant_hazard * b * h_i)));
    }
    for i in 0..dim {
        for j in 0..dim {
            let (h_i, inst_i) = first[i];
            let (h_j, inst_j) = first[j];
            let (h_ij, inst_ij) = second[i][j];
            let a_j = a * b * h_j;
            let b_j = a * h_j * b_factor;
            let q_ij = a * h_ij + a * b * h_i * h_j;
            let qt_inner_i = inst_i + instant_hazard * b * h_i;
            let qt_ij = a_j * qt_inner_i
                + a * (inst_ij + inst_j * b * h_i + instant_hazard * (b_j * h_i + b * h_ij));
            second_out[i][j] = (q_ij, qt_ij);
        }
    }
    Ok(Some(MarginalSlopeThetaSecondPartials {
        first: first_out,
        second: second_out,
    }))
}

type HazardFirstSecond = ((f64, f64), Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>);

fn survival_hazard_theta_first_second(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<HazardFirstSecond>, String> {
    let Some(hazard) = survival_cumulative_and_instant_hazard(age, cfg)? else {
        return Ok(None);
    };
    let first = survival_hazard_theta_partials(age, cfg)?
        .ok_or_else(|| "unexpected missing hazard partials".to_string())?;
    let dim = first.len();
    let mut second = vec![vec![(0.0, 0.0); dim]; dim];
    match cfg.target {
        SurvivalBaselineTarget::Linear => return Ok(None),
        SurvivalBaselineTarget::Weibull => {
            let scale = cfg
                .scale
                .ok_or_else(|| "weibull missing scale".to_string())?;
            let shape = cfg
                .shape
                .ok_or_else(|| "weibull missing shape".to_string())?;
            let log_time_ratio = age.ln() - scale.ln();
            let cumulative_hazard = hazard.0;
            let instant_hazard = hazard.1;
            let eta = shape * log_time_ratio;
            second[0][0] = (
                shape * shape * cumulative_hazard,
                shape * shape * instant_hazard,
            );
            second[0][1] = (
                -shape * cumulative_hazard * (1.0 + eta),
                -shape * instant_hazard * (2.0 + eta),
            );
            second[1][0] = second[0][1];
            second[1][1] = (
                eta * cumulative_hazard * (1.0 + eta),
                (eta + (1.0 + eta) * (1.0 + eta)) * instant_hazard,
            );
        }
        SurvivalBaselineTarget::Gompertz => {
            let rate = cfg
                .rate
                .ok_or_else(|| "gompertz missing rate".to_string())?;
            let shape = cfg
                .shape
                .ok_or_else(|| "gompertz missing shape".to_string())?;
            second[0][0] = first[0];
            second[0][1] = first[1];
            second[1][0] = first[1];
            second[1][1] = gompertz_cumulative_shape_second_derivative(age, rate, shape);
        }
        SurvivalBaselineTarget::GompertzMakeham => {
            let rate = cfg.rate.ok_or_else(|| "gm missing rate".to_string())?;
            let shape = cfg.shape.ok_or_else(|| "gm missing shape".to_string())?;
            second[0][0] = first[0];
            second[0][1] = first[1];
            second[1][0] = first[1];
            second[1][1] = gompertz_cumulative_shape_second_derivative(age, rate, shape);
            second[2][2] = first[2];
        }
    }
    Ok(Some((hazard, first, second)))
}

#[inline]
fn gompertz_cumulative_shape_second_derivative(age: f64, rate: f64, shape: f64) -> (f64, f64) {
    let x = shape * age;
    // ∂²H_G/∂shape² = rate·[t²·E/shape − 2·(shape·t·E − (E−1))/shape³]. This is
    // a difference of O(1/shape³) terms whose leading parts cancel, so its
    // floating-point accuracy is governed by x = shape·age — and the
    // cancellation is FAR worse than the first derivative's 1/shape² form.
    // Empirically the exact branch is already garbage for |x| < ~1e-4 (e.g.
    // x=1e-9 gives a ~98% relative error; x=1e-10 a ~9700% error). The old
    // `shape < 1e-10` pivot ignored `age` and so routed those small-x cases
    // through the cancelling exact form, corrupting the marginal-slope baseline
    // Hessian near small shape. Pivot on x with a wider threshold than the
    // first derivative: the 3-term Taylor (through O(x²)) holds to <1e-8 for
    // |x| < 1e-3, and the exact branch is clean above it.
    if x.abs() < 1e-3 {
        let t = age;
        (
            rate * t * t * t * (1.0 / 3.0 + x / 4.0 + x * x / 10.0),
            rate * t * t * (1.0 + x + 0.5 * x * x),
        )
    } else {
        let e = x.exp();
        let em1 = x.exp_m1();
        let n = shape * age * e - em1;
        (
            rate * (age * age * e / shape - 2.0 * n / (shape * shape * shape)),
            rate * age * age * e,
        )
    }
}

// ---------------------------------------------------------------------------
// Baseline offsets
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum BaselineOffsetEvaluator {
    LogCumulativeHazard,
    ProbitSurvival,
}

impl BaselineOffsetEvaluator {
    fn length_error(self) -> String {
        match self {
            Self::LogCumulativeHazard => SurvivalConstructionError::IncompatibleDimensions {
                reason: "survival baseline offsets require matching entry/exit lengths".to_string(),
            }
            .into(),
            Self::ProbitSurvival => {
                "survival probit baseline offsets require matching entry/exit lengths".to_string()
            }
        }
    }

    fn finite_error(self) -> &'static str {
        match self {
            Self::LogCumulativeHazard => "non-finite survival baseline offsets computed",
            Self::ProbitSurvival => "non-finite survival probit baseline offsets computed",
        }
    }

    fn evaluate(self, age: f64, cfg: &SurvivalBaselineConfig) -> Result<(f64, f64), String> {
        match self {
            Self::LogCumulativeHazard => evaluate_survival_baseline(age, cfg),
            Self::ProbitSurvival => evaluate_survival_marginal_slope_baseline(age, cfg),
        }
    }

    fn exit_is_finite(self, value: f64, age: f64) -> bool {
        match self {
            Self::LogCumulativeHazard => {
                value.is_finite() || (age == 0.0 && value == f64::NEG_INFINITY)
            }
            Self::ProbitSurvival => value.is_finite(),
        }
    }
}

fn build_survival_offsets_with_evaluator(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
    evaluator: BaselineOffsetEvaluator,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    if age_entry.len() != age_exit.len() {
        return Err(evaluator.length_error());
    }
    let n = age_entry.len();
    // Each row's three offsets are independent across i. Compute the triplets
    // in parallel, then unpack into three Array1 outputs preserving order.
    let triples: Vec<(f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(f64, f64, f64), String> {
            // Origin-entry rows are multiplied out by the survival engines, so
            // keep their entry channel finite even when the evaluator's natural
            // value at t=0 is undefined or -inf.
            let entry_age = age_entry[i];
            let e0 = if !entry_age.is_finite() {
                return Err(SurvivalConstructionError::DataValidationFailed {
                    reason: format!("non-finite entry age at row {i}"),
                }
                .into());
            } else if entry_age <= 0.0 {
                0.0
            } else {
                evaluator.evaluate(entry_age, cfg)?.0
            };
            let exit_age = age_exit[i];
            let (e1, d1) = evaluator.evaluate(exit_age, cfg)?;
            if !e0.is_finite() || !evaluator.exit_is_finite(e1, exit_age) || !d1.is_finite() {
                return Err(SurvivalConstructionError::DataValidationFailed {
                    reason: evaluator.finite_error().to_string(),
                }
                .into());
            }
            Ok((e0, e1, d1))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut eta_entry = Array1::<f64>::zeros(n);
    let mut eta_exit = Array1::<f64>::zeros(n);
    let mut derivative_exit = Array1::<f64>::zeros(n);
    for (i, (e0, e1, d1)) in triples.into_iter().enumerate() {
        eta_entry[i] = e0;
        eta_exit[i] = e1;
        derivative_exit[i] = d1;
    }
    Ok((eta_entry, eta_exit, derivative_exit))
}

/// Compute baseline target offsets for all observations.
/// Returns `(eta_entry, eta_exit, derivative_exit)`.
pub fn build_survival_baseline_offsets(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    build_survival_offsets_with_evaluator(
        age_entry,
        age_exit,
        cfg,
        BaselineOffsetEvaluator::LogCumulativeHazard,
    )
}

/// Compute probit-survival baseline target offsets for all observations.
/// Returns `(q_entry, q_exit, q_derivative_exit)` where `Phi(-q(t)) = exp(-H0(t))`.
pub fn build_survival_marginal_slope_baseline_offsets(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    build_survival_offsets_with_evaluator(
        age_entry,
        age_exit,
        cfg,
        BaselineOffsetEvaluator::ProbitSurvival,
    )
}

pub fn location_scale_uses_probit_survival_baseline(inverse_link: Option<&InverseLink>) -> bool {
    matches!(
        inverse_link,
        Some(
            InverseLink::Standard(StandardLink::Probit)
                | InverseLink::LatentCLogLog(_)
                | InverseLink::Sas(_)
                | InverseLink::BetaLogistic(_)
                | InverseLink::Mixture(_)
        )
    )
}

pub fn survival_derivative_guard_for_likelihood(likelihood_mode: SurvivalLikelihoodMode) -> f64 {
    match likelihood_mode {
        SurvivalLikelihoodMode::LocationScale
        | SurvivalLikelihoodMode::Latent
        | SurvivalLikelihoodMode::LatentBinary => DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
        SurvivalLikelihoodMode::MarginalSlope => DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
        SurvivalLikelihoodMode::Transformation | SurvivalLikelihoodMode::Weibull => 0.0,
    }
}

pub fn build_survival_time_offsets_for_likelihood(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline_cfg: &SurvivalBaselineConfig,
    likelihood_mode: SurvivalLikelihoodMode,
    inverse_link: Option<&InverseLink>,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    if likelihood_mode == SurvivalLikelihoodMode::MarginalSlope
        || (likelihood_mode == SurvivalLikelihoodMode::LocationScale
            && location_scale_uses_probit_survival_baseline(inverse_link))
    {
        build_survival_marginal_slope_baseline_offsets(age_entry, age_exit, baseline_cfg)
    } else {
        build_survival_baseline_offsets(age_entry, age_exit, baseline_cfg)
    }
}

pub fn add_survival_time_derivative_guard_offset(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    anchor_time: f64,
    derivative_guard: f64,
    eta_offset_entry: &mut Array1<f64>,
    eta_offset_exit: &mut Array1<f64>,
    derivative_offset_exit: &mut Array1<f64>,
) -> Result<(), String> {
    if derivative_guard <= 0.0 {
        return Ok(());
    }
    let n = age_entry.len();
    if age_exit.len() != n
        || eta_offset_entry.len() != n
        || eta_offset_exit.len() != n
        || derivative_offset_exit.len() != n
    {
        return Err(SurvivalConstructionError::IncompatibleDimensions {
            reason: "survival derivative-guard offset lengths must match".to_string(),
        }
        .into());
    }
    for i in 0..n {
        eta_offset_entry[i] += derivative_guard * (age_entry[i] - anchor_time);
        eta_offset_exit[i] += derivative_guard * (age_exit[i] - anchor_time);
        derivative_offset_exit[i] += derivative_guard;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct LatentSurvivalBaselineOffsets {
    pub loaded_eta_entry: Array1<f64>,
    pub loaded_eta_exit: Array1<f64>,
    pub loaded_derivative_exit: Array1<f64>,
    pub unloaded_mass_entry: Array1<f64>,
    pub unloaded_mass_exit: Array1<f64>,
    pub unloaded_hazard_exit: Array1<f64>,
}

pub fn build_latent_survival_baseline_offsets(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
    loading: HazardLoading,
) -> Result<LatentSurvivalBaselineOffsets, String> {
    if age_entry.len() != age_exit.len() {
        return Err(
            "latent survival baseline offsets require matching entry/exit lengths".to_string(),
        );
    }

    fn gompertz_components(age: f64, rate: f64, shape: f64) -> (f64, f64) {
        if shape.abs() < 1e-10 {
            // Taylor at shape=0 matching `gompertz_hazard_components`:
            //   H_G(t) = rate·t·(1 + (shape·t)/2 + (shape·t)²/6)
            //   h_G(t) = rate·(1 + shape·t + (shape·t)²/2)
            // Dropping the higher-order `shape*t` corrections silently
            // diverges this helper from its sibling for non-zero shape near
            // the cutoff and gives inconsistent loaded-vs-unloaded offsets.
            let x = shape * age;
            return (
                rate * age * (1.0 + 0.5 * x + x * x / 6.0),
                rate * (1.0 + x + 0.5 * x * x),
            );
        }
        let shape_age = shape * age;
        let cumulative_hazard = (rate / shape) * shape_age.exp_m1();
        let instant_hazard = rate * shape_age.exp();
        (cumulative_hazard, instant_hazard)
    }

    let n = age_entry.len();

    // Per-row 6-tuple is independent. Evaluate in parallel into a Vec and then
    // unpack into the six Array1 outputs in original order.
    let rows: Vec<[f64; 6]> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<[f64; 6], String> {
            let entry = age_entry[i];
            let exit = age_exit[i];
            if !entry.is_finite()
                || !exit.is_finite()
                || entry <= 0.0
                || exit <= 0.0
                || exit < entry
            {
                return Err(format!(
                    "latent survival baseline offsets require finite positive entry/exit ages with exit >= entry (row {})",
                    i + 1
                ));
            }
            match loading {
                HazardLoading::Full => {
                    let (eta_entry, _) = evaluate_survival_baseline(entry, cfg)?;
                    let (eta_exit, derivative_exit) = evaluate_survival_baseline(exit, cfg)?;
                    Ok([eta_entry, eta_exit, derivative_exit, 0.0, 0.0, 0.0])
                }
                HazardLoading::LoadedVsUnloaded => {
                    if cfg.target != SurvivalBaselineTarget::GompertzMakeham {
                        return Err(format!(
                            "HazardLoading::LoadedVsUnloaded requires --baseline-target gompertz-makeham, got {}",
                            survival_baseline_targetname(cfg.target)
                        ));
                    }
                    let rate = cfg.rate.ok_or_else(|| {
                        "gompertz-makeham latent survival is missing baseline rate".to_string()
                    })?;
                    let shape = cfg.shape.ok_or_else(|| {
                        "gompertz-makeham latent survival is missing baseline shape".to_string()
                    })?;
                    let makeham = cfg.makeham.ok_or_else(|| {
                        "gompertz-makeham latent survival is missing baseline makeham".to_string()
                    })?;
                    let (loaded_entry, _) = gompertz_components(entry, rate, shape);
                    let (loaded_exit, loaded_hazard) = gompertz_components(exit, rate, shape);
                    if !(loaded_entry.is_finite()
                        && loaded_entry > 0.0
                        && loaded_exit.is_finite()
                        && loaded_exit > 0.0
                        && loaded_hazard.is_finite()
                        && loaded_hazard > 0.0)
                    {
                        return Err(format!(
                            "gompertz-makeham latent loaded component produced a non-positive or non-finite hazard decomposition at row {}",
                            i + 1
                        ));
                    }
                    Ok([
                        loaded_entry.ln(),
                        loaded_exit.ln(),
                        loaded_hazard / loaded_exit,
                        makeham * entry,
                        makeham * exit,
                        makeham,
                    ])
                }
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut loaded_eta_entry = Array1::<f64>::zeros(n);
    let mut loaded_eta_exit = Array1::<f64>::zeros(n);
    let mut loaded_derivative_exit = Array1::<f64>::zeros(n);
    let mut unloaded_mass_entry = Array1::<f64>::zeros(n);
    let mut unloaded_mass_exit = Array1::<f64>::zeros(n);
    let mut unloaded_hazard_exit = Array1::<f64>::zeros(n);
    for (i, row) in rows.into_iter().enumerate() {
        loaded_eta_entry[i] = row[0];
        loaded_eta_exit[i] = row[1];
        loaded_derivative_exit[i] = row[2];
        unloaded_mass_entry[i] = row[3];
        unloaded_mass_exit[i] = row[4];
        unloaded_hazard_exit[i] = row[5];
    }

    Ok(LatentSurvivalBaselineOffsets {
        loaded_eta_entry,
        loaded_eta_exit,
        loaded_derivative_exit,
        unloaded_mass_entry,
        unloaded_mass_exit,
        unloaded_hazard_exit,
    })
}

// ---------------------------------------------------------------------------
// Time wiggle construction
// ---------------------------------------------------------------------------

pub fn build_survival_timewiggle_derivative_design(
    eta_exit: &Array1<f64>,
    derivative_exit: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    let mut design_derivative_exit =
        monotone_wiggle_basis_with_derivative_order(eta_exit.view(), knots, degree, 1)?;
    for i in 0..design_derivative_exit.nrows() {
        let chain = derivative_exit[i];
        for j in 0..design_derivative_exit.ncols() {
            design_derivative_exit[[i, j]] *= chain;
        }
    }
    Ok(design_derivative_exit)
}

/// Build the dynamic "baseline as prior" timewiggle runtime.
///
/// The baseline offsets are used only to initialize the wiggle knot placement
/// on a stable scalar scale.  The exact survival family evaluates the resulting
/// monotone wiggle dynamically on the current time predictor h0(t):
///
///   h(t) = g(h0(t)),   g(z) = z + w(z).
///
/// No fixed `B(eta_baseline)` design is constructed here.
pub fn build_survival_timewiggle_from_baseline(
    eta_entry: &Array1<f64>,
    eta_exit: &Array1<f64>,
    derivative_exit: &Array1<f64>,
    cfg: &LinkWiggleFormulaSpec,
) -> Result<SurvivalTimeWiggleBuild, String> {
    if eta_entry.len() != eta_exit.len() || eta_exit.len() != derivative_exit.len() {
        return Err(
            "baseline-timewiggle requires matching entry/exit/derivative lengths".to_string(),
        );
    }
    // Guard: if baseline offsets are all zero (linear baseline), the timewiggle
    // construction is degenerate — it adds only a constant, not time-varying structure.
    let all_zero = eta_entry.iter().all(|&v| v.abs() < 1e-15)
        && eta_exit.iter().all(|&v| v.abs() < 1e-15)
        && derivative_exit.iter().all(|&v| v.abs() < 1e-15);
    if all_zero {
        return Err(
            "timewiggle requires a non-linear scalar survival baseline target; \
             the provided baseline offsets are all zero (linear baseline)"
                .to_string(),
        );
    }
    let n = eta_exit.len();
    let mut seed = Array1::<f64>::zeros(2 * n);
    for i in 0..n {
        seed[i] = eta_entry[i];
        seed[n + i] = eta_exit[i];
    }
    // Use the smallest requested positive penalty order as the primary
    // coefficient-space penalty so the fitted wiggle penalty system matches
    // the public formula exactly, including the slope (`order = 1`) case.
    let (primary_order, extra_orders) = split_wiggle_penalty_orders(2, &cfg.penalty_orders);
    let wiggle_cfg = WiggleBlockConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_order: primary_order,
        double_penalty: cfg.double_penalty,
    };
    let (mut combined_block, knots) = buildwiggle_block_input_from_seed(seed.view(), &wiggle_cfg)?;
    append_selected_wiggle_penalty_orders(&mut combined_block, &extra_orders)?;
    let ncols = combined_block.design.ncols();
    Ok(SurvivalTimeWiggleBuild {
        nullspace_dims: combined_block.nullspace_dims.clone(),
        penalties: {
            combined_block
                .penalties
                .into_iter()
                .map(|ps| ps.to_global(ncols))
                .collect()
        },
        knots,
        degree: cfg.degree,
        ncols,
    })
}

pub fn append_zero_tail_columns(
    x_entry: &mut DesignMatrix,
    x_exit: &mut DesignMatrix,
    x_derivative: &mut DesignMatrix,
    tail_cols: usize,
) {
    if tail_cols == 0 {
        return;
    }
    // Wiggle tail columns are dense, so materialize everything to dense.
    // This only runs once at construction time when time-wiggles are active.
    fn append_dense(dm: &mut DesignMatrix, tail: usize) {
        let old = dm.to_dense();
        let n = old.nrows();
        let p_base = old.ncols();
        let mut out = Array2::<f64>::zeros((n, p_base + tail));
        out.slice_mut(s![.., 0..p_base]).assign(&old);
        *dm = DesignMatrix::Dense(DenseDesignMatrix::from(out));
    }
    append_dense(x_entry, tail_cols);
    append_dense(x_exit, tail_cols);
    append_dense(x_derivative, tail_cols);
}

// ---------------------------------------------------------------------------
// Resolved config (from build output back to config for serialization)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Time-varying covariate template
// ---------------------------------------------------------------------------

/// Build a time-varying covariate block by tensoring the covariate design
/// with a 1D B-spline basis on log(time).
pub fn build_time_varying_survival_covariate_template(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    time_k: usize,
    time_degree: usize,
    block_name: &str,
) -> Result<SurvivalCovariateTermBlockTemplate, String> {
    if time_k < time_degree + 1 {
        return Err(format!(
            "--{block_name}-time-k must be >= degree + 1 = {}, got {time_k}",
            time_degree + 1
        ));
    }
    let num_internal_knots = time_k - (time_degree + 1);

    let log_entry = age_entry.mapv(|t| t.max(1e-12).ln());
    let log_exit = age_exit.mapv(|t| t.max(1e-12).ln());

    let time_spec = BSplineBasisSpec {
        degree: time_degree,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(num_internal_knots),
            placement: crate::basis::BSplineKnotPlacement::Quantile,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let time_build = build_bspline_basis_1d(log_exit.view(), &time_spec)
        .map_err(|e| format!("failed to build {block_name} time-margin B-spline basis: {e}"))?;
    let time_design_exit = time_build.design.to_dense();

    let knots = match &time_build.metadata {
        BasisMetadata::BSpline1D { knots, .. } => knots.clone(),
        _ => {
            return Err(format!(
                "{block_name} time-margin basis returned unexpected metadata type"
            ));
        }
    };

    let time_build_entry = build_bspline_basis_1d(
        log_entry.view(),
        &BSplineBasisSpec {
            degree: time_degree,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Provided(knots.clone()),
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: BSplineBoundaryConditions::default(),
        },
    )
    .map_err(|e| format!("failed to evaluate {block_name} time-margin basis at entry: {e}"))?;
    let time_design_entry = time_build_entry.design.to_dense();
    let p_time = time_design_exit.ncols();
    let mut time_design_derivative_exit = Array2::<f64>::zeros((age_exit.len(), p_time));
    // Per-row derivative-basis evaluation is independent; each row owns its
    // own small `deriv_buf`. par_chunks_mut over the (n × p_time) output rows
    // hands disjoint mutable row-slices to rayon workers.
    time_design_derivative_exit
        .as_slice_mut()
        .expect("zeros are contiguous")
        .par_chunks_mut(p_time)
        .enumerate()
        .try_for_each(|(i, row_out)| -> Result<(), String> {
            let mut deriv_buf = vec![0.0_f64; p_time];
            evaluate_bspline_derivative_scalar(
                log_exit[i],
                knots.view(),
                time_degree,
                &mut deriv_buf,
            )
            .map_err(|e| {
                format!("failed to evaluate {block_name} time-margin derivative basis: {e}")
            })?;
            let chain = 1.0 / age_exit[i].max(1e-12);
            for j in 0..p_time {
                row_out[j] = deriv_buf[j] * chain;
            }
            Ok(())
        })?;

    Ok(SurvivalCovariateTermBlockTemplate::TimeVarying {
        time_basis_entry: time_design_entry,
        time_basis_exit: time_design_exit,
        time_basis_derivative_exit: time_design_derivative_exit,
        time_penalties: time_build.penalties,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalTimeBasisConfig,
        baseline_chain_rule_gradient, baseline_offset_theta_partials,
        build_survival_marginal_slope_baseline_offsets, build_survival_time_basis,
        build_survival_timewiggle_from_baseline, evaluate_survival_baseline,
        evaluate_survival_marginal_slope_baseline, gompertz_cumulative_shape_derivative,
        gompertz_cumulative_shape_second_derivative, gompertz_hazard_components,
        marginal_slope_baseline_chain_rule_gradient, marginal_slope_baseline_chain_rule_hessian,
        marginal_slope_baseline_offset_theta_partials,
        optimize_survival_baseline_config_with_gradient,
        optimize_survival_baseline_config_with_gradient_only,
        resolve_survival_marginal_slope_time_anchor_value, survival_baseline_config_from_theta,
        survival_baseline_theta_from_config,
    };
    use crate::families::survival::{OffsetChannelCurvatures, OffsetChannelResiduals};
    use crate::inference::formula_dsl::LinkWiggleFormulaSpec;
    use crate::probability::normal_cdf;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn survival_timewiggle_keeps_requested_order_one_penalty() {
        let eta_entry = array![0.1, 0.3, 0.5, 0.8];
        let eta_exit = array![0.4, 0.7, 1.0, 1.4];
        let derivative_exit = array![0.9, 1.1, 1.2, 1.3];
        let cfg = LinkWiggleFormulaSpec {
            degree: 3,
            num_internal_knots: 4,
            penalty_orders: vec![1, 2, 3],
            double_penalty: false,
        };

        let build =
            build_survival_timewiggle_from_baseline(&eta_entry, &eta_exit, &derivative_exit, &cfg)
                .expect("build survival timewiggle");

        assert_eq!(build.penalties.len(), 3);
        assert_eq!(build.nullspace_dims, vec![1, 2, 3]);
        assert!(build.ncols > 0);
    }

    #[test]
    fn marginal_slope_time_anchor_defaults_to_median_exit() {
        let age_entry = array![9.0, 1.0, 4.0, 6.0];
        let age_exit = array![20.0, 12.0, 18.0, 30.0];
        let anchor = resolve_survival_marginal_slope_time_anchor_value(&age_entry, &age_exit, None)
            .expect("resolve marginal-slope default time anchor");

        assert!(
            (anchor - 19.0).abs() <= 1e-12,
            "marginal-slope default anchor should be median exit, got {anchor}"
        );
    }

    #[test]
    fn marginal_slope_time_anchor_honors_explicit_value() {
        let age_entry = array![9.0, 1.0, 4.0, 6.0];
        let age_exit = array![20.0, 12.0, 18.0, 30.0];
        let anchor =
            resolve_survival_marginal_slope_time_anchor_value(&age_entry, &age_exit, Some(7.5))
                .expect("resolve explicit marginal-slope time anchor");

        assert!(
            (anchor - 7.5).abs() <= 1e-12,
            "explicit marginal-slope anchor should round-trip, got {anchor}"
        );
    }

    /// Derivative-contract parity for the two public baseline optimizers.
    ///
    /// After the unification onto `run_baseline_theta_optimizer`, the
    /// gradient-only and gradient+Hessian entry points differ *only* in how
    /// much derivative information they hand the outer solver — not in the
    /// surface they minimize. We exercise that invariant on a known
    /// strictly-convex quadratic in θ-space (Weibull baseline: θ = (ln scale,
    /// ln shape)) whose unique minimizer is `theta_star`, supplying the same
    /// objective as `(f, ∇f)` and as `(f, ∇f, ∇²f)`. Both contracts must
    /// recover the same minimizer config, not weakened to pass.
    #[test]
    fn baseline_optimizer_contracts_agree_on_shared_surface() {
        // SPD curvature and interior minimizer in θ-space. A is well away from
        // singular so both the analytic-Hessian and BFGS paths see the same
        // unambiguous bowl; θ* sits comfortably inside the ±6 box around the
        // θ=(0,0) seed below.
        let curvature: Array2<f64> = array![[3.0, 0.5], [0.5, 2.0]];
        let theta_star: Array1<f64> = array![2.5_f64.ln(), 1.3_f64.ln()];

        // Seed config at θ=(0,0) (scale=shape=1). The Linear early-return path
        // is not exercised here; Weibull has a genuine 2-dim θ to optimize.
        let initial = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(1.0),
            shape: Some(1.0),
            rate: None,
            makeham: None,
        };

        // θ recovered from a returned Weibull config, via the exact inverse of
        // the config→θ map the optimizers use internally.
        let recovered_theta = |cfg: &SurvivalBaselineConfig| -> Array1<f64> {
            survival_baseline_theta_from_config(cfg)
                .expect("config→θ")
                .expect("Weibull config has a θ")
        };

        // Shared quadratic surface, evaluated by mapping config→θ so every
        // contract sees the identical objective.
        let curvature_cost = curvature.clone();
        let star_cost = theta_star.clone();
        let cost_at = move |cfg: &SurvivalBaselineConfig| -> Result<f64, String> {
            let theta = survival_baseline_theta_from_config(cfg)?
                .ok_or_else(|| "expected a θ for the cost surface".to_string())?;
            let d = &theta - &star_cost;
            let ad = curvature_cost.dot(&d);
            Ok(0.5 * d.dot(&ad))
        };

        let curvature_grad = curvature.clone();
        let star_grad = theta_star.clone();
        let cost_for_grad = cost_at.clone();
        let result_grad_only = optimize_survival_baseline_config_with_gradient_only(
            &initial,
            "baseline parity (gradient-only)",
            move |cfg| {
                let cost = cost_for_grad(cfg)?;
                let theta = survival_baseline_theta_from_config(cfg)?
                    .ok_or_else(|| "expected a θ for the gradient".to_string())?;
                let gradient = curvature_grad.dot(&(&theta - &star_grad));
                Ok((cost, gradient))
            },
        )
        .expect("gradient-only baseline optimization converges");

        let curvature_hess = curvature.clone();
        let star_hess = theta_star.clone();
        let cost_for_hess = cost_at.clone();
        let result_grad_hess = optimize_survival_baseline_config_with_gradient(
            &initial,
            "baseline parity (gradient+Hessian)",
            move |cfg| {
                let cost = cost_for_hess(cfg)?;
                let theta = survival_baseline_theta_from_config(cfg)?
                    .ok_or_else(|| "expected a θ for the gradient".to_string())?;
                let gradient = curvature_hess.dot(&(&theta - &star_hess));
                Ok((cost, gradient, curvature_hess.clone()))
            },
        )
        .expect("gradient+Hessian baseline optimization converges");

        let theta_grad_only = recovered_theta(&result_grad_only);
        let theta_grad_hess = recovered_theta(&result_grad_hess);

        // Each contract recovers the true minimizer. 2e-3 is a safe,
        // un-weakened bound; both gradient paths land far tighter.
        for (label, theta) in [
            ("gradient-only", &theta_grad_only),
            ("gradient+Hessian", &theta_grad_hess),
        ] {
            let err = (theta - &theta_star)
                .mapv(f64::abs)
                .fold(0.0_f64, |a, &v| a.max(v));
            assert!(
                err <= 2e-3,
                "{label} contract recovered θ {theta:?} off true minimizer {theta_star:?} by {err:e}"
            );
        }

        // Cross-contract agreement: the three results must coincide, since the
        // only difference between the entry points is the derivative contract,
        // never the surface they minimize.
        let pairwise_max = |a: &Array1<f64>, b: &Array1<f64>| -> f64 {
            (a - b).mapv(f64::abs).fold(0.0_f64, |acc, &v| acc.max(v))
        };
        assert!(
            pairwise_max(&theta_grad_only, &theta_grad_hess) <= 2e-3,
            "gradient-only vs gradient+Hessian disagree: {theta_grad_only:?} vs {theta_grad_hess:?}"
        );
    }

    #[test]
    fn automatic_ispline_time_knots_are_sized_for_antiderivative_degree() {
        let age_entry = array![1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0];
        let age_exit = array![2.0_f64, 3.0, 5.0, 8.0, 13.0, 21.0];
        let requested_degree = 3;
        let num_internal_knots = 1;

        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: requested_degree,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            Some((num_internal_knots, 1e-2)),
        )
        .expect("automatic cubic ispline with one interior knot builds");

        let working_degree = requested_degree + 1;
        let knots = built.knots.expect("resolved ispline knots");
        assert_eq!(
            knots.len(),
            num_internal_knots + 2 * (working_degree + 1),
            "I-spline automatic knots must be clamped for the working B-spline degree"
        );
        assert_eq!(built.degree, Some(requested_degree));
        assert!(built.x_exit_time.ncols() > 0);
        assert_eq!(built.x_entry_time.ncols(), built.x_exit_time.ncols());
        assert_eq!(built.x_derivative_time.ncols(), built.x_exit_time.ncols());
    }

    #[test]
    fn ispline_time_derivative_is_nonzero_at_right_boundary() {
        let age_entry = array![1.0_f64, 1.0, 1.0];
        let age_exit = array![4.0_f64, 4.0, 4.0];
        let left = 1.0_f64.ln();
        let right = 4.0_f64.ln();
        let mid = left + 0.5 * (right - left);
        let knots = array![left, left, left, left, mid, right, right, right, right];

        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 2,
                knots,
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build right-boundary ispline time basis");

        let derivative = built.x_derivative_time.as_dense_cow();
        let max_abs = derivative.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            max_abs > 1e-8,
            "right-boundary I-spline derivative must use the left-hand endpoint slope"
        );
        for row in derivative.rows() {
            assert!(
                row.iter().any(|v| *v > 1e-8),
                "each row at the right boundary needs a positive hazard derivative"
            );
        }
    }

    #[test]
    fn marginal_slope_baseline_maps_gompertz_makeham_survival_to_probit_index() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.07),
            rate: Some(0.012),
            makeham: Some(0.003),
        };
        let age = 11.5;
        let (q, q_derivative) = evaluate_survival_marginal_slope_baseline(age, &cfg)
            .expect("evaluate marginal-slope gompertz-makeham baseline");
        let shape = cfg.shape.expect("shape");
        let rate = cfg.rate.expect("rate");
        let makeham = cfg.makeham.expect("makeham");
        let cumulative_hazard = makeham * age + (rate / shape) * ((shape * age).exp() - 1.0);
        let instant_hazard = makeham + rate * (shape * age).exp();
        let expected_survival = (-cumulative_hazard).exp();
        let actual_survival = normal_cdf(-q);
        assert!((actual_survival - expected_survival).abs() <= 1e-12);

        let h = 1e-5;
        let q_plus = evaluate_survival_marginal_slope_baseline(age + h, &cfg)
            .expect("q plus")
            .0;
        let q_minus = evaluate_survival_marginal_slope_baseline(age - h, &cfg)
            .expect("q minus")
            .0;
        let fd = (q_plus - q_minus) / (2.0 * h);
        assert!((q_derivative - fd).abs() <= 1e-7);
        assert!(instant_hazard > 0.0);
    }

    #[test]
    fn marginal_slope_baseline_is_evaluable_at_the_survival_curve_origin() {
        // Regression for #1024: the probit/marginal-slope baseline evaluator must
        // be defined at the survival-curve origin t = 0 (where S0(0) = 1, so the
        // probit index q(0) = -Phi^{-1}(1) = -inf and there is no finite offset),
        // exactly like its log-cumulative-hazard sibling `evaluate_survival_baseline`.
        // Before the fix the shared `age <= 0` hazard guard aborted, so a survival
        // prediction grid whose first node is the origin (the `Surv(time, event)`
        // right-censored shorthand) could not be evaluated for the location-scale /
        // marginal-slope likelihoods.
        let configs = [
            SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Linear,
                scale: None,
                shape: None,
                rate: None,
                makeham: None,
            },
            SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Weibull,
                scale: Some(2.5),
                shape: Some(1.3),
                rate: None,
                makeham: None,
            },
            SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Gompertz,
                scale: None,
                shape: Some(0.05),
                rate: Some(0.01),
                makeham: None,
            },
            SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::GompertzMakeham,
                scale: None,
                shape: Some(0.07),
                rate: Some(0.012),
                makeham: Some(0.003),
            },
        ];
        for cfg in &configs {
            // The probit baseline returns a finite zero offset at the origin for
            // every target (the survival surface anchors S(0) = 1 directly).
            let (q0, q0_derivative) = evaluate_survival_marginal_slope_baseline(0.0, cfg)
                .expect("marginal-slope baseline must be evaluable at the origin");
            assert_eq!(q0, 0.0);
            assert_eq!(q0_derivative, 0.0);

            // The log-cumulative-hazard sibling is likewise finite at the origin —
            // this parity is the whole point (the transformation likelihood already
            // worked because it rides this evaluator).
            let (eta0, eta0_derivative) =
                evaluate_survival_baseline(0.0, cfg).expect("log-cum-hazard baseline at origin");
            assert!(eta0_derivative.is_finite());
            assert!(eta0.is_finite() || eta0 == f64::NEG_INFINITY);

            // The batched offset builder must not abort when a query exit age is the
            // origin (this is the exact call the location-scale predict path makes on
            // the default surface grid). Entry stays at the origin, exit spans 0 -> t.
            let age_entry = array![0.0, 0.0];
            let age_exit = array![0.0, 1.5];
            let (entry, exit, derivative) =
                build_survival_marginal_slope_baseline_offsets(&age_entry, &age_exit, cfg)
                    .expect("probit baseline offsets must build through the origin");
            assert!(entry.iter().all(|v| v.is_finite()));
            assert!(exit.iter().all(|v| v.is_finite()));
            assert!(derivative.iter().all(|v| v.is_finite()));
            // The origin exit column carries no probit offset.
            assert_eq!(exit[0], 0.0);
        }
    }

    #[test]
    fn marginal_slope_baseline_offsets_use_true_gompertz_makeham_survival() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.03),
            rate: Some(0.01),
            makeham: Some(0.002),
        };
        let age_entry = array![2.0, 4.0];
        let age_exit = array![5.0, 9.0];
        let (entry, exit, derivative) =
            build_survival_marginal_slope_baseline_offsets(&age_entry, &age_exit, &cfg)
                .expect("marginal-slope baseline offsets");
        for i in 0..age_entry.len() {
            let entry_h = cfg.makeham.expect("makeham") * age_entry[i]
                + (cfg.rate.expect("rate") / cfg.shape.expect("shape"))
                    * ((cfg.shape.expect("shape") * age_entry[i]).exp() - 1.0);
            let exit_h = cfg.makeham.expect("makeham") * age_exit[i]
                + (cfg.rate.expect("rate") / cfg.shape.expect("shape"))
                    * ((cfg.shape.expect("shape") * age_exit[i]).exp() - 1.0);
            assert!((normal_cdf(-entry[i]) - (-entry_h).exp()).abs() <= 1e-12);
            assert!((normal_cdf(-exit[i]) - (-exit_h).exp()).abs() <= 1e-12);
            assert!(derivative[i].is_finite() && derivative[i] > 0.0);
        }
    }

    fn fd_marginal_slope_baseline_offset(
        age: f64,
        cfg: &SurvivalBaselineConfig,
        steps: &[f64],
    ) -> Vec<(f64, f64)> {
        let theta = survival_baseline_theta_from_config(cfg)
            .expect("theta")
            .expect("non-linear baseline");
        assert_eq!(
            steps.len(),
            theta.len(),
            "fd_marginal_slope_baseline_offset: step vector length must match θ dimension"
        );
        (0..theta.len())
            .map(|k| {
                let h = steps[k];
                let mut theta_plus = theta.clone();
                theta_plus[k] += h;
                let mut theta_minus = theta.clone();
                theta_minus[k] -= h;
                let cfg_plus =
                    survival_baseline_config_from_theta(cfg.target, &theta_plus).expect("plus cfg");
                let cfg_minus = survival_baseline_config_from_theta(cfg.target, &theta_minus)
                    .expect("minus cfg");
                let (q_p, qt_p) =
                    evaluate_survival_marginal_slope_baseline(age, &cfg_plus).expect("q+");
                let (q_m, qt_m) =
                    evaluate_survival_marginal_slope_baseline(age, &cfg_minus).expect("q-");
                ((q_p - q_m) / (2.0 * h), (qt_p - qt_m) / (2.0 * h))
            })
            .collect()
    }

    #[test]
    fn marginal_slope_baseline_theta_partials_match_fd_for_gompertz_makeham() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.04),
            rate: Some(0.013),
            makeham: Some(0.002),
        };
        let age = 17.0;
        let analytic = marginal_slope_baseline_offset_theta_partials(age, &cfg)
            .expect("partials")
            .expect("nonlinear");
        let fd = fd_marginal_slope_baseline_offset(age, &cfg, &[1e-5, 1e-5, 1e-5]);
        assert_eq!(analytic.len(), fd.len());
        for (k, ((aq, aqt), (fq, fqt))) in analytic.iter().zip(fd.iter()).enumerate() {
            assert_close(*aq, *fq, 1e-6, &format!("gm-probit q theta[{k}]"));
            assert_close(*aqt, *fqt, 1e-6, &format!("gm-probit q' theta[{k}]"));
        }
    }

    #[test]
    fn marginal_slope_baseline_theta_partials_match_fd_near_zero_gompertz_shape() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(1e-14),
            rate: Some(0.013),
            makeham: Some(0.002),
        };
        let age = 17.0;
        let analytic = marginal_slope_baseline_offset_theta_partials(age, &cfg)
            .expect("partials")
            .expect("nonlinear");
        let fd = fd_marginal_slope_baseline_offset(age, &cfg, &[1e-5, 1e-11, 1e-5]);
        assert_eq!(analytic.len(), fd.len());
        for (k, ((aq, aqt), (fq, fqt))) in analytic.iter().zip(fd.iter()).enumerate() {
            assert_close(*aq, *fq, 1e-5, &format!("near-zero gm-probit q theta[{k}]"));
            assert_close(
                *aqt,
                *fqt,
                1e-5,
                &format!("near-zero gm-probit q' theta[{k}]"),
            );
        }
    }

    fn shifted_quadratic_offset_residuals(
        age_entry: ndarray::ArrayView1<'_, f64>,
        age_exit: ndarray::ArrayView1<'_, f64>,
        base_cfg: &SurvivalBaselineConfig,
        candidate_cfg: &SurvivalBaselineConfig,
        base: &OffsetChannelResiduals,
        curvatures: &OffsetChannelCurvatures,
    ) -> OffsetChannelResiduals {
        let n = age_exit.len();
        let mut entry = base.entry.clone();
        let mut exit = base.exit.clone();
        let mut derivative = base.derivative.clone();
        for row in 0..n {
            let (_, base_exit, base_deriv) =
                baseline_marginal_slope_channels(age_exit[row], base_cfg);
            let (_, cand_exit, cand_deriv) =
                baseline_marginal_slope_channels(age_exit[row], candidate_cfg);
            let base_entry = if base.entry[row] == 0.0 {
                0.0
            } else {
                baseline_marginal_slope_channels(age_entry[row], base_cfg).1
            };
            let cand_entry = if base.entry[row] == 0.0 {
                0.0
            } else {
                baseline_marginal_slope_channels(age_entry[row], candidate_cfg).1
            };
            let delta = [
                cand_entry - base_entry,
                cand_exit - base_exit,
                cand_deriv - base_deriv,
            ];
            let mut shift = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    shift[i] += curvatures.rows[row][i][j] * delta[j];
                }
            }
            if base.entry[row] != 0.0 {
                entry[row] += shift[0];
            }
            exit[row] += shift[1];
            derivative[row] += shift[2];
        }
        OffsetChannelResiduals {
            entry,
            exit,
            derivative,
            right: base.right.clone(),
        }
    }

    fn baseline_marginal_slope_channels(age: f64, cfg: &SurvivalBaselineConfig) -> (f64, f64, f64) {
        let (q, q_t) = evaluate_survival_marginal_slope_baseline(age, cfg).expect("baseline");
        (q, q, q_t)
    }

    #[test]
    fn marginal_slope_baseline_chain_rule_hessian_matches_fd_gradient() {
        assert!(file!().ends_with(".rs"));
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.025),
            rate: Some(0.012),
            makeham: Some(0.003),
        };
        let theta = survival_baseline_theta_from_config(&cfg)
            .expect("theta")
            .expect("nonlinear");
        let age_entry = array![2.5, 0.0, 5.0];
        let age_exit = array![7.5, 11.0, 15.0];
        let base_residuals = OffsetChannelResiduals {
            entry: array![0.2, 0.0, -0.1],
            exit: array![0.6, -0.3, 0.4],
            derivative: array![-0.5, 0.25, 0.15],
            right: Array1::<f64>::zeros(3),
        };
        let curvatures = OffsetChannelCurvatures {
            rows: vec![
                [[1.4, 0.2, -0.1], [0.2, 1.1, 0.05], [-0.1, 0.05, 0.7]],
                [[0.9, -0.15, 0.0], [-0.15, 1.3, 0.12], [0.0, 0.12, 0.8]],
                [[1.2, 0.05, 0.09], [0.05, 0.95, -0.04], [0.09, -0.04, 0.6]],
            ],
        };
        let analytic = marginal_slope_baseline_chain_rule_hessian(
            age_entry.view(),
            age_exit.view(),
            &cfg,
            &base_residuals,
            &curvatures,
        )
        .expect("hessian")
        .expect("nonlinear");

        let gradient_at = |theta_candidate: &Array1<f64>| -> Array1<f64> {
            let candidate = survival_baseline_config_from_theta(cfg.target, theta_candidate)
                .expect("candidate cfg");
            let residuals = shifted_quadratic_offset_residuals(
                age_entry.view(),
                age_exit.view(),
                &cfg,
                &candidate,
                &base_residuals,
                &curvatures,
            );
            marginal_slope_baseline_chain_rule_gradient(
                age_entry.view(),
                age_exit.view(),
                &candidate,
                &residuals,
            )
            .expect("gradient")
            .expect("nonlinear")
        };

        for j in 0..theta.len() {
            let step = if j == 1 { 2e-5 } else { 1e-5 };
            let mut plus = theta.clone();
            plus[j] += step;
            let mut minus = theta.clone();
            minus[j] -= step;
            let fd_col = (&gradient_at(&plus) - &gradient_at(&minus)) / (2.0 * step);
            for i in 0..theta.len() {
                assert_close(
                    analytic[[i, j]],
                    fd_col[i],
                    2e-5,
                    &format!("baseline Hessian ({i},{j})"),
                );
            }
        }
    }

    #[test]
    fn marginal_slope_baseline_chain_rule_gradient_contracts_probit_partials() {
        assert!(file!().ends_with(".rs"));
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.03),
            rate: Some(0.01),
            makeham: Some(0.002),
        };
        let age_entry = array![3.0, 6.0];
        let age_exit = array![8.0, 12.0];
        let residuals = OffsetChannelResiduals {
            exit: array![0.7, -0.2],
            entry: array![0.1, 0.4],
            derivative: array![1.3, -0.6],
            right: Array1::<f64>::zeros(2),
        };
        let grad = marginal_slope_baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("gradient")
        .expect("nonlinear");

        let mut expected = Array1::<f64>::zeros(3);
        for i in 0..age_exit.len() {
            let exit_partials = marginal_slope_baseline_offset_theta_partials(age_exit[i], &cfg)
                .expect("exit partials")
                .expect("nonlinear");
            let entry_partials = marginal_slope_baseline_offset_theta_partials(age_entry[i], &cfg)
                .expect("entry partials")
                .expect("nonlinear");
            for k in 0..3 {
                expected[k] += residuals.exit[i] * exit_partials[k].0
                    + residuals.derivative[i] * exit_partials[k].1
                    + residuals.entry[i] * entry_partials[k].0;
            }
        }
        for k in 0..3 {
            assert_close(
                grad[k],
                expected[k],
                1e-12,
                &format!("gm-probit chain gradient theta[{k}]"),
            );
        }
    }

    /// Parity guard for the shared `baseline_chain_rule_gradient_with_partials`
    /// engine (issue #429): both public gradient functions delegate to it with a
    /// different partials provider. This test reimplements the pre-unification
    /// inline contraction (the serial reference) and asserts bit-for-bit equality
    /// against the unified engine's output for BOTH providers on the same data —
    /// the RP-eta provider (`baseline_offset_theta_partials`) and the probit-q
    /// provider (`marginal_slope_baseline_offset_theta_partials`). Any drift in
    /// the extracted contraction (length checks, theta-dim probe, exit/derivative
    /// combination, or entry gating) breaks this with an exact (0.0) tolerance.
    #[test]
    fn baseline_chain_rule_gradient_engine_matches_inline_reference() {
        assert!(file!().ends_with(".rs"));
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.028),
            rate: Some(0.011),
            makeham: Some(0.0025),
        };
        // Mixed entry interval: row 1 is origin-entry (age_entry==0, r_entry==0)
        // to exercise the entry-gating branch in the shared engine.
        let age_entry = array![3.0, 0.0, 5.5];
        let age_exit = array![8.0, 12.0, 16.0];
        let residuals = OffsetChannelResiduals {
            exit: array![0.7, -0.2, 0.45],
            entry: array![0.1, 0.0, -0.3],
            derivative: array![1.3, -0.6, 0.2],
            right: Array1::<f64>::zeros(3),
        };

        // Serial reference contraction matching the original inline body. Mirrors
        // the engine's exit+derivative/entry split and origin-entry gating.
        let reference_gradient = |partials: &dyn Fn(
            f64,
            &SurvivalBaselineConfig,
        )
            -> Result<Option<Vec<(f64, f64)>>, String>|
         -> Array1<f64> {
            let theta_dim = partials(age_exit[0], &cfg)
                .expect("probe partials")
                .expect("nonlinear")
                .len();
            let mut acc = Array1::<f64>::zeros(theta_dim);
            for i in 0..age_exit.len() {
                let p_exit = partials(age_exit[i], &cfg)
                    .expect("exit partials")
                    .expect("nonlinear");
                let r_x = residuals.exit[i];
                let r_d = residuals.derivative[i];
                for k in 0..theta_dim {
                    acc[k] += r_x * p_exit[k].0 + r_d * p_exit[k].1;
                }
                let r_e = residuals.entry[i];
                if r_e != 0.0 {
                    let p_entry = partials(age_entry[i], &cfg)
                        .expect("entry partials")
                        .expect("nonlinear");
                    for k in 0..theta_dim {
                        acc[k] += r_e * p_entry[k].0;
                    }
                }
            }
            acc
        };

        // RP-eta provider parity.
        let rp_engine = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("rp gradient")
        .expect("rp nonlinear");
        let rp_reference = reference_gradient(&baseline_offset_theta_partials);
        assert_eq!(rp_engine.len(), rp_reference.len());
        for k in 0..rp_engine.len() {
            assert_close(
                rp_engine[k],
                rp_reference[k],
                0.0,
                &format!("rp engine vs inline reference theta[{k}]"),
            );
        }

        // Probit-q provider parity.
        let probit_engine = marginal_slope_baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("probit gradient")
        .expect("probit nonlinear");
        let probit_reference = reference_gradient(&marginal_slope_baseline_offset_theta_partials);
        assert_eq!(probit_engine.len(), probit_reference.len());
        for k in 0..probit_engine.len() {
            assert_close(
                probit_engine[k],
                probit_reference[k],
                0.0,
                &format!("probit engine vs inline reference theta[{k}]"),
            );
        }
    }

    /// Finite-difference verification of the analytic θ-gradient used by the
    /// survival location-scale workflow path.
    ///
    /// At a converged β, the envelope theorem reduces the profile-NLL gradient
    /// w.r.t. the baseline-config θ to a per-row residual contraction against
    /// the per-row offset-channel partials ∂o/∂θ:
    ///
    ///   d(NLL)/dθ_k = Σ_i [ r_X[i]·∂η_exit/∂θ_k + r_E[i]·∂η_entry/∂θ_k
    ///                       + r_D[i]·∂o_D_exit/∂θ_k ]
    ///
    /// (`baseline_chain_rule_gradient`). Because β is fixed, an explicit loss
    /// `L(θ) = Σ_i [ r_X[i]·η(t_exit_i; θ) + r_E[i]·η(t_entry_i; θ)
    ///              + r_D[i]·o_D(t_exit_i; θ) ]`
    /// has gradient identically equal to the chain-rule output. Comparing the
    /// analytic gradient to a central-difference of L over `evaluate_survival_baseline`
    /// therefore exercises every piece of the chain rule (incl. the Gompertz
    /// rate / shape / Makeham partials at both entry and exit ages) without
    /// needing the full location-scale fit pipeline inside this unit-test
    /// module. If the chain rule disagrees with FD here, the workflow's
    /// gradient is wrong by exactly the same amount.
    #[test]
    fn gompertz_makeham_baseline_chain_rule_gradient_matches_finite_difference() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::GompertzMakeham,
            scale: None,
            shape: Some(0.05),
            rate: Some(0.012),
            makeham: Some(0.003),
        };
        // n = 8 small synthetic dataset spanning a realistic age range.
        let age_entry = array![5.0, 8.0, 12.0, 0.5, 20.0, 30.0, 45.0, 60.0];
        let age_exit = array![10.0, 15.0, 25.0, 4.0, 35.0, 50.0, 65.0, 80.0];
        // Synthetic per-row NLL residuals on the three offset channels. Mix of
        // signs / magnitudes / one zero-entry row (origin entry → r_E=0).
        let residuals = OffsetChannelResiduals {
            exit: array![0.42, -0.18, 0.73, -0.91, 0.05, -0.27, 0.61, -0.34],
            entry: array![-0.12, 0.31, -0.44, 0.0, 0.16, -0.22, 0.07, -0.51],
            derivative: array![1.04, -0.65, 0.18, -1.21, 0.42, -0.13, 0.88, -0.27],
            right: Array1::<f64>::zeros(8),
        };

        let analytic = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("analytic gradient ok")
        .expect("GM baseline has a θ-gradient");
        assert_eq!(analytic.len(), 3, "GM θ has 3 components");

        // Evaluate the offset-projected loss at a perturbed θ. Mirrors the
        // chain rule's algebra: the entry channel is only added for rows whose
        // r_E is nonzero (matching baseline_chain_rule_gradient's gating that
        // avoids calling evaluate_survival_baseline at age 0 for origin-entry
        // rows).
        let loss_at_cfg = |cfg_eval: &SurvivalBaselineConfig| -> f64 {
            let mut acc = 0.0;
            for i in 0..age_exit.len() {
                let (eta_exit_i, od_exit_i) =
                    evaluate_survival_baseline(age_exit[i], cfg_eval).expect("eval exit");
                acc += residuals.exit[i] * eta_exit_i + residuals.derivative[i] * od_exit_i;
                if residuals.entry[i] != 0.0 {
                    let (eta_entry_i, _) =
                        evaluate_survival_baseline(age_entry[i], cfg_eval).expect("eval entry");
                    acc += residuals.entry[i] * eta_entry_i;
                }
            }
            acc
        };

        let theta0 = survival_baseline_theta_from_config(&cfg)
            .expect("theta seed")
            .expect("GM has θ");
        // Spec requested δ = 1e-4 per axis. Use central differences over θ.
        let delta = 1e-4;
        let mut fd = Array1::<f64>::zeros(analytic.len());
        for k in 0..analytic.len() {
            let mut theta_plus = theta0.clone();
            theta_plus[k] += delta;
            let mut theta_minus = theta0.clone();
            theta_minus[k] -= delta;
            let cfg_plus =
                survival_baseline_config_from_theta(cfg.target, &theta_plus).expect("cfg(θ+δ)");
            let cfg_minus =
                survival_baseline_config_from_theta(cfg.target, &theta_minus).expect("cfg(θ-δ)");
            let lp = loss_at_cfg(&cfg_plus);
            let lm = loss_at_cfg(&cfg_minus);
            fd[k] = (lp - lm) / (2.0 * delta);
        }

        let analytic_norm = analytic.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_err = analytic
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let rel = max_err / (analytic_norm + 1e-12);
        // Print so the deliverable can quote the exact max-error number.
        eprintln!(
            "gompertz_makeham_baseline_chain_rule_gradient_matches_finite_difference: \
             analytic={analytic:?} fd={fd:?} max_err={max_err:.3e} \
             analytic_inf_norm={analytic_norm:.3e} rel={rel:.3e}"
        );
        assert!(
            rel < 1e-2,
            "analytic θ-gradient disagrees with central FD beyond 1%: \
             analytic={analytic:?}, fd={fd:?}, max_err={max_err:.3e}, \
             rel={rel:.3e} (analytic_inf_norm={analytic_norm:.3e})"
        );
    }

    /// Weibull (dim=2) companion to
    /// `gompertz_makeham_baseline_chain_rule_gradient_matches_finite_difference`.
    ///
    /// This is the FD gate for the analytic outer θ-gradient that the
    /// transformation/Weibull survival baseline optimizers now feed to BFGS
    /// (`optimize_survival_baseline_config_with_gradient_only`). At a *fixed* β
    /// the profile-NLL surface is
    /// `L(θ) = Σ_i [ r_X[i]·η(t_exit_i;θ) + r_E[i]·η(t_entry_i;θ)
    ///              + r_D[i]·o_D(t_exit_i;θ) ]`,
    /// whose exact gradient is `baseline_chain_rule_gradient`. Comparing it to a
    /// central difference of `L` over `evaluate_survival_baseline` exercises the
    /// Weibull scale/shape partials at both entry and exit ages. If this
    /// disagrees with FD, the workflow's outer gradient is wrong by the same
    /// amount.
    #[test]
    fn weibull_baseline_chain_rule_gradient_matches_finite_difference() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(11.0),
            shape: Some(1.4),
            rate: None,
            makeham: None,
        };
        let age_entry = array![5.0, 8.0, 12.0, 0.5, 20.0, 30.0, 45.0, 60.0];
        let age_exit = array![10.0, 15.0, 25.0, 4.0, 35.0, 50.0, 65.0, 80.0];
        let residuals = OffsetChannelResiduals {
            exit: array![0.42, -0.18, 0.73, -0.91, 0.05, -0.27, 0.61, -0.34],
            entry: array![-0.12, 0.31, -0.44, 0.0, 0.16, -0.22, 0.07, -0.51],
            derivative: array![1.04, -0.65, 0.18, -1.21, 0.42, -0.13, 0.88, -0.27],
            right: Array1::<f64>::zeros(8),
        };

        let analytic = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("analytic gradient ok")
        .expect("Weibull baseline has a θ-gradient");
        assert_eq!(analytic.len(), 2, "Weibull θ has 2 components");

        let loss_at_cfg = |cfg_eval: &SurvivalBaselineConfig| -> f64 {
            let mut acc = 0.0;
            for i in 0..age_exit.len() {
                let (eta_exit_i, od_exit_i) =
                    evaluate_survival_baseline(age_exit[i], cfg_eval).expect("eval exit");
                acc += residuals.exit[i] * eta_exit_i + residuals.derivative[i] * od_exit_i;
                if residuals.entry[i] != 0.0 {
                    let (eta_entry_i, _) =
                        evaluate_survival_baseline(age_entry[i], cfg_eval).expect("eval entry");
                    acc += residuals.entry[i] * eta_entry_i;
                }
            }
            acc
        };

        let theta0 = survival_baseline_theta_from_config(&cfg)
            .expect("theta seed")
            .expect("Weibull has θ");
        let delta = 1e-4;
        let mut fd = Array1::<f64>::zeros(analytic.len());
        for k in 0..analytic.len() {
            let mut theta_plus = theta0.clone();
            theta_plus[k] += delta;
            let mut theta_minus = theta0.clone();
            theta_minus[k] -= delta;
            let cfg_plus =
                survival_baseline_config_from_theta(cfg.target, &theta_plus).expect("cfg(θ+δ)");
            let cfg_minus =
                survival_baseline_config_from_theta(cfg.target, &theta_minus).expect("cfg(θ-δ)");
            let lp = loss_at_cfg(&cfg_plus);
            let lm = loss_at_cfg(&cfg_minus);
            fd[k] = (lp - lm) / (2.0 * delta);
        }

        let analytic_norm = analytic.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_err = analytic
            .iter()
            .zip(fd.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let rel = max_err / (analytic_norm + 1e-12);
        eprintln!(
            "weibull_baseline_chain_rule_gradient_matches_finite_difference: \
             analytic={analytic:?} fd={fd:?} max_err={max_err:.3e} \
             analytic_inf_norm={analytic_norm:.3e} rel={rel:.3e}"
        );
        assert!(
            rel < 1e-2,
            "analytic θ-gradient disagrees with central FD beyond 1%: \
             analytic={analytic:?}, fd={fd:?}, max_err={max_err:.3e}, \
             rel={rel:.3e} (analytic_inf_norm={analytic_norm:.3e})"
        );
    }

    // ─── baseline_offset_theta_partials — analytic vs central-difference ─

    /// Central-difference of (eta, o_D) at fixed age wrt each θ component in
    /// the theta layout defined by `survival_baseline_theta_from_config`.
    ///
    /// `steps` is per-θ-component: the caller picks the step size appropriate
    /// for each channel. Gompertz / Gompertz–Makeham need a tiny step on the
    /// shape channel near the Taylor pivot |shape| < 1e-10 (so θ±h stays on
    /// the same branch), but a normal-scale step on log_rate / log_makeham;
    /// using the tiny shape-step on every channel corrupts the log_rate
    /// channel with `eps/(2h)` cancellation noise and has nothing to do with
    /// correctness of the analytic derivative.
    fn fd_baseline_offset(
        age: f64,
        cfg: &SurvivalBaselineConfig,
        steps: &[f64],
    ) -> Vec<(f64, f64)> {
        let theta = survival_baseline_theta_from_config(cfg)
            .expect("theta")
            .expect("non-linear baseline");
        assert_eq!(
            steps.len(),
            theta.len(),
            "fd_baseline_offset: step vector length must match θ dimension"
        );
        (0..theta.len())
            .map(|k| {
                let h = steps[k];
                let mut theta_plus = theta.clone();
                theta_plus[k] += h;
                let mut theta_minus = theta.clone();
                theta_minus[k] -= h;
                let cfg_plus =
                    survival_baseline_config_from_theta(cfg.target, &theta_plus).expect("plus cfg");
                let cfg_minus = survival_baseline_config_from_theta(cfg.target, &theta_minus)
                    .expect("minus cfg");
                let (eta_p, od_p) = evaluate_survival_baseline(age, &cfg_plus).expect("eta+");
                let (eta_m, od_m) = evaluate_survival_baseline(age, &cfg_minus).expect("eta-");
                ((eta_p - eta_m) / (2.0 * h), (od_p - od_m) / (2.0 * h))
            })
            .collect()
    }

    fn assert_close(actual: f64, expected: f64, tol: f64, what: &str) {
        // `<=` so that bit-equal values satisfy tol = 0. With `<`, |a−e| < 0
        // is unsatisfiable and a zero-tolerance "must match exactly" call
        // would reject identical numbers.
        let ok = if expected.abs() < 1.0 {
            (actual - expected).abs() <= tol
        } else {
            (actual - expected).abs() <= tol * expected.abs().max(1.0)
        };
        assert!(
            ok,
            "{what}: analytic={actual:.6e} fd={expected:.6e} (tol={tol:.1e})"
        );
    }

    #[test]
    fn gompertz_offset_partials_match_central_diff() {
        // Several (rate, shape, age) combinations spanning the small-shape
        // Taylor branch (|shape| < 1e-10) and the normal branch
        // (shape >> 1e-10), plus sign-reversed shape.
        let cases = [
            (0.5_f64, 0.01_f64, 30.0_f64),
            (0.2, 0.05, 60.0),
            (1.0, 0.001, 10.0),
            (0.4, 5e-11, 25.0),
            (0.4, -5e-11, 25.0),
            (0.3, -0.02, 40.0),
            (0.8, 0.2, 5.0),
        ];
        for &(rate, shape, age) in &cases {
            let cfg = SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Gompertz,
                scale: None,
                shape: Some(shape),
                rate: Some(rate),
                makeham: None,
            };
            let analytic = baseline_offset_theta_partials(age, &cfg)
                .expect("ok")
                .expect("non-linear");
            // Keep the FD probe inside the Taylor branch for tiny |shape| so
            // the numeric derivative matches the same small-shape map as the
            // analytic helper. log_rate always uses the normal step — rate
            // is a moderate-scale parameter and a 1e-11 step would swamp the
            // FD with cancellation noise.
            let h_shape = if shape.abs() < 1e-9 { 1e-11 } else { 1e-5 };
            let fd = fd_baseline_offset(age, &cfg, &[1e-5, h_shape]);
            assert_eq!(analytic.len(), 2);
            // Gompertz θ=(log_rate, shape). Rate channel: ∂eta/∂log_rate=1, ∂o_D/∂log_rate=0.
            assert_close(
                analytic[0].0,
                fd[0].0,
                1e-7,
                &format!("gompertz ∂eta/∂log_rate (rate={rate}, shape={shape}, age={age})"),
            );
            assert_close(
                analytic[0].1,
                fd[0].1,
                1e-7,
                &format!("gompertz ∂o_D/∂log_rate (rate={rate}, shape={shape}, age={age})"),
            );
            // shape channel — larger tol because finite-differencing near
            // shape=0 amplifies rounding; 1e-5 is fine.
            assert_close(
                analytic[1].0,
                fd[1].0,
                1e-5,
                &format!("gompertz ∂eta/∂shape (rate={rate}, shape={shape}, age={age})"),
            );
            assert_close(
                analytic[1].1,
                fd[1].1,
                1e-5,
                &format!("gompertz ∂o_D/∂shape (rate={rate}, shape={shape}, age={age})"),
            );
        }
    }

    #[test]
    fn gompertz_offset_partials_log_rate_channel_is_trivial() {
        // Pure Gompertz: rate cancels in o_D, so ∂o_D/∂log_rate must be
        // exactly 0 and ∂eta/∂log_rate must be exactly 1. Verify the
        // analytic implementation returns the exact values, not FD-close.
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(0.05),
            rate: Some(0.3),
            makeham: None,
        };
        let partials = baseline_offset_theta_partials(42.0, &cfg)
            .expect("ok")
            .expect("non-linear");
        assert_eq!(partials[0].0, 1.0);
        assert_eq!(partials[0].1, 0.0);
    }

    #[test]
    fn gompertz_offset_partials_small_shape_taylor_agrees_with_direct_branch() {
        assert!(file!().ends_with(".rs"));
        // Both branches of gompertz_shape_derivatives should agree to high
        // precision at shape = 1e-10 + epsilon on the direct side vs
        // shape = 1e-10 - epsilon on the Taylor side. Here we spot-check
        // the continuity at the branch cutoff: shape slightly above and
        // slightly below 1e-10 must give values within O(shape²·t²)
        // (the Taylor truncation error).
        let age = 25.0;
        let rate = 0.4;
        let cfg_taylor = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(0.5e-10),
            rate: Some(rate),
            makeham: None,
        };
        let cfg_direct = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(2.0e-10),
            rate: Some(rate),
            makeham: None,
        };
        let p_t = baseline_offset_theta_partials(age, &cfg_taylor)
            .expect("ok")
            .expect("nl");
        let p_d = baseline_offset_theta_partials(age, &cfg_direct)
            .expect("ok")
            .expect("nl");
        // ∂eta/∂shape at shape≈0 should be t/2 = 12.5 on both sides.
        assert_close(p_t[1].0, 12.5, 1e-8, "taylor ∂eta/∂shape near 0");
        assert_close(p_d[1].0, 12.5, 1e-8, "direct ∂eta/∂shape near 0");
        // ∂o_D/∂shape at shape≈0 should be 1/2.
        assert_close(p_t[1].1, 0.5, 1e-8, "taylor ∂o_D/∂shape near 0");
        assert_close(p_d[1].1, 0.5, 1e-8, "direct ∂o_D/∂shape near 0");
    }

    // ----------------------------------------------------------------------
    // Gompertz hazard-channel shape derivatives: FD oracle + Taylor-branch
    // continuity. These feed `survival_hazard_theta_partials` /
    // `survival_hazard_theta_first_second` (the marginal-slope probit
    // baseline). Before this test, the only coverage of
    // `gompertz_cumulative_shape_{,second_}derivative` was the indirect
    // marginal-slope Hessian FD at shape=0.025, which never touches the
    // small-shape (`|shape| < 1e-10`) Taylor branch nor directly FD-checks
    // these analytic shape derivatives.
    // ----------------------------------------------------------------------

    #[test]
    fn gompertz_hazard_shape_derivatives_match_central_diff() {
        assert!(file!().ends_with(".rs"));
        // shape stays well above the 1e-10 Taylor cutoff so the exact
        // closed-form branch is exercised and the expm1/exp arithmetic is
        // numerically clean. FD on the analytic value/first-derivative
        // confirms the first and second shape derivatives.
        let cases = [
            (10.0_f64, 0.012_f64, 0.05_f64),
            (2.5, 0.5, 0.2),
            (15.0, 0.003, 0.01),
            (40.0, 0.3, 0.001),
        ];
        let h = 1e-6;
        for &(age, rate, shape) in &cases {
            // First shape derivative of (H_G, h_G) vs central diff of value.
            let (d_cum, d_inst) = gompertz_cumulative_shape_derivative(age, rate, shape);
            let (cum_p, inst_p) = gompertz_hazard_components(age, rate, shape + h);
            let (cum_m, inst_m) = gompertz_hazard_components(age, rate, shape - h);
            assert_close(
                d_cum,
                (cum_p - cum_m) / (2.0 * h),
                1e-6,
                &format!("∂H_G/∂shape (age={age}, rate={rate}, shape={shape})"),
            );
            assert_close(
                d_inst,
                (inst_p - inst_m) / (2.0 * h),
                1e-6,
                &format!("∂h_G/∂shape (age={age}, rate={rate}, shape={shape})"),
            );

            // Second shape derivative vs central diff of the first derivative.
            let (d2_cum, d2_inst) = gompertz_cumulative_shape_second_derivative(age, rate, shape);
            let (dcum_p, dinst_p) = gompertz_cumulative_shape_derivative(age, rate, shape + h);
            let (dcum_m, dinst_m) = gompertz_cumulative_shape_derivative(age, rate, shape - h);
            assert_close(
                d2_cum,
                (dcum_p - dcum_m) / (2.0 * h),
                1e-5,
                &format!("∂²H_G/∂shape² (age={age}, rate={rate}, shape={shape})"),
            );
            assert_close(
                d2_inst,
                (dinst_p - dinst_m) / (2.0 * h),
                1e-5,
                &format!("∂²h_G/∂shape² (age={age}, rate={rate}, shape={shape})"),
            );
        }
    }

    #[test]
    fn gompertz_hazard_shape_derivatives_small_shape_match_analytic_limit() {
        assert!(file!().ends_with(".rs"));
        // At small x = shape·age the shape derivatives collapse to closed-form
        // limits. These MUST hold even for large ages with tiny shapes, which
        // is precisely the regime where the (cancelling) exact branch loses all
        // precision and the x-based pivot routes to the Taylor branch.
        //   ∂H_G/∂shape   -> rate·t²/2
        //   ∂h_G/∂shape   -> rate·t
        //   ∂²H_G/∂shape² -> rate·t³/3
        //   ∂²h_G/∂shape² -> rate·t²
        // The bug this guards: the second derivative's old `shape < 1e-10`
        // pivot ignored `age`, so e.g. (age=100, shape=1e-5 -> x=1e-3) took the
        // cancelling exact branch and returned a wildly wrong curvature.
        let cases = [
            (25.0_f64, 0.4_f64, 1e-9_f64),
            (100.0, 0.4, 1e-6),   // x = 1e-4
            (100.0, 0.012, 1e-5), // x = 1e-3, the old-pivot failure point
            (50.0, 1.2, 1e-8),
        ];
        for &(age, rate, shape) in &cases {
            let t = age;
            let (d_cum, d_inst) = gompertz_cumulative_shape_derivative(age, rate, shape);
            assert_close(
                d_cum,
                rate * t * t / 2.0,
                1e-3,
                &format!("∂H_G/∂shape limit (age={age}, shape={shape})"),
            );
            assert_close(
                d_inst,
                rate * t,
                1e-3,
                &format!("∂h_G/∂shape limit (age={age}, shape={shape})"),
            );

            let (d2_cum, d2_inst) = gompertz_cumulative_shape_second_derivative(age, rate, shape);
            assert_close(
                d2_cum,
                rate * t * t * t / 3.0,
                1e-3,
                &format!("∂²H_G/∂shape² limit (age={age}, shape={shape})"),
            );
            assert_close(
                d2_inst,
                rate * t * t,
                1e-3,
                &format!("∂²h_G/∂shape² limit (age={age}, shape={shape})"),
            );
        }
    }

    #[test]
    fn gompertz_second_shape_derivative_is_accurate_in_old_pivot_gap() {
        assert!(file!().ends_with(".rs"));
        // Regression: in the band shape ∈ [1e-10, ~1e-4] with a realistic age,
        // the OLD `shape < 1e-10` pivot sent ∂²H_G/∂shape² through the
        // catastrophically-cancelling exact branch. With age=100, shape=1e-9
        // (x=1e-7) the exact branch returned ~+5e1 vs the true ~rate·t³/3.
        // Assert the implementation now matches the closed-form limit to high
        // precision throughout that band, across several decades of shape.
        let age = 100.0;
        let rate = 0.4;
        let t = age;
        let truth = rate * t * t * t / 3.0; // 1.333e5
        // Start at shape=1e-5 (x=1e-3): below this the second derivative is,
        // to better than 1e-3 relative, equal to its shape->0 limit, so the
        // limit is a valid oracle. (At x=1e-2 the true value legitimately
        // departs from the limit by ~7e-3, which is a real O(x) correction,
        // not an error — so we do not extend the band up to shape=1e-4.)
        for k in 5..=12 {
            let shape = 10f64.powi(-(k as i32)); // 1e-5 .. 1e-12
            let (d2_cum, _) = gompertz_cumulative_shape_second_derivative(age, rate, shape);
            assert_close(
                d2_cum,
                truth,
                1e-3,
                &format!("∂²H_G/∂shape² in old-pivot gap (age={age}, shape=1e-{k})"),
            );
        }
    }

    #[test]
    fn weibull_offset_partials_match_central_diff() {
        let cases = [
            (0.5_f64, 1.2_f64, 25.0_f64),
            (2.0, 0.8, 60.0),
            (0.1, 3.0, 10.0),
        ];
        for &(scale, shape, age) in &cases {
            let cfg = SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::Weibull,
                scale: Some(scale),
                shape: Some(shape),
                rate: None,
                makeham: None,
            };
            let analytic = baseline_offset_theta_partials(age, &cfg)
                .expect("ok")
                .expect("nl");
            let fd = fd_baseline_offset(age, &cfg, &[1e-5, 1e-5]);
            assert_eq!(analytic.len(), 2);
            for k in 0..2 {
                assert_close(
                    analytic[k].0,
                    fd[k].0,
                    1e-7,
                    &format!("weibull ∂eta/∂θ[{k}] (scale={scale}, shape={shape}, age={age})"),
                );
                assert_close(
                    analytic[k].1,
                    fd[k].1,
                    1e-7,
                    &format!("weibull ∂o_D/∂θ[{k}] (scale={scale}, shape={shape}, age={age})"),
                );
            }
            // Weibull o_D = shape/t is independent of scale; verify exactly.
            assert_eq!(analytic[0].1, 0.0);
        }
    }

    #[test]
    fn gompertz_makeham_offset_partials_match_central_diff() {
        let cases = [
            (0.3_f64, 0.05_f64, 0.002_f64, 40.0_f64),
            (0.5, 0.01, 0.01, 25.0),
            (0.2, 0.001, 0.005, 60.0),
            (0.4, 5e-11, 0.01, 25.0),
            (0.4, -5e-11, 0.01, 25.0),
            (0.8, 0.2, 0.05, 5.0),
        ];
        for &(rate, shape, makeham, age) in &cases {
            let cfg = SurvivalBaselineConfig {
                target: SurvivalBaselineTarget::GompertzMakeham,
                scale: None,
                shape: Some(shape),
                rate: Some(rate),
                makeham: Some(makeham),
            };
            let analytic = baseline_offset_theta_partials(age, &cfg)
                .expect("ok")
                .expect("nl");
            // See gompertz_offset_partials_match_central_diff: tiny shape-step
            // is only needed for the shape component; log_rate and
            // log_makeham take the normal-scale step.
            let h_shape = if shape.abs() < 1e-9 { 1e-11 } else { 1e-5 };
            let fd = fd_baseline_offset(age, &cfg, &[1e-5, h_shape, 1e-5]);
            assert_eq!(analytic.len(), 3);
            for k in 0..3 {
                assert_close(
                    analytic[k].0,
                    fd[k].0,
                    1e-5,
                    &format!(
                        "gm ∂eta/∂θ[{k}] (rate={rate}, shape={shape}, mk={makeham}, age={age})"
                    ),
                );
                assert_close(
                    analytic[k].1,
                    fd[k].1,
                    1e-5,
                    &format!(
                        "gm ∂o_D/∂θ[{k}] (rate={rate}, shape={shape}, mk={makeham}, age={age})"
                    ),
                );
            }
        }
    }

    #[test]
    fn linear_baseline_has_no_theta_partials() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Linear,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        };
        assert!(baseline_offset_theta_partials(5.0, &cfg).unwrap().is_none());
    }

    #[test]
    fn baseline_offset_partials_reject_non_positive_ages() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(0.01),
            rate: Some(0.5),
            makeham: None,
        };
        assert!(baseline_offset_theta_partials(0.0, &cfg).is_err());
        assert!(baseline_offset_theta_partials(-1.0, &cfg).is_err());
        assert!(baseline_offset_theta_partials(f64::NAN, &cfg).is_err());
    }

    // ─── baseline_chain_rule_gradient — mechanical and FD-vs-θ tests ─────

    /// Mechanical sanity check: with only one event observation at known
    /// (r_X, r_E, r_D, age_exit, age_entry), the Gompertz chain-rule gradient
    /// reduces to the analytic linear combination of `baseline_offset_theta_partials`.
    #[test]
    fn chain_rule_gradient_single_obs_reduces_to_pointwise_contract() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(0.05),
            rate: Some(0.3),
            makeham: None,
        };
        let age_entry = array![10.0_f64];
        let age_exit = array![25.0_f64];
        let residuals = OffsetChannelResiduals {
            exit: array![0.7_f64],
            entry: array![-0.2_f64],
            derivative: array![-0.4_f64],
            right: Array1::<f64>::zeros(1),
        };
        let grad = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("ok")
        .expect("non-linear");
        // Hand-compute: grad[k] = r_X·∂eta_exit/∂θ_k + r_D·∂o_D_exit/∂θ_k + r_E·∂eta_entry/∂θ_k.
        let p_exit = baseline_offset_theta_partials(age_exit[0], &cfg)
            .unwrap()
            .unwrap();
        let p_entry = baseline_offset_theta_partials(age_entry[0], &cfg)
            .unwrap()
            .unwrap();
        for k in 0..p_exit.len() {
            let expected = 0.7 * p_exit[k].0 + (-0.4) * p_exit[k].1 + (-0.2) * p_entry[k].0;
            assert!(
                (grad[k] - expected).abs() < 1e-12,
                "chain-rule contract mismatch at k={k}: got={:.6e} expected={:.6e}",
                grad[k],
                expected
            );
        }
    }

    /// Origin-entry rows (r_entry == 0) must skip the baseline partials call at
    /// `age_entry = 0`, which would otherwise fail the positive-age precondition.
    #[test]
    fn chain_rule_gradient_skips_entry_call_for_origin_entry_rows() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(0.05),
            rate: Some(0.3),
            makeham: None,
        };
        let age_entry = array![0.0_f64, 5.0_f64];
        let age_exit = array![10.0_f64, 20.0_f64];
        let residuals = OffsetChannelResiduals {
            exit: array![0.5_f64, 0.3_f64],
            entry: array![0.0_f64, -0.1_f64], // row 0 is origin-entry (r_E = 0)
            derivative: array![-0.2_f64, 0.0_f64],
            right: Array1::<f64>::zeros(2),
        };
        // Must not error despite age_entry[0] == 0.
        let grad = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("must not fail on origin-entry row with r_entry=0")
        .expect("non-linear");
        assert_eq!(grad.len(), 2);
        // Row 1's entry channel contributes, row 0's does not.
        let p_exit_0 = baseline_offset_theta_partials(10.0, &cfg).unwrap().unwrap();
        let p_exit_1 = baseline_offset_theta_partials(20.0, &cfg).unwrap().unwrap();
        let p_entry_1 = baseline_offset_theta_partials(5.0, &cfg).unwrap().unwrap();
        for k in 0..2 {
            let expected = 0.5 * p_exit_0[k].0
                + (-0.2) * p_exit_0[k].1
                + 0.3 * p_exit_1[k].0
                + (-0.1) * p_entry_1[k].0;
            assert!(
                (grad[k] - expected).abs() < 1e-12,
                "origin-entry contract at k={k}: got={:.6e} expected={:.6e}",
                grad[k],
                expected
            );
        }
    }

    /// Linear target has no θ-parameters; contractor returns None.
    #[test]
    fn chain_rule_gradient_linear_target_returns_none() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Linear,
            scale: None,
            shape: None,
            rate: None,
            makeham: None,
        };
        let age_entry = array![1.0_f64];
        let age_exit = array![2.0_f64];
        let residuals = OffsetChannelResiduals {
            exit: array![0.1_f64],
            entry: array![0.0_f64],
            derivative: array![0.0_f64],
            right: Array1::<f64>::zeros(1),
        };
        let grad = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("ok");
        assert!(grad.is_none());
    }

    /// End-to-end envelope-theorem check: the chain-rule gradient at
    /// residuals-evaluated-at-β-fixed matches the central FD of the
    /// unpenalized NLL with respect to θ when the OFFSETS are recomputed
    /// from the perturbed cfg and β is held at its base value.
    ///
    /// This is the mathematical content of the envelope theorem applied to
    /// the penalized-deviance cost at fixed β: if β solves ∂C/∂β = 0 at
    /// (θ, β*), then the total derivative of C at (θ±h) when β is held at
    /// β* equals the partial derivative of C wrt θ at the base — up to
    /// O(h²) in the truncation error of central differences. For THIS test
    /// we're directly differencing NLL (the unpenalized piece that carries
    /// all the θ dependence), so the envelope identity is exact up to FD
    /// truncation.
    ///
    /// The test synthesizes a plausible residual set by hand rather than
    /// running PIRLS — what we're validating is the chain-rule contractor,
    /// not the fit. A PIRLS-based end-to-end check belongs in an
    /// integration test, not this unit-test module.
    #[test]
    fn chain_rule_gradient_matches_fd_of_nll_through_offset_perturbation() {
        // Toy 3-observation case with two events (one origin-entry, one not)
        // and one censored row at large age.
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(0.03),
            rate: Some(0.25),
            makeham: None,
        };
        let age_entry = array![0.0_f64, 5.0, 8.0];
        let age_exit = array![4.0_f64, 12.0, 20.0];
        // Weighted residuals at a notional β*. Values chosen in a plausible
        // range (~same order as w·exp(η)).
        let weights = array![1.0_f64, 2.0, 0.5];
        let events = [1.0_f64, 1.0, 0.0];
        // Fake a β* that yields finite eta_entry ± eta_exit ± s values by
        // directly specifying eta quantities. Contractor only consumes the
        // residuals, so the fake is sufficient.
        let eta_entry_vals = [-100.0_f64, 0.5, 0.8]; // row 0 doesn't matter (origin entry)
        let eta_exit_vals = [0.4_f64, 0.9, 1.3];
        let s_vals = [0.7_f64, 1.1, 1.5];
        let (r_x, r_e, r_d) = {
            let mut rx = Array1::<f64>::zeros(3);
            let mut re = Array1::<f64>::zeros(3);
            let mut rd = Array1::<f64>::zeros(3);
            for i in 0..3 {
                let w = weights[i];
                let d = events[i];
                rx[i] = w * (eta_exit_vals[i].exp() - d);
                re[i] = if i == 0 {
                    0.0 // origin entry
                } else {
                    -w * eta_entry_vals[i].exp()
                };
                rd[i] = if d > 0.0 { -w * d / s_vals[i] } else { 0.0 };
            }
            (rx, re, rd)
        };
        let residuals = OffsetChannelResiduals {
            exit: r_x.clone(),
            entry: r_e.clone(),
            derivative: r_d.clone(),
            right: Array1::<f64>::zeros(3),
        };
        let grad = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect("ok")
        .expect("non-linear");

        // Construct NLL(θ) with β* held to the same eta/s values by treating
        // eta_i, s_i as fixed "linear predictor" samples and shifting by
        // (offset(θ) - offset(θ_base)). That's exactly the RP NLL with β*
        // held constant and offsets varied through θ.
        let nll = |theta_plus: &Array1<f64>| -> f64 {
            let cfg_p = survival_baseline_config_from_theta(cfg.target, theta_plus).expect("cfg_p");
            let mut sum = 0.0_f64;
            for i in 0..3 {
                let (eta_x_p, d_x_p) = evaluate_survival_baseline(age_exit[i], &cfg_p).unwrap();
                let base = evaluate_survival_baseline(age_exit[i], &cfg).unwrap();
                let d_eta_x = eta_x_p - base.0;
                let d_d_x = d_x_p - base.1;
                let eta_exit_new = eta_exit_vals[i] + d_eta_x;
                let s_new = s_vals[i] + d_d_x;
                let interval_entry = if i == 0 {
                    0.0_f64
                } else {
                    let (eta_e_p, _) = evaluate_survival_baseline(age_entry[i], &cfg_p).unwrap();
                    let base_e = evaluate_survival_baseline(age_entry[i], &cfg).unwrap();
                    let d_eta_e = eta_e_p - base_e.0;
                    let eta_entry_new = eta_entry_vals[i] + d_eta_e;
                    eta_entry_new.exp()
                };
                let w = weights[i];
                let d = events[i];
                let nll_i =
                    w * (eta_exit_new.exp() - interval_entry - d * (eta_exit_new + s_new.ln()));
                sum += nll_i;
            }
            sum
        };

        let theta_base = survival_baseline_theta_from_config(&cfg).unwrap().unwrap();
        let h = 1e-6;
        for k in 0..theta_base.len() {
            let mut tp = theta_base.clone();
            let mut tm = theta_base.clone();
            tp[k] += h;
            tm[k] -= h;
            let fd = (nll(&tp) - nll(&tm)) / (2.0 * h);
            assert!(
                (grad[k] - fd).abs() < 1e-5 * grad[k].abs().max(1.0),
                "chain-rule θ[{k}]: analytic={:.6e} fd={:.6e}",
                grad[k],
                fd
            );
        }
    }

    /// Length-mismatch surfaces as an error, not a silent contraction.
    #[test]
    fn chain_rule_gradient_rejects_length_mismatch() {
        let cfg = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Gompertz,
            scale: None,
            shape: Some(0.05),
            rate: Some(0.3),
            makeham: None,
        };
        let age_entry = array![1.0_f64, 2.0]; // length 2
        let age_exit = array![5.0_f64, 6.0, 7.0]; // length 3
        let residuals = OffsetChannelResiduals {
            exit: array![0.1_f64, 0.2, 0.3],
            entry: array![0.0_f64, 0.0, 0.0],
            derivative: array![0.0_f64, 0.0, 0.0],
            right: Array1::<f64>::zeros(3),
        };
        let err = baseline_chain_rule_gradient(
            age_entry.view(),
            age_exit.view(),
            age_exit.view(),
            &cfg,
            &residuals,
        )
        .expect_err("length mismatch must error");
        assert!(err.contains("length mismatch"), "err={err}");
    }
}
