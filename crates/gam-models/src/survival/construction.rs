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

use crate::probability::{normal_pdf, standard_normal_quantile};
use crate::survival::location_scale::{
    DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD, ResidualDistribution,
    SurvivalCovariateTermBlockTemplate, SurvivalCovariateTimeBasis,
};
use crate::survival::lognormal_kernel::HazardLoading;
use crate::survival::marginal_slope::DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD;
use crate::wiggle::{monotone_wiggle_basis_with_derivative_order, split_wiggle_penalty_orders};
use gam_linalg::matrix::{
    DenseDesignMatrix, DesignMatrix, SparseDesignMatrix, symmetrize_in_place,
};
use gam_problem::{InverseLink, StandardLink};
use gam_terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
    BasisMetadata, BasisOptions, Dense, ISplineBoundary, KnotSource, OneDimensionalBoundary,
    build_bspline_basis_1d, create_basis, evaluate_bspline_derivative_scalar,
    ispline_modelling_interval, ispline_value, ispline_value_and_first_derivative,
};
use gam_terms::inference::formula_dsl::LinkWiggleFormulaSpec;
use ndarray::{Array1, Array2, Array3, array, s};
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

impl_reason_error_boilerplate! {
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

/// Recover the fitted Weibull baseline from the single-column `log(t)`
/// time-basis coefficient.
///
/// The redundant constant column was dropped at design build (#2301 — it was the
/// intercept-confounded `−shape·log_scale` location, now carried by the mean
/// intercept), so the identified shape is the sole time coefficient `beta[0]` and
/// the identified scale is the anchor itself. The fitted baseline is
/// `shape * (log(t) - log(anchor))`.
pub fn fitted_weibull_baseline_from_linear_time_beta(
    beta: &Array1<f64>,
    anchor: f64,
) -> Option<SurvivalBaselineConfig> {
    if beta.is_empty() {
        return None;
    }
    let shape = beta[0];
    if !shape.is_finite() || shape <= 0.0 || !anchor.is_finite() || anchor <= 0.0 {
        return None;
    }
    Some(SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::Weibull,
        scale: Some(anchor),
        shape: Some(shape),
        rate: None,
        makeham: None,
    })
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
    /// [`crate::survival::location_scale::TimeBlockMonotonicity::StructuralISpline`]
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

/// Every survival likelihood mode, for the cross-mode contracts that must hold
/// for all of them (e.g. the one time-basis anchor rule). Kept exhaustive by
/// `survival_likelihood_modes_is_exhaustive`, which dispatches on the enum so a
/// new variant fails to compile until it is listed here.
pub const SURVIVAL_LIKELIHOOD_MODES: [SurvivalLikelihoodMode; 6] = [
    SurvivalLikelihoodMode::Transformation,
    SurvivalLikelihoodMode::Weibull,
    SurvivalLikelihoodMode::LocationScale,
    SurvivalLikelihoodMode::MarginalSlope,
    SurvivalLikelihoodMode::Latent,
    SurvivalLikelihoodMode::LatentBinary,
];

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

pub fn survival_baseline_theta_from_config(
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<Array1<f64>>, String> {
    let theta = match cfg.target {
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
    };
    if let Some(theta) = theta.as_ref() {
        if theta.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "{} baseline theta coordinates must be finite",
                survival_baseline_targetname(cfg.target)
            ));
        }
        // The inverse chart is also the target-specific domain validator. This
        // keeps the public encoder from emitting coordinates for an invalid
        // config (including when a caller has no data rows to evaluate).
        survival_baseline_config_from_theta(cfg.target, theta)?;
    }
    Ok(theta)
}

pub fn survival_baseline_config_from_theta(
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
}

impl BaselineDerivativeContract {
    /// Apply this contract's derivative declaration, solver class, tolerance,
    /// and iteration budget to a freshly-constructed `OuterProblem`. The
    /// bounds, initial ρ, and seed config are contract-independent and applied
    /// by [`run_baseline_theta_optimizer`].
    fn configure(
        self,
        problem: gam_solve::rho_optimizer::OuterProblem,
    ) -> gam_solve::rho_optimizer::OuterProblem {
        use gam_problem::{DeclaredHessianForm, Derivative};
        match self {
            // BFGS on a 2–3 dim problem with an exact gradient typically
            // converges in 5–10 outer evaluations.
            BaselineDerivativeContract::GradientOnly => problem
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Unavailable)
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
    ) -> Result<gam_problem::OuterEval, crate::model_types::EstimationError>,
{
    use gam_solve::rho_optimizer::OuterProblem;
    let Some(seed) = survival_baseline_theta_from_config(initial)? else {
        return Ok(initial.clone());
    };
    let dim = seed.len();
    let target = initial.target;
    let lower = seed.mapv(|v| v - 6.0);
    let upper = seed.mapv(|v| v + 6.0);
    let problem = contract
        .configure(OuterProblem::new(dim).with_prefer_gradient_only(true))
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
            ) -> Result<gam_problem::EfsEval, crate::model_types::EstimationError>,
        >,
    );
    let result = problem
        .run(&mut obj, context)
        .map_err(|e| format!("{context} failed: {e}"))?;
    if !result.converged() {
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
/// returns a fully-populated [`OuterEval`](gam_problem::OuterEval)
/// (cost + analytic gradient, optionally + analytic Hessian) for a given
/// config. Everything downstream of that — the `Rc<RefCell>` sharing that lets
/// the same user closure back both the `cost_fn` and `eval_fn`, the θ→config
/// conversion, and deriving the scalar `cost_fn` from the eval result — is
/// identical, so it lives here once. The contract-specific axis is only which
/// `HessianValue` the objective embeds, which the wrapper has already encoded
/// in the returned `OuterEval`, so this helper is contract-agnostic beyond the
/// `contract` it forwards to [`run_baseline_theta_optimizer`].
fn run_baseline_theta_optimizer_with_eval<F>(
    initial: &SurvivalBaselineConfig,
    context: &str,
    contract: BaselineDerivativeContract,
    objective: F,
) -> Result<SurvivalBaselineConfig, String>
where
    F: FnMut(&SurvivalBaselineConfig) -> Result<gam_problem::OuterEval, String>,
{
    let target = initial.target;
    let engine_context = context.to_string();
    let objective = std::rc::Rc::new(std::cell::RefCell::new(objective));
    let eval_at = move |obj: &std::rc::Rc<std::cell::RefCell<F>>,
                        theta: &Array1<f64>|
          -> Result<gam_problem::OuterEval, crate::model_types::EstimationError> {
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
        if let gam_problem::HessianValue::Dense(ref h) = eval.hessian {
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
/// `run_baseline_theta_optimizer` under the
/// `BaselineDerivativeContract::GradientOnly` contract, which advertises
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
    use gam_problem::{HessianValue, OuterEval};
    run_baseline_theta_optimizer_with_eval(
        initial,
        context,
        BaselineDerivativeContract::GradientOnly,
        move |cfg| {
            let (cost, gradient) = objective(cfg)?;
            Ok(OuterEval {
                cost,
                gradient,
                hessian: HessianValue::Unavailable,
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

    /// Infer a survival time knot vector, reporting the PUBLIC basis degree
    /// the returned vector can actually carry.
    ///
    /// `build_bspline_basis_1d` may auto-shrink the requested degree when the
    /// data cannot support it (issue #340) -- with 4 rows a degree-4 clamped
    /// vector is not constructible, so it silently returns a degree-3 one and
    /// records that in `BasisMetadata::BSpline1D::degree`. Callers must be told,
    /// or they will hand a shrunk vector to a consumer sized for the degree they
    /// asked for.
    fn infer_survival_time_knots_with_degree(
        combined: &Array1<f64>,
        knot_degree: usize,
        validation_degree: usize,
        num_internal_knots: usize,
        basis_options: BasisOptions,
    ) -> Result<(Array1<f64>, usize), String> {
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
            |placement: gam_terms::basis::BSplineKnotPlacement|
             -> Result<(Array1<f64>, usize), String> {
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
                let (knots, built_degree) = match built.metadata {
                    BasisMetadata::BSpline1D { knots, degree, .. } => {
                        (knots, degree.unwrap_or(knot_degree))
                    }
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
                // The caller's two degrees differ by a fixed raise: `i_spline()`
                // lifts the public degree to its working B-spline antiderivative
                // degree, so `knot_degree == validation_degree + raise`. When the
                // builder shrinks the vector, the public degree has to come down
                // by the same raise or the two stop describing one geometry.
                let raise = knot_degree.saturating_sub(validation_degree);
                let effective_validation_degree = built_degree.saturating_sub(raise);
                create_basis::<Dense>(
                    combined.view(),
                    KnotSource::Provided(knots.view()),
                    effective_validation_degree,
                    basis_options,
                )
                .map_err(|e| e.to_string())?;
                Ok((knots, effective_validation_degree))
            };

        if quantile_knot_inference_needs_uniform_fallback(combined, num_internal_knots) {
            inferwith(gam_terms::basis::BSplineKnotPlacement::Uniform)
        } else {
            inferwith(gam_terms::basis::BSplineKnotPlacement::Quantile)
        }
    }

    /// Knot vector only, for the callers whose consumer degree is the one they
    /// passed in (no i-spline raise, so a shrink cannot desynchronise anything).
    fn infer_survival_time_knots(
        combined: &Array1<f64>,
        knot_degree: usize,
        validation_degree: usize,
        num_internal_knots: usize,
        basis_options: BasisOptions,
    ) -> Result<Array1<f64>, String> {
        infer_survival_time_knots_with_degree(
            combined,
            knot_degree,
            validation_degree,
            num_internal_knots,
            basis_options,
        )
        .map(|(knots, _)| knots)
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
            // Single column `log t` — the Weibull baseline slope (shape). The
            // constant column `[1, ·]` this basis used to carry (#2301) is the
            // `−shape·log_scale` location, which is EXACTLY confounded with the
            // linear-predictor intercept, so it made the converged penalized
            // Hessian singular (the anchor gauge — the killed EDF trace solve and
            // the LM crawl were both downstream of that singularity). Dropping it
            // moves the whole location into the mean intercept and leaves `H`
            // nonsingular. This is valid ONLY because intercept removal (`~ x - 1`)
            // is a typed refusal at `formula_dsl.rs:2456` — the covariate block
            // ALWAYS carries an intercept to absorb the location. If intercept
            // suppression is ever implemented, the Weibull location becomes
            // unidentified without this column and the two features MUST be
            // reconciled here (re-add the constant and pin it, or keep the ban).
            let mut x_entry_time = Array2::<f64>::zeros((n, 1));
            let mut x_exit_time = Array2::<f64>::zeros((n, 1));
            let mut x_derivative_time = Array2::<f64>::zeros((n, 1));
            for i in 0..n {
                x_entry_time[[i, 0]] = log_entry[i];
                x_exit_time[[i, 0]] = log_exit[i];
                x_derivative_time[[i, 0]] = 1.0 / age_exit[i].max(SURVIVAL_TIME_FLOOR);
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

            let nullspace_dims = entry_basis
                .active_penalties
                .iter()
                .map(|penalty| penalty.nullity)
                .collect();
            let penalties = entry_basis
                .active_penalties
                .into_iter()
                .map(|penalty| penalty.matrix)
                .collect();

            Ok(SurvivalTimeBuildOutput {
                x_entry_time: entry_basis.design,
                x_exit_time: exit_basis.design,
                x_derivative_time,
                nullspace_dims,
                penalties,
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
            let requested_bspline_degree = degree
                .checked_add(1)
                .ok_or_else(|| "ispline degree overflow while building knot basis".to_string())?;
            // Every consumer below -- the derivative basis at `bspline_degree`
            // and both i-spline bases at `degree` -- is sized from these two
            // numbers, so they must describe the vector we HAVE rather than the
            // one we asked for. Inference can shrink the degree on data too
            // sparse to carry it (4 rows cannot support degree 4), and the
            // shrunk vector was previously handed to a degree-4 consumer:
            //
            //   Insufficient knots for degree 4 spline: need at least 10 knots
            //   but only 9 were provided.
            //
            // An explicit knot vector is the user's own geometry and is never
            // re-derived, so it keeps the requested degrees.
            let (knotvec, degree, bspline_degree) = if knots.is_empty() {
                let (num_internal_knots, _) = infer_knots_if_needed.ok_or_else(|| {
                    "internal error: ispline time basis requested without knot source".to_string()
                })?;
                let combined = survival_time_knot_input(&log_entry, &log_exit);
                let (knotvec, effective_degree) = infer_survival_time_knots_with_degree(
                    &combined,
                    requested_bspline_degree,
                    degree,
                    num_internal_knots,
                    BasisOptions::i_spline(),
                )?;
                let effective_bspline_degree =
                    effective_degree.checked_add(1).ok_or_else(|| {
                        "ispline degree overflow while building knot basis".to_string()
                    })?;
                (knotvec, effective_degree, effective_bspline_degree)
            } else {
                (knots, degree, requested_bspline_degree)
            };

            // ONE boundary convention for the baseline value AND its slope
            // (gam#2705).
            //
            // The Royston-Parmar baseline is `log Λ(t) = Σ_k γ_k·I_k(log t)`,
            // and the likelihood, the hazard and the predictive surface all read
            // BOTH `I_k` and `I'_k = M_k`. Those two used to be built by
            // different code paths here: the value by the shared I-spline
            // evaluator, which holds `I_k` CONSTANT past the boundary knots, and
            // the slope by a hand-rolled right-cumulative sum of a *clamped*
            // B-spline first-derivative basis, which returns the BOUNDARY SLOPE
            // there because a clamped B-spline's value extends linearly. So
            // outside the fitted knot span the two described different
            // functions, and a saved fit published a FLAT `Λ(t)` next to a
            // NONZERO `h(t) = Λ·d(log Λ)/dt` — measured on the #1564 heart-failure
            // fixture as `Λ ≡ 5.055558` with `t·h(t) ≡ 6.26088` from `t = 285`
            // out to `t = 2.85e6`, i.e. a surviving log-log slope of `1.23842`
            // in the derivative that the value does not have. `h = dΛ/dt`, so a
            // flat `Λ` forces `h = 0`; the two cannot both be the model.
            //
            // The convention is `LinearTails` rather than `Saturate` because
            // that IS the Royston-Parmar model: a *restricted* spline is linear
            // beyond its boundary knots by construction (Royston & Parmar 2002),
            // which gives the classical Weibull-shaped extrapolation
            // `Λ(t) ∝ t^c` used whenever a survival curve is projected past the
            // observed follow-up. Saturating instead asserts two things the data
            // never said: that the hazard drops to exactly zero at the last
            // observed exit time, and — on the lower tail, which
            // `default_survival_time_grid` reaches on its very first node —
            // that `Λ(t) → Λ(t_min) > 0` as `t → 0`, i.e. an atom of failures at
            // time zero and `S(0) < 1`.
            //
            // Nothing about a FIT moves: the knot vector is inferred from
            // `survival_time_knot_input(log_entry, log_exit)`, so every training
            // row is inside `[left, right]` where the two conventions are
            // bit-identical, `keep_cols` is inferred from those same interior
            // rows, and the penalty is built on `log_exit`. What moves is
            // evaluation OUTSIDE the fitted span — prediction grids, entry times
            // below the first knot, and any replay at a fresh time.
            let (x_exit_full, d_exit_log_full) = ispline_value_and_first_derivative(
                log_exit.view(),
                knotvec.view(),
                degree,
                ISplineBoundary::LinearTails,
            )
            .map_err(|e| format!("failed to build ispline exit basis and derivative: {e}"))?;
            // A row that ENTERS AT THE ORIGIN is not a row whose entry time
            // sits below the first knot — it is a row with no left truncation
            // at all, and the likelihood says so: `entry_active` is
            // `age_entry > ENTRY_AT_ORIGIN_THRESHOLD`, and the `S(entry)` factor
            // is dropped outright for the rest (`survival/base.rs`). Its entry
            // design row is therefore never read, and what it holds is a
            // conditioning choice rather than a model statement.
            //
            // `log_entry` for such a row is `ln(SURVIVAL_TIME_FLOOR) = −20.7`,
            // a NUMERICAL FLOOR and not a datum, so a linear tail evaluated
            // there would put a large arbitrary constant into a design column
            // (and into the column-scaling statistics computed from it) purely
            // as a readout of `1e-9`. The saturating basis got the right answer
            // here for the wrong reason: every time at or below the first knot
            // maps to the anchored ZERO row (`I_k(left) = 0` exactly). Keep that
            // answer, and keep it for the reason that holds — the likelihood's
            // own origin predicate — so a genuine delayed entry below the first
            // knot still receives the real extrapolation.
            let interval = ispline_modelling_interval(knotvec.view(), degree)
                .map_err(|e| format!("failed to resolve ispline modelling interval: {e}"))?;
            let mut log_entry_for_basis = log_entry.clone();
            if let Some((left, _right)) = interval {
                for i in 0..n {
                    if age_entry[i] <= crate::survival::base::ENTRY_AT_ORIGIN_THRESHOLD {
                        log_entry_for_basis[i] = left;
                    }
                }
            }
            let x_entry_full = ispline_value(
                log_entry_for_basis.view(),
                knotvec.view(),
                degree,
                ISplineBoundary::LinearTails,
            )
            .map_err(|e| format!("failed to build ispline entry basis: {e}"))?;

            let (x_entry_time, x_exit_time, keep_cols, p_time, p_time_full) = {
                let p_time_full = x_exit_full.ncols();
                if p_time_full == 0 {
                    return Err(SurvivalConstructionError::BasisConstructionFailed {
                        reason: "internal error: empty ispline time basis".to_string(),
                    }
                    .into());
                }
                if d_exit_log_full.ncols() != p_time_full
                    || d_exit_log_full.nrows() != x_exit_full.nrows()
                {
                    return Err(format!(
                        "internal error: ispline time derivative basis is {:?} but its value basis \
                         is {:?}",
                        d_exit_log_full.dim(),
                        x_exit_full.dim()
                    ));
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
                (x_entry_time, x_exit_time, keep_cols, p_time, p_time_full)
            };
            // The full-width VALUE bases are no longer needed; the retained
            // blocks above own their own storage. The full-width derivative is
            // still read below, one row at a time, so it stays.
            drop(x_entry_full);
            drop(x_exit_full);

            // `d(log Λ)/dt = d(log Λ)/d(log t) · 1/t`. The `d/d(log t)` half is
            // the M-spline block the value basis was built with, so no second
            // opinion about the exterior can arise here.
            let mut deriv_triplets = Vec::with_capacity(n * p_time.min(16));
            let mut found_nonfinite: Option<(usize, usize)> = None;
            for i in 0..n {
                let chain = 1.0 / age_exit[i].max(SURVIVAL_TIME_FLOOR);
                for (j_new, &j_old) in keep_cols.iter().enumerate() {
                    let raw_v = d_exit_log_full[[i, j_old]] * chain;
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
            // (the active `penalty_basis` matrix block). The correct curvature penalty
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
            for active_penalty in &penalty_basis.active_penalties {
                let s_mat = &active_penalty.matrix;
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
                    gam_linalg::faer_ndarray::FaerEigh::eigh(s_mat, faer::Side::Lower)
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
                    match gam_linalg::faer_ndarray::FaerEigh::eigh(s_mat, faer::Side::Lower) {
                        Ok((evals, _)) => {
                            let max_ev = evals
                                .iter()
                                .copied()
                                .fold(0.0_f64, |a, b| a.max(b.abs()))
                                .max(1.0);
                            let threshold = 100.0 * (p as f64) * f64::EPSILON * max_ev;
                            evals.iter().filter(|&&e| e <= threshold).count()
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

// ---------------------------------------------------------------------------
// Survival time-basis anchor: ONE rule, three primitives
// ---------------------------------------------------------------------------
//
// `center_survival_time_designs_at_anchor` subtracts the time-basis row at the
// anchor from every entry/exit design row, so the anchor sets the origin of the
// baseline-hazard reparameterization. WHICH time to anchor at is a function of
// exactly three things — the likelihood mode, the entry/exit data, and an
// optional caller override — so it is decided in exactly one place,
// [`resolve_survival_time_anchor_for_mode`], which every front end calls.
//
// It used to be decided in two. `materialize_survival` (the engine path behind
// `fit_from_formula` and the Python FFI) promoted the robust anchor for any
// left-truncated dataset and hardcoded the override to `None`; `gam-cli`'s
// survival path promoted it only for marginal-slope and owned the
// `--survival-time-anchor` override. Three consequences, all of them #2631:
//
//   1. The same formula, data and config produced a DIFFERENT fit depending on
//      which front end ran it — a left-truncated location-scale (or latent)
//      model centered at the robust median exit under `fit_from_formula` and at
//      the earliest entry under the CLI.
//   2. Because the override lived only in the CLI copy, and the CLI's own
//      default (transformation / Weibull) route delegates to the engine copy,
//      `--survival-time-anchor` was SILENTLY IGNORED on the default route.
//   3. `FitRequestConfigDocument` — the "complete scientific model
//      configuration" that `--survival-time-anchor` declares a conflict with —
//      had no field for the anchor at all, so a fit-request document could not
//      express what the flag it excludes expresses.
//
// A rule that lives in two places is a rule that disagrees with itself. The
// anchor is model configuration, not front-end transport, so the override now
// travels on `FitConfig` and the rule below is the only thing that reads it.

/// Validate a caller-supplied survival time-anchor override.
///
/// Honored verbatim by every likelihood mode (subject to the `SURVIVAL_TIME_FLOOR`
/// clamp that keeps `log(anchor)` finite), because a caller who names the anchor
/// is overriding the conditioning heuristic on purpose.
pub fn validate_survival_time_anchor_override(time_anchor: f64) -> Result<f64, String> {
    if !time_anchor.is_finite() || time_anchor < 0.0 {
        return Err(format!(
            "survival time anchor must be finite and non-negative, got {time_anchor}"
        ));
    }
    Ok(time_anchor.max(SURVIVAL_TIME_FLOOR))
}

/// Earliest-entry anchor — the default for data that is NOT left-truncated.
///
/// With every row entering at the time origin this is `≈ 0`, so for the
/// monotone I-spline time basis the anchor row is `≈ 0` and centering is a
/// near-no-op: the historical (pre-#751) behavior is preserved bit-for-bit on
/// ordinary right-censored data.
pub fn survival_earliest_entry_time_anchor(age_entry: &Array1<f64>) -> Result<f64, String> {
    let min_entry = age_entry
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .ok_or_else(|| "survival time anchor requires non-empty entry times".to_string())?;
    Ok(min_entry.max(SURVIVAL_TIME_FLOOR))
}

/// Robust interior anchor — the median exit age, a time on the **exit** scale
/// where the at-risk mass concentrates.
///
/// Under left truncation the earliest entry age is a genuine positive
/// *left-tail* point, and centering the time basis there leaves the centered
/// linear-trend column `X(exit) − X(anchor)` large and one-signed across all
/// rows (every exit sits far to the right of the earliest entry). That column is
/// the unpenalized polynomial null space of the 2nd-difference time penalty, so
/// the inflated one-signed column multiplies the time-block score at the
/// smoothing seed up by hundreds: the marginal-slope constrained joint Newton
/// cannot certify KKT on it and REML rejects every seed (#751), and the
/// transformation (Royston-Parmar) smoothing selection rails a penalty
/// direction and collapses the baseline to a covariate-independent surface with
/// `H` inflated ~10³× and `S(t) ≡ 0` (#1790).
///
/// Centering at the median exit keeps that column small and two-signed (some
/// exits below the median, some above) so the exit-event likelihood pins the
/// linear trend and the seed score stays bounded. The median is chosen over the
/// mean for robustness to the heavy right tail of survival times.
pub fn survival_robust_interior_time_anchor(age_exit: &Array1<f64>) -> Result<f64, String> {
    if age_exit.is_empty() {
        return Err(
            "survival robust interior time anchor requires non-empty exit times".to_string(),
        );
    }
    let mut sorted: Vec<f64> = age_exit.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let m = sorted.len();
    let median = if m % 2 == 1 {
        sorted[m / 2]
    } else {
        0.5 * (sorted[m / 2 - 1] + sorted[m / 2])
    };
    Ok(median.max(SURVIVAL_TIME_FLOOR))
}

/// The single definition of "this dataset is genuinely left-truncated".
///
/// **Any** row entering above `ENTRY_AT_ORIGIN_THRESHOLD` makes the data
/// left-truncated, not just the earliest one. Staggered entry — part of the
/// cohort observed from the time origin, the rest joining at positive delayed
/// entry times — is the ordinary shape of a real registry cohort, and it
/// exhibits the #751/#1790 inflation just as fully-delayed entry does: the
/// earliest-entry anchor is then `≈ 0`, the anchor row of a `log t` basis is
/// evaluated at `SURVIVAL_TIME_FLOOR`, and every centered exit column is
/// one-signed and large. Testing `min(entry) > threshold` instead would
/// under-trigger on exactly that shape, which is why the two former copies of
/// this predicate (`any` in the materializer, `min` inside the transformation
/// resolver) had to be collapsed to one.
///
/// The threshold is the likelihood engines' own origin convention, so "this row
/// has a delayed-entry interval" and "this dataset is left-truncated" cannot
/// drift apart.
pub fn survival_data_is_left_truncated(age_entry: &Array1<f64>) -> bool {
    age_entry
        .iter()
        .any(|&entry| entry > crate::survival::base::ENTRY_AT_ORIGIN_THRESHOLD)
}

/// **The** survival time-basis anchor rule. Every front end calls this and
/// nothing else.
///
/// * An explicit `time_anchor` wins, in every mode.
/// * Marginal-slope always takes the robust interior anchor: its `γ = 0`
///   monotone-cone seed is where #751 was measured, and the fix was applied
///   there unconditionally.
/// * Every other time-basis-carrying likelihood takes the robust interior
///   anchor **iff the data is genuinely left-truncated**. Ordinary
///   right-censored data keeps the earliest-entry anchor, which is `≈` the time
///   origin, so centering stays a near-no-op and pre-#751 behavior is preserved
///   bit-for-bit.
///
/// Re-centering is an exact affine reparameterization of the baseline offset, so
/// this choice does not change the model being fitted — only the frame the
/// smoothing selection sees it in, and the frame the saved
/// `survival_time_anchor` must replay in.
pub fn resolve_survival_time_anchor_for_mode(
    survival_mode: SurvivalLikelihoodMode,
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    time_anchor: Option<f64>,
) -> Result<f64, String> {
    if let Some(explicit) = time_anchor {
        return validate_survival_time_anchor_override(explicit);
    }
    if survival_mode == SurvivalLikelihoodMode::MarginalSlope
        || survival_data_is_left_truncated(age_entry)
    {
        survival_robust_interior_time_anchor(age_exit)
    } else {
        survival_earliest_entry_time_anchor(age_entry)
    }
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
        // Single `log t` column (#2301): the confounded constant column is gone,
        // its location absorbed by the mean intercept. See the Linear arm of
        // `build_survival_time_basis`.
        SurvivalTimeBasisConfig::Linear => Ok(array![age.ln()]),
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
            // The anchor is the ORIGIN of the baseline reparameterization, not
            // a prediction: `center_survival_time_designs_at_anchor` subtracts
            // this row from every entry and exit design row, and the fit is
            // invariant to it up to the baseline offset. So it has to be a time
            // the baseline is IDENTIFIED at.
            //
            // The default anchor for ordinary right-censored data is the
            // earliest entry, which is the time origin, which
            // `evaluate_survival_time_basis_row` floors to
            // `SURVIVAL_TIME_FLOOR = 1e-9` so `ln` stays finite — i.e. `−20.7`
            // in the basis's own coordinate, far below the first knot and a
            // readout of the floor rather than of the data. Under the saturating
            // convention that was invisible, because every time at or below the
            // first knot maps to the anchored ZERO row. Under the linear tails
            // the baseline now carries (gam#2705) it would instead re-center
            // every design column by a large constant — which is exactly the
            // #751 inflation the anchor rule exists to avoid.
            //
            // Clamping the anchor into the modelling interval says the thing
            // that is actually meant, and is numerically identical to what
            // shipped for every anchor at or below the first knot.
            let interval = ispline_modelling_interval(knots.view(), *degree)
                .map_err(|e| format!("failed to resolve ispline modelling interval: {e}"))?;
            let anchor_log_age = match interval {
                Some((left, right)) => array![log_age[0].clamp(left, right)],
                None => log_age.clone(),
            };
            let (basis_arc, _) = create_basis::<Dense>(
                anchor_log_age.view(),
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
    residuals: &crate::survival::OffsetChannelResiduals,
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
///     = Σᵢ (∂NLL_i/∂o_X\[i\])·(∂o_X_i/∂θ_k)
///       + (∂NLL_i/∂o_E\[i\])·(∂o_E_i/∂θ_k)
///       + (∂NLL_i/∂o_D\[i\])·(∂o_D_i/∂θ_k)
///       + (∂NLL_i/∂o_R\[i\])·(∂o_R_i/∂θ_k)
///
/// The four `∂NLL_i/∂o_channel` terms are the `exit`, `entry`, `derivative`,
/// `right` fields of `OffsetChannelResiduals` (sampleweight-scaled already).
/// The `∂o/∂θ_k` terms come from [`baseline_offset_theta_partials`] per obs at
/// the appropriate age.
///
/// Per the RP offset convention:
///   o_E\[i\] = eta_target(age_entry\[i\])
///   o_X\[i\] = eta_target(age_exit\[i\])
///   o_D\[i\] = d/dt eta_target(t) |_{t=age_exit\[i\]}
///   o_R\[i\] = eta_target(age_right\[i\])   (interval upper bound `R`; η-level only)
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
    residuals: &crate::survival::OffsetChannelResiduals,
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
    residuals: &crate::survival::OffsetChannelResiduals,
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

/// Complete analytic baseline chart at one age for survival marginal-slope.
///
/// `value` is `(q(t), dq(t)/dt)`. `first[k]` and `second[k][l]` are the
/// corresponding first and second partials with respect to the nonlinear
/// baseline coordinates returned by [`survival_baseline_theta_from_config`].
/// A nonlinear baseline evaluated at the survival origin has identically zero
/// value and derivatives because origin rows are anchored outside the finite
/// probit chart. Linear baselines have no coordinates and return `None`.
#[derive(Clone, Debug)]
pub struct MarginalSlopeBaselineOffsetThetaGeometry {
    pub value: (f64, f64),
    pub first: Vec<(f64, f64)>,
    pub second: Vec<Vec<(f64, f64)>>,
}

pub fn marginal_slope_baseline_offset_theta_geometry(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<MarginalSlopeBaselineOffsetThetaGeometry>, String> {
    if age == 0.0 {
        let Some(theta) = survival_baseline_theta_from_config(cfg)? else {
            return Ok(None);
        };
        let dim = theta.len();
        return Ok(Some(MarginalSlopeBaselineOffsetThetaGeometry {
            value: (0.0, 0.0),
            first: vec![(0.0, 0.0); dim],
            second: vec![vec![(0.0, 0.0); dim]; dim],
        }));
    }
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
        for j in i..dim {
            let (h_i, inst_i) = first[i];
            let (h_j, inst_j) = first[j];
            let (h_ij, inst_ij) = second[i][j];
            let a_j = a * b * h_j;
            let b_j = a * h_j * b_factor;
            let q_ij = a * h_ij + a * b * h_i * h_j;
            let qt_inner_i = inst_i + instant_hazard * b * h_i;
            let qt_ij = a_j * qt_inner_i
                + a * (inst_ij + inst_j * b * h_i + instant_hazard * (b_j * h_i + b * h_ij));
            let mixed = (q_ij, qt_ij);
            second_out[i][j] = mixed;
            second_out[j][i] = mixed;
        }
    }
    Ok(Some(MarginalSlopeBaselineOffsetThetaGeometry {
        value: (point.q, point.q_t),
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

/// Rowwise value, gradient, and Hessian of the complete marginal-slope time
/// offset with respect to a nonlinear survival-baseline chart.
///
/// The first-derivative arrays have shape `n × d` and the second-derivative
/// arrays have shape `n × d × d`, where `d = theta.len()`. The value arrays may
/// include a frozen non-baseline residual; that residual has zero derivatives.
#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeOffsetGeometry {
    pub baseline_config: SurvivalBaselineConfig,
    pub theta: Array1<f64>,
    pub offset_entry: Array1<f64>,
    pub offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    pub offset_entry_theta_first: Array2<f64>,
    pub offset_exit_theta_first: Array2<f64>,
    pub derivative_offset_exit_theta_first: Array2<f64>,
    pub offset_entry_theta_second: Array3<f64>,
    pub offset_exit_theta_second: Array3<f64>,
    pub derivative_offset_exit_theta_second: Array3<f64>,
}

fn validate_marginal_slope_baseline_row_geometry(
    row: &MarginalSlopeBaselineOffsetThetaGeometry,
    dim: usize,
    channel: &str,
) -> Result<(), String> {
    if row.first.len() != dim
        || row.second.len() != dim
        || row.second.iter().any(|axis| axis.len() != dim)
    {
        return Err(format!(
            "survival marginal-slope baseline {channel} theta dimension drifted"
        ));
    }
    if !row.value.0.is_finite()
        || !row.value.1.is_finite()
        || row
            .first
            .iter()
            .any(|&(value, derivative)| !value.is_finite() || !derivative.is_finite())
        || row
            .second
            .iter()
            .flatten()
            .any(|&(value, derivative)| !value.is_finite() || !derivative.is_finite())
    {
        return Err(format!(
            "survival marginal-slope baseline {channel} geometry must be finite"
        ));
    }
    Ok(())
}

/// Evaluate the nonlinear parametric baseline on every marginal-slope row.
///
/// This function evaluates only baseline-dependent offset geometry. It never
/// constructs or mutates time designs, wiggle knots, penalties, or linear
/// constraints. Linear baselines have no hyperparameter chart and return
/// `None`.
pub fn build_survival_marginal_slope_baseline_geometry(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<SurvivalMarginalSlopeOffsetGeometry>, String> {
    let Some(theta) = survival_baseline_theta_from_config(cfg)? else {
        // A linear baseline has no hyperparameter chart. The length check below
        // is not reached in that case, and must not be: this arm is the
        // "no chart" answer, not an error.
        if age_entry.len() != age_exit.len() {
            return Err(
                "survival marginal-slope baseline geometry requires matching entry/exit lengths"
                    .to_string(),
            );
        }
        return Ok(None);
    };
    build_survival_marginal_slope_baseline_geometry_at_theta(age_entry, age_exit, cfg, theta)
}

/// The θ-authored entry: realize the same geometry, but record the caller's own
/// `theta` VERBATIM instead of re-deriving it from `cfg`.
///
/// The two differ, and the difference is not academic. A chart evaluation is
/// `θ → cfg → rows`, and the config-authored entry above closes the loop with
/// `cfg → θ`. For a Weibull that loop is `ln(exp(θ))`, which is **not** the
/// identity in `f64`: measured over a grid on `[-3, 3]`, **17.3%** of
/// coordinates come back a ulp or more away, and `θ = 1e-5` comes back 57 269
/// ulps away.
///
/// `SurvivalMarginalSlopeFamilyHyperState` stores this `theta` as the family's
/// realized coordinates and `validate_layout` compares them to the outer
/// manifest with `to_bits()` equality — deliberately, so a workspace cannot
/// reuse row geometry from a neighbouring outer probe. With a re-derived `θ`
/// that exactness invariant fails for reasons that have nothing to do with the
/// geometry, and the inner solve refuses a point the outer optimizer is merely
/// trying to evaluate. See the #2765 measurement: at the acceptance fixture's
/// checkpoint the certificate probe refused coordinate 3 side `−` (round trip
/// `+1` ulp) and coordinate 4 on BOTH sides (`−57269` and `+3383` ulps), and
/// evaluated cleanly everywhere the round trip happened to be exact.
///
/// So: when a caller HAS a θ, that θ is the authority. `cfg` still drives every
/// row's arithmetic; only the recorded coordinates change.
pub fn build_survival_marginal_slope_baseline_geometry_at_theta(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
    theta: Array1<f64>,
) -> Result<Option<SurvivalMarginalSlopeOffsetGeometry>, String> {
    if age_entry.len() != age_exit.len() {
        return Err(
            "survival marginal-slope baseline geometry requires matching entry/exit lengths"
                .to_string(),
        );
    }
    if theta.iter().any(|value| !value.is_finite()) {
        return Err(
            "survival marginal-slope baseline theta coordinates must be finite".to_string(),
        );
    }
    // Round-trip through the public chart decoder before touching row storage.
    // This validates every target-specific config value even when `n == 0`.
    survival_baseline_config_from_theta(cfg.target, &theta)?;
    let dim = theta.len();
    let zero = || MarginalSlopeBaselineOffsetThetaGeometry {
        value: (0.0, 0.0),
        first: vec![(0.0, 0.0); dim],
        second: vec![vec![(0.0, 0.0); dim]; dim],
    };
    let rows = (0..age_exit.len())
        .into_par_iter()
        .map(
            |row_index| -> Result<
                (
                    MarginalSlopeBaselineOffsetThetaGeometry,
                    MarginalSlopeBaselineOffsetThetaGeometry,
                ),
                String,
            > {
                let entry_age = age_entry[row_index];
                if !entry_age.is_finite() || entry_age < 0.0 {
                    return Err(format!(
                        "survival marginal-slope entry age must be finite and non-negative at row {row_index}"
                    ));
                }
                let exit_age = age_exit[row_index];
                if !exit_age.is_finite() || exit_age < 0.0 {
                    return Err(format!(
                        "survival marginal-slope exit age must be finite and non-negative at row {row_index}"
                    ));
                }
                let entry = if entry_age == 0.0 {
                    zero()
                } else {
                    marginal_slope_baseline_offset_theta_geometry(entry_age, cfg)?.ok_or_else(
                        || {
                            "nonlinear survival baseline unexpectedly has no entry geometry"
                                .to_string()
                        },
                    )?
                };
                let exit = marginal_slope_baseline_offset_theta_geometry(exit_age, cfg)?
                    .ok_or_else(|| {
                        "nonlinear survival baseline unexpectedly has no exit geometry".to_string()
                    })?;
                validate_marginal_slope_baseline_row_geometry(&entry, dim, "entry")?;
                validate_marginal_slope_baseline_row_geometry(&exit, dim, "exit")?;
                Ok((entry, exit))
            },
        )
        .collect::<Result<Vec<_>, String>>()?;

    let n = rows.len();
    let mut offset_entry = Array1::<f64>::zeros(n);
    let mut offset_exit = Array1::<f64>::zeros(n);
    let mut derivative_offset_exit = Array1::<f64>::zeros(n);
    let mut offset_entry_theta_first = Array2::<f64>::zeros((n, dim));
    let mut offset_exit_theta_first = Array2::<f64>::zeros((n, dim));
    let mut derivative_offset_exit_theta_first = Array2::<f64>::zeros((n, dim));
    let mut offset_entry_theta_second = Array3::<f64>::zeros((n, dim, dim));
    let mut offset_exit_theta_second = Array3::<f64>::zeros((n, dim, dim));
    let mut derivative_offset_exit_theta_second = Array3::<f64>::zeros((n, dim, dim));
    for (row_index, (entry, exit)) in rows.into_iter().enumerate() {
        offset_entry[row_index] = entry.value.0;
        offset_exit[row_index] = exit.value.0;
        derivative_offset_exit[row_index] = exit.value.1;
        for axis in 0..dim {
            offset_entry_theta_first[[row_index, axis]] = entry.first[axis].0;
            offset_exit_theta_first[[row_index, axis]] = exit.first[axis].0;
            derivative_offset_exit_theta_first[[row_index, axis]] = exit.first[axis].1;
            for other_axis in 0..dim {
                offset_entry_theta_second[[row_index, axis, other_axis]] =
                    entry.second[axis][other_axis].0;
                offset_exit_theta_second[[row_index, axis, other_axis]] =
                    exit.second[axis][other_axis].0;
                derivative_offset_exit_theta_second[[row_index, axis, other_axis]] =
                    exit.second[axis][other_axis].1;
            }
        }
    }
    Ok(Some(SurvivalMarginalSlopeOffsetGeometry {
        baseline_config: cfg.clone(),
        theta,
        offset_entry,
        offset_exit,
        derivative_offset_exit,
        offset_entry_theta_first,
        offset_exit_theta_first,
        derivative_offset_exit_theta_first,
        offset_entry_theta_second,
        offset_exit_theta_second,
        derivative_offset_exit_theta_second,
    }))
}

/// A nonlinear baseline chart over already-prepared marginal-slope offsets.
///
/// Construction subtracts the initial parametric baseline from the prepared
/// offset channels exactly once. Candidate evaluations add a new baseline to
/// that same frozen residual. Consequently candidate theta values cannot move
/// any prepared time design, wiggle knot, penalty, or feasibility cone; only
/// the three row-offset value channels move, with analytic first and second
/// derivatives supplied by the same evaluation.
#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeFrozenOffsetChart {
    age_entry: Array1<f64>,
    age_exit: Array1<f64>,
    target: SurvivalBaselineTarget,
    initial_theta: Array1<f64>,
    lower_theta: Array1<f64>,
    upper_theta: Array1<f64>,
    fixed_offset_entry: Array1<f64>,
    fixed_offset_exit: Array1<f64>,
    fixed_derivative_offset_exit: Array1<f64>,
}

impl SurvivalMarginalSlopeFrozenOffsetChart {
    pub fn new(
        age_entry: &Array1<f64>,
        age_exit: &Array1<f64>,
        initial_config: &SurvivalBaselineConfig,
        prepared_offset_entry: &Array1<f64>,
        prepared_offset_exit: &Array1<f64>,
        prepared_derivative_offset_exit: &Array1<f64>,
    ) -> Result<Self, String> {
        let n = age_exit.len();
        if age_entry.len() != n
            || prepared_offset_entry.len() != n
            || prepared_offset_exit.len() != n
            || prepared_derivative_offset_exit.len() != n
        {
            return Err(format!(
                "survival marginal-slope frozen offset chart length mismatch: entry={}, exit={n}, prepared_entry={}, prepared_exit={}, prepared_derivative={}",
                age_entry.len(),
                prepared_offset_entry.len(),
                prepared_offset_exit.len(),
                prepared_derivative_offset_exit.len(),
            ));
        }
        if prepared_offset_entry
            .iter()
            .chain(prepared_offset_exit.iter())
            .chain(prepared_derivative_offset_exit.iter())
            .any(|value| !value.is_finite())
        {
            return Err(
                "survival marginal-slope prepared offsets must be finite before freezing"
                    .to_string(),
            );
        }
        let initial_geometry =
            build_survival_marginal_slope_baseline_geometry(age_entry, age_exit, initial_config)?
                .ok_or_else(|| {
                String::from(
                    "survival marginal-slope frozen offset chart requires a nonlinear baseline",
                )
            })?;
        let lower_theta = initial_geometry.theta.mapv(|value| value - 6.0);
        let upper_theta = initial_geometry.theta.mapv(|value| value + 6.0);
        Ok(Self {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            target: initial_config.target,
            initial_theta: initial_geometry.theta,
            lower_theta,
            upper_theta,
            fixed_offset_entry: prepared_offset_entry - &initial_geometry.offset_entry,
            fixed_offset_exit: prepared_offset_exit - &initial_geometry.offset_exit,
            fixed_derivative_offset_exit: prepared_derivative_offset_exit
                - &initial_geometry.derivative_offset_exit,
        })
    }

    pub fn target(&self) -> SurvivalBaselineTarget {
        self.target
    }

    pub fn initial_theta(&self) -> &Array1<f64> {
        &self.initial_theta
    }

    /// Declared finite domain of this frozen nonlinear chart. These are the
    /// same coordinate bounds used by the legacy standalone baseline solver,
    /// now owned by the chart so a joint solver and its terminal certificate
    /// cannot silently choose a different domain.
    pub fn theta_bounds(&self) -> (&Array1<f64>, &Array1<f64>) {
        (&self.lower_theta, &self.upper_theta)
    }

    pub fn evaluate(
        &self,
        theta: &Array1<f64>,
    ) -> Result<SurvivalMarginalSlopeOffsetGeometry, String> {
        let config = survival_baseline_config_from_theta(self.target, theta)?;
        // θ-authored: this chart was ASKED to realize `theta`, so `theta` is
        // what the geometry records. Re-deriving it from `config` closes a
        // `ln(exp(·))` loop that is not the identity in `f64`, and the family's
        // `to_bits()` manifest check then refuses the point (#2765).
        let mut geometry = build_survival_marginal_slope_baseline_geometry_at_theta(
            &self.age_entry,
            &self.age_exit,
            &config,
            theta.clone(),
        )?
        .ok_or_else(|| {
            "survival marginal-slope nonlinear baseline chart lost its theta coordinates"
                .to_string()
        })?;
        geometry.offset_entry += &self.fixed_offset_entry;
        geometry.offset_exit += &self.fixed_offset_exit;
        geometry.derivative_offset_exit += &self.fixed_derivative_offset_exit;
        Ok(geometry)
    }
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

/// Resolve the actual parametric offset chart used by a marginal-slope fit.
///
/// A nominal `Linear` target has zero derivative and therefore starts the
/// `-log(q')` barrier exactly on its guard.  Fitting consequently uses a
/// deterministic exponential-survival (Weibull shape one) offset at the
/// data-scale mean positive exit time.  This function is the shared authority
/// for fitting and persistence: saving the nominal `Linear` request would not
/// be enough to replay the fitted row likelihood.
pub fn survival_marginal_slope_offset_baseline_config(
    age_exit: &Array1<f64>,
    requested: &SurvivalBaselineConfig,
) -> SurvivalBaselineConfig {
    if requested.target == SurvivalBaselineTarget::Linear {
        SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(positive_survival_time_seed(age_exit)),
            shape: Some(1.0),
            rate: None,
            makeham: None,
        }
    } else {
        requested.clone()
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
    // Use the smallest requested derivative order as the primary exact
    // function-space roughness so the fitted penalty system matches the public
    // formula exactly, including the slope (`order = 1`) case.
    let (primary_order, extra_orders) = split_wiggle_penalty_orders(2, &cfg.penalty_orders)?;
    let mut derivative_orders = Vec::with_capacity(1 + extra_orders.len());
    derivative_orders.push(primary_order);
    derivative_orders.extend(extra_orders);
    // A FIXED-index warp: `h = h_base + B(h_base)·β_w` composes onto the
    // BASELINE offsets, which are computed from time and do not move with β. So
    // the evaluation point never crosses a knot during a solve and the boundary
    // knot's multiplicity is invisible — this block keeps the clamped generator
    // rather than the simple-ended warp one (gam#2695).
    let knots = gam_terms::basis::initializewiggle_knots_from_seed(
        seed.view(),
        cfg.degree,
        cfg.num_internal_knots,
    )?;
    // One assembly for the WHOLE order list (gam#2647). The gauge-closure
    // coordinate is a property of the assembled set, so building the primary
    // order and appending the rest would decide it on a partial set.
    let combined_block = crate::wiggle::buildwiggle_block_input_from_orders(
        seed.view(),
        &knots,
        cfg.degree,
        &derivative_orders,
        cfg.double_penalty,
    )?;
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

    let log_exit = age_exit.mapv(|t| t.max(1e-12).ln());

    let time_spec = BSplineBasisSpec {
        degree: time_degree,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(num_internal_knots),
            placement: gam_terms::basis::BSplineKnotPlacement::Quantile,
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

    let time_penalties = time_build
        .active_penalties
        .into_iter()
        .map(|penalty| penalty.matrix)
        .collect();

    finish_time_varying_survival_covariate_template(
        age_entry,
        age_exit,
        time_degree,
        knots,
        time_design_exit,
        time_penalties,
        block_name,
    )
}

/// Replay a fit-time threshold/log-scale time margin from its resolved knots.
/// Prediction and saved ALO use this path so the prediction sample can never
/// move the spline basis by re-estimating quantile knots.
pub fn replay_time_varying_survival_covariate_template(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    time_basis: &SurvivalCovariateTimeBasis,
    block_name: &str,
) -> Result<SurvivalCovariateTermBlockTemplate, String> {
    let log_exit = age_exit.mapv(|t| t.max(1e-12).ln());
    let knots = Array1::from_vec(time_basis.knots.clone());
    let time_build = build_bspline_basis_1d(
        log_exit.view(),
        &BSplineBasisSpec {
            degree: time_basis.degree,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Provided(knots.clone()),
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: BSplineBoundaryConditions::default(),
        },
    )
    .map_err(|e| format!("failed to replay {block_name} time-margin B-spline basis: {e}"))?;
    let time_design_exit = time_build.design.to_dense();
    let time_penalties = time_build
        .active_penalties
        .into_iter()
        .map(|penalty| penalty.matrix)
        .collect();
    finish_time_varying_survival_covariate_template(
        age_entry,
        age_exit,
        time_basis.degree,
        knots,
        time_design_exit,
        time_penalties,
        block_name,
    )
}

/// The slope block's follow-up margin evaluated at arbitrary times
/// (gam#2765, gam#2767).
///
/// The margin is a B-spline in `log t` on the fit's own knots. At fit time the
/// knots are placed by quantile and the design is a by-product of that build; at
/// prediction time the knots are read back from the saved model and the SAME
/// spec is re-evaluated, so a prediction sample can never move the basis. This
/// is the only B-spline evaluation on the slope time axis, so the batch
/// replay below and the per-`(row, t)` survival-curve replay cannot disagree
/// about which basis they are asking for.
pub struct SlopeTimeMarginRows {
    pub value: Array2<f64>,
    pub derivative: Array2<f64>,
}

pub fn slope_time_margin_rows(
    time_basis: &SurvivalCovariateTimeBasis,
    times: ndarray::ArrayView1<'_, f64>,
) -> Result<SlopeTimeMarginRows, String> {
    let log_times = times.mapv(|t| t.max(1e-12).ln());
    let knots = Array1::from_vec(time_basis.knots.clone());
    let p_time = knots
        .len()
        .checked_sub(time_basis.degree + 1)
        .ok_or_else(|| {
            format!(
                "the saved slope time margin has {} knots, insufficient for degree {}",
                knots.len(),
                time_basis.degree,
            )
        })?;
    if p_time == 0 {
        return Err("the replayed slope time margin has zero columns".to_string());
    }
    let mut derivative = Array2::<f64>::zeros((times.len(), p_time));
    derivative
        .as_slice_mut()
        .expect("zeros are contiguous")
        .chunks_mut(p_time)
        .enumerate()
        .try_for_each(|(row, output)| -> Result<(), String> {
            let mut derivative_log_time = vec![0.0_f64; p_time];
            evaluate_bspline_derivative_scalar(
                log_times[row],
                knots.view(),
                time_basis.degree,
                &mut derivative_log_time,
            )
            .map_err(|error| format!("failed to replay the slope time-margin tangent: {error}"))?;
            let log_time_tangent = 1.0 / times[row].max(1e-12);
            for column in 0..p_time {
                output[column] = derivative_log_time[column] * log_time_tangent;
            }
            Ok(())
        })?;
    let build = build_bspline_basis_1d(
        log_times.view(),
        &BSplineBasisSpec {
            degree: time_basis.degree,
            penalty_order: 2,
            knotspec: BSplineKnotSpec::Provided(knots),
            double_penalty: false,
            identifiability: BSplineIdentifiability::None,
            boundary: OneDimensionalBoundary::Open,
            boundary_conditions: BSplineBoundaryConditions::default(),
        },
    )
    .map_err(|e| format!("failed to replay the slope time margin: {e}"))?;
    let value = build.design.to_dense();
    if value.ncols() != p_time {
        return Err(format!(
            "the replayed slope time margin has {} columns but its knot geometry requires {p_time}",
            value.ncols(),
        ));
    }
    Ok(SlopeTimeMarginRows { value, derivative })
}

/// All three follow-up channels of a slope block, replayed from the saved
/// margin (gam#2765, gam#2767).
///
/// The row program reads the slope at three places — the row's entry time, its
/// exit time, and the exit-time rate — because the likelihood is
/// `log S(t₁) − log S(t₀)` and an event row also carries `log η′(t₁)`. Any
/// consumer that re-evaluates that program off a saved model (the leave-one-out
/// replay, above all) needs all three; handing it the exit design alone would
/// silently evaluate a time-CONSTANT slope, which is a different model.
pub struct SlopeFollowUpReplayDesigns {
    pub entry: DesignMatrix,
    pub exit: DesignMatrix,
    pub derivative_exit: DesignMatrix,
}

/// Replay every follow-up channel of a slope block from its saved margin.
pub fn replay_slope_follow_up_designs(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    time_basis: &SurvivalCovariateTimeBasis,
    covariate_design: &DesignMatrix,
) -> Result<SlopeFollowUpReplayDesigns, String> {
    if age_entry.len() != age_exit.len() {
        return Err(format!(
            "slope follow-up replay has {} entry rows against {} exit rows",
            age_entry.len(),
            age_exit.len(),
        ));
    }
    let entry_rows = slope_time_margin_rows(time_basis, age_entry.view())?;
    let exit_rows = slope_time_margin_rows(time_basis, age_exit.view())?;
    if covariate_design.nrows() != exit_rows.value.nrows() {
        return Err(format!(
            "slope follow-up replay has {} covariate rows against {} time rows",
            covariate_design.nrows(),
            exit_rows.value.nrows(),
        ));
    }
    if covariate_design.ncols() == 0 || exit_rows.value.ncols() == 0 {
        return Err(format!(
            "a follow-up-varying slope needs a non-empty tensor product, got {}x{}",
            covariate_design.ncols(),
            exit_rows.value.ncols(),
        ));
    }
    let kron = |basis: &Array2<f64>| {
        crate::survival::location_scale::rowwise_kronecker(covariate_design, basis)
    };
    Ok(SlopeFollowUpReplayDesigns {
        entry: kron(&entry_rows.value),
        exit: kron(&exit_rows.value),
        derivative_exit: kron(&exit_rows.derivative),
    })
}

/// The slope block's fitted design, rebuilt from the covariate factor and
/// the fit's own resolved time margin (gam#2765, gam#2767).
///
/// A follow-up-varying slope block does not own the covariate design its
/// term spec describes; it owns the row-wise Kronecker product
/// `X_cov ⊗ᵣ B(log t)`, evaluated at the time each row's slope is being read
/// at. At fit time that is the row's EXIT time, which is the convention the
/// block's `ParameterBlockSpec` eta already uses; at prediction time it is
/// whichever time the survival curve is being evaluated at, because `b(t)` moves
/// along the curve exactly as `q(t)` does.
///
/// Rebuilding it from the term spec alone — `p_cov` columns against a
/// `p_cov · p_time` coefficient vector — is the failure this function exists to
/// make impossible.
pub struct SlopeTimeMarginReplayDesign {
    pub value: DesignMatrix,
    pub derivative: DesignMatrix,
}

pub fn replay_slope_time_margin_value_tangent_design(
    times: ndarray::ArrayView1<'_, f64>,
    time_basis: &SurvivalCovariateTimeBasis,
    covariate_design: &DesignMatrix,
) -> Result<SlopeTimeMarginReplayDesign, String> {
    if covariate_design.nrows() != times.len() {
        return Err(format!(
            "slope time-margin replay has {} covariate rows against {} times",
            covariate_design.nrows(),
            times.len(),
        ));
    }
    let time_rows = slope_time_margin_rows(time_basis, times)?;
    if covariate_design.ncols() == 0 || time_rows.value.ncols() == 0 {
        return Err(format!(
            "a follow-up-varying slope needs a non-empty tensor product, got {}x{}",
            covariate_design.ncols(),
            time_rows.value.ncols(),
        ));
    }
    let tensor = |basis: &Array2<f64>| {
        crate::survival::location_scale::rowwise_kronecker(covariate_design, basis)
    };
    Ok(SlopeTimeMarginReplayDesign {
        value: tensor(&time_rows.value),
        derivative: tensor(&time_rows.derivative),
    })
}

fn finish_time_varying_survival_covariate_template(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    time_degree: usize,
    knots: Array1<f64>,
    time_design_exit: Array2<f64>,
    time_penalties: Vec<Array2<f64>>,
    block_name: &str,
) -> Result<SurvivalCovariateTermBlockTemplate, String> {
    if age_entry.len() != age_exit.len() {
        return Err(format!(
            "{block_name} time-margin entry/exit row mismatch: {} versus {}",
            age_entry.len(),
            age_exit.len()
        ));
    }
    let log_entry = age_entry.mapv(|t| t.max(1e-12).ln());
    let log_exit = age_exit.mapv(|t| t.max(1e-12).ln());
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
    if p_time == 0 {
        return Err(format!(
            "{block_name} time-margin basis resolved to zero columns"
        ));
    }
    let mut time_design_derivative_exit = Array2::<f64>::zeros((age_exit.len(), p_time));
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
        time_basis: SurvivalCovariateTimeBasis {
            degree: time_degree,
            knots: knots.to_vec(),
        },
        time_basis_entry: time_design_entry,
        time_basis_exit: time_design_exit,
        time_basis_derivative_exit: time_design_derivative_exit,
        time_penalties,
    })
}

#[cfg(test)]
mod tests {
    use super::{SURVIVAL_LIKELIHOOD_MODES, SURVIVAL_TIME_FLOOR, SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalLikelihoodMode, SurvivalMarginalSlopeFrozenOffsetChart, SurvivalTimeBasisConfig, baseline_chain_rule_gradient, baseline_offset_theta_partials, build_survival_marginal_slope_baseline_geometry, build_survival_marginal_slope_baseline_offsets, build_survival_time_basis, build_survival_timewiggle_from_baseline, evaluate_survival_baseline, evaluate_survival_marginal_slope_baseline, fitted_weibull_baseline_from_linear_time_beta, gompertz_cumulative_shape_derivative, gompertz_cumulative_shape_second_derivative, gompertz_hazard_components, marginal_slope_baseline_chain_rule_gradient, marginal_slope_baseline_offset_theta_partials, resolve_survival_time_anchor_for_mode, survival_baseline_config_from_theta, survival_baseline_theta_from_config, survival_data_is_left_truncated, survival_earliest_entry_time_anchor, survival_robust_interior_time_anchor, validate_survival_time_anchor_override};
    use super::{
        center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
        resolved_survival_time_basis_config_from_build,
    };
    use super::{
        DesignMatrix, SurvivalCovariateTermBlockTemplate,
        build_time_varying_survival_covariate_template, slope_time_margin_rows,
        replay_slope_follow_up_designs, replay_slope_time_margin_value_tangent_design,
    };
    use crate::probability::normal_cdf;
    use crate::survival::base::ENTRY_AT_ORIGIN_THRESHOLD;
    use crate::survival::OffsetChannelResiduals;
    use gam_terms::inference::formula_dsl::LinkWiggleFormulaSpec;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn fitted_weibull_baseline_uses_identified_anchor_and_slope() {
        // Single-column time basis (#2301): the shape is the sole coefficient.
        let fitted = fitted_weibull_baseline_from_linear_time_beta(&array![1.75], 4.5)
            .expect("valid Weibull baseline");
        assert_eq!(fitted.target, SurvivalBaselineTarget::Weibull);
        assert_eq!(fitted.scale, Some(4.5));
        assert_eq!(fitted.shape, Some(1.75));
        assert_eq!(fitted.rate, None);
        assert_eq!(fitted.makeham, None);

        // Empty coefficient vector: no slope to recover.
        assert!(
            fitted_weibull_baseline_from_linear_time_beta(&Array1::<f64>::zeros(0), 4.5).is_none()
        );
        // Non-positive shape is not a valid Weibull baseline.
        assert!(fitted_weibull_baseline_from_linear_time_beta(&array![0.0], 4.5).is_none());
        // Non-positive anchor (scale) is invalid.
        assert!(fitted_weibull_baseline_from_linear_time_beta(&array![1.0], 0.0).is_none());
    }

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
        // Anchored I-spline value basis (#2306): the anchoring removes the
        // constant direction, so the order-m roughness nullity is m−1 — the
        // order-1 penalty is positive definite (nullity 0). The old [1, 2, 3]
        // encoded the unanchored convention.
        assert_eq!(build.nullspace_dims, vec![0, 1, 2]);
        assert!(build.ncols > 0);
    }

    /// The one anchor rule (#2631), exercised across every mode and every
    /// truncation shape. Before the unification this behavior was spread over
    /// three resolvers and two front-end copies of the mode dispatch, and the
    /// copies disagreed.
    ///
    /// Marginal-slope takes the robust interior (median exit) anchor
    /// unconditionally: its `γ = 0` monotone-cone seed is where #751 was
    /// measured.
    #[test]
    fn marginal_slope_time_anchor_defaults_to_median_exit() {
        let age_entry = array![9.0, 1.0, 4.0, 6.0];
        let age_exit = array![20.0, 12.0, 18.0, 30.0];
        let anchor = resolve_survival_time_anchor_for_mode(
            SurvivalLikelihoodMode::MarginalSlope,
            &age_entry,
            &age_exit,
            None,
        )
        .expect("resolve marginal-slope default time anchor");

        // Even count: mean of the two central exits, 18 and 20.
        assert!(
            (anchor - 19.0).abs() <= 1e-12,
            "marginal-slope default anchor should be median exit, got {anchor}"
        );
    }

    /// An explicit anchor is the caller overriding the conditioning heuristic on
    /// purpose, so it wins in every mode — including the modes whose default
    /// would have been the median exit.
    #[test]
    fn explicit_time_anchor_wins_in_every_mode() {
        let age_entry = array![9.0, 1.0, 4.0, 6.0];
        let age_exit = array![20.0, 12.0, 18.0, 30.0];
        for mode in SURVIVAL_LIKELIHOOD_MODES {
            let anchor =
                resolve_survival_time_anchor_for_mode(mode, &age_entry, &age_exit, Some(7.5))
                    .expect("resolve explicit time anchor");
            assert!(
                (anchor - 7.5).abs() <= 1e-12,
                "explicit anchor must round-trip for {mode:?}, got {anchor}"
            );
        }
    }

    /// Ordinary right-censored data (`entry == 0`) keeps the earliest-entry
    /// anchor in every non-marginal-slope mode, so centering stays the near-no-op
    /// it was before #751 and prior behavior is preserved bit-for-bit.
    #[test]
    fn right_censored_data_keeps_the_earliest_entry_anchor() {
        let age_entry = Array1::<f64>::zeros(4);
        let age_exit = array![20.0, 12.0, 18.0, 30.0];
        for mode in SURVIVAL_LIKELIHOOD_MODES {
            if mode == SurvivalLikelihoodMode::MarginalSlope {
                continue;
            }
            let anchor = resolve_survival_time_anchor_for_mode(mode, &age_entry, &age_exit, None)
                .expect("resolve right-censored time anchor");
            assert!(
                (anchor - SURVIVAL_TIME_FLOOR).abs() <= 1e-18,
                "un-truncated {mode:?} must anchor at the earliest entry (floored), got {anchor}"
            );
        }
    }

    /// Genuine left truncation promotes the robust interior anchor for EVERY
    /// time-basis-carrying likelihood, not just marginal-slope. This is the rule
    /// the CLI's own copy did not implement, which is why the same data, formula
    /// and config produced a different location-scale fit on each front end
    /// (#2631).
    #[test]
    fn left_truncated_data_takes_the_robust_interior_anchor_in_every_mode() {
        let age_entry = array![9.0, 1.0, 4.0, 6.0];
        let age_exit = array![20.0, 12.0, 18.0, 30.0];
        assert!(survival_data_is_left_truncated(&age_entry));
        for mode in SURVIVAL_LIKELIHOOD_MODES {
            let anchor = resolve_survival_time_anchor_for_mode(mode, &age_entry, &age_exit, None)
                .expect("resolve left-truncated time anchor");
            assert!(
                (anchor - 19.0).abs() <= 1e-12,
                "left-truncated {mode:?} must anchor at the median exit (19.0), got {anchor}; \
                 the earliest entry (1.0) is the #751/#1790 defect"
            );
        }
    }

    /// Staggered entry — part of the cohort followed from the time origin, the
    /// rest joining later — IS left truncation. The predicate is `any(entry >
    /// threshold)`, not `min(entry) > threshold`: with a `min` test this shape
    /// would fall back to an anchor at the time-origin floor, where the centered
    /// exit columns are maximally large and one-signed. The retired
    /// transformation-specific resolver used `min` and got exactly this case
    /// wrong.
    #[test]
    fn staggered_entry_counts_as_left_truncated() {
        let age_entry = array![0.0, 9.0, 0.0, 6.0];
        let age_exit = array![20.0, 12.0, 18.0, 30.0];
        assert!(
            survival_data_is_left_truncated(&age_entry),
            "a cohort with some rows entering at positive delayed-entry times is left-truncated"
        );
        for mode in SURVIVAL_LIKELIHOOD_MODES {
            let anchor = resolve_survival_time_anchor_for_mode(mode, &age_entry, &age_exit, None)
                .expect("resolve staggered-entry time anchor");
            assert!(
                (anchor - 19.0).abs() <= 1e-12,
                "staggered-entry {mode:?} must anchor at the median exit, got {anchor}"
            );
        }
    }

    /// The origin convention is the likelihood engines' own: an entry exactly at
    /// the threshold is still "at the origin", one above it is delayed entry.
    #[test]
    fn left_truncation_predicate_uses_the_engine_origin_threshold() {
        assert!(!survival_data_is_left_truncated(&array![
            0.0,
            ENTRY_AT_ORIGIN_THRESHOLD
        ]));
        assert!(survival_data_is_left_truncated(&array![
            0.0,
            ENTRY_AT_ORIGIN_THRESHOLD * 1.000_001
        ]));
    }

    /// Odd row counts take the true middle exit; the anchor is always floored so
    /// `log(anchor)` stays finite.
    #[test]
    fn robust_interior_anchor_is_the_median_and_is_floored() {
        assert!(
            (survival_robust_interior_time_anchor(&array![30.0, 12.0, 18.0])
                .expect("odd-count median")
                - 18.0)
                .abs()
                <= 1e-12
        );
        assert_eq!(
            survival_robust_interior_time_anchor(&array![0.0, 0.0]).expect("zero exits"),
            SURVIVAL_TIME_FLOOR
        );
        assert!(survival_robust_interior_time_anchor(&Array1::<f64>::zeros(0)).is_err());
        assert!(survival_earliest_entry_time_anchor(&Array1::<f64>::zeros(0)).is_err());
    }

    /// The MECHANISM behind the rule, measured rather than asserted (#751/#1790,
    /// #2631).
    ///
    /// The robust interior anchor exists because centering a left-truncated
    /// design at the earliest entry leaves the exit columns large and ONE-SIGNED
    /// — that column is the unpenalized polynomial null space of the
    /// 2nd-difference time penalty, so its inflation multiplies the time-block
    /// score at the smoothing seed. This measures both properties directly on a
    /// staggered-entry cohort (half the rows entering at the time origin, half at
    /// positive delayed-entry times — the ordinary shape of a real registry
    /// cohort, and the case a `min(entry) > threshold` predicate would have
    /// missed):
    ///
    ///   * column magnitude `max |X_exit − X(anchor)|`
    ///   * sign balance: the fraction of rows sharing the dominant sign, per
    ///     column, worst column reported
    ///
    /// The earliest-entry anchor must be strictly worse on both, or the rule this
    /// module implements has no reason to exist.
    #[test]
    fn robust_interior_anchor_shrinks_and_balances_the_centered_time_design() {
        // Staggered entry: rows 0..5 observed from the origin, rows 6..11 with
        // positive delayed entry. Exits spread over a decade of time.
        let age_entry = array![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 5.0, 9.0, 14.0, 22.0, 30.0
        ];
        let age_exit = array![
            2.0, 4.0, 7.0, 11.0, 16.0, 23.0, 31.0, 40.0, 52.0, 68.0, 85.0, 110.0
        ];
        assert!(survival_data_is_left_truncated(&age_entry));

        let earliest = survival_earliest_entry_time_anchor(&age_entry).expect("earliest anchor");
        let robust = survival_robust_interior_time_anchor(&age_exit).expect("robust anchor");

        // Measure both anchorings on the SAME basis (knots inferred once from the
        // same times), so only the centering differs.
        let build = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 3,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            Some((4, 1e-2)),
        )
        .expect("build survival time basis");
        let resolved = resolved_survival_time_basis_config_from_build(
            &build.basisname,
            build.degree,
            build.knots.as_ref(),
            build.keep_cols.as_ref(),
            build.smooth_lambda,
        )
        .expect("resolve time basis config");

        // The quantity #1790 names is the centered design's component along the
        // TREND direction — the unpenalized null space of the 2nd-difference time
        // penalty — not any single raw column (an I-spline's last column
        // saturates at 1 over the observed range and is one-signed under every
        // anchor). For a monotone I-spline basis the row sum `Σ_j X_ij` is a
        // monotone increasing function of `t_i`, so it IS that trend coordinate,
        // and it is what "large and one-signed across all rows" is a statement
        // about.
        //
        // Returns `(max |row sum|, fraction of rows sharing the dominant sign)`.
        let measure = |anchor: f64| -> (f64, f64) {
            let mut centered = build.clone();
            let anchor_row =
                evaluate_survival_time_basis_row(anchor, &resolved).expect("anchor basis row");
            center_survival_time_designs_at_anchor(
                &mut centered.x_entry_time,
                &mut centered.x_exit_time,
                &anchor_row,
            )
            .expect("center at anchor");
            let dense = centered.x_exit_time.to_dense();
            let trend: Vec<f64> = dense.rows().into_iter().map(|row| row.sum()).collect();
            let magnitude = trend.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
            let positive = trend.iter().filter(|v| **v > 0.0).count() as f64;
            let rows = trend.len() as f64;
            let sign_fraction = (positive / rows).max(1.0 - positive / rows);
            eprintln!(
                "anchor {anchor:>10.4}: max|trend| = {magnitude:.6e}, \
                 dominant-sign fraction = {sign_fraction:.4}, trend = {trend:?}"
            );
            (magnitude, sign_fraction)
        };

        let (earliest_magnitude, earliest_sign) = measure(earliest);
        let (robust_magnitude, robust_sign) = measure(robust);

        // Measured on this fixture: earliest-entry anchor (the time-origin floor)
        // gives max|trend| = 5.000 with EVERY row positive; median-exit anchor
        // (27.0) gives 1.140 with a 6/6 sign split. The thresholds below are
        // loose around those numbers — they pin the phenomenon, not the digits.
        assert!(
            robust_magnitude < 0.5 * earliest_magnitude,
            "the robust interior anchor must materially shrink the centered trend \
             coordinate: earliest-entry anchor {earliest} gives max|trend| = \
             {earliest_magnitude}, median-exit anchor {robust} gives {robust_magnitude}"
        );
        assert!(
            (earliest_sign - 1.0).abs() <= 1e-12,
            "the earliest-entry anchor is expected to leave the trend coordinate \
             FULLY one-signed on left-truncated data — that is the #751/#1790 \
             mechanism, and a fixture where it does not hold is not exercising the \
             rule. Measured {earliest_sign}"
        );
        assert!(
            robust_sign <= 0.6,
            "the robust interior anchor must leave the trend coordinate two-signed \
             (the exit-event likelihood then pins the linear trend); measured \
             {robust_sign} of rows sharing one sign"
        );
    }

    /// `SURVIVAL_LIKELIHOOD_MODES` must list every variant, or the cross-mode
    /// contracts above silently stop covering one. The `match` is what enforces
    /// it: adding a variant to the enum breaks this compilation until the new
    /// mode is added to the array too.
    #[test]
    fn survival_likelihood_modes_is_exhaustive() {
        fn slot(mode: SurvivalLikelihoodMode) -> usize {
            match mode {
                SurvivalLikelihoodMode::Transformation => 0,
                SurvivalLikelihoodMode::Weibull => 1,
                SurvivalLikelihoodMode::LocationScale => 2,
                SurvivalLikelihoodMode::MarginalSlope => 3,
                SurvivalLikelihoodMode::Latent => 4,
                SurvivalLikelihoodMode::LatentBinary => 5,
            }
        }
        let mut seen = [false; 6];
        for mode in SURVIVAL_LIKELIHOOD_MODES {
            let slot = slot(mode);
            assert!(!seen[slot], "{mode:?} listed twice");
            seen[slot] = true;
        }
        assert!(
            seen.iter().all(|&hit| hit),
            "SURVIVAL_LIKELIHOOD_MODES is missing a variant: {seen:?}"
        );
    }

    /// A caller-supplied anchor is validated once, in one place, so every front
    /// end refuses the same values with the same message.
    #[test]
    fn time_anchor_override_rejects_non_finite_and_negative_values() {
        for bad in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                validate_survival_time_anchor_override(bad).is_err(),
                "override {bad} must be refused"
            );
            assert!(
                resolve_survival_time_anchor_for_mode(
                    SurvivalLikelihoodMode::Transformation,
                    &array![0.0, 1.0],
                    &array![5.0, 6.0],
                    Some(bad),
                )
                .is_err(),
                "the rule must refuse override {bad}"
            );
        }
        assert_eq!(
            validate_survival_time_anchor_override(0.0).expect("zero is a legal anchor"),
            SURVIVAL_TIME_FLOOR
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
    fn ispline_time_penalty_is_psd_under_nontrivial_keep_cols() {
        // PSD-invariant forward guard for the gam#979 survival hang. The I-spline
        // value-space curvature penalty on the increment coefficients is the
        // congruence `S_I = Lᵀ S_B[1:,1:] L`. When identifiability drops columns,
        // the retained block MUST be taken as a PRINCIPAL SUBMATRIX of the FULL
        // congruence (congruence first, column selection second). The historical
        // regression assembled the reduced penalty in the wrong order, producing
        // a strongly INDEFINITE matrix (measured `s0_min_eval = −9.8e7`); an
        // indefinite time penalty makes `½γᵀ S_I γ` unbounded below, the inner
        // joint-Newton follows the divergence, and the outer REML never
        // terminates — the survival marginal-slope hang.
        //
        // This test exercises the reduction with a NON-TRIVIAL `keep_cols`
        // (a proper subset, an interior column dropped) and asserts the assembled
        // penalty satisfies the PSD contract the fix guarantees. It locks the
        // invariant on the shipped code path so a future reassembly that
        // reintroduces an indefinite reduction is caught at construction rather
        // than silently as an outer-loop hang. (It is a forward invariant lock,
        // not a bit-exact replay of the removed buggy assembly.)
        let age_entry = array![1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0];
        let age_exit = array![2.0_f64, 3.0, 5.0, 8.0, 13.0, 21.0];
        let left = 1.0_f64.ln();
        let right = 21.0_f64.ln();
        let q1 = left + 0.25 * (right - left);
        let mid = left + 0.5 * (right - left);
        let q3 = left + 0.75 * (right - left);
        // Degree-2 I-spline with three interior knots -> a value-space basis wide
        // enough to drop an interior column and still leave the reduction
        // non-trivial (p_time < p_time_full).
        let knots = array![
            left, left, left, left, q1, mid, q3, right, right, right, right
        ];

        // Discover the full basis width by building with all columns retained.
        let full = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 2,
                knots: knots.clone(),
                keep_cols: Vec::new(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect("build full-width ispline time basis");
        let p_time_full = full
            .keep_cols
            .as_ref()
            .map(|k| k.len())
            .unwrap_or_else(|| full.x_exit_time.ncols());
        assert!(
            p_time_full >= 3,
            "test needs at least 3 shape-varying columns to drop an interior one; got {p_time_full}"
        );

        // Retain everything except one interior column, forcing the
        // principal-submatrix-of-the-full-congruence path.
        let keep_cols: Vec<usize> = (0..p_time_full).filter(|&j| j != 1).collect();

        let built = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 2,
                knots,
                keep_cols: keep_cols.clone(),
                smooth_lambda: 1e-2,
            },
            None,
        )
        .expect(
            "reduced ispline penalty must build (PSD contract must accept the \
             congruence-first / select-second ordering)",
        );

        assert_eq!(
            built.penalties.len(),
            1,
            "the ispline time basis should carry exactly one curvature penalty"
        );
        let s = &built.penalties[0];
        assert_eq!(s.nrows(), keep_cols.len());
        assert_eq!(s.ncols(), keep_cols.len());

        let (evals, _) = gam_linalg::faer_ndarray::FaerEigh::eigh(s, faer::Side::Lower)
            .expect("eigh of penalty");
        let evals_slice = evals.as_slice().expect("contiguous eigenvalues");
        let max_abs = evals_slice
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
            .max(1.0);
        let min_ev = evals_slice.iter().copied().fold(f64::INFINITY, f64::min);
        let tol = -100.0 * (s.nrows() as f64) * f64::EPSILON * max_abs;
        assert!(
            min_ev >= tol,
            "reduced I-spline time penalty must be PSD (gam#979): min eigenvalue \
             {min_ev:.3e} < tol {tol:.3e}, max|eig| {max_abs:.3e}"
        );
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

    /// A frozen chart evaluated at `θ` records `θ` BITWISE (#2765).
    ///
    /// `SurvivalMarginalSlopeFamilyHyperState` stores the geometry's `theta` as
    /// the family's realized coordinates, and `validate_layout` compares them
    /// to the outer manifest with `to_bits()` equality — on purpose, so a
    /// workspace cannot reuse row geometry from a neighbouring outer probe.
    /// The chart used to close a `θ → cfg → θ` loop, which for a Weibull is
    /// `ln(exp(θ))` and is not the identity in `f64`. The exactness invariant
    /// then failed for a reason that has nothing to do with the geometry, the
    /// inner solve refused a point the outer optimizer was merely trying to
    /// evaluate, and the line search read that refusal as "no improvement" —
    /// 50 times, at every halving.
    ///
    /// The witnesses below are the coordinates the #2765 acceptance fixture
    /// actually refused at, with their measured round-trip error. The test
    /// asserts the round trip really is lossy at each (so it cannot quietly
    /// stop being a witness) and then that the chart is exact anyway.
    #[test]
    fn a_frozen_baseline_chart_records_the_theta_it_was_asked_for_2765() {
        let age_entry = array![0.0, 0.75, 2.0];
        let age_exit = array![1.5, 3.0, 5.5];
        let initial_config = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: Some(2.0),
            shape: Some(1.3),
            rate: None,
            makeham: None,
        };
        let baseline = build_survival_marginal_slope_baseline_geometry(
            &age_entry,
            &age_exit,
            &initial_config,
        )
        .expect("initial baseline geometry")
        .expect("Weibull is a nonlinear chart");
        let chart = SurvivalMarginalSlopeFrozenOffsetChart::new(
            &age_entry,
            &age_exit,
            &initial_config,
            &baseline.offset_entry,
            &baseline.offset_exit,
            &baseline.derivative_offset_exit,
        )
        .expect("freeze the Weibull chart");

        // `θ₄ = ±1e-5` are the two coordinates the acceptance fixture's
        // certificate probe refused on BOTH sides; `1e-5` comes back 57_269
        // ulps away from itself through `ln(exp(·))`.
        for theta in [
            array![0.7574963781222602_f64, 1.0e-5],
            array![0.7574963781222602_f64, -1.0e-5],
            array![0.7574863781222603_f64, 0.0],
        ] {
            let lossy = theta
                .iter()
                .any(|value| value.exp().ln().to_bits() != value.to_bits());
            assert!(
                lossy,
                "this witness has stopped being one: every coordinate of {theta:?} now \
                 survives ln(exp(·)) bitwise, so it can no longer show the defect"
            );
            let realized = chart
                .evaluate(&theta)
                .expect("the chart evaluates inside its domain");
            for (axis, (want, got)) in theta.iter().zip(realized.theta.iter()).enumerate() {
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "chart axis {axis}: asked for {want:?}, recorded {got:?} — a chart must \
                     record the coordinate it was ASKED to realize, because the family's \
                     manifest check is bitwise"
                );
            }
        }
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

    #[test]
    fn marginal_slope_baseline_chain_rule_gradient_contracts_probit_partials() {
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
            (100.0, 0.012, 1e-6), // x = 1e-4, the old-pivot band (large age, tiny shape)
            (50.0, 1.2, 1e-8),
        ];
        // NOTE: every quantity below is compared against its shape->0 *limit*.
        // For the cancelling cumulative branches (∂H/∂shape, ∂²H/∂shape²,
        // ∂²h/∂shape²) the limit is the correct shape->0 target and the
        // implementation routes through Taylor in this band. But the
        // instantaneous first derivative ∂h_G/∂shape = rate·age·e^x carries NO
        // cancellation: it is exact, and its departure from the limit rate·t is
        // a genuine O(x) effect. At x=1e-3 that departure is ~1.2e-3 (> tol),
        // so the cases here keep x <= 1e-4 where the limit is a valid 1e-3
        // oracle for *all four* quantities. The cancelling-branch regression at
        // larger x is covered by gompertz_second_shape_derivative_is_accurate_in_old_pivot_gap.
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

    // ── gam#2765 / gam#2767: the slope follow-up margin replays exactly ──

    /// The predict-time replay must reproduce the fit-time margin bit for bit.
    ///
    /// This is the property the whole persistence contract rests on: at fit time
    /// the knots are placed by QUANTILE from the training exit times, and the
    /// design is a by-product of that build; at predict time only the knots
    /// survive, and the design is rebuilt from them. If those two designs are not
    /// the same matrix on the same rows, every saved follow-up-varying slope
    /// evaluates a different model than the one that was fitted — silently,
    /// because the widths still agree.
    #[test]
    fn slope_time_margin_replay_reproduces_the_fit_time_design_2765() {
        let age_exit = Array1::from_iter((1..=40).map(|i| 0.25 + 0.35 * f64::from(i)));
        let age_entry = age_exit.mapv(|t| (t - 0.2).max(1e-3));
        let fitted = build_time_varying_survival_covariate_template(
            &age_entry,
            &age_exit,
            5,
            3,
            "slope",
        )
        .expect("fit-time slope margin");
        let SurvivalCovariateTermBlockTemplate::TimeVarying {
            time_basis,
            time_basis_entry,
            time_basis_exit,
            time_basis_derivative_exit,
            ..
        } = &fitted
        else {
            panic!("a time-varying request must produce a time-varying template");
        };

        let replayed_exit = slope_time_margin_rows(time_basis, age_exit.view())
            .expect("replayed exit margin");
        assert_eq!(replayed_exit.value.dim(), time_basis_exit.dim());
        for (fit_value, replay_value) in time_basis_exit.iter().zip(replayed_exit.value.iter()) {
            assert_eq!(
                fit_value.to_bits(),
                replay_value.to_bits(),
                "the replayed exit margin must be the fitted one, not merely close"
            );
        }
        for (fit_value, replay_value) in time_basis_derivative_exit
            .iter()
            .zip(replayed_exit.derivative.iter())
        {
            assert_eq!(
                fit_value.to_bits(),
                replay_value.to_bits(),
                "the replayed exit-margin tangent must be the fitted one"
            );
        }

        // And the full three-channel replay the leave-one-out path consumes.
        let covariate = DesignMatrix::from(Array2::<f64>::from_shape_fn(
            (age_exit.len(), 2),
            |(row, col)| if col == 0 { 1.0 } else { (row as f64) * 0.05 - 1.0 },
        ));
        let replay =
            replay_slope_follow_up_designs(&age_entry, &age_exit, time_basis, &covariate)
                .expect("three-channel replay");
        let p_time = time_basis_exit.ncols();
        assert_eq!(replay.exit.ncols(), 2 * p_time);
        for (channel, fitted_margin) in [
            (&replay.entry, time_basis_entry),
            (&replay.exit, time_basis_exit),
            (&replay.derivative_exit, time_basis_derivative_exit),
        ] {
            let dense = channel
                .try_to_dense_arc("replayed slope channel")
                .expect("dense channel");
            let covariate_dense = covariate
                .try_to_dense_arc("covariate factor")
                .expect("dense covariate");
            for row in 0..age_exit.len() {
                for cov_col in 0..2 {
                    for time_col in 0..p_time {
                        let expected =
                            covariate_dense[[row, cov_col]] * fitted_margin[[row, time_col]];
                        let got = dense[[row, cov_col * p_time + time_col]];
                        assert!(
                            (expected - got).abs() <= 1e-15 * (1.0 + expected.abs()),
                            "row-wise Kronecker mismatch at ({row}, {cov_col}, {time_col}): \
                             expected {expected} got {got}"
                        );
                    }
                }
            }
        }
    }

    /// The exit design a batch replay produces is the same one a single-row
    /// replay produces at that row's time. The survival-curve path replays one
    /// `(row, t)` cell at a time, so if these disagreed a predicted curve would
    /// not pass through the batch-predicted point.
    #[test]
    fn slope_time_margin_row_replay_matches_the_batch_replay_2765() {
        let age_exit = Array1::from_iter((1..=12).map(|i| 0.4 + 0.6 * f64::from(i)));
        let age_entry = age_exit.mapv(|t| (t - 0.15).max(1e-3));
        let fitted =
            build_time_varying_survival_covariate_template(&age_entry, &age_exit, 6, 2, "slope")
                .expect("fit-time slope margin");
        let time_basis = fitted
            .resolved_time_basis()
            .expect("a time-varying template resolves a basis")
            .clone();
        let covariate = DesignMatrix::from(Array2::<f64>::from_shape_fn(
            (age_exit.len(), 2),
            |(row, col)| if col == 0 { 1.0 } else { 0.3 * (row as f64) },
        ));
        let batch = replay_slope_time_margin_value_tangent_design(
            age_exit.view(),
            &time_basis,
            &covariate,
        )
        .expect("batch replay");
        let batch_value = batch
            .value
            .try_to_dense_arc("batch value replay")
            .expect("dense batch");
        let batch_derivative = batch
            .derivative
            .try_to_dense_arc("batch tangent replay")
            .expect("dense batch tangent");
        let covariate_dense = covariate
            .try_to_dense_arc("covariate")
            .expect("dense covariate");
        for row in 0..age_exit.len() {
            let single_covariate = DesignMatrix::from(
                covariate_dense
                    .row(row)
                    .to_owned()
                    .into_shape_with_order((1, 2))
                    .expect("single covariate row"),
            );
            let single = replay_slope_time_margin_value_tangent_design(
                Array1::from_elem(1, age_exit[row]).view(),
                &time_basis,
                &single_covariate,
            )
            .expect("single-row replay");
            let single_value = single
                .value
                .try_to_dense_arc("single-row value replay")
                .expect("dense single value row");
            let single_derivative = single
                .derivative
                .try_to_dense_arc("single-row tangent replay")
                .expect("dense single tangent row");
            for col in 0..batch_value.ncols() {
                assert_eq!(
                    batch_value[[row, col]].to_bits(),
                    single_value[[0, col]].to_bits(),
                    "single-row replay disagrees with the batch at ({row}, {col})"
                );
                assert_eq!(
                    batch_derivative[[row, col]].to_bits(),
                    single_derivative[[0, col]].to_bits(),
                    "single-row tangent replay disagrees with the batch at ({row}, {col})"
                );
            }
        }
    }

    /// The defect gam#2705 names, at the level it is created: the survival
    /// I-spline time block's `x_derivative_time` must be the derivative of its
    /// own `x_exit_time`, OUTSIDE the fitted knot span as well as inside it.
    ///
    /// Before the repair the value basis saturated past the boundary knots
    /// while the derivative — hand-rolled from a CLAMPED B-spline
    /// first-derivative basis — returned the boundary slope, so a saved
    /// Royston-Parmar fit published a flat `Λ(t)` beside a nonzero
    /// `h(t) = Λ·d(log Λ)/dt`. `h = dΛ/dt`, so those cannot both be one model.
    ///
    /// The check is a central difference in `t` (not in `log t`), because `t`
    /// is the variable `x_derivative_time` is a derivative with respect to —
    /// the `1/t` chain factor is part of what has to agree.
    #[test]
    fn ispline_time_derivative_is_a_finite_difference_of_its_value_2705() {
        let n = 24usize;
        let age_entry = Array1::<f64>::zeros(n);
        let age_exit =
            Array1::from_iter((0..n).map(|i| 4.0 + 40.0 * (i as f64) / ((n - 1) as f64)));
        let build = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 3,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: 1.0,
            },
            Some((3, 1.0)),
        )
        .expect("ispline time basis builds");
        let resolved = resolved_survival_time_basis_config_from_build(
            &build.basisname,
            build.degree,
            build.knots.as_ref(),
            build.keep_cols.as_ref(),
            build.smooth_lambda,
        )
        .expect("resolved ispline config");

        // Inside the fitted span, far below it, and far above it — the last two
        // are where every `predict(...).survival_at(grid)` call lands, because
        // `default_survival_time_grid` starts at 0 and ends past `max(exit)`.
        //
        // The two BOUNDARY knots themselves (`t = 4` and `t = 44`, the extreme
        // training exits) are deliberately not central-differenced, and the
        // boundary is checked exactly below instead. An M-spline has a
        // one-sided kink at a clamped boundary knot — the trailing columns
        // vanish there like `(t_b − t)^k` — so the analytic derivative is the
        // one-sided LIMIT (zero for those columns) while a symmetric window of
        // half-width `h` averages the rising side and returns `O(h)`. Measured
        // at `t = 44`, `h = 1e-5`: analytic `0`, central difference `8.99e-8`,
        // which shrinks with `h` rather than marking a disagreement.
        let queries = [1.0_f64, 3.0, 12.0, 30.0, 43.0, 60.0, 400.0, 2_850.0];
        let step = 1.0e-5_f64;
        let mut exterior_rows_with_slope = 0usize;
        for &t in queries.iter() {
            let times = Array1::from_vec(vec![t - step, t, t + step]);
            let probe =
                build_survival_time_basis(&Array1::<f64>::zeros(3), &times, resolved.clone(), None)
                    .expect("ispline time basis replays at the query times");
            let value = probe.x_exit_time.to_dense();
            let derivative = probe.x_derivative_time.to_dense();
            let mut row_slope = 0.0_f64;
            for column in 0..value.ncols() {
                let difference = (value[[2, column]] - value[[0, column]]) / (2.0 * step);
                let analytic = derivative[[1, column]];
                row_slope += analytic.abs();
                let scale = analytic.abs().max(difference.abs()).max(1.0e-6);
                assert!(
                    (difference - analytic).abs() <= 1.0e-4 * scale,
                    "t={t}: column {column} analytic d/dt {analytic:.9e} disagrees with the \
                     central difference of its own value basis {difference:.9e}"
                );
            }
            if !(4.0..=44.0).contains(&t) {
                exterior_rows_with_slope += usize::from(row_slope > 0.0);
            }
        }
        // Non-vacuity: a basis whose exterior derivative is identically zero
        // passes every assertion above by agreeing with a flat value. The
        // Royston-Parmar tail is LINEAR, so the exterior slope is the boundary
        // slope and is nonzero on both sides.
        assert!(
            exterior_rows_with_slope >= 2,
            "the exterior must carry a nonzero boundary slope on both sides; \
             {exterior_rows_with_slope} of the exterior query times did"
        );

        // The boundary itself, exactly rather than by difference: the tail is
        // ANCHORED at the spline's own one-sided value and slope there, so
        // crossing `t_b` must not move the derivative at all and must move the
        // value by exactly `Δ(log t) · slope`.
        let boundary = 44.0_f64;
        let outside = 44.05_f64;
        let pair = build_survival_time_basis(
            &Array1::<f64>::zeros(2),
            &Array1::from_vec(vec![boundary, outside]),
            resolved.clone(),
            None,
        )
        .expect("ispline time basis replays across the boundary knot");
        let value = pair.x_exit_time.to_dense();
        let derivative = pair.x_derivative_time.to_dense();
        // `x_derivative_time` carries the `1/t` chain factor, so undo it to
        // compare slopes in the basis's own `log t` coordinate.
        let log_gap = outside.ln() - boundary.ln();
        for column in 0..value.ncols() {
            let boundary_slope = derivative[[0, column]] * boundary;
            let outside_slope = derivative[[1, column]] * outside;
            assert!(
                (outside_slope - boundary_slope).abs() <= 1.0e-12 * boundary_slope.abs().max(1.0),
                "column {column}: the tail slope {outside_slope:.9e} is not the boundary slope \
                 {boundary_slope:.9e}"
            );
            let expected = value[[0, column]] + log_gap * boundary_slope;
            assert!(
                (value[[1, column]] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
                "column {column}: the tail value {:.9e} is not the affine continuation \
                 {expected:.9e} of the boundary value {:.9e} at slope {boundary_slope:.9e}",
                value[[1, column]],
                value[[0, column]]
            );
        }
    }

    /// The fit itself must not move. Every training row is inside the knot span
    /// the training rows themselves induced, so the linear-tail convention is
    /// inert there — and a row entering AT THE ORIGIN keeps the anchored zero
    /// entry row rather than a tail evaluated at `ln(SURVIVAL_TIME_FLOOR)`,
    /// which is a readout of `1e-9` and not of the data.
    #[test]
    fn the_linear_tail_convention_is_inert_on_the_training_rows_2705() {
        let n = 16usize;
        let age_entry = Array1::<f64>::zeros(n);
        let age_exit =
            Array1::from_iter((0..n).map(|i| 2.0 + 20.0 * (i as f64) / ((n - 1) as f64)));
        let build = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 3,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: 1.0,
            },
            Some((3, 1.0)),
        )
        .expect("ispline time basis builds");
        let entry = build.x_entry_time.to_dense();
        let exit = build.x_exit_time.to_dense();
        for row in 0..n {
            for column in 0..entry.ncols() {
                assert_eq!(
                    entry[[row, column]],
                    0.0,
                    "an entry-at-origin row must carry the anchored zero row at ({row}, {column})"
                );
            }
        }
        // The exit rows are I-spline values on their own knot span, so every
        // entry stays in [0, 1]: no tail is being evaluated at a training row.
        for row in 0..n {
            for column in 0..exit.ncols() {
                let value = exit[[row, column]];
                assert!(
                    (-1.0e-12..=1.0 + 1.0e-12).contains(&value),
                    "training exit row ({row}, {column}) = {value} is outside [0, 1], so the \
                     linear tail is being evaluated on the training data"
                );
            }
        }
    }

    /// The anchor is the ORIGIN of the baseline reparameterization, and the
    /// default anchor for ordinary right-censored data is the time origin,
    /// which `evaluate_survival_time_basis_row` floors to `SURVIVAL_TIME_FLOOR`.
    /// Under a linear-tailed baseline an unclamped anchor there would re-center
    /// every design column by a large constant read off `1e-9` — the #751
    /// inflation the anchor rule exists to avoid. Clamping into the modelling
    /// interval keeps the shipped answer: `I_k(left) = 0` exactly.
    #[test]
    fn the_anchor_row_is_clamped_into_the_modelling_interval_2705() {
        let n = 16usize;
        let age_entry = Array1::<f64>::zeros(n);
        let age_exit =
            Array1::from_iter((0..n).map(|i| 5.0 + 50.0 * (i as f64) / ((n - 1) as f64)));
        let build = build_survival_time_basis(
            &age_entry,
            &age_exit,
            SurvivalTimeBasisConfig::ISpline {
                degree: 3,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: 1.0,
            },
            Some((3, 1.0)),
        )
        .expect("ispline time basis builds");
        let resolved = resolved_survival_time_basis_config_from_build(
            &build.basisname,
            build.degree,
            build.knots.as_ref(),
            build.keep_cols.as_ref(),
            build.smooth_lambda,
        )
        .expect("resolved ispline config");
        let anchor_at_origin = evaluate_survival_time_basis_row(0.0, &resolved)
            .expect("anchor row at the time origin");
        for (column, value) in anchor_at_origin.iter().enumerate() {
            assert_eq!(
                *value, 0.0,
                "the anchor row at the time origin must be exactly zero at column {column}, \
                 got {value}"
            );
        }
        // And an anchor INSIDE the span is still a real evaluation, so the
        // clamp has not turned the anchor into a constant.
        let interior =
            evaluate_survival_time_basis_row(30.0, &resolved).expect("anchor row inside the span");
        assert!(
            interior.iter().any(|value| *value > 1.0e-9),
            "an interior anchor must still evaluate the basis, got {interior:?}"
        );
    }
}
