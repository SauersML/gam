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
    build_bspline_basis_1d, create_basis, evaluate_bspline_derivative_scalar, BSplineBasisSpec,
    BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, BasisOptions, Dense, KnotSource,
};
use crate::families::gamlss::{
    append_selected_wiggle_penalty_orders, buildwiggle_block_input_from_seed,
    monotone_wiggle_basis_with_derivative_order, split_wiggle_penalty_orders, WiggleBlockConfig,
};
use crate::families::lognormal_kernel::HazardLoading;
use crate::families::survival_location_scale::{
    ResidualDistribution, SurvivalCovariateTermBlockTemplate,
};
use crate::inference::formula_dsl::LinkWiggleFormulaSpec;
use crate::matrix::{DenseDesignMatrix, DesignMatrix, SparseDesignMatrix};
use crate::probability::{normal_pdf, standard_normal_quantile};
use ndarray::{array, s, Array1, Array2};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurvivalBaselineTarget {
    /// No additional parametric target:
    /// eta_target(t) = 0, so regularized model defaults to linear log-cumulative
    /// hazard from the existing time basis.
    Linear,
    /// Parametric target: Weibull baseline encoded in eta_target(t) = log(H0(t)).
    Weibull,
    /// Parametric target: Gompertz baseline encoded in eta_target(t) = log(H0(t)).
    Gompertz,
    /// Parametric target: Gompertz-Makeham baseline encoded in eta_target(t) = log(H0(t)).
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
    ISpline {
        degree: usize,
        knots: Array1<f64>,
        keep_cols: Vec<usize>,
        smooth_lambda: f64,
    },
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
        return Err(format!(
            "non-finite survival times at row {}",
            row_index + 1
        ));
    }
    if entry_raw < 0.0 || exit_raw < 0.0 {
        return Err(format!("negative survival times at row {}", row_index + 1));
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
    Err(format!(
        "{context} requires a structural monotone survival time basis, but got '{basisname}'. \
Only `ispline` is accepted here because its basis functions enforce a monotone cumulative time effect by construction. \
`{basisname}` can fit non-monotone shapes, which can break survival semantics. \
Re-run with `--time-basis ispline`."
    ))
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
            return Err(format!(
                "unsupported --baseline-target '{other}'; use linear|weibull|gompertz|gompertz-makeham"
            ));
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
            let shape = shape.unwrap_or(0.01);
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
            let shape = shape.unwrap_or(0.01);
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
        other => Err(format!(
            "unsupported --survival-likelihood '{other}'; use transformation|weibull|location-scale|marginal-slope|latent|latent-binary"
        )),
    }
}

pub fn survival_likelihood_modename(mode: SurvivalLikelihoodMode) -> &'static str {
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
        other => Err(format!(
            "unsupported --survival-distribution '{other}'; use gaussian|gumbel|logistic"
        )),
    }
}

pub fn survival_baseline_targetname(target: SurvivalBaselineTarget) -> &'static str {
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
            return Err(format!(
                "unsupported --baseline-target '{other}'; use linear|weibull|gompertz|gompertz-makeham"
            ));
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
            shape: Some(shape.unwrap_or(0.01)),
            rate: Some(rate.unwrap_or(1.0 / time_scale_seed)),
            makeham: None,
        },
        SurvivalBaselineTarget::GompertzMakeham => SurvivalBaselineConfig {
            target,
            scale: None,
            shape: Some(shape.unwrap_or(0.01)),
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
                return Err(format!(
                    "weibull baseline parameter dimension mismatch: expected 2, got {}",
                    theta.len()
                ));
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
                return Err(format!(
                    "gompertz baseline parameter dimension mismatch: expected 2, got {}",
                    theta.len()
                ));
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
                return Err(format!(
                    "gompertz-makeham baseline parameter dimension mismatch: expected 3, got {}",
                    theta.len()
                ));
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

pub fn optimize_survival_baseline_config<F>(
    initial: &SurvivalBaselineConfig,
    context: &str,
    mut objective: F,
) -> Result<SurvivalBaselineConfig, String>
where
    F: FnMut(&SurvivalBaselineConfig) -> Result<f64, String>,
{
    use crate::solver::outer_strategy::{OuterProblem, SolverClass};
    let Some(seed) = survival_baseline_theta_from_config(initial)? else {
        return Ok(initial.clone());
    };
    let dim = seed.len();
    let target = initial.target;
    let lower = seed.mapv(|v| v - 6.0);
    let upper = seed.mapv(|v| v + 6.0);
    let problem = OuterProblem::new(dim)
        .with_solver_class(SolverClass::AuxiliaryGradientFree)
        .with_tolerance(1e-4)
        .with_max_iter(240)
        .with_bounds(lower, upper)
        .with_heuristic_lambdas(seed.to_vec());
    let cost_fn = move |_: &mut (), theta: &ndarray::Array1<f64>| {
        let cfg = survival_baseline_config_from_theta(target, theta)
            .map_err(crate::estimate::EstimationError::InvalidInput)?;
        objective(&cfg).map_err(crate::estimate::EstimationError::InvalidInput)
    };
    let mut obj =
        problem.build_objective(
            (),
            cost_fn,
            |_: &mut (),
             _: &ndarray::Array1<f64>|
             -> Result<
                crate::solver::outer_strategy::OuterEval,
                crate::estimate::EstimationError,
            > {
                Err(crate::estimate::EstimationError::InvalidInput(
                    "baseline aux optimizer: CompassSearch dispatch only calls eval_cost; \
                 eval(gradient) is unreachable by construction"
                        .to_string(),
                ))
            },
            None::<fn(&mut ())>,
            None::<
                fn(
                    &mut (),
                    &ndarray::Array1<f64>,
                ) -> Result<
                    crate::solver::outer_strategy::EfsEval,
                    crate::estimate::EstimationError,
                >,
            >,
        );
    let result = problem
        .run(&mut obj, context)
        .map_err(|e| format!("{context} failed: {e}"))?;
    survival_baseline_config_from_theta(target, &result.rho)
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
                return Err("--time-degree must be >= 1 for ispline time basis".to_string());
            }
            if time_num_internal_knots == 0 {
                return Err(
                    "--time-num-internal-knots must be > 0 for ispline time basis".to_string(),
                );
            }
            if !time_smooth_lambda.is_finite() || time_smooth_lambda < 0.0 {
                return Err("--time-smooth-lambda must be finite and >= 0".to_string());
            }
            Ok(SurvivalTimeBasisConfig::ISpline {
                degree: time_degree,
                knots: Array1::zeros(0),
                keep_cols: Vec::new(),
                smooth_lambda: time_smooth_lambda,
            })
        }
        "linear" | "bspline" => {
            require_structural_survival_time_basis(time_basis, "survival model configuration")?;
            unreachable!("non-structural survival basis unexpectedly validated");
        }
        other => Err(format!("unsupported --time-basis '{other}'; use ispline")),
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
            return Err(format!(
                "survival time basis requires finite {label} times (row {})",
                row + 1
            ));
        }
        if let Some(row) = times.iter().position(|t| *t < 0.0) {
            return Err(format!(
                "survival time basis requires non-negative {label} times (row {})",
                row + 1
            ));
        }
        Ok(times.mapv(|t| t.max(SURVIVAL_TIME_FLOOR).ln()))
    }

    let n = age_entry.len();
    if n != age_exit.len() {
        return Err("survival time basis requires matching entry/exit lengths".to_string());
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

    fn infer_survival_time_knots(
        combined: &Array1<f64>,
        degree: usize,
        num_internal_knots: usize,
        basis_options: BasisOptions,
    ) -> Result<Array1<f64>, String> {
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
                        degree,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Automatic {
                            num_internal_knots: Some(num_internal_knots),
                            placement,
                        },
                        double_penalty: false,
                        identifiability: BSplineIdentifiability::None,
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
                create_basis::<Dense>(
                    combined.view(),
                    KnotSource::Provided(knots.view()),
                    degree,
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
                    return Err("internal error: empty ispline time basis".to_string());
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
                    return Err("saved survival ispline keep_cols exceed basis width".to_string());
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
                    let v = if raw_v < 0.0 && raw_v >= -1e-12 {
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
                },
            )
            .map_err(|e| format!("failed to build ispline smoothing penalty: {e}"))?;
            if penalty_basis.design.ncols() != p_time_full + 1 {
                return Err("internal error: ispline penalty dimension mismatch".to_string());
            }
            let mut penalties = Vec::<Array2<f64>>::new();
            for s_mat in &penalty_basis.penalties {
                if s_mat.nrows() != p_time_full + 1 || s_mat.ncols() != p_time_full + 1 {
                    continue;
                }
                let reduced = s_mat.slice(ndarray::s![1.., 1..]).to_owned();
                let mut local = Array2::<f64>::zeros((p_time, p_time));
                for (i_new, &i_old) in keep_cols.iter().enumerate() {
                    for (j_new, &j_old) in keep_cols.iter().enumerate() {
                        local[[i_new, j_new]] = reduced[[i_old, j_old]];
                    }
                }
                penalties.push(local);
            }

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
            smooth_lambda: smooth_lambda.unwrap_or(1e-2),
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
            smooth_lambda: smooth_lambda.unwrap_or(1e-2),
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
                return Err("survival ISpline anchor keep_cols exceed basis width".to_string());
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
    if !age.is_finite() || age <= 0.0 {
        return Err(
            "survival ages must be finite and positive for baseline derivative evaluation"
                .to_string(),
        );
    }

    match cfg.target {
        SurvivalBaselineTarget::Linear => Ok(None),
        SurvivalBaselineTarget::Weibull => {
            // eta = shape·(log t − log scale)
            //     = shape·log t − shape·log scale
            // o_D = shape / t
            //
            // θ = (log_scale, log_shape):
            //   ∂eta/∂log_scale  = −shape          ∂o_D/∂log_scale = 0
            //   ∂eta/∂log_shape  = shape·(log t − log scale) = eta
            //   ∂o_D/∂log_shape  = shape / t = o_D
            let scale = cfg
                .scale
                .ok_or_else(|| "weibull missing scale".to_string())?;
            let shape = cfg
                .shape
                .ok_or_else(|| "weibull missing shape".to_string())?;
            if !(scale.is_finite() && shape.is_finite() && scale > 0.0 && shape > 0.0) {
                return Err("weibull baseline requires finite positive scale and shape".to_string());
            }
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
        SurvivalBaselineTarget::GompertzMakeham => {
            let rate = cfg.rate.ok_or_else(|| "gm missing rate".to_string())?;
            let shape = cfg.shape.ok_or_else(|| "gm missing shape".to_string())?;
            let makeham = cfg
                .makeham
                .ok_or_else(|| "gm missing makeham".to_string())?;
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
                return Err("gm baseline produced non-positive cumulative hazard".to_string());
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
///
/// The three `∂NLL_i/∂o_channel` terms are the `exit`, `entry`, `derivative`
/// fields of [`OffsetChannelResiduals`] (sampleweight-scaled already). The
/// `∂o/∂θ_k` terms come from [`baseline_offset_theta_partials`] per obs at
/// the appropriate age.
///
/// Per the RP offset convention:
///   o_E[i] = eta_target(age_entry[i])
///   o_X[i] = eta_target(age_exit[i])
///   o_D[i] = d/dt eta_target(t) |_{t=age_exit[i]}
///
/// so the exit and derivative partials are both evaluated at `age_exit[i]`
/// and the entry partial at `age_entry[i]`. The origin-entry case
/// (`entry_at_origin[i]`) has `r_entry[i] = 0` exactly, so we skip the
/// `baseline_offset_theta_partials(age_entry, ..)` call for those rows
/// (avoiding the `age > 0` precondition failure when age_entry is 0).
///
/// Returns `Ok(None)` when `cfg.target == Linear` (no θ-parameters).
pub fn baseline_chain_rule_gradient(
    age_entry: ndarray::ArrayView1<'_, f64>,
    age_exit: ndarray::ArrayView1<'_, f64>,
    cfg: &SurvivalBaselineConfig,
    residuals: &crate::families::survival::OffsetChannelResiduals,
) -> Result<Option<Array1<f64>>, String> {
    let n = age_exit.len();
    if age_entry.len() != n
        || residuals.exit.len() != n
        || residuals.entry.len() != n
        || residuals.derivative.len() != n
    {
        return Err(format!(
            "baseline_chain_rule_gradient: length mismatch (age_entry={}, age_exit={}, r_exit={}, r_entry={}, r_deriv={})",
            age_entry.len(),
            n,
            residuals.exit.len(),
            residuals.entry.len(),
            residuals.derivative.len(),
        ));
    }
    // Probe θ-dim via any valid age. If cfg is Linear the probe returns None
    // and we short-circuit with no θ-gradient.
    let probe_age = age_exit.iter().copied().find(|v| v.is_finite() && *v > 0.0);
    let theta_dim = match probe_age {
        Some(t) => match baseline_offset_theta_partials(t, cfg)? {
            None => return Ok(None),
            Some(v) => v.len(),
        },
        None => {
            return Err(
                "baseline_chain_rule_gradient: no valid positive age for dim probe".to_string(),
            );
        }
    };
    let mut grad = Array1::<f64>::zeros(theta_dim);
    for i in 0..n {
        let t_exit = age_exit[i];
        // Exit + derivative partials both come from the age_exit evaluation.
        let partials_exit = baseline_offset_theta_partials(t_exit, cfg)?
            .ok_or_else(|| "unexpected None from baseline partials at exit".to_string())?;
        if partials_exit.len() != theta_dim {
            return Err(format!(
                "baseline_chain_rule_gradient: theta_dim drifted ({} != {})",
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
        // baseline_offset_theta_partials would error. Gate on residual==0.
        let r_e = residuals.entry[i];
        if r_e != 0.0 {
            let t_entry = age_entry[i];
            let partials_entry = baseline_offset_theta_partials(t_entry, cfg)?
                .ok_or_else(|| "unexpected None from baseline partials at entry".to_string())?;
            for k in 0..theta_dim {
                grad[k] += r_e * partials_entry[k].0;
            }
        }
    }
    Ok(Some(grad))
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
    let n = age_exit.len();
    if age_entry.len() != n
        || residuals.exit.len() != n
        || residuals.entry.len() != n
        || residuals.derivative.len() != n
    {
        return Err(format!(
            "marginal_slope_baseline_chain_rule_gradient: length mismatch (age_entry={}, age_exit={}, r_exit={}, r_entry={}, r_deriv={})",
            age_entry.len(),
            n,
            residuals.exit.len(),
            residuals.entry.len(),
            residuals.derivative.len(),
        ));
    }

    let probe_age = age_exit.iter().copied().find(|v| v.is_finite() && *v > 0.0);
    let theta_dim = match probe_age {
        Some(t) => match marginal_slope_baseline_offset_theta_partials(t, cfg)? {
            None => return Ok(None),
            Some(v) => v.len(),
        },
        None => {
            return Err(
                "marginal_slope_baseline_chain_rule_gradient: no valid positive age for dim probe"
                    .to_string(),
            );
        }
    };

    let mut grad = Array1::<f64>::zeros(theta_dim);
    for i in 0..n {
        let partials_exit = marginal_slope_baseline_offset_theta_partials(age_exit[i], cfg)?
            .ok_or_else(|| {
                "unexpected None from marginal-slope baseline partials at exit".to_string()
            })?;
        if partials_exit.len() != theta_dim {
            return Err(format!(
                "marginal_slope_baseline_chain_rule_gradient: theta_dim drifted ({} != {})",
                partials_exit.len(),
                theta_dim
            ));
        }
        let r_x = residuals.exit[i];
        let r_d = residuals.derivative[i];
        for k in 0..theta_dim {
            let (d_q_dk, d_qt_dk) = partials_exit[k];
            grad[k] += r_x * d_q_dk + r_d * d_qt_dk;
        }

        let r_e = residuals.entry[i];
        if r_e != 0.0 {
            let partials_entry = marginal_slope_baseline_offset_theta_partials(age_entry[i], cfg)?
                .ok_or_else(|| {
                    "unexpected None from marginal-slope baseline partials at entry".to_string()
                })?;
            for k in 0..theta_dim {
                grad[k] += r_e * partials_entry[k].0;
            }
        }
    }
    Ok(Some(grad))
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
    let dhg_dshape = if shape.abs() < 1e-10 {
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

fn survival_hazard_theta_partials(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<Option<Vec<(f64, f64)>>, String> {
    if !age.is_finite() || age <= 0.0 {
        return Err(
            "survival ages must be finite and positive for baseline hazard partials".to_string(),
        );
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
                return Err("weibull baseline requires finite positive scale and shape".to_string());
            }
            let log_time_ratio = age.ln() - scale.ln();
            let cumulative_hazard = (age / scale).powf(shape);
            let instant_hazard = shape * cumulative_hazard / age;
            let eta = shape * log_time_ratio;
            Ok(Some(vec![
                (-shape * cumulative_hazard, -shape * instant_hazard),
                (eta * cumulative_hazard, (1.0 + eta) * instant_hazard),
            ]))
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
            let (cumulative_hazard, instant_hazard) = gompertz_hazard_components(age, rate, shape);
            let (d_cum_dshape, d_inst_dshape) =
                gompertz_cumulative_shape_derivative(age, rate, shape);
            Ok(Some(vec![
                (cumulative_hazard, instant_hazard),
                (d_cum_dshape, d_inst_dshape),
            ]))
        }
        SurvivalBaselineTarget::GompertzMakeham => {
            let rate = cfg.rate.ok_or_else(|| "gm missing rate".to_string())?;
            let shape = cfg.shape.ok_or_else(|| "gm missing shape".to_string())?;
            let makeham = cfg
                .makeham
                .ok_or_else(|| "gm missing makeham".to_string())?;
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
    if !age.is_finite() || age <= 0.0 {
        return Err(
            "survival ages must be finite and positive for baseline hazard evaluation".to_string(),
        );
    }

    match cfg.target {
        SurvivalBaselineTarget::Linear => Ok(None),
        SurvivalBaselineTarget::Weibull => {
            let scale = cfg.scale.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(1.0);
            if !(scale.is_finite() && shape.is_finite() && scale > 0.0 && shape > 0.0) {
                return Err("weibull baseline requires finite positive scale and shape".to_string());
            }
            let cumulative_hazard = (age / scale).powf(shape);
            let instant_hazard = shape * cumulative_hazard / age;
            Ok(Some((cumulative_hazard, instant_hazard)))
        }
        SurvivalBaselineTarget::Gompertz => {
            let rate = cfg.rate.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(0.0);
            let (cumulative_hazard, instant_hazard) = gompertz_hazard_components(age, rate, shape);
            Ok(Some((cumulative_hazard, instant_hazard)))
        }
        SurvivalBaselineTarget::GompertzMakeham => {
            let makeham = cfg.makeham.unwrap_or(0.0);
            let rate = cfg.rate.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(0.0);
            let (h_gompertz, inst_gompertz) = gompertz_hazard_components(age, rate, shape);
            Ok(Some((makeham * age + h_gompertz, makeham + inst_gompertz)))
        }
    }
}

/// Evaluate the parametric baseline target at a given age.
/// Returns `(eta_target(age), d eta_target / d age)` on the log-cumulative-hazard scale.
pub fn evaluate_survival_baseline(
    age: f64,
    cfg: &SurvivalBaselineConfig,
) -> Result<(f64, f64), String> {
    if !age.is_finite() || age <= 0.0 {
        return Err(
            "survival ages must be finite and positive for baseline target evaluation".to_string(),
        );
    }

    match cfg.target {
        SurvivalBaselineTarget::Linear => Ok((0.0, 0.0)),
        SurvivalBaselineTarget::Weibull => {
            let scale = cfg.scale.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(1.0);
            let eta = shape * (age.ln() - scale.ln());
            let derivative = shape / age;
            Ok((eta, derivative))
        }
        SurvivalBaselineTarget::Gompertz => {
            let rate = cfg.rate.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(0.0);
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
        SurvivalBaselineTarget::GompertzMakeham => {
            let makeham = cfg.makeham.unwrap_or(0.0);
            let rate = cfg.rate.unwrap_or(1.0);
            let shape = cfg.shape.unwrap_or(0.0);
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
    let Some((cumulative_hazard, instant_hazard)) =
        survival_cumulative_and_instant_hazard(age, cfg)?
    else {
        return Ok((0.0, 0.0));
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
    Ok((q, instant_hazard * survival / phi_q))
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
    let hazard_partials = survival_hazard_theta_partials(age, cfg)?
        .ok_or_else(|| "unexpected missing hazard partials for nonlinear baseline".to_string())?;
    let a = survival / phi_q;
    let a_log_derivative_factor = q * a - 1.0;
    Ok(Some(
        hazard_partials
            .into_iter()
            .map(|(d_h_cum, d_h_inst)| {
                (
                    a * d_h_cum,
                    a * (d_h_inst + instant_hazard * a_log_derivative_factor * d_h_cum),
                )
            })
            .collect(),
    ))
}

// ---------------------------------------------------------------------------
// Baseline offsets
// ---------------------------------------------------------------------------

/// Compute baseline target offsets for all observations.
/// Returns `(eta_entry, eta_exit, derivative_exit)`.
pub fn build_survival_baseline_offsets(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    if age_entry.len() != age_exit.len() {
        return Err("survival baseline offsets require matching entry/exit lengths".to_string());
    }
    let n = age_entry.len();
    let mut eta_entry = Array1::<f64>::zeros(n);
    let mut eta_exit = Array1::<f64>::zeros(n);
    let mut derivative_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (e0, _) = evaluate_survival_baseline(age_entry[i], cfg)?;
        let (e1, d1) = evaluate_survival_baseline(age_exit[i], cfg)?;
        if !e0.is_finite() || !e1.is_finite() || !d1.is_finite() {
            return Err("non-finite survival baseline offsets computed".to_string());
        }
        eta_entry[i] = e0;
        eta_exit[i] = e1;
        derivative_exit[i] = d1;
    }
    Ok((eta_entry, eta_exit, derivative_exit))
}

/// Compute marginal-slope probit baseline target offsets for all observations.
/// Returns `(q_entry, q_exit, q_derivative_exit)` where `Phi(-q(t)) = exp(-H0(t))`.
pub fn build_survival_marginal_slope_baseline_offsets(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    cfg: &SurvivalBaselineConfig,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>), String> {
    if age_entry.len() != age_exit.len() {
        return Err(
            "survival marginal-slope baseline offsets require matching entry/exit lengths"
                .to_string(),
        );
    }
    let n = age_entry.len();
    let mut eta_entry = Array1::<f64>::zeros(n);
    let mut eta_exit = Array1::<f64>::zeros(n);
    let mut derivative_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (e0, _) = evaluate_survival_marginal_slope_baseline(age_entry[i], cfg)?;
        let (e1, d1) = evaluate_survival_marginal_slope_baseline(age_exit[i], cfg)?;
        if !e0.is_finite() || !e1.is_finite() || !d1.is_finite() {
            return Err("non-finite survival marginal-slope baseline offsets computed".to_string());
        }
        eta_entry[i] = e0;
        eta_exit[i] = e1;
        derivative_exit[i] = d1;
    }
    Ok((eta_entry, eta_exit, derivative_exit))
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
            return (rate * age, rate);
        }
        let shape_age = shape * age;
        let cumulative_hazard = (rate / shape) * shape_age.exp_m1();
        let instant_hazard = rate * shape_age.exp();
        (cumulative_hazard, instant_hazard)
    }

    let n = age_entry.len();
    let mut loaded_eta_entry = Array1::<f64>::zeros(n);
    let mut loaded_eta_exit = Array1::<f64>::zeros(n);
    let mut loaded_derivative_exit = Array1::<f64>::zeros(n);
    let mut unloaded_mass_entry = Array1::<f64>::zeros(n);
    let mut unloaded_mass_exit = Array1::<f64>::zeros(n);
    let mut unloaded_hazard_exit = Array1::<f64>::zeros(n);

    for i in 0..n {
        let entry = age_entry[i];
        let exit = age_exit[i];
        if !entry.is_finite() || !exit.is_finite() || entry <= 0.0 || exit <= 0.0 || exit < entry {
            return Err(format!(
                "latent survival baseline offsets require finite positive entry/exit ages with exit >= entry (row {})",
                i + 1
            ));
        }
        match loading {
            HazardLoading::Full => {
                let (eta_entry, _) = evaluate_survival_baseline(entry, cfg)?;
                let (eta_exit, derivative_exit) = evaluate_survival_baseline(exit, cfg)?;
                loaded_eta_entry[i] = eta_entry;
                loaded_eta_exit[i] = eta_exit;
                loaded_derivative_exit[i] = derivative_exit;
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
                loaded_eta_entry[i] = loaded_entry.ln();
                loaded_eta_exit[i] = loaded_exit.ln();
                loaded_derivative_exit[i] = loaded_hazard / loaded_exit;
                unloaded_mass_entry[i] = makeham * entry;
                unloaded_mass_exit[i] = makeham * exit;
                unloaded_hazard_exit[i] = makeham;
            }
        }
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
        },
    )
    .map_err(|e| format!("failed to evaluate {block_name} time-margin basis at entry: {e}"))?;
    let time_design_entry = time_build_entry.design.to_dense();
    let p_time = time_design_exit.ncols();
    let mut time_design_derivative_exit = Array2::<f64>::zeros((age_exit.len(), p_time));
    let mut deriv_buf = vec![0.0_f64; p_time];
    for i in 0..age_exit.len() {
        deriv_buf.fill(0.0);
        evaluate_bspline_derivative_scalar(log_exit[i], knots.view(), time_degree, &mut deriv_buf)
            .map_err(|e| {
                format!("failed to evaluate {block_name} time-margin derivative basis: {e}")
            })?;
        let chain = 1.0 / age_exit[i].max(1e-12);
        for j in 0..p_time {
            time_design_derivative_exit[[i, j]] = deriv_buf[j] * chain;
        }
    }

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
        baseline_chain_rule_gradient, baseline_offset_theta_partials,
        build_survival_marginal_slope_baseline_offsets, build_survival_timewiggle_from_baseline,
        evaluate_survival_baseline, evaluate_survival_marginal_slope_baseline,
        marginal_slope_baseline_chain_rule_gradient,
        marginal_slope_baseline_offset_theta_partials, survival_baseline_config_from_theta,
        survival_baseline_theta_from_config, SurvivalBaselineConfig, SurvivalBaselineTarget,
    };
    use crate::families::survival::OffsetChannelResiduals;
    use crate::inference::formula_dsl::LinkWiggleFormulaSpec;
    use crate::probability::normal_cdf;
    use ndarray::{array, Array1};

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
            let exit_partials =
                marginal_slope_baseline_offset_theta_partials(age_exit[i], &cfg)
                    .expect("exit partials")
                    .expect("nonlinear");
            let entry_partials =
                marginal_slope_baseline_offset_theta_partials(age_entry[i], &cfg)
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
        let ok = if expected.abs() < 1.0 {
            (actual - expected).abs() < tol
        } else {
            (actual - expected).abs() < tol * expected.abs().max(1.0)
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
        };
        let grad =
            baseline_chain_rule_gradient(age_entry.view(), age_exit.view(), &cfg, &residuals)
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
        };
        // Must not error despite age_entry[0] == 0.
        let grad =
            baseline_chain_rule_gradient(age_entry.view(), age_exit.view(), &cfg, &residuals)
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
        };
        let grad =
            baseline_chain_rule_gradient(age_entry.view(), age_exit.view(), &cfg, &residuals)
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
        };
        let grad =
            baseline_chain_rule_gradient(age_entry.view(), age_exit.view(), &cfg, &residuals)
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
        };
        let err = baseline_chain_rule_gradient(age_entry.view(), age_exit.view(), &cfg, &residuals)
            .expect_err("length mismatch must error");
        assert!(err.contains("length mismatch"), "err={err}");
    }
}
