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

use ndarray::{Array1, Array2, array, s};

use crate::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, BasisOptions, Dense,
    KnotSource, build_bspline_basis_1d, create_basis, evaluate_bspline_derivative_scalar,
};
use crate::families::gamlss::{
    WiggleBlockConfig, append_selected_wiggle_penalty_orders, buildwiggle_block_input_from_seed,
    monotone_wiggle_basis_with_derivative_order, split_wiggle_penalty_orders,
};
use crate::families::survival_location_scale::{
    ResidualDistribution, SurvivalCovariateTermBlockTemplate,
};
use crate::inference::formula_dsl::LinkWiggleFormulaSpec;

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

#[derive(Clone, Debug)]
pub struct SurvivalTimeBuildOutput {
    pub x_entry_time: Array2<f64>,
    pub x_exit_time: Array2<f64>,
    pub x_derivative_time: Array2<f64>,
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
            let rate = rate.ok_or_else(|| {
                "--baseline-target gompertz requires --baseline-rate > 0".to_string()
            })?;
            let shape = shape.ok_or_else(|| {
                "--baseline-target gompertz requires --baseline-shape".to_string()
            })?;
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
            let rate = rate.ok_or_else(|| {
                "--baseline-target gompertz-makeham requires --baseline-rate > 0".to_string()
            })?;
            let shape = shape.ok_or_else(|| {
                "--baseline-target gompertz-makeham requires --baseline-shape".to_string()
            })?;
            let makeham = makeham.ok_or_else(|| {
                "--baseline-target gompertz-makeham requires --baseline-makeham > 0".to_string()
            })?;
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
        other => Err(format!(
            "unsupported --survival-likelihood '{other}'; use transformation|weibull|location-scale|marginal-slope"
        )),
    }
}

pub fn survival_likelihood_modename(mode: SurvivalLikelihoodMode) -> &'static str {
    match mode {
        SurvivalLikelihoodMode::Transformation => "transformation",
        SurvivalLikelihoodMode::Weibull => "weibull",
        SurvivalLikelihoodMode::LocationScale => "location-scale",
        SurvivalLikelihoodMode::MarginalSlope => "marginal-slope",
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
            x_entry_time: Array2::zeros((n, 0)),
            x_exit_time: Array2::zeros((n, 0)),
            x_derivative_time: Array2::zeros((n, 0)),
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
                x_entry_time,
                x_exit_time,
                x_derivative_time,
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
            let mut x_derivative_time = Array2::<f64>::zeros((n, p_time));
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
                    x_derivative_time[[i, j]] = deriv_buf[j] * chain;
                }
            }

            Ok(SurvivalTimeBuildOutput {
                x_entry_time: entry_basis.design.to_dense(),
                x_exit_time: exit_basis.design.to_dense(),
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

            let mut x_derivative_time = Array2::<f64>::zeros((n, p_time));
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
                    x_derivative_time[[i, j_new]] = d_i_log_full[j_old] * chain;
                }
            }
            if let Some((row, col)) = x_derivative_time.indexed_iter().find_map(|((i, j), v)| {
                if v.is_finite() { None } else { Some((i, j)) }
            }) {
                return Err(format!(
                    "survival ispline derivative basis produced non-finite value at row {}, column {}",
                    row + 1,
                    col + 1
                ));
            }

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
                x_entry_time,
                x_exit_time,
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
                    "survival BSpline anchor evaluation requires resolved knot metadata".to_string(),
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
                    "survival ISpline anchor evaluation requires resolved knot metadata".to_string(),
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
    design_entry: &mut Array2<f64>,
    design_exit: &mut Array2<f64>,
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
    for mut row in design_entry.rows_mut() {
        row -= &anchor_row.view();
    }
    for mut row in design_exit.rows_mut() {
        row -= &anchor_row.view();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Baseline evaluation (Gompertz, Weibull, Gompertz-Makeham)
// ---------------------------------------------------------------------------

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

    fn gompertz_components(age: f64, rate: f64, shape: f64) -> (f64, f64) {
        if shape.abs() < 1e-10 {
            return (rate * age, rate);
        }
        let shape_age = shape * age;
        let cumulative_hazard = (rate / shape) * shape_age.exp_m1();
        let instant_hazard = rate * shape_age.exp();
        (cumulative_hazard, instant_hazard)
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
            let (h, inst) = gompertz_components(age, rate, shape);
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
            let (h_gompertz, inst_gompertz) = gompertz_components(age, rate, shape);
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
    x_entry: &mut Array2<f64>,
    x_exit: &mut Array2<f64>,
    x_derivative: &mut Array2<f64>,
    tail_cols: usize,
) {
    if tail_cols == 0 {
        return;
    }
    let p_base = x_entry.ncols();
    let n = x_entry.nrows();
    let mut new_entry = Array2::<f64>::zeros((n, p_base + tail_cols));
    let mut new_exit = Array2::<f64>::zeros((n, p_base + tail_cols));
    let mut new_derivative = Array2::<f64>::zeros((n, p_base + tail_cols));
    new_entry.slice_mut(s![.., 0..p_base]).assign(x_entry);
    new_exit.slice_mut(s![.., 0..p_base]).assign(x_exit);
    new_derivative
        .slice_mut(s![.., 0..p_base])
        .assign(x_derivative);
    *x_entry = new_entry;
    *x_exit = new_exit;
    *x_derivative = new_derivative;
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
    use super::build_survival_timewiggle_from_baseline;
    use crate::inference::formula_dsl::LinkWiggleFormulaSpec;
    use ndarray::array;

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
}
