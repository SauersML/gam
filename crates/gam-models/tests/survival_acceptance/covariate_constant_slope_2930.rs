//! gam#2930: a survival marginal-slope fit whose baseline carries covariates
//! and whose slope is constant in follow-up time was refused at seed
//! validation on every seed with "best coefficient-mode candidate 0 changed
//! profile objective between value screening and derivative assembly".
//!
//! Value screening and derivative assembly evaluate the profiled criterion at
//! the same smoothing point from the same owned coefficient mode, so the two
//! scalars must agree to roundoff. These fits must therefore complete, and
//! complete with a truthful convergence certificate.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_solve::estimate::outer_eval_capture::{
    OuterSeedOrder, OuterSeedProbe, observe_next_outer_seed,
};
use gam_custom_family::{CompletionCurvatureAblation, set_completion_curvature_ablation};
use gam_solve::model_types::{CurvatureAdmissibility, CurvatureEvidence};
use ndarray::{Array1, Array2};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use gam_linalg::utils::splitmix64;

/// Planted constant slope of the latent score on the probit survival index.
const SLOPE: f64 = 0.85;
/// Planted Weibull baseline `exp(−(t/λ)^k)` of the marginal survival at `x = 0`: `log λ`. The
/// marginal probit index is the Weibull chart's own curve `q(t) = −Φ⁻¹(exp(−(t/λ)^k))`, so the
/// chart holds the planted baseline at a finite point. The level and trend match a probit index
/// `−1.15 + 0.95·log t` at `t = 1`; that index is linear in `log t`, which the chart reaches only
/// in its `k → 0`, `λ → 0` limit, where the criterion has no interior optimum (gam#2969).
const BASELINE_LOG_SCALE: f64 = 1.202;
/// Planted Weibull shape `log k`.
const BASELINE_LOG_SHAPE: f64 = 0.516;
/// Covariate effect on the marginal probit index.
const COVARIATE_EFFECT: f64 = 0.4;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gaussian(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn normal_cdf(x: f64) -> f64 {
    gam_math::probability::normal_cdf(x)
}

/// Standard-normal quantile by bisection on `Φ`, independent of the crate.
fn normal_quantile(p: f64) -> f64 {
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if normal_cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

/// Event time from `Φ(−η(T)) = u` with the closed-form Gaussian lowering
/// `η = q(t, x)·√(1 + b²) + b·z` and `q(t, x) = −Φ⁻¹(exp(−(t/λ)^k)) + shift`. The baseline index
/// at the event is `q₀ = (−Φ⁻¹(u) − b·z)/√(1 + b²) − shift`, so `(T/λ)^k = −log Φ(−q₀)`.
fn planted_event_time(u: f64, z: f64, location_shift: f64) -> f64 {
    let target = -normal_quantile(u);
    let baseline_index = (target - SLOPE * z) / (1.0 + SLOPE * SLOPE).sqrt() - location_shift;
    let cumulative_hazard = -normal_cdf(-baseline_index).ln();
    BASELINE_LOG_SCALE.exp() * cumulative_hazard.powf((-BASELINE_LOG_SHAPE).exp())
}

/// One covariate `x`, a standard-normal frozen score `z`, uniform censoring.
fn minimal_dataset(n: usize, seed: u64) -> gam_data::EncodedDataset {
    let headers = ["time", "event", "z", "x"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state = seed;
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let z = next_gaussian(&mut state);
        let x = next_gaussian(&mut state);
        let u = next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6);
        let event_time = planted_event_time(u, z, COVARIATE_EFFECT * x);
        let censor = 0.35 + 5.0 * next_unit(&mut state);
        let (time, event) = if event_time <= censor {
            (event_time, 1u8)
        } else {
            (censor, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        rows.push(StringRecord::from(vec![
            format!("{time:.17e}"),
            event.to_string(),
            format!("{z:.17e}"),
            format!("{x:.17e}"),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2930 fixture")
}

fn constant_slope_config() -> FitConfig {
    FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some("1".to_string()),
        z_column: Some("z".to_string()),
        frozen_score: true,
        baseline_target: "weibull".to_string(),
        time_num_internal_knots: 3,
        precompute_conformal: Some(false),
        ..FitConfig::default()
    }
}

fn fit_and_report(label: &str, formula: &str, data: &gam_data::EncodedDataset, config: &FitConfig) {
    let started = Instant::now();
    let result = fit_from_formula(formula, data, config)
        .unwrap_or_else(|error| panic!("[2930 {label}] `{formula}` must fit: {error}"));
    let FitResult::SurvivalMarginalSlope(fit) = result else {
        panic!("[2930 {label}] expected a SurvivalMarginalSlope fit result");
    };
    // `UnifiedFitResult`'s checked constructor refuses unconverged inner or outer
    // evidence and seals what it accepted, so reaching here is the convergence
    // certificate; the report prints that sealed evidence.
    let slope_design = fit.slope_design.design.to_dense();
    let slope = slope_design.row(0).dot(&fit.fit.blocks[2].beta) + fit.baseline_slope;
    eprintln!(
        "[2930 {label}] formula={formula:?} slope={slope:.6} log_lik={:.6e} outer_iterations={} outer_gradient_norm={:?} evidence={:?} elapsed={:.2}s",
        fit.fit.log_likelihood,
        fit.fit.outer_iterations,
        fit.fit.outer_gradient_norm,
        fit.fit.convergence_evidence(),
        started.elapsed().as_secs_f64(),
    );
    assert!(slope.is_finite(), "[2930 {label}] the slope must be finite");
}

#[test]
fn covariate_constant_slope_survival_fit_passes_seed_validation_2930() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let data = minimal_dataset(800, 0x2930_0000_0001);
    let config = constant_slope_config();
    fit_and_report("linear", "Surv(time, event) ~ x", &data, &config);
    fit_and_report("smooth", "Surv(time, event) ~ s(x, k=5)", &data, &config);
}

/// gam#2945: a learned Gaussian frailty σ moves the priced completion, whose explicit σ derivative
/// is not derived, so such a fit has neither an exact outer gradient nor a curvature certificate. It
/// is refused once, by name, before the smoothing search: not on every value+gradient evaluation,
/// and not after the search at the curvature guard.
#[test]
fn covariate_constant_slope_learned_sigma_is_refused_by_name_2945() {
    use gam_models::survival::lognormal_kernel::{FrailtyScale, FrailtySpec};

    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let data = minimal_dataset(800, 0x2930_0000_0001);
    let config = FitConfig {
        frailty: FrailtySpec::GaussianShift {
            scale: FrailtyScale::Learned { initial_sigma: 0.5 },
        },
        ..constant_slope_config()
    };
    let message = match fit_from_formula("Surv(time, event) ~ x", &data, &config) {
        Ok(_) => panic!("a learned frailty σ with the armed Jeffreys completion must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("a learned Gaussian frailty σ with the armed Jeffreys completion is refused"),
        "the refusal must name its reason, got: {message}"
    );
}

/// One outer coordinate's analytic gradient against a central difference of the criterion.
#[derive(Debug)]
struct CoordinateGrade {
    point: &'static str,
    coordinate: usize,
    analytic: f64,
    difference: f64,
    step: f64,
    /// `|D(2h) − D(h)|` at the accepted rung: the ladder's own disagreement.
    settle: f64,
    /// The accepted difference's error band, from [`derived_band`].
    band: f64,
}

/// gam#2945: the error band of a central difference `D(h)` from its own error sources, not a literal
/// floor.
///
/// Truncation: with `E(h) = D(h) − f′`, the halving ladder's disagreement `Δ₁ = D(2h) − D(h)` is
/// `E(2h) − E(h)`, so `|E(h)| ≤ |Δ₁|` whenever the error at least halves per halving,
/// `|E(2h)| ≥ 2·|E(h)|`. A central difference in its asymptotic range quarters it (`E = a·h² + …`),
/// where `|E(h)| = |Δ₁|/3`. The richer estimate `Δ₁/3 − (Δ₂ − 4Δ₁)/45`, with `Δ₂ = D(4h) − D(2h)`, is
/// not a bound: its correction `180·b·h⁴ + 3780·c·h⁶` can cancel while `64·c·h⁶` of true error remains.
/// It fell 0.2 % short on this criterion's Hessian. Every graded entry measured at most `0.335·|Δ₁|`.
///
/// Roundoff: an evaluation assembled as a floating sum of `n` summands carries Higham's
/// recursive-summation error `(n − 1)·ε·Σ|x|` beside the summands' own `ε·Σ|x|`, so `δ ≤ n·ε·Σ|x|`.
/// The criterion publishes its atoms but not their summands, so `Σ|x|` is taken at `M`, the largest
/// magnitude the two rungs took: exact when the summands share a sign, a named proxy otherwise.
/// With `δ = n·ε·M`, `D(h)` carries `δ/h` and `Δ₁` carries `1.5·δ/h`, so the band adds `2.5·δ/h`.
fn derived_band(delta_1: f64, magnitude: f64, step: f64, summands: usize) -> f64 {
    let roundoff = summands as f64 * f64::EPSILON * magnitude;
    delta_1.abs() + 2.5 * roundoff / step
}

/// Central differences of the value-only criterion along `theta[j]` on a halving ladder whose
/// first rung is one hundredth of the coordinate's scale and at most half its room inside the
/// seed box. Returns the rung that agrees best with its predecessor: its difference, its step, that
/// disagreement, and its [`derived_band`].
fn value_central_difference(
    probe: &mut dyn OuterSeedProbe,
    theta: &Array1<f64>,
    j: usize,
    summands: usize,
) -> Result<(f64, f64, f64, f64), String> {
    let layout = probe.layout();
    let room = (theta[j] - layout.lower[j])
        .min(layout.upper[j] - theta[j])
        .max(0.0);
    let mut step = (1.0e-2 * (1.0 + theta[j].abs())).min(0.5 * room);
    if !(step > 0.0) {
        return Err(format!("coordinate {j} sits on a face of the seed box"));
    }
    let mut value_at = |offset: f64| -> Result<f64, String> {
        let mut displaced = theta.clone();
        displaced[j] += offset;
        probe
            .evaluate(&displaced, OuterSeedOrder::Value)
            .map(|evaluation| evaluation.cost)
            .map_err(|error| format!("theta[{j}] displaced by {offset:e}: {error}"))
    };
    let mut estimates: Vec<(f64, f64, f64)> = Vec::with_capacity(6);
    for _ in 0..6 {
        let plus = value_at(step)?;
        let minus = value_at(-step)?;
        estimates.push((step, (plus - minus) / (2.0 * step), plus.abs().max(minus.abs())));
        step *= 0.5;
    }
    let (index, settle) = (1..estimates.len())
        .map(|i| (i, (estimates[i - 1].1 - estimates[i].1).abs()))
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .expect("the ladder has six rungs");
    let delta_1 = estimates[index - 1].1 - estimates[index].1;
    let magnitude = estimates[index - 1..=index]
        .iter()
        .fold(0.0_f64, |acc, rung| acc.max(rung.2));
    let accepted_step = estimates[index].0;
    Ok((
        estimates[index].1,
        accepted_step,
        settle,
        derived_band(delta_1, magnitude, accepted_step, summands),
    ))
}

/// One column of the analytic outer Hessian against a central difference of the analytic
/// gradient along that column's coordinate.
#[derive(Debug)]
struct HessianColumnGrade {
    point: &'static str,
    column: usize,
    analytic: Array1<f64>,
    difference: Array1<f64>,
    step: f64,
    /// `max_i |D(2h)_i − D(h)_i|` at the accepted rung.
    settle: f64,
    /// Each entry's [`derived_band`].
    band: Array1<f64>,
}

/// Central differences of the analytic gradient along `theta[j]` on the same halving ladder as
/// [`value_central_difference`]. Returns the rung whose largest entrywise disagreement with its
/// predecessor is smallest: its difference, its step, that disagreement, and each entry's
/// [`derived_band`].
fn gradient_central_difference(
    probe: &mut dyn OuterSeedProbe,
    theta: &Array1<f64>,
    j: usize,
    summands: usize,
) -> Result<(Array1<f64>, f64, f64, Array1<f64>), String> {
    let layout = probe.layout();
    let room = (theta[j] - layout.lower[j])
        .min(layout.upper[j] - theta[j])
        .max(0.0);
    let mut step = (1.0e-2 * (1.0 + theta[j].abs())).min(0.5 * room);
    if !(step > 0.0) {
        return Err(format!("coordinate {j} sits on a face of the seed box"));
    }
    let mut gradient_at = |offset: f64| -> Result<Array1<f64>, String> {
        let mut displaced = theta.clone();
        displaced[j] += offset;
        probe
            .evaluate(&displaced, OuterSeedOrder::ValueAndGradient)
            .map_err(|error| format!("theta[{j}] displaced by {offset:e}: {error}"))?
            .gradient
            .ok_or_else(|| format!("theta[{j}] displaced by {offset:e}: no gradient published"))
    };
    let mut estimates: Vec<(f64, Array1<f64>, Array1<f64>)> = Vec::with_capacity(6);
    for _ in 0..6 {
        let plus = gradient_at(step)?;
        let minus = gradient_at(-step)?;
        let magnitude = Array1::from_shape_fn(plus.len(), |i| plus[i].abs().max(minus[i].abs()));
        estimates.push((step, (&plus - &minus) / (2.0 * step), magnitude));
        step *= 0.5;
    }
    let (index, settle) = (1..estimates.len())
        .map(|i| {
            let disagreement = (&estimates[i - 1].1 - &estimates[i].1)
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()));
            (i, disagreement)
        })
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .expect("the ladder has six rungs");
    let accepted_step = estimates[index].0;
    let band = Array1::from_shape_fn(estimates[index].1.len(), |i| {
        let delta_1 = estimates[index - 1].1[i] - estimates[index].1[i];
        let magnitude = estimates[index - 1..=index]
            .iter()
            .fold(0.0_f64, |acc, rung| acc.max(rung.2[i]));
        derived_band(delta_1, magnitude, accepted_step, summands)
    });
    Ok((estimates[index].1.clone(), accepted_step, settle, band))
}

/// Grade the analytic gradient against central differences of the value-only criterion, and every
/// column of the analytic outer Hessian against central differences of the analytic gradient.
/// The outer Hessian at one point with one second-order completion term omitted: a positive control
/// the finite-difference gate must reject.
#[derive(Debug)]
struct AblatedHessian {
    point: &'static str,
    term: CompletionCurvatureAblation,
    hessian: Array2<f64>,
}

fn grade_point(
    probe: &mut dyn OuterSeedProbe,
    point: &'static str,
    theta: &Array1<f64>,
    rows: usize,
) -> Result<(Vec<CoordinateGrade>, Vec<HessianColumnGrade>, Vec<AblatedHessian>), String> {
    let evaluation = probe
        .evaluate(theta, OuterSeedOrder::ValueAndGradient)
        .map_err(|error| format!("{point}: gradient evaluation: {error}"))?;
    let coefficients = evaluation
        .selected_mode
        .as_ref()
        .map(|(beta, _)| beta.len())
        .ok_or_else(|| format!("{point}: the gradient evaluation published no selected mode"))?;
    // Higham summand count for any criterion or gradient atom: the row log-likelihood terms, the
    // penalty quadratic and the two log-determinants.
    let summands = rows + coefficients * coefficients + 2 * coefficients;
    let gradient = evaluation
        .gradient
        .ok_or_else(|| format!("{point}: the gradient evaluation published no gradient"))?;
    let curvature = probe
        .evaluate(theta, OuterSeedOrder::ValueGradientHessian)
        .map_err(|error| format!("{point}: curvature evaluation: {error}"))?;
    let hessian = curvature
        .hessian
        .ok_or_else(|| format!("{point}: the curvature evaluation published no outer Hessian"))?;
    let dim = theta.len();
    if gradient.len() != dim || hessian.dim() != (dim, dim) {
        return Err(format!(
            "{point}: derivative shapes {} / {:?} against theta dimension {dim}",
            gradient.len(),
            hessian.dim()
        ));
    }
    let mut grades = Vec::with_capacity(dim);
    let mut columns = Vec::with_capacity(dim);
    for j in 0..dim {
        let (difference, step, settle, band) = value_central_difference(probe, theta, j, summands)?;
        grades.push(CoordinateGrade {
            point,
            coordinate: j,
            analytic: gradient[j],
            difference,
            step,
            settle,
            band,
        });
        let (column_difference, column_step, column_settle, column_band) =
            gradient_central_difference(probe, theta, j, summands)?;
        columns.push(HessianColumnGrade {
            point,
            column: j,
            analytic: hessian.column(j).to_owned(),
            difference: column_difference,
            step: column_step,
            settle: column_settle,
            band: column_band,
        });
    }
    let mut ablated = Vec::with_capacity(2);
    for term in [CompletionCurvatureAblation::PsiPair, CompletionCurvatureAblation::BetaPsi] {
        set_completion_curvature_ablation(Some(term));
        let outcome = probe.evaluate(theta, OuterSeedOrder::ValueGradientHessian);
        set_completion_curvature_ablation(None);
        let hessian = outcome
            .map_err(|error| format!("{point}: curvature evaluation without {term:?}: {error}"))?
            .hessian
            .ok_or_else(|| {
                format!("{point}: the curvature evaluation without {term:?} published no outer Hessian")
            })?;
        ablated.push(AblatedHessian { point, term, hessian });
    }
    Ok((grades, columns, ablated))
}

/// Print a point's grades as the probe measures them, so the derivative evidence survives a fit
/// that runs past its time limit after the probe returns.
fn report_grades(grades: &[CoordinateGrade], columns: &[HessianColumnGrade], rho_dim: usize) {
    for grade in grades {
        let kind = if grade.coordinate < rho_dim { "rho" } else { "psi" };
        eprintln!(
            "[2930-FD] point={} coordinate={} kind={kind} analytic={:.9e} difference={:.9e} \
             relative_error={:.3e} step={:e} settle={:e} band={:e} graded={}",
            grade.point,
            grade.coordinate,
            grade.analytic,
            grade.difference,
            (grade.analytic - grade.difference).abs() / grade.analytic.abs().max(1.0),
            grade.step,
            grade.settle,
            grade.band,
            coordinate_resolved(grade),
        );
    }
    for grade in columns {
        for row in 0..grade.analytic.len() {
            eprintln!(
                "[2930-FD-HESSIAN] point={} row={row} column={} analytic={:.9e} difference={:.9e} \
                 relative_error={:.3e} step={:e} settle={:e} band={:e} graded={}",
                grade.point,
                grade.column,
                grade.analytic[row],
                grade.difference[row],
                (grade.analytic[row] - grade.difference[row]).abs() / grade.analytic[row].abs().max(1.0),
                grade.step,
                grade.settle,
                grade.band[row],
                column_resolved(grade),
            );
        }
    }
}

/// A coordinate's central difference resolves the derivative it grades when its [`derived_band`] is
/// below the derivative's own magnitude.
fn coordinate_resolved(grade: &CoordinateGrade) -> bool {
    grade.band <= grade.analytic.abs().max(grade.difference.abs())
}

/// A Hessian column's central differences resolve the column when every entry's [`derived_band`] is
/// below the column's largest analytic magnitude.
fn column_resolved(grade: &HessianColumnGrade) -> bool {
    let scale = grade
        .analytic
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    grade.band.iter().all(|band| *band <= scale)
}

/// gam#2930 consistency: every eval mode prices one criterion, the complete Jeffreys curvature
/// `M_true`, so at a ψ-bearing hyperpoint of the covariate fixture the analytic gradient, which
/// carries the completion's explicit ψ derivative, is the derivative of the value-only criterion,
/// and every column of the analytic outer Hessian, which carries the completion's ψψ and ρψ
/// curvature, is the derivative of that gradient. Graded at the lent seed and at an interior point
/// off it. The planted baseline lies inside the chart, so the fit must certify its optimum on a
/// measured positive-semidefinite outer Hessian.
#[test]
fn covariate_constant_slope_derivatives_differentiate_the_value_criterion_2930() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    #[cfg(target_os = "macos")]
    gam_gpu::configure_global_policy(gam_gpu::GpuPolicy::Off);

    let rows = 800;
    let data = minimal_dataset(rows, 0x2930_0000_0001);
    let config = constant_slope_config();
    type Grades = (Vec<CoordinateGrade>, Vec<HessianColumnGrade>, Vec<AblatedHessian>, usize, usize);
    let captured: Rc<RefCell<Option<Result<Grades, String>>>> = Rc::new(RefCell::new(None));
    let sink = Rc::clone(&captured);
    observe_next_outer_seed(
        1,
        Box::new(
            move |probe: &mut dyn OuterSeedProbe| -> Result<(), gam_solve::estimate::EstimationError> {
                let layout = probe.layout().clone();
                let outcome = (|| -> Result<Grades, String> {
                    let interior = Array1::from_shape_fn(layout.seed.len(), |i| {
                        (layout.seed[i] + 0.5).clamp(layout.lower[i], layout.upper[i])
                    });
                    let mut grades = Vec::new();
                    let mut columns = Vec::new();
                    let mut ablated = Vec::new();
                    for (point, theta) in [("seed", layout.seed.clone()), ("interior", interior)] {
                        let (point_grades, point_columns, point_ablated) =
                            grade_point(probe, point, &theta, rows)?;
                        report_grades(&point_grades, &point_columns, layout.rho_dim);
                        grades.extend(point_grades);
                        columns.extend(point_columns);
                        ablated.extend(point_ablated);
                    }
                    Ok((grades, columns, ablated, layout.rho_dim, layout.psi_dim))
                })();
                *sink.borrow_mut() = Some(outcome);
                Ok(())
            },
        ),
    );
    let refit = fit_from_formula("Surv(time, event) ~ x", &data, &config);
    let (grades, columns, ablated, rho_dim, psi_dim) = captured
        .borrow_mut()
        .take()
        .unwrap_or_else(|| panic!("the outer runner lent no ψ-bearing seed probe: {:?}", refit.as_ref().err()))
        .unwrap_or_else(|reason| panic!("the seed probe refused: {reason}"));
    assert!(psi_dim >= 1, "the probe was lent at a seed without ψ coordinates");

    // Every entry is graded against its derived band. An entry whose band is below its own magnitude
    // is an informative grade; every class of outer coordinate must carry at least one at each point.
    let class_of = |row: usize, column: Option<usize>| -> &'static str {
        match (row < rho_dim, column.map(|column| column < rho_dim)) {
            (true, None) => "gradient rho",
            (false, None) => "gradient psi",
            (true, Some(true)) => "Hessian rho-rho",
            (false, Some(false)) => "Hessian psi-psi",
            _ => "Hessian rho-psi",
        }
    };
    let mut coverage =
        std::collections::BTreeMap::<(&'static str, &'static str), (usize, usize)>::new();
    let mut failures = Vec::new();
    for grade in &grades {
        let error = (grade.analytic - grade.difference).abs();
        let tally = coverage
            .entry((grade.point, class_of(grade.coordinate, None)))
            .or_default();
        tally.0 += 1;
        if coordinate_resolved(grade) {
            tally.1 += 1;
        }
        if error > grade.band {
            failures.push(format!(
                "{} gradient [{}]: analytic {:e} vs central difference {:e}, band {:e}",
                grade.point, grade.coordinate, grade.analytic, grade.difference, grade.band
            ));
        }
    }
    assert!(failures.is_empty(), "the gradient disagrees with the value criterion: {failures:?}");

    let mut hessian_failures = Vec::new();
    for grade in &columns {
        for row in 0..grade.analytic.len() {
            let error = (grade.analytic[row] - grade.difference[row]).abs();
            let magnitude = grade.analytic[row].abs().max(grade.difference[row].abs());
            let tally = coverage
                .entry((grade.point, class_of(row, Some(grade.column))))
                .or_default();
            tally.0 += 1;
            if grade.band[row] <= magnitude {
                tally.1 += 1;
            }
            if error > grade.band[row] {
                hessian_failures.push(format!(
                    "{} Hessian [{row}, {}]: analytic {:e} vs central difference {:e}, band {:e}",
                    grade.point, grade.column, grade.analytic[row], grade.difference[row], grade.band[row]
                ));
            }
        }
    }
    assert!(
        hessian_failures.is_empty(),
        "the outer Hessian disagrees with the gradient: {hessian_failures:?}"
    );
    for ((point, class), (graded, informative)) in &coverage {
        eprintln!("[2930-FD-COVERAGE] point={point} class={class} graded={graded} informative={informative}");
    }
    for point in ["seed", "interior"] {
        for class in [
            "gradient rho",
            "gradient psi",
            "Hessian rho-rho",
            "Hessian rho-psi",
            "Hessian psi-psi",
        ] {
            let (graded, informative) = coverage.get(&(point, class)).copied().unwrap_or((0, 0));
            assert!(
                informative >= 1,
                "{point}: class {class} carries {graded} graded entries and no informative one \
                 (every central difference's band exceeds the magnitude it grades)"
            );
        }
    }

    // gam#2945 positive controls at the outer level: with one second-order completion term omitted,
    // the same bands must reject the outer Hessian.
    for control in &ablated {
        let mut rejected = 0usize;
        for grade in columns.iter().filter(|grade| grade.point == control.point) {
            for row in 0..grade.analytic.len() {
                let error = (control.hessian[[row, grade.column]] - grade.difference[row]).abs();
                if error > grade.band[row] {
                    rejected += 1;
                }
            }
        }
        eprintln!(
            "[2930-CONTROL-OUTER] point={} without={:?} rejected_entries={rejected}",
            control.point, control.term
        );
        assert!(
            rejected >= 1,
            "{}: the outer Hessian without {:?} passed every band, so the gate cannot see that term",
            control.point,
            control.term
        );
    }

    // The fit declares the analytic outer Hessian, so the certificate it mints carries a measured
    // verdict, and at the certified point that verdict must admit a local minimum.
    let refit = refit.unwrap_or_else(|error| panic!("the observed fit must complete: {error}"));
    let FitResult::SurvivalMarginalSlope(fit) = refit else {
        panic!("expected a SurvivalMarginalSlope fit result");
    };
    let certificate = fit
        .fit
        .convergence_evidence()
        .outer_certificate()
        .expect("a smoothing search mints an analytic outer certificate");
    eprintln!(
        "[2930-CERTIFICATE] stationary={} curvature={} verdict={}",
        certificate.is_stationary(),
        certificate.curvature,
        certificate.curvature_verdict(),
    );
    assert!(certificate.is_stationary(), "the certificate must clear its stationarity bound");
    assert_eq!(
        certificate.curvature,
        CurvatureEvidence::Measured { psd: true },
        "the certified point must carry a measured positive-semidefinite outer Hessian: {}",
        certificate.curvature
    );
    assert!(
        matches!(
            certificate.curvature_verdict(),
            CurvatureAdmissibility::Admissible
        ),
        "the curvature verdict must admit a local minimum: {}",
        certificate.curvature_verdict()
    );
}
