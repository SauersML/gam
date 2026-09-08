//! gam#2765 / gam#2767 probe: why does the outer REML solve refuse on a
//! follow-up-varying slope?
//!
//! The acceptance fixture fails with
//!
//! ```text
//!   solve FAILED n=2400 elapsed=1565s
//!   |Pg| = 8.392e1 against bound 2.401e-1
//!   termination = line_search_failed, StepSizeTooSmall after 50 attempts
//!   after 0 outer iteration(s)
//! ```
//!
//! Zero outer iterations and fifty rejected step sizes is not a slow optimizer;
//! it is the first line search failing outright. There are two candidate causes
//! and they call for different fixes, so this probe exists to separate them
//! rather than to guess:
//!
//! * the analytic outer gradient disagrees with the objective, so the search
//!   direction points somewhere the objective does not go; or
//! * the objective is not EVALUABLE along that direction — the inner solve
//!   refuses at the displaced ρ — in which case every trial reads as
//!   "no improvement" no matter how small the step.
//!
//! The certificate already hints at the second: `the FD-vs-analytic Hessian
//! probe could not evaluate the objective at every displaced point`. What
//! separates them is the per-evaluation log, which is why this runs the same
//! model at a size small enough to read.
//!
//! Run with `RUST_LOG=info` and grep for `outer-eval`.

use csv::StringRecord;
use gam::utils::splitmix64;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

const SLOPE_LEVEL: f64 = 0.85;
const SLOPE_TREND: f64 = -0.32;
const LOCATION_LEVEL: f64 = -1.15;
const LOCATION_TREND: f64 = 0.95;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(1e-12);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The covariate surface carried by the marginal index. Zero in the
/// intercept-only arms; a real function of `x` in the `smooth` arm, so the
/// marginal block has something to estimate.
fn planted_covariate(x: f64) -> f64 {
    0.45 * (2.4 * x).sin() - 0.25 * x
}

fn planted_eta(time: f64, z: f64, covariate: f64) -> f64 {
    let slope = SLOPE_LEVEL + SLOPE_TREND * time.ln();
    let location = LOCATION_LEVEL + LOCATION_TREND * time.ln() + covariate;
    location * (1.0 + slope * slope).sqrt() + slope * z
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

fn normal_quantile(p: f64) -> f64 {
    let cdf = |x: f64| 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2));
    let (mut low, mut high) = (-12.0_f64, 12.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if cdf(mid) < p {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn planted_event_time(u: f64, z: f64, covariate: f64) -> f64 {
    let target = -normal_quantile(u);
    let (mut low, mut high) = (-6.0_f64, 6.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if planted_eta(mid.exp(), z, covariate) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    (0.5 * (low + high)).exp()
}

/// The dataset, plus the two truth vectors an acceptance verdict needs: each
/// row's observed exit time and its latent score. A probe that only reports
/// "converged" answers a question nobody asked — the issue is whether the
/// fitted slope MOVES along follow-up and moves the planted way, and that is
/// checkable at every `n` this probe runs.
fn build_dataset(
    n: usize,
    with_covariate: bool,
) -> (gam::inference::data::EncodedDataset, Vec<f64>, Vec<f64>) {
    let headers = ["time", "event", "z", "x"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    let mut state: u64 = 0x2765_2767_5CA1_AB1E_u64;

    let mut raw_scores: Vec<f64> = Vec::with_capacity(n);
    let mut draws: Vec<f64> = Vec::with_capacity(n);
    let mut censor: Vec<f64> = Vec::with_capacity(n);
    let mut xs: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        raw_scores.push(next_gauss(&mut state));
        draws.push(next_unit(&mut state).clamp(1e-6, 1.0 - 1e-6));
        censor.push(next_unit(&mut state));
        xs.push(next_unit(&mut state) * 4.0 - 2.0);
    }
    let mean = raw_scores.iter().sum::<f64>() / n as f64;
    let variance = raw_scores.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = variance.sqrt().max(1e-12);
    let scores: Vec<f64> = raw_scores.iter().map(|v| (v - mean) / sd).collect();

    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    let mut observed_times: Vec<f64> = Vec::with_capacity(n);
    for index in 0..n {
        let z = scores[index];
        let covariate = if with_covariate {
            planted_covariate(xs[index])
        } else {
            0.0
        };
        let event_time = planted_event_time(draws[index], z, covariate);
        let censor_time = 0.35 + 5.0 * censor[index];
        let (time, event) = if event_time <= censor_time {
            (event_time, 1u8)
        } else {
            (censor_time, 0u8)
        };
        let time = time.clamp(1e-3, 1e3);
        observed_times.push(time);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            z.to_string(),
            xs[index].to_string(),
        ]));
    }
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode the #2765 probe fixture");
    (data, observed_times, scores)
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
    let n = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / n;
    let right_mean = right.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut left_var = 0.0;
    let mut right_var = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        let da = a - left_mean;
        let db = b - right_mean;
        cov += da * db;
        left_var += da * da;
        right_var += db * db;
    }
    cov / (left_var.sqrt() * right_var.sqrt()).max(1e-300)
}

fn main() {
    init_parallelism();
    gam_runtime::test_support::install_diagnostic_logger();

    let n: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(400);
    // Three arms, because the question is no longer "does the follow-up margin
    // work" but "which part of this model shape does the outer solve refuse":
    //
    //   dynamic — the shipped fixture's shape, with the margin
    //   static  — the same shape WITHOUT the margin (the control that showed
    //             the refusal is not the margin's)
    //   smooth  — a marginal formula carrying a penalized `smooth(x)` and the
    //             DEFAULT linear baseline, which is the shape every passing
    //             survival marginal-slope fixture in the tree uses, plus the
    //             margin. If this converges, the refusal belongs to the
    //             intercept-only / learned-Weibull-baseline shape rather than to
    //             the follow-up axis.
    let arm = std::env::args().nth(2).unwrap_or_else(|| "dynamic".to_string());
    let time_k: Option<usize> = if arm == "static" { None } else { Some(4) };
    let smooth_arm = arm == "smooth";

    let (data, times, _scores) = build_dataset(n, smooth_arm);
    let formula = if smooth_arm {
        "Surv(time, event) ~ smooth(x)"
    } else {
        "Surv(time, event) ~ 1"
    };
    let config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        slope_time_k: time_k,
        slope_time_degree: 2,
        time_num_internal_knots: 3,
        baseline_target: if smooth_arm {
            "linear".to_string()
        } else {
            "weibull".to_string()
        },
        ..FitConfig::default()
    };

    eprintln!(
        "[2765-probe] n={n} arm={arm} formula={formula:?} slope_time_k={time_k:?} \
         baseline={}",
        config.baseline_target
    );
    let started = std::time::Instant::now();
    match fit_from_formula(formula, &data, &config) {
        Ok(result) => {
            eprintln!(
                "[2765-probe] fit converged in {:.1}s",
                started.elapsed().as_secs_f64()
            );
            report_recovery(result, &times);
        }
        Err(error) => eprintln!(
            "[2765-probe] fit REFUSED after {:.1}s: {error}",
            started.elapsed().as_secs_f64()
        ),
    }
}

/// The acceptance verdict, stated exactly as
/// `survival_marginal_slope_recovers_a_follow_up_varying_slope_2765` states it:
/// the per-row fitted slope against the planted `b(t)`, reported as the range it
/// spans and its correlation with the truth. Reported rather than asserted,
/// because a probe's job is to produce the number at a size where it can be
/// produced repeatedly.
fn report_recovery(result: gam::FitResult, times: &[f64]) {
    let gam::FitResult::SurvivalMarginalSlope(fit) = result else {
        eprintln!("[2765-probe] not a marginal-slope fit result; no recovery verdict");
        return;
    };
    let beta = &fit.fit.blocks[2].beta;
    let design = fit.slope_design.design.to_dense();
    if design.ncols() != beta.len() || design.nrows() != times.len() {
        eprintln!(
            "[2765-probe] slope design {}x{} against {} coefficients and {} rows",
            design.nrows(),
            design.ncols(),
            beta.len(),
            times.len()
        );
        return;
    }
    let fitted: Vec<f64> = (0..times.len())
        .map(|row| design.row(row).dot(beta) + fit.baseline_slope)
        .collect();
    let truth: Vec<f64> = times
        .iter()
        .map(|t| SLOPE_LEVEL + SLOPE_TREND * t.ln())
        .collect();
    let correlation = pearson(&fitted, &truth);
    let fitted_min = fitted.iter().cloned().fold(f64::INFINITY, f64::min);
    let fitted_max = fitted.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let truth_min = truth.iter().cloned().fold(f64::INFINITY, f64::min);
    let truth_max = truth.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "[2765-probe] cols={} fitted_range=[{fitted_min:.4}, {fitted_max:.4}] \
         truth_range=[{truth_min:.4}, {truth_max:.4}] pearson={correlation:.4} \
         range={:.4} (bar 0.05, pearson bar 0.80) outer_iters={}",
        beta.len(),
        fitted_max - fitted_min,
        fit.fit.outer_iterations,
    );
}
