//! gam#1561 (docs/convergence_theory/transformation-survival.md Thm 5.1): the
//! Weibull target of the transformation likelihood is a direct summand of the
//! monotone I-spline time block, so its `(log λ, log k)` search had no θ to find.
//!
//! The target enters as the offset `k·(log t − log λ)`. The I-spline block on
//! the `log t` axis reproduces `log t` with positive increments `a` that the
//! curvature Gram annihilates, and the constant is the location intercept, so
//! `γ' = γ + k·a` maps every Weibull(k) fit onto the Linear-target fit with the
//! same likelihood and penalty. Before the direct sum the fit ran a BFGS over θ
//! along that exact flat direction (or toward the `k → 0` boundary), and its
//! answer depended on where the search stopped.

use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::faer_ndarray::FaerEigh;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    SurvivalTimeBasisConfig, build_survival_time_basis, predict_survival,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use gam_linalg::utils::splitmix64;

const N: usize = 400;
const SHAPE: f64 = 1.5;
const LOG_SCALE: f64 = 0.7;
const COVARIATE_EFFECT: f64 = 0.6;
const GRID: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

fn next_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// Weibull proportional-hazards rows `H(t | x) = exp(k (log t − log λ) + β x)`
/// with uniform administrative censoring.
fn cohort() -> (Vec<f64>, Vec<f64>, gam_data::EncodedDataset) {
    let mut state = 0x1561_0000_0005_u64;
    let mut exit = Vec::with_capacity(N);
    let mut xs = Vec::with_capacity(N);
    let mut rows = Vec::with_capacity(N);
    for _ in 0..N {
        let x = -1.0 + 2.0 * next_unit(&mut state);
        let cumulative = -next_unit(&mut state).ln();
        let event_time =
            (LOG_SCALE + (cumulative.ln() - COVARIATE_EFFECT * x) / SHAPE).exp();
        let censor = 0.3 + 5.0 * next_unit(&mut state);
        let time = event_time.min(censor);
        let event = u8::from(event_time <= censor);
        exit.push(time);
        xs.push(x);
        rows.push(StringRecord::from(vec![
            time.to_string(),
            event.to_string(),
            x.to_string(),
        ]));
    }
    let headers = ["time", "event", "x"].iter().map(|s| s.to_string()).collect();
    let data = encode_recordswith_inferred_schema(headers, rows)
        .expect("encode the gam#1561 Weibull cohort");
    (exit, xs, data)
}

/// The certificate the direct sum reads: the I-spline time block's penalty has
/// one null direction, that direction is a positive increment vector, and it
/// carries `log t` on the value rows and `1/t` on the derivative rows.
#[test]
fn weibull_offset_lies_in_the_ispline_block_null_space_1561() {
    let (exit, _, _) = cohort();
    let age_exit = Array1::from_vec(exit);
    let age_entry = Array1::<f64>::zeros(N);
    let build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        SurvivalTimeBasisConfig::ISpline {
            degree: 3,
            knots: Array1::zeros(0),
            keep_cols: Vec::new(),
        },
        Some(8),
    )
    .expect("build the I-spline time block");
    assert_eq!(build.basisname, "ispline");
    assert_eq!(build.nullspace_dims, vec![1], "the curvature Gram keeps one null direction");
    assert_eq!(build.penalties.len(), 1);

    let penalty = &build.penalties[0];
    let (evals, evecs) = penalty.eigh(faer::Side::Lower).expect("penalty spectrum");
    let top = evals.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
    let mut a = evecs.column(0).to_owned();
    if a.sum() < 0.0 {
        a.mapv_inplace(|v| -v);
    }
    let min_increment = a.iter().copied().fold(f64::INFINITY, f64::min);
    let null_residual = penalty.dot(&a).iter().fold(0.0_f64, |m, v| m.max(v.abs()));

    let x_exit: Array2<f64> = build.x_exit_time.to_dense();
    let x_derivative: Array2<f64> = build.x_derivative_time.to_dense();
    let image = x_exit.dot(&a);
    let log_t = age_exit.mapv(f64::ln);
    // `image = α·log t + c` exactly: a two-column least-squares fit.
    let mean_image = image.mean().unwrap();
    let mean_log_t = log_t.mean().unwrap();
    let centred_log_t = &log_t - mean_log_t;
    let alpha = centred_log_t.dot(&(&image - mean_image)) / centred_log_t.dot(&centred_log_t);
    let value_residual = (0..N)
        .map(|i| (image[i] - mean_image - alpha * centred_log_t[i]).abs())
        .fold(0.0_f64, f64::max);
    let derivative_image = x_derivative.dot(&a);
    let derivative_residual = (0..N)
        .map(|i| (derivative_image[i] * age_exit[i] - alpha).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "[#1561] p={} λ_null={:.3e} λ_max={top:.3e} |S a|∞={null_residual:.3e} \
         min a={min_increment:.3e} α={alpha:.6} value residual={value_residual:.3e} \
         t·derivative residual={derivative_residual:.3e}",
        a.len(),
        evals[0]
    );
    assert!(min_increment > 0.0, "the null direction is a positive increment vector: {a}");
    assert!(null_residual <= 1e-10 * top, "S a = {null_residual:.3e} against |S| = {top:.3e}");
    assert!(alpha > 0.0);
    assert!(value_residual <= 1e-10 * alpha, "X a − α log t is not constant: {value_residual:.3e}");
    assert!(
        derivative_residual <= 1e-10 * alpha,
        "t·(X' a) − α is not zero: {derivative_residual:.3e}"
    );
}

fn predict_cumulative_hazard(model: &FittedModel, data: &gam_data::EncodedDataset) -> Array2<f64> {
    let col_map: HashMap<String, usize> = data
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = Array1::<f64>::zeros(data.values.nrows());
    let grid = GRID.to_vec();
    predict_survival(
        SurvivalPredictRequest {
            model,
            data: data.values.view(),
            col_map: &col_map,
            training_headers: Some(&data.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: Some(&grid),
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("predict the fitted cumulative hazard")
    .cumulative_hazard
}

fn fit_target(data: &gam_data::EncodedDataset, target: &str) -> Array2<f64> {
    let config = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        baseline_target: target.to_string(),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("Surv(time, event) ~ x".to_string(), data, &config)
        .unwrap_or_else(|error| panic!("gam#1561 {target}-target fit failed: {error}"));
    predict_cumulative_hazard(&FittedModel::from_payload(payload), data)
}

/// The Weibull-target fit is the Linear-target fit: the same model space, the
/// same criterion, so the same fitted cumulative hazard to roundoff.
#[test]
fn weibull_target_transformation_fit_is_the_linear_target_fit_1561() {
    super::initialize_cpu_fitting();
    let (_, xs, data) = cohort();
    let linear = fit_target(&data, "linear");
    let weibull = fit_target(&data, "weibull");
    let mut worst = 0.0_f64;
    for (&w, &l) in weibull.iter().zip(linear.iter()) {
        assert!(w.is_finite() && l.is_finite() && l > 0.0);
        worst = worst.max((w - l).abs() / l);
    }
    // The planted truth, for scale: the fit recovers a Weibull hazard.
    let truth = |x: f64, t: f64| (SHAPE * (t.ln() - LOG_SCALE) + COVARIATE_EFFECT * x).exp();
    let mut truth_gap = 0.0_f64;
    for (i, &x) in xs.iter().enumerate() {
        for (j, &t) in GRID.iter().enumerate() {
            truth_gap = truth_gap.max((linear[[i, j]] / truth(x, t)).ln().abs());
        }
    }
    eprintln!(
        "[#1561] max |H_weibull − H_linear| / H_linear = {worst:.3e}; \
         max |log(H_linear / H_true)| = {truth_gap:.3}"
    );
    assert!(
        worst <= 1e-12,
        "the Weibull-target fit left the Linear-target optimum: relative gap {worst:.3e}"
    );
}
