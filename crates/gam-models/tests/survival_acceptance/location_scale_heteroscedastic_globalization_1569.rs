//! Reproduction + regression guard for #1569: the coupled smooth-scale survival
//! location-scale joint-Newton globalization must converge in the AGGRESSIVE
//! heteroscedastic regime — strong x-dependence in BOTH the AFT location and the
//! log-σ channel, where the free scale predictor `η_σ(x)` drives `exp(−η_σ)`
//! (the `inv_sigma` multiplier on the time-channel residual/gradient) over a wide
//! dynamic range and can inflate the time-block step.
//!
//! The data is a clean Gaussian AFT on log-time:  log T = μ(x) + σ(x)·ε.
//! Truth is known analytically in the mean-centered gauge; no reference tool needed.

use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::matrix::LinearOperator;
use gam_math::probability::standard_normal_quantile;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;
use std::collections::HashMap;

/// Numerical-Recipes 64-bit LCG → deterministic uniforms in [0,1).
struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Box–Muller standard normal.
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

struct HeteroFit {
    // No `converged` flag: a minted fit is the sealed convergence proof (SPEC 20).
    outer_iterations: usize,
    inner_cycles: usize,
    grad_norm: Option<f64>,
    rmse_loc: f64,
    rmse_logsig: f64,
    censor_frac: f64,
}

fn fit_heteroscedastic(
    n: usize,
    loc_amp: f64,
    scale_amp: f64,
    k_loc: usize,
    k_scale: usize,
    seed: u64,
) -> HeteroFit {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let mut rng = Lcg::new(seed);

    //   location  μ(x)   = loc_amp   * sin(πx/2)
    //   log-scale η_σ(x) = scale_amp * cos(πx/2)
    // One cycle over the covariate range x ∈ [−2, 2], which a k = 8 smooth resolves.
    // At period 1 the range held four cycles, so every certified fit shrank both
    // smooths to flat and scored exactly amplitude/√2, whatever the solver did.
    let mu = |x: f64| loc_amp * (half_pi * x).sin();
    let log_sigma = |x: f64| scale_amp * (half_pi * x).cos();

    let mut x = Vec::with_capacity(n);
    let mut exit = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    let mut censored = 0usize;
    for _ in 0..n {
        let xi = -2.0 + 4.0 * rng.unit();
        let log_t = mu(xi) + log_sigma(xi).exp() * rng.normal();
        let t = log_t.exp().max(1e-6);
        let c = (0.4 + 3.6 * rng.unit()).exp();
        let (obs, ev) = if t <= c { (t, 1.0) } else { (c, 0.0) };
        if ev < 0.5 {
            censored += 1;
        }
        x.push(xi);
        exit.push(obs);
        event.push(ev);
    }

    let headers: Vec<String> = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                "0".to_string(),
                format!("{:.17e}", exit[i]),
                format!("{:.17e}", event[i]),
                format!("{:.17e}", x[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode hetero data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some(format!("s(x, k={k_scale})")),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload(
        format!("Surv(entry, exit, event) ~ s(x, k={k_loc})"),
        &ds,
        &cfg,
    )
    .expect("gam hetero survival location-scale fit");
    let unified = payload
        .fit_result
        .clone()
        .expect("a survival location-scale payload carries its fit result");
    let thresholdspec = payload
        .resolved_termspec
        .clone()
        .expect("a survival location-scale payload carries its location spec");
    let log_sigmaspec = payload
        .resolved_termspec_noise
        .clone()
        .expect("a survival location-scale payload carries its log-scale spec");
    let model = FittedModel::from_payload(payload);

    let center = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|&z| z - m).collect()
    };
    let rmse = |a: &[f64], b: &[f64]| -> f64 {
        (a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>() / a.len() as f64).sqrt()
    };

    // The standardized index `u = (h(t) − η_t(x))/σ(x)` is invariant under
    // `(h, η_t, σ) → (c·h + b, c·η_t + b, c·σ)`, so only gauge-invariant
    // quantities are scored against the log-time truth. The fitted warp is
    // recovered at every training row from the model's own survival,
    // `h_i = η_t(x_i) + σ(x_i)·Φ⁻¹(1 − S(t_i | x_i))`, and its least-squares slope
    // `ā` on `log t_i` converts the location to log-time units: `μ̃ = η_t/ā` and
    // `log σ̃ = η_σ − log ā`. Centering on the grid removes the additive `b`.
    let col_map: HashMap<String, usize> = ds
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    let zeros = ndarray::Array1::<f64>::zeros(n);
    let predicted = predict_survival(
        SurvivalPredictRequest {
            model: &model,
            data: ds.values.view(),
            col_map: &col_map,
            training_headers: Some(&ds.headers),
            primary_offset: &zeros,
            noise_offset: &zeros,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        },
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("gam hetero survival location-scale prediction at the training rows");
    let train_loc = build_term_collection_design(ds.values.view(), &thresholdspec)
        .unwrap()
        .design
        .apply(&unified.beta_threshold())
        .to_vec();
    let train_lsig = build_term_collection_design(ds.values.view(), &log_sigmaspec)
        .unwrap()
        .design
        .apply(&unified.beta_log_sigma())
        .to_vec();
    let warp: Vec<f64> = (0..n)
        .map(|row| {
            let survival = predicted.survival[[row, 0]];
            let cumulative_hazard = predicted.cumulative_hazard[[row, 0]];
            // `Φ⁻¹(1 − S) = −Φ⁻¹(S)`; the failure mass `1 − S = −expm1(−H)` keeps
            // full precision where `S` rounds to one.
            let standardized = if survival < 0.5 {
                -standard_normal_quantile(survival).expect("survival in (0, 1)")
            } else {
                standard_normal_quantile(-(-cumulative_hazard).exp_m1())
                    .expect("failure mass in (0, 1)")
            };
            train_loc[row] + train_lsig[row].exp() * standardized
        })
        .collect();
    let log_exit: Vec<f64> = exit.iter().map(|t| t.ln()).collect();
    let log_exit_centered = center(&log_exit);
    let warp_centered = center(&warp);
    let warp_slope = warp_centered
        .iter()
        .zip(&log_exit_centered)
        .map(|(h, l)| h * l)
        .sum::<f64>()
        / log_exit_centered.iter().map(|l| l * l).sum::<f64>();
    assert!(
        warp_slope.is_finite() && warp_slope > 0.0,
        "the fitted warp must increase with log time: slope {warp_slope}"
    );

    let grid_n = 20usize;
    let (x_lo, x_hi) = (-1.9_f64, 1.9_f64);
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| x_lo + (x_hi - x_lo) * i as f64 / (grid_n as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }

    let (rmse_loc, rmse_logsig) = if unified.beta_threshold().iter().all(|v| v.is_finite())
        && unified.beta_log_sigma().iter().all(|v| v.is_finite())
    {
        let loc_design = build_term_collection_design(grid.view(), &thresholdspec).unwrap();
        let ls_design = build_term_collection_design(grid.view(), &log_sigmaspec).unwrap();
        let gam_loc = center(
            &loc_design
                .design
                .apply(&unified.beta_threshold())
                .iter()
                .map(|eta| eta / warp_slope)
                .collect::<Vec<_>>(),
        );
        let gam_lsig = center(
            &ls_design
                .design
                .apply(&unified.beta_log_sigma())
                .iter()
                .map(|eta| eta - warp_slope.ln())
                .collect::<Vec<_>>(),
        );
        let truth_loc = center(&grid_x.iter().map(|&xi| mu(xi)).collect::<Vec<_>>());
        let truth_lsig = center(&grid_x.iter().map(|&xi| log_sigma(xi)).collect::<Vec<_>>());
        (rmse(&gam_loc, &truth_loc), rmse(&gam_lsig, &truth_lsig))
    } else {
        (f64::NAN, f64::NAN)
    };
    eprintln!(
        "#1569 gauge: warp slope on log t ā={warp_slope:.6e}, rmse_loc={rmse_loc:.4}, rmse_logsig={rmse_logsig:.4}"
    );

    HeteroFit {
        outer_iterations: unified.outer_iterations,
        inner_cycles: unified.inner_cycles,
        grad_norm: unified.outer_gradient_norm,
        rmse_loc,
        rmse_logsig,
        censor_frac: censored as f64 / n as f64,
    }
}

/// Asserting regression guard for #1569.
#[test]
fn survival_location_scale_heteroscedastic_globalization_converges_1569() {
    super::initialize_cpu_fitting();
    gam_runtime::test_support::install_diagnostic_logger();
    let r = fit_heteroscedastic(180, 1.0, 1.2, 8, 8, 7);
    // Convergence is certified by construction: fit_heteroscedastic returning
    // a HeteroFit means the sealed fit minted (SPEC 20).
    assert!(
        r.rmse_loc.is_finite() && r.rmse_logsig.is_finite(),
        "#1569: non-finite truth-recovery RMSE (loc={}, logsig={})",
        r.rmse_loc,
        r.rmse_logsig,
    );
    // Provisional truth-recovery bars (re-measure + tighten when buildable).
    assert!(
        r.rmse_loc <= 0.20,
        "#1569: AFT location recovery too coarse: rmse_loc={:.4} (outer_iterations={}, \
         inner_cycles={}, outer_gradient_norm={:?}, censored fraction={:.3})",
        r.rmse_loc,
        r.outer_iterations,
        r.inner_cycles,
        r.grad_norm,
        r.censor_frac,
    );
    assert!(
        r.rmse_logsig <= 0.40,
        "#1569: log-σ recovery too coarse: rmse_logsig={:.4}",
        r.rmse_logsig
    );
}

/// The 1569 truth's rows (`log T = sin(πx/2) + e^{scale_amp·cos(πx/2)}·ε`, independent
/// censoring) as CSV records under the `entry, exit, event, x` headers. The 1569
/// fixture has `scale_amp = 1.2`.
fn heteroscedastic_records(
    n: usize,
    scale_amp: f64,
    seed: u64,
) -> (Vec<String>, Vec<csv::StringRecord>) {
    let half_pi = std::f64::consts::FRAC_PI_2;
    let mut rng = Lcg::new(seed);
    let rows = (0..n)
        .map(|_| {
            let xi = -2.0 + 4.0 * rng.unit();
            let log_t =
                (half_pi * xi).sin() + (scale_amp * (half_pi * xi).cos()).exp() * rng.normal();
            let t = log_t.exp().max(1e-6);
            let c = (0.4 + 3.6 * rng.unit()).exp();
            let (obs, ev) = if t <= c { (t, 1.0) } else { (c, 0.0) };
            csv::StringRecord::from(vec![
                "0".to_string(),
                format!("{obs:.17e}"),
                format!("{ev:.17e}"),
                format!("{xi:.17e}"),
            ])
        })
        .collect();
    let headers = ["entry", "exit", "event", "x"]
        .into_iter()
        .map(str::to_string)
        .collect();
    (headers, rows)
}

/// #2695: a saved survival location-scale model, predicting every training row at its
/// own exit time, gives back the log-likelihood its fit reports: over the plug-in
/// surfaces, `Σ d·ln λ(t) + ln S(t)` equals `UnifiedFitResult::log_likelihood`. Before
/// #2695 the family conditioned every row entering at the origin on `S(entry)`, a factor
/// that base.rs's origin contract and the prediction route both omit. At main 1a092c7443
/// these three fits predicted 215.3, 182.8 and 155.3 nats below their own fit (lane probe
/// job 1279302), and with the whole-residual kernel and the origin rule they agree to four
/// decimals (1280044).
///
/// Only the monotone-warp route is covered. The reduced parametric-AFT route (constant-scale
/// data, no noise formula) predicts 2177 nats below its fit, both on main (1279302) and with
/// this change (1280044). It is #2695's open slice and joins this test once it is fixed.
///
/// Both sides evaluate the same fitted predictors, by different straight-line programs, so
/// they may differ only by rounding. Each sums `n` rows, giving `γ_n` of the terms' absolute
/// mass `Σ |ln S| + d·|ln λ|`. Each forms one row from its predictors, with its own rounding
/// relative to that row's magnitude, counted as at least one nat. That per-row rounding is
/// an allowance of `ROW_OPERATIONS` rounded operations on each side, libm calls included,
/// not a count.
#[test]
fn a_saved_model_predicts_the_log_likelihood_it_was_fit_at_2695() {
    use gam_math::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
    use gam_models::inference::model::FittedModel;
    use gam_models::inference::model_payload_builders::fit_formula_to_payload;
    use gam_models::survival::{
        SurvivalLocationScaleTimeParameterization, SurvivalPredictEstimand, SurvivalPredictRequest,
        SurvivalPredictionCovarianceMode, predict_survival,
    };
    use std::collections::HashMap;

    const ROW_OPERATIONS: usize = 64;
    super::initialize_cpu_fitting();
    let n = 180usize;
    for (label, scale_amp, seed) in [
        ("heteroscedastic, seed 7", 1.2, 7u64),
        ("heteroscedastic, seed 3", 1.2, 3),
        ("constant scale with a noise formula, seed 7", 0.0, 7),
    ] {
        let (headers, rows) = heteroscedastic_records(n, scale_amp, seed);
        let events: Vec<f64> = rows
            .iter()
            .map(|row| row[2].parse::<f64>().expect("event indicator"))
            .collect();
        let data = encode_recordswith_inferred_schema(headers, rows).expect("encode data");
        let cfg = FitConfig {
            survival_likelihood: Some("location-scale".to_string()),
            survival_distribution: "gaussian".to_string(),
            noise_formula: Some("s(x, k=8)".to_string()),
            ..FitConfig::default()
        };
        let payload = fit_formula_to_payload(
            "Surv(entry, exit, event) ~ s(x, k=8)".to_string(),
            &data,
            &cfg,
        )
        .unwrap_or_else(|error| panic!("#2695 {label}: survival location-scale fit: {error}"));
        assert!(
            payload
                .survival_location_scale_structure
                .as_ref()
                .is_some_and(|structure| matches!(
                    structure.time_parameterization,
                    SurvivalLocationScaleTimeParameterization::MonotoneWarp
                )),
            "#2695 {label}: the fit must take the monotone-warp route this test covers"
        );
        let fitted = payload
            .fit_result
            .as_ref()
            .map(|unified| unified.log_likelihood)
            .expect("#2695: the payload carries its fit's log-likelihood");
        let model = FittedModel::from_payload(payload);
        let col_map: HashMap<String, usize> = data
            .headers
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        let zeros = ndarray::Array1::<f64>::zeros(n);
        let predicted = predict_survival(
            SurvivalPredictRequest {
                model: &model,
                data: data.values.view(),
                col_map: &col_map,
                training_headers: Some(&data.headers),
                primary_offset: &zeros,
                noise_offset: &zeros,
                time_grid: None,
                with_uncertainty: false,
                estimand: SurvivalPredictEstimand::Plugin,
            },
            SurvivalPredictionCovarianceMode::Conditional,
        )
        .unwrap_or_else(|error| panic!("#2695 {label}: survival prediction: {error}"));
        let (mut reproduced, mut mass) = (0.0_f64, 0.0_f64);
        for (row, &event) in events.iter().enumerate() {
            let log_survival = predicted.survival[[row, 0]].ln();
            let log_hazard = predicted.hazard[[row, 0]].ln();
            reproduced += event * log_hazard + log_survival;
            mass += log_survival.abs() + event * log_hazard.abs();
        }
        let gap = reproduced - fitted;
        let bound = accumulation_growth(2 * n + 2 * ROW_OPERATIONS) * (mass + n as f64);
        eprintln!(
            "#2695 {label}: fit log-likelihood {fitted:.17e}, reproduced {reproduced:.17e}, gap {gap:.3e} \
             = {:.1} u·(mass + n), bound {bound:.3e}",
            gap.abs() / (UNIT_ROUNDOFF * (mass + n as f64))
        );
        assert!(
            gap.abs() <= bound,
            "#2695 {label}: the saved model's predictions do not reproduce its fit's log-likelihood: \
             fit {fitted:.17e}, reproduced {reproduced:.17e}, gap {gap:.3e} over the rounding bound {bound:.3e}"
        );
    }
}
