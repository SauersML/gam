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
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;

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
    let result = fit_from_formula(
        &format!("Surv(entry, exit, event) ~ s(x, k={k_loc})"),
        &ds,
        &cfg,
    )
    .expect("gam hetero survival location-scale fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;

    let grid_n = 20usize;
    let (x_lo, x_hi) = (-1.9_f64, 1.9_f64);
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| x_lo + (x_hi - x_lo) * i as f64 / (grid_n as f64 - 1.0))
        .collect();
    let mut grid = Array2::<f64>::zeros((grid_n, ncols));
    for (i, &t) in grid_x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let center = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|&z| z - m).collect()
    };
    let rmse = |a: &[f64], b: &[f64]| -> f64 {
        (a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>() / a.len() as f64).sqrt()
    };

    let (rmse_loc, rmse_logsig) = if unified.beta_threshold().iter().all(|v| v.is_finite())
        && unified.beta_log_sigma().iter().all(|v| v.is_finite())
    {
        let loc_design =
            build_term_collection_design(grid.view(), &fit.fit.resolved_thresholdspec).unwrap();
        let ls_design =
            build_term_collection_design(grid.view(), &fit.fit.resolved_log_sigmaspec).unwrap();
        let gam_loc = center(&loc_design.design.apply(&unified.beta_threshold()).to_vec());
        let gam_lsig = center(&ls_design.design.apply(&unified.beta_log_sigma()).to_vec());
        let truth_loc = center(&grid_x.iter().map(|&xi| mu(xi)).collect::<Vec<_>>());
        let truth_lsig = center(&grid_x.iter().map(|&xi| log_sigma(xi)).collect::<Vec<_>>());
        (rmse(&gam_loc, &truth_loc), rmse(&gam_lsig, &truth_lsig))
    } else {
        (f64::NAN, f64::NAN)
    };

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

/// #2695: a payload written before the whole-residual kernel is refused by name
/// when both its log-σ predictor and its time-warp coefficients can move, since it
/// was then fit as a different model, one whose σ had no likelihood of its own. Two
/// payloads of the same version load, one per arm of the rule. The σ ≡ 1 arm is the
/// heteroscedastic payload with its log-σ predictor set identically to zero: its warp
/// still moves, and `u = h − η_t` under both kernels. The `h ≡ 0` arm is the
/// constant-scale fit with no noise formula, which takes the σ-scaled log-t baseline
/// (#892): it saves an all-zero warp with a free log-σ intercept, so the two kernels
/// are the same model on its rows.
#[test]
fn a_pre_2695_heteroscedastic_payload_is_refused_and_a_constant_scale_one_loads_2695() {
    use gam_models::inference::model::{FittedModel, FittedModelPayload, WHOLE_RESIDUAL_SCALE_PAYLOAD_VERSION};
    use gam_models::inference::model_payload_builders::fit_formula_to_payload;
    use gam_problem::BlockRole;

    super::initialize_cpu_fitting();
    let fit = |scale_amp: f64, noise_formula: Option<&str>| {
        let (headers, rows) = heteroscedastic_records(180, scale_amp, 7);
        let data = encode_recordswith_inferred_schema(headers, rows).expect("encode data");
        let cfg = FitConfig {
            survival_likelihood: Some("location-scale".to_string()),
            survival_distribution: "gaussian".to_string(),
            noise_formula: noise_formula.map(str::to_string),
            ..FitConfig::default()
        };
        fit_formula_to_payload("Surv(entry, exit, event) ~ s(x, k=8)".to_string(), &data, &cfg)
            .expect("#2695: survival location-scale fit")
    };
    let block_can_move = |payload: &FittedModelPayload, role: BlockRole| {
        payload
            .fit_result
            .as_ref()
            .and_then(|fit| fit.block_by_role(role))
            .map(|block| block.beta.iter().any(|value| *value != 0.0))
            .expect("#2695: the survival location-scale fit carries this block")
    };
    let stale_version = WHOLE_RESIDUAL_SCALE_PAYLOAD_VERSION - 1;

    let heteroscedastic = fit(1.2, Some("s(x, k=8)"));
    assert!(
        block_can_move(&heteroscedastic, BlockRole::Scale)
            && block_can_move(&heteroscedastic, BlockRole::Time),
        "#2695: the heteroscedastic fit must carry a moving log-σ and a moving warp, or the \
         refusal below is not the kernel-change arm"
    );
    FittedModel::from_payload(heteroscedastic.clone())
        .validate_for_persistence()
        .expect("#2695: the current payload validates");
    assert!(
        heteroscedastic.noise_offset_column.is_none(),
        "#2695: the σ ≡ 1 arm below needs a payload without a declared noise offset"
    );
    let mut unit_scale = heteroscedastic.clone();
    let mut stale = heteroscedastic;
    stale.version = stale_version;
    let error = FittedModel::from_payload(stale)
        .validate_for_persistence()
        .expect_err("#2695: a heteroscedastic payload from before the kernel change is refused");
    assert!(
        error.to_string().contains("pre-#2695 location-only kernel")
            && error
                .to_string()
                .contains("refit required: σ was unidentified under the pre-#2695 likelihood"),
        "#2695: the refusal must name the kernel change and require a refit, got: {error}"
    );

    for saved in [unit_scale.fit_result.as_mut(), unit_scale.unified.as_mut()]
        .into_iter()
        .flatten()
    {
        for block in saved.blocks.iter_mut().filter(|block| block.role == BlockRole::Scale) {
            block.beta.fill(0.0);
        }
    }
    unit_scale.survival_beta_log_sigma = unit_scale
        .survival_beta_log_sigma
        .as_ref()
        .map(|beta| vec![0.0; beta.len()]);
    assert!(
        !block_can_move(&unit_scale, BlockRole::Scale)
            && block_can_move(&unit_scale, BlockRole::Time),
        "#2695: the σ ≡ 1 payload must carry a zero log-σ predictor beside a moving warp, or \
         the load below does not exercise that arm"
    );
    unit_scale.version = stale_version;
    FittedModel::from_payload(unit_scale)
        .validate_for_persistence()
        .expect("#2695: a σ ≡ 1 payload is the same model under both kernels and loads");

    let constant_scale = fit(0.0, None);
    assert!(
        block_can_move(&constant_scale, BlockRole::Scale)
            && !block_can_move(&constant_scale, BlockRole::Time),
        "#2695: the constant-scale fit must carry a free log-σ intercept on an all-zero warp, \
         or the control below does not exercise the h ≡ 0 arm"
    );
    let mut stale = constant_scale;
    stale.version = stale_version;
    FittedModel::from_payload(stale)
        .validate_for_persistence()
        .expect("#2695: an h ≡ 0 payload is the same model under both kernels and loads");
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
        let payload =
            fit_formula_to_payload("Surv(entry, exit, event) ~ s(x, k=8)".to_string(), &data, &cfg)
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
            .unified
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
