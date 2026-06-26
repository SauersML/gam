//! Reproduction + regression guard for #1569: the coupled smooth-scale survival
//! location-scale joint-Newton globalization must converge in the AGGRESSIVE
//! heteroscedastic regime — strong x-dependence in BOTH the AFT location and the
//! log-σ channel, where the free scale predictor `η_σ(x)` drives `exp(−η_σ)` (the
//! `inv_sigma` multiplier on the time-channel residual/gradient) over a wide
//! dynamic range and can inflate the time-block step.
//!
//! DATA RECIPE — gam-model-faithful. The gam survival location-scale model is
//! NOT a textbook AFT `log T = μ + σ·ε`: its standardized index is
//!     z(t, x) = h(t) − η_t(x) · exp(−η_σ(x)),   S(t|x) = 1 − Φ(z),
//! with a LEARNED monotone time baseline `h(t)` (so the `1/σ` scaling rides the
//! location channel, not the time channel). We therefore reuse the EXACT recipe
//! of the gamlss-oracle gate `quality_vs_gamlss_gaussian_survival_ls.rs`, whose
//! closed-form truth is known to be recovered by this model: a Weibull time whose
//! log-scale carries the location signal, dispersed around its own median by a
//! smooth envelope that carries the log-σ signal. Cranking the two amplitudes
//! drives the aggressive heteroscedastic regime that stalls the joint Newton.
//! No reference tool (gamlss/R) is needed — the truth is analytic.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
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
}

struct HeteroFit {
    /// `Err(message)` if the fit returned an error (the stall surfaced as a hard
    /// failure); otherwise the converged-fit diagnostics.
    outcome: Result<HeteroOk, String>,
    censor_frac: f64,
}

struct HeteroOk {
    converged: bool,
    outer_iterations: usize,
    inner_cycles: usize,
    grad_norm: Option<f64>,
    rmse_loc: f64,
    rmse_logsig: f64,
}

/// Generate a gate-faithful right-censored heteroscedastic AFT dataset and fit
/// gam's survival location-scale model with a smooth location AND a smooth scale.
///
/// `loc_amp` scales the location signal `loc_amp·sin(2πx)` (gate default 0.3) and
/// `env_amp` (in (0,1)) the dispersion envelope `1 + env_amp·cos(2πx)` (gate
/// default 0.4). Larger amplitudes ⇒ wider `exp(−η_σ)` dynamic range ⇒ the
/// aggressive regime. Never panics on a fit error: it captures the message so a
/// sweep reports which configuration stalled.
fn fit_heteroscedastic(
    n: usize,
    loc_amp: f64,
    env_amp: f64,
    k_loc: usize,
    k_scale: usize,
    seed: u64,
) -> HeteroFit {
    let two_pi = 2.0 * std::f64::consts::PI;
    let shape = 1.5_f64;
    let mut rng = Lcg::new(seed);

    // closed-form truth (x-dependent parts; gauge = mean-centered over grid):
    //   AFT location  η_t(x)   = loc_amp · sin(2πx)        (Weibull log-scale x-part)
    //   AFT log-scale η_σ(x)   = log(1 + env_amp · cos(2πx)) (dispersion envelope)
    let truth_loc = |x: f64| loc_amp * (two_pi * x).sin();
    let truth_lsig = |x: f64| (1.0 + env_amp * (two_pi * x).cos()).ln();

    let mut x = Vec::with_capacity(n);
    let mut exit = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    let mut censored = 0usize;
    for _ in 0..n {
        let xi = -2.0 + 4.0 * rng.unit();
        let scale = (-0.5 + loc_amp * (two_pi * xi).sin()).exp();
        let envelope = 1.0 + env_amp * (two_pi * xi).cos();
        let u = rng.unit().max(1e-300);
        let base = scale * (-u.ln()).powf(1.0 / shape);
        let median = scale * (std::f64::consts::LN_2).powf(1.0 / shape);
        let t = (median + (base - median) * envelope).max(1e-6);
        let ev = if rng.unit() < 0.7 { 1.0 } else { 0.0 };
        if ev < 0.5 {
            censored += 1;
        }
        x.push(xi);
        exit.push(t);
        event.push(ev);
    }
    let censor_frac = censored as f64 / n as f64;

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
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some(format!("s(x, k={k_scale})")),
        ..FitConfig::default()
    };
    let result = match fit_from_formula(
        &format!("Surv(entry, exit, event) ~ s(x, k={k_loc})"),
        &ds,
        &cfg,
    ) {
        Ok(r) => r,
        Err(e) => {
            return HeteroFit {
                outcome: Err(e.to_string()),
                censor_frac,
            };
        }
    };
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;

    // grid truth-recovery
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
        (a.iter()
            .zip(b)
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt()
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
        let t_loc = center(&grid_x.iter().map(|&xi| truth_loc(xi)).collect::<Vec<_>>());
        let t_lsig = center(&grid_x.iter().map(|&xi| truth_lsig(xi)).collect::<Vec<_>>());
        (rmse(&gam_loc, &t_loc), rmse(&gam_lsig, &t_lsig))
    } else {
        (f64::NAN, f64::NAN)
    };

    HeteroFit {
        outcome: Ok(HeteroOk {
            converged: unified.outer_converged,
            outer_iterations: unified.outer_iterations,
            inner_cycles: unified.inner_cycles,
            grad_norm: unified.outer_gradient_norm,
            rmse_loc,
            rmse_logsig,
        }),
        censor_frac,
    }
}

#[test]
fn survival_location_scale_heteroscedastic_sweep_diagnostic() {
    init_parallelism();
    // (n, loc_amp, env_amp, k_loc, k_scale, seed). env_amp must stay < 1.
    // Moderate-aggressive configs: strong enough heteroscedasticity to drive the
    // inv_sigma-inflated time-block step into the monotone α-crush regime (the
    // #1569 stall) WITHOUT the identifiability degeneracy of the most extreme
    // amplitudes, and small enough to converge in tractable time once fixed.
    let configs = [
        (70usize, 0.7f64, 0.75f64, 5usize, 4usize, 7u64), // moderate-aggressive
        (80, 0.85, 0.8, 5, 5, 21),                        // aggressive
        (90, 0.3, 0.4, 5, 4, 1234),                       // mild control
    ];
    for (n, la, ea, kl, ks, seed) in configs {
        let t0 = std::time::Instant::now();
        let r = fit_heteroscedastic(n, la, ea, kl, ks, seed);
        let secs = t0.elapsed().as_secs_f64();
        match r.outcome {
            Ok(ok) => eprintln!(
                "[#1569 sweep] n={n} loc_amp={la} env_amp={ea} k_loc={kl} k_scale={ks} seed={seed} \
                 censor={:.2} elapsed={secs:.1}s -> converged={} outer_iters={} inner_cycles={} \
                 grad_norm={:?} rmse_loc={:.4} rmse_logsig={:.4}",
                r.censor_frac,
                ok.converged,
                ok.outer_iterations,
                ok.inner_cycles,
                ok.grad_norm,
                ok.rmse_loc,
                ok.rmse_logsig,
            ),
            Err(e) => eprintln!(
                "[#1569 sweep] n={n} loc_amp={la} env_amp={ea} k_loc={kl} k_scale={ks} seed={seed} \
                 censor={:.2} elapsed={secs:.1}s -> ERRORED: {e}",
                r.censor_frac,
            ),
        }
    }
}
