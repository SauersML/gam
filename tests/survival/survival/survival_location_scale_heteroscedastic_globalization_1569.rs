//! Reproduction + regression guard for #1569: the coupled smooth-scale survival
//! location-scale joint-Newton globalization must converge in the AGGRESSIVE
//! heteroscedastic regime — strong x-dependence in BOTH the AFT location and the
//! log-σ channel, where the free scale predictor `η_σ(x)` drives `exp(−η_σ)`
//! (the `inv_sigma` multiplier on the time-channel residual/gradient) over a wide
//! dynamic range and can inflate the time-block step.
//!
//! The data is a clean Gaussian AFT on log-time:  log T = μ(x) + σ(x)·ε.
//! Truth is known analytically in the mean-centered gauge; no reference tool needed.

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
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut rng = Lcg::new(seed);

    //   location  μ(x)   = loc_amp   * sin(2πx)
    //   log-scale η_σ(x) = scale_amp * cos(2πx)
    let mu = |x: f64| loc_amp * (two_pi * x).sin();
    let log_sigma = |x: f64| scale_amp * (two_pi * x).cos();

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

#[test]
fn survival_location_scale_heteroscedastic_sweep_diagnostic() {
    init_parallelism();
    let configs = [
        (200usize, 0.3f64, 0.4f64, 6usize, 4usize, 1234u64), // mild control
        (180, 0.8, 1.0, 8, 6, 1234),                         // moderate
        (180, 1.0, 1.2, 8, 8, 7),                            // aggressive
        (160, 1.2, 1.5, 10, 8, 42),                          // very aggressive
    ];
    for (n, la, sa, kl, ks, seed) in configs {
        let t0 = std::time::Instant::now();
        let r = fit_heteroscedastic(n, la, sa, kl, ks, seed);
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "[#1569 sweep] n={n} loc_amp={la} scale_amp={sa} k_loc={kl} k_scale={ks} seed={seed} \
             censor={:.2} elapsed={secs:.1}s -> converged=certified outer_iters={} inner_cycles={} \
             grad_norm={:?} rmse_loc={:.4} rmse_logsig={:.4}",
            r.censor_frac,
            r.outer_iterations,
            r.inner_cycles,
            r.grad_norm,
            r.rmse_loc,
            r.rmse_logsig,
        );
    }
}

/// Asserting regression guard for #1569.
#[test]
fn survival_location_scale_heteroscedastic_globalization_converges_1569() {
    init_parallelism();
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
        "#1569: AFT location recovery too coarse: rmse_loc={:.4}",
        r.rmse_loc
    );
    assert!(
        r.rmse_logsig <= 0.40,
        "#1569: log-σ recovery too coarse: rmse_logsig={:.4}",
        r.rmse_logsig
    );
}
