//! End-to-end quality: gam's location-scale accelerated-failure-time (AFT)
//! family with a *smooth covariate* must reproduce the lognormal AFT that
//! `survival::survreg(dist = "lognormal")` — the gold-standard parametric-AFT
//! reference in R — recovers on the *identical* synthetic right-censored data.
//!
//! Capability under test: lognormal location-scale AFT with a thin-plate smooth
//! covariate, requested via the survival formula
//!   `Surv(t, event) ~ x + s(z, bs="tp", k=5)`
//! fit through gam's location-scale survival likelihood
//! (`FitConfig{ survival_likelihood: "location-scale", survival_distribution:
//! "gaussian" }`). A Gaussian residual on gam's monotone-time-warp channel IS
//! the lognormal AFT (see `quality_vs_lifelines_lognormal_aft.rs` for the full
//! predictor-assembly derivation): the standardized survival index is
//!   z(t, x) = (h(t) - eta_t(x)) / sigma,   S(t|x) = 1 - Phi(z),
//! with a *location* channel `eta_t(x)` (role `Threshold`, `beta_threshold()`),
//! a constant *log-scale* channel (role `Scale`, `sigma = exp(eta_ls)`,
//! `beta_log_sigma()`), and a learned monotone transform `h(t)` of the time
//! axis. survreg fixes `h(t) = log t` and fits exactly
//!   log T = mu(x, z) + sigma * W,   W ~ Normal(0, 1),
//! by maximum likelihood under right-censoring — the *same* parametric
//! location-scale likelihood. Both engines carry a smooth in `z`: gam via
//! `s(z, bs="tp", k=5)`, survreg via `pspline(z, df = 4)` (R's penalized
//! smoothing spline term, the survreg-native analogue of a thin-plate smooth).
//!
//! Why this is the right reference. `survival::survreg` is THE mature reference
//! for parametric AFT in R; `dist = "lognormal"` targets the exact normal-on-
//! log-time likelihood gam's Gaussian location-scale family targets. The smooth
//! `s(z)` term gives a non-trivial design matrix, so the comparison exercises
//! correct parametric assembly *and* smooth-shape recovery, not just an
//! intercept/slope.
//!
//! The gauge. gam learns `h(t)` flexibly while survreg fixes `log t`, so the two
//! location channels differ by an unknown additive *gauge offset* (the absolute
//! location anchor) — exactly as in the lifelines AFT and gamlss survival-LS
//! quality tests, which re-anchor / mean-center before comparing. We therefore
//! compare the *fitted log-survival at the training points* after re-anchoring
//! gam's location to survreg's (subtracting the mean location offset). The
//! engine-agnostic invariants this measures are (a) the covariate dependence
//! through `x` and the smooth `s(z)` shape (both enter the standardized index)
//! and (b) the constant log-scale parameter `log(sigma)`.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;

/// Standard normal CDF via erfc (matches R `pnorm` / survreg's lognormal CDF).
fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Complementary error function (Numerical-Recipes rational approximation,
/// ~1e-7 absolute — far below any log-survival tolerance asserted here).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587
                                        + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// Lognormal-AFT log-survival `log S(t|x) = log(1 - Phi((log t - mu) / sigma))`.
/// We compare on the log-survival scale: it spreads the comparison across the
/// full hazard range (the spec's quantity of interest) and is what survreg's
/// `psurvreg`-equivalent reconstruction yields from `(mu, sigma)`.
fn lognormal_log_survival(t: f64, mu: f64, sigma: f64) -> f64 {
    // Clamp the survival probability away from 0 so log is finite at the largest
    // observed times; 1e-12 floor is far below any survival mass the bound sees.
    let s = (1.0 - norm_cdf((t.ln() - mu) / sigma)).max(1e-12);
    s.ln()
}

#[test]
fn gam_lognormal_location_scale_aft_smooth_matches_survreg() {
    init_parallelism();

    // ---- synthetic recipe, generated ONCE in Rust and fed IDENTICALLY to both ----
    // n=300, seed=2471. log T = eta_location + eps * sigma, with
    //   eta_location = -0.5 + 0.8*x + s(z),   s(z) = sin(pi*z) (a smooth, non-
    //   linear function of z that a k=5 thin-plate / df=4 pspline can both fit),
    //   eps ~ Normal(0,1), sigma a single constant drawn from InverseGamma(shape=2)
    //   (one scale for the whole dataset, as a constant-scale lognormal requires),
    //   censoring ~40% (right-censored at an independent exponential time).
    // The data is generated only here and the columns are handed verbatim to R, so
    // gam and survreg see byte-identical rows (no cross-engine RNG to reconcile).
    let n = 300usize;
    let mut rng = NumpyMt19937::new(2471);

    // One constant scale sigma drawn from InverseGamma(shape=2, scale=1):
    // sigma^2 = 1 / Gamma(shape=2, rate=1); Gamma(2,1) = -ln(u1) - ln(u2). With
    // shape=2 the mean of Gamma is 2 so sigma ~= 0.7, a realistic AFT scale.
    let g2 = -rng.next_f64().ln() - rng.next_f64().ln();
    let sigma_true = (1.0 / g2).sqrt();

    // Draw covariates and noise in a fixed order.
    let x: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let z: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let eps: Vec<f64> = (0..n).map(|_| rng.next_standard_normal()).collect();
    // Independent exponential censoring time; rate chosen for ~40% censoring.
    let cens_u: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();

    let mut t = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    let mut n_censored = 0usize;
    for i in 0..n {
        let s_z = (std::f64::consts::PI * z[i]).sin();
        let eta_loc = -0.5 + 0.8 * x[i] + s_z;
        let t_event = (eta_loc + eps[i] * sigma_true).exp();
        // Independent exponential censoring time, c ~ Exponential(rate = 1/2.4);
        // the multiplier 2.4 is scale-matched to median(T) ~ exp(-0.5) so that
        // P(T > c) ~ 0.40, giving the target ~40% right-censoring.
        let c = -cens_u[i].ln() * 2.4;
        if t_event <= c {
            t.push(t_event);
            event.push(1.0);
        } else {
            t.push(c.max(1e-6));
            event.push(0.0);
            n_censored += 1;
        }
    }
    let cens_frac = n_censored as f64 / n as f64;
    assert!(
        (0.25..=0.55).contains(&cens_frac),
        "expected ~40% censoring, got {cens_frac:.3} (n_censored={n_censored})"
    );

    // ---- build the gam dataset (columns: t, event, x, z) -------------------
    let headers: Vec<String> = ["t", "event", "x", "z"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", t[i]),
                format!("{:.17e}", event[i]),
                format!("{:.17e}", x[i]),
                format!("{:.17e}", z[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode lognormal-LS data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];
    let ncols = ds.headers.len();

    // ---- fit with gam: lognormal location-scale AFT with s(z, bs="tp", k=5) -
    // Gaussian-residual survival location-scale == lognormal AFT (module doc).
    // No noise_formula => a single constant log-scale (sigma) channel, matching
    // survreg's constant `sr$scale`.
    let cfg = FitConfig {
        survival_likelihood: "location-scale".to_string(),
        survival_distribution: "gaussian".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        r#"Surv(t, event) ~ x + s(z, bs="tp", k=5)"#,
        &ds,
        &cfg,
    )
    .expect("gam lognormal location-scale AFT fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;
    assert!(
        unified.outer_converged,
        "gam lognormal-LS outer optimizer did not converge: iters={} grad_norm={:?}",
        unified.outer_iterations, unified.outer_gradient_norm
    );

    let beta_location = unified.beta_threshold();
    let beta_log_sigma = unified.beta_log_sigma();
    assert!(
        beta_location
            .iter()
            .chain(beta_log_sigma.iter())
            .all(|v| v.is_finite()),
        "non-finite gam location / log-sigma coefficients"
    );

    // gam location predictor mu_gam(x_i, z_i) at the training points: rebuild the
    // frozen location (threshold) design from `resolved_thresholdspec` at the
    // observed (x, z) and apply the converged location coefficients. This is the
    // AFT location on gam's learned-time-transform gauge. Same rebuild pattern as
    // the mgcv / lifelines / gamlss quality tests.
    let mut train_grid = Array2::<f64>::zeros((n, ncols));
    for i in 0..n {
        train_grid[[i, x_idx]] = x[i];
        train_grid[[i, z_idx]] = z[i];
    }
    let loc_design = build_term_collection_design(train_grid.view(), &fit.fit.resolved_thresholdspec)
        .expect("rebuild location (threshold) design at training points");
    let gam_mu_train: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();
    assert_eq!(gam_mu_train.len(), n);

    // Constant log-scale channel: sigma = exp(eta_ls) at any (x, z) (intercept-only).
    let ls_design = build_term_collection_design(train_grid.view(), &fit.fit.resolved_log_sigmaspec)
        .expect("rebuild log-sigma design at training points");
    let gam_eta_ls: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    assert!(
        gam_eta_ls.iter().all(|&v| (v - gam_eta_ls[0]).abs() < 1e-9),
        "expected a constant (covariate-independent) log-scale channel, got {gam_eta_ls:?}"
    );
    let gam_log_sigma = gam_eta_ls[0];
    let gam_sigma = gam_log_sigma.exp();

    // ---- fit the SAME data with survreg(dist="lognormal") (mature reference) -
    // The columns are passed verbatim, so survreg fits the byte-identical rows.
    // survreg's linear predictor `predict(type = "lp")` is mu(x_i, z_i) on the
    // natural-log time scale (location), and `sr$scale` is the constant sigma.
    let r = run_r(
        &[
            Column::new("t", &t),
            Column::new("event", &event),
            Column::new("x", &x),
            Column::new("z", &z),
        ],
        r#"
        suppressPackageStartupMessages(library(survival))
        sr <- survreg(Surv(t, event) ~ x + pspline(z, df = 4),
                      data = df, dist = "lognormal")
        # Location predictor mu(x_i, z_i) at the training points (log-time scale).
        emit("mu", as.numeric(predict(sr, type = "lp")))
        # Constant scale sigma and its log (the AFT log-scale coefficient).
        emit("sigma", as.numeric(sr$scale))
        emit("log_sigma", log(as.numeric(sr$scale)))
        "#,
    );
    let ref_mu = r.vector("mu");
    let ref_sigma = r.scalar("sigma");
    let ref_log_sigma = r.scalar("log_sigma");
    assert_eq!(ref_mu.len(), n, "survreg lp length mismatch");

    // ---- re-anchor gam's location to survreg's (remove the time gauge) ------
    // gam learns h(t) while survreg fixes log t, so the absolute location anchor
    // lives on different gauges; subtract the mean location offset (the gauge
    // constant) exactly as the gamlss survival-LS quality test mean-centers its
    // surfaces. The covariate / smooth dependence is gauge-free and survives.
    let gam_mean = gam_mu_train.iter().sum::<f64>() / n as f64;
    let ref_mean = ref_mu.iter().sum::<f64>() / n as f64;
    let offset = gam_mean - ref_mean;
    let gam_mu_anchored: Vec<f64> = gam_mu_train.iter().map(|&m| m - offset).collect();

    // ---- compare fitted log-survival at the training points -----------------
    // log S(t_i | x_i, z_i) reconstructed from each engine's (mu, sigma) with the
    // identical lognormal closed form, evaluated at the observed t_i.
    let gam_log_surv: Vec<f64> = (0..n)
        .map(|i| lognormal_log_survival(t[i], gam_mu_anchored[i], gam_sigma))
        .collect();
    let ref_log_surv: Vec<f64> = (0..n)
        .map(|i| lognormal_log_survival(t[i], ref_mu[i], ref_sigma))
        .collect();

    let log_surv_corr = pearson(&gam_log_surv, &ref_log_surv);
    // log-scale coefficient = log(sigma); a single scalar so rmse == |diff|.
    let log_scale_rmse = rmse(&[gam_log_sigma], &[ref_log_sigma]);

    eprintln!(
        "lognormal location-scale AFT vs survreg: n={n} cens={cens_frac:.3} \
         sigma_true={sigma_true:.4} gam_sigma={gam_sigma:.4} ref_sigma={ref_sigma:.4} \
         gam_log_sigma={gam_log_sigma:.4} ref_log_sigma={ref_log_sigma:.4} \
         log_scale_rmse={log_scale_rmse:.4} gauge_offset={offset:.4} \
         logS_pearson={log_surv_corr:.5}"
    );

    // Bounds (spec-derived, principled):
    //  * fitted log-survival Pearson >= 0.998: both engines fit the SAME
    //    parametric lognormal location-scale likelihood (gam via PIRLS on the
    //    exact Newton Hessian, survreg via numerical MLE); once the time gauge is
    //    re-anchored the reconstructed log-survival across all 300 training points
    //    must essentially coincide. This is tight — only the k=5 thin-plate vs
    //    df=4 pspline smoothing-basis difference and optimizer noise separate them.
    assert!(
        log_surv_corr >= 0.998,
        "fitted log-survival diverges from survreg: pearson={log_surv_corr:.5}"
    );
    //  * log-scale coefficient rmse <= 0.03: log(sigma) is the second AFT
    //    parameter and is gauge-invariant (the local slope of h vs log t is ~unit).
    //    Both engines solve the identical objective for it, so they must agree to
    //    numerical-integration vs exact-MLE tolerance.
    assert!(
        log_scale_rmse <= 0.03,
        "AFT log-scale coefficient diverges from survreg: rmse={log_scale_rmse:.4} \
         (gam_log_sigma={gam_log_sigma:.4}, ref_log_sigma={ref_log_sigma:.4})"
    );
}

/// Minimal NumPy-compatible MT19937 (`np.random.seed` / `random_sample` /
/// `standard_normal`) used here purely as a deterministic, well-tested PRNG so
/// the single synthetic dataset is byte-reproducible across runs. (The data is
/// generated only in Rust and the columns are passed verbatim to R, so no
/// cross-engine RNG reconciliation is required.)
struct NumpyMt19937 {
    mt: [u32; 624],
    idx: usize,
    has_gauss: bool,
    gauss: f64,
}

impl NumpyMt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            mt,
            idx: 624,
            has_gauss: false,
            gauss: 0.0,
        }
    }

    fn generate(&mut self) {
        const MATRIX_A: u32 = 0x9908b0df;
        const UPPER: u32 = 0x80000000;
        const LOWER: u32 = 0x7fffffff;
        for i in 0..624 {
            let y = (self.mt[i] & UPPER) | (self.mt[(i + 1) % 624] & LOWER);
            let mut next = self.mt[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.mt[i] = next;
        }
        self.idx = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            self.generate();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// NumPy `random_sample`: 53-bit double in [0, 1).
    fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64; // 27 bits
        let b = (self.next_u32() >> 6) as u64; // 26 bits
        (a as f64 * 67108864.0 + b as f64) / 9007199254740992.0
    }

    /// NumPy legacy `standard_normal`: polar Marsaglia, matching `RandomState`.
    fn next_standard_normal(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            return self.gauss;
        }
        loop {
            let x1 = 2.0 * self.next_f64() - 1.0;
            let x2 = 2.0 * self.next_f64() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = f * x1;
                self.has_gauss = true;
                return f * x2;
            }
        }
    }
}
