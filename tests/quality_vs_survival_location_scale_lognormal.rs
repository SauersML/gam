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
//! The gauge. gam learns `h(t)` flexibly (a penalized monotone B-spline on
//! `log t`, see `families::survival_location_scale`) while survreg fixes
//! `h(t) = log t`. Because the location channel is *linear in its coefficients*,
//! the only engine-disagreement that the location predictor `mu(x, z)` can carry
//! that is NOT a genuine modelling difference is a single additive *gauge
//! offset* (the absolute location anchor on the warped clock) — exactly as in
//! the lifelines AFT and gamlss survival-LS quality tests, which mean-center
//! before comparing. We therefore compare the *fitted location predictor*
//! `mu(x_i, z_i)` at the training points element-wise, after mean-centering each
//! engine's predictor (removing the additive gauge constant). This is strictly
//! more faithful than reconstructing a lognormal survival curve with raw
//! `log t`, which would silently re-inject the fixed-`log t` gauge and conflate
//! gam's learned `h` with the comparison. The gauge-free invariants this
//! measures are (a) the covariate slope in `x` and (b) the smooth `s(z)` shape,
//! both of which enter `mu` additively and survive mean-centering. Separately we
//! compare the constant log-scale parameter `log(sigma)`.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

#[test]
fn gam_lognormal_location_scale_aft_smooth_matches_survreg() {
    init_parallelism();

    // ---- synthetic recipe, generated ONCE in Rust and fed IDENTICALLY to both ----
    // n=300, seed=2471. log T = eta_location + eps * sigma, with
    //   eta_location = -0.5 + 0.8*x + s(z),   s(z) = sin(pi*z) (a smooth, non-
    //   linear function of z that a k=5 thin-plate / df=4 pspline can both fit),
    //   eps ~ Normal(0,1), sigma a single constant drawn from InverseGamma(shape=2)
    //   (one scale for the whole dataset, as a constant-scale lognormal requires),
    //   censoring ~40% (right-censored at an independent exponential time with
    //   mean ~0.9, on the same scale as the event-time median).
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
        // Independent exponential censoring time c ~ Exponential(mean = 0.9):
        // T is lognormal with median ~exp(-0.5) but a heavy right tail, so a
        // censoring mean comparable to (not far above) the event median is what
        // delivers the target ~40% right-censoring. A mean of 2.4 censors far too
        // little (~22%, because it dwarfs typical event times); 0.9 puts the
        // censoring clock on the same scale as T and lands censoring in band.
        let c = -cens_u[i].ln() * 0.9;
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
    let result = fit_from_formula(r#"Surv(t, event) ~ x + s(z, bs="tp", k=5)"#, &ds, &cfg)
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
    let loc_design =
        build_term_collection_design(train_grid.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild location (threshold) design at training points");
    let gam_mu_train: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();
    assert_eq!(gam_mu_train.len(), n);

    // Constant log-scale channel: sigma = exp(eta_ls) at any (x, z) (intercept-only).
    let ls_design =
        build_term_collection_design(train_grid.view(), &fit.fit.resolved_log_sigmaspec)
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

    // ---- compare the fitted location predictor mu(x_i, z_i) element-wise ----
    // gam learns h(t) while survreg fixes log t, so the two location channels
    // sit on different time gauges; since mu is linear in its coefficients the
    // entire engine-disagreement that is NOT a real modelling difference is one
    // additive gauge constant. We mean-center each engine's location predictor
    // (removing that constant) and compare the centered predictors at the
    // identical training rows, in identical order — exactly as the gamlss
    // survival-LS quality test mean-centers its surfaces. What remains is the
    // gauge-free covariate slope in x plus the s(z) smooth shape, which is what
    // this test is designed to validate (correct parametric assembly AND smooth
    // recovery), with no fixed-log-t reconstruction to mask a warped h.
    let gam_mean = gam_mu_train.iter().sum::<f64>() / n as f64;
    let ref_mean = ref_mu.iter().sum::<f64>() / n as f64;
    let gam_mu_c: Vec<f64> = gam_mu_train.iter().map(|&m| m - gam_mean).collect();
    let ref_mu_c: Vec<f64> = ref_mu.iter().map(|&m| m - ref_mean).collect();

    let mu_corr = pearson(&gam_mu_c, &ref_mu_c);
    let mu_rel = relative_l2(&gam_mu_c, &ref_mu_c);
    // log-scale coefficient = log(sigma); compared as a scalar absolute diff.
    let log_scale_abs = (gam_log_sigma - ref_log_sigma).abs();

    eprintln!(
        "lognormal location-scale AFT vs survreg: n={n} cens={cens_frac:.3} \
         sigma_true={sigma_true:.4} gam_sigma={gam_sigma:.4} ref_sigma={ref_sigma:.4} \
         gam_log_sigma={gam_log_sigma:.4} ref_log_sigma={ref_log_sigma:.4} \
         log_scale_abs={log_scale_abs:.4} \
         mu_pearson={mu_corr:.5} mu_rel_l2={mu_rel:.4}"
    );

    // Bounds (spec-derived, principled):
    //  * centered location Pearson >= 0.99: both engines fit the SAME parametric
    //    lognormal location-scale likelihood (gam via PIRLS on the exact Newton
    //    Hessian, survreg via numerical MLE); once the additive gauge constant is
    //    removed the centered location predictor across all 300 training points
    //    is the same covariate-plus-smooth surface. It is not bit-identical — the
    //    k=5 thin-plate (gam) vs df=4 pspline (survreg) smoothing bases differ in
    //    null space and penalty, and gam's learned h(t) introduces a mild
    //    non-affine warp away from log t — so 0.99 is the tight-but-honest margin
    //    a real assembly or smooth-recovery bug would break.
    assert!(
        mu_corr >= 0.99,
        "fitted location predictor diverges from survreg: pearson={mu_corr:.5} rel_l2={mu_rel:.4}"
    );
    //  * centered location rel_l2 <= 0.15: same surface on a shared grid; the
    //    relative-L2 budget covers the tp-vs-pspline basis gap and the h-vs-log-t
    //    warp without admitting a genuinely different covariate effect.
    assert!(
        mu_rel <= 0.15,
        "fitted location predictor diverges from survreg: rel_l2={mu_rel:.4} pearson={mu_corr:.5}"
    );
    //  * |log(sigma_gam) - log(sigma_ref)| <= 0.05: log(sigma) is the second AFT
    //    parameter. gam's sigma standardizes residuals on its learned-h clock
    //    while survreg's standardizes on log t; the local slope of h vs log t is
    //    ~unit so the two scales coincide up to that warp. 0.05 (~5% on sigma)
    //    matches the principled scale-agreement margin of the lifelines lognormal
    //    AFT quality test (which fixes h = log t exactly, so gam — learning h — is
    //    entitled to no tighter a bound).
    assert!(
        log_scale_abs <= 0.05,
        "AFT log-scale coefficient diverges from survreg: |diff|={log_scale_abs:.4} \
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
