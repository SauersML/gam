//! End-to-end quality: gam's location-scale accelerated-failure-time (AFT)
//! family with a *smooth covariate* must RECOVER THE KNOWN GENERATIVE TRUTH on
//! synthetic right-censored lognormal data — not merely reproduce another tool's
//! fitted output.
//!
//! OBJECTIVE METRIC (the pass/fail claim). The data is generated from a known
//! location surface
//!   eta_location(x, z) = -0.5 + 0.8*x + sin(pi*z)
//! and a known constant scale `sigma_true`. The gauge-free, mean-centered truth
//! at training row i is therefore `truth_c[i] = 0.8*x[i] + sin(pi*z[i])` (the
//! additive -0.5 anchor cancels under centering, exactly as gam's learned-clock
//! gauge constant does). The PRIMARY assertion is that gam recovers this truth:
//!   RMSE(gam_mu_centered, truth_centered) <= a principled bar tied to the
//!   estimation noise floor (a small fraction of the location signal's spread).
//! The SECONDARY assertion is on the known scale: gam's recovered `log(sigma)`
//! must be within a principled tolerance of `log(sigma_true)`. Neither claim
//! references any other tool's fit — they measure how close gam lands to the
//! function that actually generated the data.
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
//! axis. The smooth in `z` gives a non-trivial design matrix, so recovery here
//! exercises correct parametric assembly *and* smooth-shape recovery, not just
//! an intercept/slope.
//!
//! The gauge. gam learns `h(t)` flexibly (a penalized monotone B-spline on
//! `log t`, see `families::survival_location_scale`). Because the location
//! channel is *linear in its coefficients*, the only part of the fitted location
//! predictor `mu(x, z)` that is NOT identifiable against the truth is a single
//! additive *gauge offset* (the absolute location anchor on the warped clock).
//! We therefore mean-center both gam's `mu(x_i, z_i)` and the generative truth
//! before measuring RMSE; what remains is the gauge-free covariate slope in `x`
//! plus the `s(z)` smooth shape, which is exactly what recovery should validate.
//!
//! The reference. `survival::survreg(dist = "lognormal")` — the gold-standard
//! parametric-AFT engine in R — is fit on the IDENTICAL data and kept only as a
//! BASELINE TO MATCH-OR-BEAT on the same truth-recovery metric: gam's centered
//! RMSE against the truth must be no worse than 1.10x survreg's. survreg is NOT
//! a correctness oracle here; the correctness claim is recovery of the known
//! generative function, and the baseline guards against gam being meaningfully
//! less accurate than a mature tool on the same data. We also print the centered
//! gam-vs-survreg rel_l2 for context only (not asserted).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_r};
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

    // ---- OBJECTIVE TRUTH: the known generative location surface --------------
    // The data was generated from eta_location = -0.5 + 0.8*x + sin(pi*z). The
    // gauge-free, mean-centered truth at each training row is 0.8*x + sin(pi*z)
    // with its sample mean removed (the additive -0.5 anchor, like gam's
    // learned-clock gauge constant, cancels under centering). Both gam's and
    // survreg's fitted location predictors carry an unidentifiable additive gauge
    // offset, so we mean-center every quantity before measuring recovery.
    let truth: Vec<f64> = (0..n)
        .map(|i| 0.8 * x[i] + (std::f64::consts::PI * z[i]).sin())
        .collect();
    let truth_mean = truth.iter().sum::<f64>() / n as f64;
    let truth_c: Vec<f64> = truth.iter().map(|&m| m - truth_mean).collect();

    let gam_mean = gam_mu_train.iter().sum::<f64>() / n as f64;
    let gam_mu_c: Vec<f64> = gam_mu_train.iter().map(|&m| m - gam_mean).collect();

    let ref_mean = ref_mu.iter().sum::<f64>() / n as f64;
    let ref_mu_c: Vec<f64> = ref_mu.iter().map(|&m| m - ref_mean).collect();

    // PRIMARY metric: how well does gam recover the generative location surface?
    let gam_truth_rmse = rmse(&gam_mu_c, &truth_c);
    // BASELINE: survreg's recovery of the SAME truth on the SAME data.
    let ref_truth_rmse = rmse(&ref_mu_c, &truth_c);
    // CONTEXT only (not asserted): centered gam-vs-survreg agreement.
    let mu_rel_vs_ref = relative_l2(&gam_mu_c, &ref_mu_c);

    // Signal spread of the centered truth (RMS amplitude of the location effect).
    let signal_rms = (truth_c.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();

    // SECONDARY metric: recovery of the known constant scale on the log axis.
    let log_sigma_true = sigma_true.ln();
    let gam_log_sigma_err = (gam_log_sigma - log_sigma_true).abs();
    let ref_log_sigma_err = (ref_log_sigma - log_sigma_true).abs();

    eprintln!(
        "lognormal location-scale AFT truth-recovery: n={n} cens={cens_frac:.3} \
         signal_rms={signal_rms:.4} \
         gam_truth_rmse={gam_truth_rmse:.4} ref_truth_rmse={ref_truth_rmse:.4} \
         sigma_true={sigma_true:.4} gam_sigma={gam_sigma:.4} ref_sigma={ref_sigma:.4} \
         log_sigma_true={log_sigma_true:.4} gam_log_sigma={gam_log_sigma:.4} \
         ref_log_sigma={ref_log_sigma:.4} \
         gam_log_sigma_err={gam_log_sigma_err:.4} ref_log_sigma_err={ref_log_sigma_err:.4} \
         mu_rel_l2_vs_survreg={mu_rel_vs_ref:.4}"
    );

    // PRIMARY (truth recovery, absolute). With n=300, ~40% right-censoring and a
    // k=5 thin-plate smooth, the centered location surface is estimated to a
    // small fraction of its own signal spread. The location signal here has
    // signal_rms ~ 0.75 (0.8*x on [-1,1] plus sin(pi*z)); a recovery RMSE of
    // <= 0.30 is well under half that spread and is the bar a genuine assembly
    // or smooth-recovery defect would blow through, while staying honest about
    // the censored, finite-sample estimation noise and the tp-basis null space.
    assert!(
        gam_truth_rmse <= 0.30,
        "gam failed to recover the generative location surface: \
         RMSE(gam, truth)={gam_truth_rmse:.4} (signal_rms={signal_rms:.4})"
    );
    assert!(
        gam_truth_rmse <= 0.40 * signal_rms,
        "gam location recovery error is a large fraction of the signal: \
         RMSE(gam, truth)={gam_truth_rmse:.4} > 0.40*signal_rms={:.4}",
        0.40 * signal_rms
    );

    // BASELINE (match-or-beat survreg on the SAME recovery metric). gam must be
    // no worse than 1.10x the mature parametric-AFT tool at recovering the truth;
    // it is allowed to be better. This guards against gam being meaningfully less
    // accurate than survreg without ever treating survreg's fit as ground truth.
    assert!(
        gam_truth_rmse <= 1.10 * ref_truth_rmse,
        "gam recovers the truth worse than survreg: gam_rmse={gam_truth_rmse:.4} \
         > 1.10*ref_rmse={:.4}",
        1.10 * ref_truth_rmse
    );

    // SECONDARY (known scale recovery). log(sigma_true) is a generative constant;
    // gam's recovered log-scale must land within 0.10 (~10% on sigma) of it under
    // ~40% censoring, AND no worse than survreg + 0.05 on the same target.
    assert!(
        gam_log_sigma_err <= 0.10,
        "gam failed to recover the generative log-scale: |gam_log_sigma - truth|={gam_log_sigma_err:.4} \
         (gam_log_sigma={gam_log_sigma:.4}, log_sigma_true={log_sigma_true:.4})"
    );
    assert!(
        gam_log_sigma_err <= ref_log_sigma_err + 0.05,
        "gam recovers the scale worse than survreg: gam_err={gam_log_sigma_err:.4} \
         > ref_err+0.05={:.4}",
        ref_log_sigma_err + 0.05
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
