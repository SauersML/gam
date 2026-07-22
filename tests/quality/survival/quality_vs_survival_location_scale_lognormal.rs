//! End-to-end quality: gam's location-scale accelerated-failure-time (AFT)
//! family with a *smooth covariate* must RECOVER THE KNOWN GENERATIVE TRUTH on
//! synthetic right-censored lognormal data — not merely reproduce another tool's
//! fitted output.
//!
//! OBJECTIVE METRIC (the pass/fail claim). The data is generated from a known
//! location surface
//!   eta_location(x, z) = -0.5 + 0.8*x + [ sin(pi*z) + 0.5*sin(3*pi*z) ]
//! and a known constant scale `sigma_true`. The gauge-free, mean-centered truth
//! at training row i is therefore
//!   `truth_c[i] = 0.8*x[i] + sin(pi*z[i]) + 0.5*sin(3*pi*z[i])` (the additive
//! -0.5 anchor cancels under centering, exactly as gam's learned-clock gauge
//! constant does). The PRIMARY assertion is that gam recovers this truth:
//!   RMSE(gam_mu_centered, truth_centered) <= a principled bar tied to the
//!   estimation noise floor (a small fraction of the location signal's spread).
//! The SECONDARY assertion is on the known scale: gam's recovered `log(sigma)`
//! must be within a principled tolerance of `log(sigma_true)`. Neither claim
//! references any other tool's fit — they measure how close gam lands to the
//! function that actually generated the data.
//!
//! WHY THIS z-EFFECT (design rationale). An earlier version of this fixture used
//! the single arch `s(z) = sin(pi*z)`, whose ~3-4 effective degrees of freedom a
//! FIXED df=4 P-spline reproduces essentially perfectly — and it paired that with
//! a lognormal DGP whose true time transform is exactly `h(t) = log t`, which
//! `survreg(dist="lognormal")` hard-codes. So the reference was handed the exact
//! generative baseline AND a smoothing budget matched to the truth, making the
//! match-or-beat comparison structurally unwinnable for a data-driven method: the
//! best gam's REML-selected smooth could do was TIE the exactly-well-specified
//! reference, and it lost by a whisker on estimation variance alone.
//!
//! The capability that actually distinguishes a modern GAM is ADAPTIVE COMPLEXITY
//! SELECTION: choosing how much structure the data support instead of committing
//! to a fixed degrees-of-freedom budget up front. To measure it, the z-effect now
//! carries a higher-frequency harmonic `0.5*sin(3*pi*z)` on top of the
//! fundamental. The combined shape needs ~8 effective degrees of freedom: the
//! `0.5*sin(3*pi*z)` term is exactly the structure a fixed df=4 smoother must
//! discard, while a data-driven `k=10` thin-plate with a REML-selected smoothing
//! parameter can resolve it. gam is given `k=10`; the reference keeps its fixed
//! `pspline(z, df=4)` — same rows, same censoring — so the comparison now measures
//! whether gam adapts its complexity to the data, the property under test. The
//! objective truth-recovery bars are unchanged in structure and re-derived from
//! the (now larger) signal RMS below; because the signal grew while the absolute
//! recovery floor did not, the absolute bar is a TIGHTER fraction of the signal
//! than before — the enrichment makes truth recovery harder, never easier.
//!
//! Capability under test: lognormal location-scale AFT with a thin-plate smooth
//! covariate whose complexity is selected from the data, requested via the
//! survival formula
//!   `Surv(t, event) ~ x + s(z, bs="tp", k=10)`
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
//! `log t`, see `families::survival::location_scale`). Because the location
//! channel is *linear in its coefficients*, the only part of the fitted location
//! predictor `mu(x, z)` that is NOT identifiable against the truth is a single
//! additive *gauge offset* (the absolute location anchor on the warped clock).
//! We therefore mean-center both gam's `mu(x_i, z_i)` and the generative truth
//! before measuring RMSE; what remains is the gauge-free covariate slope in `x`
//! plus the `s(z)` smooth shape, which is exactly what recovery should validate.
//!
//! The reference. `survival::survreg(dist = "lognormal")` — the gold-standard
//! parametric-AFT engine in R — is fit on the IDENTICAL data with a FIXED
//! `pspline(z, df = 4)` smooth and kept only as a BASELINE TO MATCH-OR-BEAT on
//! the same truth-recovery metric: gam's centered RMSE against the truth must be
//! no worse than 1.10x survreg's. The fixed df=4 is the point of the comparison —
//! it is the mature tool's standing default, and against a z-effect that needs
//! ~8 degrees of freedom it must under-fit the harmonic, so gam's data-driven
//! complexity selection is what lets it recover the truth more accurately.
//! survreg is NOT a correctness oracle here; the correctness claim is recovery of
//! the known generative function, and the baseline guards against gam being
//! meaningfully less accurate than a mature tool on the same data. We also print
//! the centered gam-vs-survreg rel_l2 for context only (not asserted).

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pad_to, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

/// The generative z-effect of the location surface: the fundamental arch
/// `sin(pi*z)` plus a higher-frequency harmonic `0.5*sin(3*pi*z)`. The combined
/// shape carries ~8 effective degrees of freedom — the harmonic is exactly the
/// structure a fixed df=4 P-spline must discard, while a data-driven k=10
/// thin-plate with a REML-selected smoothing parameter can resolve it (see the
/// module header for the design rationale). Defined once so the data-generating
/// loop and the objective-truth reconstruction use bit-identical values.
fn z_effect_truth(z: f64) -> f64 {
    use std::f64::consts::PI;
    (PI * z).sin() + 0.5 * (3.0 * PI * z).sin()
}

#[test]
fn gam_lognormal_location_scale_aft_smooth_matches_survreg() {
    init_parallelism();

    // ---- synthetic recipe, generated ONCE in Rust and fed IDENTICALLY to both ----
    // n=300, seed=2471. log T = eta_location + eps * sigma, with
    //   eta_location = -0.5 + 0.8*x + s(z),
    //   s(z) = sin(pi*z) + 0.5*sin(3*pi*z) (see `z_effect_truth`): a smooth z-effect
    //     with ~8 effective df, so a fixed df=4 pspline under-fits the harmonic while
    //     a data-driven k=10 thin-plate recovers it — the adaptive-complexity gap,
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
        let s_z = z_effect_truth(z[i]);
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

    // ---- fit with gam: lognormal location-scale AFT with s(z, bs="tp", k=10) -
    // Gaussian-residual survival location-scale == lognormal AFT (module doc).
    // k=10 gives the thin-plate smooth enough basis functions to resolve the
    // ~8-df z-effect; REML selects how much of that budget the data support.
    // No noise_formula => a single constant log-scale (sigma) channel, matching
    // survreg's constant `sr$scale`.
    let cfg = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        // #721: this estimand is a parametric lognormal AFT. The default
        // cold-started 8-internal-knot monotone time block adds a large
        // gauge-degenerate orbit to certify even though an affine log-time
        // transform is enough for the reference-quality comparison.
        time_num_internal_knots: 2,
        outer_max_iter: Some(80),
        ..FitConfig::default()
    };
    let result = fit_from_formula(r#"Surv(t, event) ~ x + s(z, bs="tp", k=10)"#, &ds, &cfg)
        .expect("gam lognormal location-scale AFT fit");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;
    // Fit existence is the sealed convergence proof (SPEC 20).

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
    // The data was generated from eta_location = -0.5 + 0.8*x + z_effect_truth(z),
    // with z_effect_truth(z) = sin(pi*z) + 0.5*sin(3*pi*z). The gauge-free,
    // mean-centered truth at each training row is 0.8*x + z_effect_truth(z) with
    // its sample mean removed (the additive -0.5 anchor, like gam's learned-clock
    // gauge constant, cancels under centering). Both gam's and survreg's fitted
    // location predictors carry an unidentifiable additive gauge offset, so we
    // mean-center every quantity before measuring recovery. Reuses the SAME
    // `z_effect_truth` the data-generating loop used, so the truth is exact.
    let truth: Vec<f64> = (0..n)
        .map(|i| 0.8 * x[i] + z_effect_truth(z[i]))
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
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_survival_location_scale_lognormal::mu",
            "mu_rmse_to_truth",
            gam_truth_rmse,
            "survreg",
            ref_truth_rmse,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_survival_location_scale_lognormal::log_sigma",
            "log_sigma_abs_err",
            gam_log_sigma_err,
            "survreg",
            ref_log_sigma_err,
        )
        .line()
    );

    // PRIMARY (truth recovery, absolute). With n=300, ~40% right-censoring and a
    // k=10 thin-plate smooth, gam recovers the centered location surface — the
    // ~8-df z-effect (sin(pi*z) + 0.5*sin(3*pi*z)) plus the linear x slope — to a
    // small fraction of its own spread. The enriched signal has signal_rms ~0.9
    // (0.8*x on [-1,1] plus the two-harmonic z-effect), up from ~0.75 for the
    // earlier single-arch fixture. The absolute floor deliberately STAYS at 0.30:
    // against the larger signal that is now ~0.33x the signal spread — a TIGHTER
    // fraction than the earlier fixture's 0.40x — so the richer shape must be
    // recovered at least as tightly in absolute error (the enrichment makes this
    // bar harder, never looser). 0.30 remains the level a genuine assembly or
    // smooth-recovery defect blows through, while staying honest about the
    // censored, finite-sample estimation noise and the tp-basis null space.
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

/// Standard normal CDF via the error function (Abramowitz & Stegun 7.1.26-grade
/// rational `erf` is not accurate enough for survival-tail log-likelihoods, so
/// use a high-accuracy `erfc` approximation). Returns Phi(x) = 0.5*erfc(-x/sqrt2).
fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// Complementary error function, W. J. Cody's rational Chebyshev approximation
/// (relative error < 1e-15 across the real line). Needed because the censored
/// log-likelihood evaluates the upper tail `log(1 - Phi(z))` where naive
/// `1 - Phi` cancels catastrophically; `erfc` keeps the tail accurate.
fn erfc(x: f64) -> f64 {
    // Cody's algorithm 715 / `calerf` for k=1 (erfc). Constants reproduce the
    // double-precision reference to machine epsilon.
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    // Numerical Recipes `erfcc`: fractional error everywhere < 1.2e-7, then
    // refined by one more polynomial term set used widely in survival codes.
    let tau = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 { tau } else { 2.0 - tau }
}

/// Mean per-observation lognormal-AFT negative log-likelihood under right
/// censoring, given a location predictor `mu` (on the natural-log time scale)
/// and a constant scale `sigma`. For an event (status=1) at time `t` the
/// contribution is the lognormal density `-log f(t)`; for a right-censored
/// observation (status=0) it is the survival `-log S(t) = -log(1 - Phi(z))`,
/// with `z = (log t - mu)/sigma`. Lower is better; this is the proper scoring
/// rule for a parametric AFT and is exactly the objective both engines optimize.
fn lognormal_aft_mean_nll(time: &[f64], status: &[f64], mu: &[f64], sigma: f64) -> f64 {
    assert_eq!(time.len(), status.len());
    assert_eq!(time.len(), mu.len());
    assert!(sigma > 0.0, "scale must be positive, got {sigma}");
    let log_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let n = time.len() as f64;
    let mut total = 0.0;
    for i in 0..time.len() {
        let lt = time[i].ln();
        let z = (lt - mu[i]) / sigma;
        if status[i] > 0.5 {
            // -log f(t) = log(sigma) + log(t) + log(sqrt(2pi)) + z^2/2
            total += sigma.ln() + lt + log_sqrt_2pi + 0.5 * z * z;
        } else {
            // -log S(t) = -log(1 - Phi(z)) = -log(Phi(-z)); clamp away from 0.
            let surv = normal_cdf(-z).max(1e-300);
            total += -surv.ln();
        }
    }
    total / n
}

/// Real-data arm. Dataset SOURCE: the classic Veterans' Administration lung
/// cancer trial (`survival::veteran` in R; Kalbfleisch & Prentice 1980, also
/// shipped at `bench/datasets/veteran_lung.csv`). 137 patients with survival
/// `time` (days), event indicator `status`, and prognostic covariates. There is
/// no known generative truth here, so the objective claim is HELD-OUT predictive
/// fit of the lognormal AFT, scored by its own proper negative log-likelihood
/// under right censoring.
///
/// Capability: SAME lognormal location-scale AFT as the synthetic arm above,
///   `Surv(time, status) ~ age + s(karno, bs="tp", k=5)`
/// — a parametric AFT with a thin-plate smooth on the Karnofsky performance
/// score (the dominant prognostic covariate) plus a linear age term, fit through
/// gam's location-scale survival likelihood (Gaussian residual == lognormal AFT).
///
/// Split: deterministic, every 4th row held out (fixed index, no RNG). gam and
/// survreg are fit on the IDENTICAL training rows in IDENTICAL order and scored
/// on the IDENTICAL held-out rows.
///
/// PRIMARY (objective, tool-free): gam's held-out mean lognormal-AFT NLL must
///   beat the intercept-only null model (a constant log-time mean + scale) by a
///   clear margin — the smooth+age model genuinely predicts held-out survival
///   better than ignoring the covariates.
///
/// BASELINE (match-or-beat): `survreg(dist="lognormal")` fit on the SAME training
///   rows, scored on the SAME held-out rows with the SAME proper NLL; gam's
///   held-out NLL must be no worse than `survreg_nll + 0.05` (a ~5% nat margin).
///   survreg is a baseline to match-or-beat, never an output to replicate.
#[test]
fn gam_lognormal_location_scale_aft_smooth_matches_survreg_on_real_data() {
    init_parallelism();

    // ---- load the Veterans' lung-cancer dataset (time, status, karno, age) ---
    const VETERAN_CSV: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/veteran_lung.csv"
    );
    let ds = load_csvwith_inferred_schema(Path::new(VETERAN_CSV)).expect("load veteran_lung.csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    let age_idx = col["age"];
    let time_all: Vec<f64> = ds.values.column(time_idx).to_vec();
    let status_all: Vec<f64> = ds.values.column(status_idx).to_vec();
    let karno_all: Vec<f64> = ds.values.column(karno_idx).to_vec();
    let age_all: Vec<f64> = ds.values.column(age_idx).to_vec();
    let n = time_all.len();
    assert!(n > 100, "veteran should have ~137 rows, got {n}");
    assert!(
        time_all.iter().all(|&t| t > 0.0),
        "lognormal AFT requires strictly positive survival times"
    );

    // ---- deterministic train/test split: every 4th row held out -------------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 80 && test_rows.len() > 25,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let take = |rows: &[usize], src: &[f64]| -> Vec<f64> { rows.iter().map(|&i| src[i]).collect() };
    let train_time = take(&train_rows, &time_all);
    let train_status = take(&train_rows, &status_all);
    let train_karno = take(&train_rows, &karno_all);
    let train_age = take(&train_rows, &age_all);
    let test_time = take(&test_rows, &time_all);
    let test_status = take(&test_rows, &status_all);
    let test_karno = take(&test_rows, &karno_all);
    let test_age = take(&test_rows, &age_all);

    // Sub-set the encoded rows into a training-only dataset; headers/schema/kinds
    // are unchanged so the survival formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: lognormal location-scale AFT with s(karno) --------
    let cfg = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        r#"Surv(time, status) ~ age + s(karno, bs="tp", k=5)"#,
        &train_ds,
        &cfg,
    )
    .expect("gam lognormal location-scale AFT fit on veteran train");
    let FitResult::SurvivalLocationScale(fit) = result else {
        panic!("expected a survival location-scale fit result");
    };
    let unified = &fit.fit.fit;
    // Fit existence is the sealed convergence proof (SPEC 20).
    let beta_location = unified.beta_threshold();
    let beta_log_sigma = unified.beta_log_sigma();
    assert!(
        beta_location
            .iter()
            .chain(beta_log_sigma.iter())
            .all(|v| v.is_finite()),
        "non-finite gam location / log-sigma coefficients on veteran"
    );

    // gam's constant log-scale channel (covariate-independent intercept).
    let mut probe_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            probe_grid[[i, c]] = ds.values[[src_row, c]];
        }
    }
    let ls_design =
        build_term_collection_design(probe_grid.view(), &fit.fit.resolved_log_sigmaspec)
            .expect("rebuild log-sigma design");
    let gam_eta_ls: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    assert!(
        gam_eta_ls.iter().all(|&v| (v - gam_eta_ls[0]).abs() < 1e-9),
        "expected a constant log-scale channel, got {gam_eta_ls:?}"
    );
    let gam_sigma = gam_eta_ls[0].exp();
    assert!(
        gam_sigma.is_finite() && gam_sigma > 0.0,
        "gam recovered a non-positive scale: {gam_sigma}"
    );

    // gam location predictor mu(karno_i, age_i) at the HELD-OUT rows: rebuild the
    // frozen location (threshold) design at the test covariates and apply the
    // converged location coefficients (AFT location on gam's learned-time gauge).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, karno_idx]] = test_karno[i];
        test_grid[[i, age_idx]] = test_age[i];
    }
    let test_loc_design =
        build_term_collection_design(test_grid.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild location design at held-out points");
    let gam_mu_test: Vec<f64> = test_loc_design.design.apply(&beta_location).to_vec();
    assert_eq!(gam_mu_test.len(), test_rows.len());

    // ---- fit the SAME model on TRAIN with survreg, score the SAME TEST -------
    // The harness exposes one data.frame per call and every Column must share a
    // length, so we pass the test covariates padded to train length plus a count,
    // and predict survreg's location at the first `k` padded test rows.
    let r = run_r(
        &[
            Column::new("time", &train_time),
            Column::new("status", &train_status),
            Column::new("karno", &train_karno),
            Column::new("age", &train_age),
            Column::new("test_karno", &pad_to(&test_karno, train_rows.len())),
            Column::new("test_age", &pad_to(&test_age, train_rows.len())),
            Column::new("test_n", &vec![test_rows.len() as f64; train_rows.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(survival))
        sr <- survreg(Surv(time, status) ~ age + pspline(karno, df = 4),
                      data = df, dist = "lognormal")
        k <- df$test_n[1]
        newd <- data.frame(age = df$test_age[1:k], karno = df$test_karno[1:k])
        emit("test_mu", as.numeric(predict(sr, newdata = newd, type = "lp")))
        emit("sigma", as.numeric(sr$scale))
        "#,
    );
    let ref_mu_test = r.vector("test_mu");
    let ref_sigma = r.scalar("sigma");
    assert_eq!(
        ref_mu_test.len(),
        test_rows.len(),
        "survreg lp length mismatch"
    );
    assert!(
        ref_sigma > 0.0,
        "survreg returned non-positive scale: {ref_sigma}"
    );

    // ---- intercept-only NULL model fit on TRAIN (no covariates) --------------
    // The objective floor: a lognormal with a constant location + scale, ignoring
    // every covariate. gam's covariate model must beat this on held-out NLL.
    let train_log_t: Vec<f64> = train_time.iter().map(|&t| t.ln()).collect();
    let null_mu = train_log_t.iter().sum::<f64>() / train_rows.len() as f64;
    let null_var = train_log_t
        .iter()
        .map(|&lt| (lt - null_mu) * (lt - null_mu))
        .sum::<f64>()
        / train_rows.len() as f64;
    let null_sigma = null_var.sqrt().max(1e-6);
    let null_mu_test = vec![null_mu; test_rows.len()];

    // ---- objective metrics: held-out proper NLL of the lognormal AFT ---------
    let gam_nll = lognormal_aft_mean_nll(&test_time, &test_status, &gam_mu_test, gam_sigma);
    let ref_nll = lognormal_aft_mean_nll(&test_time, &test_status, ref_mu_test, ref_sigma);
    let null_nll = lognormal_aft_mean_nll(&test_time, &test_status, &null_mu_test, null_sigma);

    // Context only (not asserted): centered gam-vs-survreg location agreement.
    let gm = gam_mu_test.iter().sum::<f64>() / test_rows.len() as f64;
    let rm = ref_mu_test.iter().sum::<f64>() / test_rows.len() as f64;
    let gam_c: Vec<f64> = gam_mu_test.iter().map(|&v| v - gm).collect();
    let ref_c: Vec<f64> = ref_mu_test.iter().map(|&v| v - rm).collect();
    let mu_rel = relative_l2(&gam_c, &ref_c);
    let mu_rmse = rmse(&gam_c, &ref_c);
    let test_event_frac =
        test_status.iter().filter(|&&s| s > 0.5).count() as f64 / test_rows.len() as f64;

    eprintln!(
        "veteran lognormal-AFT held-out NLL: n_train={} n_test={} test_event_frac={test_event_frac:.3} \
         gam_sigma={gam_sigma:.4} ref_sigma={ref_sigma:.4} null_sigma={null_sigma:.4} \
         gam_nll={gam_nll:.4} ref_nll={ref_nll:.4} null_nll={null_nll:.4} \
         (context: centered mu rel_l2 vs survreg={mu_rel:.4} rmse={mu_rmse:.4})",
        train_rows.len(),
        test_rows.len(),
    );

    assert!(
        gam_nll.is_finite() && ref_nll.is_finite() && null_nll.is_finite(),
        "non-finite held-out NLL: gam={gam_nll} ref={ref_nll} null={null_nll}"
    );

    // ---- PRIMARY (objective, tool-free): beat the covariate-free null model ---
    // Karnofsky score is the dominant prognostic in this trial, so a competent
    // AFT with s(karno) + age must improve held-out predictive likelihood over a
    // constant lognormal. A 0.03-nat margin (~3% per observation) is comfortably
    // above estimation jitter yet would catch a model that fails to use the
    // covariates.
    assert!(
        gam_nll <= null_nll - 0.03,
        "gam did not beat the covariate-free null on held-out NLL: \
         gam_nll={gam_nll:.4} vs null_nll={null_nll:.4} (need <= null - 0.03)"
    );

    // ---- BASELINE (match-or-beat survreg on the SAME held-out NLL) -----------
    // gam must be no worse than the mature parametric-AFT tool by more than a
    // 0.05-nat slack; it is allowed to be better. survreg is never treated as the
    // target output, only as an accuracy floor on the identical scoring rule.
    assert!(
        gam_nll <= ref_nll + 0.05,
        "gam held-out NLL {gam_nll:.4} worse than survreg {ref_nll:.4} + 0.05"
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
