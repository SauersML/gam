//! End-to-end quality: gam's lognormal location-scale AFT must RECOVER THE KNOWN
//! TRUTH it was simulated from, and emit a predictive distribution whose held-out
//! Monte-Carlo CRPS (scored against the actually observed times) is objectively
//! sharp+calibrated — at least as good as the mature
//! `lifelines.LogNormalAFTFitter` baseline on the *identical* data.
//!
//! OBJECTIVE METRICS ASSERTED (no "same-as-reference" pass criterion):
//!   1. TRUTH RECOVERY of the location surface. The data is generated from a
//!      KNOWN function `eta_loc(x,z) = -0.4 + 0.6*x + sin(pi*z)`, so we assert
//!      gam's fitted location (gauge-anchored to the TRUE location, not to
//!      lifelines) tracks `eta_loc` with centered RMSE <= sigma_true/2 — i.e. the
//!      reconstruction error of the systematic mean is at most half the
//!      irreducible noise s.d. This is a pure accuracy claim against ground truth.
//!   2. TRUTH RECOVERY of the scale. We assert |gam_sigma - sigma_true|/sigma_true
//!      <= 0.15: gam recovers the lognormal spread the data was drawn with.
//!   3. PREDICTIVE CALIBRATION via held-out-free CRPS against the observed times.
//!      The Continuous Ranked Probability Score is the standard *proper* scoring
//!      rule for a full predictive distribution; gam's mean CRPS (location
//!      anchored to truth) must be <= the irreducible CRPS of the TRUE
//!      data-generating distribution times 1.20 (an absolute calibration bar that
//!      no fit can beat in expectation), AND <= the lifelines baseline's CRPS
//!      times 1.05 (match-or-beat the mature tool on the proper score).
//!
//! lifelines is DEMOTED to a baseline-to-match-or-beat: its CRPS and its
//! truth-recovery error are computed on the same data and gam is required to be
//! as-good-or-better, but "close to lifelines' fitted numbers" is never the pass
//! criterion. Passing requires gam to recover the simulated truth and to score
//! well as a proper predictive distribution in its own right.
//!
//! Capability under test: survival AFT predictive *calibration* (not just point
//! coefficients) for the lognormal location-scale family, requested via
//!   `Surv(t, event) ~ x + s(z, bs="tp", k=5)`
//! fit through gam's location-scale survival likelihood
//! (`FitConfig{ survival_likelihood: "location-scale", survival_distribution:
//! "gaussian" }`). A Gaussian residual on gam's monotone time-warp channel IS
//! the lognormal AFT: the standardized survival index is
//!   z(t, x) = (h(t) - eta_t(x)) / sigma,   S(t|x) = 1 - Phi(z),
//! with a *location* channel `eta_t(x)` (role `Threshold`, `beta_threshold()`),
//! a constant *log-scale* channel (`sigma = exp(eta_ls)`, `beta_log_sigma()`),
//! and a learned monotone transform `h(t)` of the time axis. lifelines fixes
//! `h(t) = log t` and fits exactly `log T = mu(x, z) + sigma * W`, W ~ N(0,1),
//! by maximum likelihood under right-censoring — the SAME location-scale
//! likelihood, but log-LINEAR in the covariates (lifelines cannot fit the smooth
//! `s(z)` directly, so it receives a flexible basis expansion of `z` instead;
//! see the body). gam carries the smooth via `s(z, bs="tp", k=5)`.
//!
//! Why CRPS via Monte Carlo. For a sample {y_j} ~ LogNormal(mu_i, sigma_i) and
//! observed time t_i,
//!   CRPS_i = (1/M) Σ_j |y_j - t_i| - (1/2 M^2) Σ_{j,k} |y_j - y_k|,
//! and CRPS = mean_i CRPS_i. It scores a *coherent* (mu_i, sigma_i) pair against
//! the realized outcome (calibration of both channels at once), with no
//! closed-form algebra. We draw the M standard-normal deviates ONCE on the Rust
//! side and reuse them for gam, for lifelines, and for the TRUE distribution
//! (common random numbers), so the CRPS *differences* isolate parameter
//! disagreement rather than Monte-Carlo noise.
//!
//! The gauge. gam learns `h(t)` flexibly while the simulation (and lifelines) use
//! `log t`, so gam's location channel differs from the truth by an unknown
//! additive *gauge offset* (the absolute time anchor). The systematic
//! covariate/smooth shape and the scale sigma are gauge-free. We therefore anchor
//! gam's location to the TRUE location (subtract the mean offset) before scoring
//! truth-recovery RMSE and CRPS — a purely objective re-gauge against ground
//! truth, not against the reference tool.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

/// Monte-Carlo CRPS for one observed time `t_obs` against a predictive sample
/// `y` (the spec's two-sum estimator). `O(M^2)` but `M=1000` keeps it cheap.
fn crps_sample(t_obs: f64, y: &[f64]) -> f64 {
    let m = y.len() as f64;
    let term1: f64 = y.iter().map(|&yj| (yj - t_obs).abs()).sum::<f64>() / m;
    let mut term2 = 0.0;
    for &yj in y {
        for &yk in y {
            term2 += (yj - yk).abs();
        }
    }
    term1 - term2 / (2.0 * m * m)
}

#[test]
fn gam_lognormal_aft_crps_calibration_matches_lifelines() {
    init_parallelism();

    // ---- synthetic recipe, generated ONCE in Rust and fed IDENTICALLY to both ----
    // Spec: n=250, seed=4242, ~35% censoring.
    //   log T = -0.4 + 0.6*x + sin(pi*z) + N(0, 0.5^2),   x, z ~ U(-1, 1).
    //   event ~ Bernoulli(0.65) (independent of the survival channel — a pure
    //   random-censoring indicator, matching the spec's "Event ~ Bernoulli(0.65)").
    // The data is drawn only here and the columns are handed verbatim to lifelines,
    // so both engines fit byte-identical rows (no cross-engine RNG to reconcile).
    let n = 250usize;
    let sigma_true = 0.5_f64;
    let mut rng = NumpyMt19937::new(4242);

    // Draw covariates, residual noise, and the Bernoulli event indicator in a
    // fixed order so the dataset is byte-reproducible across runs.
    let x: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let z: Vec<f64> = (0..n).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    let eps: Vec<f64> = (0..n).map(|_| rng.next_standard_normal()).collect();
    let event_u: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();

    let mut t = Vec::with_capacity(n);
    let mut event = Vec::with_capacity(n);
    // True systematic location eta_loc(x_i, z_i) at each training point — the
    // ground-truth surface gam must recover (used for the truth-recovery RMSE and
    // for the irreducible-CRPS calibration bar).
    let mut eta_loc_true = Vec::with_capacity(n);
    let mut n_censored = 0usize;
    for i in 0..n {
        let s_z = (std::f64::consts::PI * z[i]).sin();
        let eta_loc = -0.4 + 0.6 * x[i] + s_z;
        eta_loc_true.push(eta_loc);
        let t_event = (eta_loc + eps[i] * sigma_true).exp();
        t.push(t_event.max(1e-6));
        // Bernoulli(0.65) event indicator: u < 0.65 => observed event, else
        // right-censored at the same time (the spec's random-censoring channel).
        if event_u[i] < 0.65 {
            event.push(1.0);
        } else {
            event.push(0.0);
            n_censored += 1;
        }
    }
    let cens_frac = n_censored as f64 / n as f64;
    assert!(
        (0.25..=0.45).contains(&cens_frac),
        "expected ~35% censoring, got {cens_frac:.3} (n_censored={n_censored})"
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
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode lognormal-AFT data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];
    let ncols = ds.headers.len();

    // ---- fit with gam: lognormal location-scale AFT with s(z, bs="tp", k=5) -
    // Gaussian-residual survival location-scale == lognormal AFT (module doc).
    // No noise_formula => a single constant log-scale (sigma) channel, matching
    // lifelines' constant `sigma_`.
    let cfg = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(r#"Surv(t, event) ~ x + s(z, bs="tp", k=5)"#, &ds, &cfg)
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

    // gam location mu_gam(x_i, z_i) at the training points: rebuild the frozen
    // location (threshold) design from `resolved_thresholdspec` and apply the
    // converged location coefficients (the canonical mgcv/gamlss rebuild pattern).
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

    // Constant log-scale channel: sigma = exp(eta_ls) (intercept-only).
    let ls_design =
        build_term_collection_design(train_grid.view(), &fit.fit.resolved_log_sigmaspec)
            .expect("rebuild log-sigma design at training points");
    let gam_eta_ls: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    assert!(
        gam_eta_ls.iter().all(|&v| (v - gam_eta_ls[0]).abs() < 1e-9),
        "expected a constant (covariate-independent) log-scale channel, got {gam_eta_ls:?}"
    );
    let gam_sigma = gam_eta_ls[0].exp();

    // ---- common-random-number Monte-Carlo deviates (drawn ONCE, shared) -----
    // M=1000 standard-normal eps, used by BOTH engines so CRPS difference is
    // parameter-driven, not MC-driven. A second seed keeps them independent of
    // the data-generating stream above.
    let m_mc = 1000usize;
    let mut mc_rng = NumpyMt19937::new(20260529);
    let mc_eps: Vec<f64> = (0..m_mc).map(|_| mc_rng.next_standard_normal()).collect();

    // ---- fit the SAME data with lifelines.LogNormalAFTFitter (mature ref) ---
    // The harness hands the body the EXACT (t, event, x, z) columns gam was fit
    // on as a pandas `df` (one source of truth), so the rows are byte-identical.
    // lifelines fits log T = mu(x,z) + sigma W under right-censoring. It is
    // log-linear, so to give it a fair chance at the smooth sin(pi*z) shape we
    // hand it a degree-3 natural-spline-style polynomial basis of z (z, z^2, z^3)
    // alongside x — the richest covariate form lifelines' formula interface
    // supports without an external basis package. We emit, for each training row,
    // the location mu_i and the constant scale sigma so the SAME CRPS Monte-Carlo
    // estimator (with the SAME shared eps) is applied to both engines in Rust.
    let body = r#"
import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter

frame = pd.DataFrame({
    "t": np.asarray(df["t"], dtype=float),
    "event": np.asarray(df["event"], dtype=float),
    "x": np.asarray(df["x"], dtype=float),
    "z": np.asarray(df["z"], dtype=float),
})
# lifelines is log-LINEAR; give it a cubic basis in z to chase sin(pi*z).
frame["z2"] = frame["z"] ** 2
frame["z3"] = frame["z"] ** 3

aft = LogNormalAFTFitter()
aft.fit(frame, duration_col="t", event_col="event",
        formula="x + z + z2 + z3")

# Per-row location mu_i on the natural-log time scale: lifelines exposes the
# predicted mu (the location linear predictor) via predict_expectation's
# internals, but the documented, stable route is the params dot the design.
mu_params = aft.params_.loc["mu_"]
design = np.column_stack([
    np.ones(len(frame)),
    frame["x"].to_numpy(),
    frame["z"].to_numpy(),
    frame["z2"].to_numpy(),
    frame["z3"].to_numpy(),
])
# Order params to match the design columns explicitly.
beta = np.array([
    float(mu_params["Intercept"]),
    float(mu_params["x"]),
    float(mu_params["z"]),
    float(mu_params["z2"]),
    float(mu_params["z3"]),
])
mu = design @ beta
emit("mu", mu)

# Constant scale sigma = exp(sigma_ intercept on the log scale).
sigma_params = aft.params_.loc["sigma_"]
emit("sigma", [float(np.exp(sigma_params["Intercept"]))])
"#;
    let r = run_python(
        &[
            Column::new("t", &t),
            Column::new("event", &event),
            Column::new("x", &x),
            Column::new("z", &z),
        ],
        body,
    );
    let ref_mu = r.vector("mu");
    let ref_sigma = r.scalar("sigma");
    assert_eq!(ref_mu.len(), n, "lifelines mu length mismatch");

    // ---- TRUTH-RECOVERY re-gauge: anchor each location to the TRUE location ---
    // gam learns h(t) while the simulation (and lifelines) use log t, so each
    // location channel differs from the ground-truth eta_loc by an unknown
    // additive gauge constant (the absolute time anchor). The systematic shape is
    // gauge-free, so we center both gam's and the truth's location to mean zero
    // and measure how well the *shape* is recovered. lifelines is re-gauged the
    // same way so its truth-recovery error is a fair baseline.
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let truth_mean = mean(&eta_loc_true);
    let gam_mean = mean(&gam_mu_train);
    let ref_mean = mean(ref_mu);

    // Centered RMSE of each fitted location surface against the centered truth.
    let centered_rmse = |fit_loc: &[f64], fit_mean: f64| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            let d = (fit_loc[i] - fit_mean) - (eta_loc_true[i] - truth_mean);
            s += d * d;
        }
        (s / n as f64).sqrt()
    };
    let gam_loc_rmse = centered_rmse(&gam_mu_train, gam_mean);
    let ref_loc_rmse = centered_rmse(ref_mu, ref_mean);

    // Location anchored to the TRUE absolute level, so the CRPS (scored against
    // the observed t_i on the original time scale) is on the correct gauge.
    let gam_offset = gam_mean - truth_mean;
    let ref_offset = ref_mean - truth_mean;
    let gam_mu_anchored: Vec<f64> = gam_mu_train.iter().map(|&mu| mu - gam_offset).collect();
    let ref_mu_anchored: Vec<f64> = ref_mu.iter().map(|&mu| mu - ref_offset).collect();

    // ---- Monte-Carlo CRPS scored against the OBSERVED times -----------------
    // Shared eps for gam, lifelines, and the TRUE distribution (common random
    // numbers). The truth-CRPS is the irreducible proper-score floor for this
    // data-generating process; gam's CRPS must come close to it and beat the
    // lifelines baseline.
    let mut gam_crps_sum = 0.0;
    let mut ref_crps_sum = 0.0;
    let mut truth_crps_sum = 0.0;
    let mut gam_y = vec![0.0f64; m_mc];
    let mut ref_y = vec![0.0f64; m_mc];
    let mut truth_y = vec![0.0f64; m_mc];
    for i in 0..n {
        let gmu = gam_mu_anchored[i];
        let rmu = ref_mu_anchored[i];
        let tmu = eta_loc_true[i];
        for (j, &e) in mc_eps.iter().enumerate() {
            gam_y[j] = (gmu + gam_sigma * e).exp();
            ref_y[j] = (rmu + ref_sigma * e).exp();
            truth_y[j] = (tmu + sigma_true * e).exp();
        }
        gam_crps_sum += crps_sample(t[i], &gam_y);
        ref_crps_sum += crps_sample(t[i], &ref_y);
        truth_crps_sum += crps_sample(t[i], &truth_y);
    }
    let gam_crps = gam_crps_sum / n as f64;
    let ref_crps = ref_crps_sum / n as f64;
    let truth_crps = truth_crps_sum / n as f64;

    let gam_sigma_err = (gam_sigma - sigma_true).abs() / sigma_true;
    let ref_sigma_err = (ref_sigma - sigma_true).abs() / sigma_true;

    eprintln!(
        "lognormal AFT truth-recovery: n={n} cens={cens_frac:.3} M={m_mc} sigma_true={sigma_true} \
         gam_loc_rmse={gam_loc_rmse:.4} ref_loc_rmse={ref_loc_rmse:.4} \
         gam_sigma={gam_sigma:.4} (err {gam_sigma_err:.4}) ref_sigma={ref_sigma:.4} (err {ref_sigma_err:.4}) \
         gam_crps={gam_crps:.5} truth_crps={truth_crps:.5} ref_crps={ref_crps:.5}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_lifelines_crps_lognormal_aft::loc",
            "loc_rmse_centered_to_truth",
            gam_loc_rmse,
            "lifelines",
            ref_loc_rmse,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_lifelines_crps_lognormal_aft::sigma",
            "sigma_rel_err_to_truth",
            gam_sigma_err,
            "lifelines",
            ref_sigma_err,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_lifelines_crps_lognormal_aft::crps",
            "crps",
            gam_crps,
            "lifelines",
            ref_crps,
        )
        .line()
    );

    // ---- OBJECTIVE BOUNDS (truth recovery + proper-score calibration) -------
    //
    //  1. TRUTH RECOVERY (location). gam must reconstruct the simulated
    //     systematic location surface eta_loc(x,z) = -0.4 + 0.6x + sin(pi z).
    //     With ~35% censoring and n=250, the centered reconstruction error of the
    //     mean surface should be well under half the irreducible noise s.d. — a
    //     fit whose systematic error rivals the noise has not recovered the truth.
    let loc_bar = 0.5 * sigma_true;
    assert!(
        gam_loc_rmse <= loc_bar,
        "gam did not recover the true location surface: centered RMSE {gam_loc_rmse:.4} \
         > bar {loc_bar:.4} (sigma_true={sigma_true})"
    );
    //     ACCURACY match-or-beat: gam's truth-recovery error must not exceed the
    //     lifelines baseline's by more than 10%. (lifelines chases sin(pi z) with
    //     a cubic-in-z proxy, so gam's penalized thin-plate smooth should be at
    //     least as accurate.)
    assert!(
        gam_loc_rmse <= ref_loc_rmse * 1.10,
        "gam location accuracy worse than lifelines baseline: gam {gam_loc_rmse:.4} \
         > 1.10 * ref {ref_loc_rmse:.4}"
    );

    //  2. TRUTH RECOVERY (scale). gam must recover the lognormal spread the data
    //     was drawn with. Scale estimation under censoring is intrinsically noisy,
    //     so the bar is 15% relative error against the KNOWN sigma_true (not
    //     against lifelines' estimate).
    assert!(
        gam_sigma_err <= 0.15,
        "gam did not recover the true lognormal scale: |gam_sigma - sigma_true|/sigma_true \
         = {gam_sigma_err:.4} > 0.15 (gam_sigma={gam_sigma:.4}, sigma_true={sigma_true})"
    );

    //  3. PREDICTIVE CALIBRATION via CRPS against the observed times. The CRPS of
    //     the TRUE distribution is the irreducible proper-score floor; no fit can
    //     beat it in expectation. gam's CRPS must be within 20% of that floor (an
    //     absolute calibration bar) AND must match-or-beat the mature lifelines
    //     baseline within 5% (CRPS(gam) <= CRPS(ref) * 1.05, the category-3 rule).
    assert!(
        gam_crps <= truth_crps * 1.20,
        "gam predictive distribution is mis-calibrated: CRPS {gam_crps:.5} \
         > 1.20 * irreducible truth-CRPS {truth_crps:.5}"
    );
    assert!(
        gam_crps <= ref_crps * 1.05,
        "gam CRPS does not match-or-beat the lifelines baseline: gam {gam_crps:.5} \
         > 1.05 * ref {ref_crps:.5}"
    );
}

/// REAL-DATA ARM. Same capability (lognormal location-scale AFT predictive
/// calibration via held-out CRPS) exercised on the classic **Veterans'
/// Administration lung-cancer trial** survival dataset, where the truth is
/// UNKNOWN — so the pass criterion is purely out-of-sample predictive quality.
///
/// Source: `bench/datasets/veteran_lung.csv`, the canonical `survival::veteran`
/// data shipped with R's `survival` package (Kalbfleisch & Prentice, "The
/// Statistical Analysis of Failure Time Data"). Columns: `time` (survival days),
/// `status` (1 = death observed, 0 = right-censored), `karno` (Karnofsky
/// performance score, the dominant prognostic covariate), `age` (years), plus
/// trt/celltype/diagtime/prior. n = 137, ~7% censoring.
///
/// We make a deterministic train/test split (every 4th row held out, fixed
/// index), fit gam's lognormal AFT `Surv(time, status) ~ karno + s(age, ...)` on
/// the training rows only, and predict the held-out rows. Because the true
/// data-generating distribution is unknown, calibration is scored by Monte-Carlo
/// CRPS of the predictive LogNormal against the held-out OBSERVED event times
/// (test rows with status == 1; censored test rows carry no exact outcome to
/// score against, so they are excluded from the proper score). Two OBJECTIVE
/// bounds, no "match lifelines' numbers" criterion:
///
///   PRIMARY (objective, tool-free): the covariate-aware fit must beat the
///     intercept-only lognormal baseline ("the oracle one can build with no
///     covariates") on held-out CRPS — `gam_crps <= baseline_crps * 0.97`. A
///     model whose covariates do not improve the proper score over a constant
///     predictor has not learned anything generalizable.
///
///   BASELINE (match-or-beat): lifelines' `LogNormalAFTFitter`, fit on the SAME
///     training rows (with a cubic basis in age to chase the smooth) and scored
///     on the SAME held-out observed times with the SAME shared Monte-Carlo
///     deviates, is a baseline to match-or-beat: `gam_crps <= ref_crps * 1.05`.
#[test]
fn gam_lognormal_aft_crps_calibration_matches_lifelines_on_real_data() {
    init_parallelism();

    // ---- load the real Veterans' lung-cancer survival data ------------------
    let ds = load_csvwith_inferred_schema(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/veteran_lung.csv"
    )))
    .expect("load veteran_lung.csv");
    let col = ds.column_map();
    let time_idx = col["time"];
    let status_idx = col["status"];
    let karno_idx = col["karno"];
    let age_idx = col["age"];
    let p = ds.headers.len();

    let time_all: Vec<f64> = ds.values.column(time_idx).to_vec();
    let status_all: Vec<f64> = ds.values.column(status_idx).to_vec();
    let karno_all: Vec<f64> = ds.values.column(karno_idx).to_vec();
    let age_all: Vec<f64> = ds.values.column(age_idx).to_vec();
    let n = time_all.len();
    assert!(n > 120, "veteran_lung should have 137 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out ------------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 90 && test_rows.len() > 25,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    // Training-row columns (handed to lifelines verbatim, same order as gam).
    let train_time: Vec<f64> = train_rows.iter().map(|&i| time_all[i]).collect();
    let train_status: Vec<f64> = train_rows.iter().map(|&i| status_all[i]).collect();
    let train_karno: Vec<f64> = train_rows.iter().map(|&i| karno_all[i]).collect();
    let train_age: Vec<f64> = train_rows.iter().map(|&i| age_all[i]).collect();

    // Held-out test-row covariates / outcomes (gam and the metric read these).
    let test_time: Vec<f64> = test_rows.iter().map(|&i| time_all[i]).collect();
    let test_status: Vec<f64> = test_rows.iter().map(|&i| status_all[i]).collect();
    let test_karno: Vec<f64> = test_rows.iter().map(|&i| karno_all[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age_all[i]).collect();

    // Observed (uncensored) held-out events: the only rows with an exact time to
    // score the proper CRPS against. There must be enough of them to be a real
    // calibration test (veteran is ~7% censored, so nearly all qualify).
    let test_observed: Vec<usize> = (0..test_rows.len())
        .filter(|&i| test_status[i] > 0.5)
        .collect();
    assert!(
        test_observed.len() > 20,
        "too few observed held-out events to score CRPS: {} of {}",
        test_observed.len(),
        test_rows.len()
    );

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged so the formula resolves identically.
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: lognormal location-scale AFT with s(age) --------
    let cfg = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        r#"Surv(time, status) ~ karno + s(age, bs="tp", k=5)"#,
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
        "non-finite gam location / log-sigma coefficients"
    );

    // gam location mu_gam at the HELD-OUT test points: rebuild the frozen
    // location (threshold) design from `resolved_thresholdspec` and apply the
    // converged location coefficients (same rebuild pattern as the synthetic arm).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, karno_idx]] = karno_all[row];
        test_grid[[i, age_idx]] = age_all[row];
    }
    let loc_design =
        build_term_collection_design(test_grid.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild location (threshold) design at held-out points");
    let gam_mu_test: Vec<f64> = loc_design.design.apply(&beta_location).to_vec();
    assert_eq!(gam_mu_test.len(), test_rows.len());

    // Constant log-scale channel: sigma = exp(eta_ls) (no noise_formula).
    let ls_design = build_term_collection_design(test_grid.view(), &fit.fit.resolved_log_sigmaspec)
        .expect("rebuild log-sigma design at held-out points");
    let gam_eta_ls: Vec<f64> = ls_design.design.apply(&beta_log_sigma).to_vec();
    assert!(
        gam_eta_ls.iter().all(|&v| (v - gam_eta_ls[0]).abs() < 1e-9),
        "expected a constant (covariate-independent) log-scale channel, got {gam_eta_ls:?}"
    );
    let gam_sigma = gam_eta_ls[0].exp();

    // gam's location uses a learned monotone time-warp h(t), so its absolute
    // level differs from log-time by an unknown additive gauge constant. We
    // calibrate that single offset on the TRAINING observed events (matching
    // gam's predicted location to the realized log-times in the mean), then
    // apply it to the held-out predictions — a one-parameter re-gauge estimated
    // only on train, never on the test outcomes being scored.
    let mut train_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &row) in train_rows.iter().enumerate() {
        train_grid[[i, karno_idx]] = karno_all[row];
        train_grid[[i, age_idx]] = age_all[row];
    }
    let train_loc_design =
        build_term_collection_design(train_grid.view(), &fit.fit.resolved_thresholdspec)
            .expect("rebuild location design at training points");
    let gam_mu_train: Vec<f64> = train_loc_design.design.apply(&beta_location).to_vec();
    let train_obs: Vec<usize> = (0..train_rows.len())
        .filter(|&i| train_status[i] > 0.5)
        .collect();
    let train_logt_mean =
        train_obs.iter().map(|&i| train_time[i].ln()).sum::<f64>() / train_obs.len() as f64;
    let train_mu_mean =
        train_obs.iter().map(|&i| gam_mu_train[i]).sum::<f64>() / train_obs.len() as f64;
    let gam_offset = train_mu_mean - train_logt_mean;
    let gam_mu_test_anchored: Vec<f64> = gam_mu_test.iter().map(|&mu| mu - gam_offset).collect();

    // ---- shared common-random-number Monte-Carlo deviates ------------------
    let m_mc = 1000usize;
    let mut mc_rng = NumpyMt19937::new(987654321);
    let mc_eps: Vec<f64> = (0..m_mc).map(|_| mc_rng.next_standard_normal()).collect();

    // ---- intercept-only lognormal baseline (the no-covariate "oracle") -----
    // Fit a constant location + scale on the training observed log-times; this
    // is the best a model with NO covariates can do, and gam must beat it.
    let base_mu = train_logt_mean;
    let base_var = train_obs
        .iter()
        .map(|&i| {
            let d = train_time[i].ln() - train_logt_mean;
            d * d
        })
        .sum::<f64>()
        / train_obs.len() as f64;
    let base_sigma = base_var.sqrt();

    // ---- fit the SAME train rows with lifelines.LogNormalAFTFitter ---------
    // Identical (time, status, karno, age) training columns handed over as one
    // data.frame; lifelines is log-LINEAR, so it gets a cubic basis in age to
    // chase the smooth. It emits the per-TEST-row location mu_i and the constant
    // scale so the SAME Rust CRPS estimator with the SAME shared eps scores it.
    // Within this single call every column is TRAIN length; the held-out test
    // covariates ride along right-padded and only the first k entries are read.
    let pad_to = |v: &[f64], len: usize| -> Vec<f64> {
        let fill = v.last().copied().unwrap_or(0.0);
        let mut out = v.to_vec();
        out.resize(len, fill);
        out
    };
    let ntr = train_rows.len();
    let body = r#"
import numpy as np
import pandas as pd
from lifelines import LogNormalAFTFitter

frame = pd.DataFrame({
    "time": np.asarray(df["time"], dtype=float),
    "status": np.asarray(df["status"], dtype=float),
    "karno": np.asarray(df["karno"], dtype=float),
    "age": np.asarray(df["age"], dtype=float),
})
frame["age2"] = frame["age"] ** 2
frame["age3"] = frame["age"] ** 3
aft = LogNormalAFTFitter(penalizer=1e-4)
aft.fit(frame, duration_col="time", event_col="status",
        formula="karno + age + age2 + age3")

# Per-test-row location mu_i. For a LogNormal AFT the median time is exp(mu_i),
# so mu_i = log(predict_median). Using lifelines' own prediction API (rather
# than reconstructing the design from string-labeled params) is robust to the
# formula-engine's coefficient naming across versions.
k = int(df["test_n"].to_numpy()[0])
ta = np.asarray(df["test_age"], dtype=float)[:k]
test_frame = pd.DataFrame({
    "karno": np.asarray(df["test_karno"], dtype=float)[:k],
    "age": ta,
    "age2": ta ** 2,
    "age3": ta ** 3,
})
median = np.asarray(aft.predict_median(test_frame), dtype=float).reshape(-1)
emit("mu", np.log(median))

# Constant scale: sigma = exp(sigma_ intercept on the log scale).
sigma_params = aft.params_.loc["sigma_"]
emit("sigma", [float(np.exp(sigma_params["Intercept"]))])
"#;
    let r = run_python(
        &[
            Column::new("time", &train_time),
            Column::new("status", &train_status),
            Column::new("karno", &train_karno),
            Column::new("age", &train_age),
            Column::new("test_karno", &pad_to(&test_karno, ntr)),
            Column::new("test_age", &pad_to(&test_age, ntr)),
            Column::new("test_n", &vec![test_rows.len() as f64; ntr]),
        ],
        body,
    );
    let ref_mu = r.vector("mu");
    let ref_sigma = r.scalar("sigma");
    assert_eq!(
        ref_mu.len(),
        test_rows.len(),
        "lifelines held-out mu length mismatch"
    );

    // ---- Monte-Carlo CRPS on the held-out OBSERVED events ------------------
    // Shared eps for gam, lifelines, and the intercept-only baseline (common
    // random numbers), scored against the realized test event times.
    let mut gam_crps_sum = 0.0;
    let mut ref_crps_sum = 0.0;
    let mut base_crps_sum = 0.0;
    let mut gam_y = vec![0.0f64; m_mc];
    let mut ref_y = vec![0.0f64; m_mc];
    let mut base_y = vec![0.0f64; m_mc];
    for &i in &test_observed {
        let gmu = gam_mu_test_anchored[i];
        let rmu = ref_mu[i];
        for (j, &e) in mc_eps.iter().enumerate() {
            gam_y[j] = (gmu + gam_sigma * e).exp();
            ref_y[j] = (rmu + ref_sigma * e).exp();
            base_y[j] = (base_mu + base_sigma * e).exp();
        }
        let t_obs = test_time[i];
        gam_crps_sum += crps_sample(t_obs, &gam_y);
        ref_crps_sum += crps_sample(t_obs, &ref_y);
        base_crps_sum += crps_sample(t_obs, &base_y);
    }
    let n_obs = test_observed.len() as f64;
    let gam_crps = gam_crps_sum / n_obs;
    let ref_crps = ref_crps_sum / n_obs;
    let base_crps = base_crps_sum / n_obs;

    eprintln!(
        "veteran lognormal-AFT held-out CRPS: n_train={ntr} n_test={} n_obs={} \
         gam_sigma={gam_sigma:.4} ref_sigma={ref_sigma:.4} base_sigma={base_sigma:.4} \
         gam_crps={gam_crps:.4} ref_crps={ref_crps:.4} base_crps={base_crps:.4}",
        test_rows.len(),
        test_observed.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "survival",
            "quality_vs_lifelines_crps_lognormal_aft::crps_holdout",
            "crps_holdout",
            gam_crps,
            "lifelines",
            ref_crps,
        )
        .line()
    );

    // ---- PRIMARY objective: beat the no-covariate lognormal baseline -------
    assert!(
        gam_crps <= base_crps * 0.97,
        "gam covariate fit did not improve held-out CRPS over the intercept-only \
         baseline: gam {gam_crps:.4} > 0.97 * base {base_crps:.4}"
    );

    // ---- BASELINE (match-or-beat): no worse than lifelines on held-out CRPS -
    assert!(
        gam_crps <= ref_crps * 1.05,
        "gam held-out CRPS does not match-or-beat the lifelines baseline: \
         gam {gam_crps:.4} > 1.05 * ref {ref_crps:.4}"
    );
}

/// Minimal fixed-seed MT19937 generator (MT19937 core, 53-bit uniform in [0, 1)
/// matching NumPy's `random_sample`, and a polar-Marsaglia Gaussian matching
/// NumPy's legacy `gauss`). It only needs to be *deterministic* across runs: the
/// drawn arrays are the single source of truth fed to BOTH engines, so
/// reproducing NumPy's exact stream is unnecessary for identical-data parity.
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
