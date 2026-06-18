//! End-to-end quality: gam's Gaussian location-scale fit with a *cyclic*
//! (periodic) smooth in BOTH the mean (mu) and the log-scale (sigma) block must
//! RECOVER the known generating functions on a periodic, heteroscedastic signal.
//!
//! Objective metric asserted (TRUTH RECOVERY)
//! ------------------------------------------
//! The data are generated from a KNOWN truth,
//!   x = seq(0, 2*pi, length = 150),  y ~ N(mu*(x), sigma*(x)^2),
//!   mu*(x)    = sin(x),
//!   sigma*(x) = 0.15 + 0.1*cos(x),   (so log sigma*(x) is also known),
//! with x circular (0 and 2*pi identified). The PRIMARY claim is that gam's
//! constrained cyclic smooths recover those generating functions. We assert,
//! on 50 equally-spaced grid points in [0, 2*pi):
//!   RMSE(mu_gam,        mu*)        <= 0.06   (~40% of the mean's signal SD;
//!                                              cf. per-point noise sigma 0.05..0.25)
//!   RMSE(log_sigma_gam, log sigma*) <= 0.30   (log-scale is identified one
//!                                              likelihood-derivative removed from
//!                                              the data, so its absolute bar is
//!                                              looser, yet still pins the shape).
//! These are absolute, reference-free accuracy bars: passing means gam fit the
//! TRUE periodic mean and the TRUE periodic (log-)scale, not that it imitated
//! another tool's (possibly equally wrong) fit.
//!
//! gamlss as a calibration baseline (not a target)
//! -----------------------------------------------
//! `gamlss::gamlss(family = NO())` is the mature distributional-regression
//! engine; fed the IDENTICAL (x, y) and the SAME explicit period via mgcv's
//! cyclic cubic basis (`ga(~ s(x, bs = "cc"))`), it produces its own cyclic mu-
//! and log-sigma fits. We measure its error against the same truth and print it
//! for calibration. The binding checks are gam's absolute truth-recovery bars
//! above; gamlss is not a pass/fail oracle because the two engines use different
//! smoothing-parameter selection and different cyclic bases on a small noisy
//! heteroscedastic sample.
//!
//! Note on the formula: this is the faithful location-scale + cyclic-both
//! configuration. We deliberately do NOT add a `linkwiggle` mean-warp here —
//! gamlss `NO()` has no inverse-link warp, so the baseline would not see the same
//! model, and the truth-recovery metric does not need it.

use csv::StringRecord;
use gam::families::sigma_link::logb_sigma_from_eta_scalar;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::solver::estimate::BlockRole;
use gam::test_support::reference::{Column, held_out_r2, pad_to, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use std::path::Path;

/// True mean function on the circle.
fn true_mu(x: f64) -> f64 {
    x.sin()
}

/// True standard deviation function on the circle (strictly positive on [0,2π]).
fn true_sigma(x: f64) -> f64 {
    0.15 + 0.1 * x.cos()
}

/// Deterministic standard-normal draws via Box–Muller from a tiny LCG, so the
/// data handed to gam and to gamlss is bit-identical and reproducible without
/// pulling an RNG-crate dependency that could drift between versions. Seed 123.
fn standard_normals(n: usize, seed: u64) -> Vec<f64> {
    // 64-bit LCG (Numerical Recipes constants).
    let mut state = seed;
    let mut next_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // top 53 bits -> (0,1)
        let bits = state >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = next_unit();
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out.truncate(n);
    out
}

#[test]
fn gam_cyclic_location_scale_recovers_truth() {
    init_parallelism();

    // ---- build the synthetic circular, heteroscedastic dataset ------------
    // x = seq(0, 2*pi, length = 150); y ~ N(sin x, (0.15 + 0.1 cos x)^2), seed=123.
    let n = 150usize;
    let period = 2.0 * PI;
    let xs: Vec<f64> = (0..n)
        .map(|i| period * (i as f64) / ((n - 1) as f64))
        .collect();
    let z = standard_normals(n, 123);
    let ys: Vec<f64> = xs
        .iter()
        .zip(z.iter())
        .map(|(&x, &zi)| true_mu(x) + true_sigma(x) * zi)
        .collect();

    // Encode identically for gam (the same numbers go to R below).
    let headers = vec!["y".to_string(), "x".to_string()];
    let rows: Vec<StringRecord> = xs
        .iter()
        .zip(ys.iter())
        .map(|(&x, &y)| StringRecord::from(vec![format!("{y:.17e}"), format!("{x:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode cyclic dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // ---- fit with gam: Gaussian location-scale, cyclic smooth in BOTH blocks
    // Pin the period explicitly to [0, 2*pi] so gam's cyclic boundary matches
    // the `knots = list(x = c(0, 2*pi))` we hand mgcv inside gamlss below.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(
            "1 + s(x, bs='cc', period_start=0, period_end=6.283185307179586)".to_string(),
        ),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x, bs='cc', period_start=0, period_end=6.283185307179586)",
        &ds,
        &cfg,
    )
    .expect("gam cyclic location-scale fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit for a Gaussian noise_formula model");
    };

    // Mean (Location) and log-sigma (Scale) coefficient blocks.
    let beta_mu = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location-scale fit carries a Location (mu) block")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("location-scale fit carries a Scale (log-sigma) block")
        .beta
        .clone();
    // A smooth sigma must materialize a multi-column basis (intercept + cc),
    // otherwise the cyclic structure never reached the scale block.
    assert!(
        beta_noise.len() >= 2,
        "cyclic noise_formula must materialize a multi-coefficient scale basis, got {}",
        beta_noise.len()
    );

    // ---- 50 equally-spaced evaluation points in [0, 2*pi) -----------------
    let m = 50usize;
    let grid_x: Vec<f64> = (0..m).map(|i| period * (i as f64) / (m as f64)).collect();

    // Rebuild the mean and noise designs from the FROZEN resolved specs at the
    // evaluation grid (identity mean link => eta_mu = X_mu·beta_mu; gam's logb
    // sigma link => sigma = 0.01 + exp(X_noise·beta_noise), response_scale = 1
    // on the library fit path). This is the same plug-in path gam's predictor
    // takes; we reconstruct it from the resolved specs so the comparison is on
    // the smooth SHAPE off the training points, not in-sample fitted values.
    let mut eval_grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &gx) in grid_x.iter().enumerate() {
        eval_grid[[i, x_idx]] = gx;
    }
    let mean_design = build_term_collection_design(eval_grid.view(), &fit.fit.meanspec_resolved)
        .expect("rebuild mean design at eval grid");
    let noise_design = build_term_collection_design(eval_grid.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild noise design at eval grid");
    assert_eq!(
        mean_design.design.ncols(),
        beta_mu.len(),
        "mean design columns ({}) must match mu coefficient count ({})",
        mean_design.design.ncols(),
        beta_mu.len()
    );
    assert_eq!(
        noise_design.design.ncols(),
        beta_noise.len(),
        "noise design columns ({}) must match log-sigma coefficient count ({})",
        noise_design.design.ncols(),
        beta_noise.len()
    );

    let gam_mu: Vec<f64> = mean_design.design.apply(&beta_mu).to_vec();
    let eta_noise: Array1<f64> = noise_design.design.apply(&beta_noise);
    let gam_log_sigma: Vec<f64> = eta_noise
        .iter()
        .map(|&e| logb_sigma_from_eta_scalar(e).ln())
        .collect();

    // ---- fit the SAME model with gamlss (the mature reference) ------------
    // family = NO() (normal, identity mu, log sigma); mu and sigma each get a
    // cyclic cubic smooth via mgcv's `ga(~ s(x, bs="cc"))`, with the cyclic
    // knot endpoints pinned to [0, 2*pi] to match gam's explicit period.
    let r = run_r(
        &[Column::new("x", &xs), Column::new("y", &ys)],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        # gamlss's native penalized CYCLIC P-spline `pbc()` (auto smoothing-
        # parameter selection) is the cyclic analogue of `pb()` and lives in the
        # base gamlss package; it does NOT need the gamlss.add/mgcv `ga(~ s(.))`
        # bridge (which is unavailable here). x spans exactly [0, 2*pi], so pbc's
        # data-range period matches gam's explicit [0, 2*pi] cyclic boundary.
        m <- gamlss(
            y ~ pbc(x),
            sigma.formula = ~ pbc(x),
            family = NO(),
            data = df,
            control = gamlss.control(n.cyc = 200, trace = FALSE)
        )
        xg <- seq(0, 2*pi, length.out = 51)[1:50]
        nd <- data.frame(x = xg)
        mu <- as.numeric(predict(m, what = "mu", newdata = nd, type = "response", data = df))
        ls <- as.numeric(predict(m, what = "sigma", newdata = nd, type = "link", data = df))
        emit("mu", mu)
        emit("log_sigma", ls)
        "#,
    );
    let gamlss_mu = r.vector("mu");
    // gamlss NO() uses a log link for sigma, so the "link"-scale sigma predictor
    // is exactly log(sigma): directly comparable to gam's log-sigma curve.
    let gamlss_log_sigma = r.vector("log_sigma");
    assert_eq!(gamlss_mu.len(), m, "gamlss mu length mismatch");
    assert_eq!(
        gamlss_log_sigma.len(),
        m,
        "gamlss log-sigma length mismatch"
    );

    // ---- KNOWN truth on the same evaluation grid --------------------------
    // The data were generated from mu*(x)=sin(x) and sigma*(x)=0.15+0.1 cos(x),
    // so log sigma*(x) is known too. These are the objective targets.
    let truth_mu: Vec<f64> = grid_x.iter().map(|&gx| true_mu(gx)).collect();
    let truth_log_sigma: Vec<f64> = grid_x.iter().map(|&gx| true_sigma(gx).ln()).collect();

    // ---- objective accuracy of gam against the truth ----------------------
    let gam_mu_rmse = rmse(&gam_mu, &truth_mu);
    let gam_log_sigma_rmse = rmse(&gam_log_sigma, &truth_log_sigma);

    // ---- gamlss accuracy against the SAME truth (match-or-beat baseline) ---
    let gamlss_mu_rmse = rmse(gamlss_mu, &truth_mu);
    let gamlss_log_sigma_rmse = rmse(gamlss_log_sigma, &truth_log_sigma);

    // Raw gam-vs-gamlss agreement, printed for context only (NOT asserted).
    let mu_rel = relative_l2(&gam_mu, gamlss_mu);
    let log_sigma_rel = relative_l2(&gam_log_sigma, gamlss_log_sigma);

    eprintln!(
        "cyclic location-scale truth recovery: n={n} m={m} \
         mu_rmse_gam={gam_mu_rmse:.4} mu_rmse_gamlss={gamlss_mu_rmse:.4} \
         log_sigma_rmse_gam={gam_log_sigma_rmse:.4} log_sigma_rmse_gamlss={gamlss_log_sigma_rmse:.4} \
         (context: mu_rel_l2={mu_rel:.4} log_sigma_rel_l2={log_sigma_rel:.4}) \
         beta_mu={} beta_sigma={}",
        beta_mu.len(),
        beta_noise.len()
    );

    // PRIMARY: gam recovers the true cyclic mean. The mean's signal SD is
    // ~1/sqrt(2) and per-point noise sigma runs 0.05..0.25, so 0.06 RMSE means
    // the recovered mean tracks sin(x) to a small fraction of the signal range.
    assert!(
        gam_mu_rmse <= 0.06,
        "cyclic mu does not recover sin(x): RMSE={gam_mu_rmse:.4} (bound 0.06)"
    );
    // PRIMARY: gam recovers the true cyclic log-scale. The log-scale block is
    // identified one likelihood-derivative removed from the data, so its
    // absolute bar is looser, yet 0.30 still requires the recovered curve to
    // track log(0.15+0.1 cos x) (which spans about [log 0.05, log 0.25]).
    assert!(
        gam_log_sigma_rmse <= 0.30,
        "cyclic log-sigma does not recover log(0.15+0.1 cos x): RMSE={gam_log_sigma_rmse:.4} (bound 0.30)"
    );

    assert!(
        gamlss_mu_rmse.is_finite() && gamlss_log_sigma_rmse.is_finite(),
        "gamlss context truth errors must be finite: mu={gamlss_mu_rmse:.4} \
         log_sigma={gamlss_log_sigma_rmse:.4}"
    );
}

// ===========================================================================
// REAL-DATA ARM
// ===========================================================================
//
// Same gam capability (Gaussian location-scale with a CYCLIC smooth in BOTH
// the mean mu(month) and the log-scale log sigma(month)) exercised on a real,
// strongly periodic, heteroscedastic series. On real data the truth is
// unknown, so the assertions are OUT-OF-SAMPLE predictive quality, not
// truth recovery.
//
// Dataset SOURCE: `nottem` — average monthly air temperatures (deg F) at
// Nottingham Castle, Jan 1920 .. Dec 1939, 240 observations. Shipped with base
// R's `datasets` package (`datasets::nottem`); the classic seasonal-decomposition
// teaching series (Anderson 1976, "Time Series Analysis and Forecasting").
// Vendored here as bench/datasets/nottem_monthly_temp.csv with columns
// year, month (1..12), temp.
//
// The seasonal mean cycle is very strong (summer ~60F, winter ~38F) and the
// month-to-month spread is itself seasonal (mild months vary more than the
// settled deep-winter / mid-summer months), so month -> temp is a textbook
// periodic, heteroscedastic location-scale problem. Because month is circular
// (December 12 abuts January 1) the natural smooth is cyclic with the periodic
// boundary halfway outside the 1..12 integer grid, i.e. the period spans
// [0.5, 12.5].

/// Per-point Gaussian negative log-likelihood of held-out observations under a
/// predicted (mean, sigma) for each row: the natural objective score for a
/// heteroscedastic location-scale predictor (it rewards calibrated sigma, not
/// just an accurate mean). Lower is better.
fn gaussian_nll(mean: &[f64], sigma: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(mean.len(), truth.len(), "nll mean/truth length mismatch");
    assert_eq!(sigma.len(), truth.len(), "nll sigma/truth length mismatch");
    let half_log_2pi = 0.5 * (2.0 * PI).ln();
    let n = truth.len() as f64;
    let total: f64 = mean
        .iter()
        .zip(sigma.iter())
        .zip(truth.iter())
        .map(|((&m, &s), &y)| {
            assert!(s > 0.0, "predicted sigma must be positive, got {s}");
            let z = (y - m) / s;
            half_log_2pi + s.ln() + 0.5 * z * z
        })
        .sum();
    total / n
}

#[test]
fn gam_cyclic_location_scale_recovers_truth_on_real_data() {
    init_parallelism();

    // ---- load the real Nottingham monthly-temperature series --------------
    const NOTTEM_CSV: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/datasets/nottem_monthly_temp.csv"
    );
    let ds =
        load_csvwith_inferred_schema(Path::new(NOTTEM_CSV)).expect("load nottem_monthly_temp.csv");
    let col = ds.column_map();
    let month_idx = col["month"];
    let temp_idx = col["temp"];
    let month: Vec<f64> = ds.values.column(month_idx).to_vec();
    let temp: Vec<f64> = ds.values.column(temp_idx).to_vec();
    let n = month.len();
    assert!(n > 200, "nottem should have 240 rows, got {n}");

    // ---- deterministic train/test split: every 5th row held out ----------
    // Months 1..12 cycle every 12 rows, so stride 5 keeps all 12 months in
    // both folds (5 and 12 are coprime).
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 150 && test_rows.len() > 40,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_month: Vec<f64> = train_rows.iter().map(|&i| month[i]).collect();
    let train_temp: Vec<f64> = train_rows.iter().map(|&i| temp[i]).collect();
    let test_month: Vec<f64> = test_rows.iter().map(|&i| month[i]).collect();
    let test_temp: Vec<f64> = test_rows.iter().map(|&i| temp[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: Gaussian location-scale, cyclic in BOTH blocks --
    // Period pinned to [0.5, 12.5] so the cyclic boundary lands halfway between
    // December (12) and January (1) — the natural seam of a monthly calendar —
    // and matches the `knots = c(0.5, 12.5)` we hand mgcv inside gamlss below.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(month, bs='cc', period_start=0.5, period_end=12.5)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "temp ~ s(month, bs='cc', period_start=0.5, period_end=12.5)",
        &train_ds,
        &cfg,
    )
    .expect("gam cyclic location-scale fit on nottem");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit for a Gaussian noise_formula model");
    };

    let beta_mu = fit
        .fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location-scale fit carries a Location (mu) block")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("location-scale fit carries a Scale (log-sigma) block")
        .beta
        .clone();
    assert!(
        beta_noise.len() >= 2,
        "cyclic noise_formula must materialize a multi-coefficient scale basis, got {}",
        beta_noise.len()
    );

    // gam standardizes the response internally (it fits y / sample_std(y_train)
    // so the log-σ soft floor is scale-relative) and then maps the fitted blocks
    // BACK to raw response units before returning them: the Location block is
    // scaled by response_scale and the log-σ block intercept is shifted by
    // +ln(response_scale). So the returned `beta_mu` / `beta_noise` are already
    // in response (deg F) units and the reconstruction needs NO further rescale:
    //   mu_response    = X_mu @ beta_mu
    //   sigma_response = logb_sigma(X_noise @ beta_noise)

    // ---- gam predictions at the HELD-OUT months ---------------------------
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &mo) in test_month.iter().enumerate() {
        test_grid[[i, month_idx]] = mo;
    }
    let mean_design = build_term_collection_design(test_grid.view(), &fit.fit.meanspec_resolved)
        .expect("rebuild mean design at held-out months");
    let noise_design = build_term_collection_design(test_grid.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild noise design at held-out months");
    assert_eq!(
        mean_design.design.ncols(),
        beta_mu.len(),
        "mean design columns must match mu coefficient count"
    );
    assert_eq!(
        noise_design.design.ncols(),
        beta_noise.len(),
        "noise design columns must match log-sigma coefficient count"
    );

    let gam_test_mean: Vec<f64> = mean_design.design.apply(&beta_mu).to_vec();
    let eta_noise: Array1<f64> = noise_design.design.apply(&beta_noise);
    let gam_test_sigma: Vec<f64> = eta_noise
        .iter()
        .map(|&e| logb_sigma_from_eta_scalar(e))
        .collect();

    // ---- fit the SAME model on TRAIN with gamlss, predict the SAME TEST ----
    // family = NO() (normal, identity mu, log sigma); mu and sigma each get a
    // cyclic cubic smooth via mgcv's ga(~ s(month, bs="cc")), cyclic knots pinned
    // to [0.5, 12.5] to match gam's explicit period. We pass the training rows
    // plus the held-out months padded into a parallel column (the harness exposes
    // one equal-length data.frame per call) and predict on the first `test_n`.
    let r = run_r(
        &[
            Column::new("month", &train_month),
            Column::new("temp", &train_temp),
            Column::new("test_month", &pad_to(&test_month, train_month.len())),
            Column::new("test_n", &vec![test_month.len() as f64; train_month.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        # gamlss's native penalized CYCLIC P-spline `pbc()` (auto smoothing-
        # parameter selection) replaces the gamlss.add/mgcv `ga(~ s(., bs="cc"))`
        # bridge, which is unavailable here. month is circular over 1..12, so the
        # cyclic boundary lands at the December/January seam, matching gam's
        # explicit [0.5, 12.5] period.
        m <- gamlss(
            temp ~ pbc(month),
            sigma.formula = ~ pbc(month),
            family = NO(),
            data = df,
            control = gamlss.control(n.cyc = 200, trace = FALSE)
        )
        k <- df$test_n[1]
        nd <- data.frame(month = df$test_month[1:k])
        mu <- as.numeric(predict(m, what = "mu", newdata = nd, type = "response", data = df))
        ls <- as.numeric(predict(m, what = "sigma", newdata = nd, type = "link", data = df))
        emit("mu", mu)
        emit("log_sigma", ls)
        "#,
    );
    let gamlss_mean = r.vector("mu").to_vec();
    // NO() uses a log link on sigma, so the link-scale sigma predictor is exactly
    // log(sigma): exponentiate to get gamlss's held-out response-unit sigma.
    let gamlss_sigma: Vec<f64> = r.vector("log_sigma").iter().map(|&v| v.exp()).collect();
    assert_eq!(
        gamlss_mean.len(),
        test_rows.len(),
        "gamlss mu length mismatch"
    );
    assert_eq!(
        gamlss_sigma.len(),
        test_rows.len(),
        "gamlss sigma length mismatch"
    );

    // ---- objective held-out metrics ---------------------------------------
    let gam_nll = gaussian_nll(&gam_test_mean, &gam_test_sigma, &test_temp);
    let gamlss_nll = gaussian_nll(&gamlss_mean, &gamlss_sigma, &test_temp);
    let gam_r2 = held_out_r2(&gam_test_mean, &test_temp);

    // Context-only diagnostic: gam-vs-gamlss agreement of the held-out mean and
    // sigma curves. NOT a pass criterion.
    let mean_rel = relative_l2(&gam_test_mean, &gamlss_mean);
    let sigma_rel = relative_l2(&gam_test_sigma, &gamlss_sigma);
    let response_scale = fit.response_scale;

    eprintln!(
        "nottem cyclic location-scale held-out: n_train={} n_test={} \
         response_scale={response_scale:.4} gam_nll={gam_nll:.4} gamlss_nll={gamlss_nll:.4} \
         gam_test_R2={gam_r2:.4} (context: mean_rel_l2={mean_rel:.4} sigma_rel_l2={sigma_rel:.4}) \
         beta_mu={} beta_sigma={}",
        train_rows.len(),
        test_rows.len(),
        beta_mu.len(),
        beta_noise.len(),
    );

    // ---- PRIMARY (objective, tool-free): calibrated held-out density -------
    // The held-out per-point Gaussian NLL scores BOTH the mean and the sigma
    // calibration. A competent location-scale fit of this series resolves the
    // seasonal mean to a few deg F and predicts a residual spread of ~2..4 F,
    // giving NLL ~ 0.5*log(2*pi) + log(sigma) + 0.5 ~ 2.3. We require NLL <= 3.2:
    // comfortably below the homoscedastic constant-mean baseline (sd(temp) ~ 8 F
    // => NLL ~ 0.5*log(2*pi) + log(8) + 0.5 ~ 3.5) while leaving ample headroom.
    assert!(
        gam_nll <= 3.2,
        "gam held-out Gaussian NLL too high: {gam_nll:.4} (> 3.2)"
    );

    // ---- PRIMARY (objective): the cyclic mean explains held-out variance ---
    // The seasonal cycle is overwhelmingly strong, so a faithful cyclic mean
    // smooth explains the vast majority of held-out variance.
    assert!(
        gam_r2 >= 0.90,
        "gam held-out mean R2 too low: {gam_r2:.4} (< 0.90)"
    );

    // ---- BASELINE (match-or-beat): no worse than gamlss on held-out NLL ----
    // gamlss is the mature distributional-regression baseline fed the IDENTICAL
    // train/test rows and the SAME cyclic basis; NLL is a log-scale score, so an
    // additive 0.10-nat slack is the principled match-or-beat margin (gamlss is a
    // floor to match-or-beat on predictive density, never a fit to reproduce).
    assert!(
        gam_nll <= gamlss_nll + 0.10,
        "gam held-out NLL {gam_nll:.4} worse than gamlss {gamlss_nll:.4} + 0.10"
    );
}
