//! #1471 validation baseline: family/smoother configurations confirmed correct
//! against KNOWN ground truth and the mgcv oracle on current `main`, pinned here
//! as REAL CI regression guards so the documented correctness cannot silently
//! drift.
//!
//! Each arm corresponds to a row of the issue's validation table. The method is
//! the same throughout: simulate `y = f(·) + noise` for a KNOWN `f`, fit gam and
//! mgcv on BYTE-IDENTICAL data (identical rows, REML), and score each engine's
//! recovery of `f` on a dense evaluation grid. The objective metric is recovery
//! error against the analytic truth (RMSE / R²), never "gam reproduces mgcv's
//! fitted output". mgcv is a mature MATCH-OR-BEAT baseline on that same
//! truth-recovery metric, not a fit to imitate.
//!
//! Documented numbers from the issue (gam 0.1.222 `b172e3ccd` vs mgcv 1.9-4):
//!   * Tweedie `s(x)` log link        : RMSE ratio gam/mgcv ≈ 1.13
//!   * `s(x) + x` (smooth + linear)    : ratio ≈ 1.05
//!   * `s(x)+s(z)`, corr(x,z)=0.90     : ratio ≈ 1.03 (concurvity)
//!   * cyclic tensor te(t,x,           : periodicity gap 0.0000; RMSE 0.018,
//!     boundary=['periodic','clamped'])  beating default `te` (0.033)
//!
//! Tolerances here are deliberately looser than the documented point estimates
//! (the documented value is the seed-specific realization; the asserted bound is
//! the principled regression floor) but never weaker than "gam matches-or-beats
//! mgcv on truth recovery to within a small Monte-Carlo margin". A real
//! regression in any of these paths — a dropped penalty, a broken cyclic seam, a
//! concurvity-driven over/under-smooth — fails the corresponding arm loudly.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, r2, rmse, run_r};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::TAU;

/// Encode (header, column-of-f64) pairs into a gam dataset. All baseline arms
/// here are purely numeric, so a single helper keeps each test focused on the
/// statistics rather than the row plumbing.
fn encode(cols: &[(&str, &[f64])]) -> EncodedDataset {
    let n = cols[0].1.len();
    for (name, c) in cols {
        assert_eq!(c.len(), n, "column {name} length mismatch");
    }
    let headers: Vec<String> = cols.iter().map(|(h, _)| (*h).to_string()).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(cols.iter().map(|(_, c)| c[i].to_string()).collect::<Vec<_>>()))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode baseline dataset")
}

/// Rebuild gam's frozen design at arbitrary covariate rows and return the linear
/// predictor `η = Xβ`. Each column index is supplied so the caller controls the
/// covariate layout; unlisted columns stay zero (their terms drop out under an
/// identity/log link).
fn gam_eta(
    fit: &gam::StandardFitResult,
    width: usize,
    assignments: &[(usize, &[f64])],
) -> Vec<f64> {
    let m = assignments[0].1.len();
    let mut pts = Array2::<f64>::zeros((m, width));
    for (idx, vals) in assignments {
        assert_eq!(vals.len(), m, "assignment length mismatch");
        for (r, &v) in vals.iter().enumerate() {
            pts[[r, *idx]] = v;
        }
    }
    let d = build_term_collection_design(pts.view(), &fit.resolvedspec)
        .expect("rebuild gam design at eval rows");
    d.design.apply(&fit.fit.beta).to_vec()
}

// ===========================================================================
// Arm 1 — Tweedie s(x), log link, power estimated.
// ===========================================================================

/// The Tweedie compound-Poisson-gamma response with a log-link smooth must
/// RECOVER a known log-mean curve and match-or-beat mgcv's `tw()` on
/// truth-recovery RMSE. Documented: gam/mgcv RMSE ratio ≈ 1.13, no EDF stall.
#[test]
fn tweedie_log_smooth_recovers_truth_and_matches_mgcv() {
    init_parallelism();

    // True log-mean is a smooth nonlinear curve; mean = exp(eta) so the response
    // is strictly positive with zero inflation (the defining Tweedie feature).
    let true_eta = |x: f64| 0.6 * (2.0 * x).sin() + 0.4 * x - 0.5;
    let n = 400usize;
    let p_true = 1.5_f64;
    let phi = 0.6_f64;

    let mut rng = StdRng::seed_from_u64(147_001);
    let unif = Uniform::new(0.0_f64, 3.0).expect("uniform x");
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = unif.sample(&mut rng);
        let mu = true_eta(xi).exp();
        // Compound Poisson–gamma draw for Tweedie 1<p<2 (Jørgensen): N ~ Pois(λ),
        // y = sum of N gamma jumps. Standard reparameterization in (μ, φ, p).
        let lambda = mu.powf(2.0 - p_true) / (phi * (2.0 - p_true));
        let gamma_shape = (2.0 - p_true) / (p_true - 1.0);
        let gamma_scale = phi * (p_true - 1.0) * mu.powf(p_true - 1.0);
        let n_jumps = poisson_sample(lambda, &mut rng);
        let mut yi = 0.0;
        for _ in 0..n_jumps {
            yi += gamma_sample(gamma_shape, gamma_scale, &mut rng);
        }
        x.push(xi);
        y.push(yi);
    }
    let zeros = y.iter().filter(|&&v| v == 0.0).count();
    assert!(zeros > 0, "Tweedie 1<p<2 must be zero-inflated; got {zeros} zeros");

    let ds = encode(&[("x", &x), ("y", &y)]);
    let x_idx = ds.column_map()["x"];
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("tweedie".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=10)", &ds, &cfg).expect("gam tweedie fit");
    let FitResult::Standard(fit) = result else {
        panic!("Tweedie(log) is a scalar GLM family => expected FitResult::Standard");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");
    assert!(
        gam_edf > 1.0 && gam_edf < 15.0,
        "tweedie smooth edf out of sane range (EDF stall?): {gam_edf:.3}"
    );

    // Dense evaluation grid: recover the MEAN curve mu = exp(eta) vs analytic
    // truth on points spanning the support.
    let grid: Vec<f64> = (0..120).map(|i| 3.0 * i as f64 / 119.0).collect();
    let truth_mu: Vec<f64> = grid.iter().map(|&xg| true_eta(xg).exp()).collect();
    let gam_mu: Vec<f64> = gam_eta(&fit, width, &[(x_idx, &grid)])
        .iter()
        .map(|e| e.exp())
        .collect();

    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x, k = 10), data = df, family = tw(), method = "REML")
            xg <- seq(0, 3, length.out = {ng})
            emit("mu", as.numeric(predict(m, newdata = data.frame(x = xg), type = "response")))
            emit("edf", sum(m$edf))
            emit("p", as.numeric(m$family$getTheta(TRUE)))
            "#,
            ng = grid.len(),
        ),
    );
    let mgcv_mu = r.vector("mu");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_mu.len(), grid.len(), "mgcv mu length mismatch");

    let gam_err = rmse(&gam_mu, &truth_mu);
    let mgcv_err = rmse(mgcv_mu, &truth_mu);
    let signal = truth_mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - truth_mu.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "tweedie s(x) log: n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_rmse_vs_truth={gam_err:.5} mgcv_rmse_vs_truth={mgcv_err:.5} \
         ratio={:.3} signal_range={signal:.4}",
        gam_err / mgcv_err.max(1e-12)
    );

    // PRIMARY: gam recovers the mean curve well below the signal scale.
    assert!(
        gam_err < 0.30 * signal,
        "tweedie smooth failed to recover the log-mean curve: rmse={gam_err:.5} \
         (signal {signal:.4})"
    );
    // MATCH-OR-BEAT: gam's truth-recovery RMSE no worse than 1.25× mgcv's. The
    // documented realization is ≈1.13; the 1.25 bound is the regression floor.
    assert!(
        gam_err <= mgcv_err * 1.25,
        "gam tweedie recovery {gam_err:.5} worse than mgcv {mgcv_err:.5} * 1.25"
    );
}

// ===========================================================================
// Arm 2 — s(x) + x : smooth plus a parametric linear term in the SAME variable.
// ===========================================================================

/// A smooth plus an explicit parametric linear term in the same covariate must
/// recover the combined truth and match-or-beat mgcv. This exercises the
/// nullspace/identifiability handling: the smooth's linear nullspace overlaps
/// the parametric `x`, and the fit must still recover f(x)+βx without aliasing.
/// Documented: gam/mgcv RMSE ratio ≈ 1.05.
#[test]
fn smooth_plus_linear_same_var_recovers_truth_and_matches_mgcv() {
    init_parallelism();

    // Truth = a genuine curve PLUS a strong linear trend in the same x.
    let truth = |x: f64| 1.2 * x + 0.8 * (1.7 * x).sin();
    let n = 300usize;
    let sigma = 0.20_f64;
    let mut rng = StdRng::seed_from_u64(147_002);
    let unif = Uniform::new(0.0_f64, 4.0).expect("uniform x");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = unif.sample(&mut rng);
        x.push(xi);
        y.push(truth(xi) + noise.sample(&mut rng));
    }

    let ds = encode(&[("x", &x), ("y", &y)]);
    let x_idx = ds.column_map()["x"];
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=10) + linear(x)", &ds, &cfg).expect("gam s(x)+x fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x)+linear(x)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    let grid: Vec<f64> = (0..120).map(|i| 4.0 * i as f64 / 119.0).collect();
    let truth_grid: Vec<f64> = grid.iter().map(|&xg| truth(xg)).collect();
    let gam_grid = gam_eta(&fit, width, &[(x_idx, &grid)]);

    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x, k = 10) + x, data = df, method = "REML")
            xg <- seq(0, 4, length.out = {ng})
            emit("pred", as.numeric(predict(m, newdata = data.frame(x = xg))))
            emit("edf", sum(m$edf))
            "#,
            ng = grid.len(),
        ),
    );
    let mgcv_grid = r.vector("pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_grid.len(), grid.len(), "mgcv pred length mismatch");

    let gam_err = rmse(&gam_grid, &truth_grid);
    let mgcv_err = rmse(mgcv_grid, &truth_grid);
    let gam_r2 = r2(&gam_grid, &truth_grid);
    eprintln!(
        "s(x)+x: n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_rmse_vs_truth={gam_err:.5} mgcv_rmse_vs_truth={mgcv_err:.5} \
         ratio={:.3} gam_r2={gam_r2:.4}",
        gam_err / mgcv_err.max(1e-12)
    );

    // PRIMARY: near-perfect recovery of the combined truth (R² well above 0).
    assert!(
        gam_r2 > 0.95,
        "s(x)+x failed to recover the combined truth: R²={gam_r2:.4}"
    );
    // MATCH-OR-BEAT: documented ratio ≈1.05; floor 1.15.
    assert!(
        gam_err <= mgcv_err * 1.15,
        "gam s(x)+x recovery {gam_err:.5} worse than mgcv {mgcv_err:.5} * 1.15"
    );
}

// ===========================================================================
// Arm 3 — s(x)+s(z) with corr(x,z)=0.90 : concurvity.
// ===========================================================================

/// Two additive smooths on STRONGLY correlated covariates (concurvity, corr=0.9)
/// must still recover the additive truth f(x)+g(z) and match-or-beat mgcv. High
/// concurvity is the classic failure mode for additive smoothers (the component
/// curves become weakly identified); recovering the SUM on a grid is the robust
/// objective metric. Documented: gam/mgcv RMSE ratio ≈ 1.03.
#[test]
fn concurvity_two_smooths_corr_090_recovers_truth_and_matches_mgcv() {
    init_parallelism();

    let f = |x: f64| (1.5 * x).sin();
    let g = |z: f64| 0.5 * z * z - 0.6 * z;
    let n = 400usize;
    let sigma = 0.20_f64;
    let rho = 0.90_f64;
    let mut rng = StdRng::seed_from_u64(147_003);
    let std_normal = Normal::new(0.0, 1.0).expect("normal");

    // z = rho*x + sqrt(1-rho^2)*eps  =>  corr(x,z) = rho exactly (both unit var).
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let noise = Normal::new(0.0, sigma).expect("normal noise");
    for _ in 0..n {
        let xi: f64 = std_normal.sample(&mut rng);
        let eps: f64 = std_normal.sample(&mut rng);
        let zi = rho * xi + (1.0 - rho * rho).sqrt() * eps;
        x.push(xi);
        z.push(zi);
        y.push(f(xi) + g(zi) + noise.sample(&mut rng));
    }
    // Confirm the realized correlation really is near 0.90.
    let corr = pearson(&x, &z);
    assert!(
        (corr - rho).abs() < 0.05,
        "realized corr(x,z)={corr:.3} should be ≈{rho:.2}"
    );

    let ds = encode(&[("x", &x), ("z", &z), ("y", &y)]);
    let cm = ds.column_map();
    let (x_idx, z_idx) = (cm["x"], cm["z"]);
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=10) + s(z, k=10)", &ds, &cfg).expect("gam s(x)+s(z) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x)+s(z)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluate the recovered ADDITIVE SUM at the training rows vs the noiseless
    // truth f(x)+g(z): the sum is identified even when the components are not.
    let truth_sum: Vec<f64> = (0..n).map(|i| f(x[i]) + g(z[i])).collect();
    let gam_sum = gam_eta(&fit, width, &[(x_idx, &x), (z_idx, &z)]);

    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x, k = 10) + s(z, k = 10), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_sum = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_sum.len(), n, "mgcv fitted length mismatch");

    // Both engines fit an intercept; center sum and truth before comparing so the
    // additive identifiability constant does not contaminate the metric.
    let center = |v: &[f64]| -> Vec<f64> {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| x - m).collect()
    };
    let truth_c = center(&truth_sum);
    let gam_c = center(&gam_sum);
    let mgcv_c = center(mgcv_sum);

    let gam_err = rmse(&gam_c, &truth_c);
    let mgcv_err = rmse(&mgcv_c, &truth_c);
    let gam_r2 = r2(&gam_c, &truth_c);
    eprintln!(
        "concurvity s(x)+s(z) corr={corr:.3}: n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_rmse_vs_truth={gam_err:.5} mgcv_rmse_vs_truth={mgcv_err:.5} \
         ratio={:.3} gam_r2={gam_r2:.4}",
        gam_err / mgcv_err.max(1e-12)
    );

    // PRIMARY: the additive SUM is recovered despite high concurvity.
    assert!(
        gam_r2 > 0.90,
        "concurvity fit failed to recover the additive sum: R²={gam_r2:.4}"
    );
    // MATCH-OR-BEAT: documented ratio ≈1.03; floor 1.15.
    assert!(
        gam_err <= mgcv_err * 1.15,
        "gam concurvity recovery {gam_err:.5} worse than mgcv {mgcv_err:.5} * 1.15"
    );
}

// ===========================================================================
// Arm 4 — cyclic tensor te(t,x, boundary=['periodic','clamped']).
// ===========================================================================

/// A mixed cyclic/clamped tensor smooth `te(t, x)` with a PERIODIC margin in `t`
/// and a CLAMPED margin in `x` must (a) genuinely enforce the periodic seam in
/// `t` — fitted surface at t=0 equals t=period — and (b) recover the periodic
/// truth at least as well as mgcv's `te(bs=c("cc","cr"))`. Documented:
/// periodicity gap ≈ 0.0000, RMSE 0.018, beating default `te` (0.033).
#[test]
fn cyclic_tensor_periodic_clamped_wraps_and_matches_mgcv() {
    init_parallelism();

    // f(t,x): periodic in t over [0,2π), smooth (non-periodic) in x over [0,1].
    let truth = |t: f64, x: f64| (t).sin() + 0.6 * (2.0 * t).cos() * x + 0.8 * x * x;
    const GT: usize = 18;
    const GX: usize = 18;
    let n = GT * GX;
    let sigma = 0.05_f64;
    let mut rng = StdRng::seed_from_u64(147_004);
    let noise = Normal::new(0.0, sigma).expect("normal");

    let mut t = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..GT {
        let ti = TAU * (i as f64) / (GT as f64); // [0,2π), seam never duplicated
        for j in 0..GX {
            let xj = j as f64 / (GX as f64 - 1.0); // [0,1] clamped
            t.push(ti);
            x.push(xj);
            y.push(truth(ti, xj) + noise.sample(&mut rng));
        }
    }

    let ds = encode(&[("t", &t), ("x", &x), ("y", &y)]);
    let cm = ds.column_map();
    let (t_idx, x_idx) = (cm["t"], cm["x"]);
    let width = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // gam analog of mgcv te(bs=c("cc","cr")): periodic t margin, clamped x margin.
    let formula =
        "y ~ te(t, x, boundary=['periodic','clamped'], period=[2*pi, None], k=8)";
    let result = fit_from_formula(formula, &ds, &cfg).expect("gam cyclic tensor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the cyclic tensor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // ---- STRUCTURE: periodic seam continuity in t (the load-bearing claim) ---
    // Fit at t=0 must equal fit at t=2π for every x: a genuine cyclic margin
    // wraps exactly; a broken seam shows a discontinuity. Evaluate on a fine x
    // sweep at both seam ends.
    let xs: Vec<f64> = (0..25).map(|i| i as f64 / 24.0).collect();
    let zeros = vec![0.0_f64; xs.len()];
    let twos = vec![TAU; xs.len()];
    let fit_at_0 = gam_eta(&fit, width, &[(t_idx, &zeros), (x_idx, &xs)]);
    let fit_at_2pi = gam_eta(&fit, width, &[(t_idx, &twos), (x_idx, &xs)]);
    let seam_gap = fit_at_0
        .iter()
        .zip(&fit_at_2pi)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // ---- truth recovery on a held-out interpolation grid (off training nodes) -
    const GIT: usize = 13;
    const GIX: usize = 13;
    let mut gt = Vec::new();
    let mut gx = Vec::new();
    let mut gtruth = Vec::new();
    for i in 0..GIT {
        let ti = TAU * (i as f64 + 0.5) / (GIT as f64);
        for j in 0..GIX {
            let xj = (j as f64 + 0.5) / (GIX as f64);
            gt.push(ti);
            gx.push(xj);
            gtruth.push(truth(ti, xj));
        }
    }
    let gam_grid = gam_eta(&fit, width, &[(t_idx, &gt), (x_idx, &gx)]);

    // mgcv baseline: te(bs=c("cc","cr")) with the cyclic t knots pinned to the
    // [0,2π] support; predicts on the SAME interpolation grid (scored vs truth).
    let mut t_all = t.clone();
    t_all.extend_from_slice(&gt);
    let mut x_all = x.clone();
    x_all.extend_from_slice(&gx);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, gt.len()));
    let mut w = vec![1.0_f64; n];
    w.extend(std::iter::repeat_n(0.0, gt.len()));

    let r = run_r(
        &[
            Column::new("t", &t_all),
            Column::new("x", &x_all),
            Column::new("y", &y_all),
            Column::new("w", &w),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$w > 0, ]
        m <- gam(y ~ te(t, x, bs = c("cc", "cr"), k = c(8, 8)),
                 data = train, method = "REML",
                 knots = list(t = c(0, 2 * pi)))
        grid <- df[df$w == 0, ]
        emit("grid_pred", as.numeric(predict(m, newdata = grid)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_grid = r.vector("grid_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_grid.len(), gtruth.len(), "mgcv grid length mismatch");

    let gam_err = rmse(&gam_grid, &gtruth);
    let mgcv_err = rmse(mgcv_grid, &gtruth);
    eprintln!(
        "cyclic tensor te(cc,cr): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         seam_gap={seam_gap:.2e} gam_rmse_vs_truth={gam_err:.5} \
         mgcv_rmse_vs_truth={mgcv_err:.5} ratio={:.3}",
        gam_err / mgcv_err.max(1e-12)
    );

    // PRIMARY (structure): the periodic seam wraps to numerical zero. The
    // documented gap is 0.0000; a broken cyclic-basis closure (sign/threshold
    // bug) leaves a real discontinuity. 1e-6 is far below the signal (≈O(1)).
    assert!(
        seam_gap < 1e-6,
        "cyclic margin does not wrap: fit(t=0) vs fit(t=2π) max gap {seam_gap:.3e}"
    );
    // PRIMARY (recovery): recovers the periodic surface well below signal scale.
    let signal = gtruth.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - gtruth.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        gam_err < 0.10 * signal,
        "cyclic tensor failed to recover truth: rmse={gam_err:.5} (signal {signal:.4})"
    );
    // MATCH-OR-BEAT: no worse than 1.20× mgcv on truth recovery.
    assert!(
        gam_err <= mgcv_err * 1.20,
        "gam cyclic tensor recovery {gam_err:.5} worse than mgcv {mgcv_err:.5} * 1.20"
    );
}

// ===========================================================================
// Arm 5 — CI coverage under a NON-IDENTITY link (Poisson / log).
// ===========================================================================

/// Count intervals that bracket the truth.
fn covered(lower: &[f64], upper: &[f64], truth: &[f64]) -> usize {
    lower
        .iter()
        .zip(upper)
        .zip(truth)
        .filter(|((lo, hi), t)| **lo <= **t && **t <= **hi)
        .count()
}

/// A 95% confidence interval is only correct if it covers the truth ~95% of the
/// time. The existing Gaussian-identity coverage test exercises the trivial
/// Jacobian (dμ/dη ≡ 1); this arm exercises the NON-trivial log-link delta
/// method under a discrete Poisson response, where a wrong response-scale SE
/// transform would silently mis-cover. We draw many Poisson replicates around a
/// KNOWN log-mean, form gam's 95% response-scale mean intervals, and measure
/// empirical coverage against the truth — then assert (a) gam is calibrated to
/// nominal and (b) it covers at least as well as mgcv's `predict.gam(type=
/// "response", se.fit=TRUE)` intervals on the identical data.
#[test]
fn poisson_response_ci_is_calibrated_and_matches_mgcv() {
    init_parallelism();
    let n = 250usize;
    let replicates = 24usize;
    let nominal = 0.95_f64;

    // Shared design across replicates (sorted x); only the Poisson draw changes.
    let mut drng = StdRng::seed_from_u64(147_005);
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform x");
    let mut x: Vec<f64> = (0..n).map(|_| unif.sample(&mut drng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    // eta = 1.0 + 0.9*sin(2π x); mu = exp(eta) in ~[1.1, 6.7] (well away from 0,
    // so the Gaussian-approximate interval is a fair comparison for both tools).
    let mu_true: Vec<f64> = x.iter().map(|&v| (1.0 + 0.9 * (TAU * v).sin()).exp()).collect();

    let poisson_log = LikelihoodSpec::new(
        ResponseFamily::Poisson,
        InverseLink::Standard(StandardLink::Log),
    );

    let total = n * replicates;
    let mut gam_cov = 0usize;
    let mut mgcv_cov = 0usize;

    for rep in 0..replicates {
        let mut rng = StdRng::seed_from_u64(700 + rep as u64);
        let y: Vec<f64> = mu_true
            .iter()
            .map(|&m| poisson_sample(m, &mut rng) as f64)
            .collect();

        let ds = encode(&[("x", &x), ("y", &y)]);
        let cfg = FitConfig {
            family: Some("poisson".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula("y ~ s(x)", &ds, &cfg).expect("gam poisson fit");
        let FitResult::Standard(fit) = result else {
            panic!("poisson => FitResult::Standard");
        };
        let design = build_term_collection_design(ds.values.view(), &fit.resolvedspec)
            .expect("rebuild poisson design at training points");
        let dense = design.design.to_dense();
        let offset = Array1::<f64>::zeros(n);
        let pred = predict_gamwith_uncertainty(
            dense,
            fit.fit.beta.view(),
            offset.view(),
            poisson_log.clone(),
            &fit.fit,
            &PredictUncertaintyOptions {
                confidence_level: nominal,
                covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
                mean_interval_method: MeanIntervalMethod::Delta,
                includeobservation_interval: false,
                apply_bias_correction: false,
                edgeworth_one_sided: false,
                boundary_correction: false,
                ood_inflation: false,
                multi_point_joint: false,
                ..PredictUncertaintyOptions::default()
            },
        )
        .expect("gam poisson response-scale uncertainty");
        gam_cov += covered(&pred.mean_lower.to_vec(), &pred.mean_upper.to_vec(), &mu_true);

        let r = run_r(
            &[Column::new("x", &x), Column::new("y", &y)],
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x), data = df, family = poisson(), method = "REML")
            p <- predict(m, newdata = df, se.fit = TRUE, type = "response")
            z <- qnorm(0.975)
            emit("lower", as.numeric(p$fit - z * p$se.fit))
            emit("upper", as.numeric(p$fit + z * p$se.fit))
            "#,
        );
        mgcv_cov += covered(r.vector("lower"), r.vector("upper"), &mu_true);
    }

    let gam_coverage = gam_cov as f64 / total as f64;
    let mgcv_coverage = mgcv_cov as f64 / total as f64;
    let gam_err = (gam_coverage - nominal).abs();
    let mgcv_err = (mgcv_coverage - nominal).abs();
    eprintln!(
        "poisson response-scale 95% CI coverage: reps={replicates} n={n} \
         gam_cov={gam_coverage:.4} mgcv_cov={mgcv_coverage:.4} nominal={nominal} \
         gam_err={gam_err:.4} mgcv_err={mgcv_err:.4}"
    );

    // PRIMARY: gam's own response-scale Poisson intervals are calibrated. The
    // band (±0.07) is slightly looser than the Gaussian-identity case because
    // the log-link delta method plus the discrete Poisson draw add MC noise.
    assert!(
        gam_err <= 0.07,
        "gam 95% response-scale Poisson CI miscalibrated: empirical coverage \
         {gam_coverage:.4} outside {nominal} ± 0.07"
    );
    // MATCH-OR-BEAT: gam calibrates at least as well as mgcv (MC slack 0.04).
    assert!(
        gam_err <= mgcv_err + 0.04,
        "gam CI calibration worse than mgcv: gam_err {gam_err:.4} > mgcv_err {mgcv_err:.4} + 0.04"
    );
}

// ---------------------------------------------------------------------------
// Small numeric helpers (no external RNG-distribution deps for Poisson/gamma).
// ---------------------------------------------------------------------------

/// Pearson correlation of two equal-length samples.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa.sqrt() * sbb.sqrt()).max(1e-300)
}

/// Knuth Poisson sampler — adequate for the small λ regime of the Tweedie DGP.
fn poisson_sample(lambda: f64, rng: &mut StdRng) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let mut k = 0u32;
    let mut p = 1.0;
    loop {
        p *= unif.sample(rng);
        if p <= l {
            return k;
        }
        k += 1;
        if k > 10_000 {
            return k; // numerical safety net; never reached for the DGP's λ
        }
    }
}

/// Marsaglia–Tsang gamma sampler (shape > 0), returning a draw with the given
/// scale. Used to build the compound-Poisson–gamma Tweedie response.
fn gamma_sample(shape: f64, scale: f64, rng: &mut StdRng) -> f64 {
    let normal = Normal::new(0.0, 1.0).expect("normal");
    let unif = Uniform::new(0.0_f64, 1.0).expect("uniform");
    if shape < 1.0 {
        // Boost: Gamma(shape) = Gamma(shape+1) * U^(1/shape).
        let u: f64 = unif.sample(rng);
        return gamma_sample(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z: f64 = normal.sample(rng);
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = unif.sample(rng);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v * scale;
        }
    }
}
