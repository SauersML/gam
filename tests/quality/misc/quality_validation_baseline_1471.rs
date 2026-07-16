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
//! APPLES-TO-APPLES (the matched-model contract — every arm honours all three):
//!
//!   1. SAME BASIS. gam and mgcv use the SAME explicit `bs` on both sides, never
//!      a bare `s()` that resolves to each tool's own default. The Gaussian arms
//!      (s(x)+x, concurvity) pin `bs="cr"` on BOTH sides — the cubic-regression
//!      spline is constructed identically in both engines (no thin-plate
//!      knot-selection nondeterminism). The Tweedie arm pins `bs="tp"` on BOTH
//!      sides (the clean thin-plate cell; see the #1477 note below). No arm uses
//!      a bare default `s()` against an explicit basis.
//!
//!   2. SAME PENALTY MODEL. gam's smooth `double_penalty` flag is the exact analog
//!      of mgcv's `select=`. The matched pairs are: gam `double_penalty=false`
//!      ↔ mgcv ordinary penalized fit (`select=FALSE`, the default), and gam
//!      `double_penalty=true` ↔ mgcv `select=TRUE`. Every arm fixes gam's
//!      `double_penalty` EXPLICITLY and sets mgcv's `select=` to the matching
//!      value, so neither side silently double-penalizes the nullspace while the
//!      other does not. (gam's bare smooth default is `double_penalty=true` and
//!      mgcv's bare default is `select=FALSE`; pinning both removes that mismatch.)
//!
//!   3. SAME FAMILY PARAMETERS. The Tweedie variance power `p` is FIXED at the
//!      same value (1.5) on both sides — gam hard-codes `p=1.5` (it does not
//!      estimate the power), so the mgcv call uses `tw(p=1.5)` (the fixed-power
//!      Tweedie), NOT `tw()` (which estimates `p`). This arm therefore makes no
//!      "power estimated" claim; both engines fit the SAME fixed-`p` model.
//!
//! Tolerances here are deliberately looser than the documented point estimates
//! (the documented value is the seed-specific realization; the asserted bound is
//! the principled regression floor) but never weaker than "gam matches-or-beats
//! mgcv on truth recovery to within a small Monte-Carlo margin". A real
//! regression in any of these paths — a dropped penalty, a broken cyclic seam, a
//! concurvity-driven over/under-smooth — fails the corresponding arm loudly.
//!
//! RECONCILIATION WITH #1476 / #1477:
//!
//!   * #1477 proves the Tweedie `s(x)` mean is BIASED *specifically* on the
//!     explicit P-spline basis (`bs='ps'`) — right-boundary blow-up — while
//!     gamfit's own `bs='cr'` and the thin-plate `bs='tp'` recover truth. Arm 1
//!     pins `bs="tp"` (matched on both sides) — the clean cell — and adds an
//!     explicit right-boundary fitted-mean assertion so a #1477-style boundary
//!     blow-up cannot leak into the thin-plate path undetected. Arm 1 does NOT
//!     certify the `bs='ps'` Tweedie path; that broken cell is tracked by #1477.
//!
//!   * #1476 was the default double-penalty nullspace mis-allocation that
//!     collapsed one smooth's PER-PARTIAL effect to EDF≈0 under concurvity. With
//!     the projector fix (#26ab264e3) and a MATCHED penalty model (gam
//!     `double_penalty=false` ↔ mgcv ordinary), the per-partial recovery should
//!     now hold. Arm 3 therefore asserts the matched-model per-term contract
//!     directly: each smooth keeps real EDF (neither collapses to ≈0), each
//!     partial curve recovers its own truth component, and both smoothing
//!     parameters are finite and sane. A regression that reintroduces the
//!     nullspace collapse fails these per-term bars loudly.

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, r2, rmse, run_r};
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
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
        .map(|i| {
            StringRecord::from(
                cols.iter()
                    .map(|(_, c)| c[i].to_string())
                    .collect::<Vec<_>>(),
            )
        })
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
// Arm 1 — Tweedie s(x), log link, FIXED variance power p = 1.5 (matched both
//         sides). gam does NOT estimate p; this arm makes NO "power estimated"
//         claim — it fits the same fixed-p model in both engines.
// ===========================================================================

/// The Tweedie compound-Poisson-gamma response with a log-link smooth must
/// RECOVER a known log-mean curve and match-or-beat mgcv's FIXED-power
/// `tw(p=1.5)` on truth-recovery RMSE. Documented: gam/mgcv RMSE ratio ≈ 1.13,
/// no EDF stall.
///
/// MATCHED MODEL (see the module header contract):
///   * SAME BASIS: `bs="tp"`, `k=10` on BOTH sides (the clean thin-plate cell,
///     NOT the #1477-biased `bs='ps'`).
///   * SAME PENALTY MODEL: gam `double_penalty=false` ↔ mgcv ordinary
///     (`select=FALSE`).
///   * SAME FAMILY PARAM: variance power fixed at `p=1.5` on both sides. gam
///     hard-codes `p=1.5`; the mgcv call uses `tw(p=1.5)` (fixed), NOT `tw()`
///     (which estimates `p`). No power-estimation claim is asserted.
///
/// Averaged over several seeds to suppress the compound-Poisson-gamma draw's
/// Monte-Carlo noise, and with an explicit right-boundary fitted-mean check
/// (the #1477 guard): the thin-plate mean must track truth at the right edge.
#[test]
fn tweedie_log_smooth_recovers_truth_and_matches_mgcv() {
    init_parallelism();

    // True log-mean is a smooth nonlinear curve; mean = exp(eta) so the response
    // is strictly positive with zero inflation (the defining Tweedie feature).
    let true_eta = |x: f64| 0.6 * (2.0 * x).sin() + 0.4 * x - 0.5;
    let n = 400usize;
    let p_true = 1.5_f64;
    let phi = 0.6_f64;
    let seeds: [u64; 5] = [147_001, 147_011, 147_021, 147_031, 147_041];

    let grid: Vec<f64> = (0..120).map(|i| 3.0 * i as f64 / 119.0).collect();
    let truth_mu: Vec<f64> = grid.iter().map(|&xg| true_eta(xg).exp()).collect();
    let signal = truth_mu.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - truth_mu.iter().cloned().fold(f64::INFINITY, f64::min);
    let last = grid.len() - 1;
    let boundary_truth = truth_mu[last];

    let mut gam_err_sum = 0.0;
    let mut mgcv_err_sum = 0.0;
    let mut boundary_ok = 0usize;

    for &seed in &seeds {
        let mut rng = StdRng::seed_from_u64(seed);
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
        assert!(
            zeros > 0,
            "Tweedie 1<p<2 must be zero-inflated; got {zeros} zeros"
        );

        let ds = encode(&[("x", &x), ("y", &y)]);
        let x_idx = ds.column_map()["x"];
        let width = ds.headers.len();

        let cfg = FitConfig {
            family: Some("tweedie".to_string()),
            ..FitConfig::default()
        };
        // gam: matched basis (tp) + matched penalty model (double_penalty=false).
        let result = fit_from_formula("y ~ s(x, bs=\"tp\", k=10, double_penalty=false)", &ds, &cfg)
            .expect("gam tweedie fit");
        let FitResult::Standard(fit) = result else {
            panic!("Tweedie(log) is a scalar GLM family => expected FitResult::Standard");
        };
        let gam_edf = fit.fit.edf_total().expect("gam reports total edf");
        assert!(
            gam_edf > 1.0 && gam_edf < 15.0,
            "tweedie smooth edf out of sane range (EDF stall?): {gam_edf:.3}"
        );

        let gam_mu: Vec<f64> = gam_eta(&fit, width, &[(x_idx, &grid)])
            .iter()
            .map(|e| e.exp())
            .collect();

        // mgcv: SAME basis (tp), SAME penalty model (ordinary, select=FALSE), and
        // the SAME FIXED variance power p=1.5 — tw(p=1.5), NOT tw() (no estimate).
        let r = run_r(
            &[Column::new("x", &x), Column::new("y", &y)],
            &format!(
                r#"
                suppressPackageStartupMessages(library(mgcv))
                m <- gam(y ~ s(x, bs = "tp", k = 10), data = df,
                         family = tw(p = 1.5), select = FALSE, method = "REML")
                xg <- seq(0, 3, length.out = {ng})
                emit("mu", as.numeric(predict(m, newdata = data.frame(x = xg), type = "response")))
                emit("edf", sum(m$edf))
                "#,
                ng = grid.len(),
            ),
        );
        let mgcv_mu = r.vector("mu");
        let mgcv_edf = r.scalar("edf");
        assert_eq!(mgcv_mu.len(), grid.len(), "mgcv mu length mismatch");

        let gam_err = rmse(&gam_mu, &truth_mu);
        let mgcv_err = rmse(mgcv_mu, &truth_mu);
        gam_err_sum += gam_err;
        mgcv_err_sum += mgcv_err;

        // BOUNDARY (#1477 guard): the right-edge fitted mean must track truth.
        // The clean tp cell stays within a tight factor of the analytic mean at
        // x=3; the #1477 ps blow-up overshoots by >2×. Require 1.4× (tighter than
        // before) at the rightmost evaluation point.
        let boundary_gam = gam_mu[last];
        if boundary_gam <= boundary_truth * 1.4 && boundary_gam >= boundary_truth / 1.4 {
            boundary_ok += 1;
        }

        eprintln!(
            "tweedie s(x) tp p=1.5 seed={seed}: n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
             gam_rmse_vs_truth={gam_err:.5} mgcv_rmse_vs_truth={mgcv_err:.5} \
             ratio={:.3} gam_mu(x=3)={boundary_gam:.4} truth(x=3)={boundary_truth:.4}",
            gam_err / mgcv_err.max(1e-12)
        );
    }

    let ns = seeds.len() as f64;
    let gam_err = gam_err_sum / ns;
    let mgcv_err = mgcv_err_sum / ns;
    eprintln!(
        "tweedie s(x) tp p=1.5 MEAN over {} seeds: gam_rmse={gam_err:.5} mgcv_rmse={mgcv_err:.5} \
         ratio={:.3} signal={signal:.4} boundary_ok={boundary_ok}/{}",
        seeds.len(),
        gam_err / mgcv_err.max(1e-12),
        seeds.len()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_validation_baseline_1471::tweedie",
            "rmse_vs_truth",
            gam_err,
            "mgcv",
            mgcv_err,
        )
        .line()
    );

    // PRIMARY: gam recovers the mean curve well below the signal scale (seed-mean).
    assert!(
        gam_err < 0.30 * signal,
        "tweedie smooth failed to recover the log-mean curve: mean rmse={gam_err:.5} \
         (signal {signal:.4})"
    );
    // MATCH-OR-BEAT: gam's seed-mean truth-recovery RMSE no worse than 1.25× mgcv's.
    // The documented realization is ≈1.13; the 1.25 bound is the regression floor.
    assert!(
        gam_err <= mgcv_err * 1.25,
        "gam tweedie recovery {gam_err:.5} worse than mgcv {mgcv_err:.5} * 1.25"
    );
    // BOUNDARY: the right-edge mean must track truth on at least 4 of 5 seeds (a
    // genuine #1477-style blow-up fails every seed, not an MC-tail one).
    assert!(
        boundary_ok >= 4,
        "tweedie tp mean shows a #1477-style right-boundary distortion: \
         only {boundary_ok}/5 seeds kept gam(x=3) within 1.4× of truth {boundary_truth:.4}"
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
    // gam: matched basis (cr) + matched penalty model (double_penalty=false). The
    // parametric `linear(x)` term is unpenalized on both sides (mgcv `+ x` too).
    let result = fit_from_formula(
        "y ~ s(x, bs=\"cr\", k=10, double_penalty=false) + linear(x)",
        &ds,
        &cfg,
    )
    .expect("gam s(x)+x fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x)+linear(x)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    let grid: Vec<f64> = (0..120).map(|i| 4.0 * i as f64 / 119.0).collect();
    let truth_grid: Vec<f64> = grid.iter().map(|&xg| truth(xg)).collect();
    let gam_grid = gam_eta(&fit, width, &[(x_idx, &grid)]);

    // mgcv: SAME basis (cr), SAME penalty model (ordinary, select=FALSE).
    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x, bs = "cr", k = 10) + x, data = df,
                     select = FALSE, method = "REML")
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
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_validation_baseline_1471::smooth_plus_linear",
            "rmse_vs_truth",
            gam_err,
            "mgcv",
            mgcv_err,
        )
        .line()
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
/// curves become weakly identified). Documented: gam/mgcv RMSE ratio ≈ 1.03.
///
/// MATCHED MODEL + PER-TERM CONTRACT (the #1476 / #26ab264e3 guard): both engines
/// fit `s(x, bs="cr") + s(z, bs="cr")` with a MATCHED ordinary penalty model
/// (gam `double_penalty=false` ↔ mgcv `select=FALSE`). With the projector fix
/// (#26ab264e3) the per-partial recovery should hold even under concurvity, so
/// this arm now asserts the FULL per-term contract — not just the additive sum:
///   (a) the additive SUM recovers truth and match-or-beats mgcv (unchanged);
///   (b) EACH smooth keeps real EDF — neither collapses to ≈0 (the #1476 mode);
///   (c) EACH partial curve recovers its own truth component (s_x↔f, s_z↔g);
///   (d) BOTH smoothing parameters are finite and sane.
/// A regression that reintroduces the nullspace collapse fails (b)/(c) loudly.
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
    // gam: matched basis (cr) + matched ordinary penalty model (double_penalty=false).
    let result = fit_from_formula(
        "y ~ s(x, bs=\"cr\", k=10, double_penalty=false) + s(z, bs=\"cr\", k=10, double_penalty=false)",
        &ds,
        &cfg,
    )
    .expect("gam s(x)+s(z) fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for s(x)+s(z)");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Evaluate the recovered ADDITIVE SUM at the training rows vs the noiseless
    // truth f(x)+g(z): the sum is identified even when the components are not.
    let truth_sum: Vec<f64> = (0..n).map(|i| f(x[i]) + g(z[i])).collect();
    let gam_sum = gam_eta(&fit, width, &[(x_idx, &x), (z_idx, &z)]);

    // --- per-term partials (the #1476 / #26ab264e3 contract) -------------------
    // Isolate each smooth's partial curve by evaluating the frozen design with
    // ONLY that covariate set (the other smooth contributes its constant value at
    // 0, folded into the additive-identifiability constant — centered out below).
    // s_x partial vs f(x); s_z partial vs g(z), both on a dense in-support grid.
    let pgrid_x: Vec<f64> = (0..120).map(|i| -2.5 + 5.0 * i as f64 / 119.0).collect();
    let pgrid_z = pgrid_x.clone();
    let gam_px = gam_eta(&fit, width, &[(x_idx, &pgrid_x)]);
    let gam_pz = gam_eta(&fit, width, &[(z_idx, &pgrid_z)]);
    let truth_px: Vec<f64> = pgrid_x.iter().map(|&v| f(v)).collect();
    let truth_pz: Vec<f64> = pgrid_z.iter().map(|&v| g(v)).collect();

    // gam per-term EDF: edf_by_block is aligned 1:1 with the smoothing parameters
    // (one penalty per cr smooth under double_penalty=false), so block 0 = s(x),
    // block 1 = s(z). lambdas carries the per-term smoothing parameter.
    let gam_edf_blocks = fit.fit.edf_by_block();
    let gam_lambdas: Vec<f64> = fit.fit.lambdas.to_vec();
    assert!(
        gam_edf_blocks.len() >= 2,
        "expected >=2 per-term EDF blocks for s(x)+s(z), got {}",
        gam_edf_blocks.len()
    );
    assert!(
        gam_lambdas.len() >= 2,
        "expected >=2 per-term smoothing parameters, got {}",
        gam_lambdas.len()
    );
    let edf_sx = gam_edf_blocks[0];
    let edf_sz = gam_edf_blocks[1];

    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("y", &y),
        ],
        &format!(
            r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x, bs = "cr", k = 10) + s(z, bs = "cr", k = 10), data = df,
                 select = FALSE, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        # per-smooth EDF (summary s.table rows are ordered s(x) then s(z))
        st <- summary(m)$s.table
        emit("edf_per", as.numeric(st[, "edf"]))
        emit("sp", as.numeric(m$sp))
        # per-term partial curves on the SAME dense grids gam scores
        pgx <- seq(-2.5, 2.5, length.out = {ng})
        tx <- predict(m, newdata = data.frame(x = pgx, z = 0), type = "terms")
        emit("px", as.numeric(tx[, "s(x)"]))
        tz <- predict(m, newdata = data.frame(x = 0, z = pgx), type = "terms")
        emit("pz", as.numeric(tz[, "s(z)"]))
        "#,
            ng = pgrid_x.len(),
        ),
    );
    let mgcv_sum = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    let mgcv_edf_per = r.vector("edf_per");
    let mgcv_sp = r.vector("sp");
    let mgcv_px = r.vector("px");
    let mgcv_pz = r.vector("pz");
    assert_eq!(mgcv_sum.len(), n, "mgcv fitted length mismatch");
    assert_eq!(mgcv_edf_per.len(), 2, "mgcv must report 2 per-smooth EDFs");
    assert_eq!(mgcv_sp.len(), 2, "mgcv must report 2 smoothing parameters");

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
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_validation_baseline_1471::concurvity_sum",
            "rmse_vs_truth",
            gam_err,
            "mgcv",
            mgcv_err,
        )
        .line()
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

    // ====================================================================
    // PER-TERM CONTRACT (the #1476 / #26ab264e3 regression guard). Now that
    // the projector fix and the matched ordinary penalty model are in place,
    // each smooth must keep real complexity and recover its OWN component.
    // ====================================================================

    // (b) EDF non-collapse: neither smooth may be smoothed to a straight line
    // (EDF≈1) or annihilated (EDF≈0). The #1476 nullspace mis-allocation drives
    // exactly one component's EDF to ≈0 while the other absorbs both signals;
    // require BOTH gam smooths to carry genuine curvature (>1.5 EDF), and assert
    // the same of mgcv so the bar is a real matched property, not gam-specific.
    eprintln!(
        "concurvity per-term: gam edf_sx={edf_sx:.3} edf_sz={edf_sz:.3} \
         mgcv edf={:.3},{:.3} | gam sp={:.3e},{:.3e} mgcv sp={:.3e},{:.3e}",
        mgcv_edf_per[0], mgcv_edf_per[1], gam_lambdas[0], gam_lambdas[1], mgcv_sp[0], mgcv_sp[1]
    );
    assert!(
        edf_sx > 1.5 && edf_sz > 1.5,
        "a concurvity smooth collapsed (the #1476 nullspace mis-allocation): \
         gam EDF s(x)={edf_sx:.3}, s(z)={edf_sz:.3} (both must exceed 1.5)"
    );
    assert!(
        mgcv_edf_per[0] > 1.5 && mgcv_edf_per[1] > 1.5,
        "mgcv reference itself collapsed a smooth — the per-term bar is unfair: \
         mgcv EDF s(x)={:.3}, s(z)={:.3}",
        mgcv_edf_per[0],
        mgcv_edf_per[1]
    );

    // (c) per-PARTIAL truth recovery: each isolated smooth curve must track its
    // OWN truth component (s_x↔f, s_z↔g), centered to drop the additive constant.
    // R² of each partial against its truth component — and match-or-beat mgcv's
    // own partials on the same metric (mgcv is the mature concurvity baseline).
    let truth_px_c = center(&truth_px);
    let truth_pz_c = center(&truth_pz);
    let gam_px_c = center(&gam_px);
    let gam_pz_c = center(&gam_pz);
    let mgcv_px_c = center(mgcv_px);
    let mgcv_pz_c = center(mgcv_pz);

    let px_r2 = r2(&gam_px_c, &truth_px_c);
    let pz_r2 = r2(&gam_pz_c, &truth_pz_c);
    let px_err = rmse(&gam_px_c, &truth_px_c);
    let pz_err = rmse(&gam_pz_c, &truth_pz_c);
    let mgcv_px_err = rmse(&mgcv_px_c, &truth_px_c);
    let mgcv_pz_err = rmse(&mgcv_pz_c, &truth_pz_c);
    eprintln!(
        "concurvity per-partial recovery: s(x) gam_r2={px_r2:.4} gam_rmse={px_err:.5} \
         mgcv_rmse={mgcv_px_err:.5} | s(z) gam_r2={pz_r2:.4} gam_rmse={pz_err:.5} \
         mgcv_rmse={mgcv_pz_err:.5}"
    );
    // Each partial recovers its own component well (not just the sum). A collapsed
    // component would have near-zero or negative R² against its truth piece.
    assert!(
        px_r2 > 0.80,
        "s(x) partial failed to recover f(x): R²={px_r2:.4} (component collapse?)"
    );
    assert!(
        pz_r2 > 0.80,
        "s(z) partial failed to recover g(z): R²={pz_r2:.4} (component collapse?)"
    );
    // MATCH-OR-BEAT mgcv per-partial: no worse than 1.30× mgcv's partial RMSE
    // (per-partial is intrinsically noisier under concurvity than the sum, so the
    // floor is looser than the 1.15 sum bar — but still a real matched bound).
    assert!(
        px_err <= mgcv_px_err * 1.30,
        "gam s(x) partial {px_err:.5} worse than mgcv {mgcv_px_err:.5} * 1.30"
    );
    assert!(
        pz_err <= mgcv_pz_err * 1.30,
        "gam s(z) partial {pz_err:.5} worse than mgcv {mgcv_pz_err:.5} * 1.30"
    );

    // (d) both smoothing parameters finite, positive and not railed to the
    // degenerate extremes (sp→∞ ⇒ a straight line; sp→0 ⇒ interpolation). gam's
    // lambdas and mgcv's sp are the same penalty-weight coordinate.
    for (i, &lam) in gam_lambdas.iter().take(2).enumerate() {
        assert!(
            lam.is_finite() && lam > 1e-8 && lam < 1e12,
            "gam smoothing parameter {i} out of sane range (railed?): {lam:.3e}"
        );
    }
    for (i, &sp) in mgcv_sp.iter().enumerate() {
        assert!(
            sp.is_finite() && sp > 0.0,
            "mgcv smoothing parameter {i} not finite/positive: {sp:.3e}"
        );
    }
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
    let formula = "y ~ te(t, x, boundary=['periodic','clamped'], period=[2*pi, None], k=8)";
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
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_validation_baseline_1471::cyclic_tensor",
            "rmse_vs_truth",
            gam_err,
            "mgcv",
            mgcv_err,
        )
        .line()
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
/// nominal and (b) it covers at least as well as mgcv on the identical data.
///
/// MATCHED COVARIANCE (the finding-#6 fix): gam uses
/// `SmoothingCorrected` (`H⁻¹ + J·Var(ρ̂)·Jᵀ` — the conditional
/// posterior PLUS the first-order smoothing-parameter-uncertainty correction).
/// mgcv's matching covariance is `Vc`, obtained with `unconditional = TRUE` in
/// `predict.gam`. The mgcv call below therefore passes `unconditional = TRUE`,
/// so BOTH intervals carry the smoothing-uncertainty term — not gam's `Vc`
/// against mgcv's conditional `Vp`. Basis/penalty are matched too (`bs="cr"`,
/// ordinary `select=FALSE` ↔ gam `double_penalty=false`).
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
    let mu_true: Vec<f64> = x
        .iter()
        .map(|&v| (1.0 + 0.9 * (TAU * v).sin()).exp())
        .collect();

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
        let result = fit_from_formula("y ~ s(x, bs=\"cr\", k=10, double_penalty=false)", &ds, &cfg)
            .expect("gam poisson fit");
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
                covariance_mode: InferenceCovarianceMode::SmoothingCorrected,
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
        gam_cov += covered(
            &pred.mean_lower.to_vec(),
            &pred.mean_upper.to_vec(),
            &mu_true,
        );

        let r = run_r(
            &[Column::new("x", &x), Column::new("y", &y)],
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x, bs = "cr", k = 10), data = df, family = poisson(),
                     select = FALSE, method = "REML")
            # unconditional = TRUE -> Vc (adds the smoothing-parameter-uncertainty
            # term), the match for gam's SmoothingCorrected mode.
            p <- predict(m, newdata = df, se.fit = TRUE, type = "response",
                         unconditional = TRUE)
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
    eprintln!(
        "{}",
        QualityPair::error(
            "misc",
            "quality_validation_baseline_1471::poisson_ci_coverage",
            "ci_calibration_error",
            gam_err,
            "mgcv",
            mgcv_err,
        )
        .line()
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
