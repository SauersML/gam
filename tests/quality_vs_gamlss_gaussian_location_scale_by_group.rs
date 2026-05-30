//! End-to-end quality: gam's Gaussian location-scale fit with a **by-group**
//! smooth in *both* the mean and the log-σ model must match `gamlss` — the
//! standard GAMLSS-style distributional-regression engine — on the same data.
//!
//! What this benchmarks against, and why
//! -------------------------------------
//! gam builds a single joint design for a Gaussian location-scale model where
//! `s(x, bs="tp", by=group)` enters the mean and `noise_formula="s(x, bs="tp",
//! by=group)"` enters log σ. The `by=` factor expands each smooth into one
//! block per group level, with a block-diagonal penalty: group A's smooth and
//! group B's smooth are penalized (and hence shrunk) independently in both the
//! mean and the variance model. The danger is that gam's joint design assembly
//! cross-wires the by= columns between groups, or between the mean and log-σ
//! blocks, so that one group's structure leaks into the other or the σ-block
//! picks up the mean's basis.
//!
//! The mature reference is `gamlss::gamlss(family=NO())`, which has no by=
//! machinery: the textbook way to get group-specific mean *and* σ smooths there
//! is explicit stratification — fit one independent `gamlss` per group with a
//! `tp` smooth (via `ga(~ s(x, bs="tp"))`) in both `mu` and `sigma`. Two
//! independent per-group GAMLSS fits are exactly the population gam's
//! block-diagonal by= construction is supposed to reproduce. If gam's by=
//! semantics propagate correctly to both the mean and log-σ blocks, the gam
//! per-group fitted μ and σ curves must coincide with the corresponding
//! stratified gamlss curves.
//!
//! Data (seed 321, n = 200, 100 per group), fed IDENTICALLY to both engines:
//!   group A: x ~ U(0,1), y ~ N(sin(2πx),  (0.10 + 0.10 sin(πx))^2)
//!   group B: x ~ U(0,1), y ~ N(0.5 + 0.3 sin(3πx), (0.12 + 0.08 x)^2)
//! Group A has a smooth heteroscedastic σ (a hump); group B a near-linear σ
//! ramp. The two groups differ in BOTH mean shape and σ shape, so a leak
//! between by= blocks would visibly distort one curve toward the other.
//!
//! Metric / bound (per group, on a shared 50-point grid spanning that group's
//! observed x-range): relative L2 of fitted μ < 0.02 AND relative L2 of
//! log σ < 0.04. Both engines REML/ML-fit penalized `tp` smooths to the same
//! data, so the mean curves should track to ~1% as in the plain-smooth mgcv
//! benchmark (rel_l2 ~0.005 there); the σ curve is a second, noisier estimand
//! (variance is estimated from squared residuals), hence the slightly looser
//! 0.04 on log σ — still tight enough that a by=/log-σ block leak fails it. The
//! bounds are NOT loosened to whitelist a divergence; a genuine gap is a real
//! finding.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_GROUP: usize = 100;
const GRID_POINTS: usize = 50;

/// True mean for group A.
fn mean_a(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
/// True sigma for group A (smooth heteroscedastic hump).
fn sigma_a(x: f64) -> f64 {
    0.10 + 0.10 * (std::f64::consts::PI * x).sin()
}
/// True mean for group B.
fn mean_b(x: f64) -> f64 {
    0.5 + 0.3 * (3.0 * std::f64::consts::PI * x).sin()
}
/// True sigma for group B (near-linear ramp).
fn sigma_b(x: f64) -> f64 {
    0.12 + 0.08 * x
}

/// `seq(a, b, length.out = GRID_POINTS)` replicated EXACTLY so the gam grid and
/// the R `seq(...)` grid are bit-for-bit identical given identical (a, b).
fn linspace(a: f64, b: f64) -> Vec<f64> {
    (0..GRID_POINTS)
        .map(|i| a + (b - a) * (i as f64) / ((GRID_POINTS - 1) as f64))
        .collect()
}

#[test]
fn gam_location_scale_by_group_matches_gamlss() {
    init_parallelism();

    // ---- synthesize the two-group location-scale data (seed 321) ----------
    // Group A rows first, then group B, so the inferred categorical levels are
    // ["A", "B"] -> codes [0.0, 1.0] in first-seen insertion order.
    let mut rng = StdRng::seed_from_u64(321);
    let ux = Uniform::new(0.0_f64, 1.0_f64).expect("uniform x");
    let std_normal = Normal::new(0.0_f64, 1.0_f64).expect("standard normal");

    let headers = vec!["y".to_string(), "x".to_string(), "group".to_string()];
    let mut rows: Vec<StringRecord> = Vec::with_capacity(2 * N_PER_GROUP);

    // We also keep the raw (x, y) per group so the R side can subset by group
    // and so we can build the per-group prediction grids over the SAME x-range.
    let mut x_a = Vec::with_capacity(N_PER_GROUP);
    let mut x_b = Vec::with_capacity(N_PER_GROUP);

    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_a(x) + sigma_a(x) * std_normal.sample(&mut rng);
        x_a.push(x);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "A".to_string(),
        ]));
    }
    for _ in 0..N_PER_GROUP {
        let x = ux.sample(&mut rng);
        let y = mean_b(x) + sigma_b(x) * std_normal.sample(&mut rng);
        x_b.push(x);
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            "B".to_string(),
        ]));
    }

    // Columns shipped to R verbatim (same numbers gam encodes). group is sent
    // as a 0/1 code matching gam's first-seen encoding (A=0, B=1); R turns it
    // back into a factor for stratification.
    let mut y_all = Vec::with_capacity(2 * N_PER_GROUP);
    let mut x_all = Vec::with_capacity(2 * N_PER_GROUP);
    let mut g_all = Vec::with_capacity(2 * N_PER_GROUP);
    for r in &rows {
        y_all.push(r.get(0).expect("y field").parse::<f64>().expect("parse y"));
        x_all.push(r.get(1).expect("x field").parse::<f64>().expect("parse x"));
        g_all.push(if r.get(2) == Some("A") { 0.0 } else { 1.0 });
    }

    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode by-group data");
    let col = data.column_map();
    let x_idx = col["x"];
    let group_idx = col["group"];

    // ---- fit gam: Gaussian location-scale, by-group tp smooth on BOTH μ and
    //      log σ ----------------------------------------------------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x, bs='tp', by=group)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp', by=group)", &data, &cfg)
        .expect("gam Gaussian location-scale by-group fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit result for a noise_formula model");
    };

    // Mean (μ) coefficient block: identity link => μ = X_mean · β_mean.
    let beta_mean = &fit
        .fit
        .fit
        .block_by_role(gam::solver::estimate::BlockRole::Location)
        .expect("location-scale fit must carry a Location (mean) block")
        .beta;
    // log-σ (Scale) coefficient block: log σ = X_noise · β_scale (log link).
    let beta_scale = &fit
        .fit
        .fit
        .block_by_role(gam::solver::estimate::BlockRole::Scale)
        .expect("location-scale fit must carry a Scale (log-σ) block")
        .beta;

    // Build the per-group prediction grids over each group's observed x-range.
    let (a_lo, a_hi) = x_a
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let (b_lo, b_hi) = x_b
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let grid_a = linspace(a_lo, a_hi);
    let grid_b = linspace(b_lo, b_hi);

    let ncols = data.headers.len();

    // gam μ and σ on each per-group grid: set the group code so the by= term
    // activates only that group's block, and read off X·β through the frozen
    // resolved specs (mean uses meanspec_resolved, σ uses noisespec_resolved).
    let predict_group = |grid: &[f64], code: f64| -> (Vec<f64>, Vec<f64>) {
        let mut design_pts = Array2::<f64>::zeros((grid.len(), ncols));
        for (i, &xv) in grid.iter().enumerate() {
            design_pts[[i, x_idx]] = xv;
            design_pts[[i, group_idx]] = code;
        }
        let mean_design =
            build_term_collection_design(design_pts.view(), &fit.fit.meanspec_resolved)
                .expect("rebuild mean design at grid");
        let noise_design =
            build_term_collection_design(design_pts.view(), &fit.fit.noisespec_resolved)
                .expect("rebuild noise design at grid");
        let mu = mean_design.design.apply(beta_mean).to_vec();
        let log_sigma = noise_design.design.apply(beta_scale).to_vec();
        let sigma: Vec<f64> = log_sigma.iter().map(|v| v.exp()).collect();
        (mu, sigma)
    };

    let (gam_mu_a, gam_sigma_a) = predict_group(&grid_a, 0.0);
    let (gam_mu_b, gam_sigma_b) = predict_group(&grid_b, 1.0);

    // ---- fit the SAME data with gamlss, stratified by group (the mature
    //      GAMLSS reference). One independent gamlss per group, tp smooth via
    //      ga(~ s(x, bs="tp")) in BOTH the mu and sigma models, NO() family.
    //      predictAll returns μ and σ on the response scale. ------------------
    let grid_concat: Vec<f64> = grid_a.iter().chain(grid_b.iter()).copied().collect();
    let r = run_r(
        &[
            Column::new("y", &y_all),
            Column::new("x", &x_all),
            Column::new("g", &g_all),
            // Two extra columns of equal length (200) carrying the 100+100
            // grid points (padded) so we can ship the exact prediction grid.
            Column::new("gridx", &pad_to(&grid_concat, 2 * N_PER_GROUP)),
        ],
        r#"
        suppressPackageStartupMessages(library(gamlss))
        suppressPackageStartupMessages(library(gamlss.add))
        df$g <- factor(ifelse(df$g < 0.5, "A", "B"), levels = c("A", "B"))
        gridx <- df$gridx[1:100]
        gA <- gridx[1:50]
        gB <- gridx[51:100]

        fit_one <- function(sub, grid) {
          m <- gamlss(y ~ ga(~ s(x, bs = "tp")),
                      sigma.formula = ~ ga(~ s(x, bs = "tp")),
                      family = NO(), data = sub,
                      control = gamlss.control(trace = FALSE))
          nd <- data.frame(x = grid)
          pa <- predictAll(m, newdata = nd, data = sub, type = "response")
          list(mu = as.numeric(pa$mu), sigma = as.numeric(pa$sigma))
        }

        ra <- fit_one(df[df$g == "A", ], gA)
        rb <- fit_one(df[df$g == "B", ], gB)
        emit("mu_a", ra$mu)
        emit("sigma_a", ra$sigma)
        emit("mu_b", rb$mu)
        emit("sigma_b", rb$sigma)
        "#,
    );

    let ref_mu_a = r.vector("mu_a");
    let ref_sigma_a = r.vector("sigma_a");
    let ref_mu_b = r.vector("mu_b");
    let ref_sigma_b = r.vector("sigma_b");
    assert_eq!(ref_mu_a.len(), GRID_POINTS, "gamlss group-A μ grid length");
    assert_eq!(ref_mu_b.len(), GRID_POINTS, "gamlss group-B μ grid length");

    // log σ comparison: relative L2 on log σ (the natural scale of the σ-model).
    let log_vec = |v: &[f64]| -> Vec<f64> { v.iter().map(|s| s.ln()).collect() };

    let mu_a_rel = relative_l2(&gam_mu_a, ref_mu_a);
    let mu_b_rel = relative_l2(&gam_mu_b, ref_mu_b);
    let logsig_a_rel = relative_l2(&log_vec(&gam_sigma_a), &log_vec(ref_sigma_a));
    let logsig_b_rel = relative_l2(&log_vec(&gam_sigma_b), &log_vec(ref_sigma_b));

    eprintln!(
        "by-group location-scale vs gamlss: \
         A mu_rel={mu_a_rel:.4} logsig_rel={logsig_a_rel:.4} | \
         B mu_rel={mu_b_rel:.4} logsig_rel={logsig_b_rel:.4}"
    );

    // Per-group μ must track gamlss to ~1% (same penalized tp objective on the
    // same stratum); the 0.02 bar mirrors the plain-smooth mgcv benchmark.
    assert!(
        mu_a_rel < 0.02,
        "group-A mean diverges from gamlss: rel_l2={mu_a_rel:.4}"
    );
    assert!(
        mu_b_rel < 0.02,
        "group-B mean diverges from gamlss: rel_l2={mu_b_rel:.4}"
    );
    // log σ is a noisier (squared-residual) estimand; 0.04 is principled-loose
    // but still fails on any cross-block leak between groups or μ/log-σ.
    assert!(
        logsig_a_rel < 0.04,
        "group-A log-sigma diverges from gamlss: rel_l2={logsig_a_rel:.4}"
    );
    assert!(
        logsig_b_rel < 0.04,
        "group-B log-sigma diverges from gamlss: rel_l2={logsig_b_rel:.4}"
    );
}

/// Pad `v` with trailing zeros to `len` so it can ride as a fixed-width column.
fn pad_to(v: &[f64], len: usize) -> Vec<f64> {
    assert!(v.len() <= len, "pad_to: vector longer than target length");
    let mut out = v.to_vec();
    out.resize(len, 0.0);
    out
}
