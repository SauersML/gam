//! End-to-end quality: gam's 1-D thin-plate *by-factor* (interaction) smooth
//! must match `mgcv` — the mature, standard GAM implementation — when a separate
//! smooth is fit per factor level.
//!
//! This benchmarks gam's `s(x, by=g, bs='tp')` against
//! `mgcv::gam(y ~ g + s(x, by=g, bs='tp', k=15), method="REML")` with `g` a
//! factor. mgcv's `by=<factor>` mechanism creates an independent thin-plate
//! basis block per level, so each level gets its OWN smooth function — the
//! canonical way to fit treatment-specific / group-specific curves in GAM
//! practice. The parametric `+ g` carries the per-level intercept (each
//! by-level smooth is sum-to-zero centered in both engines), matching the
//! unpenalized treatment-coded factor main effect gam adds automatically for a
//! categorical `by=`. We give the two engines byte-identical data (same x, same
//! A/B labels, same y) and assert:
//!   1. per-LEVEL fitted values agree pointwise (relative L2 over each group),
//!   2. total effective degrees of freedom agree (same overall complexity), and
//!   3. the two recovered group curves are genuinely DISTINCT — the by-factor
//!      mechanism must not collapse the levels onto one shared smooth.
//!
//! Both engines REML-fit the same penalized objective, so close agreement is the
//! correct expectation and a real divergence is a real bug. The truth is
//! group-A: sin(6πx), group-B: cos(4πx) at σ=0.06 — two clearly different shapes,
//! so a collapse-to-shared-smooth bug would blow up the per-group L2.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N_PER_LEVEL: usize = 80;
const SIGMA: f64 = 0.06;
const SEED: u64 = 111;

/// group-A truth: sin(6πx); group-B truth: cos(4πx).
fn truth(x: f64, group_a: bool) -> f64 {
    let pi = std::f64::consts::PI;
    if group_a {
        (6.0 * pi * x).sin()
    } else {
        (4.0 * pi * x).cos()
    }
}

#[test]
fn gam_thin_plate_by_factor_matches_mgcv() {
    init_parallelism();

    // ---- synthesize identical data for both engines ----------------------
    // x ~ U[0,1] per level; group label in {A, B}; y = truth + N(0, σ).
    // The "A" rows are emitted first so the categorical inference maps A->0,
    // B->1 (first-appearance level order); both engines see the same labels.
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");

    let n = 2 * N_PER_LEVEL;
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g = Vec::<String>::with_capacity(n);
    let mut g_code = Vec::<f64>::with_capacity(n); // 0.0=A, 1.0=B (handed to R numeric)
    let mut y = Vec::<f64>::with_capacity(n);
    let mut is_a = Vec::<bool>::with_capacity(n);

    // group A block first, then group B block (fixes level order A->0, B->1).
    for group_a in [true, false] {
        let mut xs: Vec<f64> = (0..N_PER_LEVEL).map(|_| ux.sample(&mut rng)).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &xi in &xs {
            let yi = truth(xi, group_a) + noise.sample(&mut rng);
            x.push(xi);
            y.push(yi);
            g.push(if group_a {
                "A".to_string()
            } else {
                "B".to_string()
            });
            g_code.push(if group_a { 0.0 } else { 1.0 });
            is_a.push(group_a);
        }
    }

    // ---- fit with gam: y ~ s(x, by=g, bs='tp', k=15), REML ----------------
    let headers = vec!["x".to_string(), "g".to_string(), "y".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![x[i].to_string(), g[i].clone(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode by-factor dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let g_idx = col["g"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, by=g, bs='tp', k=15)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian by-factor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted values at the training points: rebuild the frozen design at the
    // observed (x, g) and apply beta (identity link => design*beta = mean). The
    // factor column carries its encoded level index (0=A, 1=B), so each row is
    // evaluated against ITS OWN group's smooth block.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, g_idx]] = g_code[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(gam_fitted.len(), n, "gam fitted length mismatch");

    // Distinctness on a TRUE shared grid: evaluate the level-A and level-B
    // smooths at the SAME x points (g held at A=0, then at B=1) by rebuilding
    // the design twice on one common linspace. Because both predictions share
    // the grid, a collapse to a single shared smooth would force these two
    // curves to coincide; their disagreement is unambiguous proof of distinct
    // by-factor blocks (no reliance on the two groups' x draws lining up).
    const N_GRID: usize = 120;
    let mut grid_a = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
    let mut grid_b = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
    for j in 0..N_GRID {
        let xj = j as f64 / (N_GRID as f64 - 1.0);
        grid_a[[j, x_idx]] = xj;
        grid_a[[j, g_idx]] = 0.0; // level A
        grid_b[[j, x_idx]] = xj;
        grid_b[[j, g_idx]] = 1.0; // level B
    }
    let curve_a: Vec<f64> = build_term_collection_design(grid_a.view(), &fit.resolvedspec)
        .expect("rebuild design on grid (level A)")
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    let curve_b: Vec<f64> = build_term_collection_design(grid_b.view(), &fit.resolvedspec)
        .expect("rebuild design on grid (level B)")
        .design
        .apply(&fit.fit.beta)
        .to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) --------------
    // g arrives as a numeric 0/1 column; rebuild it as the matching factor so
    // mgcv's by=<factor> creates a separate tp basis per level. The factor main
    // effect `+ g` is REQUIRED to match gam's model: with an unordered `by=`
    // factor mgcv centers each level smooth (sum-to-zero), so the per-level
    // intercept difference lives ONLY in the parametric `g` term — exactly the
    // unpenalized treatment-coded main effect gam adds automatically for a
    // categorical `by=`. Omitting `+ g` would leave mgcv structurally unable to
    // represent that level offset, making the comparison apples-to-oranges.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("g", &g_code),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        df$g <- factor(ifelse(df$g < 0.5, "A", "B"), levels = c("A", "B"))
        m <- gam(y ~ g + s(x, by = g, bs = "tp", k = 15), data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), n, "mgcv fitted length mismatch");

    // ---- split fitted values by group and compare per level --------------
    let mut gam_a = Vec::new();
    let mut gam_b = Vec::new();
    let mut mgcv_a = Vec::new();
    let mut mgcv_b = Vec::new();
    for i in 0..n {
        if is_a[i] {
            gam_a.push(gam_fitted[i]);
            mgcv_a.push(mgcv_fitted[i]);
        } else {
            gam_b.push(gam_fitted[i]);
            mgcv_b.push(mgcv_fitted[i]);
        }
    }

    let rel_a = relative_l2(&gam_a, &mgcv_a);
    let rel_b = relative_l2(&gam_b, &mgcv_b);
    let corr_a = pearson(&gam_a, &mgcv_a);
    let corr_b = pearson(&gam_b, &mgcv_b);

    // Distinctness on the shared grid (curve_a, curve_b evaluated at identical
    // x): sin(6πx) and cos(4πx) over [0,1] are nearly uncorrelated, so a low
    // cross-group correlation proves the by-factor blocks did NOT collapse onto
    // one shared smooth. Both curves live on the same x, so a collapse would
    // force corr -> 1.
    let cross_corr = pearson(&curve_a, &curve_b);
    // Per-level fitted ranges over the shared grid, for the diagnostic line.
    let a_min = curve_a.iter().cloned().fold(f64::INFINITY, f64::min);
    let a_max = curve_a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let b_min = curve_b.iter().cloned().fold(f64::INFINITY, f64::min);
    let b_max = curve_b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);

    eprintln!(
        "tp by-factor: n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} edf_rel={edf_rel:.3}\n  \
         A: rel_l2={rel_a:.4} pearson={corr_a:.5}\n  \
         B: rel_l2={rel_b:.4} pearson={corr_b:.5}\n  \
         cross-group pearson(A,B)={cross_corr:.4} \
         (A range [{a_min:.3},{a_max:.3}] B range [{b_min:.3},{b_max:.3}])",
    );

    // (1) Per-level agreement. Both engines REML-fit the same per-level tp
    // penalized objective; at σ=0.06 / 80 pts the smooth is well-determined, so
    // each group's fitted curve must essentially coincide. 0.06 relative L2 is a
    // tight bound that still tolerates basis-convention differences (gam's
    // default tp null space vs mgcv's) while catching any real divergence.
    assert!(
        rel_a < 0.06,
        "group A fitted smooth diverges from mgcv: rel_l2={rel_a:.4} (pearson={corr_a:.5})"
    );
    assert!(
        rel_b < 0.06,
        "group B fitted smooth diverges from mgcv: rel_l2={rel_b:.4} (pearson={corr_b:.5})"
    );
    assert!(
        corr_a > 0.999,
        "group A fitted smooth should be near-identical to mgcv: pearson={corr_a:.5}"
    );
    assert!(
        corr_b > 0.999,
        "group B fitted smooth should be near-identical to mgcv: pearson={corr_b:.5}"
    );

    // (2) Same overall complexity. Per-group EDF attribution is basis/null-space
    // convention sensitive; the robust, convention-stable quantity is the TOTAL
    // EDF summed over both level blocks (plus the shared g main effect), which
    // both engines report via sum(m$edf). With matched k=15 and REML the totals
    // should agree closely; 20% relative tolerates tp null-space convention
    // differences while still rejecting a wrong penalty structure.
    assert!(
        edf_rel < 0.20,
        "total effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );

    // (3) Distinctness — the by-factor mechanism must keep the levels separate.
    // sin(6πx) vs cos(4πx) on [0,1] are nearly orthogonal; the two recovered
    // gam curves, evaluated on the SAME x grid, must have |corr| well below 1.
    // A collapse-to-shared-smooth bug would force corr -> 1.
    assert!(
        cross_corr.abs() < 0.7,
        "by-factor smooths collapsed toward a single shared curve: cross-group pearson={cross_corr:.4}"
    );
}
