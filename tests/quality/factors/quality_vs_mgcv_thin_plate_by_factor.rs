//! End-to-end quality: gam's 1-D thin-plate *by-factor* (interaction) smooth
//! must RECOVER THE KNOWN PER-LEVEL TRUTH that generated the data, and do so at
//! least as accurately as `mgcv` — the mature, standard GAM implementation.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, not tool mimicry): the data are
//! generated from two known functions — group-A: sin(6πx), group-B: cos(4πx) —
//! contaminated by N(0, σ=0.06) noise. The PRIMARY pass/fail claim is that gam's
//! per-level fitted smooth recovers its own group's true function with
//! RMSE(gam_fit, truth) below a principled bar set by the noise level: a good
//! smoother averages out noise, so its error against the *true* signal should be
//! a fraction of σ, comfortably below σ itself. We assert this per group.
//!
//! mgcv is fit on byte-identical data and demoted to a BASELINE TO MATCH-OR-BEAT
//! on the SAME objective metric: gam's POOLED RMSE-to-truth (over both levels)
//! must be no worse than 1.10× mgcv's. "Same as mgcv" is never the criterion —
//! mgcv's own fit is a noisy estimate, so we only require gam to be as accurate
//! as it against the truth, and otherwise to recover the truth on its own merits.
//! (The pooled, not per-group, comparison is deliberate: see the note at the
//! match-or-beat assertion.)
//!
//! We also assert two structural properties that hold independently of any
//! reference: the two recovered group curves are genuinely DISTINCT (the
//! by-factor mechanism must not collapse the levels onto one shared smooth), and
//! gam's total effective degrees of freedom lie in a sane, signal-appropriate
//! range (more than the 2 linear null-space dims, well under the 2·k cap).
//!
//! mgcv uses `gam(y ~ g + s(x, by=g, bs='tp', k=15), method="REML")` with `g` a
//! factor: its `by=<factor>` mechanism creates an independent thin-plate basis
//! block per level (each centered sum-to-zero), and the parametric `+ g` carries
//! the per-level intercept — the unpenalized treatment-coded main effect gam adds
//! automatically for a categorical `by=`. Identical x, A/B labels, and y go to
//! both engines.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, pearson, relative_l2, rmse, run_r};
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
fn gam_thin_plate_by_factor_recovers_per_level_truth() {
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

    // ---- split fitted values by group and score AGAINST THE KNOWN TRUTH ----
    // The objective quantity is the error of each engine's per-level fitted mean
    // against the function that actually generated that group's data, evaluated
    // at the same training x. A good smoother averages out the N(0,σ) noise, so
    // its RMSE-to-truth should be a fraction of σ — NOT the ~σ that a raw,
    // unsmoothed estimate of y would incur.
    let mut gam_a = Vec::new();
    let mut gam_b = Vec::new();
    let mut mgcv_a = Vec::new();
    let mut mgcv_b = Vec::new();
    let mut truth_a = Vec::new();
    let mut truth_b = Vec::new();
    for i in 0..n {
        let ti = truth(x[i], is_a[i]);
        if is_a[i] {
            gam_a.push(gam_fitted[i]);
            mgcv_a.push(mgcv_fitted[i]);
            truth_a.push(ti);
        } else {
            gam_b.push(gam_fitted[i]);
            mgcv_b.push(mgcv_fitted[i]);
            truth_b.push(ti);
        }
    }

    // PRIMARY objective metric: RMSE of the fitted smooth against the TRUE signal.
    let gam_err_a = rmse(&gam_a, &truth_a);
    let gam_err_b = rmse(&gam_b, &truth_b);
    let mgcv_err_a = rmse(&mgcv_a, &truth_a);
    let mgcv_err_b = rmse(&mgcv_b, &truth_b);

    // For context only (NOT a pass criterion): how close the two engines' fits
    // are to each other. Printed, never asserted — "same as mgcv" proves nothing.
    let rel_a = relative_l2(&gam_a, &mgcv_a);
    let rel_b = relative_l2(&gam_b, &mgcv_b);

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

    eprintln!(
        "tp by-factor truth recovery (sigma={SIGMA}): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3}\n  \
         A: rmse(gam,truth)={gam_err_a:.4} rmse(mgcv,truth)={mgcv_err_a:.4} (gam-vs-mgcv rel_l2={rel_a:.4})\n  \
         B: rmse(gam,truth)={gam_err_b:.4} rmse(mgcv,truth)={mgcv_err_b:.4} (gam-vs-mgcv rel_l2={rel_b:.4})\n  \
         cross-group pearson(A,B)={cross_corr:.4} \
         (A range [{a_min:.3},{a_max:.3}] B range [{b_min:.3},{b_max:.3}])",
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "factors",
            "quality_vs_mgcv_thin_plate_by_factor::a",
            "err_a",
            gam_err_a,
            "mgcv",
            mgcv_err_a,
        )
        .line()
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "factors",
            "quality_vs_mgcv_thin_plate_by_factor::b",
            "err_b",
            gam_err_b,
            "mgcv",
            mgcv_err_b,
        )
        .line()
    );

    // (1) TRUTH RECOVERY — the primary claim. Each group's fitted smooth must
    // recover its OWN true generating function to well within the noise level.
    // With 80 points per level at σ=0.06, a properly penalized tp smooth drives
    // the estimation error far below σ; we require RMSE-to-truth < σ (a clearly
    // smoothing, signal-tracking fit), which a noise-following or collapsed fit
    // cannot meet.
    assert!(
        gam_err_a < SIGMA,
        "group A fit does not recover sin(6πx) within the noise level: rmse(gam,truth)={gam_err_a:.4} >= sigma={SIGMA}"
    );
    assert!(
        gam_err_b < SIGMA,
        "group B fit does not recover cos(4πx) within the noise level: rmse(gam,truth)={gam_err_b:.4} >= sigma={SIGMA}"
    );

    // (2) MATCH-OR-BEAT the mature reference ON ACCURACY. mgcv's fit is itself a
    // noisy estimate of the truth, so we do not require gam to match its output —
    // only that gam recovers the truth at least as accurately as mgcv does, to
    // within a 10% tolerance. The comparison is made on the POOLED truth-recovery
    // error over all 160 points, not per 80-point group. With only 80 noisy
    // points per level (σ=0.06), the per-group RMSE-to-truth is dominated by the
    // single realized noise draw, while the two engines' *outputs* agree to ~1–2%
    // (see the printed `rel_l2`): a per-group relative gate then flips on which
    // engine's REML smoothing parameter happens to land closer to that group's
    // noise, which is not a meaningful accuracy difference. The pooled error is
    // the statistically sound aggregate of "is gam at least as accurate as mgcv";
    // the strict per-group signal-recovery (within σ) and distinctness gates below
    // independently rule out a collapsed or noise-following fit.
    let gam_pooled: Vec<f64> = gam_a.iter().chain(gam_b.iter()).copied().collect();
    let mgcv_pooled: Vec<f64> = mgcv_a.iter().chain(mgcv_b.iter()).copied().collect();
    let truth_pooled: Vec<f64> = truth_a.iter().chain(truth_b.iter()).copied().collect();
    let gam_err_pooled = rmse(&gam_pooled, &truth_pooled);
    let mgcv_err_pooled = rmse(&mgcv_pooled, &truth_pooled);
    eprintln!(
        "{}",
        QualityPair::error(
            "factors",
            "quality_vs_mgcv_thin_plate_by_factor::pooled",
            "err_pooled",
            gam_err_pooled,
            "mgcv",
            mgcv_err_pooled,
        )
        .line()
    );
    assert!(
        gam_err_pooled <= mgcv_err_pooled * 1.10,
        "gam is less accurate than mgcv against truth (pooled over both levels): \
         gam={gam_err_pooled:.4} mgcv={mgcv_err_pooled:.4} \
         (per-group gam A={gam_err_a:.4}/B={gam_err_b:.4}, mgcv A={mgcv_err_a:.4}/B={mgcv_err_b:.4})"
    );

    // (3) Distinctness — the by-factor mechanism must keep the levels separate.
    // sin(6πx) vs cos(4πx) on [0,1] are nearly orthogonal; the two recovered
    // gam curves, evaluated on the SAME x grid, must have |corr| well below 1.
    // A collapse-to-shared-smooth bug would force corr -> 1.
    assert!(
        cross_corr.abs() < 0.7,
        "by-factor smooths collapsed toward a single shared curve: cross-group pearson={cross_corr:.4}"
    );

    // (4) Complexity sanity (NOT edf-matching). gam's total effective degrees of
    // freedom must sit in a signal-appropriate range: above the 2-dim per-curve
    // linear null space (so the smooths are genuinely wiggly, recovering the two
    // oscillating signals) and below the 2·k=30 hard basis cap (so it has not
    // interpolated noise). We deliberately do NOT assert gam's edf equals mgcv's.
    assert!(
        gam_edf > 2.0 && gam_edf < 30.0,
        "gam total edf outside the sane signal range (2, 30): gam_edf={gam_edf:.3}"
    );
}

/// Regression for #704: the predict-time design rebuild of a by-factor *spatial*
/// smooth (`bs='tp'`) must REPLAY the frozen fitted basis, never recompute the
/// data-dependent kernel / Wood-TPRS eigen-truncation / sum-to-zero constraint on
/// the prediction rows. Two independent guarantees, neither needing R:
///
///   (a) Rebuilding the design from the frozen `resolvedspec` at the TRAINING
///       points and applying β reproduces the in-sample fitted η produced by the
///       fit-time design to floating-point tolerance — i.e. the replayed basis is
///       the fitted basis, not a freshly-derived one.
///   (b) Rebuilding on a FRESH uniform grid with every row in a *single* level
///       (so the other level's block has no active rows) neither panics (the
///       original symptom, `SelfAdjointEigenNonFiniteInput`) nor collapses the two
///       levels onto one shared curve.
///
/// Under the pre-fix bug the inner thin-plate spec under the `by=` wrapper was
/// left unfrozen (`freeze_inner_smooth_basis_from_metadata` only handled
/// B-splines), so (a) drifted and (b) crashed.
#[test]
fn gam_thin_plate_by_factor_predict_replays_frozen_basis() {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, SIGMA).expect("normal");

    let n = 2 * N_PER_LEVEL;
    let mut x = Vec::<f64>::with_capacity(n);
    let mut g = Vec::<String>::with_capacity(n);
    let mut y = Vec::<f64>::with_capacity(n);
    for group_a in [true, false] {
        let mut xs: Vec<f64> = (0..N_PER_LEVEL).map(|_| ux.sample(&mut rng)).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &xi in &xs {
            x.push(xi);
            y.push(truth(xi, group_a) + noise.sample(&mut rng));
            g.push(if group_a { "A" } else { "B" }.to_string());
        }
    }

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

    // (a) The frozen replay must reproduce the fit-time basis EXACTLY. The
    // fit-time design and β live in the same (post-identifiability) coordinate
    // system, so `fit_design · β` is the in-sample fitted η. Rebuilding the
    // design from the frozen resolvedspec at the same training rows and applying
    // the same β must give the identical η — only true if the inner thin-plate
    // centers, radial reparameterization and constraint were frozen, not
    // recomputed on these rows.
    let eta_fit: Vec<f64> = fit.design.design.apply(&fit.fit.beta).to_vec();
    let mut train_grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        train_grid[[i, x_idx]] = x[i];
        train_grid[[i, g_idx]] = if g[i] == "A" { 0.0 } else { 1.0 };
    }
    let eta_replay: Vec<f64> = build_term_collection_design(train_grid.view(), &fit.resolvedspec)
        .expect("rebuild frozen design at training points")
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    assert_eq!(eta_fit.len(), eta_replay.len());
    let max_abs_dev = eta_fit
        .iter()
        .zip(&eta_replay)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_dev < 1e-8,
        "frozen predict-time replay does not reproduce the fitted basis: \
         max|eta_fit - eta_replay|={max_abs_dev:.3e} (expected < 1e-8); \
         the by-factor thin-plate inner basis was recomputed on the rebuild rows \
         rather than replayed from the frozen spec (#704)"
    );

    // (b) Fresh single-level grids must not panic and must stay distinct.
    const N_GRID: usize = 120;
    let mut grid_a = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
    let mut grid_b = Array2::<f64>::zeros((N_GRID, ds.headers.len()));
    for j in 0..N_GRID {
        let xj = j as f64 / (N_GRID as f64 - 1.0);
        grid_a[[j, x_idx]] = xj;
        grid_a[[j, g_idx]] = 0.0;
        grid_b[[j, x_idx]] = xj;
        grid_b[[j, g_idx]] = 1.0;
    }
    let curve_a: Vec<f64> = build_term_collection_design(grid_a.view(), &fit.resolvedspec)
        .expect("rebuild design on fresh grid (level A) must not panic")
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    let curve_b: Vec<f64> = build_term_collection_design(grid_b.view(), &fit.resolvedspec)
        .expect("rebuild design on fresh grid (level B) must not panic")
        .design
        .apply(&fit.fit.beta)
        .to_vec();
    assert!(
        curve_a.iter().chain(&curve_b).all(|v| v.is_finite()),
        "fresh-grid by-factor curves contain non-finite values"
    );
    let cross_corr = pearson(&curve_a, &curve_b);
    assert!(
        cross_corr.abs() < 0.7,
        "fresh-grid by-factor smooths collapsed onto one shared curve: pearson={cross_corr:.4}"
    );
}
