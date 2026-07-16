//! #1757 / #1689 (ROOT 2) — DEFAULT-RANK accuracy gate for the Duchon spatial
//! smoother.
//!
//! WHY THIS TEST EXISTS. The existing `quality_vs_mgcv_duchon_2d` /
//! `quality_vs_mgcv_duchon_regimes` tests all fit with an EXPLICIT center count
//! (`duchon(x, k=..)` / `duchon(x, z, k=..)`), so they never exercise the
//! adaptive default `default_num_centers` (`crates/gam-terms/src/basis/types.rs`)
//! that a user gets when they write `duchon(x)` / `duchon(x, z)` with no `k`.
//! That default is the subject of the #1757 (2-D duchon 14–23× slower than mgcv)
//! and #1689 (thin-plate row) perf issues: it grows the basis rank with `n`
//! (`ceil(8·d_factor·n^0.4)`, floored at `min(200, n/8)`), so a moderate-`n`
//! spatial fit carries several× mgcv's fixed low-rank `k≈27–30`, inflating every
//! inner solve.
//!
//! The perf fix is to make that default SATURATE (stop growing ~linearly with
//! `n`). But a capacity cut is only safe if it does NOT degrade accuracy — an
//! ungated default reduction could silently under-smooth complex surfaces at
//! large `n` with no test catching it. THIS test is that gate: it fits at the
//! DEFAULT rank (no `k`) and asserts, on a known truth, that the default
//! recovers the surface AND match-or-beats mgcv on accuracy-vs-truth — at both a
//! small (`n=400`) and a moderate (`n=1500`) sample size, exactly the regime the
//! reduction targets. It must pass BEFORE the reduction (the default is
//! over-provisioned, so it trivially recovers) and AFTER (the whole point). If a
//! future capacity cut goes too far, the match-or-beat assertion here goes red.
//!
//! OBJECTIVE METRIC: TRUTH RECOVERY (RMSE of gam's fitted surface against the
//! KNOWN generating function on a dense interior grid), with mgcv demoted to a
//! respected match-or-beat baseline on the SAME accuracy-vs-truth metric (within
//! a 10% margin). Same doctrine and the same principled bars as the explicit-`k`
//! siblings — the surfaces and noise levels are reused from those calibrated
//! tests, so the recovery bars carry over; only the center count is now the
//! adaptive default instead of a fixed `k`.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

// ─── 1-D: default-rank `duchon(x)` on the two-component wave ─────────────────

/// Richer-but-smooth 1-D truth reused from `quality_vs_mgcv_duchon_regimes`
/// (regime B): `sin(2πx) + 0.4·sin(6πx)`, range ≈ [-1.3, 1.3]. The 3-period
/// component genuinely needs a non-trivial basis rank, so an over-aggressive
/// default cut would show up as lost recovery here.
fn truth_1d(t: f64) -> f64 {
    let tp = std::f64::consts::PI * t;
    (2.0 * tp).sin() + 0.4 * (6.0 * tp).sin()
}

/// Fit `y ~ duchon(x)` at the DEFAULT center count (no `k`) and assert (1) truth
/// recovery below `abs_bar` and (2) match-or-beat mgcv's `s(x, bs="ds",
/// m=c(2,0))` at a strong fixed low-rank `k_mgcv`, on accuracy-vs-truth.
fn assert_duchon_1d_default_rank(n: usize, sigma: f64, seed: u64, k_mgcv: usize, abs_bar: f64) {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, sigma).expect("normal");
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth_1d(t) + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic 1d dataset");
    let col = ds.column_map();
    let x_idx = col["x"];

    // DEFAULT rank: `duchon(x)` with no `k` -> default_num_centers.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ duchon(x)", &ds, &cfg).expect("gam default-rank duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian Duchon smooth");
    };

    // Dense interior test grid (avoid extrapolation edges).
    let m = 201usize;
    let x_test: Vec<f64> = (0..m)
        .map(|i| 0.005 + 0.99 * i as f64 / (m as f64 - 1.0))
        .collect();
    let y_truth: Vec<f64> = x_test.iter().map(|&t| truth_1d(t)).collect();

    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for (i, &t) in x_test.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild default-rank Duchon design at test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // mgcv baseline on byte-identical training data, strong fixed low-rank.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&x_test);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, m));
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, m));

    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("y", &y_all),
            Column::new("is_train", &is_train),
        ],
        &format!(
            r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$is_train > 0.5, ]
        grid  <- df[df$is_train < 0.5, ]
        m <- gam(y ~ s(x, bs = "ds", k = {k_mgcv}, m = c(2, 0)), data = train, method = "REML")
        emit("fitted", as.numeric(predict(m, newdata = grid)))
        emit("edf", sum(m$edf))
        emit("sp", as.numeric(m$sp))
        emit("reml", as.numeric(m$gcv.ubre))
        "#
        ),
    );
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(mgcv_fitted.len(), m, "mgcv prediction length mismatch");
    let mgcv_edf = r.scalar("edf");
    let mgcv_sp = r.vector("sp");
    let mgcv_reml = r.scalar("reml");

    let gam_truth_rmse = rmse(&gam_fitted, &y_truth);
    let mgcv_truth_rmse = rmse(mgcv_fitted, &y_truth);
    let rel_gam_vs_mgcv = relative_l2(&gam_fitted, mgcv_fitted);

    eprintln!(
        "duchon-default-rank-1d: n={n} sigma={sigma} k_mgcv={k_mgcv} \
         gam_truth_rmse={gam_truth_rmse:.4} mgcv_truth_rmse={mgcv_truth_rmse:.4} \
         (context: rel_l2(gam,mgcv)={rel_gam_vs_mgcv:.4})"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            &format!("quality_vs_mgcv_duchon_default_rank::1d_n{n}"),
            "truth_rmse",
            gam_truth_rmse,
            "mgcv",
            mgcv_truth_rmse,
        )
        .line()
    );

    // #1561 λ-selection diagnostic (pure instrumentation; no pass criterion). gam
    // uses MORE default centers than mgcv's k here yet recovers worse, so the gap
    // is smoothing-parameter SELECTION, not capacity. Emit both sides' selected
    // smoothing state to compare gam's REML argmin against mgcv's head-to-head.
    eprintln!(
        "lambda_diag test=duchon_default_rank_1d n={n} k_mgcv={k_mgcv} \
         gam_reml={:.4} gam_edf_total={:.4} \
         gam_lambdas={:?} gam_log_lambdas={:?} \
         gam_edf_by_block={:?} gam_block_trace={:?} \
         mgcv_reml={mgcv_reml:.4} mgcv_edf={mgcv_edf:.4} mgcv_sp={mgcv_sp:?}",
        fit.fit.reml_score,
        fit.fit.edf_total().expect("gam reports total edf"),
        fit.fit.lambdas.to_vec(),
        fit.fit.log_lambdas.to_vec(),
        fit.fit.edf_by_block().to_vec(),
        fit.fit.penalty_block_trace().to_vec(),
    );

    // (1) Non-degeneracy / recovery bar (reused from the regime-B calibration).
    assert!(
        gam_truth_rmse < abs_bar,
        "default-rank 1-D Duchon failed to recover the truth: \
         RMSE-vs-truth={gam_truth_rmse:.4} (bar {abs_bar})"
    );
    // (2) Match-or-beat mgcv on accuracy-vs-truth (within 10%).
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "default-rank 1-D Duchon recovers the truth worse than mgcv: \
         gam RMSE-vs-truth={gam_truth_rmse:.4} > 1.10 * mgcv={mgcv_truth_rmse:.4}"
    );
}

#[test]
fn gam_duchon_1d_default_rank_matches_mgcv_small_n() {
    // n=400: default_num_centers(400,1) ≈ 88 centers (>> mgcv's k=30); the
    // reduction target is to shrink this without losing the recovery below.
    assert_duchon_1d_default_rank(400, 0.07, 99, 30, 0.12);
}

#[test]
fn gam_duchon_1d_default_rank_matches_mgcv_moderate_n() {
    // n=1500: default_num_centers(1500,1) ≈ 187 centers — the moderate-n regime
    // the #1689/#1757 perf issues flag. More data than n=400, same smooth truth,
    // so the recovery bar carries over; this pins that a saturating default keeps
    // accuracy here.
    assert_duchon_1d_default_rank(1500, 0.07, 100, 30, 0.12);
}

// ─── 2-D: default-rank `duchon(x, z)` on the two-bump surface ────────────────

/// Known smooth 2-D surface reused from `quality_vs_mgcv_duchon_2d`: a sum of
/// two Gaussian bumps over the unit square. Range ≈ [0, ~1.4].
fn truth_surface_2d(x: f64, z: f64) -> f64 {
    let bump = |cx: f64, cz: f64, s: f64, a: f64| {
        let d2 = (x - cx).powi(2) + (z - cz).powi(2);
        a * (-d2 / (2.0 * s * s)).exp()
    };
    bump(0.3, 0.3, 0.18, 1.0) + bump(0.7, 0.65, 0.22, 0.8)
}

/// Fit `y ~ duchon(x, z)` at the DEFAULT center count (no `k`) and assert (1)
/// truth recovery below `abs_bar` and (2) match-or-beat mgcv's `s(x, z,
/// bs="ds", k=k_mgcv, m=c(2,0))` on accuracy-vs-truth.
fn assert_duchon_2d_default_rank(n: usize, sigma: f64, seed: u64, k_mgcv: usize, abs_bar: f64) {
    init_parallelism();

    let mut rng = StdRng::seed_from_u64(seed);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, sigma).expect("normal");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(truth_surface_2d(xi, zi) + noise.sample(&mut rng));
    }

    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![x[i].to_string(), z[i].to_string(), y[i].to_string()])
        })
        .collect();
    let ds =
        encode_recordswith_inferred_schema(headers, rows).expect("encode synthetic 2d dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    // DEFAULT rank: `duchon(x, z)` with no `k` -> default_num_centers.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ duchon(x, z)", &ds, &cfg).expect("gam default-rank 2d duchon fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian 2-D Duchon smooth");
    };

    // Dense interior test grid on [0.05, 0.95]^2 (avoid extrapolation).
    let g = 25usize; // 25x25 = 625 interior grid points
    let coord = |i: usize| 0.05 + 0.90 * i as f64 / (g as f64 - 1.0);
    let mut gx = Vec::with_capacity(g * g);
    let mut gz = Vec::with_capacity(g * g);
    let mut y_truth = Vec::with_capacity(g * g);
    for i in 0..g {
        for j in 0..g {
            let (xi, zi) = (coord(i), coord(j));
            gx.push(xi);
            gz.push(zi);
            y_truth.push(truth_surface_2d(xi, zi));
        }
    }
    let m = gx.len();

    let mut grid = Array2::<f64>::zeros((m, ds.headers.len()));
    for i in 0..m {
        grid[[i, x_idx]] = gx[i];
        grid[[i, z_idx]] = gz[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild default-rank 2-D Duchon design at test grid");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // mgcv baseline on byte-identical training data, strong fixed low-rank.
    let mut x_all = x.clone();
    x_all.extend_from_slice(&gx);
    let mut z_all = z.clone();
    z_all.extend_from_slice(&gz);
    let mut y_all = y.clone();
    y_all.extend(std::iter::repeat_n(0.0, m));
    let mut is_train = vec![1.0; n];
    is_train.extend(std::iter::repeat_n(0.0, m));

    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("z", &z_all),
            Column::new("y", &y_all),
            Column::new("is_train", &is_train),
        ],
        &format!(
            r#"
        suppressPackageStartupMessages(library(mgcv))
        train <- df[df$is_train > 0.5, ]
        grid  <- df[df$is_train < 0.5, ]
        m <- gam(y ~ s(x, z, bs = "ds", k = {k_mgcv}, m = c(2, 0)), data = train, method = "REML")
        emit("fitted", as.numeric(predict(m, newdata = grid)))
        emit("edf", sum(m$edf))
        emit("sp", as.numeric(m$sp))
        emit("reml", as.numeric(m$gcv.ubre))
        "#
        ),
    );
    let mgcv_fitted = r.vector("fitted");
    assert_eq!(mgcv_fitted.len(), m, "mgcv prediction length mismatch");
    let mgcv_edf = r.scalar("edf");
    let mgcv_sp = r.vector("sp");
    let mgcv_reml = r.scalar("reml");

    let gam_truth_rmse = rmse(&gam_fitted, &y_truth);
    let mgcv_truth_rmse = rmse(mgcv_fitted, &y_truth);
    let rel_gam_vs_mgcv = relative_l2(&gam_fitted, mgcv_fitted);
    let rms_truth = (y_truth.iter().map(|t| t * t).sum::<f64>() / y_truth.len() as f64).sqrt();

    eprintln!(
        "duchon-default-rank-2d: n={n} grid={m} sigma={sigma} k_mgcv={k_mgcv} \
         gam_truth_rmse={gam_truth_rmse:.4} mgcv_truth_rmse={mgcv_truth_rmse:.4} \
         rms_truth={rms_truth:.4} (context: rel_l2(gam,mgcv)={rel_gam_vs_mgcv:.4})"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            &format!("quality_vs_mgcv_duchon_default_rank::2d_n{n}"),
            "truth_rmse",
            gam_truth_rmse,
            "mgcv",
            mgcv_truth_rmse,
        )
        .line()
    );

    // #1561 λ-selection diagnostic (pure instrumentation; no pass criterion). Emit
    // both sides' selected smoothing state so the artifact lets us compare gam's
    // REML argmin against mgcv's head-to-head on the 2-D Duchon surface.
    eprintln!(
        "lambda_diag test=duchon_default_rank_2d n={n} k_mgcv={k_mgcv} \
         gam_reml={:.4} gam_edf_total={:.4} \
         gam_lambdas={:?} gam_log_lambdas={:?} \
         gam_edf_by_block={:?} gam_block_trace={:?} \
         mgcv_reml={mgcv_reml:.4} mgcv_edf={mgcv_edf:.4} mgcv_sp={mgcv_sp:?}",
        fit.fit.reml_score,
        fit.fit.edf_total().expect("gam reports total edf"),
        fit.fit.lambdas.to_vec(),
        fit.fit.log_lambdas.to_vec(),
        fit.fit.edf_by_block().to_vec(),
        fit.fit.penalty_block_trace().to_vec(),
    );

    // (1) Non-degeneracy / recovery bar (reused from the k=49 2-D calibration).
    assert!(
        gam_truth_rmse < abs_bar,
        "default-rank 2-D Duchon failed to recover the surface: \
         RMSE-vs-truth={gam_truth_rmse:.4} (signal RMS≈{rms_truth:.4}, bar {abs_bar})"
    );
    // (2) Match-or-beat mgcv on accuracy-vs-truth (within 10%).
    assert!(
        gam_truth_rmse <= mgcv_truth_rmse * 1.10,
        "default-rank 2-D Duchon recovers the truth worse than mgcv: \
         gam RMSE-vs-truth={gam_truth_rmse:.4} > 1.10 * mgcv={mgcv_truth_rmse:.4}"
    );
}

#[test]
fn gam_duchon_2d_default_rank_matches_mgcv_small_n() {
    // n=400: default_num_centers(400,2) ≈ 100 centers (~2× mgcv's k=49).
    assert_duchon_2d_default_rank(400, 0.10, 20260530, 49, 0.15);
}

#[test]
fn gam_duchon_2d_default_rank_matches_mgcv_moderate_n() {
    // n=1500: default_num_centers(1500,2) ≈ 187 centers — the moderate-n regime
    // the #1757 14–23× slowdown is measured in. Same surface/noise as n=400, so
    // the recovery bar carries over; this pins that a saturating default keeps
    // 2-D accuracy at the sample size where the perf gap is worst.
    assert_duchon_2d_default_rank(1500, 0.10, 20260531, 49, 0.15);
}
