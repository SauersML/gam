//! End-to-end OBJECTIVE quality: gam's 1-D Matérn/GP smooth
//! (`matern(x, nu=2.5)`) must RECOVER the known generating function on which the
//! synthetic data were built.
//!
//! The data are `y = f(x) + N(0, σ²)` for a *known* `f` (a sum of two sinusoids)
//! and `σ = 0.08`. Because the truth is known, the quality claim is TRUTH
//! RECOVERY, not "looks like mgcv": we assert
//!
//!   * PRIMARY: `RMSE(gam_fit, truth)` on a dense interior grid is below a
//!     principled bar tied to the noise level — the smooth must strip the noise
//!     and reconstruct `f`. The bar is `σ` (0.08); a faithful penalized GP fit on
//!     n=180 points recovers a smooth signal to well under one noise standard
//!     deviation. (A noisy/overfit smooth would float up toward σ or beyond.)
//!
//!   * MATCH-OR-BEAT: gam's recovery error is no worse than mgcv's GP smooth
//!     (`bs="gp"`, ν=5/2) by more than 10%: `rmse_gam <= rmse_mgcv * 1.10`. mgcv
//!     is the mature GP-smooth standard, demoted here from "ground truth" to a
//!     baseline accuracy gam must match or beat on the SAME recovery metric.
//!
//! mgcv's `bs="gp"` selects the correlation function via the FIRST element of
//! its `m` argument (`?mgcv::gp.smooth`): 1=spherical, 2=power-exponential,
//! 3=Matérn ν=3/2, 4=Matérn ν=5/2, 5=Matérn ν=7/2. We pass `m = 4` (ν=5/2) so
//! mgcv fits the same kernel family gam's `matern(x, nu=2.5)` implements, making
//! the recovery comparison apples-to-apples. Both engines select the smoothing
//! parameter by REML. We still print the gam-vs-mgcv relative-L2 for context,
//! but it is NOT the pass criterion — matching a peer tool's noisy fit proves
//! nothing about quality; reconstructing the truth does.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, r2, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

#[test]
fn gam_matern_smooth_recovers_truth() {
    init_parallelism();

    // ---- synthetic data, fed IDENTICALLY to gam and mgcv ------------------
    // x ∈ [0,1]; y = 1 + 0.8·sin(4π·x) + 0.4·cos(2π·x) + N(0, 0.08²); n=180.
    let n = 180usize;
    let mut rng = StdRng::seed_from_u64(456);
    let ux = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, 0.08).expect("gaussian noise");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let truth = |t: f64| {
        1.0 + 0.8 * (4.0 * std::f64::consts::PI * t).sin()
            + 0.4 * (2.0 * std::f64::consts::PI * t).cos()
    };
    let y: Vec<f64> = x
        .iter()
        .map(|&t| truth(t) + noise.sample(&mut rng))
        .collect();

    // ---- fit with gam: y ~ matern(x, nu=2.5, k=20), REML ------------------
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| csv::StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode matern dataset");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ matern(x, nu=2.5, k=20)", &ds, &cfg).expect("gam matern fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian GAM fit for matern() smooth");
    };

    // ---- shared dense evaluation grid -------------------------------------
    // Interior of [0,1] to avoid GP-kernel edge behavior dominating the metric.
    // The harness packs every column into one CSV/data.frame, so all columns
    // must share one length; we therefore size the grid to `n` and pass x, y,
    // and xg as three length-`n` columns. The grid is independent of the data
    // x's — it is a regular sweep of the interior — but the same grid is used by
    // both gam and mgcv, so the comparison is element-wise aligned.
    let grid_n = n;
    let x_grid: Vec<f64> = (0..grid_n)
        .map(|i| 0.005 + 0.99 * i as f64 / (grid_n - 1) as f64)
        .collect();

    // gam fitted function at the grid: rebuild the design from the frozen spec
    // (identity link ⇒ design·beta = mean). Column order matches headers: x@0.
    let mut g = Array2::<f64>::zeros((grid_n, 2));
    for (i, &t) in x_grid.iter().enumerate() {
        g[[i, 0]] = t;
        g[[i, 1]] = 0.0;
    }
    let grid_design = build_term_collection_design(g.view(), &fit.resolvedspec)
        .expect("rebuild matern design at grid points");
    let gam_grid: Vec<f64> = grid_design.design.apply(&fit.fit.beta).to_vec();

    // The KNOWN generating function at the same grid — the recovery target.
    let truth_grid: Vec<f64> = x_grid.iter().map(|&t| truth(t)).collect();

    // ---- fit the SAME model with mgcv (the mature GP-smooth reference) -----
    // bs="gp" with m=4 selects the Matérn ν=5/2 kernel (the first m entry is the
    // correlation-function index; 4 == Matérn 5/2). The range parameter is left
    // at mgcv's data-driven default. REML selects the smoothing parameter,
    // matching gam. We predict on x_grid.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("y", &y),
            Column::new("xg", &x_grid),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        # x, y, xg arrive as three aligned, all-finite columns of identical length
        # (the Rust side generated them); fit on (x, y), predict on the xg grid.
        fit_df  <- data.frame(x = df$x, y = df$y)
        grid_df <- data.frame(x = df$xg)
        m <- gam(y ~ s(x, bs = "gp", k = 20, m = 4), data = fit_df, method = "REML")
        emit("grid_fit", as.numeric(predict(m, newdata = grid_df)))
        "#,
    );
    let mgcv_grid = r.vector("grid_fit");
    assert_eq!(
        mgcv_grid.len(),
        grid_n,
        "mgcv grid prediction length mismatch"
    );

    // ---- OBJECTIVE quality: recovery of the known truth -------------------
    let noise_sigma = 0.08_f64;
    let rmse_gam = rmse(&gam_grid, &truth_grid);
    let rmse_mgcv = rmse(mgcv_grid, &truth_grid);
    // Context only (NOT a pass criterion): how close gam's fit is to mgcv's.
    let rel_to_mgcv = relative_l2(&gam_grid, mgcv_grid);

    eprintln!(
        "matern(x,nu=2.5) TRUTH RECOVERY: n={n} grid={grid_n} sigma={noise_sigma:.3} \
         rmse_gam_vs_truth={rmse_gam:.4} rmse_mgcv_vs_truth={rmse_mgcv:.4} \
         gam/mgcv_ratio={:.3} rel_l2(gam,mgcv)={rel_to_mgcv:.4}",
        rmse_gam / rmse_mgcv.max(1e-12)
    );

    // PRIMARY: gam strips the noise and reconstructs the signal. On n=180 points
    // a faithful penalized Matérn ν=5/2 GP recovers a smooth signal to well under
    // one noise standard deviation; tie the bar to σ itself.
    assert!(
        rmse_gam < noise_sigma,
        "matern smooth fails to recover the truth: rmse(gam, truth)={rmse_gam:.4} >= sigma={noise_sigma:.3}"
    );

    // MATCH-OR-BEAT: gam's recovery is no worse than the mature GP smooth
    // (mgcv bs='gp', ν=5/2) by more than 10% on the SAME recovery metric.
    assert!(
        rmse_gam <= rmse_mgcv * 1.10,
        "matern recovery worse than mgcv baseline: rmse(gam, truth)={rmse_gam:.4} > 1.10 * rmse(mgcv, truth)={:.4}",
        rmse_mgcv * 1.10
    );
}

// Source: the `quakes` dataset (1000 seismic events near Fiji, 1964), shipped by
// base R (`datasets::quakes`) and exported here to bench/datasets/quakes.csv.
// Columns: lat, long, depth, mag, stations. We model earthquake magnitude as a
// 2-D Matérn ν=5/2 spatial smooth over (long, lat).
const QUAKES_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/quakes.csv");

/// #2395: K random train/test partitions for the real-data (quakes) arm. The 2-D
/// Matérn GP fit (k=60 on ~800 rows) is the expensive one here, so K=5 keeps the
/// 2*K=10 fits near ~5x the former single-split cost, inside the current
/// slowest-normal-test envelope, while still cutting the metric's std error ~2.2x.
const K_SPLITS: usize = 5;
/// Held-out fraction per partition (~80/20, matching the former split scale).
const HOLDOUT: f64 = 0.20;

/// Deterministic uniform(0,1) hash of (row, split) via splitmix64 — row `i` is in
/// the TEST fold of partition `split` iff it maps below `HOLDOUT`. No RNG dep; gam
/// and mgcv, fed the SAME masks, partition byte-identically.
fn is_heldout(i: usize, split: usize) -> bool {
    let mut z = (i as u64)
        .wrapping_add((split as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
    u < HOLDOUT
}

/// REAL-DATA arm: the SAME 2-D Matérn ν=5/2 spatial-smooth capability, exercised
/// on `quakes` (Fiji earthquakes). The truth is unknown on real data, so quality
/// is OUT-OF-SAMPLE predictive accuracy of the spatial surface.
///
/// #2395 K-split averaging: the former single deterministic hold-out flipped the
/// gam-vs-mgcv sign across splits (single-split noise). We now score K random
/// train/test partitions and average. gam and mgcv are scored on the SAME K
/// partitions (identical 0/1 fold masks shipped into the R body):
///
///   PRIMARY (objective, tool-free): the (long, lat) Matérn surface is informative
///     (AVERAGED held-out R2 > 0) AND at least as good as the mature reference
///     (`gam_R2_avg >= mgcv_R2_avg - 0.02`). Magnitude is only weakly spatially
///     structured (mgcv itself reaches R²≈0.085), so an absolute R²≥0.10 bar would
///     measure the data, not gam — the match-or-beat philosophy is used instead.
///
///   BASELINE (match-or-beat): mgcv fits the SAME 2-D GP smooth
///     (`s(long, lat, bs="gp", m=4)`, ν=5/2) on the SAME training rows and predicts
///     the SAME held-out rows of each partition; gam's AVERAGED held-out RMSE must
///     be no worse than `mgcv_rmse_avg * 1.10`.
#[test]
fn gam_matern_smooth_recovers_truth_on_real_data() {
    init_parallelism();

    // ---- load the quakes dataset (long, lat -> mag) -----------------------
    let ds = load_csvwith_inferred_schema(Path::new(QUAKES_CSV)).expect("load quakes.csv");
    let col = ds.column_map();
    let long_idx = col["long"];
    let lat_idx = col["lat"];
    let mag_idx = col["mag"];
    let long: Vec<f64> = ds.values.column(long_idx).to_vec();
    let lat: Vec<f64> = ds.values.column(lat_idx).to_vec();
    let mag: Vec<f64> = ds.values.column(mag_idx).to_vec();
    let n = long.len();
    assert!(n > 900, "quakes should have ~1000 rows, got {n}");
    let p = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };

    let mut gam_rmses = Vec::with_capacity(K_SPLITS);
    let mut gam_r2s = Vec::with_capacity(K_SPLITS);
    let mut fold_data: Vec<Vec<f64>> = Vec::with_capacity(K_SPLITS);
    let mut fold_names: Vec<String> = Vec::with_capacity(K_SPLITS);
    let mut gam_edf_repr = f64::NAN;

    for split in 0..K_SPLITS {
        let train_rows: Vec<usize> = (0..n).filter(|&i| !is_heldout(i, split)).collect();
        let test_rows: Vec<usize> = (0..n).filter(|&i| is_heldout(i, split)).collect();
        assert!(
            train_rows.len() > 600 && test_rows.len() > 120,
            "#2395 matern split {split} degenerate: train={} test={}",
            train_rows.len(),
            test_rows.len()
        );
        let test_long: Vec<f64> = test_rows.iter().map(|&i| long[i]).collect();
        let test_lat: Vec<f64> = test_rows.iter().map(|&i| lat[i]).collect();
        let test_mag: Vec<f64> = test_rows.iter().map(|&i| mag[i]).collect();

        let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
        for (out_row, &src_row) in train_rows.iter().enumerate() {
            for c in 0..p {
                train_values[[out_row, c]] = ds.values[[src_row, c]];
            }
        }
        let mut train_ds = ds.clone();
        train_ds.values = train_values;

        let result = fit_from_formula("mag ~ matern(long, lat, nu=2.5, k=60)", &train_ds, &cfg)
            .expect("gam 2-D matern fit on quakes train");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard Gaussian GAM fit for 2-D matern() spatial smooth");
        };
        if split == 0 {
            gam_edf_repr = fit.fit.edf_total().expect("gam reports total edf");
        }

        let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
        for i in 0..test_rows.len() {
            test_grid[[i, long_idx]] = test_long[i];
            test_grid[[i, lat_idx]] = test_lat[i];
        }
        let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
            .expect("rebuild 2-D matern design at held-out quakes points");
        let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
        gam_rmses.push(rmse(&gam_test_pred, &test_mag));
        gam_r2s.push(r2(&gam_test_pred, &test_mag));

        fold_data.push(
            (0..n)
                .map(|i| if is_heldout(i, split) { 1.0 } else { 0.0 })
                .collect(),
        );
        fold_names.push(format!("fold{split}"));
    }

    // ---- mgcv on the SAME K partitions (full data.frame + K fold masks) ----
    let mut columns: Vec<Column> = vec![
        Column::new("long", &long),
        Column::new("lat", &lat),
        Column::new("mag", &mag),
    ];
    for (name, data) in fold_names.iter().zip(fold_data.iter()) {
        columns.push(Column::new(name, data));
    }
    let r = run_r(
        &columns,
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            K <- {K_SPLITS}
            rmses <- numeric(K)
            r2s <- numeric(K)
            for (s in 0:(K - 1)) {{
              fold <- df[[paste0("fold", s)]]
              tr <- data.frame(long = df$long[fold < 0.5], lat = df$lat[fold < 0.5],
                               mag = df$mag[fold < 0.5])
              te <- data.frame(long = df$long[fold >= 0.5], lat = df$lat[fold >= 0.5])
              obs <- df$mag[fold >= 0.5]
              m <- gam(mag ~ s(long, lat, bs = "gp", k = 60, m = 4), data = tr, method = "REML")
              p <- as.numeric(predict(m, newdata = te))
              rmses[s + 1] <- sqrt(mean((p - obs)^2))
              r2s[s + 1] <- 1 - sum((obs - p)^2) / sum((obs - mean(obs))^2)
            }}
            emit("mgcv_rmses", rmses)
            emit("mgcv_r2s", r2s)
            "#
        ),
    );
    let mgcv_rmses = r.vector("mgcv_rmses");
    let mgcv_r2s = r.vector("mgcv_r2s");
    assert_eq!(mgcv_rmses.len(), K_SPLITS, "mgcv per-split rmse count mismatch");
    assert_eq!(mgcv_r2s.len(), K_SPLITS, "mgcv per-split r2 count mismatch");

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let gam_rmse_avg = mean(&gam_rmses);
    let gam_r2_avg = mean(&gam_r2s);
    let mgcv_rmse_avg = mean(mgcv_rmses);
    let mgcv_r2_avg = mean(mgcv_r2s);

    eprintln!(
        "quakes matern(long,lat,nu=2.5) #2395 K={K_SPLITS}-split avg: gam_edf(split0)={gam_edf_repr:.3} \
         gam_test_R2_avg={gam_r2_avg:.4} mgcv_test_R2_avg={mgcv_r2_avg:.4} \
         gam_test_rmse_avg={gam_rmse_avg:.4} mgcv_test_rmse_avg={mgcv_rmse_avg:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_matern_smooth::test",
            "test_rmse",
            gam_rmse_avg,
            "mgcv",
            mgcv_rmse_avg,
        )
        .line()
    );

    // ---- PRIMARY: informative AND at least as good as the mature reference ---
    assert!(
        gam_r2_avg > 0.0 && gam_r2_avg >= mgcv_r2_avg - 0.02,
        "gam's averaged held-out spatial R2 {gam_r2_avg:.4} is not informative or trails mgcv \
         {mgcv_r2_avg:.4} (the mature reference) by more than 0.02"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on averaged held-out RMSE.
    assert!(
        gam_rmse_avg <= mgcv_rmse_avg * 1.10,
        "gam averaged held-out RMSE {gam_rmse_avg:.4} exceeds mgcv {mgcv_rmse_avg:.4} * 1.10"
    );

    // ---- complexity sanity: spatial edf in a sane range (not matched) ------
    assert!(
        gam_edf_repr > 1.0 && gam_edf_repr < 60.0,
        "gam effective dof out of sane range: edf(split0)={gam_edf_repr:.3}"
    );
}
