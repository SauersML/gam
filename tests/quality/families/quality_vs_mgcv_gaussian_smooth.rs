//! End-to-end quality: gam's penalized Gaussian smooth must PREDICT well on
//! held-out data — not merely reproduce mgcv's in-sample fit.
//!
//! The lidar benchmark (`logratio ~ s(range)`) is real data with no known
//! ground-truth function, so the objective quality of a smoother is its
//! out-of-sample predictive accuracy.
//!
//! #2395 K-split averaging: the former single deterministic hold-out put the
//! gam-vs-mgcv margin on a knife-edge that flipped sign across splits (pure
//! single-split noise). We now score K random train/test partitions and average
//! the held-out metric. gam and mgcv are scored on the SAME K partitions
//! (identical 0/1 fold masks shipped into the R body), so the paired comparison
//! stays honest; only the split noise is averaged away. Objective bars on gam's
//! OWN averaged predictions:
//!
//!   PRIMARY (objective, tool-free): AVERAGED held-out `test_R2 >= 0.55` — gam's
//!     smooth genuinely explains held-out variance, well above the constant-mean
//!     predictor (R2 = 0).
//!
//!   BASELINE (match-or-beat): mgcv fits the SAME training rows and predicts the
//!     SAME held-out rows of each partition; gam's AVERAGED held-out RMSE must be
//!     no worse than `mgcv_rmse_avg * 1.10`. Averaging a lower-variance metric
//!     against the same bar is strictly harder than the former single split.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, r2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// #2395: K random train/test partitions per arm. The lidar smooth fit is
/// sub-second, so 2*K=20 fits stay far inside the fast envelope while cutting the
/// held-out metric's standard error ~sqrt(K)=3.2x.
const K_SPLITS: usize = 10;
/// Held-out fraction per partition (~80/20, matching the former split scale).
const HOLDOUT: f64 = 0.20;

/// Deterministic uniform(0,1) hash of (row, split_key) via splitmix64 — row `i`
/// is in the TEST fold of partition `split_key` iff it maps below `HOLDOUT`. No
/// RNG dep; gam and mgcv, fed the SAME masks, partition byte-identically.
fn is_heldout(i: usize, split_key: usize) -> bool {
    let mut z = (i as u64)
        .wrapping_add((split_key as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
    u < HOLDOUT
}

/// Fit gam (`gam_formula`) + mgcv (`mgcv_formula`) on `K_SPLITS` random partitions
/// of the lidar data and return averaged (gam_rmse, gam_r2, mgcv_rmse) plus a
/// representative single-split edf (split 0). `seed_base` offsets the partition
/// stream so the two arms average over INDEPENDENT partition families. mgcv scores
/// the SAME K partitions: the per-row 0/1 masks are shipped as `fold0..fold{K-1}`
/// columns and R loops over them, one subprocess.
fn gs_kfold_lidar(gam_formula: &str, mgcv_formula: &str, seed_base: usize) -> (f64, f64, f64, f64) {
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");
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

    for k in 0..K_SPLITS {
        let split_key = seed_base + k;
        let train_rows: Vec<usize> = (0..n).filter(|&i| !is_heldout(i, split_key)).collect();
        let test_rows: Vec<usize> = (0..n).filter(|&i| is_heldout(i, split_key)).collect();
        assert!(
            train_rows.len() > 100 && test_rows.len() > 20,
            "#2395 gaussian_smooth split {k} degenerate: train={} test={}",
            train_rows.len(),
            test_rows.len()
        );
        let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
        let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

        let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
        for (out_row, &src_row) in train_rows.iter().enumerate() {
            for c in 0..p {
                train_values[[out_row, c]] = ds.values[[src_row, c]];
            }
        }
        let mut train_ds = ds.clone();
        train_ds.values = train_values;

        let result = fit_from_formula(gam_formula, &train_ds, &cfg).expect("gam fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit");
        };
        if k == 0 {
            gam_edf_repr = fit.fit.edf_total().expect("gam reports total edf");
        }

        let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
        for (i, &r) in test_range.iter().enumerate() {
            test_grid[[i, range_idx]] = r;
        }
        let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
            .expect("rebuild design at held-out points");
        let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
        gam_rmses.push(rmse(&gam_test_pred, &test_logratio));
        gam_r2s.push(r2(&gam_test_pred, &test_logratio));

        fold_data.push(
            (0..n)
                .map(|i| if is_heldout(i, split_key) { 1.0 } else { 0.0 })
                .collect(),
        );
        fold_names.push(format!("fold{k}"));
    }

    let mut columns: Vec<Column> = vec![
        Column::new("range", &range),
        Column::new("logratio", &logratio),
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
            for (k in 0:(K - 1)) {{
              fold <- df[[paste0("fold", k)]]
              tr <- data.frame(range = df$range[fold < 0.5], logratio = df$logratio[fold < 0.5])
              te <- data.frame(range = df$range[fold >= 0.5])
              obs <- df$logratio[fold >= 0.5]
              m <- gam({mgcv_formula}, data = tr, method = "REML")
              p <- as.numeric(predict(m, newdata = te))
              rmses[k + 1] <- sqrt(mean((p - obs)^2))
            }}
            emit("mgcv_rmses", rmses)
            "#
        ),
    );
    let mgcv_rmses = r.vector("mgcv_rmses");
    assert_eq!(
        mgcv_rmses.len(),
        K_SPLITS,
        "mgcv per-split rmse count mismatch"
    );

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    (
        mean(&gam_rmses),
        mean(&gam_r2s),
        mean(mgcv_rmses),
        gam_edf_repr,
    )
}

#[test]
fn gam_smooth_predicts_lidar_better_than_baseline() {
    init_parallelism();

    let (gam_rmse, gam_r2, mgcv_rmse, gam_edf) =
        gs_kfold_lidar("logratio ~ s(range)", "logratio ~ s(range)", 0);
    eprintln!(
        "lidar s(range) #2395 K={K_SPLITS}-split avg (seed base 0): gam_edf(split0)={gam_edf:.3} \
         gam_test_R2_avg={gam_r2:.4} gam_test_rmse_avg={gam_rmse:.4} mgcv_test_rmse_avg={mgcv_rmse:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_mgcv_gaussian_smooth::default_basis",
            "test_rmse",
            gam_rmse,
            "mgcv",
            mgcv_rmse,
        )
        .line()
    );

    assert!(
        gam_r2 >= 0.55,
        "gam's averaged held-out predictive R2 too low: {gam_r2:.4} (< 0.55)"
    );
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam averaged held-out RMSE {gam_rmse:.4} exceeds mgcv {mgcv_rmse:.4} * 1.10"
    );
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: edf(split0)={gam_edf:.3}"
    );
}

/// Real-data arm for the SAME Gaussian smooth capability using gam's explicit
/// P-spline basis `s(range, bs="ps")`, over an INDEPENDENT family of K random
/// partitions (offset partition stream) so the two arms corroborate rather than
/// duplicate.
///
/// Dataset SOURCE: `bench/datasets/lidar.csv` — the classic LIDAR scatterplot
/// distributed with R's `SemiPar` package (Ruppert, Wand & Carroll,
/// *Semiparametric Regression*, 2003). Real measurements; objective quality is
/// held-out predictive accuracy.
#[test]
fn gam_smooth_predicts_lidar_better_than_baseline_on_real_data() {
    init_parallelism();

    let (gam_rmse, gam_r2, mgcv_rmse, gam_edf) = gs_kfold_lidar(
        "logratio ~ s(range, bs=\"ps\")",
        "logratio ~ s(range, bs = \"ps\")",
        1000,
    );
    eprintln!(
        "lidar s(range,bs=ps) #2395 K={K_SPLITS}-split avg (seed base 1000): \
         gam_edf(split0)={gam_edf:.3} gam_test_R2_avg={gam_r2:.4} \
         gam_test_rmse_avg={gam_rmse:.4} mgcv_test_rmse_avg={mgcv_rmse:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "families",
            "quality_vs_mgcv_gaussian_smooth::ps_basis",
            "test_rmse",
            gam_rmse,
            "mgcv",
            mgcv_rmse,
        )
        .line()
    );

    assert!(
        gam_r2 >= 0.55,
        "gam's averaged held-out predictive R2 too low: {gam_r2:.4} (< 0.55)"
    );
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam averaged held-out RMSE {gam_rmse:.4} exceeds mgcv {mgcv_rmse:.4} * 1.10"
    );
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: edf(split0)={gam_edf:.3}"
    );
}
