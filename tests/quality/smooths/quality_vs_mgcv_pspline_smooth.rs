//! End-to-end quality: gam's 1-D **p-spline** smooth (`bs='ps'`) is judged by
//! its OBJECTIVE held-out predictive accuracy on real data, not by how closely
//! it reproduces mgcv's fitted curve.
//!
//! mgcv's `bs='ps'` is the canonical Eilers-Marx penalized B-spline: a cubic
//! (degree-3) B-spline basis with a discrete *second-order difference* penalty
//! on adjacent coefficients (mgcv defaults `m=c(2,2)` -> degree 3, penalty
//! order 2). gam's `s(range, bs='ps')` builds exactly this basis
//! (`term_builder`: `"ps"` -> B-spline, `degree=3`, `penalty_order=2`).
//!
//! We use the canonical `lidar` smoothing benchmark (`logratio ~ range`).
//!
//! #2395 K-split averaging: the former single deterministic hold-out put the
//! gam-vs-mgcv margin on a knife-edge that flipped sign across splits (pure
//! single-split noise). We now score K random train/test partitions and average
//! the held-out metric. gam and mgcv are scored on the SAME K partitions
//! (identical 0/1 fold masks shipped into the R body, which reconstructs each
//! split internally), so the paired comparison stays honest; only the split noise
//! is averaged away. The p-spline is fit on each partition's training rows ONLY
//! and used to predict its held-out rows. Objective bars on gam's OWN averaged
//! predictions:
//!   1. AVERAGED test R^2 >= 0.55 — the smooth explains the majority of held-out
//!      variance in `logratio`.
//!   2. gam's AVERAGED held-out RMSE <= mgcv's AVERAGED held-out RMSE * 1.10 —
//!      mgcv is a BASELINE TO MATCH-OR-BEAT on out-of-sample accuracy. Averaging a
//!      lower-variance metric against the same bar is strictly harder than the
//!      former single split, never a weakening.

use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, r2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Root-mean-square error of `pred` against `truth`.
fn rmse_pair(pred: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(pred.len(), truth.len(), "rmse length mismatch");
    let n = pred.len() as f64;
    let s: f64 = pred.iter().zip(truth).map(|(p, y)| (p - y) * (p - y)).sum();
    (s / n.max(1.0)).sqrt()
}

/// #2395: K random train/test partitions per arm. The lidar ps(k=15) fit is
/// sub-second, so 2*K=20 fits stay far inside the fast envelope while cutting the
/// held-out metric's standard error ~sqrt(K)=3.2x — the noise that made this
/// near-miss flip sign across single splits.
const K_SPLITS: usize = 10;
/// Held-out fraction per partition (~80/20, matching the former split scale).
const HOLDOUT: f64 = 0.20;

/// Deterministic uniform(0,1) hash of (row, split_key) via splitmix64 — row `i`
/// is in the TEST fold of partition `split_key` iff it maps below `HOLDOUT`. No
/// RNG dep; the mask is a pure function of (i, split_key), so gam and mgcv — fed
/// the SAME masks — partition byte-identically.
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

/// Fit gam + mgcv on `K_SPLITS` random partitions of the lidar data and return
/// averaged (gam_rmse, gam_r2, mgcv_rmse) over the K held-out sets plus a
/// representative single-split edf (split 0). `seed_base` offsets the partition
/// stream so the two arms average over INDEPENDENT partition families. mgcv
/// scores the SAME K partitions: the per-row 0/1 masks are shipped as
/// `fold0..fold{K-1}` columns and R loops over them, one subprocess.
fn ps_kfold_lidar(seed_base: usize) -> (f64, f64, f64, f64) {
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

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
            "#2395 ps split {k} degenerate: train={} test={}",
            train_rows.len(),
            test_rows.len()
        );
        let range_test: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
        let logratio_test: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

        let mut train_values = Array2::<f64>::zeros((train_rows.len(), ds.headers.len()));
        for (new_row, &orig) in train_rows.iter().enumerate() {
            train_values.row_mut(new_row).assign(&ds.values.row(orig));
        }
        let train_ds = EncodedDataset {
            headers: ds.headers.clone(),
            values: train_values,
            schema: ds.schema.clone(),
            column_kinds: ds.column_kinds.clone(),
        };
        let result = fit_from_formula("logratio ~ s(range, bs='ps', k=15)", &train_ds, &cfg)
            .expect("gam p-spline fit on training rows");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for a gaussian p-spline smooth");
        };
        if k == 0 {
            gam_edf_repr = fit.fit.edf_total().expect("gam reports total edf");
        }

        let mut test_grid = Array2::<f64>::zeros((test_rows.len(), ds.headers.len()));
        for (i, &r) in range_test.iter().enumerate() {
            test_grid[[i, range_idx]] = r;
        }
        let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
            .expect("rebuild design at held-out points");
        let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
        gam_rmses.push(rmse_pair(&gam_test_pred, &logratio_test));
        gam_r2s.push(r2(&gam_test_pred, &logratio_test));

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
              m <- gam(logratio ~ s(range, bs = "ps", k = 15, m = c(2, 2)), data = tr, method = "REML")
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
fn gam_pspline_generalizes_on_lidar() {
    init_parallelism();

    let (gam_rmse, gam_r2, mgcv_rmse, gam_edf) = ps_kfold_lidar(0);
    eprintln!(
        "lidar s(range, bs='ps', k=15) #2395 K={K_SPLITS}-split avg (seed base 0): \
         gam_edf(split0)={gam_edf:.3} gam_test_R2_avg={gam_r2:.4} \
         gam_test_rmse_avg={gam_rmse:.4} mgcv_test_rmse_avg={mgcv_rmse:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_pspline_smooth",
            "test_rmse",
            gam_rmse,
            "mgcv",
            mgcv_rmse,
        )
        .line()
    );

    // (1) gam's p-spline must GENERALIZE: averaged held-out R^2 clears the floor.
    assert!(
        gam_r2 >= 0.55,
        "gam p-spline does not generalize on held-out lidar: mean test R^2 = {gam_r2:.4} (need >= 0.55)"
    );
    // (2) MATCH-OR-BEAT the mature baseline on averaged out-of-sample accuracy.
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam averaged held-out RMSE {gam_rmse:.4} exceeds mgcv {mgcv_rmse:.4} by > 10%"
    );
    // (3) sane complexity (NOT edf-matching): more than a line, well under k=15.
    assert!(
        gam_edf > 1.0 && gam_edf < 15.0,
        "gam effective complexity out of range: edf(split0) = {gam_edf:.3} (expected 1 < edf < 15)"
    );
}

/// Second real-data arm exercising the SAME cubic p-spline (`s(range, bs='ps')`)
/// capability on `lidar`, over an INDEPENDENT family of K random partitions
/// (offset partition stream) and a DIFFERENT objective lens (an absolute held-out
/// RMSE bar). The two former distinct deterministic holdouts corroborated each
/// other; the two independent K-partition families now do the same.
///
/// Dataset SOURCE: the classic LIDAR smoothing benchmark of Sigrist (1994),
/// distributed with the SemiPar R package (`data(lidar)`).
#[test]
fn gam_pspline_generalizes_on_lidar_on_real_data() {
    init_parallelism();

    let (gam_rmse, gam_r2, mgcv_rmse, gam_edf) = ps_kfold_lidar(1000);
    eprintln!(
        "lidar s(range, bs='ps', k=15) #2395 K={K_SPLITS}-split avg (seed base 1000): \
         gam_edf(split0)={gam_edf:.3} gam_test_R2_avg={gam_r2:.4} \
         gam_test_rmse_avg={gam_rmse:.4} mgcv_test_rmse_avg={mgcv_rmse:.4}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_pspline_smooth::real_data_alt",
            "test_rmse",
            gam_rmse,
            "mgcv",
            mgcv_rmse,
        )
        .line()
    );

    // (1) ABSOLUTE held-out accuracy bar (tool-free): averaged RMSE within the
    //     lidar noise floor. A flat-mean predictor scores ~0.18, so <= 0.08 is a
    //     strong, real generalization claim.
    assert!(
        gam_rmse <= 0.08,
        "gam averaged held-out RMSE {gam_rmse:.4} exceeds the absolute lidar accuracy bar (0.08)"
    );
    // (2) generalization floor on averaged explained held-out variance.
    assert!(
        gam_r2 >= 0.55,
        "gam p-spline does not generalize on held-out lidar: mean test R^2 = {gam_r2:.4} (need >= 0.55)"
    );
    // (3) MATCH-OR-BEAT the mature baseline on averaged out-of-sample RMSE.
    assert!(
        gam_rmse <= mgcv_rmse * 1.10,
        "gam averaged held-out RMSE {gam_rmse:.4} exceeds mgcv {mgcv_rmse:.4} by > 10%"
    );
    // (4) sane complexity (NOT edf-matching).
    assert!(
        gam_edf > 1.0 && gam_edf < 15.0,
        "gam effective complexity out of range: edf(split0) = {gam_edf:.3} (expected 1 < edf < 15)"
    );
}
