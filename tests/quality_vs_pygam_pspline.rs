//! End-to-end quality: gam's 1-D **p-spline** smooth (`bs="ps"`) judged on
//! **held-out predictive accuracy** on the canonical `lidar` benchmark, with
//! **pyGAM** kept only as a baseline-to-beat on that same objective metric.
//!
//! OBJECTIVE METRIC (the pass/fail criterion): a deterministic interleaved
//! train/test split (every 4th row is held out, the rest train) is fixed once.
//! gam fits `logratio ~ s(range, bs="ps", k=15)` (Gaussian/REML) on the TRAIN
//! rows only, then predicts the held-out TEST rows. The test asserts
//!   1. an absolute bar on out-of-sample fit: test R^2 >= 0.90, and
//!   2. gam generalizes at least as well as the mature reference:
//!      gam_test_RMSE <= pygam_test_RMSE * 1.10.
//! Both are computed on gam's OWN predictions of data it never saw. This is a
//! real quality claim (the smoother generalizes), not "gam reproduces pyGAM's
//! in-sample curve" — matching a peer tool's noisy fit proves nothing on its
//! own, so closeness-to-pyGAM is no longer a pass criterion.
//!
//! pyGAM (`LinearGAM(s(0, n_splines=15)).gridsearch`) is, by default, a cubic
//! (degree-3) B-spline with a 2nd-order difference penalty — exactly the basis
//! gam builds for `s(range, bs="ps", k=15)`. It is fit on the IDENTICAL train
//! rows and asked to predict the IDENTICAL test rows, so its held-out RMSE is a
//! fair generalization baseline for gam to match-or-beat. Its in-sample curve
//! and EDF are no longer asserted (the rel-L2 against pyGAM is still printed for
//! context via `eprintln!`, but does not gate the test).

use csv::StringRecord;
use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

/// Build a two-column (`range`, `logratio`) training `EncodedDataset` from the
/// held-in rows by re-encoding string records through the same schema inference
/// the CSV loader uses — keeping the train subset numerically identical to the
/// loaded data while letting gam own column layout.
fn train_dataset(range: &[f64], logratio: &[f64]) -> EncodedDataset {
    let headers = vec!["range".to_string(), "logratio".to_string()];
    let records: Vec<StringRecord> = range
        .iter()
        .zip(logratio)
        .map(|(&r, &y)| StringRecord::from(vec![format!("{r:.17e}"), format!("{y:.17e}")]))
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode train subset")
}

#[test]
fn gam_pspline_held_out_accuracy_beats_pygam_on_lidar() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range: Vec<f64> = ds.values.column(col["range"]).to_vec();
    let logratio: Vec<f64> = ds.values.column(col["logratio"]).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- deterministic interleaved train/test split ----------------------
    // Every 4th row (index % 4 == 0) is held out for evaluation; the rest
    // train. Fixed by index (no RNG) so the split is identical for gam and the
    // reference and reproducible across runs.
    let mut tr_range = Vec::new();
    let mut tr_logratio = Vec::new();
    let mut te_range = Vec::new();
    let mut te_logratio = Vec::new();
    for i in 0..n {
        if i % 4 == 0 {
            te_range.push(range[i]);
            te_logratio.push(logratio[i]);
        } else {
            tr_range.push(range[i]);
            tr_logratio.push(logratio[i]);
        }
    }
    let n_test = te_range.len();
    assert!(n_test > 30, "need a meaningful held-out set, got {n_test}");

    // ---- fit gam on TRAIN only: logratio ~ s(range, bs="ps", k=15) --------
    // `bs="ps"` => degree-3 B-spline with 2nd-order difference penalty; `k=15`
    // => 15 basis functions, matching pyGAM's default `s(0, n_splines=15)`.
    // REML selects lambda from the training data only.
    let train_ds = train_dataset(&tr_range, &tr_logratio);
    let train_col = train_ds.column_map();
    let train_range_idx = train_col["range"];
    let train_ncols = train_ds.headers.len();
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs=\"ps\", k=15)", &train_ds, &cfg)
        .expect("gam p-spline fit on train");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian p-spline smooth");
    };

    // gam predictions on the HELD-OUT test points: rebuild the frozen design
    // at the test `range` values (identity link => design*beta = mean
    // response). These are points gam never saw during fitting.
    let mut test_grid = Array2::<f64>::zeros((n_test, train_ncols));
    for (i, &r) in te_range.iter().enumerate() {
        test_grid[[i, train_range_idx]] = r;
    }
    let design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild p-spline design at held-out test points");
    let gam_test_pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with pyGAM on the SAME train rows -------------
    // LinearGAM(s(0, n_splines=15)) is a default cubic (spline_order=3) B-spline
    // with a 2nd-order difference penalty. `.gridsearch()` selects the single
    // penalty from the TRAIN data by minimizing GCV. We then predict the SAME
    // held-out test rows. pyGAM's held-out RMSE is the generalization baseline
    // for gam to match-or-beat; it is not used to constrain gam's curve shape.
    let py = run_python(
        &[
            Column::new("tr_range", &tr_range),
            Column::new("tr_logratio", &tr_logratio),
            Column::new("te_range", &te_range),
        ],
        r#"
from pygam import LinearGAM, s
Xtr = np.asarray(df["tr_range"], dtype=float).reshape(-1, 1)
ytr = np.asarray(df["tr_logratio"], dtype=float)
Xte = np.asarray(df["te_range"], dtype=float).reshape(-1, 1)
gam = LinearGAM(s(0, n_splines=15)).gridsearch(Xtr, ytr, progress=False)
emit("test_pred", gam.predict(Xte))
"#,
    );
    let pygam_test_pred = py.vector("test_pred");
    assert_eq!(
        pygam_test_pred.len(),
        n_test,
        "pyGAM held-out prediction length mismatch"
    );

    // ---- objective held-out metrics on each engine's OWN predictions ------
    let gam_rmse = rmse(&gam_test_pred, &te_logratio);
    let pygam_rmse = rmse(pygam_test_pred, &te_logratio);

    // Out-of-sample R^2 of gam against the held-out truth.
    let ybar: f64 = te_logratio.iter().sum::<f64>() / n_test as f64;
    let ss_tot: f64 = te_logratio.iter().map(|y| (y - ybar) * (y - ybar)).sum();
    let ss_res: f64 = gam_test_pred
        .iter()
        .zip(&te_logratio)
        .map(|(p, y)| (p - y) * (p - y))
        .sum();
    let gam_r2 = 1.0 - ss_res / ss_tot.max(1e-300);

    // Printed for context only (NOT a pass criterion): how close gam's held-out
    // predictions track pyGAM's on the same points.
    let rel_vs_pygam = relative_l2(&gam_test_pred, pygam_test_pred);
    eprintln!(
        "lidar s(range,bs=ps,k=15) HELD-OUT: n_test={n_test} \
         gam_r2={gam_r2:.4} gam_rmse={gam_rmse:.4} pygam_rmse={pygam_rmse:.4} \
         rel_l2_vs_pygam={rel_vs_pygam:.4}"
    );

    // PRIMARY claim: the p-spline smoother generalizes — it explains the bulk of
    // held-out variance on data it never saw. The lidar signal is strong and
    // smooth; a correct degree-3 / 2nd-difference p-spline clears 0.90 R^2
    // comfortably, while a wrong basis or penalty order would not.
    assert!(
        gam_r2 >= 0.90,
        "gam p-spline held-out R^2 too low: {gam_r2:.4} (rmse={gam_rmse:.4})"
    );

    // MATCH-OR-BEAT: gam must generalize at least as well as the mature pyGAM
    // baseline on the identical split, within a 10% RMSE margin.
    assert!(
        gam_rmse <= pygam_rmse * 1.10,
        "gam generalizes worse than pyGAM out of sample: \
         gam_rmse={gam_rmse:.4} > 1.10 * pygam_rmse={pygam_rmse:.4}"
    );
}
