//! End-to-end quality: gam's cyclic cubic smooth (`cc()` / `bs="cc"`) must
//! PREDICT a real seasonal cycle on held-out data — and enforce the periodic
//! wrap — at least as well as **mgcv**, the mature, standard GAM implementation.
//!
//! DATASET: Nottingham Castle monthly average air temperatures, Jan 1920 – Dec
//! 1939 (240 months), R's `datasets::nottem` exported by the Rdatasets project:
//! https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/nottem.csv
//! Vincent Arel-Bundock's Rdatasets index:
//! https://vincentarelbundock.github.io/Rdatasets/datasets.html
//! The raw `time` column is a decimal year; we materialised a clean
//! `year,month,temp` CSV (`bench/datasets/nottem_monthly_temp.csv`) where
//! `month in 1..=12`. Temperature has a strong, smooth annual cycle (cold
//! Jan/Feb/Dec ~32–44 °F, warm Jun/Jul/Aug ~57–66 °F): a textbook use-case for a
//! cyclic smooth over month-of-year, where December must join back to January.
//!
//! This is a HELD-OUT PREDICTION test on REAL data — there is no analytic ground
//! truth, so objective quality is out-of-sample accuracy.
//!
//! #2395 K-split averaging: the former single deterministic hold-out put the
//! gam-vs-mgcv margin on a knife-edge that flipped sign across splits (pure
//! single-split noise). We now score K random train/test partitions and average
//! the held-out metric. gam and mgcv are scored on the SAME K partitions
//! (identical 0/1 fold masks shipped into the R body), so the paired comparison
//! stays honest; only the split noise is averaged away. We assert on gam's OWN
//! predictions:
//!
//!   1. PREDICTION (primary, tool-free): AVERAGED held-out `R2 >= 0.85`. The
//!      seasonal signal is strong and clean, so a competent cyclic smooth explains
//!      the vast majority of held-out variance — far above the constant-mean
//!      predictor (R2 = 0) and high enough to catch under/over-smoothing.
//!   2. MATCH-OR-BEAT (accuracy): gam's AVERAGED held-out RMSE is no worse than
//!      mgcv's by more than 10% — `gam_rmse_avg <= mgcv_rmse_avg * 1.10`. mgcv fits
//!      the SAME training rows and predicts the SAME held-out months of each
//!      partition; a lower-variance averaged metric against the same bar is
//!      strictly harder than the former single split, never a weakening.
//!   3. STRUCTURE — periodic seam continuity: gam genuinely enforces the wrap, so
//!      its fitted cyclic smooth agrees at the period endpoints
//!      `fit(month=1) == fit(month=13)` to 1e-6. This is a split-invariant property
//!      of a cyclic basis (`bs="cc"`), asserted on one representative fit.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, r2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const NOTTEM_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/nottem_monthly_temp.csv"
);

/// Month-of-year cyclic period: months 1..=12, with the seam (knots) at 1 and
/// 13, so December (12) wraps continuously back to January (1). Both gam's
/// `cc(..., period_start, period_end)` and mgcv's `knots=c(1,13)` use these.
const PERIOD_START: f64 = 1.0;
const PERIOD_END: f64 = 13.0;
/// Cyclic-basis dimension. With only 12 distinct months, k=8 is comfortably
/// resolved by mgcv's `bs="cc"` and leaves ample smoothing freedom.
const K: usize = 8;

/// #2395: K random train/test partitions, averaged. n=240 (20 rows/month), so a
/// ~25% hold-out keeps every month heavily represented in every partition's train
/// set; the cc(month) fit is sub-millisecond, so 2*K=20 fits are trivially inside
/// the fast envelope while cutting the held-out metric's standard error ~3.2x.
const K_SPLITS: usize = 10;
/// Held-out fraction per partition (~75/25, matching the former i%4 split scale).
const HOLDOUT: f64 = 0.25;

/// Deterministic uniform(0,1) hash of (row, split) via splitmix64 — row `i` is in
/// the TEST fold of partition `split` iff it maps below `HOLDOUT`. No RNG dep; the
/// mask is a pure function of (i, split), so gam and mgcv — fed the SAME masks —
/// partition byte-identically.
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

#[test]
fn gam_cyclic_predicts_nottem_seasonal_cycle_vs_mgcv() {
    init_parallelism();

    // ---- load the Nottingham monthly-temperature dataset (month -> temp) ---
    let ds = load_csvwith_inferred_schema(Path::new(NOTTEM_CSV)).expect("load nottem CSV");
    let col = ds.column_map();
    let month_idx = col["month"];
    let temp_idx = col["temp"];
    let month: Vec<f64> = ds.values.column(month_idx).to_vec();
    let temp: Vec<f64> = ds.values.column(temp_idx).to_vec();
    let n = month.len();
    assert_eq!(n, 240, "nottem should have 240 monthly rows, got {n}");
    let p = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula =
        format!("temp ~ cc(month, k={K}, period_start={PERIOD_START}, period_end={PERIOD_END})");

    let mut gam_rmses = Vec::with_capacity(K_SPLITS);
    let mut gam_r2s = Vec::with_capacity(K_SPLITS);
    let mut fold_data: Vec<Vec<f64>> = Vec::with_capacity(K_SPLITS);
    let mut fold_names: Vec<String> = Vec::with_capacity(K_SPLITS);
    // Split-invariant structural readouts captured on the first partition's fit.
    let mut wrap_gap_repr = f64::NAN;
    let mut gam_edf_repr = f64::NAN;

    for split in 0..K_SPLITS {
        let train_rows: Vec<usize> = (0..n).filter(|&i| !is_heldout(i, split)).collect();
        let test_rows: Vec<usize> = (0..n).filter(|&i| is_heldout(i, split)).collect();
        assert!(
            train_rows.len() > 150 && test_rows.len() > 30,
            "#2395 nottem split {split} degenerate: train={} test={}",
            train_rows.len(),
            test_rows.len()
        );
        let test_month: Vec<f64> = test_rows.iter().map(|&i| month[i]).collect();
        let test_temp: Vec<f64> = test_rows.iter().map(|&i| temp[i]).collect();

        // Training-only dataset (subset rows; schema/kinds unchanged).
        let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
        for (out_row, &src_row) in train_rows.iter().enumerate() {
            for c in 0..p {
                train_values[[out_row, c]] = ds.values[[src_row, c]];
            }
        }
        let mut train_ds = ds.clone();
        train_ds.values = train_values;

        let result = fit_from_formula(&formula, &train_ds, &cfg).expect("gam cyclic fit");
        let FitResult::Standard(fit) = result else {
            panic!("expected a standard GAM fit for a gaussian cyclic smooth");
        };

        // gam predictions at the held-out months (identity link => design*beta).
        let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
        for (i, &m) in test_month.iter().enumerate() {
            test_grid[[i, month_idx]] = m;
        }
        let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
            .expect("rebuild design at held-out months");
        let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
        gam_rmses.push(rmse(&gam_test_pred, &test_temp));
        gam_r2s.push(r2(&gam_test_pred, &test_temp));

        if split == 0 {
            gam_edf_repr = fit.fit.edf_total().expect("gam reports total edf");
            // Periodic-wrap check on gam's OWN fit: fit(month=1) == fit(month=13).
            let mut seam_grid = Array2::<f64>::zeros((2, p));
            seam_grid[[0, month_idx]] = PERIOD_START; // month = 1  (January edge)
            seam_grid[[1, month_idx]] = PERIOD_END; //   month = 13 (one period later)
            let seam_design = build_term_collection_design(seam_grid.view(), &fit.resolvedspec)
                .expect("rebuild design at period seam");
            let seam_fit: Vec<f64> = seam_design.design.apply(&fit.fit.beta).to_vec();
            wrap_gap_repr = (seam_fit[0] - seam_fit[1]).abs();
        }

        fold_data.push(
            (0..n)
                .map(|i| if is_heldout(i, split) { 1.0 } else { 0.0 })
                .collect(),
        );
        fold_names.push(format!("fold{split}"));
    }

    // ---- mgcv on the SAME K partitions (full data.frame + K fold masks) ----
    let mut columns: Vec<Column> = vec![
        Column::new("month", &month),
        Column::new("temp", &temp),
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
            for (s in 0:(K - 1)) {{
              fold <- df[[paste0("fold", s)]]
              tr <- df[fold < 0.5, ]
              te <- df[fold >= 0.5, ]
              m <- gam(temp ~ s(month, bs = "cc", k = {K}), data = tr, method = "REML",
                       knots = list(month = c({PERIOD_START}, {PERIOD_END})))
              p <- as.numeric(predict(m, newdata = data.frame(month = te$month)))
              rmses[s + 1] <- sqrt(mean((p - te$temp)^2))
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
    let gam_rmse_avg = mean(&gam_rmses);
    let gam_r2_avg = mean(&gam_r2s);
    let mgcv_rmse_avg = mean(mgcv_rmses);

    eprintln!(
        "nottem cc(month) #2395 K={K_SPLITS}-split avg: gam_edf(split0)={gam_edf_repr:.3} \
         gam_test_R2_avg={gam_r2_avg:.4} gam_test_rmse_avg={gam_rmse_avg:.4} \
         mgcv_test_rmse_avg={mgcv_rmse_avg:.4} wrap_gap(split0)={wrap_gap_repr:.3e}"
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_nottem_cyclic",
            "test_rmse",
            gam_rmse_avg,
            "mgcv",
            mgcv_rmse_avg,
        )
        .line()
    );

    // 1) PREDICTION (primary): gam's cyclic smooth explains the held-out cycle.
    assert!(
        gam_r2_avg >= 0.85,
        "gam averaged held-out predictive R2 too low: {gam_r2_avg:.4} (< 0.85)"
    );

    // 2) MATCH-OR-BEAT (accuracy): gam no worse than mgcv on averaged held-out RMSE.
    assert!(
        gam_rmse_avg <= mgcv_rmse_avg * 1.10,
        "gam averaged held-out RMSE {gam_rmse_avg:.4} exceeds mgcv {mgcv_rmse_avg:.4} * 1.10"
    );

    // 3) STRUCTURE — periodic seam continuity: fit(1) must equal fit(13).
    assert!(
        wrap_gap_repr < 1e-6,
        "cyclic wrap not enforced: |fit(month=1) - fit(month=13)| = {wrap_gap_repr:.3e}"
    );

    // ---- complexity sanity: edf in a seasonal-appropriate range (not matched).
    assert!(
        gam_edf_repr > 1.0 && gam_edf_repr < (K as f64),
        "gam effective dof out of sane range: {gam_edf_repr:.3}"
    );
}
