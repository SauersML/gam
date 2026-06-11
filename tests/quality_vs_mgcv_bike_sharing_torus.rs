//! End-to-end quality: gam's *doubly* periodic tensor smooth on the torus
//! S¹ × S¹ must PREDICT a real bivariate-circular surface on held-out cells —
//! at least as well as **mgcv**, the mature standard — AND enforce the periodic
//! wrap on BOTH margins.
//!
//! DATASET: UCI "Bike Sharing" hourly rental counts, Capital Bikeshare
//! (Washington D.C.), 2011–2012, 17 379 hourly records.
//!   Source (UCI ML Repository): https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
//!   Raw `hour.csv` mirror used to build this file:
//!     https://raw.githubusercontent.com/rachelleperez/Bike-Sharing/master/hour.csv
//! The raw rows carry an hour-of-day `hr` (0–23) and a calendar date `dteday`.
//! Both coordinates are genuinely PERIODIC: the hour wraps 23→0 (midnight), and
//! the day-of-year wraps Dec→Jan. We materialised a clean toroidal CSV
//! (`bench/datasets/bike_sharing_torus.csv`) by:
//!   * mapping each record to a continuous within-year angle
//!     `season = ((day_of_year − 1) / days_in_year) · 365`  (leap-year aware,
//!     so 2011 and 2012 share one [0, 365) circle), and the hour `hour ∈ [0,24)`;
//!   * binning `season` into 73 equal arcs (≈5-day) and `hour` into its 24
//!     integer slots, then taking the **cell mean of `log1p(cnt)`** (7–11 raw
//!     records per cell). Averaging denoises the count process into a smooth,
//!     fully-populated 73×24 toroidal surface; `log1p` is the natural variance-
//!     stabilising scale for over-dispersed counts.
//! The cell centres sit at `season = (b + 0.5)/73 · 365` and `hour = h`, so with
//! domain `[0, 365)` × `[0, 24)` the Dec→Jan and 23→0 seams each fall exactly one
//! cell-spacing across the wrap — the physically correct toroidal geometry.
//!
//! This is a HELD-OUT PREDICTION test on REAL data — there is no analytic ground
//! truth, so objective quality is out-of-sample accuracy. A deterministic split
//! holds out every 5th cell (sorted season-major). We fit the doubly-periodic
//! tensor smooth on the training cells by REML, predict the held-out cells, and
//! assert on gam's OWN predictions:
//!
//!   1. PREDICTION (primary, tool-free): held-out `R2 >= 0.92`. The rental
//!      surface has a strong, smooth structure (a sharp twin-peaked commute
//!      cycle over the day, modulated by a seasonal envelope), so a competent
//!      doubly-periodic tensor smooth explains the vast majority of held-out
//!      variance — far above the constant-mean predictor (R2 = 0) and high enough
//!      to catch under/over-smoothing on either margin.
//!   2. MATCH-OR-BEAT (accuracy): gam's held-out RMSE is no worse than mgcv's by
//!      more than 10% — `gam_test_rmse <= mgcv_test_rmse * 1.10`. mgcv
//!      (`te(season, hour, bs=c("cc","cc"))`) fits the SAME training cells and
//!      predicts the SAME held-out cells; it is the mature toroidal baseline to
//!      match-or-beat on ACCURACY, never a fitted target to reproduce.
//!   3. STRUCTURE — toroidal seam continuity on BOTH margins (the load-bearing
//!      property of a cyclic basis): gam's fitted surface agrees at season=0 vs
//!      season=365 for every hour (the seasonal seam) and symmetrically at
//!      hour=0 vs hour=24 for every season (the diurnal seam), exact to 1e-6. A
//!      sign/threshold bug in either periodic-basis closure surfaces here as a
//!      wrap discontinuity invisible to an interior RMSE check.
//!
//! The identical (season, hour, log_count) cells, in the same order, with the
//! identical periods (365, 24) and origins (0, 0), are handed to both gam and
//! mgcv. "rel_l2 to mgcv's fitted surface" is computed for context only and is
//! deliberately NOT a pass criterion: matching another tool's fit proves nothing
//! about correctness — both could overfit alike.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, r2, relative_l2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const BIKE_TORUS_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/bike_sharing_torus.csv"
);

/// Seasonal period: a full year mapped to a [0, 365) circle, Dec wrapping to Jan.
const SEASON_PERIOD: f64 = 365.0;
/// Diurnal period: 24 hours, hour 23 wrapping to hour 0.
const HOUR_PERIOD: f64 = 24.0;
/// Per-margin cyclic-basis dimension: more freedom on the (richer) diurnal axis
/// is unnecessary — the sharp commute structure lives in `hour`, the slow
/// envelope in `season` — so give the seasonal margin a generous `k` and the
/// diurnal margin enough to resolve the twin commute peaks without saturating.
const K_SEASON: usize = 12;
const K_HOUR: usize = 10;

#[test]
fn gam_torus_predicts_bike_sharing_diurnal_seasonal_cycle_vs_mgcv() {
    init_parallelism();

    // ---- load the cleaned toroidal bike-sharing surface --------------------
    let ds = load_csvwith_inferred_schema(Path::new(BIKE_TORUS_CSV)).expect("load bike torus CSV");
    let col = ds.column_map();
    let season_idx = col["season"];
    let hour_idx = col["hour"];
    let log_idx = col["log_count"];
    let season: Vec<f64> = ds.values.column(season_idx).to_vec();
    let hour: Vec<f64> = ds.values.column(hour_idx).to_vec();
    let log_count: Vec<f64> = ds.values.column(log_idx).to_vec();
    let n = season.len();
    assert_eq!(
        n,
        73 * 24,
        "bike torus should be a full 73x24 grid, got {n}"
    );

    // ---- deterministic train/test split: every 5th cell is held out --------
    // Rows are season-major (then hour) in the CSV, so striding by 5 scatters the
    // held-out cells across the whole torus rather than along one ring.
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 1300 && test_rows.len() > 300,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let test_season: Vec<f64> = test_rows.iter().map(|&i| season[i]).collect();
    let test_hour: Vec<f64> = test_rows.iter().map(|&i| hour[i]).collect();
    let test_log: Vec<f64> = test_rows.iter().map(|&i| log_count[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: doubly-periodic tensor smooth, REML -------------
    // `boundary=['periodic','periodic']` + `period=[365, 24]` + `origin=[0, 0]`
    // is gam's exact analog of mgcv's te(bs=c('cc','cc')) with the matching
    // cyclic knot ranges: two cyclic B-spline marginals tensor-producted on the
    // torus. The explicit origins pin each domain to [0, period) so the seam sits
    // at the physical Dec→Jan / midnight wrap (not at the data minimum 2.5/0).
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = format!(
        "log_count ~ te(season, hour, boundary=['periodic','periodic'], \
         period=[{SEASON_PERIOD}, {HOUR_PERIOD}], origin=[0, 0], k=[{K_SEASON}, {K_HOUR}])"
    );
    let result = fit_from_formula(&formula, &train_ds, &cfg).expect("gam torus tensor fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the bike-sharing torus tensor smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Helper: evaluate gam's fitted surface at arbitrary (season, hour) rows by
    // rebuilding the design from the frozen spec (identity link => design·β = mean).
    let gam_predict = |seasons: &[f64], hours: &[f64]| -> Vec<f64> {
        assert_eq!(seasons.len(), hours.len());
        let m = seasons.len();
        let mut pts = Array2::<f64>::zeros((m, p));
        for r in 0..m {
            pts[[r, season_idx]] = seasons[r];
            pts[[r, hour_idx]] = hours[r];
        }
        let d = build_term_collection_design(pts.view(), &fit.resolvedspec)
            .expect("rebuild bike torus design");
        d.design.apply(&fit.fit.beta).to_vec()
    };

    // gam predictions on the held-out cells.
    let gam_test_pred = gam_predict(&test_season, &test_hour);

    // ---- fit the SAME model on TRAIN with mgcv te(bs=c("cc","cc")) ---------
    // mgcv needs explicit cyclic knot ranges per margin so its cyclic closure
    // matches the [0, period) data support. The held-out cells ride along with a
    // zero weight so they are predicted but never fitted; both engines therefore
    // train on the identical cells and predict the identical cells.
    let mut season_all: Vec<f64> = train_rows.iter().map(|&i| season[i]).collect();
    season_all.extend_from_slice(&test_season);
    let mut hour_all: Vec<f64> = train_rows.iter().map(|&i| hour[i]).collect();
    hour_all.extend_from_slice(&test_hour);
    let mut log_all: Vec<f64> = train_rows.iter().map(|&i| log_count[i]).collect();
    log_all.extend(std::iter::repeat_n(0.0, test_rows.len())); // placeholders, weight 0
    let mut wts: Vec<f64> = std::iter::repeat_n(1.0, train_rows.len()).collect();
    wts.extend(std::iter::repeat_n(0.0, test_rows.len()));

    let r = run_r(
        &[
            Column::new("season", &season_all),
            Column::new("hour", &hour_all),
            Column::new("log_count", &log_all),
            Column::new("w", &wts),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            train <- df[df$w > 0, ]
            m <- gam(log_count ~ te(season, hour, bs = c("cc", "cc"), k = c({K_SEASON}, {K_HOUR})),
                     data = train, method = "REML",
                     knots = list(season = c(0, {SEASON_PERIOD}), hour = c(0, {HOUR_PERIOD})))
            test <- df[df$w == 0, ]
            emit("test_pred", as.numeric(predict(m, newdata = test)))
            emit("edf", sum(m$edf))
            "#
        ),
    );
    let mgcv_test_pred = r.vector("test_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective held-out accuracy ---------------------------------------
    let gam_r2 = r2(&gam_test_pred, &test_log);
    let gam_test_rmse = rmse(&gam_test_pred, &test_log);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_log);
    // For context only (NOT a pass gate): how close the two fitted surfaces are.
    let rel_gam_vs_mgcv = relative_l2(&gam_test_pred, mgcv_test_pred);

    // ---- toroidal seam continuity on BOTH margins (load-bearing property) --
    // Seasonal seam: f(season=0, hour) vs f(season=365, hour) across a dense set
    // of hours. Diurnal seam: f(season, hour=0) vs f(season, hour=24) across a
    // dense set of seasons. A genuine doubly-periodic basis has identical design
    // rows — hence identical fitted values — one full period apart in either
    // margin.
    let hour_sweep: Vec<f64> = (0..48).map(|k| HOUR_PERIOD * (k as f64) / 48.0).collect();
    let season_sweep: Vec<f64> = (0..48).map(|k| SEASON_PERIOD * (k as f64) / 48.0).collect();
    let zeros_h: Vec<f64> = std::iter::repeat_n(0.0, hour_sweep.len()).collect();
    let period_h: Vec<f64> = std::iter::repeat_n(SEASON_PERIOD, hour_sweep.len()).collect();
    let zeros_s: Vec<f64> = std::iter::repeat_n(0.0, season_sweep.len()).collect();
    let period_s: Vec<f64> = std::iter::repeat_n(HOUR_PERIOD, season_sweep.len()).collect();

    let season_seam_0 = gam_predict(&zeros_h, &hour_sweep);
    let season_seam_p = gam_predict(&period_h, &hour_sweep);
    let season_seam_gap = season_seam_0
        .iter()
        .zip(season_seam_p.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let hour_seam_0 = gam_predict(&season_sweep, &zeros_s);
    let hour_seam_p = gam_predict(&season_sweep, &period_s);
    let hour_seam_gap = hour_seam_0
        .iter()
        .zip(hour_seam_p.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    eprintln!(
        "bike torus te(cc,cc): n_train={} n_test={} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_test_R2={gam_r2:.5} gam_test_rmse={gam_test_rmse:.5} mgcv_test_rmse={mgcv_test_rmse:.5} \
         rel_l2_gam_vs_mgcv={rel_gam_vs_mgcv:.5} \
         season_seam_gap={season_seam_gap:.3e} hour_seam_gap={hour_seam_gap:.3e}",
        train_rows.len(),
        test_rows.len()
    );

    // (1) PREDICTION — the primary objective claim. The diurnal×seasonal rental
    // surface is strong and smooth; a correct doubly-periodic tensor smooth
    // explains the overwhelming majority of held-out variance. R2 >= 0.92 is the
    // principled bar: far above the constant-mean predictor and high enough to
    // catch under/over-smoothing on either periodic margin.
    assert!(
        gam_r2 >= 0.92,
        "gam under-explained the held-out bike-sharing torus: test R2={gam_r2:.5} < 0.92"
    );
    // (2) MATCH-OR-BEAT mgcv ON ACCURACY (not on fitted output). gam's held-out
    // error must be no worse than the mature toroidal baseline's by more than
    // 10%. This demotes mgcv to an accuracy baseline, never a target.
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam's held-out accuracy lags the mgcv baseline: gam_rmse={gam_test_rmse:.5} \
         mgcv_rmse={mgcv_test_rmse:.5} (allowed gam <= mgcv*1.10)"
    );
    // EDF sanity (NOT matched to mgcv): the recovered surface must be genuinely
    // wiggly (sharp commute peaks + seasonal envelope) yet far below the
    // k_season*k_hour saturation cap.
    assert!(
        gam_edf > 8.0 && (gam_edf as usize) < K_SEASON * K_HOUR,
        "gam edf outside a signal-appropriate range for the bike torus: {gam_edf:.3}"
    );
    // (3) The defining contract of a doubly-periodic basis: value continuity
    // across BOTH wraps, exact up to float error. Each torus seam must close to
    // < 1e-6; any larger gap is a sign/threshold bug in a periodic-basis closure.
    assert!(
        season_seam_gap < 1e-6,
        "seasonal seam not closed: max |f(0,hour) - f(365,hour)| = {season_seam_gap:.3e}"
    );
    assert!(
        hour_seam_gap < 1e-6,
        "diurnal seam not closed: max |f(season,0) - f(season,24)| = {hour_seam_gap:.3e}"
    );
}
