//! End-to-end OBJECTIVE quality: gam's *mixed-boundary* cylinder tensor smooth
//! on REAL solar-geometry data must PREDICT held-out solar zenith angles well —
//! beating a constant-mean baseline by a wide margin and matching-or-beating
//! mgcv — and must satisfy the intrinsic azimuthal-seam contract of the cyclic
//! margin.
//!
//! DATA (real, freely downloadable): the `sza` dataset from the R `gt` package
//! (twice-hourly solar zenith angles by month & latitude),
//! https://vincentarelbundock.github.io/Rdatasets/csv/gt/sza.csv — saved
//! verbatim at `bench/datasets/solar_zenith_angle.csv`. The solar zenith angle
//! (degrees from vertical) is a smooth, deterministic function of the date and
//! the time of day at a fixed latitude. We fix latitude = 20° and model
//!
//!     sza  ~  te( month_angle , tst_hours )
//!
//! as a CYLINDER S¹ × ℝ:
//!   * **month_angle** is the PERIODIC (azimuthal) margin. Calendar month
//!     jan…dec is mapped to θ = 2π·(month_index)/12, which genuinely WRAPS:
//!     the solar geometry one full year apart is identical, so the surface at
//!     θ = 0 (start of January) must equal the surface at θ = 2π. The data
//!     bear this out — at solar noon the December zenith (≈41.8°) is essentially
//!     the January zenith (≈43°): the seam closes physically.
//!   * **tst_hours** (true solar time, 04:00–12:00 as decimal hours) is the
//!     LINEAR / clamped margin. Morning solar time does NOT wrap here, and the
//!     surface is free at the two ends — the asymmetry that makes this a
//!     cylinder, not a torus.
//!
//! gam builds this with
//! `te(month_angle, tst_hours, boundary=['periodic','clamped'],
//!     period=[2*pi, None])`: a cyclic θ margin tensor-producted with a clamped,
//! non-periodic time margin. mgcv builds the identical construction with
//! `te(month_angle, tst_hours, bs = c("cc", "ps"))` and serves PURELY as a
//! match-or-beat BASELINE on held-out accuracy — never as the definition of
//! "correct".
//!
//! OBJECTIVE METRICS ASSERTED:
//!   1. HELD-OUT PREDICTION (primary, tool-free): a deterministic train/test
//!      split (every 4th row held out) is fit on the training rows only; gam's
//!      predictions on the held-out rows achieve test R² ≥ 0.95. The solar
//!      signal is essentially noiseless and strongly structured, so a competent
//!      cylinder smooth explains nearly all held-out variance, far above the
//!      constant-mean predictor (R² = 0).
//!   2. BASELINE (match-or-beat): gam's held-out RMSE ≤ mgcv's × 1.10 on the
//!      SAME split and the SAME rows.
//!   3. AZIMUTHAL SEAM CONTINUITY (θ contract, exact math): gam's fitted surface
//!      is identical at θ=0 and θ=2π for every solar time, to float error — the
//!      load-bearing property of the cyclic month margin.
//!
//! CRITICAL plumbing: the IDENTICAL (month_angle, tst_hours, sza) rows, in the
//! IDENTICAL order, with the IDENTICAL 2π period, are handed to both gam and
//! mgcv.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, r2, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::TAU;
use std::path::Path;

const SZA_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/solar_zenith_angle.csv"
);

/// Map a calendar-month abbreviation to its 0-based index (jan=0 … dec=11).
fn month_index(m: &str) -> usize {
    const MONTHS: [&str; 12] = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    MONTHS
        .iter()
        .position(|&x| x == m)
        .unwrap_or_else(|| panic!("unrecognized month {m:?} in solar zenith dataset"))
}

/// Parse a `tst` field ("HHMM", e.g. "0830") into decimal hours (8.5).
fn tst_to_hours(tst: &str) -> f64 {
    assert_eq!(tst.len(), 4, "tst {tst:?} is not HHMM");
    let hh: f64 = tst[0..2].parse().expect("tst hour");
    let mm: f64 = tst[2..4].parse().expect("tst minute");
    hh + mm / 60.0
}

#[test]
fn gam_solar_zenith_cylinder_predicts_heldout_and_closes_seam() {
    init_parallelism();

    // ---- load the real solar-zenith dataset, fix latitude = 20° -----------
    // We keep a single latitude so the problem is a clean 2-D cylinder
    // (month × time), and drop the night-time rows where the zenith angle is
    // undefined (NA). The remaining rows are kept in FILE ORDER so the identical
    // sequence reaches gam and mgcv. month → θ = 2π·month_index/12 makes the
    // calendar genuinely periodic (Dec end ≡ Jan start); tst → decimal hours is
    // the linear morning-time margin.
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(Path::new(SZA_CSV))
        .expect("open solar_zenith_angle.csv");

    let mut month_angle: Vec<f64> = Vec::new();
    let mut tst_hours: Vec<f64> = Vec::new();
    let mut sza: Vec<f64> = Vec::new();
    for rec in reader.records() {
        let rec = rec.expect("read sza row");
        // columns: rownames, latitude, month, tst, sza
        let latitude = rec.get(1).expect("latitude col").trim();
        let month = rec.get(2).expect("month col").trim();
        let tst = rec.get(3).expect("tst col").trim();
        let sza_raw = rec.get(4).expect("sza col").trim();
        if latitude != "20" {
            continue;
        }
        if sza_raw.is_empty() {
            // night-time / sun below horizon: zenith undefined in this dataset.
            continue;
        }
        month_angle.push(TAU * (month_index(month) as f64) / 12.0);
        tst_hours.push(tst_to_hours(tst));
        sza.push(sza_raw.parse().expect("parse sza degrees"));
    }
    let n = sza.len();
    assert!(
        n > 120,
        "solar zenith (lat=20, non-NA) should have ~150 rows, got {n}"
    );

    // ---- deterministic train/test split: every 4th row held out ----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 80 && test_rows.len() > 25,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_angle: Vec<f64> = train_rows.iter().map(|&i| month_angle[i]).collect();
    let train_time: Vec<f64> = train_rows.iter().map(|&i| tst_hours[i]).collect();
    let train_sza: Vec<f64> = train_rows.iter().map(|&i| sza[i]).collect();
    let test_angle: Vec<f64> = test_rows.iter().map(|&i| month_angle[i]).collect();
    let test_time: Vec<f64> = test_rows.iter().map(|&i| tst_hours[i]).collect();
    let test_sza: Vec<f64> = test_rows.iter().map(|&i| sza[i]).collect();

    // ---- build gam's training dataset from the numeric columns ------------
    let headers = ["month_angle", "tst_hours", "sza"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..train_rows.len())
        .map(|r| {
            StringRecord::from(vec![
                train_angle[r].to_string(),
                train_time[r].to_string(),
                train_sza[r].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode solar zenith train");
    let col = ds.column_map();
    let angle_idx = col["month_angle"];
    let time_idx = col["tst_hours"];

    // ---- fit gam on TRAIN: mixed-boundary cylinder tensor smooth, REML ----
    // `boundary=['periodic','clamped']` + `period=[2*pi, None]` is gam's exact
    // analog of mgcv's te(bs=c('cc','ps')): a cyclic month margin (period 2π)
    // tensor-producted with a clamped, non-periodic solar-time margin.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let formula = "sza ~ te(month_angle, tst_hours, boundary=['periodic','clamped'], period=[2*pi, None], k=8)";
    let result = fit_from_formula(formula, &ds, &cfg).expect("gam cylinder fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the solar-zenith cylinder smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Helper: evaluate gam's fitted surface at arbitrary (θ, time) rows by
    // rebuilding the design from the frozen spec (identity link => design·β=mean).
    let gam_predict = |angles: &[f64], times: &[f64]| -> Vec<f64> {
        assert_eq!(angles.len(), times.len());
        let m = angles.len();
        let mut pts = Array2::<f64>::zeros((m, ds.headers.len()));
        for r in 0..m {
            pts[[r, angle_idx]] = angles[r];
            pts[[r, time_idx]] = times[r];
        }
        let d = build_term_collection_design(pts.view(), &fit.resolvedspec)
            .unwrap_or_else(|e| panic!("rebuild solar-zenith cylinder design: {e:?}"));
        d.design.apply(&fit.fit.beta).to_vec()
    };

    // gam predictions at the held-out rows.
    let gam_test_pred = gam_predict(&test_angle, &test_time);
    let gam_test_r2 = r2(&gam_test_pred, &test_sza);
    let gam_test_rmse = rmse(&gam_test_pred, &test_sza);

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST -----
    // The identical training rows AND the held-out rows ride along as columns
    // (test rows padded to the train length; only the first `test_n` are read
    // back), so mgcv sees bit-identical θ values and the same 2π period.
    let pad = |v: &[f64]| -> Vec<f64> {
        let fill = v.last().copied().unwrap_or(0.0);
        let mut out = v.to_vec();
        out.resize(train_rows.len(), fill);
        out
    };
    let r = run_r(
        &[
            Column::new("month_angle", &train_angle),
            Column::new("tst_hours", &train_time),
            Column::new("sza", &train_sza),
            Column::new("test_angle", &pad(&test_angle)),
            Column::new("test_time", &pad(&test_time)),
            Column::new("test_n", &vec![test_rows.len() as f64; train_rows.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(sza ~ te(month_angle, tst_hours, bs = c("cc", "ps"), k = c(8, 8)),
                 data = df, method = "REML",
                 knots = list(month_angle = c(0, 2 * pi)))
        emit("edf", sum(m$edf))
        k <- df$test_n[1]
        newd <- data.frame(month_angle = df$test_angle[1:k],
                           tst_hours   = df$test_time[1:k])
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        "#,
    );
    let mgcv_test_pred = r.vector("test_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_sza);

    // ---- (3) intrinsic θ-seam continuity (the cyclic-margin contract) ------
    // Compare gam's surface at θ=0 vs θ=2π across a dense set of solar times. A
    // genuine cyclic month margin has identical design rows — hence identical
    // fitted values — at coordinates separated by exactly one period in θ.
    let time_grid: Vec<f64> = (0..41).map(|kk| 4.0 + 8.0 * (kk as f64) / 40.0).collect();
    let theta_zeros: Vec<f64> = std::iter::repeat_n(0.0, time_grid.len()).collect();
    let theta_taus: Vec<f64> = std::iter::repeat_n(TAU, time_grid.len()).collect();
    let seam_0 = gam_predict(&theta_zeros, &time_grid);
    let seam_tau = gam_predict(&theta_taus, &time_grid);
    let theta_seam_gap = seam_0
        .iter()
        .zip(seam_tau.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // Context-only diagnostic (printed, never a pass criterion): how closely
    // gam tracks mgcv on the held-out rows. Matching another tool proves nothing
    // about correctness, so it is reported but not asserted.
    let rel_to_mgcv = relative_l2(&gam_test_pred, mgcv_test_pred);

    eprintln!(
        "solar-zenith te(cc,ps) cylinder: n={n} n_train={} n_test={} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         mgcv_test_rmse={mgcv_test_rmse:.4} theta_seam_gap={theta_seam_gap:.3e} \
         rel_to_mgcv(context)={rel_to_mgcv:.5}",
        train_rows.len(),
        test_rows.len(),
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "manifolds",
            "quality_vs_mgcv_solar_zenith_cylinder",
            "test_rmse",
            gam_test_rmse,
            "mgcv",
            mgcv_test_rmse,
        )
        .line()
    );

    // ---- (1) PRIMARY: held-out prediction ---------------------------------
    // Solar zenith is an essentially noiseless, strongly structured function of
    // month and solar time. A correct cylinder REML fit explains nearly all the
    // held-out variance; R²≥0.95 is far above the constant-mean baseline (0) and
    // a real basis/penalty bug (e.g. a broken periodic margin) drops well below.
    assert!(
        gam_test_r2 >= 0.95,
        "gam held-out predictive R² too low: {gam_test_r2:.4} (< 0.95)"
    );

    // ---- (2) BASELINE (match-or-beat): no worse than mgcv on held-out RMSE -
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- (3) θ-seam continuity (exact cyclic-margin math) -----------------
    // Value continuity across the calendar wrap, exact up to float error. Any
    // gap > 1e-6 is a sign/threshold bug in gam's periodic closure — a
    // mathematical contract of the cyclic basis, asserted directly.
    assert!(
        theta_seam_gap < 1e-6,
        "month θ-seam not closed: max |f(0,t) - f(2π,t)| = {theta_seam_gap:.3e}"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 2.0 && gam_edf < 50.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
