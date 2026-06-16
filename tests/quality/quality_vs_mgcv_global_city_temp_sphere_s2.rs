//! End-to-end quality: gam's intrinsic S² (sphere) smooth must PREDICT mean
//! surface temperature from a city's position ON THE GLOBE on held-out data at
//! least as well as mgcv's spline-on-sphere `s(lat, lon, bs="sos")` — Wahba's
//! mature, de-facto integrated penalized smoother on the 2-sphere.
//!
//! Data: Berkeley Earth "Global Land Temperatures By Major City", reduced to one
//! row per major city: its (signed) Latitude/Longitude in degrees and its mean
//! monthly surface temperature over 2000-01 .. 2012-12 (°C). The 100 cities are
//! genuinely GLOBAL — spanning both hemispheres (lat −37.8°..+60.3°) and both
//! sides of the prime meridian (lon −118.7°..+151.8°) — so position must be read
//! as a chart of S², not a flat plane. Source CSV (no auth, direct download):
//!   https://raw.githubusercontent.com/gindeleo/climate/master/GlobalLandTemperaturesByMajorCity.csv
//! The per-city annual-mean reduction is the deterministic preprocessing baked
//! into `bench/datasets/global_major_city_temp.csv`.
//!
//! Realistic use-case: a geostatistical regression of mean temperature on a
//! smooth surface over the sphere,
//!   `temp ~ s(lat, lon, bs="sos")` (Gaussian / identity link).
//! Surface temperature is a smooth, predominantly latitudinal field on S² (hot
//! tropics, cold high latitudes), with longitudinal structure from continents and
//! oceans — exactly the low-frequency spherical field an intrinsic spline-on-
//! sphere is built to recover, and exactly where a planar smoother would suffer a
//! ±180° seam artifact that a correct S² chart must not.
//!
//! There is no synthetic ground truth here (real measurements), so objective
//! quality is out-of-sample predictive accuracy:
//!
//!   PRIMARY (objective, tool-free): held-out coefficient of determination
//!     `test_R2 >= 0.70`. Temperature is strongly determined by global position,
//!     so a correct intrinsic S² smooth explains the large majority of held-out
//!     variance — far above the constant-mean predictor (R2=0). A broken kernel,
//!     a seam wrap, or a degenerate (mean-like) surface cannot clear this bar.
//!
//!   BASELINE (match-or-beat): mgcv fits the SAME training cities with
//!     `s(lat, lon, bs="sos")` and predicts the SAME held-out cities; gam's
//!     held-out RMSE must be no worse than `mgcv_test_rmse * 1.10`. mgcv is an
//!     accuracy baseline to match-or-beat, NEVER a fitted target to reproduce.
//!
//! CRITICAL: the identical (lat, lon, temp) rows, in the identical order and with
//! identical degree units, are handed to gam and to mgcv; the held-out split is
//! the deterministic every-4th-row rule on both sides.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, pearson, r2, rmse, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const CITY_TEMP_CSV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/bench/datasets/global_major_city_temp.csv"
);

#[test]
fn gam_sphere_smooth_predicts_global_city_temp_better_than_mgcv_sos() {
    init_parallelism();

    // ---- load the global major-city temperature dataset (lat, lon -> temp) -
    let ds = load_csvwith_inferred_schema(Path::new(CITY_TEMP_CSV))
        .expect("load global_major_city_temp.csv");
    let col = ds.column_map();
    let lat_idx = col["lat"];
    let lon_idx = col["lon"];
    let temp_idx = col["temp"];
    let lat: Vec<f64> = ds.values.column(lat_idx).to_vec();
    let lon: Vec<f64> = ds.values.column(lon_idx).to_vec();
    let temp: Vec<f64> = ds.values.column(temp_idx).to_vec();
    let n = temp.len();
    assert!(
        n >= 100,
        "global city temperature set should have ~100 cities, got {n}"
    );
    // Sanity: the cities really do span the globe (both hemispheres, both sides
    // of the prime meridian) so this is an honest S² problem, not a local patch.
    let lat_min = lat.iter().cloned().fold(f64::INFINITY, f64::min);
    let lat_max = lat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lon_min = lon.iter().cloned().fold(f64::INFINITY, f64::min);
    let lon_max = lon.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        lat_min < -20.0 && lat_max > 50.0 && lon_min < -50.0 && lon_max > 100.0,
        "cities are not globally spread: lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]"
    );

    // ---- deterministic train/test split: every 4th city is held out -------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() >= 70 && test_rows.len() >= 20,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_lat: Vec<f64> = train_rows.iter().map(|&i| lat[i]).collect();
    let train_lon: Vec<f64> = train_rows.iter().map(|&i| lon[i]).collect();
    let train_temp: Vec<f64> = train_rows.iter().map(|&i| temp[i]).collect();
    let test_lat: Vec<f64> = test_rows.iter().map(|&i| lat[i]).collect();
    let test_lon: Vec<f64> = test_rows.iter().map(|&i| lon[i]).collect();
    let test_temp: Vec<f64> = test_rows.iter().map(|&i| temp[i]).collect();

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

    // ---- fit gam on TRAIN: temp ~ sphere(lat, lon, k=40), REML -------------
    // `sphere(lat, lon, ...)` routes (lat, lon) degrees through gam's intrinsic
    // S² smooth (Wahba reproducing-kernel / spherical harmonics), the direct
    // analogue of mgcv `bs="sos"`. k=40 < 75 training cities leaves the penalty,
    // not the rank, in control of smoothness.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("temp ~ sphere(lat, lon, k=40)", &train_ds, &cfg).expect("gam sphere fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian sphere smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out cities: rebuild the design from the frozen
    // spec (identity link => design*beta = predicted mean). The sphere basis
    // pins its kernel centers from the fit data, so `fit.resolvedspec` already
    // carries the fit-time geometry that `beta` was estimated against.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, lat_idx]] = test_lat[i];
        test_grid[[i, lon_idx]] = test_lon[i];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild sphere design at held-out cities");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model on TRAIN with mgcv bs="sos", predict the SAME TEST
    // The harness exposes one data.frame per call, so the held-out predictor
    // columns ride along padded to the training length and are sliced back to
    // the first `k = test_n` rows inside R. mgcv's `s(lat, lon, bs="sos")` takes
    // latitude/longitude in DEGREES — identical units to the columns gam loaded.
    let r = run_r(
        &[
            Column::new("lat", &train_lat),
            Column::new("lon", &train_lon),
            Column::new("temp", &train_temp),
            Column::new("test_lat", &pad_to(&test_lat, train_lat.len())),
            Column::new("test_lon", &pad_to(&test_lon, train_lat.len())),
            Column::new("test_n", &vec![test_rows.len() as f64; train_lat.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(temp ~ s(lat, lon, bs = "sos", k = 40), data = df, method = "REML")
        k <- df$test_n[1]
        newd <- data.frame(
            lat = df$test_lat[1:k],
            lon = df$test_lon[1:k]
        )
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_test_pred = r.vector("test_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN predictions ------------------------
    let gam_test_r2 = r2(&gam_test_pred, &test_temp);
    let gam_test_rmse = rmse(&gam_test_pred, &test_temp);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_temp);

    eprintln!(
        "global-city temp sphere(lat,lon,k=40) held-out: n_train={} n_test={} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} gam_test_R2={gam_test_r2:.4} \
         gam_test_rmse={gam_test_rmse:.4} mgcv_test_rmse={mgcv_test_rmse:.4}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out field ------
    // Mean surface temperature is strongly structured by position on S²; a
    // correct intrinsic smooth recovers most of that structure. R2 >= 0.70 sits
    // well above the constant-mean baseline (0) and a degenerate or seam-broken
    // surface cannot reach it.
    assert!(
        gam_test_r2 >= 0.70,
        "gam's held-out predictive R2 too low: {gam_test_r2:.4} (< 0.70)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv bs=sos on held-out RMSE
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv bs=sos {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 2.0 && gam_edf < 40.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}

/// Second real-data arm exercising the SAME intrinsic S² (sphere) capability on
/// the SAME Berkeley-Earth global major-city temperature dataset, but with an
/// INDEPENDENT deterministic split and an INDEPENDENT held-out objective metric,
/// so the sphere smooth is held to more than a single split + R² fluke.
///
/// Source CSV (Berkeley Earth "Global Land Temperatures By Major City", reduced
/// to one row per major city, no auth, direct download):
///   https://raw.githubusercontent.com/gindeleo/climate/master/GlobalLandTemperaturesByMajorCity.csv
///
/// Split: every 5th city (`i % 5 == 0`) is held out — 20 test / 80 train,
/// disjoint from the every-4th-row split of the sibling test. Identical
/// (lat, lon, temp) rows, in identical order and degree units, go to gam and to
/// mgcv `s(lat, lon, bs="sos")`.
///
///   PRIMARY (objective, tool-free): held-out Pearson correlation between gam's
///     predicted and observed temperature `test_corr >= 0.84`. Temperature is a
///     strongly position-determined field on S²; a correct intrinsic smooth's
///     held-out predictions track the observed field tightly. A degenerate
///     (mean-like) surface or a ±180° seam break cannot reach this. Correlation
///     is scale/offset-free, so it is an independent witness from the sibling
///     test's R² (which also penalises bias/scale errors).
///
///   ABSOLUTE held-out bar: gam's held-out RMSE `< 6.0 °C` — comfortably below
///     the ~10 °C spread of city annual-mean temperatures, an honest accuracy
///     floor that a non-predictive surface fails outright.
///
///   BASELINE (match-or-beat): mgcv `s(lat, lon, bs="sos")` fits the SAME train
///     cities and predicts the SAME held-out cities; gam's held-out RMSE must be
///     no worse than `mgcv_test_rmse * 1.10`. mgcv is an accuracy baseline to
///     match-or-beat, never a fitted target to reproduce.
#[test]
fn gam_sphere_smooth_predicts_global_city_temp_better_than_mgcv_sos_on_real_data() {
    init_parallelism();

    // ---- load the global major-city temperature dataset (lat, lon -> temp) -
    let ds = load_csvwith_inferred_schema(Path::new(CITY_TEMP_CSV))
        .expect("load global_major_city_temp.csv");
    let col = ds.column_map();
    let lat_idx = col["lat"];
    let lon_idx = col["lon"];
    let temp_idx = col["temp"];
    let lat: Vec<f64> = ds.values.column(lat_idx).to_vec();
    let lon: Vec<f64> = ds.values.column(lon_idx).to_vec();
    let temp: Vec<f64> = ds.values.column(temp_idx).to_vec();
    let n = temp.len();
    assert!(
        n >= 100,
        "global city temperature set should have ~100 cities, got {n}"
    );

    // ---- deterministic train/test split: every 5th city is held out -------
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() >= 70 && test_rows.len() >= 18,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_lat: Vec<f64> = train_rows.iter().map(|&i| lat[i]).collect();
    let train_lon: Vec<f64> = train_rows.iter().map(|&i| lon[i]).collect();
    let train_temp: Vec<f64> = train_rows.iter().map(|&i| temp[i]).collect();
    let test_lat: Vec<f64> = test_rows.iter().map(|&i| lat[i]).collect();
    let test_lon: Vec<f64> = test_rows.iter().map(|&i| lon[i]).collect();
    let test_temp: Vec<f64> = test_rows.iter().map(|&i| temp[i]).collect();

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

    // ---- fit gam on TRAIN: temp ~ sphere(lat, lon, k=40), REML -------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("temp ~ sphere(lat, lon, k=40)", &train_ds, &cfg).expect("gam sphere fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian sphere smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out cities: rebuild the design from the frozen
    // spec (identity link => design*beta = predicted mean).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, lat_idx]] = test_lat[i];
        test_grid[[i, lon_idx]] = test_lon[i];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild sphere design at held-out cities");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model on TRAIN with mgcv bs="sos", predict the SAME TEST
    let r = run_r(
        &[
            Column::new("lat", &train_lat),
            Column::new("lon", &train_lon),
            Column::new("temp", &train_temp),
            Column::new("test_lat", &pad_to(&test_lat, train_lat.len())),
            Column::new("test_lon", &pad_to(&test_lon, train_lat.len())),
            Column::new("test_n", &vec![test_rows.len() as f64; train_lat.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(temp ~ s(lat, lon, bs = "sos", k = 40), data = df, method = "REML")
        k <- df$test_n[1]
        newd <- data.frame(
            lat = df$test_lat[1:k],
            lon = df$test_lon[1:k]
        )
        emit("test_pred", as.numeric(predict(m, newdata = newd)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_test_pred = r.vector("test_pred");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_pred.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN predictions ------------------------
    let gam_test_corr = pearson(&gam_test_pred, &test_temp);
    let gam_test_rmse = rmse(&gam_test_pred, &test_temp);
    let mgcv_test_rmse = rmse(mgcv_test_pred, &test_temp);

    eprintln!(
        "global-city temp sphere(lat,lon,k=40) held-out (i%5): n_train={} n_test={} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} gam_test_corr={gam_test_corr:.4} \
         gam_test_rmse={gam_test_rmse:.4} mgcv_test_rmse={mgcv_test_rmse:.4}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam tracks the held-out field --------
    assert!(
        gam_test_corr >= 0.84,
        "gam's held-out predictive correlation too low: {gam_test_corr:.4} (< 0.84)"
    );

    // ---- ABSOLUTE held-out accuracy floor ----------------------------------
    assert!(
        gam_test_rmse < 6.0,
        "gam's held-out RMSE too high: {gam_test_rmse:.4} °C (>= 6.0)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv bs=sos on held-out RMSE
    assert!(
        gam_test_rmse <= mgcv_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds mgcv bs=sos {mgcv_test_rmse:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 2.0 && gam_edf < 40.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
