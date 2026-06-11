//! Bug: an intrinsic `sphere(lat, lon)` smooth is *not single-valued at the
//! poles* when predicted from a saved model. The north pole `lat = +π/2` is
//! one physical point `(0,0,1)`; every longitude addresses the same point, so
//! a correct predictor returns the same value across longitude there. The SOS
//! basis itself is single-valued at the pole — the defect was in the
//! saved-model predict-time input handling, which clamped the pole latitude
//! back into the *training* range.
//!
//! `FittedModel::axis_clip_to_training_ranges` (`src/inference/model.rs`) clips
//! each continuous new-data column to the `(min, max)` observed in training.
//! It exempts sphere *longitude* (periodic) and parametric/linear axes, but it
//! used to clamp sphere *latitude* to the sampled range. A finite cover of S²
//! never reaches the pole exactly (extreme training latitude here ≈ 87°), so a
//! `lat = π/2` request was clamped to ≈ 87°, where the Wahba SOS kernel's
//! `cos(lat)·cos(lat_c)·cos(Δlon)` term has not damped — and the predictor
//! swept a spurious `cos(lon)` profile at what is physically a single point.
//!
//! The fix teaches the clip that sphere latitude is a closed-manifold
//! coordinate: its clip bounds are the manifold's intrinsic domain
//! (`[-π/2, π/2]` radians or `[-90, 90]` degrees), not the sampled range. Then
//! `lat = ±π/2` reaches the true pole and the SOS basis delivers its intended
//! single-valued pole.
//!
//! These tests drive the real `gam` CLI fit→save→predict pipeline (the single
//! predict entry point reached by both the CLI and the Python `gamfit` path)
//! and assert the pole longitude spread is a tiny fraction of the equatorial
//! signal spread. They cover three angles:
//!   1. north pole, radians mode (the original repro),
//!   2. south pole, radians mode (the `-π/2` bound),
//!   3. north pole, degrees mode (the `90.0` bound — guards against a
//!      regression that hardcodes `π/2` regardless of the `radians` flag).

use gam::test_support::cli_harness::{read_prediction_means, run_or_panic};
use std::f64::consts::PI;
use std::path::Path;
use std::process::Command;

/// Deterministic Fibonacci-lattice cover of S². Returns (lat, lon) in radians,
/// lon wrapped to `[-π, π]`. The lattice never lands exactly on a pole, which
/// is precisely the condition that exposed the clamp.
fn fibonacci_lattice(n: usize) -> Vec<(f64, f64)> {
    let golden = PI * (3.0 - 5.0_f64.sqrt());
    let two_pi = 2.0 * PI;
    (0..n)
        .map(|i| {
            let sin_lat = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let lat = sin_lat.clamp(-1.0, 1.0).asin();
            let mut lon = (golden * i as f64) % two_pi;
            if lon > PI {
                lon -= two_pi;
            } else if lon < -PI {
                lon += two_pi;
            }
            (lat, lon)
        })
        .collect()
}

/// Write a training CSV with a pure-longitude signal `y = cos(lon)`. Columns
/// are emitted in the requested unit (radians or degrees) so the formula's
/// `radians=` flag matches the data.
fn write_training_csv(path: &Path, n: usize, radians: bool) {
    let mut writer = csv::Writer::from_path(path).expect("create training csv");
    writer
        .write_record(["lat", "lon", "y"])
        .expect("write header");
    let to_unit = if radians { 1.0 } else { 180.0 / PI };
    for (lat, lon) in fibonacci_lattice(n) {
        let y = lon.cos();
        writer
            .write_record([
                format!("{:.12}", lat * to_unit),
                format!("{:.12}", lon * to_unit),
                format!("{y:.12}"),
            ])
            .expect("write training row");
    }
    writer.flush().expect("flush training csv");
}

/// Write a predict CSV: one ring of `n_lon` longitudes at a fixed latitude.
fn write_ring_csv(path: &Path, lat: f64, n_lon: usize, radians: bool) {
    let mut writer = csv::Writer::from_path(path).expect("create predict csv");
    writer
        .write_record(["lat", "lon", "y"])
        .expect("write header");
    let to_unit = if radians { 1.0 } else { 180.0 / PI };
    for j in 0..n_lon {
        // Span (-π, π) without duplicating the seam endpoint.
        let lon = -PI + 2.0 * PI * (j as f64) / (n_lon as f64);
        writer
            .write_record([
                format!("{:.12}", lat * to_unit),
                format!("{:.12}", lon * to_unit),
                "0.0".to_string(),
            ])
            .expect("write predict row");
    }
    writer.flush().expect("flush predict csv");
}

fn spread(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

/// Fit on a Fibonacci cover and return (pole-ring spread, equator-ring spread)
/// from the saved-model predict pipeline. `pole_lat` is the pole to probe
/// (`±π/2` in radians). Values are emitted to CSV in the unit selected by
/// `radians`.
fn pole_and_equator_spread(radians: bool, pole_lat_rad: f64) -> (f64, f64) {
    let dir = tempfile::tempdir().expect("create tempdir");
    let train_path = dir.path().join("train.csv");
    let model_path = dir.path().join("model.json");
    let pole_path = dir.path().join("pole.csv");
    let equator_path = dir.path().join("equator.csv");
    let pole_out = dir.path().join("pole_out.csv");
    let equator_out = dir.path().join("equator_out.csv");

    write_training_csv(&train_path, 800, radians);
    write_ring_csv(&pole_path, pole_lat_rad, 24, radians);
    write_ring_csv(&equator_path, 0.0, 24, radians);

    let formula = if radians {
        "y ~ sphere(lat, lon, radians=true)"
    } else {
        "y ~ sphere(lat, lon)"
    };

    let mut fit_cmd = Command::new(gam::gam_binary!());
    fit_cmd
        .arg("fit")
        .arg(&train_path)
        .arg(formula)
        .args(["--family", "gaussian"])
        .arg("--out")
        .arg(&model_path);
    run_or_panic(fit_cmd, "gam fit sphere(lat, lon)");
    assert!(model_path.is_file(), "gam fit did not write {model_path:?}");

    for (input, out, label) in [
        (&pole_path, &pole_out, "gam predict (pole ring)"),
        (&equator_path, &equator_out, "gam predict (equator ring)"),
    ] {
        let mut cmd = Command::new(gam::gam_binary!());
        cmd.arg("predict")
            .arg(&model_path)
            .arg(input)
            .arg("--out")
            .arg(out)
            .args(["--mode", "posterior-mean"]);
        run_or_panic(cmd, label);
    }

    let pole = read_prediction_means(&pole_out);
    let equator = read_prediction_means(&equator_out);
    assert!(
        pole.iter().all(|v| v.is_finite()) && equator.iter().all(|v| v.is_finite()),
        "predictions must be finite"
    );
    (spread(&pole), spread(&equator))
}

fn assert_pole_single_valued(radians: bool, pole_lat_rad: f64, label: &str) {
    let (pole_spread, equator_spread) = pole_and_equator_spread(radians, pole_lat_rad);
    // The equatorial signal is cos(lon), spread ≈ 2.0; this anchors the test
    // (if it is near zero the fit failed and the pole assertion is vacuous).
    assert!(
        equator_spread > 0.5,
        "[{label}] equatorial spread {equator_spread} is too small — the fit \
         did not recover the cos(lon) signal, so the pole check is vacuous",
    );
    // The pole is a single physical point: its prediction must not vary with
    // longitude. Allow 2% of the equatorial signal spread as slack.
    let tol = 0.02 * equator_spread;
    assert!(
        pole_spread < tol,
        "[{label}] pole (lat={pole_lat_rad}) is a single point but the \
         saved-model prediction sweeps a longitude profile of spread \
         {pole_spread} (equator spread {equator_spread}, tol {tol}) — predict \
         clamped the pole latitude back into the training range, reintroducing \
         a pole artefact",
    );
}

#[test]
fn sphere_north_pole_predict_is_single_valued_radians() {
    assert!(file!().ends_with(".rs"));
    assert_pole_single_valued(true, PI / 2.0, "north/radians");
}

#[test]
fn sphere_south_pole_predict_is_single_valued_radians() {
    assert!(file!().ends_with(".rs"));
    assert_pole_single_valued(true, -PI / 2.0, "south/radians");
}

#[test]
fn sphere_north_pole_predict_is_single_valued_degrees() {
    assert!(file!().ends_with(".rs"));
    assert_pole_single_valued(false, PI / 2.0, "north/degrees");
}
