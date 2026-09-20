//! gam#3425: the F15 synthetic chicago Poisson model
//! `y ~ s(time) + s(tmpd) + te(pm10, o3)` on `_chicago_like(1, 1000)` must end
//! on a certified outer optimum.
//!
//! The rows are drawn by NumPy `default_rng(1)` in
//! `tests/pygam_covariance_refusal_test.py::_chicago_like` and stored as CSV
//! (17-significant-digit `repr` floats) because the generator is not
//! reproducible from Rust. On main the `s(time)` refinement to 47 internal
//! knots stopped on `ArcUnprogressingStallCheckpoint` with a PSD analytic outer
//! Hessian and `|Pg| = 3.0e-2` against a stationarity bound of `1.5e-2`.
//!
//! A fit is only minted from a converged optimization, so a returned payload is
//! itself the certificate.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;

const CHICAGO_LIKE_SEED_1_N_1000: &str =
    include_str!("fixtures/poisson_chicago_like_3425_seed1_n1000.csv");

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{}", record.args());
        }
    }

    fn flush(&self) {}
}

static STDERR_LOGGER: StderrLogger = StderrLogger;

fn dataset(csv_text: &str) -> EncodedDataset {
    let mut reader = csv::Reader::from_reader(csv_text.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .expect("fixture header")
        .iter()
        .map(str::to_string)
        .collect();
    let rows: Vec<StringRecord> = reader
        .records()
        .map(|record| record.expect("fixture row"))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture")
}

#[test]
fn poisson_chicago_like_seed1_n1000_certifies_3425() {
    // The outer-search trace is the evidence this test exists to capture when
    // it fails; the global logger may already be owned by another test in this
    // binary, in which case that sink receives the records instead.
    if log::set_logger(&STDERR_LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Debug);
    }
    let data = dataset(CHICAGO_LIKE_SEED_1_N_1000);
    let config = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload(
        "y ~ s(time) + s(tmpd) + te(pm10, o3)".to_string(),
        &data,
        &config,
    )
    .unwrap_or_else(|error| panic!("the outer search must reach a certified optimum: {error}"));
    let fit = payload.fit_result.as_ref().expect("the payload carries its fit");
    assert!(
        fit.reml_score().is_some_and(f64::is_finite),
        "the certified fit has a finite criterion"
    );
}
