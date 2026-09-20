//! gam#3305: a one-smooth binomial `y ~ s(x)` on 200 rows must end on a
//! certified outer optimum.
//!
//! The fixtures are the rows of the two Python tests that raised
//! `RemlConvergenceError` (`test_diagnose_auto_selects_classification_metrics_for_binary_family`
//! and `test_gamclassifier_score_is_accuracy_and_metrics_panel_is_sane`):
//! `x = sort(U(-3, 3))`, `y ~ Bernoulli(logistic(slope·x))` with slope 2.0 and
//! 2.5, drawn by NumPy `default_rng(20260601)` and `default_rng(20260602)`, and
//! stored as CSV because the generator is not reproducible from Rust. The truth
//! is linear in `x`, so the wiggle penalty's optimum lies far out on the λ→∞
//! ridge (`rho_checkpoint = [22.89, -4.69]` in the report).
//!
//! A fit is only minted from a converged optimization, so a returned payload is
//! itself the certificate.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;

const SLOPE_2_0_SEED_20260601: &str =
    include_str!("fixtures/binomial_outer_certificate_3305_seed20260601.csv");
const SLOPE_2_5_SEED_20260602: &str =
    include_str!("fixtures/binomial_outer_certificate_3305_seed20260602.csv");

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

fn assert_certified(label: &str, csv_text: &str) {
    let data = dataset(csv_text);
    let config = FitConfig {
        family: Some("binomial".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ s(x)".to_string(), &data, &config)
        .unwrap_or_else(|error| {
            panic!("[{label}] the outer search must reach a certified optimum: {error}")
        });
    let fit = payload.fit_result.as_ref().expect("the payload carries its fit");
    assert!(
        fit.reml_score().is_some_and(f64::is_finite),
        "[{label}] the certified fit has a finite criterion"
    );
}

#[test]
fn binomial_smooth_slope_2_0_certifies_3305() {
    assert_certified("slope 2.0 seed 20260601", SLOPE_2_0_SEED_20260601);
}

#[test]
fn binomial_smooth_slope_2_5_certifies_3305() {
    assert_certified("slope 2.5 seed 20260602", SLOPE_2_5_SEED_20260602);
}
