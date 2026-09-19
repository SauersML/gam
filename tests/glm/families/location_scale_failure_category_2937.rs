//! #2937: a location-scale fit refused by its own spec checks raises the
//! category of that refusal.
//!
//! `fit_location_scale_terms` and its entry points handed back text, and
//! `fit_binomial_location_scale_model` recorded that text as
//! `FitFailure::Unclassified`, so Python raised the bare `FitError`. On main
//! b88dad1cad a binomial location-scale fit with a smooth log-σ reported
//! `variant=FitFailure::Unclassified category=Unclassified` with the message
//! "fit_binomial_location_scale_terms: Bernoulli binomial location-scale data
//! identify only the composite q = -threshold / sigma; ..." (sw1a job 1234148).
//! The builders now return `FitFailure`, so the refusal reaches the boundary
//! as an input failure with its message unchanged.

use csv::StringRecord;
use gam::{
    FailureCategory, FitConfig, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};

fn rows(
    headers: &[&str],
    n: usize,
    row: impl Fn(usize) -> Vec<String>,
) -> gam::data::EncodedDataset {
    let headers = headers.iter().map(|h| h.to_string()).collect::<Vec<_>>();
    let rows = (0..n).map(|i| StringRecord::from(row(i))).collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2937 fixture")
}

fn grid(i: usize, n: usize) -> f64 {
    -1.0 + 2.0 * (i as f64) / ((n - 1) as f64)
}

#[test]
fn binomial_location_scale_spec_refusal_raises_its_category_2937() {
    init_parallelism();
    let n = 80;
    let data = rows(&["y", "x"], n, |i| {
        vec![u8::from(grid(i, n) > 0.0).to_string(), grid(i, n).to_string()]
    });
    let config = FitConfig {
        family: Some("binomial".to_string()),
        noise_formula: Some("s(x, k=6)".to_string()),
        ..FitConfig::default()
    };
    let error = match fit_from_formula("y ~ s(x, k=8)", &data, &config) {
        Ok(_) => panic!("#2937: a smooth Bernoulli log-sigma must be refused, not fitted"),
        Err(error) => error,
    };
    eprintln!(
        "[#2937] variant={} category={:?} message={error}",
        error.variant_name(),
        error.failure_category()
    );
    assert_eq!(error.failure_category(), FailureCategory::Input, "{error}");
    assert_eq!(error.variant_name(), "FitFailure::Input", "{error}");
    assert!(
        error
            .to_string()
            .contains("identify only the composite q = -threshold / sigma"),
        "the message must stay the validator's: {error}"
    );
}
