//! #2937: a survival transformation fit whose smoothing search cannot certify
//! raises the category of the engine error that stopped it.
//!
//! `fit_survival_transformation_model` used to hand back text, and `fit_model`
//! recorded that text as `FitFailure::Unclassified`, so Python raised the bare
//! `FitError`. On main 666558f622 this fixture reported
//! `variant=FitFailure::Unclassified category=Unclassified` with the message
//! "REML smoothing optimization failed to converge: survival transformation
//! smoothing-parameter selection (dim=3): ..." (job 1217032). The outer search's
//! `EstimationError` now reaches the boundary whole, so the class names
//! convergence and the message is the engine's.
//!
//! Which convergence refusal ends the search is the engine's choice. By
//! 19143a1511 it had become `EstimationError::TrialPointRefused` ("survival
//! transformation inner P-IRLS at this trial rho ended with status
//! LmStepSearchExhausted ...", sw1a job 1277503), so the pin names the category,
//! the typed variant and the route's own words, not one verdict's text.

use csv::StringRecord;
use gam::{
    FailureCategory, FitConfig, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};

/// Events at short times and censoring at long times: the time effect separates
/// the outcome, and the smoothing search cannot certify an optimum.
fn separable_survival_rows(n: usize) -> gam::data::EncodedDataset {
    let headers = vec!["time".to_string(), "event".to_string(), "x".to_string()];
    let rows = (0..n)
        .map(|i| {
            let time = if i < n / 2 {
                1.0 + i as f64 * 0.01
            } else {
                50.0 + i as f64
            };
            let event = u8::from(i < n / 2);
            let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            StringRecord::from(vec![time.to_string(), event.to_string(), x.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2937 fixture")
}

#[test]
fn survival_transformation_search_failure_raises_its_category_2937() {
    init_parallelism();
    let data = separable_survival_rows(80);
    let config = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        ..FitConfig::default()
    };
    let error = match fit_from_formula("Surv(time, event) ~ s(x, k=8)", &data, &config) {
        Ok(_) => panic!("#2937: the separable fixture must refuse, not mint a fit"),
        Err(error) => error,
    };
    eprintln!(
        "[#2937] variant={} category={:?} message={error}",
        error.variant_name(),
        error.failure_category()
    );
    assert_eq!(
        error.failure_category(),
        FailureCategory::Convergence,
        "{error}"
    );
    assert!(
        error.variant_name().starts_with("EstimationError::"),
        "the engine error must reach the boundary typed, got {}",
        error.variant_name()
    );
    assert!(
        error.to_string().contains("survival transformation"),
        "the message must stay the engine's: {error}"
    );
}
