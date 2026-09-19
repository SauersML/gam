//! #2937: a survival location-scale fit with a non-Linear baseline runs a
//! baseline-config search at materialization, fitting the model at every
//! candidate. The search returned text, and `From<String> for WorkflowError`
//! filed its every failure as `WorkflowError::InvalidConfig`, so Python raised
//! `InvalidConfigurationError` for a search that did not converge. On main
//! da56cb7d6f this fixture reported `variant=WorkflowError::InvalidConfig
//! category=Input` with "workflow survival location-scale baseline failed: Outer
//! smoothing-parameter optimization did not certify a stationary optimum ..."
//! (sw1a job 1283012). The search's verdict now reaches the boundary typed, as a
//! convergence failure with the same message, while a candidate spec the data
//! cannot support stays a configuration refusal.

use csv::StringRecord;
use gam::{
    FailureCategory, FitConfig, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};

fn survival_rows(n: usize, row: impl Fn(usize) -> (f64, u8)) -> gam::data::EncodedDataset {
    let headers = vec!["time".to_string(), "event".to_string(), "x".to_string()];
    let rows = (0..n)
        .map(|i| {
            let (time, event) = row(i);
            let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            StringRecord::from(vec![time.to_string(), event.to_string(), x.to_string()])
        })
        .collect::<Vec<_>>();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #2937 fixture")
}

const FORMULA: &str = "Surv(time, event) ~ s(x, k=6)";

fn weibull_location_scale() -> FitConfig {
    FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        baseline_target: "weibull".to_string(),
        ..FitConfig::default()
    }
}

#[test]
fn survival_location_scale_baseline_search_failure_raises_its_category_2937() {
    init_parallelism();
    // Eight rows: too few for the search to certify a stationary baseline.
    let data = survival_rows(8, |i| (1.0 + i as f64, u8::from(i % 3 == 0)));
    let error = match fit_from_formula(FORMULA, &data, &weibull_location_scale()) {
        Ok(_) => panic!("#2937: the eight-row fixture must refuse, not mint a fit"),
        Err(error) => error,
    };
    eprintln!(
        "[#2937] variant={} category={:?} message={error}",
        error.variant_name(),
        error.failure_category()
    );
    assert_eq!(error.failure_category(), FailureCategory::Convergence, "{error}");
    assert!(
        error.variant_name().starts_with("EstimationError::"),
        "the search's verdict must reach the boundary typed, got {}",
        error.variant_name()
    );
    assert!(
        error
            .to_string()
            .contains("workflow survival location-scale baseline"),
        "the message must stay the search's: {error}"
    );

    // A configuration refusal on the same route stays one: tied exit times give
    // the time basis no knot domain.
    let tied = survival_rows(30, |i| (5.0, u8::from(i % 2 == 0)));
    let refusal = match fit_from_formula(FORMULA, &tied, &weibull_location_scale()) {
        Ok(_) => panic!("#2937: tied exit times must refuse"),
        Err(error) => error,
    };
    assert_eq!(refusal.variant_name(), "WorkflowError::InvalidConfig", "{refusal}");
}
