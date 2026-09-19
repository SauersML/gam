//! #2937: a Bernoulli marginal-slope fit refused by its own spec check raises
//! the category of that refusal.
//!
//! `fit_bernoulli_marginal_slope_terms` returns `FitFailure`, but the text its
//! helpers return reached it through `From<String>`, which records a string as
//! unclassified, so Python raised the bare `FitError`. The spec check validates
//! the response, weights, score and offsets the fit is handed, so a response
//! outside {0, 1} now reaches the boundary as an input failure with its message
//! unchanged.

use csv::StringRecord;
use gam::{
    FailureCategory, FitConfig, FitRequest, encode_recordswith_inferred_schema, fit_model,
    init_parallelism, materialize,
};

#[test]
fn bernoulli_marginal_slope_spec_refusal_raises_its_category_2937() {
    init_parallelism();
    let n = 80;
    let headers = vec!["y".to_string(), "x".to_string(), "z".to_string()];
    let records = (0..n)
        .map(|i| {
            let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            let z = (1.7 * (i as f64)).sin();
            let y = u8::from(x + 0.5 * z > 0.0);
            StringRecord::from(vec![y.to_string(), x.to_string(), z.to_string()])
        })
        .collect::<Vec<_>>();
    let data = encode_recordswith_inferred_schema(headers, records)
        .expect("encode the #2937 marginal-slope fixture");
    let config = FitConfig {
        slope_formula: Some("x".to_string()),
        z_column: Some("z".to_string()),
        ..FitConfig::default()
    };
    let mut materialized = materialize("y ~ x", &data, &config)
        .expect("materialize the Bernoulli marginal-slope request");
    let FitRequest::BernoulliMarginalSlope(ref mut request) = materialized.request else {
        panic!("expected a BernoulliMarginalSlope fit request");
    };
    // The spec check reads the response exactly: 0.5 is not an outcome.
    request.spec.y[0] = 0.5;
    let error = match fit_model(materialized.request) {
        Ok(_) => panic!("#2937: a response outside {{0, 1}} must be refused, not fitted"),
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
            .contains("bernoulli-marginal-slope requires binary y in {0,1}"),
        "the message must stay the spec check's: {error}"
    );
}
