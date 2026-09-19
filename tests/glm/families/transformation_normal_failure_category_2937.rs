//! #2937: a transformation-normal fit refused by its response basis raises the
//! category of that refusal.
//!
//! `fit_transformation_normal` returns `FitFailure`, but the text its helpers
//! return reached it through `From<String>`, which records a string as
//! unclassified, so Python raised the bare `FitError`. The response basis
//! validates the response and the configuration it is built from, so its
//! refusal now reaches the boundary as an input failure with its message
//! unchanged.

use csv::StringRecord;
use gam::{
    FailureCategory, FitConfig, FitRequest, encode_recordswith_inferred_schema, fit_model,
    init_parallelism, materialize,
};

#[test]
fn transformation_normal_response_basis_refusal_raises_its_category_2937() {
    init_parallelism();
    let n = 60;
    let headers = vec!["x".to_string(), "y".to_string()];
    let records = (0..n)
        .map(|i| {
            let x = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            let y = x + 0.25 * (7.0 * x).sin();
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect::<Vec<_>>();
    let data = encode_recordswith_inferred_schema(headers, records)
        .expect("encode the #2937 transformation-normal fixture");
    let config = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let mut materialized = materialize("y ~ s(x, k=6)", &data, &config)
        .expect("materialize the transformation-normal request");
    let FitRequest::TransformationNormal(ref mut request) = materialized.request else {
        panic!("expected a TransformationNormal fit request");
    };
    // An I-spline response basis needs degree at least 1; the builder refuses 0.
    request.config.response_degree = 0;
    let error = match fit_model(materialized.request) {
        Ok(_) => panic!("#2937: a degree-0 response basis must be refused, not fitted"),
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
            .contains("response_degree must be >= 1 for the I-spline basis, got 0"),
        "the message must stay the response basis builder's: {error}"
    );
}
