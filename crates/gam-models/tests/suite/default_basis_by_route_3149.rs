//! #3149: a formula default nobody sized starts at the penalized-resolution
//! pilot only where the resolution loop grows it (the standard formula
//! workflow). A location-scale fit has no such loop, so its mean `s(x)` keeps
//! the provisioned default, adequate without growth: 12 cubic functions, 11
//! columns after its sum-to-zero constraint, where #3191's pilot handed it 6
//! functions with nothing to grow them.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitRequest, materialize};
use gam_terms::smooth::build_term_collection_design;

fn dataset(n: usize) -> EncodedDataset {
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let x = i as f64 / (n - 1) as f64;
            let y = (2.0 * std::f64::consts::PI * 3.0 * x).sin() + 0.1 * ((i * 7919) % 13) as f64;
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    let headers = ["x", "y"].iter().map(|s| s.to_string()).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

#[test]
fn location_scale_mean_smooth_keeps_the_provisioned_default_3149() {
    let data = dataset(1000);
    let config = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x)".to_string()),
        ..FitConfig::default()
    };
    let materialized = materialize("y ~ s(x)", &data, &config).expect("location-scale request");
    let FitRequest::GaussianLocationScale(request) = materialized.request else {
        panic!("expected a Gaussian location-scale request");
    };
    let mean_design = build_term_collection_design(data.values.view(), &request.spec.meanspec)
        .expect("mean design");
    let columns = mean_design.smooth.terms[0].coeff_range.len();
    assert_eq!(
        columns, 11,
        "a location-scale mean s(x) at n = 1000 must keep the provisioned 11-column default, got {columns}"
    );
}
