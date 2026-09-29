//! #2937: a survival location-scale fit files a request the data or the
//! likelihood cannot support as a configuration refusal, and nothing else.
//!
//! The issue was the converse: `From<String> for WorkflowError` filed every
//! failure of the location-scale baseline-config search as
//! `WorkflowError::InvalidConfig`, so Python raised `InvalidConfigurationError`
//! for a search that did not converge (main da56cb7d6f, sw1a job 1283012). #3413
//! retired that search: the baseline's θ is a family-owned outer axis selected
//! with ρ, so a selection that does not converge is the outer search's own typed
//! error. What stays on this route are the two refusals below, each raised
//! before any fit runs.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism};

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

fn weibull_location_scale(noise_formula: Option<&str>) -> FitConfig {
    FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        baseline_target: "weibull".to_string(),
        noise_formula: noise_formula.map(str::to_string),
        ..FitConfig::default()
    }
}

#[test]
fn survival_location_scale_baseline_refusals_are_configuration_errors_2937() {
    init_parallelism();

    // A target the likelihood cannot read: with a constant scale and no time
    // wiggle the fit replaces its time warp with `-log t` on the location
    // channel and reads no time offset, so the target has no parameter.
    let data = survival_rows(30, |i| (1.0 + i as f64, u8::from(i % 3 == 0)));
    let unread = match fit_from_formula(FORMULA, &data, &weibull_location_scale(None)) {
        Ok(_) => panic!("#2937: an unread baseline target must refuse"),
        Err(error) => error,
    };
    assert_eq!(unread.variant_name(), "WorkflowError::InvalidConfig", "{unread}");

    // Tied exit times give the time basis no knot domain.
    let tied = survival_rows(30, |i| (5.0, u8::from(i % 2 == 0)));
    let refusal = match fit_from_formula(FORMULA, &tied, &weibull_location_scale(Some("x"))) {
        Ok(_) => panic!("#2937: tied exit times must refuse"),
        Err(error) => error,
    };
    assert_eq!(refusal.variant_name(), "WorkflowError::InvalidConfig", "{refusal}");
}
