//! Family auto-detection for a two-level label response, and the row-count
//! floor that runs before any family is inferred (pyGAM audit DOC-18 / F7).
//!
//! These go through `fit_from_formula_with_notes` on a CSV-encoded dataset,
//! the same service and encoder `gam fit` uses, so they pin the CLI as well
//! as the library.

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{
    FitConfig, FitResult, StandardFitResult, fit_from_formula_with_notes,
};
use gam_terms::fit_notes::FitNotes;
use ndarray::Array1;

const N: usize = 240;

/// `x` on a grid and a binary outcome whose event rate rises with `x`,
/// thresholded against a golden-ratio sequence so the fixture is deterministic.
fn outcome_rows() -> Vec<(f64, bool)> {
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    (0..N)
        .map(|i| {
            let x = i as f64 / (N - 1) as f64;
            let probability = 1.0 / (1.0 + (-(3.0 * x - 1.5)).exp());
            let u = (i as f64 * golden).fract();
            (x, u < probability)
        })
        .collect()
}

fn encode(headers: &[&str], rows: Vec<Vec<String>>) -> EncodedDataset {
    encode_recordswith_inferred_schema(
        headers.iter().map(|h| h.to_string()).collect(),
        rows.into_iter().map(StringRecord::from).collect(),
    )
    .expect("encode fixture")
}

/// The outcome spelled as labels. `yes` is written first so encounter order and
/// sorted order disagree: the coding must follow sorted order.
fn label_data() -> EncodedDataset {
    let rows = outcome_rows()
        .into_iter()
        .map(|(x, event)| vec![x.to_string(), if event { "yes" } else { "no" }.to_string()])
        .collect();
    encode(&["x", "y"], rows)
}

fn numeric_data() -> EncodedDataset {
    let rows = outcome_rows()
        .into_iter()
        .map(|(x, event)| vec![x.to_string(), if event { "1" } else { "0" }.to_string()])
        .collect();
    encode(&["x", "y"], rows)
}

fn fit_standard(data: &EncodedDataset, family: Option<&str>) -> (StandardFitResult, FitNotes) {
    let config = FitConfig {
        family: family.map(str::to_string),
        ..FitConfig::default()
    };
    let outcome = fit_from_formula_with_notes("y ~ s(x)", data, &config)
        .unwrap_or_else(|error| panic!("fit with family {family:?} failed: {error}"));
    let FitResult::Standard(fit) = outcome.result else {
        panic!("expected a standard fit");
    };
    (fit, outcome.inference_notes)
}

fn fitted_probabilities(fit: &StandardFitResult) -> Array1<f64> {
    let eta = fit.design.design.to_dense().dot(&fit.fit.beta);
    eta.mapv(|value| 1.0 / (1.0 + (-value).exp()))
}

#[test]
fn a_two_level_label_response_is_auto_detected_as_binomial() {
    let (labels, notes) = fit_standard(&label_data(), None);
    let family = labels
        .fit
        .likelihood_family
        .as_ref()
        .expect("the fit records its family");
    assert!(
        family.is_binomial(),
        "two labels must infer binomial, got {family:?}"
    );

    // Sorted order codes 'no' = 0 and 'yes' = 1, so the fit is the numeric 0/1
    // fit of the same outcome, coefficient for coefficient.
    let (numeric, _) = fit_standard(&numeric_data(), Some("binomial"));
    assert_eq!(labels.fit.beta.len(), numeric.fit.beta.len());
    for (a, b) in labels.fit.beta.iter().zip(numeric.fit.beta.iter()) {
        assert!(
            (a - b).abs() <= 1e-8 * (1.0 + b.abs()),
            "label fit {a} vs numeric fit {b}"
        );
    }

    // The fitted probabilities are P(y = 'yes'): they rise with x, and the
    // unpenalized intercept's score equation makes them average to the
    // observed 'yes' share.
    let mu = fitted_probabilities(&labels);
    assert!(
        mu[N - 1] > 0.7 && mu[0] < 0.3,
        "P(yes) must rise with x: {} .. {}",
        mu[0],
        mu[N - 1]
    );
    let observed = outcome_rows().iter().filter(|(_, event)| *event).count() as f64 / N as f64;
    let fitted = mu.mean().expect("nonempty");
    assert!(
        (fitted - observed).abs() < 1e-3,
        "mean P(yes) {fitted} vs observed {observed}"
    );

    assert!(
        notes
            .iter()
            .any(|note| note.contains("'no' = 0") && note.contains("'yes' = 1")),
        "the summary must state the level coding: {:?}",
        notes
    );

    // Naming the family explicitly codes the labels the same way.
    let (explicit, _) = fit_standard(&label_data(), Some("binomial"));
    assert_eq!(explicit.fit.beta, labels.fit.beta);
}

fn assert_too_few_rows(data: &EncodedDataset, config: &FitConfig, case: &str) {
    let error = match fit_from_formula_with_notes("y ~ s(x)", data, config) {
        Ok(_) => panic!("{case}: a fit with no residual contrast must be refused"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("too few rows"), "{case}: {error}");
}

#[test]
fn a_single_row_reports_too_few_rows_before_the_family_is_inferred() {
    let one_row = encode(&["x", "y"], vec![vec!["0.5".to_string(), "1".to_string()]]);
    // Auto-detection would call y = [1] a degenerate binomial, and an explicit
    // family would call the lone x value a constant covariate; the row count
    // is the actual problem in every case.
    for family in [None, Some("binomial"), Some("gaussian"), Some("poisson")] {
        let config = FitConfig {
            family: family.map(str::to_string),
            ..FitConfig::default()
        };
        assert_too_few_rows(&one_row, &config, &format!("standard, family {family:?}"));
    }
    let location_scale = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x)".to_string()),
        ..FitConfig::default()
    };
    assert_too_few_rows(&one_row, &location_scale, "location-scale");

    // Zero-weight rows carry no information, so they do not count.
    let zero_weights = encode(
        &["x", "y", "w"],
        (0..4)
            .map(|i| {
                vec![
                    (i as f64 / 3.0).to_string(),
                    (i % 2).to_string(),
                    "0".to_string(),
                ]
            })
            .collect(),
    );
    let weighted = FitConfig {
        weight_column: Some("w".to_string()),
        ..FitConfig::default()
    };
    assert_too_few_rows(&zero_weights, &weighted, "all-zero weights");
}
