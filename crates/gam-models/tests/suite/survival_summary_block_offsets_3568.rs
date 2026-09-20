//! gam#3568: a saved survival fit's term tables read the covariate predictor
//! where the fit put it. A Royston-Parmar fit (Weibull, transformation) holds
//! `[time basis | covariates]` in one block, so the covariate columns start
//! after the time prologue; a location-scale survival fit leads with its
//! time-transform block. Reading `resolved_termspec` at global index 0 put the
//! time coefficients under the covariates' names.
use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::inference::saved_summary::{SummaryPayload, saved_model_summary};

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Shape of the Weibull event law below.
const SHAPE: f64 = 1.5;
/// Age scale of the log-scale effect below.
const AGE_SCALE: f64 = 20.0;

/// Proportional-hazards Weibull data: `H(t | age) = (t / λ(age))^k` with
/// `λ(age) = 10 exp(−(age − 60)/20)`, so `log H = k log t + k (age − 60)/20 −
/// k log 10` and the log-cumulative-hazard coefficient on `age` is `k / 20`.
fn weibull_ph_data(seed: u64) -> EncodedDataset {
    let headers = ["entry", "exit", "event", "age"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut state = seed;
    let rows = (0..500)
        .map(|_| {
            let age = 40.0 + 40.0 * next_unit(&mut state);
            let scale = (-(age - 60.0) / AGE_SCALE).exp() * 10.0;
            let u = next_unit(&mut state).max(f64::MIN_POSITIVE);
            let latent = scale * (-u.ln()).powf(1.0 / SHAPE);
            let censor = 20.0 * next_unit(&mut state);
            StringRecord::from(vec![
                "0".to_string(),
                format!("{:.17e}", latent.min(censor)),
                u8::from(latent <= censor).to_string(),
                format!("{age:.17e}"),
            ])
        })
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

fn summary(formula: &str, data: &EncodedDataset, config: &FitConfig) -> SummaryPayload {
    let payload = fit_formula_to_payload(formula.to_string(), data, config).expect("fit");
    saved_model_summary(&FittedModel::from_payload(payload)).expect("summary")
}

fn survival_config(likelihood: &str) -> FitConfig {
    FitConfig {
        survival_likelihood: Some(likelihood.to_string()),
        ..FitConfig::default()
    }
}

/// `(estimate, statistic)` of the `age` row of a summary's parametric table.
fn age_row(summary: &SummaryPayload, label: &str) -> (f64, f64) {
    assert_eq!(
        summary.parametric_terms_unavailable, None,
        "{label}: parametric table absent"
    );
    let names: Vec<&str> = summary
        .parametric_terms
        .iter()
        .map(|row| row.name.as_str())
        .collect();
    let row = summary
        .parametric_terms
        .iter()
        .find(|row| row.name == "age")
        .unwrap_or_else(|| panic!("{label}: no `age` row in {names:?}"));
    let statistic = row
        .statistic
        .unwrap_or_else(|| panic!("{label}: `age` row has no Wald statistic"));
    (row.estimate, statistic)
}

#[test]
fn royston_parmar_covariate_rows_read_past_the_time_prologue_3568() {
    let data = weibull_ph_data(0x3568);
    let truth = SHAPE / AGE_SCALE;
    let weibull = summary("Surv(entry, exit, event) ~ age", &data, &survival_config("weibull"));
    let transformation = summary(
        "Surv(entry, exit, event) ~ age",
        &data,
        &survival_config("transformation"),
    );
    let (weibull_estimate, weibull_statistic) = age_row(&weibull, "weibull");
    let (transformation_estimate, transformation_statistic) =
        age_row(&transformation, "transformation");
    // With ~300 events over an age spread of SD ≈ 11.5 the standard error is
    // about 1/(√300 · 11.5) ≈ 0.005, so four of them is 0.02. Reading the
    // time prologue's coefficients instead lands on the log-t slope (≈ k).
    for (label, estimate) in [
        ("weibull", weibull_estimate),
        ("transformation", transformation_estimate),
    ] {
        assert!(
            (estimate - truth).abs() <= 0.02,
            "{label}: age estimate {estimate} vs the generating log-H coefficient {truth}"
        );
    }
    // Both fits are proportional-hazards models of the same data and differ
    // only in the baseline's flexibility, so their covariate tables agree. At
    // #3087 the Weibull Wald statistic was about five times the
    // transformation one.
    let ratio = weibull_statistic / transformation_statistic;
    assert!(
        (2.0 / 3.0..=1.5).contains(&ratio),
        "weibull vs transformation age Wald statistic: {weibull_statistic} vs \
         {transformation_statistic}"
    );
}

#[test]
fn royston_parmar_smooth_rows_name_and_test_the_covariate_3568() {
    let data = weibull_ph_data(0x3568);
    for likelihood in ["weibull", "transformation"] {
        let summary = summary(
            "Surv(entry, exit, event) ~ s(age)",
            &data,
            &survival_config(likelihood),
        );
        assert_eq!(
            summary.smooth_terms_unavailable, None,
            "{likelihood}: per-smooth table absent"
        );
        let names: Vec<&str> = summary
            .smooth_terms
            .iter()
            .map(|row| row.name.as_str())
            .collect();
        assert_eq!(names.len(), 1, "{likelihood}: one row for s(age): {names:?}");
        assert!(names[0].contains("age"), "{likelihood}: {names:?}");
        let row = &summary.smooth_terms[0];
        assert!(
            row.edf.is_finite() && row.edf > 0.0,
            "{likelihood}: edf {}",
            row.edf
        );
        // A strong monotone age effect (≈ 15 standard errors): the smooth's
        // test must reject.
        let p = row.p_value.unwrap_or_else(|| {
            panic!("{likelihood}: no p-value ({:?})", row.p_value_unavailable)
        });
        assert!(p < 1e-6, "{likelihood}: s(age) p-value {p}");
    }
}

#[test]
fn location_scale_survival_tables_name_the_covariate_terms_3568() {
    let data = weibull_ph_data(0x3568);
    let config = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some("age".to_string()),
        ..FitConfig::default()
    };
    let summary = summary("Surv(entry, exit, event) ~ s(age)", &data, &config);
    assert_eq!(
        summary.smooth_terms_unavailable, None,
        "per-smooth table absent"
    );
    let names: Vec<&str> = summary
        .smooth_terms
        .iter()
        .map(|row| row.name.as_str())
        .collect();
    assert_eq!(names.len(), 1, "one row for s(age): {names:?}");
    assert!(names[0].contains("age"), "{names:?}");
    let row = &summary.smooth_terms[0];
    assert!(row.edf.is_finite() && row.edf > 0.0, "edf {}", row.edf);
    let p = row
        .p_value
        .unwrap_or_else(|| panic!("no p-value ({:?})", row.p_value_unavailable));
    assert!(p < 1e-6, "s(age) p-value {p}");
    assert_eq!(
        summary.parametric_terms_unavailable, None,
        "parametric table absent"
    );
}
