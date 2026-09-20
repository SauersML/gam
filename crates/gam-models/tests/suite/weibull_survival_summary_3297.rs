//! gam#3297: the summary of a Weibull (Royston-Parmar) survival fit reports no
//! scalar dispersion, because that family's scale contract has none, instead
//! of refusing the whole summary (and with it Python's `repr`).
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::inference::saved_summary::saved_model_summary;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn weibull_survival_summary_reports_no_scalar_dispersion_3297() {
    let headers = ["entry", "exit", "event", "age"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut state = 0x3297u64;
    let rows = (0..500)
        .map(|_| {
            let age = 40.0 + 40.0 * next_unit(&mut state);
            let scale = (-(age - 60.0) / 20.0).exp() * 10.0;
            let u = next_unit(&mut state).max(f64::MIN_POSITIVE);
            let latent = scale * (-u.ln()).powf(1.0 / 1.5);
            let censor = 20.0 * next_unit(&mut state);
            StringRecord::from(vec![
                "0".to_string(),
                format!("{:.17e}", latent.min(censor)),
                u8::from(latent <= censor).to_string(),
                format!("{age:.17e}"),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let config = FitConfig {
        survival_likelihood: Some("weibull".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload(
        "Surv(entry, exit, event) ~ s(age)".to_string(),
        &data,
        &config,
    )
    .expect("fit");
    let summary = saved_model_summary(&FittedModel::from_payload(payload)).expect("summary");
    assert_eq!(
        summary.scale, None,
        "Royston-Parmar has no scalar dispersion"
    );
    assert!(summary.deviance.is_finite());
    assert!(!summary.coefficients.is_empty());
    // A full likelihood with no dispersion: the conditional AIC is defined and
    // spends no scale degree of freedom; the WPS correction needs a
    // coefficient-covariance scale this family does not have.
    let criteria = &summary.information_criteria;
    assert_eq!(criteria.scale_dof, Some(0.0));
    let log_likelihood = summary.log_likelihood.expect("log-likelihood");
    let edf = summary.edf_total.expect("conditional EDF");
    let aic = criteria.aic_conditional.expect("conditional AIC");
    assert!((aic - (-2.0 * log_likelihood + 2.0 * edf)).abs() <= 1e-9 * aic.abs().max(1.0));
    assert_eq!(criteria.aic_corrected, None);
    assert!(criteria.aic_corrected_unavailable.is_some());
}
