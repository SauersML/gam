//! #2997: `summary()` of a Bernoulli marginal-slope fit carries one row per
//! smooth of BOTH predictors — the marginal formula's and the slope
//! formula's — each read against its own block's λ and coefficient range.
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::utils::splitmix64;
use gam_math::probability::normal_cdf;
use gam_models::fit_orchestration::FitConfig;
use gam_models::inference::model::FittedModel;
use gam_models::inference::model_payload_builders::fit_formula_to_payload;
use gam_models::inference::saved_summary::saved_model_summary;

fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state).max(f64::MIN_POSITIVE);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[test]
fn bms_summary_has_one_row_per_smooth_of_both_predictors_2997() {
    let headers = ["y", "x", "w", "z"].iter().map(|s| s.to_string()).collect();
    let mut state = 0x2997u64;
    let rows = (0..1000)
        .map(|_| {
            let x = 4.0 * next_unit(&mut state) - 2.0;
            let w = 4.0 * next_unit(&mut state) - 2.0;
            let z = next_gauss(&mut state);
            let eta = 0.5 * x.sin() + (0.6 + 0.2 * w.tanh()) * z;
            let y = u8::from(next_unit(&mut state) < normal_cdf(eta));
            StringRecord::from(vec![
                y.to_string(),
                format!("{x:.17e}"),
                format!("{w:.17e}"),
                format!("{z:.17e}"),
            ])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode dataset");
    let config = FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1 + s(w)".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("y ~ s(x)".to_string(), &data, &config).expect("fit");
    let model = FittedModel::from_payload(payload);
    let summary = saved_model_summary(&model).expect("summary");
    assert_eq!(
        summary.smooth_terms_unavailable, None,
        "per-smooth table absent; λ = {:?}",
        summary.lambdas
    );
    let names: Vec<&str> = summary
        .smooth_terms
        .iter()
        .map(|row| row.name.as_str())
        .collect();
    assert_eq!(
        names.len(),
        2,
        "one row per smooth of both predictors: {names:?}"
    );
    assert!(
        names.iter().any(|name| name.contains('x')),
        "marginal s(x) row: {names:?}"
    );
    assert!(
        names.iter().any(|name| name.contains('w')),
        "slope s(w) row: {names:?}"
    );
    let smooth_predictors: Vec<(Option<&str>, &str)> = summary
        .smooth_terms
        .iter()
        .map(|row| (row.predictor, row.name.as_str()))
        .collect();
    assert!(
        smooth_predictors
            .iter()
            .any(|(predictor, name)| *predictor == Some("marginal") && name.contains('x'))
            && smooth_predictors
                .iter()
                .any(|(predictor, name)| *predictor == Some("slope") && name.contains('w')),
        "each smooth row names its predictor: {smooth_predictors:?}"
    );
    // Both formulas carry an intercept; each is read at its own block's
    // coefficient, so the parametric table holds one per predictor, tagged.
    assert_eq!(
        summary.parametric_terms_unavailable, None,
        "parametric table absent"
    );
    let parametric: Vec<(Option<&str>, &str)> = summary
        .parametric_terms
        .iter()
        .map(|row| (row.predictor, row.name.as_str()))
        .collect();
    assert_eq!(
        parametric,
        vec![(Some("marginal"), "Intercept"), (Some("slope"), "Intercept")],
        "one intercept row per predictor"
    );
    // The slope predictor is the fitted baseline plus the slope block, so the
    // slope intercept is read at the slope block's coefficient when baseline
    // and intercept recover the simulated mean slope 0.6 (`E tanh(w) = 0`).
    let baseline_slope = model.baseline_slope.expect("fitted slope baseline");
    let mean_slope = baseline_slope + summary.parametric_terms[1].estimate;
    assert!(
        (mean_slope - 0.6).abs() < 0.15,
        "slope baseline {baseline_slope} + slope intercept {} = {mean_slope}, simulated 0.6",
        summary.parametric_terms[1].estimate
    );
    for row in &summary.smooth_terms {
        assert!(
            row.edf.is_finite() && row.edf > 0.0,
            "{}: edf {}",
            row.name,
            row.edf
        );
        let p = row
            .p_value
            .unwrap_or_else(|| panic!("{}: no p-value", row.name));
        assert!((0.0..=1.0).contains(&p), "{}: p {p}", row.name);
    }
}
