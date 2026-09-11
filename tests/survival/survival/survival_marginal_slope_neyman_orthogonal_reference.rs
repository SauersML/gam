//! CTN composition validates its sampling boundary before any survival fit.
//! Cross-fitting alone is not a claim of orthogonality.
use gam::{CtnStage1Recipe, FitConfig, encode_recordswith_inferred_schema, fit_from_formula};
use gam::transformation_normal::TransformationNormalConfig;

#[test]
fn survival_ctn_requires_explicit_independence_groups_or_folds() {
    let headers = vec!["entry".into(), "exit".into(), "event".into(), "score".into()];
    let rows = (0..12).map(|i| csv::StringRecord::from(vec![
        "0".into(), (i + 1).to_string(), (i % 2).to_string(), (i as f64 / 12.0).to_string(),
    ])).collect::<Vec<_>>();
    let data = encode_recordswith_inferred_schema(headers, rows).unwrap();
    let recipe = CtnStage1Recipe::new("score", "1", TransformationNormalConfig::default(), None, None).unwrap();
    let config = FitConfig { ctn_stage1: Some(recipe), survival_likelihood: Some("marginal-slope".into()),
        slope_formula: Some("1".into()), ..FitConfig::default() };
    let result = fit_from_formula("Surv(entry, exit, event) ~ 1", &data, &config);
    let error = match result { Ok(_) => panic!("CTN accepted unspecified folds"), Err(error) => error };
    assert!(error.to_string().contains("requires fold_column or group_column"));
}
