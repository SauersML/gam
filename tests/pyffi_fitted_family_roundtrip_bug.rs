use gam::inference::model::FittedFamily;

#[test]
fn fitted_family_marginal_slope_deserializes_without_base_link_for_backward_compat() {
    let raw = r#"{\"family_kind\":\"marginal-slope\",\"likelihood\":{\"response\":\"Binomial\",\"inverse\":{\"Standard\":\"Logit\"},\"scale\":\"Fixed\"},\"frailty\":{\"kind\":\"none\"}}"#;
    let parsed: Result<FittedFamily, _> = serde_json::from_str(raw);
    assert_eq!(parsed.is_ok(), true);
}
