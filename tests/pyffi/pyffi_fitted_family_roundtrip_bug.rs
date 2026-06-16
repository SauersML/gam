use gam::inference::model::FittedFamily;

#[test]
fn fitted_family_marginal_slope_deserializes_without_base_link_for_backward_compat() {
    // Saved fits produced before `base_link` was added to FittedFamily::MarginalSlope
    // omit that field entirely. Deserialization must still succeed so older artifacts
    // remain loadable; the field defaults to `None`.
    let raw = concat!(
        "{",
        "\"family_kind\": \"marginal-slope\",",
        "\"likelihood\": {",
        "\"response\": \"Binomial\",",
        "\"link\": { \"Standard\": \"Logit\" }",
        "},",
        "\"frailty\": { \"frailty_kind\": \"none\" }",
        "}",
    );
    let parsed: Result<FittedFamily, _> = serde_json::from_str(raw);
    assert!(
        parsed.is_ok(),
        "expected backward-compat deserialization to succeed, got {:?}",
        parsed.err()
    );
    match parsed.unwrap() {
        FittedFamily::MarginalSlope { base_link, .. } => {
            assert!(
                base_link.is_none(),
                "missing base_link must default to None"
            );
        }
        other => panic!("expected FittedFamily::MarginalSlope, got {other:?}"),
    }
}
