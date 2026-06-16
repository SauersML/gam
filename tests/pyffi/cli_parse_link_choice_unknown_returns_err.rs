use gam::inference::formula_dsl::parse_link_choice;

#[test]
fn cli_parse_link_choice_unknown_returns_err() {
    let result = parse_link_choice(Some("not-a-real-link"), false);
    assert!(
        result.is_err(),
        "Expected unknown link string to return an error instead of parsing successfully"
    );
}
