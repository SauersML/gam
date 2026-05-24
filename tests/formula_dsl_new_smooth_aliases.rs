//! Parser-level tests for the new smooth-family formula aliases.

use gam::inference::formula_dsl::{ParsedTerm, parse_formula};

fn smooth_options(
    parsed: &gam::inference::formula_dsl::ParsedFormula,
) -> &std::collections::BTreeMap<String, String> {
    for term in &parsed.terms {
        if let ParsedTerm::Smooth { options, .. } = term {
            return options;
        }
    }
    panic!("no smooth term in parsed formula");
}

#[test]
fn sphere_alias_names_all_parse_to_sphere_type() {
    for name in ["sphere", "sos", "spherical"] {
        let f = format!("y ~ {name}(lat, lon, k=10)");
        let parsed = parse_formula(&f).unwrap_or_else(|e| panic!("`{f}` parse failed: {e}"));
        let opts = smooth_options(&parsed);
        let ty = opts.get("type").map(String::as_str).unwrap_or("");
        assert_eq!(ty, "sphere", "alias `{name}` parsed to type=`{ty}`");
    }
}

#[test]
fn periodic_alias_names_all_parse_to_cyclic_type() {
    for name in ["cyclic", "periodic", "cc", "cp"] {
        let f = format!("y ~ {name}(t, k=10, period_start=0, period_end=6.283185307179586)");
        let parsed = parse_formula(&f).unwrap_or_else(|e| panic!("`{f}` parse failed: {e}"));
        let opts = smooth_options(&parsed);
        let ty = opts.get("type").map(String::as_str).unwrap_or("");
        assert_eq!(
            ty, "cyclic",
            "periodic alias `{name}` parsed to type=`{ty}`"
        );
    }
}

#[test]
fn sphere_method_aliases_parse_consistently() {
    for raw_method in [
        "wahba",
        "kernel",
        "harmonic",
        "harmonics",
        "spherical_harmonics",
        "spherical-harmonics",
        "sh",
    ] {
        let f = format!("y ~ sphere(lat, lon, k=10, method={raw_method})");
        let parsed = parse_formula(&f).unwrap_or_else(|e| panic!("`{f}` parse failed: {e}"));
        let opts = smooth_options(&parsed);
        assert!(
            opts.get("method").is_some(),
            "method= dropped at parse for `{f}`"
        );
    }
}

#[test]
fn bc_option_with_list_value_parses() {
    let f = "y ~ te(theta, h, bc=['periodic', 'natural'], period=[2*pi, None], k=5)";
    let parsed = parse_formula(f).unwrap_or_else(|e| panic!("`{f}` parse failed: {e}"));
    let opts = smooth_options(&parsed);
    let bc = opts.get("bc").expect("bc option present");
    assert!(
        bc.contains("periodic") && bc.contains("natural"),
        "bc list value mangled at parse: {bc}",
    );
}

#[test]
fn matern_nu_decimal_and_fraction_both_parse() {
    for nu in ["1/2", "0.5", "3/2", "1.5", "5/2", "2.5", "half"] {
        let f = format!("y ~ matern(x, nu={nu})");
        let _ = parse_formula(&f).unwrap_or_else(|e| panic!("`{f}` parse failed: {e}"));
    }
}

#[test]
fn periodic_period_decimal_and_2pi_expressions_parse() {
    for spec in [
        "period=6.283185307179586",
        "period=2*pi",
        "period=1.0",
        "period=100",
    ] {
        let f = format!("y ~ s(t, periodic=true, {spec})");
        let _ = parse_formula(&f).unwrap_or_else(|e| panic!("`{f}` parse failed: {e}"));
    }
}

#[test]
fn difference_smooth_options_parse_by_and_sz() {
    let parsed = parse_formula("y ~ s(x, by=group, k=8) + s(group, x, bs='sz')")
        .expect("difference-smooth formula parses");
    let mut saw_by = false;
    let mut saw_sz = false;
    for term in &parsed.terms {
        if let ParsedTerm::Smooth { options, vars, .. } = term {
            if options.get("by").map(String::as_str) == Some("group") {
                saw_by = true;
            }
            if options.get("bs").map(String::as_str) == Some("sz") {
                assert_eq!(vars, &vec!["group".to_string(), "x".to_string()]);
                saw_sz = true;
            }
        }
    }
    assert!(saw_by, "by= option was not retained");
    assert!(saw_sz, "bs=sz option was not retained");
}
