use gam::inference::formula_dsl::{ParsedTerm, parse_formula};

fn smooth_options(formula: &str) -> (Vec<String>, std::collections::BTreeMap<String, String>) {
    let parsed = parse_formula(formula).expect(formula);
    parsed
        .terms
        .into_iter()
        .find_map(|term| match term {
            ParsedTerm::Smooth { vars, options, .. } => Some((vars, options)),
            _ => None,
        })
        .expect("smooth term")
}

#[test]
fn parses_factor_smooth_bs_forms_and_aliases() {
    for (formula, expected_bs, expected_vars) in [
        ("y ~ s(x, fac, bs=\"fs\")", "fs", vec!["x", "fac"]),
        ("y ~ fs(x, fac)", "fs", vec!["x", "fac"]),
        ("y ~ s(x, fac, bs=fs, k=10)", "fs", vec!["x", "fac"]),
        ("y ~ s(x, fac, bs=\"sz\")", "sz", vec!["x", "fac"]),
        ("y ~ sz(x, fac)", "sz", vec!["x", "fac"]),
    ] {
        let (vars, options) = smooth_options(formula);
        assert_eq!(vars, expected_vars);
        assert_eq!(options.get("bs").map(String::as_str), Some(expected_bs));
    }
}

#[test]
fn parses_by_smooth_options() {
    for (formula, expected_by, expected_k) in [
        ("y ~ s(x, by=fac)", "fac", None),
        ("y ~ s(x, by=group, k=8)", "group", Some("8")),
    ] {
        let (vars, options) = smooth_options(formula);
        assert_eq!(vars, vec!["x"]);
        assert_eq!(options.get("by").map(String::as_str), Some(expected_by));
        assert_eq!(options.get("k").map(String::as_str), expected_k);
    }
}
