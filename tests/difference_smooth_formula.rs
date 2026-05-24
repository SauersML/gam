use gam::inference::formula_dsl::{ParsedTerm, parse_term};

#[test]
fn smooth_by_argument_is_preserved_in_formula_dsl() {
    let term = parse_term("s(Time, by=Group, k=8)").expect("parse by smooth");
    match term {
        ParsedTerm::Smooth { vars, options, .. } => {
            assert_eq!(vars, vec!["Time".to_string()]);
            assert_eq!(options.get("by").map(String::as_str), Some("Group"));
            assert_eq!(options.get("k").map(String::as_str), Some("8"));
        }
        other => panic!("expected smooth term, got {other:?}"),
    }
}

#[test]
fn sz_alias_is_preserved_for_term_builder() {
    let term = parse_term("s(Group, Time, bs=\"sz\")").expect("parse sz smooth");
    match term {
        ParsedTerm::Smooth { vars, options, .. } => {
            assert_eq!(vars, vec!["Group".to_string(), "Time".to_string()]);
            assert_eq!(options.get("bs").map(String::as_str), Some("sz"));
        }
        other => panic!("expected smooth term, got {other:?}"),
    }
}
