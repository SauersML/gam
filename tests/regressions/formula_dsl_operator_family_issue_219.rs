//! RED tests for https://github.com/SauersML/gam/issues/219
//!
//! The Pest grammar in `src/inference/formula_dsl.rs` does not include `:` as
//! an interaction operator. These tests pin the *desired* behavior of the
//! documented Wilkinson-Rogers operator family (`+`, `:`, `*`, `/`, `^`,
//! `I(...)`) and of the smooth/group constructors that are advertised in
//! the README and docs. They are expected to fail until the grammar +
//! materializer accept the colon interaction operator (and any other gaps
//! these probes surface).
//!
//! Each test is written to fail loudly and explicitly rather than be
//! marked `#[ignore]`, so the gap is visible in CI output.

use gam::inference::formula_dsl::{parse_formula, parse_formula_dsl};

#[test]
fn colon_interaction_in_dsl_string_parses() {
    let parsed = parse_formula_dsl("y ~ x1 + x2 + x1:x2")
        .expect("formula `y ~ x1 + x2 + x1:x2` should parse; `:` is the documented interaction op");
    assert_eq!(parsed.response_expr, "y");
    assert_eq!(
        parsed.rhs_terms.len(),
        3,
        "expected three top-level terms (x1, x2, x1:x2), got: {:?}",
        parsed.rhs_terms
    );
    assert!(
        parsed
            .rhs_terms
            .iter()
            .any(|t| t.replace(' ', "") == "x1:x2"),
        "expected an `x1:x2` interaction term in rhs_terms, got: {:?}",
        parsed.rhs_terms
    );
}

#[test]
fn star_expansion_in_dsl_string_parses() {
    // `a*b` is the Wilkinson-Rogers expansion shorthand for `a + b + a:b`.
    // At minimum, the grammar must accept `a*b` as a single top-level term.
    let parsed = parse_formula_dsl("y ~ x1*x2")
        .expect("formula `y ~ x1*x2` should parse; `*` is the documented crossing op");
    assert_eq!(parsed.response_expr, "y");
    assert!(
        !parsed.rhs_terms.is_empty(),
        "expected at least one RHS term, got none"
    );
}

#[test]
fn slash_nesting_in_dsl_string_parses() {
    // `a/b` is the documented nesting operator.
    let parsed = parse_formula_dsl("y ~ x1/x2")
        .expect("formula `y ~ x1/x2` should parse; `/` is the documented nesting op");
    assert_eq!(parsed.response_expr, "y");
    assert!(
        !parsed.rhs_terms.is_empty(),
        "expected at least one RHS term, got none"
    );
}

#[test]
fn caret_power_in_dsl_string_parses() {
    // `(x1 + x2)^2` is the documented power-crossing operator.
    let parsed = parse_formula_dsl("y ~ (x1 + x2)^2")
        .expect("formula `y ~ (x1 + x2)^2` should parse; `^` is documented");
    assert_eq!(parsed.response_expr, "y");
    assert!(!parsed.rhs_terms.is_empty());
}

#[test]
fn identity_wrapper_in_dsl_string_parses() {
    // `I(x1 + x2)` should pass `x1 + x2` through as a single arithmetic term.
    let parsed = parse_formula_dsl("y ~ I(x1 + x2)")
        .expect("formula `y ~ I(x1 + x2)` should parse; `I()` is documented");
    assert_eq!(parsed.rhs_terms.len(), 1, "I(...) should be one RHS term");
}

#[test]
fn full_mixed_operator_formula_from_issue_219_parses() {
    // The exact repro from the bug report.
    let formula = "y ~ x1 + x2 + x1:x2 + x1*x2 + x1/x2 + group(g)";
    let parsed = parse_formula_dsl(formula).unwrap_or_else(|e| {
        panic!("formula from issue #219 failed to parse: {e}\nformula: {formula}")
    });
    assert_eq!(parsed.response_expr, "y");
    assert!(
        parsed.rhs_terms.len() >= 6,
        "expected six top-level RHS terms, got {} ({:?})",
        parsed.rhs_terms.len(),
        parsed.rhs_terms
    );
}

#[test]
fn high_level_parser_accepts_colon_interaction() {
    // The structured parser surface (used by the workflow materializer) should
    // also accept colon interactions, not just the lightweight DSL string parse.
    match parse_formula("y ~ x1 + x2 + x1:x2") {
        Ok(_) => {}
        Err(e) => panic!(
            "high-level parse_formula() rejected `y ~ x1 + x2 + x1:x2`: {e:?}\n\
             This is the documented R-style interaction operator and must be supported."
        ),
    }
}

#[test]
fn nested_smooth_with_interaction_inside_parses() {
    // Smooth wrappers should compose with interaction terms inside.
    let formula = "y ~ s(x1) + s(x2) + x1:x2";
    let parsed = parse_formula_dsl(formula)
        .unwrap_or_else(|e| panic!("formula `{formula}` should parse: {e}"));
    assert!(
        parsed.rhs_terms.iter().any(|t| t.contains(':')),
        "expected an interaction term containing `:` in rhs_terms, got: {:?}",
        parsed.rhs_terms
    );
}

#[test]
fn tensor_and_interaction_terms_coexist() {
    // `te(...)` tensor smooths should sit alongside `:` interactions in one formula.
    let formula = "y ~ te(x1, x2) + ti(x1, x2) + x1:x2";
    let parsed = parse_formula_dsl(formula)
        .unwrap_or_else(|e| panic!("formula `{formula}` should parse: {e}"));
    assert!(parsed.rhs_terms.len() >= 3, "rhs: {:?}", parsed.rhs_terms);
}

#[test]
fn group_constructor_with_interaction_parses() {
    let formula = "y ~ x1:x2 + group(g)";
    let parsed = parse_formula_dsl(formula)
        .unwrap_or_else(|e| panic!("formula `{formula}` should parse: {e}"));
    assert_eq!(parsed.rhs_terms.len(), 2, "rhs: {:?}", parsed.rhs_terms);
}
