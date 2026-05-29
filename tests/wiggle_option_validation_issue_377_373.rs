//! Regression tests for https://github.com/SauersML/gam/issues/377 and
//! https://github.com/SauersML/gam/issues/373.
//!
//! `linkwiggle(...)` and `timewiggle(...)` share `parse_linkwiggle_formulaspec`.
//! Two defects in its option-validation block:
//!
//!   * #377 — `degree`/`internal_knots` were read with the lossy `option_usize`
//!     (and `double_penalty` with the lossy `option_bool`), so a present-but-
//!     unparseable value (`-3`, `abc`, `6.5`, `ture`) was silently discarded
//!     and the built-in default substituted as if the option were absent. A
//!     user typo or out-of-range value thus changed the fitted model with no
//!     diagnostic. Compare: `internal_knots=0` was correctly rejected.
//!   * #373 — the `degree`/`internal_knots` semantic-validation messages
//!     hardcoded the literal `"linkwiggle()"`, so a bad `timewiggle(...)`
//!     option was reported as a `linkwiggle()` error, naming the wrong term.
//!
//! These tests pin the corrected behavior end-to-end through the public
//! parser API.

use gam::inference::formula_dsl::parse_linkwiggle_formulaspec;
use std::collections::BTreeMap;

fn opts(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

// ---------------------------------------------------------------------------
// #377 — strict integer parsing
// ---------------------------------------------------------------------------

#[test]
fn invalid_internal_knots_error_message_is_actionable() {
    for bad in ["-3", "abc", "6.5"] {
        let raw = format!("linkwiggle(internal_knots={bad})");
        let err = parse_linkwiggle_formulaspec(&opts(&[("internal_knots", bad)]), &raw)
            .expect_err("invalid internal_knots must be rejected");
        assert!(
            err.contains("internal_knots") && err.contains("not a non-negative integer"),
            "expected a strict-integer error for internal_knots={bad}, got: {err}"
        );
    }
}

#[test]
fn invalid_degree_values_are_rejected() {
    for bad in ["-3", "abc", "6.5"] {
        let raw = format!("linkwiggle(degree={bad})");
        let err = parse_linkwiggle_formulaspec(&opts(&[("degree", bad)]), &raw)
            .expect_err("invalid degree must be rejected");
        assert!(
            err.contains("degree") && err.contains("not a non-negative integer"),
            "expected a strict-integer error for degree={bad}, got: {err}"
        );
    }
}

#[test]
fn invalid_double_penalty_value_is_rejected() {
    let err = parse_linkwiggle_formulaspec(
        &opts(&[("internal_knots", "3"), ("double_penalty", "ture")]),
        "linkwiggle(double_penalty=ture)",
    )
    .expect_err("invalid double_penalty must be rejected");
    assert!(
        err.contains("double_penalty") && err.contains("not a boolean"),
        "expected a strict-boolean error, got: {err}"
    );
}

#[test]
fn semantically_invalid_zero_internal_knots_still_rejected() {
    // `0` parses as a usize but is semantically invalid; the strict switch
    // must not regress this existing rejection.
    let err = parse_linkwiggle_formulaspec(
        &opts(&[("internal_knots", "0")]),
        "linkwiggle(internal_knots=0)",
    )
    .expect_err("internal_knots=0 must still be rejected");
    assert!(err.contains("requires internal_knots > 0"), "got: {err}");
}

#[test]
fn valid_options_parse_and_round_trip() {
    let spec = parse_linkwiggle_formulaspec(
        &opts(&[
            ("degree", "3"),
            ("internal_knots", "5"),
            ("double_penalty", "false"),
        ]),
        "linkwiggle(degree=3, internal_knots=5, double_penalty=false)",
    )
    .expect("valid options must parse");
    assert_eq!(spec.degree, 3);
    assert_eq!(spec.num_internal_knots, 5);
    assert!(!spec.double_penalty);
}

// ---------------------------------------------------------------------------
// #373 — error messages name the actual term, not always "linkwiggle()"
// ---------------------------------------------------------------------------

#[test]
fn timewiggle_internal_knots_error_names_timewiggle() {
    let err = parse_linkwiggle_formulaspec(
        &opts(&[("internal_knots", "0")]),
        "timewiggle(internal_knots=0)",
    )
    .expect_err("timewiggle(internal_knots=0) must be rejected");
    assert!(
        err.contains("timewiggle() requires internal_knots > 0"),
        "error must name timewiggle(): {err}"
    );
    assert!(
        !err.contains("linkwiggle()"),
        "error must not mislabel timewiggle as linkwiggle(): {err}"
    );
}

#[test]
fn timewiggle_degree_error_names_timewiggle() {
    let err = parse_linkwiggle_formulaspec(
        &opts(&[("degree", "0"), ("internal_knots", "3")]),
        "timewiggle(degree=0)",
    )
    .expect_err("timewiggle(degree=0) must be rejected");
    assert!(
        err.contains("timewiggle() requires degree >= 1"),
        "error must name timewiggle(): {err}"
    );
    assert!(
        !err.contains("linkwiggle()"),
        "error must not mislabel timewiggle as linkwiggle(): {err}"
    );
}

#[test]
fn linkwiggle_term_still_named_linkwiggle() {
    let err = parse_linkwiggle_formulaspec(
        &opts(&[("internal_knots", "0")]),
        "linkwiggle(internal_knots=0)",
    )
    .expect_err("linkwiggle(internal_knots=0) must be rejected");
    assert!(
        err.contains("linkwiggle() requires internal_knots > 0"),
        "error must name linkwiggle(): {err}"
    );
}
