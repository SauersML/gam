//! Bug hunt (#1129): the mgcv-style `family(link)` parser rejected the canonical
//! default-link spellings `poisson(log)`, `Gamma(log)`, and `gaussian(identity)`
//! as "unknown family".
//!
//! `resolve_family` is the single inference seam every entry point (CLI, formula
//! API, `gamfit.fit` / FFI) routes through. The mgcv parenthesized form added in
//! 62ea9ad3d canonicalized `family(link)` to a flat `family-link` string and then
//! matched it against a hand-enumerated table. That table only spelled out
//! link-suffixed arms for families whose siblings historically carried a
//! *non-default* link (`binomial-probit`, `tweedie-log`, …). The three families
//! whose default link was never enumerated — `poisson`, `gamma`, `gaussian` —
//! had no `poisson-log` / `gamma-log` / `gaussian-identity` arm, so the canonical
//! default-link spellings fell through to "unknown family". mgcv users carry over
//! exactly these spellings (`poisson(log)`, `Gamma(log)`, `gaussian(identity)`).
//!
//! The fix parses `family(link)` structurally: resolve the family *head*, then
//! validate the link argument against it and apply it. This test pins the
//! behaviour from several angles a regression of the same root cause would trip:
//!
//!   1. the three originally-rejected default-link spellings now resolve to their
//!      documented `(family, link)` pairs;
//!   2. `family(default_link)` is *equivalent* to the bare `family` (a default
//!      link spelled out changes nothing but the pin);
//!   3. the link-changing forms that already worked still work (`binomial(probit)`
//!      etc.), proving the structural rewrite did not regress them;
//!   4. illegal family/link pairings (`gaussian(logit)`, `poisson(probit)`) and
//!      unknown link names (`poisson(banana)`) are rejected with precise messages,
//!      not silently mis-resolved;
//!   5. a malformed empty link (`poisson()`) still reports "unknown family".

use gam::resolve_family;
use gam::types::{InverseLink, LikelihoodSpec, ResponseColumnKind, ResponseFamily, StandardLink};
use ndarray::{Array1, array};

/// A small positive numeric response — agnostic to the family being asserted,
/// since `family=...` always wins over auto-inference in `resolve_family`.
fn y() -> Array1<f64> {
    array![0.3, 1.2, 2.5, 4.0, 0.7, 3.1]
}

fn resolve(name: &str) -> Result<LikelihoodSpec, String> {
    resolve_family(
        Some(name),
        None,
        None,
        y().view(),
        ResponseColumnKind::Numeric,
        "y",
    )
}

/// The three canonical default-link spellings the bug rejected must resolve to
/// their documented `(family, link)` pairs.
#[test]
fn mgcv_default_link_parenthesized_spellings_resolve() {
    let poisson = resolve("poisson(log)").expect("poisson(log) must resolve, not 'unknown family'");
    assert_eq!(poisson.response, ResponseFamily::Poisson);
    assert_eq!(poisson.link, InverseLink::Standard(StandardLink::Log));

    // Capital G as R writes it (`Gamma()`); the resolver lowercases.
    let gamma = resolve("Gamma(log)").expect("Gamma(log) must resolve, not 'unknown family'");
    assert_eq!(gamma.response, ResponseFamily::Gamma);
    assert_eq!(gamma.link, InverseLink::Standard(StandardLink::Log));

    let gaussian =
        resolve("gaussian(identity)").expect("gaussian(identity) must resolve, not 'unknown family'");
    assert_eq!(gaussian.response, ResponseFamily::Gaussian);
    assert_eq!(gaussian.link, InverseLink::Standard(StandardLink::Identity));
}

/// `family(default_link)` must be exactly the bare `family` — a spelled-out
/// default link changes the resolved family/link, nothing else.
#[test]
fn parenthesized_default_link_equals_bare_family() {
    for (paren, bare) in [
        ("poisson(log)", "poisson"),
        ("gamma(log)", "gamma"),
        ("gaussian(identity)", "gaussian"),
        ("binomial(logit)", "binomial"),
        ("beta(logit)", "beta"),
        ("tweedie(log)", "tweedie"),
        ("negative_binomial(log)", "negative_binomial"),
    ] {
        let p = resolve(paren).unwrap_or_else(|e| panic!("{paren} must resolve: {e}"));
        let b = resolve(bare).unwrap_or_else(|e| panic!("{bare} must resolve: {e}"));
        assert_eq!(
            p.response, b.response,
            "{paren} and {bare} must resolve to the same response family"
        );
        assert_eq!(
            p.link.link_function(),
            b.link.link_function(),
            "{paren} and {bare} must resolve to the same link"
        );
    }
}

/// The link-changing parenthesized forms that already worked must keep working —
/// the structural rewrite must not regress them.
#[test]
fn link_changing_parenthesized_forms_still_resolve() {
    let probit = resolve("binomial(probit)").expect("binomial(probit) must resolve");
    assert_eq!(probit.response, ResponseFamily::Binomial);
    assert_eq!(probit.link, InverseLink::Standard(StandardLink::Probit));

    let cloglog = resolve("binomial(cloglog)").expect("binomial(cloglog) must resolve");
    assert_eq!(cloglog.link, InverseLink::Standard(StandardLink::CLogLog));

    // Mixed case both sides of the paren, as mgcv users sometimes write.
    let mixed = resolve("Binomial(Probit)").expect("Binomial(Probit) must resolve");
    assert_eq!(mixed.link, InverseLink::Standard(StandardLink::Probit));
}

/// An illegal family/link pairing must be rejected with a precise message —
/// never silently mis-resolved to the default link.
#[test]
fn illegal_family_link_pairings_are_rejected() {
    for bad in ["gaussian(logit)", "poisson(probit)", "gamma(identity)"] {
        let err = resolve(bad).expect_err(&format!("{bad} is an illegal pairing and must error"));
        assert!(
            err.contains("not supported for family"),
            "{bad} must report an illegal-link message, got: {err}"
        );
    }
}

/// An unknown link name must be rejected as an unknown link, not silently
/// accepted nor reported as an unknown family.
#[test]
fn unknown_link_name_is_rejected() {
    let err = resolve("poisson(banana)").expect_err("poisson(banana) must error");
    assert!(
        err.contains("unknown link") && err.contains("banana"),
        "poisson(banana) must report an unknown-link message naming the link, got: {err}"
    );
}

/// Malformed empty parens fall through to the standard "unknown family" error.
#[test]
fn malformed_empty_link_reports_unknown_family() {
    let err = resolve("poisson()").expect_err("poisson() must error");
    assert!(
        err.contains("unknown family"),
        "poisson() must fall through to 'unknown family', got: {err}"
    );
}
