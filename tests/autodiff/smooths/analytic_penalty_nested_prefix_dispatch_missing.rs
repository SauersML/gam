use std::fs;

#[test]
fn analytic_penalty_nested_prefix_kind_is_missing_from_pyffi_json_dispatch() {
    let pyffi = fs::read_to_string("crates/gam-pyffi/src/lib.rs")
        .expect("BUG: could not read crates/gam-pyffi/src/lib.rs");
    let registry = fs::read_to_string("crates/gam-terms/src/analytic_penalties/manifest.rs")
        .expect("BUG: could not read crates/gam-terms/src/analytic_penalties/manifest.rs");

    assert!(
        registry.contains("register!(NestedPrefix, NestedPrefixPenalty);"),
        "BUG: NestedPrefix is not registered in analytic_penalty_registry macro"
    );

    assert!(
        pyffi.contains("\"nested_prefix\"") || pyffi.contains("\"nestedprefix\""),
        "BUG: pyffi build_analytic_penalty_registry_from_json does not dispatch NestedPrefix even though AnalyticPenaltyKind includes it"
    );
}
