//! #1254 regression guard: the `nested_prefix` analytic-penalty descriptor must
//! be DISPATCHED (built into a real `AnalyticPenaltyKind::NestedPrefix`), not
//! silently dropped, by the JSON descriptor parser.
//!
//! HISTORY (#1254 re-fix): the original guard was a pair of source-text greps —
//! it asserted that `crates/gam-pyffi/src/lib.rs` *contained the substring*
//! `"nested_prefix"`. After the analytic-penalty JSON dispatch was unified into
//! `gam_models::fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors`
//! (pyffi no longer parses these descriptors itself), the only remaining
//! occurrence of `"nested_prefix"` in pyffi's `lib.rs` is a DOC COMMENT. So the
//! grep passed purely off prose — the real dispatch arm could be deleted and the
//! test would stay green. That is a vacuous guard.
//!
//! This version exercises the real dispatch path: it builds a `nested_prefix`
//! descriptor through the production parser and asserts the resulting registry
//! actually contains a `NestedPrefix` penalty. It also pins the negative
//! (unknown kind is rejected, not silently ignored) so the guard catches a
//! regression in BOTH directions.

use gam::solver::fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors;
use serde_json::json;

/// A latent block named `z` with shape (n=4, d=3), matching the `target` the
/// descriptors below reference. The parser requires at least one latent block.
fn latents() -> serde_json::Value {
    json!({
        "z": { "name": "z", "n": 4, "d": 3 },
    })
}

#[test]
fn nested_prefix_descriptor_is_dispatched_not_dropped() {
    let latents = latents();
    let penalties = json!([
        {
            "kind": "nested_prefix",
            "target": "z",
            "prefix_sizes": [1, 2],
            "shell_weights": [1.0, 0.5],
            "tier": "psi"
        }
    ]);

    let registry = build_analytic_penalty_registry_from_descriptors(Some(&latents), Some(&penalties))
        .expect("nested_prefix descriptor must build a registry");

    // The descriptor must produce exactly one penalty, tagged `nested_prefix`.
    // A silently-dropped kind would yield an empty registry (the #1254 symptom);
    // a mis-dispatched kind would carry the wrong tag.
    let tags: Vec<&str> = registry
        .penalties
        .iter()
        .map(|penalty| penalty.kind_tag())
        .collect();
    assert_eq!(
        tags,
        vec!["nested_prefix"],
        "BUG (#1254): nested_prefix descriptor was not dispatched to a \
         NestedPrefix penalty (got tags {tags:?})"
    );
}

#[test]
fn nested_prefix_accepts_hyphenated_and_uppercase_kind() {
    // The parser lowercases and replaces '-' with '_', so these alias spellings
    // must dispatch identically — pinning the normalization the dispatch relies on.
    let latents = latents();
    for kind in ["Nested-Prefix", "NESTED_PREFIX", "nestedprefix"] {
        let penalties = json!([
            {
                "kind": kind,
                "target": "z",
                "prefix_sizes": [1, 2],
                "shell_weights": [1.0, 0.5],
                "tier": "psi"
            }
        ]);
        let registry =
            build_analytic_penalty_registry_from_descriptors(Some(&latents), Some(&penalties))
                .unwrap_or_else(|err| panic!("kind alias {kind:?} must dispatch: {err}"));
        let tags: Vec<&str> = registry
            .penalties
            .iter()
            .map(|penalty| penalty.kind_tag())
            .collect();
        assert_eq!(
            tags,
            vec!["nested_prefix"],
            "kind alias {kind:?} must dispatch to nested_prefix (got {tags:?})"
        );
    }
}

#[test]
fn unknown_analytic_penalty_kind_is_rejected_not_silently_dropped() {
    // Negative control: an unrecognized kind must ERROR, not be silently skipped.
    // This is what makes the positive test above meaningful — it proves the
    // parser genuinely discriminates kinds rather than ignoring everything.
    let latents = latents();
    let penalties = json!([
        { "kind": "definitely_not_a_real_penalty_kind", "target": "z" }
    ]);
    let result = build_analytic_penalty_registry_from_descriptors(Some(&latents), Some(&penalties));
    assert!(
        result.is_err(),
        "an unknown analytic-penalty kind must be rejected, not silently dropped"
    );
}
