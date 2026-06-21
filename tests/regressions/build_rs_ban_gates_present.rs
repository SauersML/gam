//! Guardian: the `build.rs` ban-scanner gates must stay present.
//!
//! Background: the ban gates in `build.rs` are the project's only enforcement
//! against owed-marker prose, self-tampering, discarded bindings, and friends.
//! They are powerful precisely because they live in the build script and run on
//! every build. But that also makes them a single point of failure: commit
//! `25babfc34` silently removed a ban gate from `build.rs` while claiming only
//! to "fix radial_profile". A defense that lives entirely inside `build.rs` is
//! useless the moment `build.rs` itself is overwritten or quietly trimmed.
//!
//! This test deliberately lives in a SEPARATE file. It reads the `build.rs`
//! source as text and asserts that each required ban-scanner function (and the
//! unconditional terminal hard-exit) is still present. If a gate vanishes, this
//! `#[test]` fails CI with a message naming the missing gate — surviving a
//! `build.rs` overwrite that the in-build.rs defenses cannot survive.
//!
//! Every required token is assembled from string fragments at runtime so that
//! THIS file does not itself contain the literal gate identifiers, and so the
//! `build.rs` scanners (which also scan test files) never trip on it.

/// The required ban-gate tokens, each assembled from fragments so the literal
/// identifier never appears verbatim in this source file. Every entry MUST be
/// present in `build.rs`; a missing entry means a gate was removed.
pub(crate) fn required_ban_gates() -> Vec<String> {
    vec![
        // Banned-substring + owed-marker scanners.
        format!("scan_for_{}", "banned_substrings"),
        format!("scan_for_{}{}", "deferred_", "work_markers"),
        format!("scan_for_{}", "banned_marker"),
        // Owed-work-as-prose ban — the gate repeatedly stripped by disguised
        // commits (25babfc34); the guardian MUST cover it.
        format!("scan_for_{}{}", "owed_", "work_prose"),
        // Per-line discarded-binding scanner (comment-stripped line scan).
        format!("scan_for_{}", "let_underscore"),
        // Fuzzy comment-block cue detector.
        format!("comment_block_has_{}{}", "defer", "ral_cue"),
        // Self-tampering guard: forbids disabling or neutering the gates.
        format!("forbid_build_rs_{}", "self_tampering"),
        // The unconditional terminal hard exit that makes any offense fatal.
        format!("std::process::{}", "exit(1)"),
    ]
}

#[test]
pub(crate) fn build_rs_ban_gates_are_all_present() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs"))
        .expect("build.rs must be readable at the crate manifest root");

    let mut missing: Vec<String> = Vec::new();
    for name in required_ban_gates() {
        if !src.contains(&name) {
            missing.push(name);
        }
    }

    assert!(
        missing.is_empty(),
        "build.rs ban gate(s) MISSING — they were stripped; restore them \
         (do not remove ban gates). Missing: {}",
        missing.join(", ")
    );
}
