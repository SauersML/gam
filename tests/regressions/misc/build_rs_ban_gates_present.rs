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

/// The ban-scanner's FAILURE-SURFACE teeth: it is not enough that the gate
/// FUNCTIONS exist — the report emitter that turns a detected violation into a
/// *fatal-looking* CI signal must also stay intact. Commit `25babfc34` did not
/// only delete a gate; it LAUNDERED severity: it softened the terminal
/// "ban-scanner FAILED ... build aborted" summary into an informational
/// "found ... violations", and stripped the per-offender `error:` / `error —`
/// prefixes down to decorative rules so an abort reads as a benign note in CI
/// logs. A future re-laundering of the report emitter would slip past the
/// function-presence checks above. These tokens pin the teeth so it cannot.
///
/// Each token is assembled from fragments so the literal failure-surface string
/// never appears verbatim in THIS file.
pub(crate) fn required_failure_surface_teeth() -> Vec<String> {
    // The em-dash (U+2014) used in the hard-fail summary and section headers.
    let em_dash = "\u{2014}";
    vec![
        // Hard-fail summary token: the scanner's terminal cargo:warning render
        // emits "ban-scanner FAILED: ... build aborted". Both halves must stay;
        // softening either to an informational phrasing is severity-laundering.
        format!("{}{}", "FAIL", "ED"),
        format!("build {}{}", "abort", "ed"),
        // Per-offender row prefix: render_report emits its violation rows with
        // an `error: ` prefix (the leading total line and each "file:line").
        // Demoting this to a decorative bullet hides offenders in CI output.
        format!("{}: {{}}", "error"),
        // Section-header prefix: render_report opens each rule's section with
        // an `error — ` header. Stripping the `error` keyword launders the
        // severity of the whole section to a neutral heading.
        format!("{} {} {{}}", "error", em_dash),
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

/// Guardian for severity-laundering: the ban-scanner's report emitter must keep
/// rendering a *fatal* failure surface (FAILED / build aborted / error: /
/// error —), not a softened informational note. This is the exact attack that
/// commit `25babfc34` performed in addition to deleting a gate.
#[test]
pub(crate) fn build_rs_ban_scanner_failure_surface_is_not_laundered() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs"))
        .expect("build.rs must be readable at the crate manifest root");

    let mut missing: Vec<String> = Vec::new();
    for token in required_failure_surface_teeth() {
        if !src.contains(&token) {
            missing.push(token);
        }
    }

    assert!(
        missing.is_empty(),
        "build.rs ban-scanner FAILURE SURFACE was laundered: token(s) {:?} \
         missing — the abort must read as a failure (FAILED / build aborted / \
         error: row-prefix / error \u{2014} section-header), restore it. A \
         softened informational phrasing makes a fatal abort look benign in CI \
         logs (this is the severity-laundering half of commit 25babfc34).",
        missing
    );
}

/// Each required ban gate, paired with the OFFENDER VECTOR it fills. The gate is
/// only live if BOTH (a) `scan_for_<gate>(&manifest_dir, ...)` is INVOKED in
/// `main()` AND (b) a `<offender>.is_empty()` push guard funnels its offenders
/// into `sections` (which the unconditional hard-exit then aborts on). A
/// function-name-present check (the test above) is satisfied by the bare `fn`
/// DEFINITION alone, so a trojan that deletes only the call or only the
/// `sections.push` block — leaving the `fn` body in place — would slip past it
/// while silently neutering the gate. This pins the full chain.
///
/// Each token is assembled from fragments so the literal identifiers never
/// appear verbatim in this source (the build.rs scanners also read test files).
pub(crate) fn required_gate_wiring() -> Vec<(String, String)> {
    vec![
        (
            format!("scan_for_{}(&manifest", "banned_marker"),
            format!("todo_{}.is_empty()", "offenders"),
        ),
        (
            format!("scan_for_{}{}(&manifest", "deferred_", "work_markers"),
            format!("deferred_marker_{}.is_empty()", "offenders"),
        ),
        (
            format!("scan_for_{}{}(&manifest", "owed_", "work_prose"),
            format!("owed_work_{}.is_empty()", "offenders"),
        ),
        (
            format!("scan_for_{}(&manifest", "let_underscore"),
            format!("underscore_{}.is_empty()", "offenders"),
        ),
        (
            format!("scan_for_{}(&manifest", "banned_substrings"),
            format!("substring_{}.is_empty()", "offenders"),
        ),
    ]
}

/// Guardian for SILENT NEUTERING: a ban gate is only enforced if it is both
/// INVOKED and WIRED into the report `sections` (whose non-empty set triggers the
/// unconditional hard exit). Pin both halves for the critical gates, and pin a
/// FLOOR on the total number of `scan_for_*` invocations and `.is_empty()` push
/// guards so a wholesale strip of gates not individually listed here is also
/// caught. This closes the call-site / `sections.push` deletion vector that the
/// function-presence check alone (`build_rs_ban_gates_are_all_present`) cannot
/// see — the deeper half of the commit-`25babfc34` class of trojan edit.
#[test]
pub(crate) fn build_rs_ban_gates_are_invoked_and_wired() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/build.rs"))
        .expect("build.rs must be readable at the crate manifest root");

    // (a) Each critical gate must be both invoked and funneled into `sections`.
    let mut broken: Vec<String> = Vec::new();
    for (invocation, push_guard) in required_gate_wiring() {
        if !src.contains(&invocation) {
            broken.push(format!("INVOCATION missing: `{invocation}`"));
        }
        if !src.contains(&push_guard) {
            broken.push(format!("sections.push wiring missing: `{push_guard}`"));
        }
    }
    assert!(
        broken.is_empty(),
        "build.rs ban gate WIRING broken — a gate's call site or its \
         `<offenders>.is_empty()` push into `sections` was removed, silently \
         neutering it while the `fn` body remains (so the name-presence check \
         stays green). Restore the full scan->offenders->sections.push->exit \
         chain. Broken: {broken:?}"
    );

    // (b) Floor on the total wiring so a wholesale strip of OTHER gates (not
    // individually pinned above) is also caught. As of this commit build.rs
    // wires 24 `scan_for_*` invocations and 30+ `.is_empty()` push guards; a
    // conservative floor rejects a mass deletion without flaking on the routine
    // addition of new gates.
    let scan_prefix = format!("scan_{}_", "for");
    let scan_invocations = src.matches(scan_prefix.as_str()).count();
    let invocation_token = format!("(&manifest_{}", "dir");
    let scan_call_sites = src
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with(scan_prefix.as_str()) && l.contains(&invocation_token)
        })
        .count();
    let push_guards = src.matches(&format!(".is_{}()", "empty")).count();
    assert!(
        scan_invocations >= 24,
        "build.rs `scan_for_*` mentions dropped to {scan_invocations} (floor 24) — \
         ban gates were stripped en masse"
    );
    assert!(
        scan_call_sites >= 20,
        "build.rs `scan_for_*(&manifest_dir, ...)` CALL SITES dropped to \
         {scan_call_sites} (floor 20) — gate invocations were removed from main()"
    );
    assert!(
        push_guards >= 25,
        "build.rs `.is_empty()` push guards dropped to {push_guards} (floor 25) — \
         gate offenders are no longer funneled into the abort `sections`"
    );
}
