//! Tests for the approximation-ledger build-time scanner.
//!
//! The scanner itself lives in `build.rs` and runs at compile time. These
//! tests reach into a self-contained re-implementation of the same scanner
//! logic — exposed here as plain functions over an in-memory string fixture
//! — and assert that the classification rules behave as documented in
//! `src/approx_ledger.rs`. We do not invoke `build.rs` directly because
//! Cargo only runs build scripts as part of compilation, not as a library
//! callable from a test.
//!
//! The fixture deliberately contains a hand-wavy marker WITHOUT an
//! `ApproxKind` annotation, so the scan must catch it. A second fixture
//! pairs the same marker with an annotation and must pass clean.

use gam::approx_ledger::{ApproxKind, LEDGER_WINDOW, LedgerSite};

const APPROX_LEDGER_VARIANTS: &[&str] = &[
    "Exact",
    "NumericalApproximation",
    "StatisticalApproximation",
    "SurrogateObjective",
    "TemporarySolverDamping",
];

const HANDWAVY_MARKERS: &[&str] = &["bandaid", "hack", "magic", "FIXME"];

fn is_marker_in_comment(line: &str, marker_lower: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let comment_pos = lower.find("//");
    let marker_pos = lower.find(marker_lower);
    match (comment_pos, marker_pos) {
        (Some(cp), Some(mp)) => mp > cp,
        _ => false,
    }
}

fn window_has_ledger_annotation(lines: &[&str], idx: usize) -> bool {
    let lo = idx.saturating_sub(LEDGER_WINDOW);
    let hi = (idx + LEDGER_WINDOW + 1).min(lines.len());
    for line in &lines[lo..hi] {
        for variant in APPROX_LEDGER_VARIANTS {
            if line.contains(variant) {
                return true;
            }
        }
    }
    false
}

fn scan(content: &str) -> Vec<(usize, &'static str)> {
    let lines: Vec<&str> = content.lines().collect();
    let lower_lines: Vec<String> = lines.iter().map(|l| l.to_ascii_lowercase()).collect();
    let mut hits = Vec::new();
    for (idx, _) in lines.iter().enumerate() {
        for marker in HANDWAVY_MARKERS {
            let m_lower = marker.to_ascii_lowercase();
            if !lower_lines[idx].contains(&m_lower) {
                continue;
            }
            if !is_marker_in_comment(lines[idx], &m_lower) {
                continue;
            }
            if window_has_ledger_annotation(&lines, idx) {
                continue;
            }
            hits.push((idx + 1, *marker));
        }
    }
    hits
}

#[test]
fn unclassified_bandaid_is_caught() {
    // Fixture: a real-looking comment block with a bandaid marker but NO
    // ApproxKind annotation anywhere within LEDGER_WINDOW. The scan must
    // flag it; otherwise the build-time guarantee from `build.rs` is
    // broken and unclassified hand-wavy code can land on main.
    let content = "fn step() {\n\
        // The inner loop floors the weights at 1e-12 to keep the\n\
        // factorization positive-definite. This is a bandaid for the\n\
        // pure-exp link's σ → 0 singularity.\n\
        let _ = 0;\n\
    }\n";
    let hits = scan(content);
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one offender, got {:?}",
        hits
    );
    assert_eq!(hits[0].1, "bandaid");
}

#[test]
fn classified_bandaid_passes() {
    // Same fixture, now paired with an explicit ApproxKind reference in
    // the same comment block. The scan must not flag it.
    let content = "fn step() {\n\
        // The inner loop floors the weights at 1e-12 to keep the\n\
        // factorization positive-definite. This is a bandaid for the\n\
        // pure-exp link's σ → 0 singularity.\n\
        // ApproxKind: NumericalApproximation { backward_error_bound: ... }\n\
        let _ = 0;\n\
    }\n";
    let hits = scan(content);
    assert!(hits.is_empty(), "expected no offenders, got {:?}", hits);
}

#[test]
fn marker_outside_comment_is_ignored() {
    // A function or string literal that happens to contain the substring
    // "hack" must NOT trigger the scan — we only police actual prose
    // comments, not identifiers.
    let content = "fn hack_score() -> f64 { 0.0 }\n\
        let s = \"a magic value\";\n";
    let hits = scan(content);
    assert!(
        hits.is_empty(),
        "identifiers/strings must not trigger: {:?}",
        hits
    );
}

#[test]
fn annotation_outside_window_does_not_rescue() {
    // The annotation sits well past LEDGER_WINDOW lines from the marker.
    // The scan must still flag the marker — the contract requires the
    // classification to be visible in the same comment neighborhood.
    let mut content = String::from("// ApproxKind: Exact\n");
    for _ in 0..(LEDGER_WINDOW + 2) {
        content.push_str("// filler line\n");
    }
    content.push_str("// stray bandaid in here\n");
    let hits = scan(&content);
    assert_eq!(
        hits.len(),
        1,
        "expected the stray marker to be flagged: {:?}",
        hits
    );
}

#[test]
fn ledger_site_const_constructs() {
    // Sanity check: the public LedgerSite constructor and ApproxKind
    // variants compile and discriminate correctly. Not a scanner test,
    // but it ensures the module exposed by `pub mod approx_ledger` is
    // usable by callers that want compile-time site annotations.
    const SITE: LedgerSite = LedgerSite::new(
        "tests/approx_ledger_invariants.rs::ledger_site_const_constructs",
        ApproxKind::SurrogateObjective {
            description: "test stub",
        },
    );
    assert!(matches!(SITE.kind, ApproxKind::SurrogateObjective { .. }));
}
