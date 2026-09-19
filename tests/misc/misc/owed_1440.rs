//! Owed-work regression for #1440 — all production finite differences removed;
//! the hermetic FD scanner is the genuine, allowlist-confined arbiter.
//!
//! ## What #1440 requires
//!
//! Production code must contain no finite difference (central difference is still
//! FD), except where it is theoretically impossible — or, in practice, far more
//! expensive — to do better. The authoritative check is the hermetic scanner
//! `tests/autodiff/no_production_finite_differences.rs`: if it is green, the only
//! FD left in `src/` is test-only or a tracked, justified sanction.
//!
//! ## The defect this guards (the #1440 hole)
//!
//! The scanner strips `fd-ok`/`FD-OK` audit-marker regions before searching for
//! banned FD markers. Originally it did so in EVERY file, gating only a *whole
//! file* exemption on the `SANCTIONED_FD_FILES` allowlist. That meant a fresh
//! finite difference could hide behind a per-line `// fd-ok:` marker in ANY file
//! — the allowlist's "single, tracked source of truth" promise was false. Two
//! real production finite differences were once exempted this way while the
//! allowlist claimed to be EMPTY.
//!
//! ## The fix
//!
//! The scanner now confines `fd-ok` markers to the allowlist
//! (`fd_ok_markers_are_confined_to_the_allowlist`): a non-test file that uses an
//! `fd-ok` marker but is not allowlisted is itself a violation. Since
//! d7ec08b5cf the allowlist is EMPTY (see
//! `no_production_file_is_sanctioned_for_finite_differences`).
//!
//! ## What this test guards
//!
//! It is a SOURCE-CONTRACT meta-guard (no gam dependency): it reads the scanner
//! source via `include_str!` and asserts the confinement invariant and the
//! allowlist's contents survive, so a future edit cannot silently restore the
//! per-line-marker hole or sanction a production file without this guard
//! naming it.

const SCANNER_SRC: &str = include_str!("no_production_finite_differences.rs");

/// The confinement guard test — the core #1440 invariant — must exist and must
/// flag non-allowlisted files that use `fd-ok` markers.
#[test]
fn scanner_confines_fd_ok_markers_to_the_allowlist() {
    assert!(
        SCANNER_SRC.contains("fn fd_ok_markers_are_confined_to_the_allowlist"),
        "#1440: the scanner must keep the confinement guard that forbids a \
         non-allowlisted file from using `fd-ok` markers — without it, a fresh \
         finite difference can hide behind a per-line exemption in any file"
    );
    // The guard must actually key off the allowlist and the marker tokens.
    assert!(
        SCANNER_SRC.contains("fd_ok_markers_allowed")
            && SCANNER_SRC.contains("FD-OK:")
            && SCANNER_SRC.contains("fd-ok:"),
        "#1440: the confinement guard must test allowlist membership against the \
         actual `fd-ok`/`FD-OK` marker tokens"
    );
}

/// The whole-file exemption must remain keyed on the tracked allowlist, and the
/// allowlist must still be the single source of truth (a named constant), so the
/// exemption cannot be granted ad hoc.
#[test]
fn scanner_keeps_a_tracked_allowlist_constant() {
    assert!(
        SCANNER_SRC.contains("const SANCTIONED_FD_FILES"),
        "#1440: the tracked FD allowlist constant must remain the single source of \
         truth for sanctioned finite differences"
    );
    assert!(
        SCANNER_SRC.contains("fn sanctioned_fd_allowlist_files_exist"),
        "#1440: the allowlist must keep its existence guard so it cannot rot into a \
         stale, over-broad exemption"
    );
    assert!(
        SCANNER_SRC.contains("fn every_fd_ok_marker_in_the_tree_carries_a_justification"),
        "#1440: every fd-ok marker must keep its mandatory justification guard"
    );
}

/// Extract the ENTRIES of the scanner's `SANCTIONED_FD_FILES` literal.
///
/// This test used to ask `SCANNER_SRC.contains(site)`, which is the wrong
/// witness. A path string occurs in the scanner source for several reasons that
/// are not membership: inside a NEGATIVE assertion
/// (`assert!(!fd_ok_markers_allowed(...))`), inside the allowlist's own prose,
/// or inside a comment recording that an entry was removed. Both sites this
/// test named passed that check while being on the allowlist ZERO times — and
/// one of them, `fd_audit.rs`, does not exist in the tree at all any more.
///
/// So parse the literal instead: take the text between the const's `&[` and its
/// closing `];`, drop comment-only lines, and collect the quoted entries. The
/// empty list is written on one line (`&[];`); a populated one closes on a line
/// of its own. Any other shape is refused, so a reshaped literal cannot parse
/// as empty.
fn sanctioned_allowlist_entries() -> Vec<String> {
    const HEAD: &str = "const SANCTIONED_FD_FILES: &[&str] = &[";
    let start = SCANNER_SRC.find(HEAD).unwrap_or_else(|| {
        panic!(
            "#1440: the scanner must keep the tracked FD allowlist constant \
             (`{HEAD}` not found)"
        )
    });
    let body = &SCANNER_SRC[start + HEAD.len()..];
    if body.starts_with("];") {
        return Vec::new();
    }
    let end = body
        .find("\n];")
        .unwrap_or_else(|| panic!("#1440: the FD allowlist must remain a closed literal list"));

    let mut entries: Vec<String> = Vec::new();
    for line in body[..end].lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        // Anything after a trailing `//` on an entry line is justification prose.
        let code = match trimmed.find("//") {
            Some(at) => &trimmed[..at],
            None => trimmed,
        };
        let mut rest = code;
        while let Some(open) = rest.find('"') {
            let after = &rest[open + 1..];
            let Some(close) = after.find('"') else { break };
            entries.push(after[..close].to_string());
            rest = &after[close + 1..];
        }
    }
    entries
}

/// No production file is sanctioned for finite differences. 1bc46ac508 replaced
/// the runner's FD audit (the Ridders probe in `rho_optimizer/run_plan.rs` and
/// its certificate store in `estimate/outer_eval_capture.rs`) with an analytic
/// seed probe, and d7ec08b5cf removed the last entry, `rho_optimizer/run.rs`,
/// when opt dropped the `set_finite_difference_bounds` hook it delegated. The
/// scanner's own tests prove no production FD remains outside the list.
/// Sanctioning a file again is a deliberate, reviewed change that restates this
/// test with the site and its justification.
///
/// Checked against the PARSED allowlist, not against raw scanner source text.
#[test]
fn no_production_file_is_sanctioned_for_finite_differences() {
    let entries = sanctioned_allowlist_entries();
    assert!(
        entries.is_empty(),
        "#1440: SANCTIONED_FD_FILES must stay empty unless a reviewed change \
         restates this guard with the new site and its justification. Parsed \
         allowlist: {entries:?}"
    );
}
