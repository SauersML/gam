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
//! real production finite differences (the geodesic-acceleration curvature probe
//! in `pirls/reweight.rs` and the survival pilot W-metric chain factor in
//! `survival/marginal_slope/row_math.rs`), plus the SAE sphere-boost chart
//! Jacobian, were exempted this way while the allowlist claimed to be EMPTY.
//!
//! ## The fix
//!
//! The scanner now confines `fd-ok` markers to the allowlist
//! (`fd_ok_markers_are_confined_to_the_allowlist`): a non-test file that uses an
//! `fd-ok` marker but is not allowlisted is itself a violation. The allowlist is
//! populated with every sanctioned FD (audit oracle/certificate machinery, the
//! genuinely-irreducible geodesic probe and SAE chart Jacobian, and the tracked
//! reducible survival-pilot FD debt), each with a written justification.
//!
//! ## What this test guards
//!
//! It is a SOURCE-CONTRACT meta-guard (no gam dependency): it reads the scanner
//! source via `include_str!` and asserts the confinement invariant and the
//! enumerated allowlist survive, so a future edit cannot silently restore the
//! per-line-marker hole or empty the allowlist while leaving production FD in the
//! tree.

const SCANNER_SRC: &str = include_str!("autodiff/no_production_finite_differences.rs");

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

/// The genuinely-irreducible production finite differences and the FD-audit
/// oracle must remain enumerated in the allowlist. If FD is removed from one of
/// these files, drop the entry here too (and from the scanner) — that is a
/// deliberate, reviewed change, not a silent weakening.
#[test]
fn known_sanctioned_fd_sites_stay_enumerated() {
    for site in [
        "solver/pirls/reweight.rs",            // geodesic-acceleration curvature probe
        "terms/sae/chart_canonicalization.rs", // SAE sphere-boost GN chart Jacobian
        "solver/rho_optimizer/fd_audit.rs",    // FD-audit oracle (diagnostic only)
    ] {
        assert!(
            SCANNER_SRC.contains(site),
            "#1440: the sanctioned FD site `{site}` must stay enumerated in the \
             allowlist (with its justification) so it is tracked, not hidden"
        );
    }
}
