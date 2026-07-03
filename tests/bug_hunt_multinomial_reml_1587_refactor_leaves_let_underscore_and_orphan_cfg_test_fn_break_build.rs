//! Bug hunt: the #1587 "reference-symmetric centered multinomial penalty"
//! refactor left dead scaffolding that trips the `build.rs` hygiene scanner, so
//! the workspace no longer builds from a clean checkout at HEAD.
//!
//! The two most recent multinomial commits —
//!   * `97a88873f` fix(#1587): wire reference-symmetric centered multinomial
//!     penalty through outer REML
//!   * `725a28656` fix(#1587): un-ignore the multinomial reference-class
//!     invariance oracle
//! moved every smooth term's penalty onto the joint `M⊗S_t` carrier and stopped
//! attaching the old per-(class,term) `I⊗S_t` block penalty. That left two
//! orphans, both flagged by the always-fatal `build.rs` ban scanner:
//!
//!   1. `crates/gam-models/src/multinomial_reml.rs:407` — the per-class block
//!      builder no longer reads `n_terms`, and the dangling binding was silenced
//!      with `let _ = n_terms;`. `build.rs::scan_for_let_underscore` bans every
//!      `let _` discard in production source ("let _ binding (bare `_`, `_name`,
//!      or all-underscore tuple pattern)").
//!
//!   2. `crates/gam-custom-family/src/penalty_labels.rs:73-74` — the
//!      `#[cfg(test)]`-gated `pub(crate) fn penalty_label_layout(...)` shim (a
//!      zero-`joint_specs` wrapper around `penalty_label_layout_with_joint`) is
//!      now called from nowhere. It trips TWO rules at once: `#[cfg(test)] on a
//!      src/ item` (the scanner only exempts a private `mod tests` /
//!      `mod test_support` / `mod tests_*` / `mod *_tests`, never a free fn) and
//!      `pub(crate)/pub(super) item in src/ with ZERO consumers anywhere`.
//!
//! On any state that forces `build.rs` to re-run (a clean `cargo build`, `cargo
//! test`, a fresh `--release` build, or the `maturin` wheel build) the scanner
//! reports 3 violations across 3 rules and `std::process::exit(1)`:
//!
//!     error: crates/gam-models/src/multinomial_reml.rs:407: let _ = n_terms;
//!     error: crates/gam-custom-family/src/penalty_labels.rs:73: [pub(crate) fn
//!            penalty_label_layout(] #[cfg(test)]
//!     error: crates/gam-custom-family/src/penalty_labels.rs:74:
//!            [penalty_label_layout] pub(crate) fn penalty_label_layout(
//!     cargo:warning=ban-scanner FAILED: 3 violation(s) across 3 rule(s)
//!
//! so `cargo build` / `cargo test` exit 1 and the `gamfit` wheel cannot be built.
//! (A *cached* `target/` can hide it: while the build-script fingerprint is still
//! valid the script is skipped and a stale incremental build appears to succeed —
//! which is how this rode onto `main`. A fresh build, the wheel build, and CI all
//! re-run it and fail.)
//!
//! While the bug is present the crate cannot build, so this integration-test
//! target cannot even be produced — `cargo test` fails. Once the #1587 fallout is
//! cleaned (drop the `let _ = n_terms;` discard, and delete the orphaned
//! `penalty_label_layout` shim), the workspace builds again and the checks below
//! — which re-implement the same two detections `build.rs` uses, over the two
//! offending production files — find nothing and the test passes, with no further
//! edits.

use std::path::{Path, PathBuf};

/// Strip a `//` line comment so a *mention* of a banned construct in comment
/// prose is not counted, matching `build.rs`'s comment masking.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Faithful-enough port of `build.rs::stripped_line_has_let_underscore` for the
/// scalar forms that occur here: `let _ = ...`, `let _: T = ...`,
/// `let _name = ...`, `let mut _name = ...`. (Tuple-discard patterns are not
/// needed for this file and are intentionally omitted.)
fn line_has_let_underscore(code: &str) -> bool {
    let bytes = code.as_bytes();
    let mut i = 0usize;
    while i + 3 < bytes.len() {
        if &bytes[i..i + 3] == b"let"
            && (i == 0 || !is_ident_byte(bytes[i - 1]))
            && bytes[i + 3].is_ascii_whitespace()
        {
            let mut j = i + 3;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            // optional `mut`
            if j + 3 <= bytes.len()
                && &bytes[j..j + 3] == b"mut"
                && j + 3 < bytes.len()
                && bytes[j + 3].is_ascii_whitespace()
            {
                j += 3;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
            }
            if j < bytes.len() && bytes[j] == b'_' {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// True if the stripped line carries a `#[cfg(test)]` (or `#[cfg(all(test,…))]`
/// etc.) attribute — a deliberately loose check sufficient for these files.
fn line_is_cfg_test_attr(code: &str) -> bool {
    let t = code.trim();
    t.starts_with("#[") && t.contains("cfg(") && t.contains("test")
}

fn read(rel: &str) -> (PathBuf, String) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join(rel);
    let content = std::fs::read_to_string(&path)
        .expect("cannot read offending production source file under the workspace root");
    (path, content)
}

/// Lines that live inside a `#[cfg(test)] mod …` block are exempt from the
/// `let _` ban (production code is the scope). Returns a per-line "is test
/// region" mask by tracking brace depth after a `#[cfg(test)]`-introduced
/// module. Single test module per file is the only shape present here.
fn test_region_mask(content: &str) -> Vec<bool> {
    let lines: Vec<&str> = content.lines().collect();
    let mut mask = vec![false; lines.len()];
    let mut i = 0usize;
    while i < lines.len() {
        let stripped = strip_line_comment(lines[i]);
        if line_is_cfg_test_attr(stripped) {
            // Find the next item line; if it is a `mod`, mask the whole block.
            let mut j = i + 1;
            while j < lines.len() {
                let t = strip_line_comment(lines[j]).trim().to_string();
                if t.is_empty() || t.starts_with("#[") {
                    j += 1;
                    continue;
                }
                break;
            }
            if j < lines.len()
                && strip_line_comment(lines[j])
                    .trim_start()
                    .starts_with("mod ")
            {
                // Mask from the cfg(test) line until the matching closing brace.
                let mut depth: i32 = 0;
                let mut seen_open = false;
                let mut k = i;
                while k < lines.len() {
                    mask[k] = true;
                    for ch in strip_line_comment(lines[k]).chars() {
                        if ch == '{' {
                            depth += 1;
                            seen_open = true;
                        } else if ch == '}' {
                            depth -= 1;
                        }
                    }
                    if seen_open && depth <= 0 {
                        break;
                    }
                    k += 1;
                }
                i = k + 1;
                continue;
            }
        }
        i += 1;
    }
    mask
}

#[test]
fn multinomial_reml_production_source_has_no_banned_let_underscore_discard() {
    let rel = "crates/gam-models/src/multinomial_reml.rs";
    let (path, content) = read(rel);
    let mask = test_region_mask(&content);

    let mut offenders: Vec<String> = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        if mask.get(idx).copied().unwrap_or(false) {
            continue; // inside a #[cfg(test)] module — allowed
        }
        let code = strip_line_comment(line);
        if line_has_let_underscore(code) {
            offenders.push(format!("{}:{}: {}", rel, idx + 1, line.trim()));
        }
    }

    assert!(
        offenders.is_empty(),
        "production `let _` discard(s) present; build.rs::scan_for_let_underscore \
         aborts the whole workspace build (exit 1) on these — `cargo build`, \
         `cargo test`, and the gamfit wheel all fail. The #1587 refactor left a \
         dangling `let _ = n_terms;` after the per-block penalty path was removed; \
         use `n_terms` or delete the binding. Offenders in {}: {:?}",
        path.display(),
        offenders
    );
}

#[test]
fn penalty_labels_has_no_cfg_test_gated_free_function() {
    // The scanner only exempts a private `mod tests` / `mod test_support` /
    // `mod tests_*` / `mod *_tests` after `#[cfg(test)]`. A `#[cfg(test)]`
    // directly gating a free `fn` (here the orphaned `penalty_label_layout`
    // shim) is a banned `#[cfg(test)] on a src/ item`.
    let rel = "crates/gam-custom-family/src/penalty_labels.rs";
    let (path, content) = read(rel);
    let lines: Vec<&str> = content.lines().collect();

    let mut offenders: Vec<String> = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        let code = strip_line_comment(line);
        if !line_is_cfg_test_attr(code) {
            continue;
        }
        // Walk to the first real item line.
        let mut j = idx + 1;
        while j < lines.len() {
            let t = strip_line_comment(lines[j]).trim().to_string();
            if t.is_empty() || t.starts_with("#[") {
                j += 1;
                continue;
            }
            break;
        }
        if j >= lines.len() {
            continue;
        }
        let item = strip_line_comment(lines[j]).trim_start();
        // Exempt the allowed test submodule forms.
        let is_exempt_mod = item.strip_prefix("mod ").map(|rest| {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            name == "tests"
                || name == "test_support"
                || name.starts_with("tests_")
                || name.ends_with("_tests")
        }) == Some(true);
        if is_exempt_mod {
            continue;
        }
        // Any non-`mod` item gated by `#[cfg(test)]` is the banned pattern; a
        // free `fn` is the concrete #1587 orphan.
        offenders.push(format!("{}:{}: gates `{}`", rel, idx + 1, item));
    }

    assert!(
        offenders.is_empty(),
        "`#[cfg(test)]`-gated non-module item(s) present; build.rs::\
         scan_for_cfg_test_on_pub_items aborts the whole workspace build (exit 1) \
         on these. The #1587 refactor orphaned the test-only \
         `penalty_label_layout` shim (a zero-`joint_specs` wrapper around \
         `penalty_label_layout_with_joint`); delete it. Offenders in {}: {:?}",
        path.display(),
        offenders
    );
}

/// Belt-and-braces: confirm both offending production files are reachable from
/// the workspace root so a future move/rename does not silently turn the checks
/// above into vacuous passes.
#[test]
fn offending_source_files_exist() {
    for rel in [
        "crates/gam-models/src/multinomial_reml.rs",
        "crates/gam-custom-family/src/penalty_labels.rs",
    ] {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path: &Path = &root.join(rel);
        assert!(path.is_file(), "expected production file missing: {rel}");
    }
}
