//! Bug hunt: the workspace does not build from a clean checkout at HEAD.
//!
//! Commit 700e6c5a2 ("perf(#932): revert SAE recon logdet channels to hand
//! row_jets_for_logdet (25-57x), keep jet as oracle") reverted the SAE
//! reconstruction log-det channels from the Taylor-jet path back to the hand
//! closed form, and in doing so demoted the jet path to a test-only oracle. The
//! revert landed FIVE distinct `build.rs` ban-scanner violations across FOUR
//! rules, all in the one freshly-rewired file
//! `crates/gam-sae/src/manifold/construction_row_jet_logdet_channels.rs`:
//!
//!   1. `#[allow(...)]` / `#[expect(...)]` lint silencer (scan_for_banned_allow):
//!        construction_row_jet_logdet_channels.rs:333
//!          #[allow(clippy::too_many_arguments)]
//!          fn fill_row_jets_hand_softmax(...)
//!   2. underscore-prefixed fn parameter (scan_for_underscore_fn_args):
//!        construction_row_jet_logdet_channels.rs:772
//!          fn refill_jet_window(..., _n: usize, ...)   // unused after the revert
//!   3. `#[cfg(test)]` on a src/ item, ×2 (scan_for_cfg_test_on_pub_items):
//!        construction_row_jet_logdet_channels.rs:646  #[cfg(test)] pub(crate) fn row_jets_for_logdet_batch4
//!        construction_row_jet_logdet_channels.rs:694  #[cfg(test)] fn batch4_assemble<const K: usize>
//!   4. `pub(crate)` item in src/ with ZERO consumers anywhere:
//!        construction_row_jet_logdet_channels.rs:647  pub(crate) fn row_jets_for_logdet_batch4
//!      (the `#[cfg(test)]` gate removes it from production while it stays
//!       `pub(crate)`, so no production caller and no test reference reach it).
//!
//! On any state that forces `build.rs` to re-run — a fresh `cargo build`,
//! `cargo test`, a `--release` build, or the `maturin` wheel build — the
//! ban scanner fires and the build script aborts:
//!
//!   cargo:warning=ban-scanner FAILED: 5 violation(s) across 4 rule(s); build aborted
//!   error: 5 ban violations across 4 rules
//!
//! so the whole workspace (and the `gamfit` wheel) exits non-zero. (A *cached*
//! `target/` hides it: a still-valid build-script fingerprint skips the scan
//! and a stale incremental build appears to succeed — which is how this rode
//! onto `main`. A fresh `--release` build, the wheel build, and CI all re-run
//! it and fail.)
//!
//! While the bug is present the crate cannot build, so this integration-test
//! target cannot even be produced — `cargo test` fails to compile it. Once the
//! violations are resolved (give `row_jets_for_logdet_batch4` a real production
//! caller or move both oracles into a `#[cfg(test)] mod`, consume or delete the
//! `_n` parameter, and refactor away the `too_many_arguments` silencer per the
//! ban's own remedy), the workspace builds again and the three checks below —
//! which re-implement, over this one file, the same patterns `build.rs` flags —
//! find nothing and pass, with no further edits.

use std::fs;
use std::path::PathBuf;

/// Path to the offending file, relative to the workspace root
/// (`CARGO_MANIFEST_DIR` is the root, where the `gam` crate lives).
fn offending_file() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates/gam-sae/src/manifold/construction_row_jet_logdet_channels.rs")
}

fn read_offending_file() -> String {
    let path = offending_file();
    fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// Strip a `//` line comment (everything from the first `//`), so a *mention*
/// of an attribute inside comment prose is not counted — matching build.rs's
/// `strip_file_lines` comment masking for these line-level checks.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

/// True if `code` carries an attribute-context lint silencer: `allow(` or
/// `expect(` immediately following (after optional whitespace) a `#[` or `#![`
/// attribute opener, with at least one lint token before the closing `)`.
/// Mirrors `build.rs::scan_for_banned_allow`.
fn line_has_silencer(code: &str) -> bool {
    if !code.contains("#[") {
        return false;
    }
    for silencer in ["allow(", "expect("] {
        let mut search_from = 0usize;
        while let Some(rel_idx) = code[search_from..].find(silencer) {
            let abs_match = search_from + rel_idx;
            let mut k = abs_match;
            while k > 0 && code.as_bytes()[k - 1].is_ascii_whitespace() {
                k -= 1;
            }
            let prefix = &code[..k];
            let is_attr = prefix.ends_with("#[") || prefix.ends_with("#![");
            let start = abs_match + silencer.len();
            if is_attr {
                if let Some(end_rel) = code[start..].find(')') {
                    let inside = &code[start..start + end_rel];
                    if inside.split(',').any(|tok| !tok.trim().is_empty()) {
                        return true;
                    }
                }
            }
            search_from = start;
            if search_from >= code.len() {
                break;
            }
        }
    }
    false
}

#[test]
fn logdet_channels_file_has_no_lint_silencing_attributes() {
    let content = read_offending_file();
    let mut offenders: Vec<String> = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        if line_has_silencer(strip_line_comment(line)) {
            offenders.push(format!("line {}: {}", idx + 1, line.trim()));
        }
    }
    assert!(
        offenders.is_empty(),
        "construction_row_jet_logdet_channels.rs carries lint-silencing allow/expect \
         attribute(s); build.rs's scan_for_banned_allow aborts the whole workspace build \
         (cargo build, cargo test, and the gamfit wheel all fail) on these: {offenders:?}"
    );
}

/// Extract the parameter-list text of `fn <name>` from `src`: the substring
/// between the first `(` after the function name and its matching `)`.
fn parameter_list(src: &str, name: &str) -> String {
    let needle = format!("fn {name}");
    let fn_at = src
        .find(&needle)
        .unwrap_or_else(|| panic!("function `{name}` not found in the file"));
    let open_rel = src[fn_at..]
        .find('(')
        .expect("function signature has no opening paren");
    let open = fn_at + open_rel;
    let bytes = src.as_bytes();
    let mut depth = 0i32;
    for (offset, byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return src[open + 1..open + offset].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced parens in signature of `{name}`");
}

/// Top-level (depth-0 within the param list) parameter names that are
/// underscore-prefixed, i.e. an `_ident` token sitting immediately before a
/// `:` at the outermost comma level. Reference/type underscores after the `:`
/// are not parameter names and are ignored. Mirrors the intent of
/// `build.rs::scan_for_underscore_fn_args`.
fn underscore_param_names(param_list: &str) -> Vec<String> {
    let bytes = param_list.as_bytes();
    let mut depth = 0i32; // nesting from <...>, (...), [...] in types
    let mut token = String::new();
    let mut hits = Vec::new();
    let mut at_name_position = true; // true at the start of each top-level param
    for &byte in bytes {
        let ch = byte as char;
        match ch {
            '<' | '(' | '[' => {
                depth += 1;
                token.clear();
            }
            '>' | ')' | ']' => {
                depth -= 1;
                token.clear();
            }
            ',' if depth == 0 => {
                at_name_position = true;
                token.clear();
            }
            ':' if depth == 0 => {
                if at_name_position && token.starts_with('_') && token.len() > 1 {
                    hits.push(token.clone());
                }
                at_name_position = false;
                token.clear();
            }
            c if c.is_alphanumeric() || c == '_' => token.push(c),
            _ => token.clear(),
        }
    }
    hits
}

#[test]
fn refill_jet_window_has_no_underscore_param() {
    let content = read_offending_file();
    let params = parameter_list(&content, "refill_jet_window");
    let banned = underscore_param_names(&params);
    assert!(
        banned.is_empty(),
        "`refill_jet_window` declares underscore-prefixed parameter(s) {banned:?}; \
         build.rs bans these (use the value, restructure the API, or delete the param) \
         and aborts the whole workspace build. Parameter list was: ({params})"
    );
}

/// Recognize a `#[cfg(test)]`-style attribute line (the outer `#[...]` form
/// that gates only the next item), e.g. `#[cfg(test)]`, `#[cfg(all(test, ...))]`,
/// `#[cfg(any(test, ...))]`. The inner `#![cfg(test)]` form (which gates a whole
/// module body) is intentionally not matched here.
fn is_cfg_test_outer_attr(stripped: &str) -> bool {
    let t = stripped.trim();
    t.starts_with("#[") && !t.starts_with("#![") && t.contains("cfg(") && {
        // crude word-boundary check for the bare `test` cfg token.
        let inner = t;
        inner.contains("(test)")
            || inner.contains("(test,")
            || inner.contains(" test)")
            || inner.contains(" test,")
            || inner.contains("(test ")
    }
}

/// True if the line is blank, a doc/line comment, or any `#[...]` attribute
/// other than the `#[cfg(test)]` form — i.e. a line we walk *past* when looking
/// upward from a fn signature for a directly-attached `#[cfg(test)]` gate.
fn is_skippable_attr_or_doc(stripped: &str) -> bool {
    let t = stripped.trim();
    t.is_empty() || t.starts_with("//") || (t.starts_with("#[") && !is_cfg_test_outer_attr(t))
}

/// Returns true iff the definition `fn <name>` is directly annotated by a
/// `#[cfg(test)]` outer attribute (walking upward across stacked attributes and
/// doc comments to the contiguous attribute block immediately above the
/// signature). A fn relocated *inside* a `#[cfg(test)] mod tests { ... }` would
/// NOT carry its own `#[cfg(test)]`, so this correctly stops flagging it once
/// the oracle is moved into a private test module (the ban's own remedy).
fn fn_is_directly_cfg_test_gated(content: &str, name: &str) -> bool {
    let lines: Vec<&str> = content.lines().collect();
    let needle = format!("fn {name}");
    let Some(def_line) = lines.iter().position(|l| l.contains(&needle)) else {
        panic!("function `{name}` not found in the file");
    };
    let mut i = def_line;
    while i > 0 {
        i -= 1;
        let line = lines[i];
        if is_cfg_test_outer_attr(line) {
            return true;
        }
        if is_skippable_attr_or_doc(line) {
            continue;
        }
        // First line that is neither blank, comment, nor an attribute: the
        // contiguous attribute block above the signature has ended.
        break;
    }
    false
}

#[test]
fn demoted_logdet_oracles_are_not_cfg_test_gated_src_items() {
    let content = read_offending_file();
    let mut offenders: Vec<&str> = Vec::new();
    // Both functions the revert demoted to test-only oracles. The first is also
    // `pub(crate)`, so its `#[cfg(test)]` gate simultaneously trips the
    // ZERO-consumers rule (no production caller, no test reference).
    for name in ["row_jets_for_logdet_batch4", "batch4_assemble"] {
        if fn_is_directly_cfg_test_gated(&content, name) {
            offenders.push(name);
        }
    }
    assert!(
        offenders.is_empty(),
        "function(s) {offenders:?} in construction_row_jet_logdet_channels.rs are gated by a \
         top-level `#[cfg(test)]` on a src/ item; build.rs's scan_for_cfg_test_on_pub_items \
         bans this (and the `pub(crate)` one additionally trips the ZERO-consumers rule), \
         aborting the whole workspace build. Give them a real production caller or move them \
         into a private `#[cfg(test)] mod`."
    );
}
