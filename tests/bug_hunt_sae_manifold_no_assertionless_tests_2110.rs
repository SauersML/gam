//! #2110 defense-in-depth: no assertion-less `#[test]` may live in the gam-sae
//! manifold test tree — checked at HARNESS time, so it fires even when a cached
//! `target/` lets the workspace-root `build.rs` skip its own scan.
//!
//! #2110 (assertion-less #2101 birth-locus probes aborting the whole build) is
//! the latest in a recurring pattern: a Claude-authored SAE commit lands a
//! print-only diagnostic `#[test]` under `crates/gam-sae/src/manifold/`, which
//! trips `build.rs::scan_for_useless_tests` and breaks `cargo build`/`--release`
//! and the `gamfit` wheel. The build.rs ban is the primary gate, but it only
//! re-runs when its build-script fingerprint is invalidated — a stale
//! incremental `target/` skips it, which is *exactly* how the #2110 probes rode
//! onto `main` (see the issue's "A cached `target/` hides this" note). This test
//! runs every `cargo test` regardless of build-script caching, over the whole
//! `crates/gam-sae/src/manifold` directory rather than a single file, so the
//! *next* print-only probe in this subsystem is caught here even if the build.rs
//! fingerprint is warm.
//!
//! The assertion detector mirrors `build.rs::line_is_assertion_shaped` /
//! `body_reaches_assertion`: a `#[test]` is fine if its body — or, following one
//! hop of local helper delegation — reaches an `assert*!`/`panic!`/
//! `unreachable!`/`todo!`/`unimplemented!` macro or a propagating `?`, or if it
//! is guarded by `#[should_panic]`. `.expect(...)`/`.unwrap()` do NOT count, by
//! design (matching build.rs). Because the workspace only builds when build.rs's
//! own stricter scan finds zero offenders tree-wide, a green build guarantees
//! this scoped scan finds zero too; it exists to catch the *regression window*
//! between "print-only test committed" and "someone runs a fresh build".

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

fn manifold_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("crates/gam-sae/src/manifold")
}

/// Strip a `//` line comment (mirrors build.rs comment masking). Not
/// block-comment aware — matching the line-oriented scanner it mirrors.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

/// A propagating `?`: a `?` followed (after optional spaces) by end-of-line or
/// one of `; , . )`. Mirrors `build.rs::line_contains_propagating_question`.
fn has_propagating_question(code: &str) -> bool {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] == b'?' {
            let mut k = i + 1;
            while k < n && bytes[k] == b' ' {
                k += 1;
            }
            if k == n || matches!(bytes[k], b';' | b',' | b'.' | b')') {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// True when a code line carries an assertion-shaped construct, matching the
/// full set recognised by `build.rs::line_is_assertion_shaped`: the fixed
/// assert/panic macro set, a propagating `?`, and — critically — any macro or
/// bare call whose identifier follows the `assert_*` / `expect_*` / `require_*`
/// / `ensure_*` helper convention (build.rs::line_contains_assertion_helper_*).
/// Missing the helper-convention recognizers makes this detector STRICTER than
/// build.rs and produces false positives on legitimate helper-asserted tests.
fn line_is_assertion_shaped(code: &str) -> bool {
    const MACROS: [&str; 10] = [
        "assert!(",
        "assert_eq!(",
        "assert_ne!(",
        "debug_assert!(",
        "debug_assert_eq!(",
        "debug_assert_ne!(",
        "panic!(",
        "unreachable!(",
        "todo!(",
        "unimplemented!(",
    ];
    MACROS.iter().any(|m| code.contains(m))
        || has_propagating_question(code)
        || contains_assertion_helper(code)
}

/// Mirrors `build.rs::line_contains_assertion_helper_macro` +
/// `line_contains_assertion_helper_call`: an identifier followed by `!(`/`![`/
/// `!{` (macro) or `(` (bare call) whose name starts with an assertion-helper
/// prefix. Resolved purely by name convention, exactly as build.rs does — no
/// callee body is inspected.
fn contains_assertion_helper(code: &str) -> bool {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        // Find an ident immediately followed by `!<delim>` (macro) or `(` (call).
        let is_macro = bytes[i] == b'!'
            && matches!(bytes.get(i + 1), Some(b'(') | Some(b'[') | Some(b'{'));
        let is_call = bytes[i] == b'(' && !(i > 0 && bytes[i - 1] == b'!');
        if !is_macro && !is_call {
            i += 1;
            continue;
        }
        let mut start = i;
        while start > 0 && is_ident_byte(bytes[start - 1]) {
            start -= 1;
        }
        if start < i {
            let name = &code[start..i];
            if name.starts_with("assert_")
                || name.starts_with("expect_")
                || name.starts_with("require_")
                || name.starts_with("ensure_")
            {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Collect identifiers in bare-call position (`ident(`, not `ident!(`) on a
/// stripped line, so a `#[test]` that only asserts *inside* a local helper it
/// calls is not mis-flagged. Conservative superset of
/// `build.rs::collect_called_idents` (turbofish handling folded in by stepping
/// over a `>`-terminated `::<…>` segment before the `(`).
fn collect_called_idents(code: &str, out: &mut Vec<String>) {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] != b'(' || (i > 0 && bytes[i - 1] == b'!') {
            i += 1;
            continue;
        }
        let mut id_end = i;
        if i > 0 && bytes[i - 1] == b'>' {
            // Step back over a balanced `<…>` and a preceding `::` (turbofish).
            let mut depth = 0i32;
            let mut p = i - 1;
            let mut open = None;
            loop {
                match bytes[p] {
                    b'>' => depth += 1,
                    b'<' => {
                        depth -= 1;
                        if depth == 0 {
                            open = Some(p);
                            break;
                        }
                    }
                    _ => {}
                }
                if p == 0 {
                    break;
                }
                p -= 1;
            }
            if let Some(lt) = open
                && lt >= 2
                && bytes[lt - 1] == b':'
                && bytes[lt - 2] == b':'
            {
                id_end = lt - 2;
            }
        }
        let mut start = id_end;
        while start > 0 {
            let b = bytes[start - 1];
            if b == b'_' || b.is_ascii_alphanumeric() {
                start -= 1;
            } else {
                break;
            }
        }
        if start < id_end {
            out.push(code[start..id_end].to_string());
        }
        i += 1;
    }
}

/// `(name, first_body_line, last_body_line)` for every `fn <name>(…) { … }` in
/// the file, indexed by brace balance from the signature.
fn index_fns(stripped: &[String]) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let n = stripped.len();
    let mut i = 0usize;
    while i < n {
        let s = &stripped[i];
        // A `fn` token in declaration position (word-boundary), skipping `fn(`
        // pointer types which have no following identifier.
        if let Some(name) = fn_name_on_line(s) {
            // Walk to the opening `{` and match its close.
            let mut depth = 0i32;
            let mut started = false;
            let mut k = i;
            let open = loop {
                if k >= n {
                    break None;
                }
                if stripped[k].contains('{') {
                    break Some(k);
                }
                // Signature without a brace before `;` → not a body (trait decl).
                if stripped[k].contains(';') && !stripped[k].contains('{') {
                    break None;
                }
                k += 1;
            };
            if let Some(open_line) = open {
                let mut close_line = open_line;
                let mut kk = open_line;
                while kk < n {
                    for ch in stripped[kk].chars() {
                        if ch == '{' {
                            depth += 1;
                            started = true;
                        } else if ch == '}' {
                            depth -= 1;
                        }
                    }
                    if started && depth == 0 {
                        close_line = kk;
                        break;
                    }
                    kk += 1;
                }
                out.push((name, open_line, close_line));
                i = open_line + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Extract the function name from a stripped line that declares `fn <name>(`.
fn fn_name_on_line(s: &str) -> Option<String> {
    let idx = find_word(s, "fn")?;
    let rest = s[idx + 2..].trim_start();
    // A generic like `fn foo<T>(…)` still starts with the identifier.
    let name: String = rest
        .chars()
        .take_while(|c| *c == '_' || c.is_ascii_alphanumeric())
        .collect();
    if name.is_empty() {
        None // `fn(` pointer type
    } else {
        Some(name)
    }
}

/// Find `word` as a whitespace/boundary-delimited token in `s`.
fn find_word(s: &str, word: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let wb = word.as_bytes();
    let mut i = 0usize;
    while i + wb.len() <= bytes.len() {
        if &bytes[i..i + wb.len()] == wb {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after = i + wb.len();
            let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
            if before_ok && after_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

/// Whether the body `[open..=close]` reaches an assertion directly or via up to
/// `depth` hops of local-helper delegation (cycle-guarded by `visited`).
fn body_reaches_assertion(
    stripped: &[String],
    open: usize,
    close: usize,
    fns: &HashMap<String, (usize, usize)>,
    visited: &mut Vec<usize>,
    depth: usize,
) -> bool {
    for line in &stripped[open..=close.min(stripped.len() - 1)] {
        if line_is_assertion_shaped(line) {
            return true;
        }
    }
    if depth == 0 {
        return false;
    }
    let mut callees = Vec::new();
    for line in &stripped[open..=close.min(stripped.len() - 1)] {
        collect_called_idents(line, &mut callees);
    }
    for callee in &callees {
        if let Some(&(h_open, h_close)) = fns.get(callee) {
            if visited.contains(&h_open) {
                continue;
            }
            visited.push(h_open);
            if body_reaches_assertion(stripped, h_open, h_close, fns, visited, depth - 1) {
                return true;
            }
        }
    }
    false
}

/// Report `(line, signature)` of each `#[test]` fn in `content` that reaches no
/// assertion (directly or via ≤3 delegation hops) and carries no
/// `#[should_panic]`.
fn assertionless_tests(content: &str) -> Vec<(usize, String)> {
    let raw: Vec<&str> = content.lines().collect();
    let stripped: Vec<String> = raw.iter().map(|l| strip_line_comment(l).to_string()).collect();
    let n = raw.len();

    // Index local fns so a test can delegate its assertions to a helper.
    let indexed = index_fns(&stripped);
    let mut fn_map: HashMap<String, (usize, usize)> = HashMap::new();
    for (name, open, close) in &indexed {
        fn_map.entry(name.clone()).or_insert((*open, *close));
    }

    let mut offenders = Vec::new();
    let mut i = 0usize;
    while i < n {
        if !stripped[i].contains("#[test]") {
            i += 1;
            continue;
        }
        // Walk past attributes/blank lines to the `fn` line, noting should_panic.
        let mut has_should_panic = stripped[i].contains("#[should_panic");
        let mut j = i + 1;
        while j < n {
            let tt = stripped[j].trim();
            if tt.is_empty() {
                j += 1;
                continue;
            }
            if tt.starts_with("#[") || tt.starts_with("#![") {
                if stripped[j].contains("#[should_panic") {
                    has_should_panic = true;
                }
                j += 1;
                continue;
            }
            break;
        }
        if j >= n {
            break;
        }
        let sig_line = j;
        // Locate this test's body braces.
        let mut depth = 0i32;
        let mut started = false;
        let mut open = sig_line;
        let mut k = sig_line;
        while k < n {
            for ch in stripped[k].chars() {
                if ch == '{' {
                    if !started {
                        open = k;
                    }
                    depth += 1;
                    started = true;
                } else if ch == '}' {
                    depth -= 1;
                }
            }
            if started && depth == 0 {
                break;
            }
            k += 1;
        }
        let close = k.min(n - 1);
        let mut visited = vec![open];
        let reached = body_reaches_assertion(&stripped, open, close, &fn_map, &mut visited, 3);
        if !has_should_panic && !reached {
            offenders.push((sig_line + 1, raw[sig_line].trim().to_string()));
        }
        i = close + 1;
    }
    offenders
}

/// Every `.rs` file directly under `crates/gam-sae/src/manifold` (the subsystem
/// where the assertion-less-probe failure mode recurs) must be free of
/// assertion-less `#[test]` functions.
#[test]
fn gam_sae_manifold_tests_are_not_assertionless() {
    let dir = manifold_dir();
    let entries = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut all_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    let mut files_scanned = 0usize;
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        files_scanned += 1;
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        for (line, sig) in assertionless_tests(&content) {
            all_offenders.push((path.clone(), line, sig));
        }
    }
    assert!(
        files_scanned > 10,
        "expected to scan the manifold test tree, only saw {files_scanned} .rs files — \
         has the directory layout moved? (guarding against a vacuous pass)"
    );
    assert!(
        all_offenders.is_empty(),
        "{} assertion-less #[test] function(s) under crates/gam-sae/src/manifold — each trips \
         build.rs::scan_for_useless_tests and aborts the whole workspace build (cargo build/test, \
         --release, and the gamfit wheel):\n{}",
        all_offenders.len(),
        all_offenders
            .iter()
            .map(|(p, ln, sig)| format!(
                "  {}:{ln}: {sig}",
                p.file_name().and_then(|s| s.to_str()).unwrap_or("?")
            ))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// Guard the guard: the detector must classify an `eprintln!`-only `#[test]` as
/// assertion-less, must clear an `assert!`-bearing one and a `#[should_panic]`
/// one, and must follow a helper-delegated assertion (so it does not mis-flag
/// tests whose assertions live in a called helper — the escape hatch that keeps
/// the whole-directory scan free of false positives).
#[test]
fn detector_flags_only_genuinely_assertionless_tests() {
    let sample = r#"
fn check(v: i32) {
    assert_eq!(v, 4);
}

#[test]
fn print_only() {
    eprintln!("no assertions");
    let _ = compute().expect("value");
}

#[test]
fn direct_assert() {
    assert!(2 + 2 == 4);
}

#[test]
fn via_helper() {
    check(2 + 2);
}

#[test]
#[should_panic(expected = "boom")]
fn expected_panic() {
    trigger();
}
"#;
    let offenders = assertionless_tests(sample);
    assert_eq!(
        offenders.len(),
        1,
        "expected exactly `print_only` to be flagged, got {offenders:?}"
    );
    assert!(
        offenders[0].1.contains("print_only"),
        "wrong test flagged: {offenders:?}"
    );
}
