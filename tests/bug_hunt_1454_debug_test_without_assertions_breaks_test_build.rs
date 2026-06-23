//! Regression: the crate's TEST build does not succeed from a clean checkout at
//! HEAD, because a recently-added debug `#[test]` carries no assertion and trips
//! `build.rs`'s useless-test ban.
//!
//! `build.rs` runs a first-party hygiene scanner (`scan_for_useless_tests` →
//! `test_body_reaches_assertion`) over every `*.rs` file under `src/`. The rule
//! requires every `#[test]` function to *verify something* — its body (directly
//! or through a delegated helper) must reach an assertion-shaped construct:
//! `assert!` / `assert_eq!` / `assert_ne!` / a `debug_assert*` / a panic-family
//! macro / a `?`-propagation / an `assert_*`/`expect_*`/`require_*`/`ensure_*`-
//! named helper. Plain `.unwrap()` / `.expect("…")` / `eprintln!` do NOT count
//! (they are not assertion-shaped), so a test that only prints is rejected.
//!
//! Commit a6309b3bc ("debug(#1454): expose a_uv/auvd12 + auvd12 FD-localizer to
//! split jet-recovery vs composition") added
//!
//!   src/families/survival/marginal_slope/tests.rs
//!     `#[test] fn flex_bidir_auvd12_fd_1454()`
//!
//! whose body computes a 2-D central finite difference of the base intercept
//! Hessian `a_uv` and compares it to the analytic `auvd12`, but only ever
//! `eprintln!`s the discrepancy — there is no `assert!`. The scanner therefore
//! aborts the build:
//!
//!   error: 1 ban violation across 1 rule
//!   error — #[test] function without assertions (test must verify something —
//!     add assert! / assert_eq! / ? / #[should_panic] or delete the test)
//!     error: src/families/survival/marginal_slope/tests.rs:8281: #[test]
//!
//! Because the scanner runs in `build.rs`, this aborts the WHOLE test build:
//! `cargo test` exits non-zero before any test target is even compiled, so the
//! entire suite (every `tests/*.rs`, every `#[cfg(test)]` module) cannot run.
//! `cargo build --lib` / the release binary / the wheel are unaffected (the
//! offending function lives in a `#[cfg(test)]` module that those profiles do
//! not pull in to a degree that re-keys `build.rs` against it in the same way),
//! so this is specifically a red `cargo test` / CI test-job break. A *cached*
//! `target/` can also mask it whenever `build.rs`'s fingerprint stays valid and
//! the script is skipped.
//!
//! While the bug is present the crate's test build cannot be produced, so this
//! integration-test target itself cannot even be compiled — `cargo test` fails
//! for everyone. Once `flex_bidir_auvd12_fd_1454` is given a genuine assertion
//! (e.g. asserting the FD `worst` gap is below a tolerance — which is the whole
//! point of an FD localizer), routed through an assertion helper, marked
//! `#[should_panic]`, or deleted, `build.rs` stops flagging it, the test build
//! succeeds again, and the source-level check below passes with no further
//! edits.
//!
//! The check is scoped to the exact function the offending commit added (looked
//! up by name in its specific file), and mirrors the build-time recognizer, so
//! it neither under- nor over-fires on unrelated tests.

use std::fs;
use std::path::Path;

const TARGET_REL: &str = "src/families/survival/marginal_slope/tests.rs";
const TARGET_FN: &str = "flex_bidir_auvd12_fd_1454";

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else if first >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

/// Replace block-comment bodies with spaces (newlines preserved) so braces and
/// assertion tokens that appear only inside `/* … */` cannot confuse the scan.
fn strip_block_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    let mut in_block = false;
    while i < bytes.len() {
        if in_block {
            if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                out.push_str("  ");
                i += 2;
                in_block = false;
                continue;
            }
            out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
            i += 1;
            continue;
        }
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            out.push_str("  ");
            i += 2;
            in_block = true;
            continue;
        }
        let ch_len = utf8_len(bytes[i]);
        for b in &bytes[i..(i + ch_len).min(bytes.len())] {
            out.push(*b as char);
        }
        i += ch_len;
    }
    out
}

/// Remove `//` line comments and string-literal bodies from one line. This is
/// what lets brace-matching survive `format!("{u},{v}")` and what stops a stray
/// `assert` substring inside a string/comment from masking the real defect.
fn strip_line_comment_and_strings(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    let mut in_str = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            break;
        }
        if b == b'"' {
            in_str = true;
            i += 1;
            continue;
        }
        let ch_len = utf8_len(b);
        for cb in &bytes[i..(i + ch_len).min(bytes.len())] {
            out.push(*cb as char);
        }
        i += ch_len;
    }
    out
}

fn cleaned_source(src: &str) -> String {
    strip_block_comments(src)
        .lines()
        .map(strip_line_comment_and_strings)
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// The body text (between the matching `{` `}`) of `fn <name>` in `cleaned`, or
/// `None` if the function is absent. Strings/comments are already removed, so
/// brace matching is reliable.
fn fn_body<'a>(cleaned: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("fn {name}");
    let bytes = cleaned.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = cleaned[from..].find(&needle) {
        let kw = from + rel;
        let before_ok = kw == 0 || !is_ident_byte(bytes[kw - 1]);
        let after = kw + needle.len();
        let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            // Find the body `{` after the signature (the first `{` past the
            // parameter/return list). A `#[test] fn ...()` has no generic body
            // brace ambiguity here.
            if let Some(open_rel) = cleaned[after..].find('{') {
                let open = after + open_rel;
                let mut depth = 0i32;
                let mut j = open;
                while j < bytes.len() {
                    match bytes[j] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(&cleaned[open + 1..j]);
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
            }
        }
        from = kw + needle.len();
    }
    None
}

/// Mirror of `build.rs`'s per-line `line_is_assertion_shaped`: does this line
/// carry an assertion-shaped construct the useless-test gate accepts?
fn line_is_assertion_shaped(s: &str) -> bool {
    // The realistic fix for an FD localizer is an `assert!`/`assert_eq!`; the
    // panic-family and `debug_assert*` forms the build-time recognizer also
    // accepts are themselves banned by build.rs, so a maintainer would not
    // reach for them — and embedding those exact tokens here would trip the
    // banned-substring scanner. Recognizing the assert family is sufficient and
    // well-posed.
    const MACROS: [&str; 3] = ["assert!(", "assert_eq!(", "assert_ne!("];
    if MACROS.iter().any(|m| s.contains(m)) {
        return true;
    }
    // `?`-propagation (only meaningful in a Result-returning test, but the gate
    // accepts it regardless).
    if s.contains('?') {
        return true;
    }
    // Assertion-helper naming convention the gate also accepts.
    for prefix in ["assert_", "expect_", "require_", "ensure_"] {
        if let Some(pos) = s.find(prefix) {
            let after = &s[pos + prefix.len()..];
            if after.starts_with(|c: char| c == '_' || c.is_ascii_alphanumeric()) {
                return true;
            }
        }
    }
    false
}

/// Does any `*.rs` file under `src/` OTHER than a `tests.rs` / `#[cfg(test)]`
/// test module reference `.<field>`? Used to witness the dead-code break: a
/// `pub(crate)` field read only from cfg(test) code is dead in every non-test
/// build and fails `-D dead-code`.
fn field_read_outside_tests(field: &str) -> bool {
    fn walk(dir: &Path, field_tok: &str) -> bool {
        let Ok(entries) = fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if walk(&path, field_tok) {
                    return true;
                }
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            // Skip the dedicated test files; those are exactly the cfg(test)
            // readers that do NOT keep the field alive in a non-test build.
            if path.file_name().and_then(|f| f.to_str()) == Some("tests.rs") {
                continue;
            }
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            let cleaned = cleaned_source(&content);
            // A read looks like `.auvd12` not preceded by an ident char (so we
            // do not match the `auvd12:` field declaration or `field_read...`).
            let tok = format!(".{field_tok}");
            let bytes = cleaned.as_bytes();
            let mut from = 0usize;
            while let Some(rel) = cleaned[from..].find(&tok) {
                let at = from + rel;
                let after = at + tok.len();
                let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
                if after_ok {
                    return true;
                }
                from = at + tok.len();
            }
        }
        false
    }
    let manifest = env!("CARGO_MANIFEST_DIR");
    walk(&Path::new(manifest).join("src"), field)
}

/// Is `<field>:` declared as a struct field anywhere under `src/`?
fn field_is_declared(field: &str) -> bool {
    fn walk(dir: &Path, decl: &str) -> bool {
        let Ok(entries) = fs::read_dir(dir) else {
            return false;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if walk(&path, decl) {
                    return true;
                }
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            if cleaned_source(&content).contains(decl) {
                return true;
            }
        }
        false
    }
    let manifest = env!("CARGO_MANIFEST_DIR");
    walk(&Path::new(manifest).join("src"), &format!("{field}: Array2<f64>"))
}

/// Break B: the #1454 debug commit added `auvd12` to
/// `SurvivalFlexTimepointBiDirectionalExact`, read ONLY from the cfg(test)
/// `flex_bidir_auvd12_fd_1454`. In every non-test build (release wheel, the
/// integration-test lib link) that field is dead code, so `-D dead-code`
/// (implied by the workspace `warnings = "deny"`) fails the build. Passes once
/// the field is wired into non-test code or removed with the debug machinery.
#[test]
fn debug_field_auvd12_must_be_used_outside_tests_or_removed() {
    if !field_is_declared("auvd12") {
        // Field removed (debug machinery reverted) — no dead-code break.
        return;
    }
    assert!(
        field_read_outside_tests("auvd12"),
        "`auvd12` (SurvivalFlexTimepointBiDirectionalExact, added by #1454 debug \
         commit a6309b3bc) is declared but read only from cfg(test) code, so it is \
         dead in every non-test build and fails `-D dead-code` (workspace \
         warnings=\"deny\"). This is masked at HEAD only because build.rs's \
         useless-test ban aborts first (see the sibling test). Wire it into the \
         non-test path or delete the temp debug field."
    );
}

#[test]
fn flex_bidir_auvd12_fd_1454_debug_test_must_assert_or_be_absent() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(manifest).join(TARGET_REL);
    let content =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let cleaned = cleaned_source(&content);

    let Some(body) = fn_body(&cleaned, TARGET_FN) else {
        // Function deleted entirely — the build-time ban can no longer fire.
        return;
    };

    // Whole-function `#[should_panic]` would also satisfy the gate; accept it.
    let has_should_panic = content.contains("#[should_panic");

    let reaches_assertion = has_should_panic || body.lines().any(line_is_assertion_shaped);

    assert!(
        reaches_assertion,
        "`fn {TARGET_FN}` in {TARGET_REL} is a #[test] with no assertion-shaped \
         construct (only eprintln!/.unwrap()/.expect()). build.rs's useless-test \
         ban rejects it and aborts the ENTIRE `cargo test` build, so no test in \
         the crate can run. Give the FD localizer a real assertion (e.g. assert \
         the worst FD gap is below tolerance), route it through an assertion \
         helper, mark it #[should_panic], or delete it."
    );
}
