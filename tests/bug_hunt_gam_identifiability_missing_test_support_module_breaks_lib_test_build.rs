//! Regression (sibling of #1578 / #1579; same #1521 carve, #1567-style
//! test-build variant): the carved `gam-identifiability` crate's `#[cfg(test)]`
//! code references a crate-root `test_support` module — and bare helper
//! functions it is meant to export — that the carve never created in the crate.
//! Its **library compiles**, but its **lib-test build does not**:
//!
//! ```ignore
//! $ cargo test -p gam-identifiability --no-run
//! error[E0432]: unresolved import `crate::test_support`
//!   --> crates/gam-identifiability/src/canonical.rs:2452:16
//!    | use crate::test_support::spec_from_dense_with_priority;
//!    | help: a similar path exists: `gam_linalg::test_support`
//! error[E0425]: cannot find function `spec_from_dense` in this scope
//!   --> crates/gam-identifiability/src/audit.rs:3056:13   (×13)
//! error: could not compile `gam-identifiability` (lib test) due to 14 previous errors
//! ```
//!
//! The helpers `spec_from_dense` / `spec_from_dense_with_priority` are defined
//! only inside `canonical.rs`'s own `#[cfg(test)] mod tests` (canonical.rs:3145
//! / :3163); `audit.rs`'s test module calls `spec_from_dense(…)` as if it were
//! in scope, and `canonical.rs` itself imports `crate::test_support::…`. Both
//! expect a shared crate-root `test_support` module (the pattern
//! `crates/gam-linalg/src/test_support.rs` already follows) that was not carried
//! into `gam-identifiability`.
//!
//! Because CI runs a workspace-wide `cargo nextest run` / `cargo test` with no
//! `-p` filter, this aborts the entire lib-test phase (exactly like the
//! previously-closed #1567 / #1566 carve test-build breaks).
//!
//! # The invariant this test pins
//!
//! If any `gam-identifiability` source references `crate::test_support`, the
//! crate must declare a `test_support` module at its crate root. The minimal
//! fix adds `crates/gam-identifiability/src/test_support.rs` exporting
//! `spec_from_dense` / `spec_from_dense_with_priority` (and a
//! `#[cfg(test)] mod test_support;` in `lib.rs`), with `audit.rs`/`canonical.rs`
//! importing the helpers from it — which resolves both the `E0432` import and
//! the 13 `E0425` out-of-scope calls. This guard then passes with no edit.
//!
//! This test lives in the `gam` crate and inspects the carved sources
//! statically, so it does not itself require `gam-identifiability` to compile.

use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Does `code` contain a `crate::test_support` path (excluding `$crate::` and
/// `my_crate::`)?
fn references_crate_test_support(code: &str) -> bool {
    let needle = "crate::test_support";
    let bytes = code.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = code[from..].find(needle) {
        let at = from + rel;
        let prev_ok = at == 0 || {
            let p = bytes[at - 1];
            !(p.is_ascii_alphanumeric() || p == b'_' || p == b'$')
        };
        // The matched token must end at a non-ident byte so `crate::test_supportX`
        // does not count.
        let after = at + needle.len();
        let end_ok = code[after..]
            .chars()
            .next()
            .map(|c| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(true);
        if prev_ok && end_ok {
            return true;
        }
        from = at + needle.len();
    }
    false
}

/// Does the crate-root module (`lib.rs` plus anything it `include!`s) declare a
/// `test_support` module (`mod test_support` / `pub mod test_support`,
/// optionally `#[cfg(test)]`)?
fn root_declares_test_support(src_dir: &Path) -> bool {
    let lib = src_dir.join("lib.rs");
    let mut text = match fs::read_to_string(&lib) {
        Ok(t) => t,
        Err(_) => return false,
    };
    let needle = "include!";
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(needle) {
        let at = from + rel;
        if let Some(q1) = text[at..].find('"') {
            let s = at + q1 + 1;
            if let Some(q2) = text[s..].find('"') {
                let inc = src_dir.join(&text[s..s + q2]);
                if let Ok(extra) = fs::read_to_string(&inc) {
                    text.push('\n');
                    text.push_str(&extra);
                }
                from = s + q2 + 1;
                continue;
            }
        }
        from = at + needle.len();
    }
    text.lines().any(|line| {
        let t = line.trim_start();
        t.starts_with("mod test_support")
            || t.starts_with("pub mod test_support")
            || t.starts_with("pub(crate) mod test_support")
    })
}

#[test]
fn gam_identifiability_declares_test_support_if_referenced() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src_dir = Path::new(manifest).join("crates/gam-identifiability/src");

    let mut sources = Vec::new();
    rust_sources(&src_dir, &mut sources);
    assert!(
        !sources.is_empty(),
        "expected gam-identifiability sources under {}",
        src_dir.display()
    );

    let mut referencing: Vec<PathBuf> = Vec::new();
    for path in &sources {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if references_crate_test_support(&src) {
            referencing.push(path.clone());
        }
    }

    if referencing.is_empty() {
        // Nothing references `crate::test_support`; nothing to enforce.
        return;
    }

    assert!(
        root_declares_test_support(&src_dir),
        "gam-identifiability references `crate::test_support` in {} source file(s) \
         (e.g. {}) but declares no `test_support` module at its crate root, so \
         `cargo test -p gam-identifiability --no-run` fails (E0432 + 13× E0425 \
         `spec_from_dense`) and the workspace lib-test phase aborts. The #1521 \
         carve never carried the shared test-helper module into the crate. Add \
         `crates/gam-identifiability/src/test_support.rs` (exporting \
         `spec_from_dense` / `spec_from_dense_with_priority`) and a \
         `#[cfg(test)] mod test_support;` in `lib.rs`.",
        referencing.len(),
        referencing[0].display(),
    );
}
