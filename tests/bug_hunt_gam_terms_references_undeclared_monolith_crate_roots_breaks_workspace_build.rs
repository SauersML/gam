//! Regression: the `gam-terms` workspace crate does not compile standalone, so
//! `cargo build --workspace`, `cargo build -p gam-terms` (and its reverse
//! dependents `gam-models` / `gam-sae` / `gam-solve`), and the `gamfit` wheel
//! build (`maturin develop`/`build`, which compiles `gam-pyffi` → … →
//! `gam-terms`) all abort at HEAD.
//!
//! # What's wrong
//!
//! The #1521 "crate carve" physically moved a large slice of the engine into
//! `crates/gam-terms/src/`, but those source files still address sibling
//! subsystems through **crate-root paths of the OLD monolithic `gam` crate**.
//! As of commit `ba7b1f35f` the carve is partly done but `gam-terms` still
//! leaks these monolith-only roots as `crate::…` (126 compile errors, down from
//! 225 a few commits earlier, so the set is actively shrinking):
//! `crate::estimate`, `crate::solver`, `crate::families`, `crate::pirls`,
//! `crate::custom_family`, `crate::mixture_link`, `crate::outer_subsample`,
//! `crate::util`, and `crate::types` — plus deeper submodule paths into roots
//! it *does* now declare, e.g. `crate::inference::formula_dsl`.
//!
//! Those roots exist only in the monolithic `gam` crate — as real modules
//! (`pub mod outer_subsample;`, `pub mod types;`, `pub mod util;` in
//! `src/lib.rs`) or as crate-root re-exports (`pub use gam_models as families;`,
//! `pub use gam_solve as solver;`). The carved `gam-terms` crate declares none
//! of them at its own crate root (its `lib.rs` only declares `basis`,
//! `construction`, `latent`, `smooth`, `structure`, `term_builder`, … and the
//! macros `bail_invalid_basis` / `bail_dim_basis` / `gpu_bail`). The compiler
//! even prints the fix:
//!
//! ```ignore
//! error[E0432]: unresolved import `crate::estimate`
//!  --> crates/gam-terms/src/smooth/prelude.rs:40:5
//! error[E0433]: failed to resolve: could not find `solver` in the crate root
//!  --> crates/gam-terms/src/smooth/design_construction.rs:3588:16
//!    | use crate::solver::rho_optimizer::OuterProblem;
//!    help: a similar path exists: `gam_solve::…`
//! error[E0433]: failed to resolve: could not find `outer_subsample` in the crate root
//!    | *current_row_set.borrow_mut() = crate::outer_subsample::RowSet::All;
//!    help: crate::gam_problem::outer_subsample::RowSet
//! ```
//!
//! The monolithic `gam` crate cannot build either, because it lists `gam-terms`
//! as a hard `[dependencies]` entry (`Cargo.toml`), so `cargo build`,
//! `cargo build --workspace`, and `cargo test` all abort at HEAD. The `gam` CLI
//! on `$PATH` keeps working only because it is a binary pre-built from an
//! earlier, compiling commit.
//!
//! # The invariant this test pins
//!
//! Every `crate::<root>` path used in a `gam-terms` library source file must
//! resolve to something the `gam-terms` crate provides at its own root: a
//! top-level module, a macro it defines, or a name it re-exports in `lib.rs`.
//! This test collects the crate's provided root names (every identifier in
//! `lib.rs`, every `macro_rules!` name across the crate, plus the cross-crate
//! `analytic_penalty_registry!` macro) and asserts that no library source
//! references a `crate::<root>` outside that set.
//!
//! The correct fix finishes the carve: rewrite the leaked references to the
//! crates that now own those subsystems (`gam_solve::…`, `gam_models::…`,
//! `gam_problem::…`, …) or import the macro, so the `crate::`-rooted references
//! disappear. Either way the set computed below empties and this guard passes,
//! with no edit to the test.
//!
//! This test lives in the `gam` crate, which compiles regardless of the state
//! of `gam-terms`, and inspects the carved sources statically — so it neither
//! requires `gam-terms` to compile nor shells out to `cargo`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Recursively collect every `.rs` file under `dir`.
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

/// Every maximal `[A-Za-z_][A-Za-z0-9_]*` identifier token in `src`.
fn identifiers(src: &str) -> impl Iterator<Item = String> + '_ {
    let bytes = src.as_bytes();
    let mut i = 0usize;
    std::iter::from_fn(move || {
        while i < bytes.len() {
            let b = bytes[i];
            let starts = b.is_ascii_alphabetic() || b == b'_';
            if starts {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                return Some(src[start..i].to_string());
            }
            i += 1;
        }
        None
    })
}

/// Replace `//` line comments, `/* … */` block comments, and string-literal
/// bodies with spaces (newlines preserved). A `crate::solver::…` written in a
/// doc comment or a string is prose, not a path the compiler resolves, so it
/// cannot make the crate fail to build — the invariant this test pins. Scanning
/// the raw text instead false-flagged such prose (e.g. a `//! crate::families`
/// design note), reporting a clean, compiling crate as broken. Mirror the
/// comment/string stripping the sibling scanner tests already apply.
fn strip_comments_and_strings(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        // Block comment.
        if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            out.push_str("  ");
            i += 2;
            while i < bytes.len() && !(bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/')) {
                out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            if i < bytes.len() {
                out.push_str("  ");
                i += 2;
            }
            continue;
        }
        // Line comment.
        if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        // String literal (with backslash escapes).
        if b == b'"' {
            out.push(' ');
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    out.push_str("  ");
                    i += 2;
                    continue;
                }
                out.push(if bytes[i] == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
            if i < bytes.len() {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        out.push(b as char);
        i += 1;
    }
    out
}

/// First path segment of every `crate::<seg>` occurrence in `src`.
fn crate_root_refs(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let needle = "crate::";
    let bytes = src.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = src[from..].find(needle) {
        let at = from + rel;
        // Require a word boundary before `crate` so `my_crate::x` does not match.
        let before_ok =
            at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
        let seg_start = at + needle.len();
        let mut j = seg_start;
        while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
            j += 1;
        }
        if before_ok && j > seg_start {
            out.push(src[seg_start..j].to_string());
        }
        from = seg_start.max(at + needle.len());
    }
    out
}

#[test]
fn gam_terms_only_references_crate_roots_it_provides() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let terms_src = Path::new(manifest).join("crates/gam-terms/src");
    let lib_rs = terms_src.join("lib.rs");

    let lib_text = fs::read_to_string(&lib_rs)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", lib_rs.display()));

    let mut sources = Vec::new();
    rust_sources(&terms_src, &mut sources);
    assert!(
        !sources.is_empty(),
        "expected gam-terms sources under {}",
        terms_src.display()
    );

    // Provided root names: every identifier declared/re-exported in lib.rs
    // (modules, re-exported types, locally-defined macros) — a deliberate
    // over-approximation so any genuinely-local root is treated as provided.
    let mut provided: BTreeSet<String> = identifiers(&lib_text).collect();
    // Every `macro_rules!`-defined macro anywhere in the crate is callable as
    // `crate::<name>!`.
    for path in &sources {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        // Scan the identifier token stream for `macro_rules! NAME`.
        let toks: Vec<String> = identifiers(&src).collect();
        for w in toks.windows(2) {
            if w[0] == "macro_rules" {
                provided.insert(w[1].clone());
            }
        }
    }
    // `analytic_penalty_registry!` is provided via a cross-crate macro import
    // (it resolves today — it is not one of the leaked monolith roots).
    provided.insert("analytic_penalty_registry".to_string());

    // Referenced crate roots in library (non-test) sources. `*tests*.rs` files
    // hold `#[cfg(test)]`-only references (e.g. `crate::test_support`) that do
    // not participate in the failing *library* build, so they are excluded.
    let mut undeclared: BTreeMap<String, PathBuf> = BTreeMap::new();
    for path in &sources {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.contains("test") {
            continue;
        }
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        for root in crate_root_refs(&strip_comments_and_strings(&src)) {
            if !provided.contains(&root) {
                undeclared.entry(root).or_insert_with(|| path.clone());
            }
        }
    }

    assert!(
        undeclared.is_empty(),
        "gam-terms references {} crate-root path(s) it does not provide, so the \
         crate does not compile standalone and `cargo build --workspace` / the \
         gamfit wheel build abort. The #1521 carve left these monolith-only \
         roots addressed as `crate::…`:\n{}\nFinish the carve by routing each to \
         the crate that now owns it (gam_solve::, gam_models::, gam_problem::, …) \
         or importing the macro, so the `crate::`-rooted references disappear.",
        undeclared.len(),
        undeclared
            .iter()
            .map(|(root, path)| format!("  - `crate::{root}` (e.g. {})", path.display()))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}
