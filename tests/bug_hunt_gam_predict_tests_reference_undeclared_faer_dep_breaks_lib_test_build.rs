//! Regression (#1567): the `gam-predict` crate's `#[cfg(test)]` modules
//! reference the `faer` crate (e.g. `use faer::sparse::{SparseColMat, Triplet};`
//! in `crates/gam-predict/src/linalg.rs`'s test module), but the #1521 crate
//! split carved `gam-predict` out of the monolithic `gam` crate WITHOUT carrying
//! a `faer` dependency declaration into `crates/gam-predict/Cargo.toml`.
//!
//! `faer` is referenced only from test code, so `cargo build -p gam-predict`
//! stays green — but `cargo test -p gam-predict --no-run` fails to compile with
//!
//! ```ignore
//! error[E0433]: failed to resolve: use of unresolved module or unlinked crate `faer`
//!  --> crates/gam-predict/src/linalg.rs:328:9
//! ```
//!
//! Because CI runs a workspace-wide `cargo nextest run --lib` with no `-p`
//! filter, this aborts the entire lib-test phase. The minimal fix is to declare
//! `faer` (matching the `0.24.0` the rest of the workspace pins) under
//! `[dev-dependencies]` of `crates/gam-predict/Cargo.toml`, since the only
//! references live in `#[cfg(test)]` code.
//!
//! This test (it lives in the `gam` crate, which compiles regardless) asserts
//! the invariant *statically*: if any `gam-predict` source file references the
//! `faer` crate, then `gam-predict`'s `Cargo.toml` MUST declare `faer` in
//! `[dependencies]` or `[dev-dependencies]`. Declaring the dev-dependency makes
//! `gam-predict`'s test build compile again and makes this guard pass, with no
//! further edits.

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

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Does `src` reference the `faer` crate as a path root? Matches `faer::` and
/// `use faer ;`/`use faer::…` where the `faer` token is a whole word (so it does
/// not match `gam_faer`, `myfaer`, or the substring inside `faerie`).
fn references_faer_crate(src: &str) -> bool {
    let bytes = src.as_bytes();
    let needle = "faer";
    let mut from = 0usize;
    while let Some(rel) = src[from..].find(needle) {
        let at = from + rel;
        let after = at + needle.len();
        let before_ok = at == 0 || !is_ident_byte(bytes[at - 1]);
        // The token must be followed by `::` (path use) or end-of-ident so that
        // `faer::sparse`, `faer::Mat`, or a bare `use faer;` all count, while
        // `faerie` (ident byte after) and `something_faer_x` do not.
        let next = bytes.get(after).copied().unwrap_or(b' ');
        let path_sep = next == b':' && bytes.get(after + 1).copied() == Some(b':');
        let word_boundary = !is_ident_byte(next);
        if before_ok && (path_sep || word_boundary) {
            return true;
        }
        from = after;
    }
    false
}

/// Does the `[dependencies]` / `[dev-dependencies]` section of `cargo_toml`
/// declare a `faer` dependency? Looks for a `faer` key (`faer = …` or
/// `faer.workspace = …` / `faer = { … }`) on a non-comment line.
fn cargo_declares_faer(cargo_toml: &str) -> bool {
    for line in cargo_toml.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            continue;
        }
        // Accept `faer = ...`, `faer.workspace = ...`, `faer = { ... }`.
        let rest = match trimmed.strip_prefix("faer") {
            Some(r) => r,
            None => continue,
        };
        let next = rest.as_bytes().first().copied().unwrap_or(b' ');
        // A real dependency key is `faer` followed by `=`, `.`, or whitespace
        // (then `=`), never another identifier byte (which would be `faer_*`).
        if next == b'=' || next == b'.' || next == b' ' || next == b'\t' {
            return true;
        }
    }
    false
}

#[test]
fn gam_predict_declares_faer_if_its_sources_reference_it() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let predict_root = Path::new(manifest).join("crates/gam-predict");
    let src_dir = predict_root.join("src");

    let mut sources = Vec::new();
    rust_sources(&src_dir, &mut sources);
    assert!(
        !sources.is_empty(),
        "expected to find gam-predict sources under {}",
        src_dir.display()
    );

    let mut referencing: Vec<PathBuf> = Vec::new();
    for path in &sources {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if references_faer_crate(&src) {
            referencing.push(path.clone());
        }
    }

    if referencing.is_empty() {
        // No source references `faer`; nothing to enforce.
        return;
    }

    let cargo_path = predict_root.join("Cargo.toml");
    let cargo = fs::read_to_string(&cargo_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cargo_path.display()));

    assert!(
        cargo_declares_faer(&cargo),
        "gam-predict references the `faer` crate in {} source file(s) \
         (e.g. {}) but `{}` declares no `faer` dependency. The #1521 split moved \
         faer-using tests into gam-predict without its dependency, so \
         `cargo test -p gam-predict --no-run` fails with E0433 and the \
         workspace `--lib` test phase aborts (#1567). Declare `faer` under \
         `[dev-dependencies]` (it is referenced only from `#[cfg(test)]` code), \
         matching the `0.24.0` the rest of the workspace pins.",
        referencing.len(),
        referencing[0].display(),
        cargo_path.display(),
    );
}
