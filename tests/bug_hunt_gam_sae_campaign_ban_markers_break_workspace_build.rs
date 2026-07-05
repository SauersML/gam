//! Bug hunt: the workspace does not build from a clean checkout at HEAD because
//! the latest campaign commit left `build.rs` ban-marker violations in
//! `gam-sae` production source.
//!
//! `build.rs` ships a self-defense lint gate that scans every workspace source
//! file and **aborts the build** (exit 101) when it finds any of a fixed set of
//! banned constructs: bare-`_` `let` bindings, `#[allow(...)]` / `#[expect(...)]`
//! lint escape hatches, `TODO` markers, or a `#[cfg(test)]` applied to a
//! non-`mod` `src/` item. The gate is intentional; the defect is that commit
//! `f232d6c2c412ff5ae3ea15d5152c8dcec5aed944` ("campaign: gam#2138 ...") added
//! all of the following, so the gate now fires on every fresh build:
//!
//!   crates/gam-sae/src/manifold/kronecker.rs:411  TODO(#974)
//!   crates/gam-sae/src/manifold/kronecker.rs:415  #[allow(dead_code)]
//!   crates/gam-sae/src/manifold/kronecker.rs:436  TODO(#974)
//!   crates/gam-sae/src/manifold/kronecker.rs:438  #[allow(dead_code)]
//!   crates/gam-sae/src/manifold/kronecker.rs:440  #[allow(clippy::too_many_arguments)]
//!   crates/gam-sae/src/manifold/kronecker.rs:659  let _ = m;
//!   crates/gam-sae/src/manifold/curl.rs:50        #[cfg(test)] use std::f64::consts::PI;
//!   crates/gam-sae/src/frames.rs:47               TODO(#974)
//!   crates/gam-sae/src/frames.rs:49               #[allow(dead_code)]
//!
//! The build script runs for the root `gam` crate, which the CLI, every
//! `tests/*.rs` integration target, and the `gamfit` wheel all depend on, so
//! `cargo build`, `cargo test`, and `maturin build`/`maturin develop` all abort
//! with:
//!
//!     error: crates/gam-sae/src/manifold/kronecker.rs:659: let _ = m;
//!     ...
//!     3  TODO marker
//!     4  #[allow(...)] / #[expect(...)]
//!     2  let _ binding
//!     1  #[cfg(test)] on src/ item
//!
//! While the bug is present the root crate's build script panics, so this
//! integration target cannot even be produced — `cargo test` fails before any
//! assertion runs, which is itself the symptom. Once a maintainer removes the
//! banned markers from the three files (rewiring the frames_engaged whitened
//! assembly so the `#[allow(dead_code)]` / `TODO(#974)` scaffolding is no longer
//! needed, dropping the unused `let _ = m;`, and moving the test-only `use PI`
//! inside `mod tests`), the workspace builds again and the checks below — which
//! re-scan the same three files for the same banned constructs build.rs forbids
//! — find none and the test passes, with no further edits.

use std::fs;
use std::path::{Path, PathBuf};

/// Files the offending campaign commit touched; each currently carries at least
/// one construct build.rs bans in production (non-test-module) source.
const TARGET_FILES: &[&str] = &[
    "crates/gam-sae/src/manifold/kronecker.rs",
    "crates/gam-sae/src/manifold/curl.rs",
    "crates/gam-sae/src/frames.rs",
];

/// Split a source file into its production prefix (everything before a
/// `#[cfg(test)] mod tests { ... }` block, where legitimate test-only helpers
/// are allowed to live) and return that prefix's lines with 1-based numbers.
///
/// The ban that breaks the build fires on the *production* region — every
/// offending line above sits before the file's trailing `mod tests {`.
fn production_lines(source: &str) -> Vec<(usize, &str)> {
    let mut cut = source.lines().count();
    for (idx, line) in source.lines().enumerate() {
        if line.trim_start().starts_with("mod tests") {
            cut = idx;
            break;
        }
    }
    source
        .lines()
        .enumerate()
        .take(cut)
        .map(|(i, l)| (i + 1, l))
        .collect()
}

/// Report every banned construct build.rs forbids that is present in the
/// production region of `source`, as `(line_number, description)`.
fn banned_markers(source: &str) -> Vec<(usize, String)> {
    let lines = production_lines(source);
    let mut hits = Vec::new();
    for (i, (lineno, raw)) in lines.iter().enumerate() {
        let line = raw.trim_start();

        // `#[allow(...)]` / `#![allow(...)]` / `#[expect(...)]` / `#![expect(...)]`
        if line.starts_with("#[allow(")
            || line.starts_with("#![allow(")
            || line.starts_with("#[expect(")
            || line.starts_with("#![expect(")
        {
            hits.push((*lineno, format!("lint escape hatch: `{}`", line)));
        }

        // Bare-underscore `let` binding (`let _ = ...`, `let _: T = ...`).
        let is_let_underscore = {
            let mut rest = line.strip_prefix("let ");
            if rest.is_none() {
                rest = line.strip_prefix("let\t");
            }
            rest.map(|r| {
                let r = r.trim_start();
                r.starts_with("_ ") || r.starts_with("_=") || r.starts_with("_:")
            })
            .unwrap_or(false)
        };
        if is_let_underscore {
            hits.push((*lineno, format!("discard `let _` binding: `{}`", line)));
        }

        // Bare TODO marker (build.rs bans the deferral note outright).
        if line.contains("TODO") {
            hits.push((*lineno, format!("TODO deferral marker: `{}`", line)));
        }

        // `#[cfg(test)]` gating a non-`mod` item (here: a `use`). A test-only
        // import must live *inside* `mod tests`, not at module scope.
        if line == "#[cfg(test)]" {
            if let Some((_, next_raw)) = lines.get(i + 1) {
                let next = next_raw.trim_start();
                if !next.starts_with("mod ") && !next.is_empty() {
                    hits.push((
                        *lineno,
                        format!("#[cfg(test)] on a non-mod item: next line `{}`", next),
                    ));
                }
            }
        }
    }
    hits
}

#[test]
fn gam_sae_campaign_source_is_free_of_build_banned_markers() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut report = String::new();
    let mut total = 0usize;

    for rel in TARGET_FILES {
        let path: PathBuf = root.join(Path::new(rel));
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let hits = banned_markers(&source);
        if !hits.is_empty() {
            report.push_str(&format!("\n{rel}:\n"));
            for (lineno, desc) in &hits {
                report.push_str(&format!("    :{lineno}  {desc}\n"));
                total += 1;
            }
        }
    }

    assert!(
        total == 0,
        "gam-sae production source carries {total} build.rs-banned marker(s); the \
         root `gam` crate's build script aborts on these (exit 101), so `cargo \
         build`, `cargo test`, and the `gamfit` wheel all fail to build at HEAD:\n{report}"
    );
}
