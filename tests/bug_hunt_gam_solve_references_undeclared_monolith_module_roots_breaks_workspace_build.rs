//! Regression (sibling of the gam-terms carve breakage, see Related issue):
//! the carved `gam-solve` crate also references **module roots of the old
//! monolithic `gam` crate** that it does not provide at its own crate root, so
//! it cannot compile standalone. This failure is currently *latent* — the
//! workspace build aborts earlier on `gam-terms` (a `gam-solve` dependency), so
//! `cargo` never reaches `gam-solve` — but once `gam-terms` is fixed this is the
//! next domino: `cargo build --workspace` / the `gamfit` wheel build will then
//! fail in `gam-solve`.
//!
//! # What's wrong
//!
//! The #1521 carve moved engine source into `crates/gam-solve/src/`, but those
//! files still address sibling subsystems through `crate::<module>` paths that
//! only resolve inside the monolithic `gam` crate. `gam-solve`'s crate root
//! (`lib.rs`, which `include!`s `mod.rs`) declares none of these snake_case
//! module roots, yet the sources contain plain `use crate::<module>::…` /
//! `crate::<module>::…` references to them:
//!
//! ```ignore
//! use crate::probability::{ … };                                  // mixture_link.rs
//! let psis = crate::psis::pareto_smooth_weights(&raw_influence);   // objective.rs
//! use crate::terms::sae::manifold::{ … };                         // continuation_path.rs
//! -> crate::inference::certificates::Verdict                       // topology_selector.rs
//! -> crate::rho_prior_eval::RhoPriorEval                           // gradient_hessian.rs
//! pub(crate) rho_uncertainty_problem_size: crate::rho_uncertainty::RhoUncertaintyProblemSize  // run.rs
//! ```
//!
//! Each of `probability`, `psis`, `terms`, `inference`, `rho_prior_eval`,
//! `rho_uncertainty` is a crate-root item of the monolithic `gam` crate
//! (`src/psis.rs`, `src/inference/…`, `src/rho_prior_eval.rs`,
//! `src/rho_uncertainty.rs`, the `terms` alias, …), not of `gam-solve`. These
//! are bare `use`/type/call paths — they cannot be satisfied by a macro import
//! or a glob, so they are genuine `E0432`/`E0433` errors. The correct fix
//! finishes the carve: repoint them at the crates that now own those subsystems
//! (`gam_math::probability`, `gam_problem::…`, `gam_terms::…`, …) so the
//! `crate::`-rooted module references disappear.
//!
//! # The invariant this test pins
//!
//! Every `crate::<snake_case_module>::…` path used in a `gam-solve` library
//! source must resolve to a module the crate provides at its own root (declared
//! or re-exported in `lib.rs` / its `include!`d root module). The check is
//! deliberately narrowed to snake_case *module* paths in real (non-comment,
//! non-`$crate`, non-macro-call) code, so it has no macro/type false positives
//! (verified: the same logic reports zero leaks for the healthy `gam-models`
//! crate). It computes the provided set from the crate's root module text, so it
//! stays correct as the carve progresses; it currently reports the six leaks
//! above and empties once they are repointed.
//!
//! This test lives in the `gam` crate and inspects the carved sources
//! statically, so it does not require `gam-solve` to compile.

use std::collections::BTreeMap;
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

/// Concatenated text of the crate root module: `lib.rs` plus any file it pulls
/// in via `include!("…")` at the top level (gam-solve's `lib.rs` is just
/// `include!("mod.rs")`).
fn root_module_text(src_dir: &Path) -> String {
    let lib = src_dir.join("lib.rs");
    let mut text =
        fs::read_to_string(&lib).unwrap_or_else(|e| panic!("cannot read {}: {e}", lib.display()));
    // Naive `include!("rel/path.rs")` scan.
    let needle = "include!";
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(needle) {
        let at = from + rel;
        // Find the quoted argument after `include!`.
        if let Some(q1) = text[at..].find('"') {
            let s = at + q1 + 1;
            if let Some(q2) = text[s..].find('"') {
                let inc_rel = text[s..s + q2].to_string();
                let inc_path = src_dir.join(&inc_rel);
                if let Ok(extra) = fs::read_to_string(&inc_path) {
                    text.push('\n');
                    text.push_str(&extra);
                }
                from = s + q2 + 1;
                continue;
            }
        }
        from = at + needle.len();
    }
    text
}

fn identifiers(src: &str) -> Vec<String> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            out.push(src[start..i].to_string());
        } else {
            i += 1;
        }
    }
    out
}

/// First segment of every `crate::<seg>::` occurrence in a code line where
/// `<seg>` is snake_case (a module name) — excluding `$crate::…` and macro
/// calls (the `::` requirement already excludes `crate::name!(…)`).
fn crate_module_roots(code: &str) -> Vec<String> {
    let bytes = code.as_bytes();
    let needle = "crate::";
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = code[from..].find(needle) {
        let at = from + rel;
        // Reject `$crate::` and `my_crate::` (ident/`$` byte right before).
        let prev_ok = at == 0 || {
            let p = bytes[at - 1];
            !(p.is_ascii_alphanumeric() || p == b'_' || p == b'$')
        };
        let seg_start = at + needle.len();
        let mut j = seg_start;
        while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
            j += 1;
        }
        // Require a following `::` so this is a module path, not a leaf item or
        // a `crate::name!` macro call.
        let followed_by_path = code[j..].starts_with("::");
        let seg = &code[seg_start..j];
        let snake = !seg.is_empty()
            && seg
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            && seg
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_lowercase() || c == '_');
        if prev_ok && followed_by_path && snake {
            out.push(seg.to_string());
        }
        from = seg_start.max(at + needle.len());
    }
    out
}

#[test]
fn gam_solve_only_references_module_roots_it_provides() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src_dir = Path::new(manifest).join("crates/gam-solve/src");

    let provided: std::collections::BTreeSet<String> = identifiers(&root_module_text(&src_dir))
        .into_iter()
        .collect();

    let mut sources = Vec::new();
    rust_sources(&src_dir, &mut sources);
    assert!(
        !sources.is_empty(),
        "expected gam-solve sources under {}",
        src_dir.display()
    );

    let mut undeclared: BTreeMap<String, (PathBuf, String)> = BTreeMap::new();
    for path in &sources {
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.contains("test") {
            continue;
        }
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        for line in src.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("///") || trimmed.starts_with("//!") || trimmed.starts_with("//")
            {
                continue;
            }
            let code = line.split("//").next().unwrap_or(line);
            for root in crate_module_roots(code) {
                if !provided.contains(&root) {
                    undeclared
                        .entry(root)
                        .or_insert_with(|| (path.clone(), line.trim().to_string()));
                }
            }
        }
    }

    assert!(
        undeclared.is_empty(),
        "gam-solve references {} crate-root module path(s) it does not provide, so \
         the crate cannot compile standalone — the next workspace-build domino \
         after the gam-terms carve breakage. The #1521 carve left these \
         monolith-only module roots addressed as `crate::…`:\n{}\nRepoint each at \
         the crate that now owns it (gam_math::, gam_problem::, gam_terms::, …).",
        undeclared.len(),
        undeclared
            .iter()
            .map(|(root, (path, line))| format!(
                "  - `crate::{root}` ({}): {}",
                path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
                line
            ))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}
