//! Bug hunt: the workspace does not build from a clean checkout at HEAD.
//!
//! `build.rs` enforces a crate-wide ban on lint-silencing attributes: every
//! `#[allow(...)]` / `#![allow(...)]` / `#[expect(...)]` / `#![expect(...)]`
//! anywhere in the tree is a violation (`scan_for_banned_allow`, build.rs).
//! The stated contract (build.rs:415) is:
//!
//!   "#[allow(...)] / #[expect(...)] (any lint, anywhere — fix the underlying
//!    code instead of silencing the lint)"
//!
//! Commit 7e8545f20 ("fix(#1026/#1154): land the encode.rs basin-warmup")
//! reintroduced a silencer at `crates/gam-sae/src/encode.rs:1102`:
//!
//!     #[allow(clippy::too_many_arguments)]
//!     fn certify_with_basin_warmup(...)
//!
//! On any state that forces `build.rs` to re-run (a clean build — `cargo
//! build`, `cargo test`, a fresh `--release` build, or the `maturin` wheel
//! build all do), the attribute audit fires and the build script calls
//! `std::process::exit(1)` (build.rs:795):
//!
//!     error: #[allow(...)] / #[expect(...)] (any lint, anywhere ...)  (1 hit)
//!       error: crates/gam-sae/src/encode.rs:1102: #[allow(clippy::too_many_arguments)]
//!     cargo:warning=ban-scanner FAILED: 1 violation(s) across 1 rule(s); build aborted
//!
//! so `cargo build` / `cargo test` exit 1 and the `gamfit` wheel cannot be
//! built. (A *cached* `target/` can hide it: when the build-script fingerprint
//! is still valid the script is skipped and a stale incremental build appears
//! to succeed — which is how this rode onto `main`. A fresh build, the wheel
//! build, and CI all re-run it and fail.)
//!
//! While the bug is present the crate cannot build, so this integration-test
//! target cannot even be produced — `cargo test` fails. Once the silencer is
//! removed (split the function's arguments into a struct, or otherwise make the
//! `too_many_arguments` lint not fire, per the ban's own remedy), the workspace
//! builds again and the check below — which re-implements the same
//! attribute-context detection `build.rs` uses, over every `.rs` file in the
//! tree — finds no silencer and the test passes, with no further edits.
//!
//! The check deliberately masks `//`-line-comment text so a *mention* of the
//! attribute syntax inside a comment (e.g. the documented
//! `// without an #[allow(unused_variables)] suppression.` in
//! `crates/gam-solve/src/gpu_kernels/sigma_cubature.rs`) is not flagged, exactly
//! as `build.rs`'s `strip_file_lines` masking does.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// Directory names `build.rs::collect_scannable_files` skips.
fn is_skipped_dir(name: &str) -> bool {
    (name.starts_with('.') && name != ".github")
        || name == "target"
        || name.starts_with("target-")
        || name == "node_modules"
        || name == "__pycache__"
        || name == "pydeps"
        || name == "site-packages"
        || name == "venv"
        || name == "dist"
        || name == "build"
        || name == "site"
}

/// Collect every scannable `.rs` file under `dir`, recursively, skipping the
/// same directories build.rs does. `build.rs` itself is exempt from the ban.
fn rust_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
        if path.is_dir() {
            if !is_skipped_dir(name) {
                rust_files(root, &path, out);
            }
            continue;
        }
        if path.extension().and_then(OsStr::to_str) != Some("rs") {
            continue;
        }
        // build.rs names the attribute syntax as part of the scanner's own
        // contract and is exempt (build.rs:scan_for_banned_allow).
        if let Ok(rel) = path.strip_prefix(root) {
            if rel == Path::new("build.rs") {
                continue;
            }
        }
        out.push(path);
    }
}

/// Strip a `//` line comment (everything from the first `//`), so a *mention*
/// of `#[allow(...)]` inside comment prose is not counted — matching build.rs's
/// `strip_file_lines` comment masking for this line-level check.
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
fn source_tree_has_no_lint_silencing_allow_or_expect_attributes() {
    // `CARGO_MANIFEST_DIR` is the workspace root (the `gam` crate lives there).
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let mut files = Vec::new();
    rust_files(&root, &root, &mut files);
    assert!(
        !files.is_empty(),
        "no .rs files found under {}",
        root.display()
    );

    let mut offenders: Vec<String> = Vec::new();
    for file in &files {
        let content = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if !content.contains("allow(") && !content.contains("expect(") {
            continue;
        }
        for (idx, line) in content.lines().enumerate() {
            let code = strip_line_comment(line);
            if line_has_silencer(code) {
                let rel = file
                    .strip_prefix(&root)
                    .unwrap_or(file)
                    .to_string_lossy()
                    .into_owned();
                offenders.push(format!("{rel}:{}: {}", idx + 1, line.trim()));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "lint-silencing allow/expect attribute(s) present; build.rs's \
         scan_for_banned_allow aborts the whole workspace build (exit 1) on these — \
         `cargo build`, `cargo test`, and the gamfit wheel all fail: {offenders:?}"
    );
}
