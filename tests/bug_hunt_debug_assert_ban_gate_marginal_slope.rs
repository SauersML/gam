//! Regression guard for issue #530: `debug_assert!{,_eq,_ne}!` must never
//! reappear anywhere under `src/`.
//!
//! Background: the `debug_assert*` family compiles to nothing in release, so
//! any shape/finiteness invariant guarded by one is *silently unchecked* in
//! the shipped wheels (`maturin build --release`) — the exact failure mode the
//! `build.rs` ban-gate exists to prevent. #461 introduced three
//! `debug_assert_eq!` calls in `src/families/marginal_slope_orthogonal.rs`
//! (`influence_block_design` / `residualize_influence_columns`); they were
//! converted to release-active `assert_eq!`.
//!
//! Why this test exists in addition to the build-script gate: `build.rs` only
//! re-runs its scan when `build.rs` or `src/terms/penalties` change
//! (`cargo:rerun-if-changed`), so its verdict *caches* — a `debug_assert*`
//! added elsewhere in `src/` on a warm `target/` would not re-trip the gate
//! until something it watches changes. This `#[test]` re-scans the whole `src/`
//! tree on every `cargo test`, independent of that cache, so a regression is
//! caught from a different angle than the build-time gate.
//!
//! The banned needles are assembled with `concat!` so this test file does not
//! itself contain the literal substring the gate (and this test) ban — the
//! gate scans test files too for this family (`test_aware = false`).

use std::fs;
use std::path::{Path, PathBuf};

/// The three banned needles, each split so the literal never appears verbatim
/// in this source file. Mirrors `build.rs`'s `debug_assert*` ban entries.
fn banned_needles() -> [String; 3] {
    [
        concat!("debug_", "assert!(").to_string(),
        concat!("debug_", "assert_eq!(").to_string(),
        concat!("debug_", "assert_ne!(").to_string(),
    ]
}

/// Produce a "code-only" view of a line: blank out the contents of `"..."` and
/// `'...'` literals and drop a trailing `//` line comment, so a banned needle
/// that appears *inside a string or comment* (e.g. an error message or doc line
/// that names the macro) is not a false positive. This is a faithful, minimal
/// version of the gate's `strip_strings_and_comments`; block comments and raw
/// strings are out of scope, exactly as the gate documents.
fn strip_code_only(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_str = false;
    let mut in_char = false;
    while let Some(c) = chars.next() {
        if in_str {
            if c == '\\' {
                // Skip the escaped char so an escaped quote does not close.
                out.push(' ');
                if chars.next().is_some() {
                    out.push(' ');
                }
            } else if c == '"' {
                in_str = false;
                out.push('"');
            } else {
                out.push(' ');
            }
            continue;
        }
        if in_char {
            if c == '\\' {
                out.push(' ');
                if chars.next().is_some() {
                    out.push(' ');
                }
            } else if c == '\'' {
                in_char = false;
                out.push('\'');
            } else {
                out.push(' ');
            }
            continue;
        }
        match c {
            '"' => {
                in_str = true;
                out.push('"');
            }
            '\'' => {
                in_char = true;
                out.push('\'');
            }
            '/' if chars.peek() == Some(&'/') => {
                // Rest of the line is a comment.
                break;
            }
            _ => out.push(c),
        }
    }
    out
}

/// Recursively collect `.rs` files under `dir`, skipping build artifact dirs.
fn collect_rs_files(dir: &Path, acc: &mut Vec<PathBuf>) {
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if name.starts_with('.') || name == "target" || name.starts_with("target-") {
            continue;
        }
        if path.is_dir() {
            collect_rs_files(&path, acc);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            acc.push(path);
        }
    }
}

#[test]
fn no_debug_assert_family_anywhere_in_src() {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    assert!(
        src_root.is_dir(),
        "src/ tree not found at {}",
        src_root.display()
    );

    let needles = banned_needles();
    let mut files = Vec::new();
    collect_rs_files(&src_root, &mut files);
    assert!(
        !files.is_empty(),
        "scanned 0 .rs files under {} — traversal is broken, not a clean tree",
        src_root.display()
    );

    let mut offenders: Vec<String> = Vec::new();
    for path in &files {
        let content = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for (idx, raw) in content.lines().enumerate() {
            let code = strip_code_only(raw);
            for needle in &needles {
                if code.contains(needle.as_str()) {
                    offenders.push(format!("{}:{}: {}", path.display(), idx + 1, raw.trim()));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "found {} banned debug_assert* call(s) under src/ \
         (they compile to nothing in release — make them release-active \
         `assert*!`/`Result`, or delete them):\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}
