use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    fs::write(out_dir.join("lint_errors.rs"), "").expect("failed to write lint_errors.rs");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after the Unix epoch")
        .as_secs();
    println!("cargo:rustc-env=GAM_BUILD_TIMESTAMP={timestamp}");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"));

    // Outright TO-DO ban (split below to avoid self-trigger in this file).
    let needle: &str = concat!("TO", "DO");
    let mut offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_banned_marker(&manifest_dir, &manifest_dir, needle, &mut offenders);
    if !offenders.is_empty() {
        eprintln!();
        eprintln!("error: {} markers are banned. just do it now.", needle);
        eprintln!();
        for (rel, line_no, line) in &offenders {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            eprintln!("  {}:{}: {}", rel.display(), line_no, snippet);
        }
        eprintln!();
        eprintln!(
            "error: {} {} marker(s) found. just do it now.",
            offenders.len(),
            needle
        );
        std::process::exit(1);
    }

    // `#[allow(unused_*)]` and `#[allow(dead_code)]` ban. Unused-* lints
    // (unused_assignments, unused_variables, unused_imports, unused_mut,
    // unused_must_use, unused_macros, unused_doc_comments, unused_attributes,
    // unused_parens, unused_braces, ...) catch dead code or dead writes that
    // signal a logic error. `dead_code` is the same story for unused items.
    // Silencing them locally hides bugs — use or delete instead. Rewrite the
    // code so the offending binding is actually read, or remove it.
    let mut allow_offenders: Vec<(PathBuf, usize, String, String)> = Vec::new();
    scan_for_banned_allow(&manifest_dir, &manifest_dir, &mut allow_offenders);
    if !allow_offenders.is_empty() {
        eprintln!();
        eprintln!(
            "error: {} banned `#[allow(...)]` attribute(s) found. \
             `unused_*` and `dead_code` are not allowed to be silenced — \
             use the value or delete the binding.",
            allow_offenders.len()
        );
        eprintln!();
        for (rel, line_no, lint, line) in &allow_offenders {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            eprintln!("  {}:{}: [allow({})] {}", rel.display(), line_no, lint, snippet);
        }
        eprintln!();
        std::process::exit(1);
    }

    // `let _name = ...` ban (underscore-prefixed binding with a non-empty
    // suffix). Such bindings silence `unused_variables` without naming the
    // intent. Explicit alternatives exist for every legitimate case:
    //   - drop result:        `let _ = expr;` or `drop(expr);`
    //   - RAII guard scope:   `{ let g = lock(); ...use g... }` or
    //                         `let g = lock(); std::mem::drop(g);` at exit
    //   - keep alive to end:  bind with a real name and read it
    // The bare `_` pattern (`let _ = ...`, `let _: T = ...`) is fine and
    // not flagged here.
    let mut underscore_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_let_underscore_binding(&manifest_dir, &manifest_dir, &mut underscore_offenders);
    if !underscore_offenders.is_empty() {
        eprintln!();
        eprintln!(
            "error: {} `let _name = ...` binding(s) found. \
             Underscore-prefixed names silence unused-variable warnings — \
             use a real name and read it, or use `let _ = ...` / `drop(...)`.",
            underscore_offenders.len()
        );
        eprintln!();
        for (rel, line_no, line) in &underscore_offenders {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            eprintln!("  {}:{}: {}", rel.display(), line_no, snippet);
        }
        eprintln!();
        std::process::exit(1);
    }

    // Ignored-test ban: every `#[ignore]` / `#[ignore = "..."]` attribute is
    // a test the suite no longer enforces. Stop the build until they are
    // either deleted or restored to the running suite.
    let mut ignored: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_ignored_tests(&manifest_dir, &manifest_dir, &mut ignored);
    if !ignored.is_empty() {
        eprintln!();
        eprintln!(
            "error: {} `#[ignore]` test attribute(s) found.",
            ignored.len()
        );
        eprintln!(
            "       Ignored tests are silently dead. Either delete the test \
             or remove the `#[ignore]` attribute so it runs again."
        );
        eprintln!();
        for (rel, line_no, line) in &ignored {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            eprintln!("  {}:{}: {}", rel.display(), line_no, snippet);
        }
        eprintln!();
        std::process::exit(1);
    }
}

fn scan_for_banned_marker(
    root: &Path,
    dir: &Path,
    needle: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        if !content.contains(needle) {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            if line.contains(needle) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Scan for `#[allow(<lint>)]` and `#[allow(..., <lint>, ...)]`. Matches the
/// lint name as a whole word so `unused_assignments_foo` would not collide.
/// Only Rust files are inspected; build.rs is exempt (it documents the bans
/// it enforces, including the lint name).
fn scan_for_banned_allow(
    root: &Path,
    dir: &Path,
    lint: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("allow(") || !content.contains(lint) {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            // Find `allow(` then look inside the parenthesized list for `lint`
            // as a comma/whitespace-delimited token. This avoids matching
            // identifiers that merely contain the lint name as a substring
            // and also avoids matching the lint name in unrelated string
            // literals or comments.
            let bytes = line.as_bytes();
            let mut search_from = 0usize;
            let mut matched = false;
            while let Some(rel_idx) = line[search_from..].find("allow(") {
                let start = search_from + rel_idx + "allow(".len();
                // Find matching ')' on the same line.
                let Some(end_rel) = line[start..].find(')') else {
                    break;
                };
                let inside = &line[start..start + end_rel];
                for tok in inside.split(',') {
                    if tok.trim() == lint {
                        matched = true;
                        break;
                    }
                }
                if matched {
                    break;
                }
                search_from = start + end_rel + 1;
                if search_from >= bytes.len() {
                    break;
                }
            }
            if matched {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

fn scan_for_ignored_tests(root: &Path, dir: &Path, offenders: &mut Vec<(PathBuf, usize, String)>) {
    visit_files(root, dir, &mut |rel, content| {
        // Skip the build script itself: it names the `#[ignore]` attribute
        // as part of this scanner's own contract.
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        // Only Rust files carry test attributes.
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("#[ignore") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            // Match `#[ignore]` and `#[ignore = "..."]`. The `#[ignore` prefix
            // is unique enough that no non-attribute construct collides with it.
            if trimmed.starts_with("#[ignore]") || trimmed.starts_with("#[ignore =") {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Walk `dir` recursively, calling `visitor(rel_path, content)` for every
/// scannable file. Centralizes the directory-skip and extension rules.
fn visit_files(root: &Path, dir: &Path, visitor: &mut dyn FnMut(&Path, &str)) {
    let read = match fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    for entry in read.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
        if path
            .strip_prefix(root)
            .ok()
            .is_some_and(|rel| rel.starts_with("bench/runtime/pydeps"))
        {
            continue;
        }
        if name.starts_with('.')
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
        {
            continue;
        }
        if path.is_dir() {
            visit_files(root, &path, visitor);
            continue;
        }
        let ext = path.extension().and_then(OsStr::to_str).unwrap_or("");
        let basename = path.file_name().and_then(OsStr::to_str).unwrap_or("");
        let scannable = matches!(
            ext,
            "rs" | "py" | "toml" | "yml" | "yaml" | "sh" | "bash" | "json"
        ) || basename == "build.rs"
            || basename == "Makefile";
        if !scannable {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let content = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        visitor(&rel, &content);
    }
}
