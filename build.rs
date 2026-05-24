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
    let mut todo_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_banned_marker(&manifest_dir, &manifest_dir, needle, &mut todo_offenders);

    // `#[allow(unused_*)]` and `#[allow(dead_code)]` ban. Unused-* lints
    // (unused_assignments, unused_variables, unused_imports, unused_mut,
    // unused_must_use, unused_macros, unused_doc_comments, unused_attributes,
    // unused_parens, unused_braces, ...) catch dead code or dead writes that
    // signal a logic error. `dead_code` is the same story for unused items.
    // Silencing them locally hides bugs — use or delete instead.
    let mut allow_offenders: Vec<(PathBuf, usize, String, String)> = Vec::new();
    scan_for_banned_allow(&manifest_dir, &manifest_dir, &mut allow_offenders);

    // `let _...` ban (any underscore-leading let pattern). This covers
    // `let _ = expr;`, `let _: T = expr;`, `let _name = expr;`,
    // `let mut _name = expr;`. Every underscore-leading binding silences
    // the type system's unused-value feedback. Use or delete: bind a real
    // name and read it, or call the expression for its effect without a
    // binding (e.g. `drop(expr)` or just `expr;` for `Result` consumers
    // wrapped behind an explicit check).
    let mut underscore_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_let_underscore(&manifest_dir, &manifest_dir, &mut underscore_offenders);

    // Ignored-test ban: every `#[ignore]` / `#[ignore = "..."]` attribute is
    // a test the suite no longer enforces. Stop the build until they are
    // either deleted or restored to the running suite.
    let mut ignored: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_ignored_tests(&manifest_dir, &manifest_dir, &mut ignored);

    // Banned substrings in code (skipping `//` line comments and `"..."` /
    // `'..'` literals via a lexer-lite). Each entry is matched against the
    // stripped code portion of every line. Build.rs is exempt (it names the
    // needles as part of this scanner's contract).
    let mut substring_offenders: Vec<(PathBuf, usize, &'static str, String)> = Vec::new();
    scan_for_banned_substrings(&manifest_dir, &manifest_dir, &mut substring_offenders);

    // `unsafe` requires a `// SAFETY:` justification within the same line or
    // one of the 3 preceding non-blank lines. Keeps `unsafe` legal but makes
    // every use cite a reason.
    let mut unsafe_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_unsafe_without_safety(&manifest_dir, &manifest_dir, &mut unsafe_offenders);

    // `#[should_panic]` without `expected = "..."` catches any panic and
    // masks unrelated bugs. Require an `expected` string.
    let mut should_panic_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_bare_should_panic(&manifest_dir, &manifest_dir, &mut should_panic_offenders);

    let mut sections: Vec<Section> = Vec::new();

    if !todo_offenders.is_empty() {
        sections.push(Section {
            title: format!("{} marker", needle),
            rows: todo_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !allow_offenders.is_empty() {
        sections.push(Section {
            title: "#[allow(unused_* / dead_code)]".to_string(),
            rows: allow_offenders
                .iter()
                .map(|(r, l, lint, s)| (r.clone(), *l, Some(format!("allow({lint})")), s.clone()))
                .collect(),
        });
    }

    if !underscore_offenders.is_empty() {
        sections.push(Section {
            title: "let _ binding (bare `_` or `_name`)".to_string(),
            rows: underscore_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !ignored.is_empty() {
        sections.push(Section {
            title: "#[ignore] test".to_string(),
            rows: ignored
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !substring_offenders.is_empty() {
        // Group by needle label so each banned pattern gets its own section.
        let mut by_label: std::collections::BTreeMap<&'static str, Vec<(PathBuf, usize, String)>> =
            std::collections::BTreeMap::new();
        for (rel, line_no, label, line) in &substring_offenders {
            by_label
                .entry(*label)
                .or_default()
                .push((rel.clone(), *line_no, line.clone()));
        }
        for (label, rows_in) in by_label {
            sections.push(Section {
                title: label.to_string(),
                rows: rows_in
                    .into_iter()
                    .map(|(r, l, s)| (r, l, None, s))
                    .collect(),
            });
        }
    }

    if !unsafe_offenders.is_empty() {
        sections.push(Section {
            title: "unsafe without `// SAFETY:`".to_string(),
            rows: unsafe_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !should_panic_offenders.is_empty() {
        sections.push(Section {
            title: "#[should_panic] without `expected = \"...\"`".to_string(),
            rows: should_panic_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if sections.is_empty() {
        return;
    }

    render_report(&sections);
    std::process::exit(1);
}

/// One banned-pattern report section: a title plus the file/line rows that
/// matched. `tag` is an optional per-row marker (e.g. `allow(dead_code)`)
/// rendered between line number and snippet.
struct Section {
    title: String,
    rows: Vec<(PathBuf, usize, Option<String>, String)>,
}

fn render_report(sections: &[Section]) {
    let total: usize = sections.iter().map(|s| s.rows.len()).sum();
    eprintln!();
    eprintln!(
        "error: {} ban violation{} across {} rule{}",
        total,
        if total == 1 { "" } else { "s" },
        sections.len(),
        if sections.len() == 1 { "" } else { "s" },
    );
    for section in sections {
        eprintln!();
        eprintln!(
            "── {}  ({} hit{})",
            section.title,
            section.rows.len(),
            if section.rows.len() == 1 { "" } else { "s" },
        );
        for (rel, line_no, tag, line) in &section.rows {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            match tag {
                Some(t) => eprintln!("  {}:{}: [{}] {}", rel.display(), line_no, t, snippet),
                None => eprintln!("  {}:{}: {}", rel.display(), line_no, snippet),
            }
        }
    }
    eprintln!();
    eprintln!("summary:");
    for section in sections {
        eprintln!("  {:>5}  {}", section.rows.len(), section.title);
    }
    eprintln!();
}

/// Build the table of banned substrings. Each entry is matched in the
/// "stripped" portion of each line (string/char literal contents replaced by
/// spaces, `//` line comments dropped). The label is shown in the error
/// output so the offender knows which rule fired.
fn banned_substrings() -> &'static [(&'static str, &'static str)] {
    &[
        // Debug residue.
        ("dbg!(", "dbg!"),
        // Runtime "not done yet" markers — same family as the TO-DO text ban.
        ("todo!(", "todo!"),
        ("unimplemented!(", "unimplemented!"),
        // Direct panics. Tests should use `assert*!` macros which carry
        // failure context; production code should propagate `Result`.
        ("panic!(", "panic!"),
        // Memory-leak primitives.
        ("mem::forget(", "mem::forget"),
        ("Box::leak(", "Box::leak"),
        // Environment-variable reads (see feedback_no_env_vars memory).
        ("env::var(", "env::var"),
        ("env::var_os(", "env::var_os"),
        // Discarded inner values in `if let` and match arms — same family
        // as the `let _...` ban.
        ("if let Ok(_)", "if let Ok(_)"),
        ("if let Some(_)", "if let Some(_)"),
        ("if let Err(_)", "if let Err(_)"),
        ("Ok(_) =>", "Ok(_) =>"),
        ("Err(_) =>", "Err(_) =>"),
        ("Some(_) =>", "Some(_) =>"),
        // Poison-panic on lock acquisition.
        (".lock().unwrap()", ".lock().unwrap()"),
        (".read().unwrap()", ".read().unwrap()"),
        (".write().unwrap()", ".write().unwrap()"),
        // Redundant boolean comparisons.
        ("== true", "== true"),
        ("== false", "== false"),
        ("!= true", "!= true"),
        ("!= false", "!= false"),
        // Sleep hides latency / introduces flakiness.
        ("thread::sleep(", "thread::sleep"),
        ("std::thread::sleep(", "std::thread::sleep"),
    ]
}

fn scan_for_banned_substrings(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, &'static str, String)>,
) {
    let needles = banned_substrings();
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let stripped = strip_strings_and_comments(line);
            for (needle, label) in needles {
                if stripped.contains(needle) {
                    offenders.push((rel.to_path_buf(), idx + 1, *label, line.to_string()));
                }
            }
        }
    });
}

/// Walks `unsafe` keyword sites and requires a `// SAFETY:` comment within
/// the same line or one of the 3 preceding non-blank lines. The `unsafe`
/// match is by-word so identifiers like `unsafely_typed` do not fire.
fn scan_for_unsafe_without_safety(
    root: &Path,
    dir: &Path,
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
        if !content.contains("unsafe") {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let stripped = strip_strings_and_comments(line);
            if !line_has_keyword(&stripped, "unsafe") {
                continue;
            }
            if line.contains("SAFETY:") {
                continue;
            }
            // Scan up to 3 preceding non-blank lines for `// SAFETY:`.
            let mut justified = false;
            let mut seen = 0usize;
            let mut k = idx;
            while k > 0 && seen < 3 {
                k -= 1;
                let prev = lines[k];
                if prev.trim().is_empty() {
                    continue;
                }
                seen += 1;
                if prev.contains("SAFETY:") {
                    justified = true;
                    break;
                }
            }
            if !justified {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

fn scan_for_bare_should_panic(
    root: &Path,
    dir: &Path,
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
        if !content.contains("#[should_panic") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("#[should_panic]")
                || (trimmed.starts_with("#[should_panic")
                    && !trimmed.contains("expected"))
            {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Replace string-literal contents and `'...'` char-literal contents with
/// spaces, and truncate at `//` line comments. Preserves length where the
/// caller relies on column alignment; for substring search this only needs
/// to remove confusing content, which it does. Raw strings (`r"..."`,
/// `r#"..."#`) and block comments (`/* ... */`) are not modeled — the
/// scanner is a line-level heuristic.
fn strip_strings_and_comments(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    let mut in_str = false;
    let mut str_quote: u8 = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if c == b'\\' && i + 1 < bytes.len() {
                out.push(b' ');
                out.push(b' ');
                i += 2;
                continue;
            }
            if c == str_quote {
                in_str = false;
                out.push(c);
            } else {
                out.push(b' ');
            }
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            break;
        }
        if c == b'"' || c == b'\'' {
            in_str = true;
            str_quote = c;
            out.push(c);
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| line.to_string())
}

/// True if `line` contains `kw` as a whole word (boundaries on both sides
/// are non-identifier bytes or string ends).
fn line_has_keyword(line: &str, kw: &str) -> bool {
    let bytes = line.as_bytes();
    let kw_bytes = kw.as_bytes();
    if kw_bytes.is_empty() || bytes.len() < kw_bytes.len() {
        return false;
    }
    let mut i = 0usize;
    while i + kw_bytes.len() <= bytes.len() {
        if &bytes[i..i + kw_bytes.len()] == kw_bytes {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok = i + kw_bytes.len() == bytes.len()
                || !is_ident_byte(bytes[i + kw_bytes.len()]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
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

/// Scan for `#[allow(<lint>)]` / `#![allow(<lint>)]` where `<lint>` is in the
/// banned set: any token starting with `unused` (covers `unused_assignments`,
/// `unused_variables`, `unused_imports`, `unused_mut`, `unused_must_use`,
/// `unused_macros`, `unused_doc_comments`, `unused_attributes`, etc.) or
/// exactly `dead_code`. Lint names are matched as whole comma-delimited
/// tokens inside the `allow(...)` parenthesized list, so identifiers that
/// merely contain the substring are not flagged. The leading `clippy::`
/// path prefix is tolerated. Build.rs itself is exempt (it names the lints
/// as part of this scanner's own contract).
fn scan_for_banned_allow(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        if !content.contains("allow(") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            // Strip line comments (`//`, `///`, `//!`) so that doc/comment
            // text mentioning the attribute syntax is not flagged. String
            // literals containing `allow(` would still be matched, but the
            // codebase does not embed such literals.
            let code = match line.find("//") {
                Some(pos) => &line[..pos],
                None => line,
            };
            // Only match when `allow(` is part of an attribute on this
            // line: preceded somewhere by `#[` or `#![`.
            if !code.contains("#[") && !code.contains("#![") {
                continue;
            }
            let mut search_from = 0usize;
            while let Some(rel_idx) = code[search_from..].find("allow(") {
                let start = search_from + rel_idx + "allow(".len();
                let Some(end_rel) = code[start..].find(')') else {
                    break;
                };
                let inside = &code[start..start + end_rel];
                for tok in inside.split(',') {
                    let t = tok.trim().trim_start_matches("clippy::");
                    if t.starts_with("unused") || t == "dead_code" {
                        offenders.push((
                            rel.to_path_buf(),
                            idx + 1,
                            t.to_string(),
                            line.to_string(),
                        ));
                    }
                }
                search_from = start + end_rel + 1;
                if search_from >= code.len() {
                    break;
                }
            }
        }
    });
}

/// Scan for any `let _...` binding (bare `_`, `_name`, `mut _`, `mut _name`).
/// Skips lines whose `let` is inside a `//`-line-comment or a string literal.
/// Multi-line raw strings and `/* ... */` blocks are out of scope: the scanner
/// is a line-level heuristic, not a full parser. Build.rs is exempt.
fn scan_for_let_underscore(
    root: &Path,
    dir: &Path,
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
        if !content.contains("let ") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            if line_has_let_underscore(line) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// Returns true when `line` contains a `let` (optionally followed by `mut`)
/// whose pattern starts with `_`. Lexer-lite: tracks `//` line comments and
/// string literals so the keyword check does not false-fire on commentary
/// or string content.
fn line_has_let_underscore(line: &str) -> bool {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    let mut in_str = false;
    let mut str_quote: u8 = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if c == b'\\' && i + 1 < bytes.len() {
                i += 2;
                continue;
            }
            if c == str_quote {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            return false;
        }
        if c == b'"' || c == b'\'' {
            in_str = true;
            str_quote = c;
            i += 1;
            continue;
        }
        if c == b'l'
            && i + 3 <= bytes.len()
            && &bytes[i..i + 3] == b"let"
            && (i == 0 || !is_ident_byte(bytes[i - 1]))
            && i + 3 < bytes.len()
            && bytes[i + 3].is_ascii_whitespace()
        {
            let mut j = i + 3;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j + 4 <= bytes.len()
                && &bytes[j..j + 3] == b"mut"
                && bytes[j + 3].is_ascii_whitespace()
            {
                j += 3;
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
            }
            if j < bytes.len() && bytes[j] == b'_' {
                return true;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
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
