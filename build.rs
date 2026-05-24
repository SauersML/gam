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

    // `mem::transmute` requires a `// SAFETY:` justification, same shape
    // as the `unsafe` rule. Transmute is loud enough that the dedicated
    // section keeps it visible even when surrounded by other `unsafe`.
    let mut transmute_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_transmute_without_safety(&manifest_dir, &manifest_dir, &mut transmute_offenders);

    // `panic!(` in non-test code requires a `// SAFETY:` justification.
    // Mirrors the `unsafe` rule: legal but every site cites a reason.
    let mut panic_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_panic_without_safety(&manifest_dir, &manifest_dir, &mut panic_offenders);

    // `#[cfg(any())]` is permanently false, `#[cfg(all())]` permanently
    // true. Both are dead-by-construction guards in the same family as
    // `const X: bool = false;`.
    let mut dead_cfg_gate_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_dead_cfg_gates(&manifest_dir, &manifest_dir, &mut dead_cfg_gate_offenders);

    // `#[should_panic]` without `expected = "..."` catches any panic and
    // masks unrelated bugs. Require an `expected` string.
    let mut should_panic_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_bare_should_panic(&manifest_dir, &manifest_dir, &mut should_panic_offenders);

    // `eprintln!` / `eprint!` carrying `{:?}` / `{:#?}` debug formatting is
    // a hand-rolled `dbg!`. Test-aware; build.rs exempt.
    let mut debug_eprintln_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_debug_eprintln(&manifest_dir, &manifest_dir, &mut debug_eprintln_offenders);

    // Bare `const <ident>: bool = <literal>;` items are dead-by-construction
    // guards (rustc's `dead_code` cannot prove unreachability through them)
    // or constant truths that should not exist. Real toggles belong in
    // `cfg`/`feature`/runtime config.
    let mut dead_guard_const_offenders: Vec<(PathBuf, usize, String)> = Vec::new();
    scan_for_dead_guard_consts(
        &manifest_dir,
        &manifest_dir,
        &mut dead_guard_const_offenders,
    );

    // Persistent unimplemented/todo/unreachable removal audit. Compares the
    // current set of marker-bearing functions against the on-disk ledger and
    // flags any function whose marker disappeared without a real implementation
    // taking its place (trivial body, or replaced by another panic-shape
    // macro). The ledger is rewritten in place; pruning happens automatically
    // for legitimately-removed sites. Build.rs cannot reach the section
    // builder below without finishing this call, so the audit's ledger
    // write always lands even when other scanners report offenders.
    let mut history_violations: Vec<(PathBuf, usize, String, String)> = Vec::new();
    run_unimplemented_history_audit(&manifest_dir, &mut history_violations);

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

    if !transmute_offenders.is_empty() {
        sections.push(Section {
            title: "mem::transmute without `// SAFETY:` justification".to_string(),
            rows: transmute_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !panic_offenders.is_empty() {
        sections.push(Section {
            title: "panic!() in non-test code without `// SAFETY:` justification".to_string(),
            rows: panic_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !dead_cfg_gate_offenders.is_empty() {
        sections.push(Section {
            title:
                "#[cfg(any())]/#[cfg(all())] (empty-arg cfg — permanently false/true)"
                    .to_string(),
            rows: dead_cfg_gate_offenders
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

    if !debug_eprintln_offenders.is_empty() {
        sections.push(Section {
            title:
                "eprintln!/eprint! with {:?} debug formatting (use real logging or delete)"
                    .to_string(),
            rows: debug_eprintln_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !dead_guard_const_offenders.is_empty() {
        sections.push(Section {
            title:
                "const <name>: bool = <literal> (dead-by-construction guard — use cfg or delete)"
                    .to_string(),
            rows: dead_guard_const_offenders
                .iter()
                .map(|(r, l, s)| (r.clone(), *l, None, s.clone()))
                .collect(),
        });
    }

    if !history_violations.is_empty() {
        sections.push(Section {
            title: "unimplemented!/todo!/unreachable! removed without real implementation \
                    (history audit — ban_history.txt)"
                .to_string(),
            rows: history_violations
                .iter()
                .map(|(file, line, sig, reason)| {
                    (file.clone(), *line, Some(reason.clone()), sig.clone())
                })
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

/// Build the table of banned substrings. Each entry is `(needle, label,
/// test_aware)` matched against the "stripped" portion of every line
/// (string/char literal contents replaced by spaces, `//` line comments
/// dropped). When `test_aware` is true the match is suppressed in test
/// scope (test directories or inside `#[cfg(test)]` blocks).
fn banned_substrings() -> &'static [(&'static str, &'static str, bool)] {
    &[
        // Debug residue. Acceptable in test code.
        ("dbg!(", "dbg!", true),
        // Runtime "not done yet" markers — same family as the TO-DO text
        // ban. Acceptable in test scaffolding.
        ("todo!(", "todo!", true),
        ("unimplemented!(", "unimplemented!", true),
        // `unreachable!(` is a drop-in synonym for `unimplemented!(` from
        // the same panic family — propagate via `Result` or restructure so
        // the impossible branch is not expressible.
        ("unreachable!(", "unreachable!", true),
        // Direct panics are handled by `scan_for_panic_without_safety`
        // (a dedicated scanner that requires a `// SAFETY:` justification
        // for non-test panics), not by the lexical substring ban.
        // Vacuous assertions. Neither form belongs anywhere — tests use
        // `assert_eq!` / real predicates, production code uses `Result`.
        ("assert!(true)", "assert!(true)", false),
        ("assert!(false)", "assert!(false)", false),
        // Process termination bypasses `Drop`. Build.rs uses
        // `std::process::exit(1)` legitimately at end-of-report and is
        // already exempt from every scanner.
        ("std::process::exit(", "std::process::exit", false),
        ("process::exit(", "process::exit", false),
        ("std::process::abort(", "std::process::abort", false),
        ("process::abort(", "process::abort", false),
        // Library code writing to stdout pollutes downstream consumers.
        // Tests, examples, and benches legitimately print.
        ("println!(", "println!", true),
        ("print!(", "print!", true),
        // Memory-leak primitives. Tests sometimes leak intentionally.
        ("mem::forget(", "mem::forget", true),
        ("Box::leak(", "Box::leak", true),
        // Environment-variable reads (see feedback_no_env_vars memory).
        // Strict everywhere.
        ("env::var(", "env::var", false),
        ("env::var_os(", "env::var_os", false),
        // Env-var iteration is the same read-from-environment loophole as
        // `env::var(...)`; banning the singular form without these would
        // leave `env::vars().any(|(k,v)| k == "FOO" && v == "1")` as a
        // behavior-identical workaround.
        ("env::vars(", "env::vars", false),
        ("env::vars_os(", "env::vars_os", false),
        // `if let Ok/Some/Err(_) = …` — `.is_ok()` / `.is_some()` /
        // `.is_err()` are strictly cleaner. Match-arm forms `Ok(_) =>`
        // etc. are NOT banned: when the inner type is `()` (e.g.
        // `Result<(), E>` from `channel.send`), `Ok(_) =>` is the
        // idiomatic variant-only check and there is nothing to bind.
        // Tripwire — strict everywhere.
        ("if let Ok(_)", "if let Ok(_)", false),
        ("if let Some(_)", "if let Some(_)", false),
        ("if let Err(_)", "if let Err(_)", false),
        // Redundant boolean comparisons — redundant everywhere.
        ("== true", "== true", false),
        ("== false", "== false", false),
        ("!= true", "!= true", false),
        ("!= false", "!= false", false),
        // Sleep hides latency / introduces flakiness. Tests sometimes
        // need a real sleep (e.g. timing scheduler races).
        ("thread::sleep(", "thread::sleep", true),
        ("std::thread::sleep(", "std::thread::sleep", true),
        // Busy-wait primitives. `while Instant::now() < until { spin_loop() }`
        // is a worse `thread::sleep` (pins a core instead of parking).
        // Strict everywhere; legitimate uses are vanishingly rare.
        ("spin_loop()", "spin_loop", false),
        ("thread::yield_now(", "thread::yield_now", false),
        ("std::thread::yield_now(", "std::thread::yield_now", false),
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
        let mask = compute_test_mask(content, rel);
        for (idx, line) in content.lines().enumerate() {
            let stripped = strip_strings_and_comments(line);
            let in_test = mask.get(idx).copied().unwrap_or(false);
            for (needle, label, test_aware) in needles {
                if *test_aware && in_test {
                    continue;
                }
                if stripped.contains(needle) {
                    offenders.push((rel.to_path_buf(), idx + 1, *label, line.to_string()));
                }
            }
        }
    });
}

/// Compute a per-line bitmap of "is this line in test scope?". A line is in
/// test scope if either the file is a test/bench file (under `tests/`,
/// `bench/`, `benches/`, or `crates/*/tests|benches/`), or the line is
/// inside a brace block annotated with `#[cfg(test)]` / `#[cfg(all(test,
/// ...))]` / `#[cfg(any(test, ...))]`.
///
/// Brace tracking uses `strip_strings_and_comments` per line to ignore
/// braces inside string literals and `//` comments. Block comments and
/// raw strings are out of scope (same limitation as the other scanners).
fn compute_test_mask(content: &str, rel: &Path) -> Vec<bool> {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let mut mask = vec![false; n];

    // File-level test scope.
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    let file_is_test = rel_str.starts_with("tests/")
        || rel_str.starts_with("bench/")
        || rel_str.starts_with("benches/")
        || path_matches_crates_test(&rel_str);
    if file_is_test {
        for m in &mut mask {
            *m = true;
        }
        return mask;
    }

    // Brace-tracked cfg(test) regions. We maintain a stack of entry brace
    // depths: when a `#[cfg(test)]`-style attribute is seen, we wait for
    // the next `{` to open the gated block, then pop when depth returns to
    // the entry level.
    let mut depth: i32 = 0;
    // Pending attribute: when Some, the very next `{` (which may be on the
    // same line as the attribute or several lines later) opens a gated
    // block.
    let mut pending_attr = false;
    // Stack of entry depths (depth at which the gate opens; we exit when
    // depth drops back to this value).
    let mut gate_stack: Vec<i32> = Vec::new();

    for (idx, raw) in lines.iter().enumerate() {
        let stripped = strip_strings_and_comments(raw);

        // Detect cfg-test attribute on this line. Attribute syntax is
        // `#[cfg(test)]`, `#[cfg(all(test, ...))]`, `#[cfg(any(test,
        // ...))]`. We accept the attribute anywhere on the line.
        if is_cfg_test_attr_line(&stripped) {
            pending_attr = true;
        }

        // Walk braces on this line.
        let bytes = stripped.as_bytes();
        // The line counts as "in test" if the line's starting depth is
        // already inside a gate. Compute before brace walk so the
        // attribute line itself and the brace-open line are not marked
        // (they belong to enclosing scope), and the brace-close line that
        // exits the gate is also not marked. Lines strictly inside the
        // gate ARE marked.
        let inside_at_line_start = !gate_stack.is_empty();
        mask[idx] = inside_at_line_start;

        for &b in bytes {
            if b == b'{' {
                depth += 1;
                if pending_attr {
                    // The brace that just opened belongs to the cfg(test)
                    // attribute target. The gate's "entry depth" is the
                    // outer depth, i.e. depth - 1: we exit when depth
                    // drops back to that value.
                    gate_stack.push(depth - 1);
                    pending_attr = false;
                }
            } else if b == b'}' {
                if let Some(&entry) = gate_stack.last() {
                    if depth - 1 == entry {
                        gate_stack.pop();
                    }
                }
                depth -= 1;
            }
        }
    }

    mask
}

/// Match `crates/<name>/tests/...` or `crates/<name>/benches/...`.
fn path_matches_crates_test(rel: &str) -> bool {
    let Some(rest) = rel.strip_prefix("crates/") else {
        return false;
    };
    // Skip the crate name segment.
    let Some(slash) = rest.find('/') else {
        return false;
    };
    let tail = &rest[slash + 1..];
    tail.starts_with("tests/") || tail.starts_with("benches/")
}

/// Recognize `#[cfg(test)]`, `#[cfg(all(test, ...))]`, `#[cfg(any(test,
/// ...))]` on a stripped line. We match by locating `cfg(` after a `#[`
/// and inspecting the argument list for a bare `test` token (either
/// directly, or inside an `all(...)`/`any(...)` whose token list contains
/// `test`).
fn is_cfg_test_attr_line(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if bytes[i] == b'#' && bytes[i + 1] == b'[' {
            // Find `cfg(` after this.
            let rest = &stripped[i + 2..];
            // Allow `cfg_attr` to NOT match — we want `cfg(...)`.
            if let Some(pos) = rest.find("cfg(") {
                // Ensure it's `cfg(` not e.g. `cfg_attr(` (`cfg_attr`
                // would include the underscore before `(`, but `find`
                // returns the first match; check the preceding byte to
                // ensure the `c` of `cfg(` is not preceded by an
                // identifier byte).
                let abs = i + 2 + pos;
                let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
                if before_ok {
                    // Extract the parenthesized argument list (balance
                    // parens).
                    let args_start = abs + 4; // past `cfg(`
                    if let Some(end) = find_matching_paren(&bytes[args_start..]) {
                        let args = &stripped[args_start..args_start + end];
                        if cfg_args_contain_test(args) {
                            return true;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    false
}

/// Given bytes starting just *after* an opening `(`, find the index of
/// the matching closing `)`. Returns the position of the `)` relative to
/// the slice start, or None if unbalanced on this line.
fn find_matching_paren(bytes: &[u8]) -> Option<usize> {
    let mut depth: i32 = 1;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// True if the argument list inside `cfg(...)` mentions `test` either as
/// a bare token or inside an `all(...)` / `any(...)` whose own token
/// list contains `test`. The check is recursive in spirit but
/// implemented by scanning for a `test` whole-word token anywhere in the
/// flattened argument string — `cfg(all(test, foo))`, `cfg(any(foo,
/// test))`, and `cfg(test)` all reduce to "contains the bare word
/// `test`", which is the desired behavior since any of these spellings
/// gates the annotated item to the test build.
fn cfg_args_contain_test(args: &str) -> bool {
    let bytes = args.as_bytes();
    let kw = b"test";
    let mut i = 0usize;
    while i + kw.len() <= bytes.len() {
        if &bytes[i..i + kw.len()] == kw {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok =
                i + kw.len() == bytes.len() || !is_ident_byte(bytes[i + kw.len()]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
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
            // `unsafe impl Send` / `unsafe impl Sync` — the SAFETY
            // rationale belongs on the type's documentation, not the impl
            // line. Skip both spellings, whitespace-tolerant.
            if is_unsafe_marker_impl(&stripped) {
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

/// Flags `const <ident>: bool = false;` and `const <ident>: bool = true;`
/// item declarations. A bare boolean constant set to a literal is either a
/// dead-by-construction guard (`if !RUN_BENCH { return; }` patterns that
/// rustc's `dead_code` lint cannot catch because the const *could* in
/// principle be flipped) or a constant truth that does not need to exist.
/// Real toggles belong in `cfg`/`feature` gates or runtime configuration.
/// Build.rs is exempt.
fn scan_for_dead_guard_consts(
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
        if !content.contains("const ") {
            return;
        }
        for (idx, line) in content.lines().enumerate() {
            let stripped = strip_strings_and_comments(line);
            if !line_has_keyword(&stripped, "const") {
                continue;
            }
            if line_matches_bool_literal_const(&stripped) {
                offenders.push((rel.to_path_buf(), idx + 1, line.to_string()));
            }
        }
    });
}

/// True if `stripped` contains a `const <ident>: bool = <true|false>;`
/// item-style declaration (visibility prefix tolerated; whitespace flexible
/// throughout). Operates on the stripped line.
fn line_matches_bool_literal_const(stripped: &str) -> bool {
    let bytes = stripped.as_bytes();
    let kw = b"const";
    let mut i = 0usize;
    'outer: while i + kw.len() <= bytes.len() {
        if &bytes[i..i + kw.len()] != kw {
            i += 1;
            continue;
        }
        let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
        let after_ok =
            i + kw.len() < bytes.len() && !is_ident_byte(bytes[i + kw.len()]);
        if !before_ok || !after_ok {
            i += 1;
            continue;
        }
        // Skip whitespace after `const`.
        let mut j = i + kw.len();
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        // Identifier (one or more ident bytes, must start with non-digit).
        if j >= bytes.len() || !(bytes[j].is_ascii_alphabetic() || bytes[j] == b'_') {
            i += 1;
            continue;
        }
        while j < bytes.len() && is_ident_byte(bytes[j]) {
            j += 1;
        }
        // Whitespace, then `:`.
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b':' {
            i += 1;
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        // Type token `bool`.
        let bool_kw = b"bool";
        if j + bool_kw.len() > bytes.len() || &bytes[j..j + bool_kw.len()] != bool_kw {
            i += 1;
            continue;
        }
        let after = j + bool_kw.len();
        if after < bytes.len() && is_ident_byte(bytes[after]) {
            i += 1;
            continue;
        }
        j = after;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        if j >= bytes.len() || bytes[j] != b'=' {
            i += 1;
            continue;
        }
        j += 1;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        // Match `true` or `false`.
        for lit in [b"true".as_ref(), b"false".as_ref()] {
            if j + lit.len() <= bytes.len() && &bytes[j..j + lit.len()] == lit {
                let after_lit = j + lit.len();
                let bound_ok =
                    after_lit == bytes.len() || !is_ident_byte(bytes[after_lit]);
                if bound_ok {
                    // Optional whitespace then `;`.
                    let mut k = after_lit;
                    while k < bytes.len() && bytes[k].is_ascii_whitespace() {
                        k += 1;
                    }
                    if k < bytes.len() && bytes[k] == b';' {
                        return true;
                    }
                }
            }
        }
        i += 1;
        continue 'outer;
    }
    false
}

/// Flags lines where `eprintln!(` or `eprint!(` appears together with a
/// `{...:?}` / `{...:#?}` debug format spec. This is the hand-rolled `dbg!`
/// pattern: same intent (dump a value with `Debug` formatting to stderr),
/// different spelling. Test-aware via the shared `compute_test_mask`
/// machinery; build.rs is exempt.
///
/// The `eprintln!(` / `eprint!(` token check uses the stripped line (so
/// string-literal occurrences are ignored), but the `:?}` / `:#?}` probe
/// looks at the original line because `strip_strings_and_comments` blanks
/// out string-literal contents and would erase the format spec.
fn scan_for_debug_eprintln(
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
        if !content.contains("eprint") {
            return;
        }
        let mask = compute_test_mask(content, rel);
        for (idx, line) in content.lines().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let stripped = strip_strings_and_comments(line);
            if !stripped.contains("eprintln!(") && !stripped.contains("eprint!(") {
                continue;
            }
            if line.contains(":?}") || line.contains(":#?}") {
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
        let mask = compute_test_mask(content, rel);
        for (idx, line) in content.lines().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
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

// ─────────────────────────────────────────────────────────────────────────────
// Unimplemented-removal history audit.
//
// The lexical bans catch direct uses of `unimplemented!(` / `todo!(` /
// `unreachable!(` in non-test code. They do NOT catch the more subtle cheat
// of *deleting* a marker without writing real code in its place — e.g.
// `fn foo() { unimplemented!(); }` becoming `fn foo() { Err(NotImpl) }`. To
// catch that, we maintain a persistent ledger of every (file, fn-signature)
// pair that currently holds at least one marker. On each build we diff the
// ledger against the current scan: any pair that disappeared must now have a
// substantive function body, otherwise the build fails.
//
// The ledger is a tab-separated, sorted, line-oriented file at the repo root
// (`ban_history.txt`) so git merges resolve naturally as line-level diffs.
// Build.rs rewrites the file on every run; humans should not hand-edit it.
// ─────────────────────────────────────────────────────────────────────────────

const HISTORY_LEDGER_FILENAME: &str = "ban_history.txt";

/// The marker macros tracked by the audit. Same family as the lexical bans
/// for `unimplemented!` / `todo!` / `unreachable!` — every one is a runtime
/// "not done yet" panic dressed differently.
const HISTORY_MARKERS: &[&str] = &["unimplemented!(", "todo!(", "unreachable!("];

/// Macros that, if present anywhere in a function body, mean the body is
/// "still a panic" — even if the original marker was syntactically removed.
/// Includes the tracked markers themselves (catches "swap unimplemented for
/// unreachable") plus bare `panic!(` (catches "swap for a generic panic").
const HISTORY_BODY_REJECT_MACROS: &[&str] = &[
    "unimplemented!(",
    "todo!(",
    "unreachable!(",
    "panic!(",
];

/// Minimum number of substantive (non-blank, non-comment, non-pure-brace,
/// non-panic) statement lines a function body must contain to count as
/// "actually implemented" after marker removal. Trivial single-line returns
/// (`Err(...)`, `Ok(())`, `default()`) fall below this threshold and are
/// flagged — if a function legitimately has nothing to do, the call site
/// should be deleted instead of leaving a stub.
const HISTORY_MIN_SUBSTANTIVE_BODY_LINES: usize = 2;

fn run_unimplemented_history_audit(
    manifest_dir: &Path,
    violations: &mut Vec<(PathBuf, usize, String, String)>,
) {
    let ledger_path = manifest_dir.join(HISTORY_LEDGER_FILENAME);
    println!("cargo:rerun-if-changed={}", ledger_path.display());

    // Step 1: scan the current tree. For each .rs file (non-test scope only)
    // collect the (file, normalized-fn-signature) pairs that contain at least
    // one history marker, along with the kinds of markers present.
    let mut current: std::collections::BTreeMap<
        (String, String),
        (PathBuf, usize, Vec<String>),
    > = std::collections::BTreeMap::new();
    let mut file_contents: std::collections::BTreeMap<String, String> =
        std::collections::BTreeMap::new();

    visit_files(manifest_dir, manifest_dir, &mut |rel, content| {
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        if rel.extension().and_then(OsStr::to_str) != Some("rs") {
            return;
        }
        file_contents.insert(rel_str.clone(), content.to_string());

        let lines: Vec<&str> = content.lines().collect();
        let mask = compute_test_mask(content, rel);
        for (idx, line) in lines.iter().enumerate() {
            if mask.get(idx).copied().unwrap_or(false) {
                continue;
            }
            let stripped = strip_strings_and_comments(line);
            for &marker in HISTORY_MARKERS {
                if !stripped.contains(marker) {
                    continue;
                }
                if let Some((sig, (open, _close))) = find_enclosing_fn(&lines, idx) {
                    let kind = marker.trim_end_matches('(').to_string();
                    let entry = current
                        .entry((rel_str.clone(), sig.clone()))
                        .or_insert_with(|| (rel.to_path_buf(), open + 1, Vec::new()));
                    if !entry.2.contains(&kind) {
                        entry.2.push(kind);
                    }
                    break;
                }
            }
        }
    });

    // Step 2: load the previous ledger (empty on first run).
    let previous = load_history_ledger(&ledger_path);

    // Step 3: build the next ledger from the current scan. Legitimately-
    // removed entries are simply absent from the next ledger; flagged
    // entries are carried forward so future builds re-check them.
    let mut next_ledger: std::collections::BTreeMap<(String, String), Vec<String>> =
        std::collections::BTreeMap::new();
    for ((file, sig), (_, _, kinds)) in &current {
        next_ledger.insert((file.clone(), sig.clone()), kinds.clone());
    }

    for ((file, sig), prev_kinds) in &previous {
        if current.contains_key(&(file.clone(), sig.clone())) {
            continue;
        }
        let state = match file_contents.get(file) {
            None => HistoryBodyState::FnAbsent,
            Some(c) => body_state_for_signature(c, sig),
        };
        match state {
            HistoryBodyState::FnAbsent | HistoryBodyState::Substantive => {
                // Pruned: legitimate removal (whole fn deleted, or body now
                // substantive).
            }
            HistoryBodyState::Trivial {
                fn_open_line,
                snippet,
            } => {
                violations.push((
                    PathBuf::from(file),
                    fn_open_line + 1,
                    sig.clone(),
                    format!(
                        "marker(s) [{}] removed but body still trivial / panic-shaped — first body line: {}",
                        prev_kinds.join(","),
                        snippet,
                    ),
                ));
                // Keep entry alive so a real fix is required to clear it.
                next_ledger.insert((file.clone(), sig.clone()), prev_kinds.clone());
            }
        }
    }

    // Step 4: write the next ledger if it differs from disk. Avoid spurious
    // mtime churn on identical content.
    save_history_ledger(&ledger_path, &next_ledger);
}

fn load_history_ledger(
    path: &Path,
) -> std::collections::BTreeMap<(String, String), Vec<String>> {
    let mut out: std::collections::BTreeMap<(String, String), Vec<String>> =
        std::collections::BTreeMap::new();
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return out,
    };
    for line in content.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(3, '\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let file = parts[0].to_string();
        let sig = parts[1].to_string();
        let kinds: Vec<String> = parts[2]
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        out.insert((file, sig), kinds);
    }
    out
}

fn save_history_ledger(
    path: &Path,
    ledger: &std::collections::BTreeMap<(String, String), Vec<String>>,
) {
    let mut out = String::new();
    out.push_str(
        "# Persistent audit of `unimplemented!(` / `todo!(` / `unreachable!(`\n\
         # sites. Auto-managed by build.rs — do NOT hand-edit. Each non-comment\n\
         # line is tab-separated:\n\
         #   <relative_path>\\t<normalized_fn_signature>\\t<comma_marker_kinds>\n\
         # When an entry disappears from the source tree, build.rs inspects the\n\
         # enclosing function's body. Trivial bodies (under a few code lines)\n\
         # and bodies that still contain any panic-shape macro fail the build,\n\
         # so deleting a `unimplemented!` without writing a real implementation\n\
         # is caught. Legitimately-implemented removals auto-prune.\n\
         #\n\
         # Concurrent branches that both remove markers will produce additive\n\
         # line-level diff conflicts that resolve by union — the next build\n\
         # then re-validates each surviving entry.\n",
    );
    for ((file, sig), kinds) in ledger {
        out.push_str(file);
        out.push('\t');
        out.push_str(sig);
        out.push('\t');
        out.push_str(&kinds.join(","));
        out.push('\n');
    }
    if let Ok(existing) = fs::read_to_string(path) {
        if existing == out {
            return;
        }
    }
    if fs::write(path, out).is_err() {
        // Non-fatal: write failures (e.g. read-only checkout) leave the
        // ledger stale rather than failing the build. Future runs retry.
    }
}

enum HistoryBodyState {
    FnAbsent,
    Trivial {
        fn_open_line: usize,
        snippet: String,
    },
    Substantive,
}

/// Walk the file looking for a function whose normalized signature matches
/// `target_sig`. Returns the body shape: absent, trivial, or substantive.
/// "Substantive" requires at least `HISTORY_MIN_SUBSTANTIVE_BODY_LINES`
/// non-blank, non-comment, non-pure-brace code lines AND no occurrence of
/// any `HISTORY_BODY_REJECT_MACROS` macro in the body.
fn body_state_for_signature(content: &str, target_sig: &str) -> HistoryBodyState {
    let lines: Vec<&str> = content.lines().collect();
    let mut idx = 0usize;
    while idx < lines.len() {
        let stripped = strip_strings_and_comments(lines[idx]);
        if !line_has_keyword(&stripped, "fn") {
            idx += 1;
            continue;
        }
        if let Some((sig, (open, close))) = find_fn_body_at(&lines, idx) {
            if sig == target_sig {
                let mut code_lines = 0usize;
                let mut first_snippet: Option<String> = None;
                for j in open..=close {
                    let raw = lines[j];
                    let s = strip_strings_and_comments(raw);
                    let t = s.trim();
                    if t.is_empty() {
                        continue;
                    }
                    if t.chars().all(|c| matches!(c, '{' | '}' | ' ')) {
                        continue;
                    }
                    for &m in HISTORY_BODY_REJECT_MACROS {
                        if s.contains(m) {
                            return HistoryBodyState::Trivial {
                                fn_open_line: open,
                                snippet: raw
                                    .trim()
                                    .chars()
                                    .take(120)
                                    .collect::<String>(),
                            };
                        }
                    }
                    if first_snippet.is_none() {
                        first_snippet =
                            Some(raw.trim().chars().take(120).collect::<String>());
                    }
                    code_lines += 1;
                }
                if code_lines < HISTORY_MIN_SUBSTANTIVE_BODY_LINES {
                    return HistoryBodyState::Trivial {
                        fn_open_line: open,
                        snippet: first_snippet
                            .unwrap_or_else(|| "<empty body>".to_string()),
                    };
                }
                return HistoryBodyState::Substantive;
            }
            idx = close + 1;
            continue;
        }
        idx += 1;
    }
    HistoryBodyState::FnAbsent
}

/// Walk backward from `at_line` looking for the most recent line that bears
/// a `fn` keyword AND whose resulting body braces enclose `at_line`. Returns
/// `(normalized_signature, (body_open_line, body_close_line))`.
fn find_enclosing_fn(
    lines: &[&str],
    at_line: usize,
) -> Option<(String, (usize, usize))> {
    let mut start = at_line + 1;
    while start > 0 {
        start -= 1;
        let stripped = strip_strings_and_comments(lines[start]);
        if !line_has_keyword(&stripped, "fn") {
            continue;
        }
        if let Some((sig, (open, close))) = find_fn_body_at(lines, start) {
            if open <= at_line && at_line <= close {
                return Some((sig, (open, close)));
            }
        }
    }
    None
}

/// Starting at `fn_line` (a line containing the `fn` keyword), find the
/// function body's opening `{` and matching `}`. Brace counting uses
/// `strip_strings_and_comments` per line so braces inside string literals
/// or `//` comments don't perturb depth. Block comments and raw strings
/// are out of scope (same limitation as the other scanners). Returns the
/// normalized signature text — everything from `fn_line` up to (and
/// excluding) the body's opening `{`, whitespace-collapsed.
fn find_fn_body_at(
    lines: &[&str],
    fn_line: usize,
) -> Option<(String, (usize, usize))> {
    let mut depth: i32 = 0;
    let mut body_open: Option<usize> = None;
    for j in fn_line..lines.len() {
        let s = strip_strings_and_comments(lines[j]);
        for b in s.bytes() {
            match b {
                b'{' => {
                    depth += 1;
                    if body_open.is_none() {
                        body_open = Some(j);
                    }
                }
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(open) = body_open {
                            let mut sig = String::new();
                            for k in fn_line..=open {
                                let ss = strip_strings_and_comments(lines[k]);
                                let cut = if k == open {
                                    ss.find('{').unwrap_or(ss.len())
                                } else {
                                    ss.len()
                                };
                                sig.push_str(&ss[..cut]);
                                sig.push(' ');
                            }
                            let normalized: String =
                                sig.split_whitespace().collect::<Vec<_>>().join(" ");
                            return Some((normalized, (open, j)));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}
