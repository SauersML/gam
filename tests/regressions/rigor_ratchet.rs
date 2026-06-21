//! Rigor ratchet — track aggregate rigor metrics as committed state so that any
//! silent weakening of the codebase's strictness becomes a visible, reviewable
//! ledger change rather than a hidden side effect of a "graceful fix" commit.
//!
//! CONCEPT
//! -------
//! Instead of trying to detect every individual hack, this test tracks a small
//! set of AGGREGATE rigor metrics (counts of panic guards, asserts, error
//! returns, build.rs ban gates, and a NaN-emission ceiling) as numbers stored
//! in the committed ledger `rigor_ledger.txt`. Removing a guard, deleting an
//! assertion, laundering an error into a silent fallback, or dropping a ban
//! gate all NECESSARILY drop one of these metrics. The ratchet asserts the
//! metrics never silently decrease (and the NaN ceiling never silently rises).
//!
//! To legitimately lower a floor metric (or raise the NaN ceiling) you MUST edit
//! `rigor_ledger.txt`. That edit shows up in the diff — a reviewer sees
//! `panic_guards = 194 -> 192` and asks why — instead of the weakening hiding
//! inside an unrelated change. Strictness is thereby made MONOTONE over time:
//! it can only drop deliberately and visibly, never silently.
//!
//! METRICS (computed from the NON-test `src/` tree; test scopes excluded):
//!   - panic_guards   = occurrences of `panic!(`                       (floor)
//!   - asserts        = `assert!(` + `assert_eq!(` + `assert_ne!(`     (floor)
//!   - result_guards  = occurrences of `return Err(`                   (floor)
//!   - ban_gates      = `fn scan_for_` in build.rs + the self-tamper
//!                      gate `forbid_build_rs_self_tampering`          (floor)
//!   - nan_emissions  = occurrences of `f64::NAN`                    (CEILING)
//!
//! The test recomputes each metric from the working tree using the SAME
//! test-scope filter that was used to seed the ledger, and fails if a floor
//! metric is below its ledger value or the NaN ceiling is above its ledger
//! value. If the tree cannot be read (e.g. packaged build without sources),
//! the test skips gracefully with an eprintln.

use std::fs;
use std::path::{Path, PathBuf};

/// Repository root, derived from this crate's manifest dir.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Recursively collect every `.rs` file under `dir`, sorted for determinism.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// `true` iff the line contains a `mod tests {` / `mod test {` opener.
///
/// Mirrors the seeding regex `\bmod\s+tests?\b.*\{` with a hand-rolled scan so
/// the test has no external dependencies.
fn line_opens_mod_test(line: &str) -> bool {
    // Find a `mod` token bounded by non-identifier chars.
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while let Some(pos) = line[i..].find("mod") {
        let start = i + pos;
        let end = start + 3;
        let before_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_ident_byte(bytes[end]);
        if before_ok && after_ok {
            // Skip whitespace after `mod`.
            let mut j = end;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            // Match `test` then optional `s`.
            let rest = &line[j..];
            if rest.starts_with("test") {
                let after_kw_idx = j + 4;
                let mut k = after_kw_idx;
                if k < bytes.len() && bytes[k] == b's' {
                    k += 1;
                }
                // Token must end (word boundary) and a `{` must appear later.
                let boundary = k >= bytes.len() || !is_ident_byte(bytes[k]);
                if boundary && line[k..].contains('{') {
                    return true;
                }
            }
        }
        i = end;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b == b'_' || b.is_ascii_alphanumeric()
}

#[derive(Default, Clone, Copy)]
struct Counts {
    panic_guards: u64,
    asserts: u64,
    result_guards: u64,
    nan_emissions: u64,
}

/// Count metrics on the NON-test lines of one source file.
///
/// Test scopes are skipped via a brace-depth state machine: a scope opens on a
/// line that either (a) follows after `#[cfg(test)]` was seen and contains an
/// opening brace, or (b) is a `mod tests {` / `mod test {` opener. The scope
/// closes when brace depth returns to the level recorded at the opener. This is
/// byte-for-byte the same algorithm used to seed `rigor_ledger.txt`.
fn count_non_test(text: &str, counts: &mut Counts) {
    let mut in_test = false;
    let mut test_depth: i64 = 0;
    let mut depth: i64 = 0;
    let mut armed = false; // saw `#[cfg(test)]`, awaiting the block's opening brace

    for line in text.split('\n') {
        let opens = line.matches('{').count() as i64;
        let closes = line.matches('}').count() as i64;

        if !in_test {
            if line.contains("#[cfg(test)]") {
                armed = true;
            }
            let is_mod_test = line_opens_mod_test(line);
            if (armed && opens > 0) || is_mod_test {
                in_test = true;
                test_depth = depth;
                armed = false;
                depth += opens - closes;
                continue;
            }
            counts.panic_guards += line.matches("panic!(").count() as u64;
            counts.asserts += line.matches("assert!(").count() as u64
                + line.matches("assert_eq!(").count() as u64
                + line.matches("assert_ne!(").count() as u64;
            counts.result_guards += line.matches("return Err(").count() as u64;
            counts.nan_emissions += line.matches("f64::NAN").count() as u64;
            depth += opens - closes;
        } else {
            depth += opens - closes;
            if depth <= test_depth {
                in_test = false;
            }
        }
    }
}

/// Count build.rs ban gates: `fn scan_for_` plus `forbid_build_rs_self_tampering`.
fn count_ban_gates(build_rs: &str) -> u64 {
    build_rs.matches("fn scan_for_").count() as u64
        + build_rs.matches("forbid_build_rs_self_tampering").count() as u64
}

/// Parse a `name = N` line from the ledger.
fn parse_ledger(text: &str, name: &str) -> Option<u64> {
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == name {
                return v.trim().parse::<u64>().ok();
            }
        }
    }
    None
}

#[test]
pub(crate) fn rigor_does_not_silently_regress() {
    let root = repo_root();
    let src_dir = root.join("src");
    let ledger_path = root.join("rigor_ledger.txt");
    let build_rs_path = root.join("build.rs");

    if !src_dir.is_dir() {
        eprintln!(
            "rigor_ratchet: skipping — src/ not readable at {} (packaged build?)",
            src_dir.display()
        );
        return;
    }

    let ledger_text = match fs::read_to_string(&ledger_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "rigor_ratchet: skipping — cannot read ledger {}: {e}",
                ledger_path.display()
            );
            return;
        }
    };

    // Recompute src-derived metrics.
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    if files.is_empty() {
        eprintln!("rigor_ratchet: skipping — no .rs files found under src/");
        return;
    }
    let mut counts = Counts::default();
    for path in &files {
        match fs::read_to_string(path) {
            Ok(text) => count_non_test(&text, &mut counts),
            Err(e) => {
                eprintln!(
                    "rigor_ratchet: skipping — cannot read source {}: {e}",
                    path.display()
                );
                return;
            }
        }
    }

    let build_rs = match fs::read_to_string(&build_rs_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "rigor_ratchet: skipping — cannot read {}: {e}",
                build_rs_path.display()
            );
            return;
        }
    };
    let ban_gates = count_ban_gates(&build_rs);

    // Compare each metric to the ledger. Floor metrics must not be BELOW the
    // ledger; the NaN ceiling must not be ABOVE the ledger.
    let mut failures: Vec<String> = Vec::new();

    let mut check_floor = |name: &str, current: u64| {
        match parse_ledger(&ledger_text, name) {
            Some(floor) => {
                if current < floor {
                    failures.push(format!(
                        "{name}: {floor} -> {current} (DROPPED by {})",
                        floor - current
                    ));
                }
            }
            None => failures.push(format!("{name}: missing from rigor_ledger.txt")),
        }
    };

    check_floor("panic_guards", counts.panic_guards);
    check_floor("asserts", counts.asserts);
    check_floor("result_guards", counts.result_guards);
    check_floor("ban_gates", ban_gates);

    // NaN ceiling: must not increase.
    match parse_ledger(&ledger_text, "nan_emissions") {
        Some(ceiling) => {
            if counts.nan_emissions > ceiling {
                failures.push(format!(
                    "nan_emissions: {ceiling} -> {} (CEILING EXCEEDED by {})",
                    counts.nan_emissions,
                    counts.nan_emissions - ceiling
                ));
            }
        }
        None => failures.push("nan_emissions: missing from rigor_ledger.txt".to_string()),
    }

    assert!(
        failures.is_empty(),
        "RIGOR REGRESSED: the codebase's aggregate strictness dropped silently.\n  {}\n\n\
         These metrics are committed state in rigor_ledger.txt and must be MONOTONE: \
         floor metrics (panic_guards/asserts/result_guards/ban_gates) may not decrease \
         and nan_emissions may not increase. A drop means a guard, assertion, error \
         return, or ban gate was removed/weakened (or a NaN emission was added). \
         If this weakening is intentional and justified, it must be made EXPLICIT by \
         editing rigor_ledger.txt in the same commit — so the decrement is visible and \
         reviewable in the diff. Strictness must not silently drop.",
        failures.join("\n  ")
    );
}
