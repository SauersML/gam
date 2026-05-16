use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

/// Variants of `src/approx_ledger::ApproxKind`. Kept as bare identifiers so
/// the scanner can match the same names that appear in source comments and
/// in `pub enum ApproxKind`. If the enum gains a variant, add it here.
const APPROX_LEDGER_VARIANTS: &[&str] = &[
    "Exact",
    "NumericalApproximation",
    "StatisticalApproximation",
    "SurrogateObjective",
    "TemporarySolverDamping",
];

/// Hand-wavy markers that must be paired with an `ApproxKind` annotation in
/// nearby comment lines. Case-insensitive match on whole tokens. The TO-DO
/// ban from the original build script is preserved as an outright ban
/// (no annotation can rescue a TO-DO — just do it now), but the other
/// markers are *allowed* when classified.
const HANDWAVY_MARKERS: &[&str] = &["bandaid", "hack", "magic", "FIXME"];

/// Lines on either side of a marker line within which an `ApproxKind`
/// reference counts as annotation. Mirrors `LEDGER_WINDOW` in
/// `src/approx_ledger.rs`.
const LEDGER_WINDOW: usize = 8;

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

    // Approximation-ledger scan: hand-wavy markers ("bandaid", "hack",
    // "magic", "FIXME") are allowed only when the surrounding comment
    // window names an ApproxKind variant from `src/approx_ledger.rs`.
    let mut unclassified: Vec<(PathBuf, usize, &'static str, String)> = Vec::new();
    scan_for_unclassified_handwavy(&manifest_dir, &manifest_dir, &mut unclassified);
    if !unclassified.is_empty() {
        eprintln!();
        eprintln!(
            "error: {} unclassified hand-wavy marker(s) found.",
            unclassified.len()
        );
        eprintln!(
            "       Pair the marker with an ApproxKind annotation from \
             src/approx_ledger.rs (one of: {}) within {} lines, or replace \
             it with precise wording.",
            APPROX_LEDGER_VARIANTS.join(", "),
            LEDGER_WINDOW
        );
        eprintln!();
        for (rel, line_no, marker, line) in &unclassified {
            let trimmed = line.trim();
            let snippet: String = trimmed.chars().take(160).collect();
            eprintln!("  {}:{} [{}]: {}", rel.display(), line_no, marker, snippet);
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

fn scan_for_ignored_tests(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
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

fn scan_for_unclassified_handwavy(
    root: &Path,
    dir: &Path,
    offenders: &mut Vec<(PathBuf, usize, &'static str, String)>,
) {
    visit_files(root, dir, &mut |rel, content| {
        // Skip the build script itself: it legitimately mentions every
        // marker by name as part of its own scanner contract.
        // Normalize Windows backslashes so the exemption comparison matches
        // on every host.
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str == "build.rs" {
            return;
        }
        let lower = content.to_ascii_lowercase();
        if !HANDWAVY_MARKERS
            .iter()
            .any(|m| lower.contains(&m.to_ascii_lowercase()))
        {
            return;
        }
        let lines: Vec<&str> = content.lines().collect();
        let lower_lines: Vec<String> = lines.iter().map(|l| l.to_ascii_lowercase()).collect();
        for (idx, line) in lines.iter().enumerate() {
            for marker in HANDWAVY_MARKERS {
                let m_lower = marker.to_ascii_lowercase();
                if !lower_lines[idx].contains(&m_lower) {
                    continue;
                }
                if !is_marker_in_comment(line, &m_lower) {
                    continue;
                }
                if window_has_ledger_annotation(&lines, idx) {
                    continue;
                }
                offenders.push((rel.to_path_buf(), idx + 1, marker, line.to_string()));
            }
        }
    });
}

/// True when the marker token sits inside a line comment (`//`) — strings
/// and identifiers like `hack_score()` or test names should not be flagged.
/// Lightweight heuristic: we just require the marker to appear *after* a
/// `//` on the same line.
fn is_marker_in_comment(line: &str, marker_lower: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    let comment_pos = lower.find("//");
    let marker_pos = lower.find(marker_lower);
    match (comment_pos, marker_pos) {
        (Some(cp), Some(mp)) => mp > cp,
        _ => false,
    }
}

/// True when any line within `LEDGER_WINDOW` of `idx` references one of the
/// `ApproxKind` variant names. The scan is case-sensitive on the variant
/// (they are CamelCase identifiers); a bare mention anywhere in the window
/// is sufficient.
fn window_has_ledger_annotation(lines: &[&str], idx: usize) -> bool {
    let lo = idx.saturating_sub(LEDGER_WINDOW);
    let hi = (idx + LEDGER_WINDOW + 1).min(lines.len());
    for line in &lines[lo..hi] {
        for variant in APPROX_LEDGER_VARIANTS {
            if line.contains(variant) {
                return true;
            }
        }
    }
    false
}

/// Walk `dir` recursively, calling `visitor(rel_path, content)` for every
/// scannable file. Centralizes the directory-skip and extension rules so
/// the two scans share them.
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
