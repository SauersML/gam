//! #1440 production finite-difference ban scanner, registered as a first-class
//! top-level integration binary (`tests/no_production_finite_differences.rs`) so
//! the `for test_file in tests/*.rs` CI loop in `.github/workflows/test.yml`
//! discovers and runs it. It previously lived as a buried submodule of the
//! `autodiff` harness (`tests/autodiff/misc/...`), which cargo compiles as the
//! `autodiff` target — a name the `tests/*.rs` glob never matched — so the ban
//! never actually ran in CI.
//!
//! It also scans the WHOLE workspace: the root crate's `src` plus every
//! `crates/*/src`. The #1521 monolith carve-out moved nearly all production code
//! down into workspace crates, so scanning only `root/src` (as the buried copy
//! did) left the real production tree — including the actual FFI/solver/terms
//! code — completely uncovered.

use std::fs;
use std::path::{Component, Path, PathBuf};

const BANNED_PRODUCTION_MARKERS: &[&str] = &[
    "finite difference",
    "finite-difference",
    "finite_diff",
    "finitediff",
    "central difference",
    "central fd",
    "central_fd",
    "central-diff",
    "central_diff",
    "fd-vs",
    "numdiff",
    "numerical gradient",
    "numerical_gradient",
    "numeric gradient",
    "numeric_gradient",
    "richardson",
];

/// True when `source` contains the identifier token `fd` used as a piece of a
/// compound identifier (`fd_grad`, `grad_fd`, `fd-vs`).
///
/// This replaces the four bare substrings `fd_` / `_fd` / `fd-` / `-fd` that
/// [`BANNED_PRODUCTION_MARKERS`] used to carry. The scan is a plain
/// `production.contains(marker)`, so those substrings fired INSIDE longer
/// alphanumeric tokens that have nothing to do with finite differences:
/// `empirical_fdr`, `split_lr_fdr` and `family_fdr_certificate` (FDR = false
/// discovery rate, a multiple-testing statistic) and the hex seed constant
/// `0xd6e8_feb8_6659_fd93` were all reported as production finite differences.
///
/// The rule those markers were reaching for is token-aware:
///
/// * the maximal run of ASCII-alphanumeric characters around the `fd` must be
///   exactly `fd`, so `fdr` / `fd93` / `feb8` can never match, AND
/// * at least one immediately adjacent character must be `_` or `-`, so a bare
///   file-descriptor variable (`let fd = 3;`) is not treated as an FD marker.
///   That adjacency is precisely what the original markers were encoding.
fn contains_fd_identifier_token(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut i = 0usize;
    while i + 1 < bytes.len() {
        if !(bytes[i].eq_ignore_ascii_case(&b'f') && bytes[i + 1].eq_ignore_ascii_case(&b'd')) {
            i += 1;
            continue;
        }
        let before = i.checked_sub(1).map(|prev| bytes[prev]);
        let after = bytes.get(i + 2).copied();
        // The maximal alphanumeric run must be exactly `fd`.
        let token_is_exactly_fd = !before.is_some_and(|byte| byte.is_ascii_alphanumeric())
            && !after.is_some_and(|byte| byte.is_ascii_alphanumeric());
        // …and it must be joined to something by `_`/`-`, i.e. a compound name.
        let has_adjacent_separator =
            matches!(before, Some(b'_' | b'-')) || matches!(after, Some(b'_' | b'-'));
        if token_is_exactly_fd && has_adjacent_separator {
            return true;
        }
        i += 1;
    }
    false
}

/// Files that are PERMITTED to use `FD-OK:` / `END-FD-OK` / `fd-ok:` audit
/// markers to exempt regions from the production-FD ban.
///
/// This is the single, tracked source of truth for sanctioned finite-difference
/// in the source tree (#1440). A bare `fd-ok` comment in any file NOT listed
/// here is ignored by the scanner, so a new finite difference can never hide
/// behind a freshly-sprinkled exemption — it must be justified here first, in
/// review, with the reason it is irreducible.
///
/// Paths are matched as suffixes of the file path (forward-slash normalised), so
/// `crates/gam-solve/src/rho_optimizer/fd_audit.rs` matches regardless of the
/// absolute prefix. Each entry MUST carry a justification comment and MUST be a
/// full workspace-crate path (the #1521 carve-out relocated every entry out of
/// the old `root/src/...` monolith and down into a `crates/<crate>/src/...`
/// tree).
const SANCTIONED_FD_FILES: &[&str] = &[
    // Declares the `hessian_fd_probes` module, whose pub accessors hand tests the
    // dense effective Hessian and the converged inner state at a fixed theta.
    // The accessors difference nothing; the tests that call them form the
    // differences.
    "crates/gam-solve/src/estimate/mod.rs",
    // Implements the external `opt` crate's `set_finite_difference_bounds` hook by
    // delegation. gam takes no finite difference there.
    "crates/gam-solve/src/rho_optimizer/run.rs",
];

/// The root crate `src` plus every `crates/*/src`, in stable sorted order. This
/// is the whole production tree the #1521 carve-out spread across the workspace;
/// scanning only `root/src` (as the buried autodiff-harness copy did) left the
/// real FFI/solver/terms code uncovered, so the ban silently passed.
fn workspace_production_src_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let root_src = root.join("src");
    if root_src.is_dir() {
        dirs.push(root_src);
    }
    if let Ok(entries) = fs::read_dir(root.join("crates")) {
        let mut crate_dirs: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .collect();
        crate_dirs.sort();
        for crate_dir in crate_dirs {
            let src = crate_dir.join("src");
            if src.is_dir() {
                dirs.push(src);
            }
        }
    }
    dirs
}

/// Collect every `.rs` file across the whole workspace production tree.
fn collect_workspace_production_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in workspace_production_src_dirs(root) {
        collect_rust_files(&dir, &mut files);
    }
    files
}

/// True when `path` is on the [`SANCTIONED_FD_FILES`] allowlist and may
/// therefore use `fd-ok` audit markers.
fn fd_ok_markers_allowed(path: &Path) -> bool {
    let normalised: String = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => name.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/");
    SANCTIONED_FD_FILES
        .iter()
        .any(|allowed| normalised.ends_with(allowed))
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source directory") {
        let entry = entry.expect("read source entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, out);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("._"))
        {
            continue;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn is_test_only_source_file(path: &Path) -> bool {
    let file_name = path.file_name().and_then(|name| name.to_str());
    if file_name.is_some_and(|name| name.starts_with("._")) {
        return true;
    }
    // Split-out unit-test submodules follow the repo's `tests_<topic>.rs` naming
    // convention and are always declared `#[cfg(test)] mod tests_<topic>;` in
    // their parent (e.g. crates/gam-sae/src/manifold/tests_row_jet_and_outer_objective_780.rs,
    // crates/gam-solve/src/estimate/tests_diagnostics.rs). They live alongside the
    // production source under `src/` rather than under a `tests/` directory, so the
    // path-component check below never sees them. Recognize the `tests_` prefix the
    // same way the `tests.rs` exact name and the `_tests.rs` suffix already are —
    // FD comparison against the analytic gradient is sanctioned inside tests.
    if matches!(file_name, Some("tests.rs" | "test_support.rs"))
        || file_name.is_some_and(|name| name.ends_with("_tests.rs") || name.starts_with("tests_"))
    {
        return true;
    }

    path.components().any(|component| {
        let Component::Normal(name) = component else {
            return false;
        };
        let Some(name) = name.to_str() else {
            return false;
        };
        // Exact-name test directories (`…/tests/…`, `…/testing/…`) AND whole
        // test-support CRATES. The #1521 carve-out moved the FD test harness into
        // its own `gam-test-support` crate; scanning `crates/*/src` now reaches it,
        // so a crate directory whose name marks it as test-support infrastructure
        // (`gam-test-support`) is test-only in its entirety — FD IS sanctioned in
        // test harnesses. Match the `test-support` / `test_support` substring so the
        // crate-dir component (`gam-test-support`) is covered without listing it.
        matches!(name, "tests" | "test_support" | "testing")
            || name.contains("test-support")
            || name.contains("test_support")
    })
}

fn strip_cfg_test_blocks(source: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0usize;
    while let Some(attr_start) = find_next_cfg_test_only_attr(source, cursor) {
        out.push_str(&source[cursor..attr_start]);
        let Some(item_delimiter) = find_next_cfg_test_item_delimiter(source, attr_start) else {
            cursor = source.len();
            break;
        };
        if source.as_bytes()[item_delimiter] == b';' {
            cursor = item_delimiter + 1;
            continue;
        }
        let open_brace = item_delimiter;
        let Some(block_end) = find_matching_code_brace(source, open_brace) else {
            cursor = source.len();
            break;
        };
        cursor = block_end;
    }
    out.push_str(&source[cursor..]);
    out
}

fn strip_fd_ok_regions(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut in_fd_ok_region = false;
    for line in source.split_inclusive('\n') {
        let has_region_start = line.contains("FD-OK:");
        let has_region_end = line.contains("END-FD-OK");
        let has_line_marker = line.contains("fd-ok:");
        if in_fd_ok_region || has_region_start || has_line_marker {
            preserve_newline_shape(line, &mut out);
        } else {
            out.push_str(line);
        }
        if has_region_start {
            in_fd_ok_region = true;
        }
        if has_region_end {
            in_fd_ok_region = false;
        }
    }
    out
}

fn preserve_newline_shape(line: &str, out: &mut String) {
    if line.ends_with('\n') {
        out.push('\n');
    } else {
        out.push(' ');
    }
}

fn find_next_cfg_test_only_attr(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i = skip_block_comment(bytes, i + 2);
            }
            b'"' => {
                i = skip_cooked_string(bytes, i + 1);
            }
            b'r' if raw_string_hashes_at(bytes, i).is_some() => {
                let hashes = raw_string_hashes_at(bytes, i).expect("checked raw string");
                i = skip_raw_string(bytes, i + 1 + hashes + 1, hashes);
            }
            b'#' if bytes.get(i..i + 6) == Some(b"#[cfg(") => {
                let attr_start = i;
                let Some(attr_end) = bytes[attr_start..]
                    .iter()
                    .position(|byte| *byte == b']')
                    .map(|end| attr_start + end + 1)
                else {
                    return None;
                };
                if cfg_attr_is_test_only(&source[attr_start..attr_end]) {
                    return Some(attr_start);
                }
                i = attr_end;
            }
            _ => {
                i += 1;
            }
        }
    }
    None
}

fn cfg_attr_is_test_only(attr: &str) -> bool {
    let compact: String = attr.chars().filter(|ch| !ch.is_whitespace()).collect();
    if compact == "#[cfg(test)]" {
        return true;
    }

    let Some(all_args) = compact
        .strip_prefix("#[cfg(all(")
        .and_then(|value| value.strip_suffix("))]"))
    else {
        return false;
    };
    cfg_all_args_have_direct_test_clause(all_args)
}

fn cfg_all_args_have_direct_test_clause(args: &str) -> bool {
    let bytes = args.as_bytes();
    let mut depth = 0usize;
    let mut arg_start = 0usize;
    for (i, byte) in bytes.iter().enumerate() {
        if *byte == b'(' {
            depth += 1;
        } else if *byte == b')' {
            depth = depth.saturating_sub(1);
        } else if *byte == b',' && depth == 0 {
            if args[arg_start..i].trim() == "test" {
                return true;
            }
            arg_start = i + 1;
        }
    }
    args[arg_start..].trim() == "test"
}

fn strip_prose(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                if i < bytes.len() {
                    out.push('\n');
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                let comment_start = i;
                i = skip_block_comment(bytes, i + 2);
                preserve_comment_spacing(&source[comment_start..i], &mut out);
            }
            b'"' => {
                i = skip_cooked_string(bytes, i + 1);
                out.push_str("\"\"");
            }
            b'r' if raw_string_hashes_at(bytes, i).is_some() => {
                let hashes = raw_string_hashes_at(bytes, i).expect("checked raw string");
                i = skip_raw_string(bytes, i + 1 + hashes + 1, hashes);
                out.push_str("r\"\"");
            }
            _ => {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    out
}

fn preserve_comment_spacing(comment: &str, out: &mut String) {
    for byte in comment.bytes() {
        if byte == b'\n' {
            out.push('\n');
        }
    }
    out.push(' ');
}

fn find_next_cfg_test_item_delimiter(source: &str, start: usize) -> Option<usize> {
    find_next_code_byte_where(source, start, |value| matches!(value, b'{' | b';'))
}

fn find_next_code_byte_where(
    source: &str,
    start: usize,
    mut matches_byte: impl FnMut(u8) -> bool,
) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i = skip_block_comment(bytes, i + 2);
            }
            b'"' => {
                i = skip_cooked_string(bytes, i + 1);
            }
            b'r' if raw_string_hashes_at(bytes, i).is_some() => {
                let hashes = raw_string_hashes_at(bytes, i).expect("checked raw string");
                i = skip_raw_string(bytes, i + 1 + hashes + 1, hashes);
            }
            value if matches_byte(value) => return Some(i),
            _ => i += 1,
        }
    }
    None
}

fn find_matching_code_brace(source: &str, open_brace: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    assert_eq!(bytes[open_brace], b'{');
    let mut depth = 1usize;
    let mut i = open_brace + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i = skip_block_comment(bytes, i + 2);
            }
            b'"' => {
                i = skip_cooked_string(bytes, i + 1);
            }
            b'r' if raw_string_hashes_at(bytes, i).is_some() => {
                let hashes = raw_string_hashes_at(bytes, i).expect("checked raw string");
                i = skip_raw_string(bytes, i + 1 + hashes + 1, hashes);
            }
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                i += 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => i += 1,
        }
    }
    None
}

fn skip_cooked_string(bytes: &[u8], mut i: usize) -> usize {
    let mut escaped = false;
    while i < bytes.len() {
        let b = bytes[i];
        i += 1;
        if escaped {
            escaped = false;
        } else if b == b'\\' {
            escaped = true;
        } else if b == b'"' {
            break;
        }
    }
    i
}

fn raw_string_hashes_at(bytes: &[u8], i: usize) -> Option<usize> {
    if bytes.get(i) != Some(&b'r') {
        return None;
    }
    let mut j = i + 1;
    while bytes.get(j) == Some(&b'#') {
        j += 1;
    }
    (bytes.get(j) == Some(&b'"')).then_some(j - i - 1)
}

fn skip_raw_string(bytes: &[u8], mut i: usize, hashes: usize) -> usize {
    while i < bytes.len() {
        if bytes[i] == b'"' && (0..hashes).all(|h| bytes.get(i + 1 + h) == Some(&b'#')) {
            return (i + 1 + hashes).min(bytes.len());
        }
        i += 1;
    }
    bytes.len()
}

fn skip_block_comment(bytes: &[u8], mut i: usize) -> usize {
    let mut depth = 1usize;
    while i + 1 < bytes.len() {
        if bytes[i] == b'/' && bytes[i + 1] == b'*' {
            depth += 1;
            i += 2;
        } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
            depth -= 1;
            i += 2;
            if depth == 0 {
                return i;
            }
        } else {
            i += 1;
        }
    }
    bytes.len()
}

#[test]
fn cfg_test_out_of_line_module_does_not_strip_following_production_item() {
    let source = r#"
#[cfg(test)]
mod tests;

fn production_fd_grad() {
}
"#;

    let stripped = strip_cfg_test_blocks(source);
    assert!(stripped.contains("production_fd_grad"));
}

#[test]
fn cfg_all_test_module_is_stripped_but_cfg_any_test_module_is_not() {
    let test_only = r#"
#[cfg(all(test, target_os = "linux"))]
mod tests {
    fn fd_grad() {}
}

fn production_code() {}
"#;
    let not_test_only = r#"
#[cfg(any(test, feature = "diagnostics"))]
fn production_visible_fd_grad() {}
"#;

    let stripped_test_only = strip_cfg_test_blocks(test_only);
    assert!(!stripped_test_only.contains("fd_grad"));
    assert!(stripped_test_only.contains("production_code"));

    let stripped_not_test_only = strip_cfg_test_blocks(not_test_only);
    assert!(stripped_not_test_only.contains("production_visible_fd_grad"));
}

#[test]
fn test_only_source_file_classification_covers_out_of_line_test_modules() {
    for path in [
        "crates/gam-models/src/families/gamlss/tests.rs",
        "crates/gam-solve/src/estimate/continuous_order_tests.rs",
        "crates/gam-custom-family/src/test_support.rs",
        "crates/gam-test-support/src/fd_checker.rs",
        "crates/gam-test-support/src/testing/mod.rs",
        "crates/gam-terms/src/foo/tests/bar.rs",
        "crates/gam-terms/src/foo/._bar.rs",
    ] {
        assert!(is_test_only_source_file(Path::new(path)), "{path}");
    }

    assert!(!is_test_only_source_file(Path::new(
        "crates/gam-inference/src/quadrature.rs"
    )));
}

#[test]
fn production_marker_scan_ignores_comment_prose() {
    let source = r#"
//! exact composition: no finite differences anywhere
fn production_code() {
    let message = "finite-difference fallback is forbidden here";
    let fd_grad = 1.0;
}
"#;

    let production = strip_prose(&strip_cfg_test_blocks(source)).to_lowercase();
    assert!(!production.contains("finite difference"));
    assert!(!production.contains("finite-difference"));
    assert!(production.contains("fd_"));
}

#[test]
fn production_marker_scan_ignores_explicit_fd_ok_audit_regions() {
    let source = r#"
fn production_code() {
    let fd_grad = 1.0;
    // FD-OK: diagnostic audit block, not model math
    let fd_directional = 2.0;
    let fd_error = 0.1;
    // END-FD-OK
    let fd_step = 1.0e-4; // fd-ok: diagnostic audit scalar
}
"#;

    let production =
        strip_prose(&strip_fd_ok_regions(&strip_cfg_test_blocks(source))).to_lowercase();
    assert!(production.contains("fd_grad"));
    assert!(!production.contains("fd_directional"));
    assert!(!production.contains("fd_error"));
    assert!(!production.contains("fd_step"));
}

#[test]
fn sanctioned_fd_allowlist_membership_is_correct() {
    // #1440: the SAE #1273 outer-ρ files carry NO finite difference (the fallback
    // was removed; the descent direction is the plain analytic gradient), so they
    // are NOT allowlisted.
    assert!(!fd_ok_markers_allowed(Path::new(
        "/Users/anyone/gam/crates/gam-sae/src/manifold/outer_objective.rs"
    )));
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-sae/src/manifold/construction.rs"
    )));
    // The outer-gradient FD audit is gone (#2901). `run_plan.rs` ran its Ridders
    // probe at an armed seed and `outer_eval_capture.rs` held its record; the
    // runner now lends an armed test an analytic seed probe, and the test forms
    // its own difference. Neither file, nor the retired `fd_audit.rs` before
    // them, is allowlisted, so a finite difference cannot creep back into them
    // unreviewed.
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-solve/src/rho_optimizer/run_plan.rs"
    )));
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-solve/src/estimate/outer_eval_capture.rs"
    )));
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-solve/src/rho_optimizer/fd_audit.rs"
    )));
    assert!(fd_ok_markers_allowed(Path::new(
        "/Users/anyone/gam/crates/gam-solve/src/rho_optimizer/run.rs"
    )));
    // The dead geodesic-acceleration probe was removed, so the P-IRLS update
    // path is not allowlisted.
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-solve/src/pirls/reweight.rs"
    )));
    // The sphere-boost GN chart Jacobian is analytic in production; its
    // finite-difference oracle is test-only, so the production file is not
    // allowlisted.
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-sae/src/chart_canonicalization.rs"
    )));
    // (#1440) The survival pilot W-metric chain is now the closed-form
    // `c_derivatives` chain, so `row_math.rs` carries no FD and is NOT allowlisted.
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-models/src/survival/marginal_slope/row_math.rs"
    )));
    // Files NOT carrying a documented sanction are not on the allowlist.
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-solve/src/reml/gradient_hessian.rs"
    )));
    assert!(!fd_ok_markers_allowed(Path::new(
        "crates/gam-inference/src/quadrature.rs"
    )));
}

#[test]
fn sanctioned_fd_allowlist_files_exist() {
    // Every allowlisted path must point at a real source file, so the allowlist
    // cannot silently rot into a stale, over-broad exemption (#1440). Entries are
    // full workspace-crate paths, resolved against the workspace root (the root
    // crate's `CARGO_MANIFEST_DIR`).
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for allowed in SANCTIONED_FD_FILES {
        let candidate = root.join(allowed);
        assert!(
            candidate.exists(),
            "sanctioned FD allowlist entry does not exist: {}",
            candidate.display()
        );
    }
}

#[test]
fn fd_ok_markers_are_confined_to_the_allowlist() {
    // #1440 single-source-of-truth invariant: a file may use `fd-ok`/`FD-OK`
    // audit markers ONLY if it is on `SANCTIONED_FD_FILES`. Otherwise a fresh
    // finite difference could hide behind a per-line `// fd-ok:` marker in any
    // file, defeating the allowlist (the scanner strips fd-ok regions before the
    // banned-marker search). This guard pins the allowlist as the genuine,
    // reviewed enumeration of every sanctioned FD in the tree: introducing FD in a
    // new file forces a reviewed allowlist entry (with a justification) here first.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = collect_workspace_production_files(root);
    let mut offenders = Vec::new();
    for path in &files {
        if is_test_only_source_file(path) {
            continue;
        }
        if fd_ok_markers_allowed(path) {
            continue;
        }
        let source = fs::read_to_string(path).expect("read source file");
        let uses_marker = source.lines().any(|line| {
            line.contains("FD-OK:") || line.contains("fd-ok:") || line.contains("END-FD-OK")
        });
        if uses_marker {
            offenders.push(path.display().to_string());
        }
    }
    assert!(
        offenders.is_empty(),
        "#1440: these non-test files use `fd-ok` audit markers but are NOT on the \
         SANCTIONED_FD_FILES allowlist — a finite difference may not hide behind a \
         per-line exemption; add the file to the allowlist (with a justification) \
         in review or remove the FD:\n{}",
        offenders.join("\n")
    );
}

/// A justification token is the non-whitespace text that must follow an `fd-ok`
/// audit marker. Returns `true` when the line carries a real reason after the
/// marker, `false` for a bare `fd-ok` / `fd-ok:` with nothing meaningful after.
fn fd_ok_line_has_justification(line: &str) -> bool {
    // Honour both the region opener (`FD-OK:`) and the per-line marker
    // (`fd-ok:`). For each occurrence the text AFTER the marker must contain at
    // least a few non-whitespace characters of explanation.
    for marker in ["FD-OK:", "fd-ok:"] {
        if let Some(pos) = line.find(marker) {
            let rest = line[pos + marker.len()..].trim();
            if rest.chars().filter(|c| !c.is_whitespace()).count() >= 4 {
                return true;
            }
        }
    }
    false
}

#[test]
fn fd_ok_justification_detector_behaves() {
    assert!(fd_ok_line_has_justification(
        "let h = STEP; // fd-ok: descent-direction only (#1273)"
    ));
    assert!(fd_ok_line_has_justification(
        "    // FD-OK: diagnostic audit block, not model math"
    ));
    // Bare markers with no reason are rejected.
    assert!(!fd_ok_line_has_justification("let h = STEP; // fd-ok:"));
    assert!(!fd_ok_line_has_justification("// fd-ok: !!"));
    // A line without any marker is irrelevant (returns false).
    assert!(!fd_ok_line_has_justification("let h = STEP;"));
}

#[test]
fn every_fd_ok_marker_in_the_tree_carries_a_justification() {
    // #1440: an `fd-ok` exemption must always state WHY the finite difference is
    // irreducible, so a bare `// fd-ok` can never silently mask a new FD. Every
    // marker line across the whole workspace tree is required to carry a reason.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = collect_workspace_production_files(root);
    let mut violations = Vec::new();
    for path in files {
        let source = fs::read_to_string(&path).expect("read source file");
        for (lineno, line) in source.lines().enumerate() {
            // `END-FD-OK` is a region closer; it never needs a justification.
            if line.contains("END-FD-OK") {
                continue;
            }
            let has_marker = line.contains("FD-OK:") || line.contains("fd-ok:");
            if has_marker && !fd_ok_line_has_justification(line) {
                violations.push(format!(
                    "{}:{} bare fd-ok marker without a justification: {}",
                    path.display(),
                    lineno + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "fd-ok audit markers must carry a justification (#1440):\n{}",
        violations.join("\n")
    );
}

#[test]
fn fd_identifier_token_accepts_real_finite_difference_names() {
    // `fd` joined to something by `_`/`-` is what the four deleted substring
    // markers (`fd_`, `_fd`, `fd-`, `-fd`) were reaching for.
    for accepted in [
        "let grad_fd_check = 1.0;",
        "fd_vs analytic",
        "assert_eq!(check_fd, analytic);",
        "let fd_grad = 1.0;",
        "fd-vs-analytic",
        // The two names this file's own strip tests plant in synthetic sources
        // MUST stay catchable: `fd` delimited by `_` on both sides.
        "fn production_fd_grad() {}",
        "fn production_visible_fd_grad() {}",
    ] {
        assert!(
            contains_fd_identifier_token(accepted),
            "should be an FD marker: {accepted}"
        );
    }
}

#[test]
fn fd_identifier_token_rejects_fdr_statistics_hex_seeds_and_file_descriptors() {
    // The measured false positives on main: FDR (false discovery rate) names in
    // `crates/gam-sae/src/sparse_dict/*` and a hex seed constant in
    // `crates/gam-sae/src/manifold/support_seed.rs`. Plus a bare file-descriptor
    // binding, which has no adjacent `_`/`-` and so was never matched even by
    // the old markers.
    for rejected in [
        "fdr",
        "let empirical_fdr = 0.05;",
        "split_lr_fdr",
        "family_fdr_certificate",
        "null_battery_rank1_shell_fdr_controlled",
        "let seed = 0xd6e8_feb8_6659_fd93;",
        "let fd = 3;",
        "fn open() -> Fd { }",
    ] {
        assert!(
            !contains_fd_identifier_token(&rejected.to_lowercase()),
            "should NOT be an FD marker: {rejected}"
        );
    }
}

#[test]
fn production_code_has_no_finite_difference_markers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let files = collect_workspace_production_files(root);
    let mut violations = Vec::new();
    for path in files {
        if is_test_only_source_file(&path) {
            continue;
        }
        // A file on the tracked SANCTIONED_FD_FILES allowlist carries its FD with
        // a documented, reviewed justification (#1440) and is exempt as a whole.
        // This is the single place such an exemption is recorded, so the FD is
        // tracked rather than silently tolerated.
        if fd_ok_markers_allowed(&path) {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read source file");
        let production =
            strip_prose(&strip_fd_ok_regions(&strip_cfg_test_blocks(&source))).to_lowercase();
        for marker in BANNED_PRODUCTION_MARKERS {
            if production.contains(marker) {
                violations.push(format!("{} contains `{}`", path.display(), marker));
            }
        }
        if contains_fd_identifier_token(&production) {
            violations.push(format!(
                "{} contains an `fd` identifier token (e.g. `fd_grad` / `grad_fd`)",
                path.display()
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "finite-difference/numerical-gradient markers are forbidden in production code:\n{}",
        violations.join("\n")
    );
}
