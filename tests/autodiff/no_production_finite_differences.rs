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
    "fd_",
    "_fd",
    "fd-",
    "-fd",
    "numdiff",
    "numerical gradient",
    "numerical_gradient",
    "numeric gradient",
    "numeric_gradient",
    "richardson",
];

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
    if matches!(file_name, Some("tests.rs" | "test_support.rs"))
        || file_name.is_some_and(|name| name.ends_with("_tests.rs"))
    {
        return true;
    }

    path.components().any(|component| {
        matches!(
            component,
            Component::Normal(name)
                if matches!(name.to_str(), Some("tests" | "test_support" | "testing"))
        )
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
        match byte {
            b'(' => depth += 1,
            b')' => depth = depth.saturating_sub(1),
            b',' if depth == 0 => {
                if args[arg_start..i].trim() == "test" {
                    return true;
                }
                arg_start = i + 1;
            }
            _ => {}
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
        "src/families/gamlss/tests.rs",
        "src/solver/estimate/continuous_order_tests.rs",
        "src/families/custom_family/test_support.rs",
        "src/test_support/fd_checker.rs",
        "src/testing/mod.rs",
        "src/foo/tests/bar.rs",
        "src/foo/._bar.rs",
    ] {
        assert!(is_test_only_source_file(Path::new(path)), "{path}");
    }

    assert!(!is_test_only_source_file(Path::new(
        "src/inference/quadrature.rs"
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
fn production_code_has_no_finite_difference_markers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_rust_files(&root.join("src"), &mut files);
    let mut violations = Vec::new();
    for path in files {
        if is_test_only_source_file(&path) {
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
    }

    assert!(
        violations.is_empty(),
        "finite-difference/numerical-gradient markers are forbidden in production code:\n{}",
        violations.join("\n")
    );
}
