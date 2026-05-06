use std::fs;
use std::path::{Path, PathBuf};

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
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn strip_cfg_test_blocks(source: &str) -> String {
    let mut out = String::new();
    let mut cursor = 0usize;
    while let Some(rel_attr) = source[cursor..].find("#[cfg(test)]") {
        let attr_start = cursor + rel_attr;
        out.push_str(&source[cursor..attr_start]);
        let Some(open_brace) = find_next_code_byte(source, attr_start, b'{') else {
            cursor = source.len();
            break;
        };
        let Some(block_end) = find_matching_code_brace(source, open_brace) else {
            cursor = source.len();
            break;
        };
        cursor = block_end;
    }
    out.push_str(&source[cursor..]);
    out
}

fn find_next_code_byte(source: &str, start: usize, needle: u8) -> Option<usize> {
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
            value if value == needle => return Some(i),
            _ => i += 1,
        }
    }
    None
}

fn find_matching_code_brace(source: &str, open_brace: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    debug_assert_eq!(bytes[open_brace], b'{');
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
fn production_code_has_no_finite_difference_markers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    collect_rust_files(&root.join("src"), &mut files);
    let mut violations = Vec::new();
    for path in files {
        if path.ends_with("src/testing/mod.rs") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("read source file");
        let production = strip_cfg_test_blocks(&source).to_lowercase();
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
