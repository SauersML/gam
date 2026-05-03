use std::fs;
use std::path::{Path, PathBuf};

const BANNED_PRODUCTION_MARKERS: &[&str] = &[
    "finite difference",
    "finite-difference",
    "central difference",
    "central fd",
    "fd-vs",
    "numerical gradient",
    "numeric gradient",
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
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.trim_start().starts_with("#[cfg(test)]") {
            i += 1;
            while i < lines.len() && !lines[i].contains('{') {
                i += 1;
            }
            if i == lines.len() {
                break;
            }
            let mut depth = 0isize;
            loop {
                depth += lines[i].matches('{').count() as isize;
                depth -= lines[i].matches('}').count() as isize;
                i += 1;
                if i == lines.len() || depth <= 0 {
                    break;
                }
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    out
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
