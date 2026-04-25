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
}

fn scan_for_banned_marker(
    root: &Path,
    dir: &Path,
    needle: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
) {
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
            scan_for_banned_marker(root, &path, needle, offenders);
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
        if !content.contains(needle) {
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
        for (idx, line) in content.lines().enumerate() {
            if line.contains(needle) {
                offenders.push((rel.clone(), idx + 1, line.to_string()));
            }
        }
    }
}
