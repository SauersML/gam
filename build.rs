use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
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
    let mut rs_files: Vec<PathBuf> = Vec::new();
    scan_for_banned_marker(
        &manifest_dir,
        &manifest_dir,
        needle,
        &mut offenders,
        &mut rs_files,
    );
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

    check_rustfmt(&manifest_dir, &rs_files);
}

/// Run `rustfmt --check` on every `.rs` file the marker scan walked over.
///
/// The Rust CI's `cargo fmt --all -- --check` job has been historically
/// fragile: `b554e8c`, `c224757`, `b807d48`, and `b2c4de6` were all pure
/// rustfmt-fixup commits landed reactively after the gate had already
/// failed on `main`. Each fixup re-applied rustfmt without closing the
/// loop, so the next batch of refactors quietly reintroduced violations.
/// Mirroring the check inside `build.rs` shifts the failure left: the
/// same `rustfmt --check` invocation now runs as part of any `cargo
/// build` / `cargo test` and rejects the build before the diff can
/// leave the developer's machine.
///
/// If `rustfmt` is not on PATH (some minimal CI images install only
/// `rustc` / `cargo`), the check is skipped quietly so the build still
/// succeeds — the workflow's dedicated Rustfmt job remains the
/// authoritative gate.
fn check_rustfmt(manifest_dir: &Path, files: &[PathBuf]) {
    if files.is_empty() {
        return;
    }
    // Probe first: distinguishes "rustfmt not installed" (the rustup
    // shim exits non-zero with `the 'rustfmt' binary ... is not
    // applicable`) from real formatting failures. Without the probe a
    // CI job that didn't install the `rustfmt` component (e.g. the
    // `test` / `clippy` / `python-tests` jobs in `.github/workflows/
    // test.yml`, which use `dtolnay/rust-toolchain` without the
    // `rustfmt` component) would have its `cargo build` invocation
    // fail here for the wrong reason.
    let rustfmt_available = Command::new("rustfmt")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !rustfmt_available {
        return;
    }
    // Match the workflow exactly: `cargo fmt --all -- --check` resolves
    // edition / style-edition from each crate's `rustfmt.toml`. Running
    // rustfmt with `current_dir(manifest_dir)` lets it discover the same
    // root `rustfmt.toml`, so we don't need to (and must not) override
    // the edition on the CLI here — that would silently diverge from the
    // gate the CI job enforces.
    let mut cmd = Command::new("rustfmt");
    cmd.current_dir(manifest_dir).arg("--check").args(files);
    let output = match cmd.output() {
        Ok(out) => out,
        Err(_) => return,
    };
    if output.status.success() {
        return;
    }
    eprintln!();
    eprintln!(
        "error: rustfmt detected unformatted Rust source. run `cargo fmt --all` and rebuild."
    );
    eprintln!();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stderr.trim().is_empty() {
        eprintln!("{}", stderr);
    }
    if !stdout.trim().is_empty() {
        eprintln!("{}", stdout);
    }
    std::process::exit(1);
}

fn scan_for_banned_marker(
    root: &Path,
    dir: &Path,
    needle: &str,
    offenders: &mut Vec<(PathBuf, usize, String)>,
    rs_files: &mut Vec<PathBuf>,
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
            scan_for_banned_marker(root, &path, needle, offenders, rs_files);
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
        if ext == "rs" {
            rs_files.push(path.clone());
        }
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
