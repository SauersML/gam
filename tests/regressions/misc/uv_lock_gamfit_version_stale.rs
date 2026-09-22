//! The release version must be the same in every manifest that CARRIES one.
//!
//! uv.lock is not one of them, and that is why this file is named for it. gam#3157 moved uv.lock's
//! `gamfit` entry to the dynamic form `uv lock` writes for a project whose version the build
//! backend derives — `name = "gamfit"` with `source = { editable = "." }` and no version line —
//! and `build.rs::require_uv_lock_gamfit_dynamic` refuses a version line there by name ("gamfit's
//! version is dynamic; relock with `uv lock`"). This test still read uv.lock for a version and
//! panicked with "uv.lock has no gamfit package version" when it found none, which is every
//! revision since that commit. Two rules for one line, pointing opposite ways; the scanner's is
//! the one that measured uv's own output, so uv.lock is not read here.
//!
//! `pyproject.toml`'s `[tool.gamfit] version` is the release line. `build.rs` holds
//! `crates/gam-pyffi/Cargo.toml` to it (`require_toml_version`) and refuses a static version in
//! `[project]` (`require_dynamic_project_version`), and its tree-wide scan flags any line naming
//! both `gamfit` and a `0.1.` version that is not the release line. `Cargo.lock`'s `gam-pyffi`
//! entry is the one manifest that carries the version and escapes all three: its `version = ` line
//! sits on its own, naming neither `gamfit` nor the crate. That is what this test asserts.
use std::path::Path;
use std::{fs, io};

fn quoted_value_after<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let raw = line.trim().strip_prefix(prefix)?.trim();
    let rest = raw.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// The release line: `pyproject.toml`'s first `version = ` line, which is `[tool.gamfit]`'s. The
/// file's own comment says so, and `[project]` carries none.
fn release_line(path: &Path) -> String {
    let content = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("read {}: {err}", path.display());
    });
    content
        .lines()
        .find_map(|line| quoted_value_after(line, "version = "))
        .unwrap_or_else(|| panic!("{} has no version line", path.display()))
        .to_string()
}

fn cargo_lock_package_version(path: &Path, package_name: &str) -> Option<String> {
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return None,
        Err(err) => panic!("read {}: {err}", path.display()),
    };
    let mut inside_package = false;
    let mut inside_target = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[[package]]" {
            inside_package = true;
            inside_target = false;
            continue;
        }
        if !inside_package {
            continue;
        }
        if trimmed.starts_with('[') {
            inside_package = false;
            inside_target = false;
            continue;
        }
        if let Some(name) = quoted_value_after(trimmed, "name = ") {
            inside_target = name == package_name;
            continue;
        }
        if inside_target {
            if let Some(version) = quoted_value_after(trimmed, "version = ") {
                return Some(version.to_string());
            }
        }
    }
    panic!("{} has no {package_name} package version", path.display());
}

/// `Cargo.lock`'s `gam-pyffi` entry must carry the release line.
///
/// A lockfile absent from the tree is not a stale one: `Cargo.lock` is regenerated on the first
/// build, so a checkout without it has nothing to disagree. A lockfile that is present and names a
/// different version is a release that landed in the manifests and not in the lock, and nothing
/// else in the tree catches it.
#[test]
fn cargo_lock_gam_pyffi_carries_the_release_line() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let release = release_line(&root.join("pyproject.toml"));
    let Some(cargo_lock) = cargo_lock_package_version(&root.join("Cargo.lock"), "gam-pyffi") else {
        return;
    };
    assert_eq!(
        release, cargo_lock,
        "Cargo.lock's gam-pyffi version must be pyproject.toml's [tool.gamfit] release line"
    );
}
