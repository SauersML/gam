use std::fs;
use std::path::Path;

fn quoted_value_after<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let raw = line.trim().strip_prefix(prefix)?.trim();
    let rest = raw.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn manifest_version(path: &Path) -> String {
    let content = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("read {}: {err}", path.display());
    });
    content
        .lines()
        .find_map(|line| quoted_value_after(line, "version = "))
        .unwrap_or_else(|| panic!("{} has no version line", path.display()))
        .to_string()
}

fn uv_lock_gamfit_version(path: &Path) -> String {
    let content = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("read {}: {err}", path.display());
    });
    let mut inside_package = false;
    let mut inside_gamfit = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[[package]]" {
            inside_package = true;
            inside_gamfit = false;
            continue;
        }
        if !inside_package {
            continue;
        }
        if trimmed.starts_with('[') {
            inside_package = false;
            inside_gamfit = false;
            continue;
        }
        if let Some(name) = quoted_value_after(trimmed, "name = ") {
            inside_gamfit = name == "gamfit";
            continue;
        }
        if inside_gamfit {
            if let Some(version) = quoted_value_after(trimmed, "version = ") {
                return version.to_string();
            }
        }
    }
    panic!("{} has no gamfit package version", path.display());
}

fn cargo_lock_package_version(path: &Path, package_name: &str) -> String {
    let content = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!("read {}: {err}", path.display());
    });
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
                return version.to_string();
            }
        }
    }
    panic!("{} has no {package_name} package version", path.display());
}

#[test]
fn gamfit_version_is_consistent_across_manifests() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let pyproject = manifest_version(&root.join("pyproject.toml"));
    let pyffi = manifest_version(&root.join("crates/gam-pyffi/Cargo.toml"));
    let uv_lock = uv_lock_gamfit_version(&root.join("uv.lock"));
    let cargo_lock = cargo_lock_package_version(&root.join("Cargo.lock"), "gam-pyffi");

    assert_eq!(
        pyproject, pyffi,
        "pyproject.toml and crates/gam-pyffi/Cargo.toml must release together"
    );
    assert_eq!(
        pyproject, uv_lock,
        "uv.lock editable gamfit package must match pyproject.toml"
    );
    assert_eq!(
        pyproject, cargo_lock,
        "Cargo.lock gam-pyffi package must match pyproject.toml"
    );
}
