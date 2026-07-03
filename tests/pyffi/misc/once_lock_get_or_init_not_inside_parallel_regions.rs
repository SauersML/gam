use std::fs;
use std::path::{Path, PathBuf};

/// #1253 regression guard: no `OnceLock::get_or_init` may run *inside* a rayon
/// parallel region (first-init landing on a worker thread risks a deadlock /
/// non-deterministic init). The original bug was in `survival/base.rs`.
///
/// IMPORTANT (#1253 re-fix): the first version of this guard scanned only
/// `CARGO_MANIFEST_DIR/src` — i.e. the *root* `gam` crate's `src/`. After the
/// #1521 crate carve, essentially all production code (including the
/// survival/quadrature paths that originally tripped this) moved into
/// `crates/<crate>/src`, leaving the root `src/` with a handful of thin
/// re-export shims and ZERO `get_or_init` call sites. The guard therefore
/// scanned an empty target and passed vacuously — it could no longer catch a
/// reintroduction of the very bug it exists to prevent. This version walks
/// EVERY workspace crate's `src/` tree (plus the root `src/`), so all ~90
/// real `get_or_init` call sites are covered.
#[test]
fn once_lock_get_or_init_not_inside_parallel_regions() {
    let workspace_root = workspace_root();
    let src_roots = workspace_src_roots(&workspace_root);
    assert!(
        !src_roots.is_empty(),
        "BUG: found no crate src/ trees to scan under {} — the guard would be \
         vacuous (see #1253)",
        workspace_root.display()
    );

    // Sanity tripwire: the scan must actually reach the production crates that
    // carry `get_or_init`. If this count ever drops to zero the guard has gone
    // blind again (e.g. another relocation), exactly the #1253 failure mode.
    let mut total_get_or_init = 0usize;
    let mut offenders: Vec<String> = Vec::new();

    let mut stack: Vec<PathBuf> = src_roots;
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries {
            let entry = entry.expect("dir entry failed");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let content = fs::read_to_string(&path).expect("read file failed");
            let lines: Vec<&str> = content.lines().collect();
            for i in 0..lines.len() {
                let line = lines[i];
                if line.contains("get_or_init(") {
                    total_get_or_init += 1;
                    // The risk this gate guards against is a `get_or_init` running
                    // *inside* a rayon parallel closure (first-init landing on a
                    // worker thread). A blind ±N-line window mis-fires when an
                    // unrelated `fn` that merely happens to sit near a parallel
                    // loop also uses a cache. Walk backwards within the SAME
                    // function body (stop at the previous top-level `fn ` opener)
                    // and only flag if a parallel iterator was opened before this
                    // line without an intervening function boundary.
                    let mut opened_parallel = false;
                    for j in (0..i).rev() {
                        let prev = lines[j];
                        // Top-level `fn ` declaration ends the enclosing-scope walk.
                        let trimmed = prev.trim_start();
                        if trimmed.starts_with("fn ")
                            || trimmed.starts_with("pub fn ")
                            || trimmed.starts_with("pub(crate) fn ")
                            || trimmed.starts_with("async fn ")
                        {
                            break;
                        }
                        if prev.contains(".par_iter(") || prev.contains(".into_par_iter(") {
                            opened_parallel = true;
                            break;
                        }
                    }
                    if opened_parallel {
                        offenders.push(format!("{}:{}", path.display(), i + 1));
                    }
                }
            }
        }
    }

    assert!(
        total_get_or_init > 0,
        "BUG (#1253): scanned the workspace and found ZERO `get_or_init` call \
         sites — the guard has gone blind again (code likely relocated out of \
         the scanned src/ roots). Re-point `workspace_src_roots` at the new \
         layout."
    );

    assert!(
        offenders.is_empty(),
        "found get_or_init in proximity of par_iter/into_par_iter:\n{}",
        offenders.join("\n")
    );
}

/// Locate the workspace root by walking up from this test crate's manifest dir
/// until a `Cargo.toml` containing a `[workspace]` table is found.
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    loop {
        let manifest = dir.join("Cargo.toml");
        if let Ok(text) = fs::read_to_string(&manifest) {
            // A line-anchored `[workspace]` table marks the root manifest.
            if text
                .lines()
                .any(|l| l.trim_start().starts_with("[workspace]"))
            {
                return dir;
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            // Fall back to the manifest dir if no workspace table is found
            // (single-crate layout): scanning its own src/ is still correct.
            None => return PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        }
    }
}

/// Every `src/` directory whose Rust sources ship in the build: the root crate's
/// `src/` plus each `crates/<crate>/src/`.
fn workspace_src_roots(root: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let root_src = root.join("src");
    if root_src.is_dir() {
        roots.push(root_src);
    }
    let crates_dir = root.join("crates");
    if let Ok(entries) = fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            let crate_src = entry.path().join("src");
            if crate_src.is_dir() {
                roots.push(crate_src);
            }
        }
    }
    roots
}
