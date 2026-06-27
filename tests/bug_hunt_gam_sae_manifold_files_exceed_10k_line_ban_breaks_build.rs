//! Bug hunt: the workspace does not build from a clean checkout at HEAD.
//!
//! `build.rs` enforces issue #780's line-count gate — every git-tracked file
//! must stay at or below `MAX_TRACKED_FILE_LINES = 10_000` newline-counted
//! lines (`scan_for_oversized_tracked_files`). Two `gam-sae` source files have
//! grown past that ceiling:
//!
//!   * crates/gam-sae/src/manifold/construction.rs  — 10012 lines
//!   * crates/gam-sae/src/manifold/tests.rs         — 10114 lines
//!
//! Both crossed 10_000 at commit 50d8d17e3 ("perf(#1557): pin SAE arrow-Schur
//! per-row faer GEMM to Par::Seq"): construction.rs went 9995 -> 10012 and
//! tests.rs 9984 -> 10114. On any state that forces `build.rs` to re-run (a
//! clean build — `cargo build`, `cargo test`, or the `maturin` wheel build all
//! do), the line-count audit fires and the build script aborts:
//!
//!   error — tracked file over 10k lines (split the file; issue #780 line-count gate)  (2 hits)
//!     error: crates/gam-sae/src/manifold/construction.rs:10012: 10012 lines; limit is 10000
//!     error: crates/gam-sae/src/manifold/tests.rs:10114: 10114 lines; limit is 10000
//!   error: 3 ban violations across 2 rules
//!
//! so `cargo build` / `cargo test` exit 101 and the `gamfit` wheel cannot be
//! built. (A *cached* `target/` can hide it: when the build-script fingerprint
//! is still valid the script is skipped and a stale incremental build appears
//! to succeed — which is how this rode onto `main`. A fresh `--release` build,
//! the wheel build, and CI all re-run it and fail.)
//!
//! While the bug is present the crate cannot build, so this integration-test
//! target cannot even be produced — `cargo test` fails. Once the two files are
//! split into cohesively-named submodules (each <= 10_000 lines), the crate
//! builds again and the check below — which counts newline bytes exactly as the
//! `build.rs` audit does, over every `.rs` file under `crates/gam-sae/src/` —
//! finds no oversized file and the test passes, with no further edits.

use std::fs;
use std::path::{Path, PathBuf};

/// Same line definition as `build.rs::count_file_lines`: count of `\n` bytes.
fn newline_count(path: &Path) -> usize {
    let bytes = fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    bytes.iter().filter(|byte| **byte == b'\n').count()
}

/// Collect every `.rs` file under `dir`, recursively.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn gam_sae_manifold_sources_respect_the_10k_line_gate() {
    // `CARGO_MANIFEST_DIR` is the workspace root (the `gam` crate lives there).
    const LIMIT: usize = 10_000;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sae_src = root.join("crates/gam-sae/src");
    assert!(
        sae_src.is_dir(),
        "expected gam-sae source tree at {}",
        sae_src.display()
    );

    let mut files = Vec::new();
    rust_files(&sae_src, &mut files);
    assert!(
        !files.is_empty(),
        "no .rs files found under {}",
        sae_src.display()
    );

    let mut oversized: Vec<(String, usize)> = Vec::new();
    for file in &files {
        let lines = newline_count(file);
        if lines > LIMIT {
            let rel = file
                .strip_prefix(&root)
                .unwrap_or(file)
                .to_string_lossy()
                .into_owned();
            oversized.push((rel, lines));
        }
    }

    assert!(
        oversized.is_empty(),
        "tracked gam-sae source file(s) exceed the {LIMIT}-line gate (issue #780); \
         build.rs aborts the whole workspace build on these: {oversized:?}"
    );
}
