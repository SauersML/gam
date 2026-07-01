//! Bug hunt: the workspace does not build from a clean checkout at HEAD — and
//! the break rode all the way onto a RELEASE commit
//! (`0e6f90fde release: gam 0.3.139 / gamfit 0.1.241`).
//!
//! `build.rs` runs an always-fatal hygiene ban scanner over every tracked file
//! before the crate compiles; any violation makes the build script call
//! `std::process::exit(1)` (build.rs:795), so `cargo build`, `cargo test`, a
//! fresh `--release` build, and the `maturin` wheel build all abort. At HEAD the
//! scanner reports 13 violations across 6 rules. The three most clear-cut are
//! fresh scaffolding regressions:
//!
//!   1. A TEMPORARY EXPLORATORY PROBE committed into the production source tree:
//!      `crates/gam-solve/src/estimate/gaussian_obs_coverage_probe.rs` — its own
//!      header says "EXPLORATORY probe (temporary) for #1765". It trips the
//!      `println!` ban SEVEN times (`eprintln!(...)`, lines 172-195), the
//!      `#[test] without assertions` rule (line 202), and — via its module
//!      declaration `#[cfg(test)] mod gaussian_obs_coverage_probe;`
//!      (`crates/gam-solve/src/estimate/mod.rs:130`) — the `#[cfg(test)] on a
//!      src/ item` rule (the scanner only exempts a private `mod tests` /
//!      `mod test_support` / `mod tests_*` / `mod *_tests`, never an arbitrary
//!      probe module).
//!
//!   2. `crates/gam-sae/src/manifold/construction.rs` is 10059 lines — over the
//!      10_000-line tracked-file limit (build.rs `scan_for_oversized_tracked_files`,
//!      issue #780).
//!
//!   3. `crates/gam-sae/src/sparse_dict/tests.rs:549` carries a `#[ignore = ...]`
//!      attribute on `real_olmo_sparse_dict_ev_vs_k_parity`, and line 486 puts a
//!      redundant `#[cfg(test)]` on the `read_npy_f32_2d` helper `fn`.
//!
//! (The remaining violations: a stale `uv.lock:307 version = "0.1.240"` that the
//! 0.3.139/0.1.241 release bump did not update — the "non-latest gamfit version
//! reference" rule.)
//!
//! Observed (verbatim, on `cargo build -p gam`):
//!
//!     error: 13 ban violations across 6 rules
//!     error — println!  (7 hits)   ...gaussian_obs_coverage_probe.rs:172 eprintln!(...)
//!     error — #[test] function without assertions  (1 hit)
//!     error — #[cfg(test)] on src/ item  (2 hits)
//!     error — tracked file over 10k lines  (1 hit) ...construction.rs:10059
//!     error — #[ignore] test  (1 hit)
//!     error — non-latest gamfit version reference (expected 0.1.241)  (1 hit)
//!     cargo:warning=ban-scanner FAILED: 13 violation(s); build aborted
//!
//! Expected: a clean checkout / release tag builds; wheel + CLI produce.
//!
//! Root cause: exploratory/measurement scaffolding (the #1765 coverage probe,
//! the #1026 banked-data parity test) was committed without running the ban
//! scanner, and `construction.rs` grew past the split threshold. A *cached*
//! `target/` hides all of this (a valid build-script fingerprint skips the
//! re-scan), which is how it rode onto `main` and into the release; a fresh
//! build, the wheel build, and CI all re-run `build.rs` and fail.
//!
//! While the bug is present the crate cannot build, so this integration-test
//! target cannot even be produced — `cargo test` fails at compile. Once the
//! temporary probe is removed, `construction.rs` is split under 10k lines, and
//! the `#[ignore]` test is cleaned, the workspace builds again and the checks
//! below — which re-implement the simply-detectable rules over the offending
//! files — find nothing and the test passes, with no further edits. Each check
//! treats an absent file as "fixed" (a valid fix may delete the probe outright).

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Read a workspace-relative file as raw bytes, or `None` if it is absent.
fn read(rel: &str) -> Option<Vec<u8>> {
    std::fs::read(root().join(rel)).ok()
}

/// Strip a `//` line comment (byte-wise, offset-preserving) so a *mention* of a
/// banned construct inside comment prose is not counted — matching build.rs's
/// comment masking for these line-level checks.
fn strip_line_comment(line: &[u8]) -> &[u8] {
    let mut i = 0usize;
    while i + 1 < line.len() {
        if line[i] == b'/' && line[i + 1] == b'/' {
            return &line[..i];
        }
        i += 1;
    }
    line
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || hay.len() < needle.len() {
        return false;
    }
    hay.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn head_carries_no_build_aborting_scaffolding_ban_violations() {
    let mut offenders: Vec<String> = Vec::new();

    // 1) The temporary exploratory probe must not sit in production src/ with
    //    stdout/stderr printing or a bare `#[test]` (the `println!` /
    //    `#[test]-without-assertions` bans). A valid fix deletes the file.
    const PROBE: &str = "crates/gam-solve/src/estimate/gaussian_obs_coverage_probe.rs";
    if let Some(bytes) = read(PROBE) {
        for (idx, raw) in bytes.split(|&b| b == b'\n').enumerate() {
            let code = strip_line_comment(raw);
            if contains(code, b"println!(") || contains(code, b"eprintln!(") {
                offenders.push(format!(
                    "[println! in src/ probe] {PROBE}:{}",
                    idx + 1
                ));
            }
        }
    }

    // 2) construction.rs must be within the 10_000-line tracked-file limit.
    const BIG: &str = "crates/gam-sae/src/manifold/construction.rs";
    if let Some(bytes) = read(BIG) {
        // build.rs counts lines the same way `str::lines` does.
        let n_lines = String::from_utf8_lossy(&bytes).lines().count();
        if n_lines > 10_000 {
            offenders.push(format!(
                "[>10k lines] {BIG}: {n_lines} lines (limit 10000)"
            ));
        }
    }

    // 3) The sparse-dict test file must carry no `#[ignore]` attribute.
    const SD: &str = "crates/gam-sae/src/sparse_dict/tests.rs";
    if let Some(bytes) = read(SD) {
        for (idx, raw) in bytes.split(|&b| b == b'\n').enumerate() {
            let code = strip_line_comment(raw);
            let trimmed: Vec<u8> = code.iter().copied().skip_while(|b| b.is_ascii_whitespace()).collect();
            if trimmed.starts_with(b"#[ignore") {
                offenders.push(format!("[#[ignore] test] {SD}:{}", idx + 1));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "build.rs ban-scanner violation(s) present at HEAD; the scanner calls \
         std::process::exit(1), so `cargo build`, `cargo test`, and the gamfit \
         wheel all abort on a clean checkout of the 0.3.139/0.1.241 release. \
         Offenders: {offenders:#?}"
    );
}
