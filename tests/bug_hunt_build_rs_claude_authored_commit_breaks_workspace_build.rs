//! Bug hunt: the workspace does not build from a clean checkout at HEAD because
//! the most recent commit touching `build.rs` was authored by a non-human.
//!
//! `build.rs` ships a self-defense gate, `forbid_claude_build_rs_edits`
//! (build.rs:985), wired into `fn main()` at build.rs:45. On every build-script
//! run it queries
//!
//!     git log -1 --format=%an|%ae -- build.rs
//!
//! and, if the resulting author string contains `claude` or `anthropic`
//! (case-insensitive), **panics** (build.rs:1009):
//!
//!     thread 'main' panicked at build.rs:1010:9:
//!     Sorry Claude! build.rs was last edited by a non-human author
//!     (Claude|noreply@anthropic.com). ...
//!
//! Commit `d3fa40b806fbca43238a3d4601d5154dd4471904`
//! ("fix(build): ban-scanner follows turbofish generic-helper calls") edited
//! `build.rs` and is authored by `Claude <noreply@anthropic.com>`. It is the
//! last commit touching `build.rs`, so the gate now fires on every fresh build.
//!
//! The build script runs for the root `gam` crate, which the CLI, every
//! `tests/*.rs` integration target, and the `gamfit` wheel all depend on. With
//! the gate tripping, `cargo build`, `cargo test`, and `maturin build` all abort
//! with exit status 101:
//!
//!     error: failed to run custom build command for `gam v0.3.129`
//!     Caused by: process didn't exit successfully: build-script-build (exit status: 101)
//!       thread 'main' panicked at build.rs:1010:9: Sorry Claude! ...
//!
//! (A *cached* `target/` whose build-script fingerprint is still valid skips the
//! script and hides the breakage — which is how a green-looking incremental
//! build can coexist with a red clean build, the wheel build, and CI.)
//!
//! The gate is correct and must not be weakened or removed; the defect is that a
//! non-human (Claude) commit to `build.rs` was allowed onto `main`, leaving the
//! tree unbuildable. The remedy is for a human maintainer to re-commit `build.rs`
//! (e.g. `git commit --amend --reset-author`, or a fresh human-authored commit
//! that re-touches `build.rs`) so its last author is a human again.
//!
//! While the bug is present the `gam` crate's build script panics, so this
//! integration-test target cannot even be produced — `cargo test` fails. Once a
//! human re-authors the latest `build.rs` commit, the workspace builds again and
//! the check below — which re-runs the exact same `git log` query `build.rs`
//! itself uses and applies the same author predicate — finds a human author and
//! the test passes, with no further edits.

use std::path::PathBuf;
use std::process::Command;

/// Re-implementation of `build.rs::forbid_claude_build_rs_edits`' author
/// predicate: the offending author string contains `claude` or `anthropic`,
/// case-insensitively.
fn is_non_human_author(info: &str) -> bool {
    let lower = info.to_lowercase();
    lower.contains("claude") || lower.contains("anthropic")
}

#[test]
fn build_rs_last_commit_is_human_authored() {
    // `CARGO_MANIFEST_DIR` is the workspace root (the `gam` crate lives there),
    // exactly the `manifest_dir` build.rs queries against.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let output = Command::new("git")
        .arg("-C")
        .arg(&root)
        .arg("log")
        .arg("-1")
        .arg("--format=%an|%ae")
        .arg("--")
        .arg("build.rs")
        .output()
        .expect("failed to run git log for build.rs author audit");

    assert!(
        output.status.success(),
        "git log for build.rs author failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let info = String::from_utf8_lossy(&output.stdout);
    let info = info.trim().to_string();

    assert!(
        !info.is_empty(),
        "git reported no commit history for build.rs (cannot audit its author)"
    );

    assert!(
        !is_non_human_author(&info),
        "build.rs was last committed by a non-human author ({info}); build.rs's own \
         forbid_claude_build_rs_edits gate (build.rs:985) panics the build script on \
         this, so `cargo build`, `cargo test`, and the gamfit wheel all abort (exit \
         101). A human maintainer must re-author the latest build.rs commit."
    );
}
