//! Bug hunt: the workspace does not build from a clean checkout at HEAD.
//!
//! Commit `a1d610d21` ("fix(sae/#2101): birth born atoms as real rank-2
//! circles, not DC-row constants (producer)", 2026-07-03) landed a new
//! diagnostic-probe source file
//! `crates/gam-sae/src/manifold/tests_2101_birth_locus_probe.rs` (wired into the
//! module tree by `crates/gam-sae/src/manifold/mod.rs:194`,
//! `mod tests_2101_birth_locus_probe;`). It contains TWO `#[test]` functions —
//!
//!   * `probe_2101_birth_locus_disjoint_6circle_ordered_beta_bernoulli`   (line 32)
//!   * `probe_2101_proper_circle_seed_survival`          (line 156)
//!
//! — and each is a pure `eprintln!`-only trajectory dump. Neither body contains
//! a single assertion-shaped construct: no `assert!` / `assert_eq!` /
//! `assert_ne!`, no `panic!` / `unreachable!` / `todo!`, no propagating `?`, and
//! no `#[should_panic]`. (They call `.expect(...)` / `.unwrap()`, but the ban
//! scanner deliberately does NOT count those — see
//! `build.rs::line_is_assertion_shaped`.)
//!
//! The workspace-root `build.rs` runs a ban scanner
//! (`scan_for_useless_tests`, section title
//! "#[test] function without assertions (test must verify something — add
//! assert! / assert_eq! / ? / #[should_panic] or delete the test)") over every
//! `.rs` file under the tree. Both probes trip it, so on any state that forces
//! `build.rs` to re-run — a fresh `cargo build`, `cargo test`, a `--release`
//! build, or the `maturin` wheel build — the build script aborts:
//!
//!   error: crates/gam-sae/src/manifold/tests_2101_birth_locus_probe.rs:156: #[test]
//!   summary: 2  #[test] function without assertions ...
//!   💥 maturin failed / Cargo build finished with "exit status: 101"
//!
//! The root `gam` crate is a (transitive) dependency of `gam-pyffi`, so the
//! `gamfit` Python wheel build hits the very same abort — `gamfit` cannot be
//! built or imported at HEAD. (A *cached* `target/` hides it: a still-valid
//! build-script fingerprint skips the scan, which is how this rode onto `main`.
//! A fresh `--release`/wheel build and CI all re-run it and fail.)
//!
//! While the bug is present the workspace cannot build, so this integration
//! test target cannot even be compiled — `cargo test` fails to produce it, which
//! is the failing state. Once the probes carry a real assertion (converting them
//! into genuine pass/fail guards), are moved out of the ban's reach, or are
//! deleted, the workspace builds again and the check below — which re-implements
//! the ban's own rule over that one file — finds nothing and passes, with no
//! further edits.

use std::fs;
use std::path::PathBuf;

/// The freshly-added probe file, relative to the workspace root
/// (`CARGO_MANIFEST_DIR` is the root, where the `gam` crate lives).
fn probe_file() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates/gam-sae/src/manifold/tests_2101_birth_locus_probe.rs")
}

/// Strip a `//` line comment so a mention of an assertion token inside comment
/// prose is not counted, mirroring `build.rs`'s comment masking.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

/// A propagating `?` operator: a `?` immediately followed (after optional
/// spaces) by end-of-line or one of `; , . )`. Mirrors
/// `build.rs::line_contains_propagating_question`.
fn has_propagating_question(code: &str) -> bool {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        if bytes[i] == b'?' {
            let mut k = i + 1;
            while k < n && bytes[k] == b' ' {
                k += 1;
            }
            if k == n || matches!(bytes[k], b';' | b',' | b'.' | b')') {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// True when a code line carries an assertion-shaped construct, matching the
/// set recognised by `build.rs::line_is_assertion_shaped`.
fn line_is_assertion_shaped(code: &str) -> bool {
    const MACROS: [&str; 9] = [
        "assert!(",
        "assert_eq!(",
        "assert_ne!(",
        "debug_assert!(",
        "debug_assert_eq!(",
        "debug_assert_ne!(",
        "panic!(",
        "unreachable!(",
        "todo!(",
    ];
    MACROS.iter().any(|m| code.contains(m))
        || code.contains("unimplemented!(")
        || has_propagating_question(code)
}

/// Report each `#[test]` function in `content` whose body contains no
/// assertion-shaped construct and is not guarded by `#[should_panic]`.
/// Returns `(line, signature)` pairs (1-based line of the `fn`).
fn assertionless_tests(content: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let mut offenders = Vec::new();
    let mut i = 0usize;
    while i < n {
        let stripped = strip_line_comment(lines[i]);
        if !stripped.contains("#[test]") {
            i += 1;
            continue;
        }
        // Walk forward past attributes/blank/comment lines to the `fn` line,
        // noting any `#[should_panic]` on the way (it exempts the test).
        let mut has_should_panic = stripped.contains("#[should_panic");
        let mut j = i + 1;
        while j < n {
            let t = strip_line_comment(lines[j]);
            let tt = t.trim();
            if tt.is_empty() {
                j += 1;
                continue;
            }
            if tt.starts_with("#[") || tt.starts_with("#![") {
                if t.contains("#[should_panic") {
                    has_should_panic = true;
                }
                j += 1;
                continue;
            }
            break;
        }
        if j >= n {
            break;
        }
        let sig_line = j;
        let signature = lines[sig_line].trim().to_string();
        // Find the body braces by counting `{`/`}` from the signature onward.
        let mut depth: i32 = 0;
        let mut started = false;
        let mut body_has_assertion = false;
        let mut k = sig_line;
        while k < n {
            let code = strip_line_comment(lines[k]);
            if started && line_is_assertion_shaped(code) {
                body_has_assertion = true;
            }
            for ch in code.chars() {
                if ch == '{' {
                    depth += 1;
                    started = true;
                } else if ch == '}' {
                    depth -= 1;
                }
            }
            if started && depth == 0 {
                break;
            }
            k += 1;
        }
        if !has_should_panic && !body_has_assertion {
            offenders.push((sig_line + 1, signature));
        }
        i = k + 1;
    }
    offenders
}

/// The probe file must not contain any assertion-less `#[test]` function.
///
/// Fix-agnostic: a missing file (the probes were deleted or moved) yields no
/// offenders and passes, exactly as adding real assertions does.
#[test]
fn birth_locus_probe_has_no_assertionless_tests() {
    let path = probe_file();
    let content = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(_) => return, // file removed → the ban no longer fires → pass
    };
    let offenders = assertionless_tests(&content);
    assert!(
        offenders.is_empty(),
        "crates/gam-sae/src/manifold/tests_2101_birth_locus_probe.rs has {} \
         #[test] function(s) with no assertion-shaped construct, which trips the \
         build.rs `scan_for_useless_tests` ban and aborts the whole workspace \
         build (cargo build/test, --release, and the gamfit wheel):\n{}",
        offenders.len(),
        offenders
            .iter()
            .map(|(ln, sig)| format!("  line {ln}: {sig}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// Guard the guard: the scanner must actually classify an `eprintln!`-only
/// `#[test]` as assertion-less, and an `assert!`-bearing one as fine. Without
/// this, a scanner that silently matched nothing would make the check above
/// vacuously green.
#[test]
fn scanner_flags_assertionless_and_clears_asserting_tests() {
    let sample = r#"
#[test]
fn probe_only_prints() {
    eprintln!("no assertions here");
    let _ = do_thing().expect("value");
}

#[test]
fn real_test() {
    assert_eq!(2 + 2, 4);
}

#[test]
#[should_panic(expected = "boom")]
fn expected_panic() {
    trigger();
}
"#;
    let offenders = assertionless_tests(sample);
    assert_eq!(
        offenders.len(),
        1,
        "expected exactly the print-only test to be flagged, got {offenders:?}"
    );
    assert!(
        offenders[0].1.contains("probe_only_prints"),
        "wrong test flagged: {offenders:?}"
    );
}
