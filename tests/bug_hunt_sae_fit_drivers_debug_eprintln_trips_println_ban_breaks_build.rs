//! Bug hunt: the workspace (and the `gamfit` wheel) does not build from a clean
//! checkout at HEAD — a stray debug `eprintln!` diagnostic trips the `println!`
//! substring ban.
//!
//! Commit `eb767f1ba` ("fix(sae/#2089): treat total co-collapse as a recoverable
//! infeasible-ρ probe + guard reseed numerics", 2026-07-04) left two debug
//! diagnostic dumps in library (non-test) code inside
//! `crates/gam-sae/src/manifold/fit_drivers.rs`, in the streaming Arrow–Schur
//! joint-fit driver `run_joint_fit_arrow_schur_streaming`:
//!
//!   * line 5136: `eprintln!("[DIAG cs={chunk_size} range=({start},{end})] …")`
//!     (gated on the small-`n` chunk branch)
//!   * line 5169: `eprintln!("[DIAG-ITER cs={chunk_size}] s_trace=… …")`
//!     (gated on `n_total <= 40`)
//!
//! The workspace-root `build.rs` runs a banned-substring hygiene scanner
//! (`scan_for_banned_substrings`) whose needle list (build.rs ~line 1630)
//! includes the literal `"println!("` — "Library code writing to stdout pollutes
//! downstream consumers." The scanner does a plain
//! `stripped_line.contains("println!(")` over every non-test `.rs` line, and
//! **`"eprintln!(".contains("println!(")` is `true`** (`e`+`println!(`). So each
//! of the two `eprintln!(` diagnostics is counted as a `println!` violation, and
//! on any state that forces `build.rs` to re-run — a fresh `cargo build`, `cargo
//! test`, a `--release` build, or the `maturin` wheel build — the build script
//! aborts:
//!
//!   error: crates/gam-sae/src/manifold/fit_drivers.rs:5169: eprintln!(
//!   summary: 2  println!
//!   💥 maturin failed / Cargo build finished with "exit status: 101"
//!
//! The root `gam` crate is a (transitive) dependency of `gam-pyffi`, so the
//! `gamfit` Python wheel build hits the very same abort — `gamfit` cannot be
//! built or imported at HEAD (`maturin develop`/`maturin build` both fail here;
//! observed directly). A *cached* `target/` hides it: a still-valid build-script
//! fingerprint skips the scan, which is how it rode onto `main`; a fresh
//! `--release` / wheel build re-runs the scanner and fails.
//!
//! These are stray debug instrumentation dumps (trajectory `s_trace` / gate
//! values), not intended output — the correct fix is to delete them (or route
//! through real logging), exactly as the ban is designed to force. `eprintln!`
//! with Display formatting is otherwise not caught by the *debug*-eprintln
//! scanner (`scan_for_debug_eprintln` only flags `{:?}`/`{:#?}`), so the
//! `println!`-substring rule is the only gate these two lines trip — which is
//! why the whole build hinges on them.
//!
//! While the bug is present the workspace cannot build, so this integration test
//! target cannot even be compiled — `cargo test` fails to produce it, which is
//! the failing state. Once the two diagnostics are removed (or moved out of the
//! ban's reach), the workspace builds again and the check below — which
//! re-implements the ban's own `println!(`-substring rule over that one file —
//! finds nothing and passes, with no further edits.
//!
//! Related build-break siblings: #2119, #2110, #2029 (each a different HEAD
//! commit whose new source trips a build.rs hygiene ban and breaks the wheel).

use std::fs;
use std::path::PathBuf;

/// The offending SAE driver file, relative to the workspace root
/// (`CARGO_MANIFEST_DIR` is the root, where the `gam` crate lives).
fn fit_drivers_file() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates/gam-sae/src/manifold/fit_drivers.rs")
}

/// The exact banned needle from `build.rs::banned_substrings` that
/// `eprintln!(` collides with.
const PRINTLN_NEEDLE: &str = "println!(";

/// Remove the contents of double-quoted string literals (replacing them with an
/// empty pair) so a mention of the needle *inside a string* — the common false
/// positive for long multi-line error messages — is not counted. Mirrors the
/// intent of `build.rs::strip_file_lines`. Line-local approximation: good enough
/// because the two `eprintln!(` call sites are bare code, and the ban's own
/// stripper likewise leaves the macro invocation token intact.
fn strip_string_literals(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut in_string = false;
    let mut escaped = false;
    for ch in line.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        out.push(ch);
    }
    out
}

/// Strip a `//` line comment so a mention of the needle in comment prose is not
/// counted.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

/// Every 1-based line in `content` whose code (comments and string literals
/// stripped) contains the banned `println!(` substring — this is exactly what
/// `scan_for_banned_substrings` flags, and it matches `eprintln!(` too.
fn println_substring_offenders(content: &str) -> Vec<(usize, String)> {
    content
        .lines()
        .enumerate()
        .filter_map(|(idx, raw)| {
            let code = strip_string_literals(strip_line_comment(raw));
            if code.contains(PRINTLN_NEEDLE) {
                Some((idx + 1, raw.trim().to_string()))
            } else {
                None
            }
        })
        .collect()
}

/// `fit_drivers.rs` must contain no `println!(`-substring line (bare `println!`
/// OR `eprintln!` in library code), because the build.rs ban aborts the whole
/// workspace/wheel build on any such line.
///
/// Fix-agnostic: a missing file (the driver was renamed) yields no offenders and
/// passes, exactly as deleting the two diagnostics does.
#[test]
fn fit_drivers_has_no_println_substring_diagnostics() {
    let path = fit_drivers_file();
    let content = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(_) => return, // file removed/renamed → ban no longer fires here → pass
    };
    let offenders = println_substring_offenders(&content);
    assert!(
        offenders.is_empty(),
        "crates/gam-sae/src/manifold/fit_drivers.rs has {} line(s) containing the \
         banned `println!(` substring (bare `println!` or `eprintln!(` — the two \
         debug diagnostics left by eb767f1ba). The build.rs \
         `scan_for_banned_substrings` `println!` rule matches each and aborts the \
         whole workspace build (cargo build/test, --release, and the gamfit \
         wheel):\n{}",
        offenders.len(),
        offenders
            .iter()
            .map(|(ln, code)| format!("  line {ln}: {code}"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
}

/// Guard the guard: the substring rule must actually classify an `eprintln!(`
/// call as a `println!(` match (that collision is the whole mechanism), and must
/// leave innocuous lines alone. Without this, a scanner that silently matched
/// nothing would make the check above vacuously green.
#[test]
fn println_substring_rule_matches_eprintln_and_clears_innocuous() {
    let sample = "\
        let x = 1;\n\
        eprintln!(\"[DIAG] value={x:.3e}\");\n\
        let y = format!(\"println!( inside a string is fine\");\n\
        println!(\"stdout write\");\n\
        // a comment mentioning println!( must not count\n";
    let offenders = println_substring_offenders(sample);
    let lines: Vec<usize> = offenders.iter().map(|(ln, _)| *ln).collect();
    assert_eq!(
        lines,
        vec![2, 4],
        "expected the eprintln!( (line 2) and println!( (line 4) to be flagged, \
         and the in-string mention (line 3) plus the comment (line 5) to be \
         cleared; got {offenders:?}"
    );
}
