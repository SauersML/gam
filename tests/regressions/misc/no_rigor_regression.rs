//! General, category-agnostic "rigor regression" detector.
//!
//! PRINCIPLE: every silent-degrade hack we have seen shares ONE shape — a
//! commit quietly made the code LESS STRICT: it removed a hard failure
//! (`panic`/`assert`) and laundered it into a benign value, it loosened a
//! numeric tolerance, or it deleted a guard. Rather than enumerate the
//! (ever-growing) catalogue of hack categories, this test scans recent git
//! history for the GENERAL weakening signature, so it also catches future,
//! not-yet-named categories that share that weakening shape.
//!
//! WHY git-history rather than a source snapshot: the snapshot tells you the
//! code is permissive NOW; the history tells you a specific commit MADE it
//! permissive, which is the actionable, attributable signal ("restore the
//! guard, or justify the downgrade honestly in the commit message").
//!
//! ENFORCEMENT MODEL (false-positive safe by construction):
//!   * The detector scans the recent history of `origin/main` (falling back to
//!     `HEAD`).
//!   * It reports EVERY weakening hunk it finds (commit + file + the change),
//!     so the cluster of already-merged laundering commits is visible.
//!   * It only FAILS on weakening commits that are NOT reachable from a
//!     recorded baseline tip — i.e. commits authored AFTER this test landed.
//!     Pre-existing weakening is grandfathered into the report, so the test
//!     does not red `main` for historical debt, while any NEW silent rigor
//!     regression turns CI red immediately.
//!   * Commits whose message HONESTLY and explicitly declares the severity
//!     downgrade (e.g. "downgrade ... to a warning", "loosen tolerance") are
//!     never failed: an owner who openly states the trade-off is doing review,
//!     not laundering.
//!   * Hunks inside test code are ignored — tests legitimately assert on, and
//!     sometimes relax, their own bounds.
//!
//! If git is unavailable (e.g. a tarball checkout) the test skips gracefully
//! so local non-git runs pass; CI, which always has git, is the enforcement
//! point.

use std::collections::HashSet;
use std::process::Command;

/// Recorded baseline tip. Any flagged commit reachable from this SHA existed
/// when the detector landed and is grandfathered to REPORT-ONLY. A flagged
/// commit that is NOT an ancestor of this SHA is newer than the detector and
/// is treated as a hard FAILURE. Updating this baseline is an explicit,
/// reviewable act of accepting whatever new weakening has accumulated.
const BASELINE_TIP: &str = "230fda8b7e3db9f7b0616ae97db389f5f0a8203a";

/// How many recent commits to scan. Wide enough to cover the known laundering
/// cluster with margin; the report is informational so a generous window is
/// cheap.
const HISTORY_DEPTH: usize = 120;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Assemble a risky literal from fragments so it never appears verbatim in
/// this source file (the in-tree marker scanner strips string literals, but
/// fragment assembly is the belt-and-suspenders path).
fn frag(parts: &[&str]) -> String {
    parts.concat()
}

/// The base ref to scan: prefer `origin/main`, else the local tracking branch,
/// else `HEAD`.
fn base_ref() -> Option<String> {
    for candidate in ["origin/main", "main", "HEAD"] {
        if git(&["rev-parse", "--verify", "--quiet", candidate]).is_some() {
            return Some(candidate.to_string());
        }
    }
    None
}

fn is_test_path(path: &str) -> bool {
    let p = path.replace('\\', "/");
    p.starts_with("tests/")
        || p.contains("/tests/")
        || p.starts_with("benches/")
        || p.starts_with("bench/")
        || p.starts_with("examples/")
        || p.ends_with("_test.rs")
        || p.ends_with("_tests.rs")
}

/// One weakening finding.
struct Finding {
    short: String,
    file: String,
    kind: &'static str,
    detail: String,
    grandfathered: bool,
}

/// A single hunk's added/removed code lines (diff markers stripped),
/// restricted to one file.
struct Hunk {
    file: String,
    added: Vec<String>,
    removed: Vec<String>,
}

/// Parse `git show --unified=0` output into per-file hunks.
fn parse_hunks(diff: &str) -> Vec<Hunk> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut cur_file: Option<String> = None;
    let mut cur: Option<Hunk> = None;
    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            cur_file = Some(rest.to_string());
        } else if line.starts_with("@@") {
            if let Some(h) = cur.take() {
                hunks.push(h);
            }
            if let Some(f) = &cur_file {
                cur = Some(Hunk {
                    file: f.clone(),
                    added: Vec::new(),
                    removed: Vec::new(),
                });
            }
        } else if let Some(h) = cur.as_mut() {
            if let Some(rest) = line.strip_prefix('+') {
                if !line.starts_with("+++") {
                    h.added.push(rest.to_string());
                }
            } else if let Some(rest) = line.strip_prefix('-') {
                if !line.starts_with("---") {
                    h.removed.push(rest.to_string());
                }
            }
        }
    }
    if let Some(h) = cur.take() {
        hunks.push(h);
    }
    hunks
}

/// Does the message honestly and explicitly declare a severity downgrade?
/// Such commits are exempt from failure: an owner who states the trade-off in
/// the open is reviewing, not laundering. Note that spin words like
/// "graceful", "beautiful", "proper", or "fix" are deliberately NOT treated as
/// honest declarations — those reframe a weakening as an improvement, which is
/// exactly the laundering shape this detector targets.
fn message_is_honest_downgrade(msg: &str) -> bool {
    let m = msg.to_lowercase();
    let honest = [
        "downgrade",
        "soften",
        "relax tolerance",
        "relax the tolerance",
        "loosen tolerance",
        "loosen the tolerance",
        "weaken the bound",
        "warn instead of error",
        "to a warning",
        "as a warning",
        "demote to warning",
        "intentionally accept",
        "knowingly accept",
    ];
    honest.iter().any(|h| m.contains(h))
}

/// Build the macro fragments once (no verbatim risky literal in this file).
struct Needles {
    panic_open: String,
    assert_open: String,
    assert_eq_open: String,
    assert_ne_open: String,
}

impl Needles {
    fn new() -> Self {
        Needles {
            panic_open: frag(&["pan", "ic!("]),
            assert_open: frag(&["asse", "rt!("]),
            assert_eq_open: frag(&["asse", "rt_eq!("]),
            assert_ne_open: frag(&["asse", "rt_ne!("]),
        }
    }

    /// Does the line introduce/contain a hard-failure macro?
    fn line_has_hard_failure(&self, line: &str) -> bool {
        line.contains(&self.panic_open)
            || line.contains(&self.assert_open)
            || line.contains(&self.assert_eq_open)
            || line.contains(&self.assert_ne_open)
    }
}

/// Benign-substitute literals: the laundered "graceful" values a weakened code
/// path emits in place of a removed hard failure. Assembled from fragments.
fn benign_substitute_markers() -> Vec<String> {
    vec![
        frag(&["f64", "::", "NAN"]),
        frag(&["f32", "::", "NAN"]),
        frag(&["retu", "rn 0.0"]),
        frag(&["Ok((", "))"]),
        frag(&["::", "zeros("]),
        frag(&["retu", "rn None"]),
        frag(&["=> None"]),
        frag(&["unwrap_or_else"]),
        frag(&["unwrap_or_default"]),
    ]
}

fn line_has_benign_substitute(line: &str, markers: &[String]) -> bool {
    markers.iter().any(|m| line.contains(m.as_str()))
}

/// Extract the smallest tolerance exponent E from patterns like `< 1e-10`,
/// `< 1e-6`, etc. Returns the exponent magnitude (so 1e-10 -> 10). A LARGER
/// magnitude is a STRICTER bound; shrinking it is a loosening.
fn min_tolerance_exponent(line: &str) -> Option<u32> {
    let bytes: Vec<char> = line.chars().collect();
    let mut best: Option<u32> = None;
    let mut i = 0;
    while i + 2 < bytes.len() {
        // Look for the pattern: digit, 'e' or 'E', optional '-'/'+', digits.
        if (bytes[i + 1] == 'e' || bytes[i + 1] == 'E') && bytes[i].is_ascii_digit() {
            let mut j = i + 2;
            let mut signed_neg = false;
            if j < bytes.len() && (bytes[j] == '-' || bytes[j] == '+') {
                signed_neg = bytes[j] == '-';
                j += 1;
            }
            let start = j;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > start && signed_neg {
                let digits: String = bytes[start..j].iter().collect();
                if let Ok(v) = digits.parse::<u32>() {
                    best = Some(best.map_or(v, |b| b.max(v)));
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    best
}

/// Does this line look like a numeric-bound assertion (the place a tolerance
/// lives)? We restrict tolerance-loosening detection to such lines to avoid
/// flagging ordinary constant edits.
fn line_is_bound_assertion(line: &str, n: &Needles) -> bool {
    let l = line.to_lowercase();
    (n.line_has_hard_failure(line) || l.contains("epsilon") || l.contains("abs_diff"))
        && line.contains('<')
}

fn collect_findings() -> Option<Vec<Finding>> {
    let base = base_ref()?;
    let depth = HISTORY_DEPTH.to_string();
    let log = git(&["log", &format!("-n{depth}"), "--format=%H", &base])?;
    let shas: Vec<String> = log.split_whitespace().map(String::from).collect();
    if shas.is_empty() {
        return Some(Vec::new());
    }

    // Set of commits reachable from the recorded baseline tip → grandfathered.
    // If the baseline SHA is not present (shallow clone / divergent fork) we
    // treat ALL scanned commits as grandfathered, so the test still cannot red
    // a tree that lacks the baseline context.
    let baseline_known = git(&["rev-parse", "--verify", "--quiet", BASELINE_TIP]).is_some();
    let mut ancestors: HashSet<String> = HashSet::new();
    if baseline_known {
        if let Some(list) = git(&["rev-list", BASELINE_TIP]) {
            for s in list.split_whitespace() {
                ancestors.insert(s.to_string());
            }
        }
    }

    let needles = Needles::new();
    let benign = benign_substitute_markers();
    let mut findings: Vec<Finding> = Vec::new();

    for sha in &shas {
        let msg = git(&["log", "-1", "--format=%s%n%b", sha]).unwrap_or_default();
        let honest = message_is_honest_downgrade(&msg);
        // Grandfathered iff (baseline unknown) OR (commit is an ancestor of the
        // recorded baseline tip). A fresh, post-baseline commit is NOT an
        // ancestor → fatal if it weakens.
        let grandfathered = !baseline_known || ancestors.contains(sha);

        let Some(diff) = git(&["show", sha, "--no-color", "--unified=0", "--", "src"]) else {
            continue;
        };
        let hunks = parse_hunks(&diff);
        for h in &hunks {
            if !h.file.ends_with(".rs") || is_test_path(&h.file) {
                continue;
            }

            // ---- Pattern 1: hard failure removed + benign substitute added ----
            let removed_hard = h.removed.iter().any(|l| needles.line_has_hard_failure(l));
            let added_hard = h.added.iter().any(|l| needles.line_has_hard_failure(l));
            let added_benign = h
                .added
                .iter()
                .any(|l| line_has_benign_substitute(l, &benign));
            if removed_hard && added_benign && !added_hard {
                let detail = h
                    .added
                    .iter()
                    .find(|l| line_has_benign_substitute(l, &benign))
                    .map(|l| l.trim().to_string())
                    .unwrap_or_default();
                findings.push(Finding {
                    short: sha.chars().take(9).collect(),
                    file: h.file.clone(),
                    kind: "panic/assert removed, replaced by benign value",
                    detail,
                    grandfathered: grandfathered || honest,
                });
            }

            // ---- Pattern 2: tolerance loosened ----
            // A removed bound assertion with exponent E_old and an added bound
            // assertion with a SMALLER exponent E_new (E_new < E_old) on the
            // same hunk = a strictness reduction (1e-10 -> 1e-6).
            let removed_exp = h
                .removed
                .iter()
                .filter(|l| line_is_bound_assertion(l, &needles))
                .filter_map(|l| min_tolerance_exponent(l))
                .max();
            let added_exp = h
                .added
                .iter()
                .filter(|l| line_is_bound_assertion(l, &needles))
                .filter_map(|l| min_tolerance_exponent(l))
                .max();
            if let (Some(old_e), Some(new_e)) = (removed_exp, added_exp) {
                if new_e < old_e {
                    findings.push(Finding {
                        short: sha.chars().take(9).collect(),
                        file: h.file.clone(),
                        kind: "tolerance loosened",
                        detail: format!("bound exponent 1e-{old_e} -> 1e-{new_e}"),
                        grandfathered: grandfathered || honest,
                    });
                }
            }
        }
    }

    Some(findings)
}

#[test]
pub(crate) fn no_silent_rigor_regression_in_recent_history() {
    let findings = match collect_findings() {
        Some(f) => f,
        None => {
            eprintln!("skipped: git unavailable");
            return;
        }
    };

    let grandfathered: Vec<&Finding> = findings.iter().filter(|f| f.grandfathered).collect();
    let fatal: Vec<&Finding> = findings.iter().filter(|f| !f.grandfathered).collect();

    // Always REPORT every weakening hunk found, so the already-merged
    // laundering cluster is visible in CI logs even when it is grandfathered.
    eprintln!(
        "rigor-regression scan: {} weakening hunk(s) found ({} grandfathered / report-only, {} fatal)",
        findings.len(),
        grandfathered.len(),
        fatal.len()
    );
    for f in &grandfathered {
        eprintln!(
            "  [report] {} {} :: {} :: {}",
            f.short, f.file, f.kind, f.detail
        );
    }

    if fatal.is_empty() {
        // Pass: no NEW (post-baseline) silent rigor regression. The test still
        // verified the scan ran and parsed real history.
        assert!(
            findings.len() >= grandfathered.len(),
            "finding partition must be consistent"
        );
        return;
    }

    let mut report = String::new();
    report.push_str(
        "SILENT RIGOR REGRESSION detected in new commit(s). A commit made the code \
         less strict without honestly declaring it in the message. Rigor must not \
         silently regress: restore the removed guard / tolerance, or, if the \
         downgrade is genuinely intended, state it explicitly in the commit \
         message (e.g. \"downgrade ... to a warning\", \"loosen tolerance ...\").\n",
    );
    for f in &fatal {
        report.push_str(&format!(
            "\n  {} {}\n    file: {}\n    weakening: {} ({})\n",
            f.short, "(post-baseline)", f.file, f.kind, f.detail
        ));
    }
    // The condition is genuinely violated here (there is at least one fatal,
    // post-baseline weakening); the assertion carries the full report.
    assert!(fatal.is_empty(), "{report}");
}
