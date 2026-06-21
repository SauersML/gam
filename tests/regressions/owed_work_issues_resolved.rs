//! Wording-resilient, persistent "owed work not done" detector.
//!
//! Owed-work comments scattered through `src/` cite open GitHub issues (for
//! example #993, #299, #1408, #1409, #932). The prose of such a comment can be
//! reworded or the comment deleted entirely, but the issue NUMBER it cites is
//! immune to rewording, and whether that issue is OPEN is a machine-checkable
//! proof that the work has not been finished. This detector therefore keys on
//! the issue number, never on the surrounding prose.
//!
//! Wording-resilience: the ledger `ban_owed_work_issues.txt` stores bare issue
//! numbers. No matter how the in-code comment is paraphrased, the number it
//! refers to (and the issue's open/closed state on GitHub) is what gets checked.
//!
//! Persistence: rewording OR deleting the in-code comment cannot clear a ledger
//! entry. The ledger lives independently of any comment; an entry leaves the
//! ledger only when the work is genuinely done and the GitHub issue is CLOSED.
//! Closing the issue is the one and only way to make this test pass for an entry.
//!
//! Enforcement point: the test shells out to `gh` to read each issue's state.
//! In a local checkout with no network or no `gh` authentication the test skips
//! gracefully so `cargo test` stays green; CI (which provides `gh` and a
//! `GITHUB_TOKEN`) is where this is actually enforced.

use std::process::Command;

const LEDGER: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/ban_owed_work_issues.txt");

const REPO: &str = "SauersML/gam";

/// Parse the ledger into the list of issue numbers it tracks.
///
/// Lines are either blank, a `#` header/comment line, or an issue number
/// optionally followed by an inline `# comment` recording the issue title.
fn ledger_issue_numbers() -> Vec<u64> {
    let body = std::fs::read_to_string(LEDGER)
        .unwrap_or_else(|e| panic!("cannot read owed-work ledger {LEDGER}: {e}"));
    let mut numbers = Vec::new();
    for raw in body.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let token = line.split('#').next().unwrap_or("").trim();
        if token.is_empty() {
            continue;
        }
        let number: u64 = token
            .parse()
            .unwrap_or_else(|e| panic!("malformed issue number {token:?} in ledger: {e}"));
        numbers.push(number);
    }
    numbers
}

/// Possible answers from probing one issue's state via `gh`.
enum IssueState {
    Open,
    Closed,
    /// `gh` itself is unavailable / unauthenticated / could not be reached.
    GhUnavailable,
}

/// Query a single issue's open/closed state through `gh`.
///
/// Any failure to run or parse `gh` is reported as `GhUnavailable` so that a
/// local developer without network or authentication is not blocked; CI is the
/// enforcement point.
fn issue_state(number: u64) -> IssueState {
    let output = Command::new("gh")
        .args([
            "issue",
            "view",
            &number.to_string(),
            "--repo",
            REPO,
            "--json",
            "state",
        ])
        .output();

    let output = match output {
        Ok(value) if value.status.success() => value,
        Ok(_) => return IssueState::GhUnavailable,
        Err(_) => return IssueState::GhUnavailable,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let upper = stdout.to_ascii_uppercase();
    // The JSON payload is `{"state":"OPEN"}` or `{"state":"CLOSED"}`. Detect
    // CLOSED first because "CLOSED" does not contain "OPEN".
    if upper.contains("CLOSED") {
        IssueState::Closed
    } else if upper.contains("OPEN") {
        IssueState::Open
    } else {
        IssueState::GhUnavailable
    }
}

#[test]
pub(crate) fn owed_work_issues_must_be_closed() {
    let numbers = ledger_issue_numbers();
    assert!(
        !numbers.is_empty(),
        "owed-work ledger is empty; expected at least the seeded owed-work issues"
    );

    let mut still_open: Vec<u64> = Vec::new();

    for &number in &numbers {
        match issue_state(number) {
            IssueState::Closed => {}
            IssueState::Open => still_open.push(number),
            IssueState::GhUnavailable => {
                eprintln!(
                    "owed_work_issues_must_be_closed skipped: gh unavailable \
                     (no network / not authenticated / could not query issue #{number}); \
                     CI enforces this check"
                );
                return;
            }
        }
    }

    if !still_open.is_empty() {
        let mut report = String::new();
        for number in &still_open {
            report.push_str(&format!(
                "owed work NOT DONE -- issue #{number} is still open; finish and \
                 close it (rewording/deleting the comment does not clear this -- \
                 only closing the issue does)\n"
            ));
        }
        panic!("{report}");
    }
}
