//! Regression: a binomial `s(x) + link(type=sas)` fit through the standard CLI
//! fit path must succeed.
//!
//! It once aborted with
//!
//!   "The P-IRLS inner loop did not converge within 3 iterations.
//!    Last gradient norm was 9.798574e0."
//!
//! because an outer-driven inner-PIRLS iteration cap, throttled to 3 during
//! the ARC search, leaked into the final evaluation at the optimum, whose
//! P-IRLS needs many more iterations (the SAS search drives η to extreme
//! magnitudes). The capped `MaxIterationsReached` was escalated to a fatal
//! `EstimationError::PirlsDidNotConverge` and the whole fit aborted.
//!
//! #3536 removed the outer-driven cap entirely: every inner solve runs under
//! the configured P-IRLS budget at one tolerance. This test pins that the fit
//! completes and never reports a 3-iteration budget.

use std::process::Command;

#[test]
fn sas_link_binomial_fit_finalize_does_not_abort_under_throttled_inner_cap() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_sas_link_finalize_cap.csv"
    );
    let out = tempfile::Builder::new()
        .suffix(".gam")
        .tempfile()
        .expect("temp output path");

    let output = Command::new(gam_test_support::gam_binary!())
        .arg("fit")
        .arg(fixture)
        .arg("y ~ s(x) + link(type=sas)")
        .arg("--out")
        .arg(out.path())
        .output()
        .expect("spawn gam fit");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");

    // The original defect: a search-time inner-iteration cap (≈3) applied to
    // the FINAL evaluation at the optimum, whose P-IRLS needs more iterations.
    assert!(
        !combined.contains("did not converge within 3 iterations"),
        "SAS-link fit aborted under a 3-iteration inner-PIRLS budget; every \
         inner solve must run under the configured budget (#3536).\n\
         stderr tail: {}",
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );

    // And the documented link must actually fit through the CLI.
    assert!(
        output.status.success(),
        "gam fit with link(type=sas) failed (exit {:?}).\nstderr tail: {}",
        output.status.code(),
        stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n")
    );
}
