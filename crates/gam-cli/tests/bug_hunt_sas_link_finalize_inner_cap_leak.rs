//! Bug hunt: a binomial `s(x) + link(type=sas)` fit aborts through the standard
//! CLI fit path with
//!
//!   "The P-IRLS inner loop did not converge within 3 iterations.
//!    Last gradient norm was 9.798574e0."
//!
//! This is a SECOND, distinct failure mode of the parameterized-link outer arm
//! from the lambda-count mismatch (the cap-guard at optimizer.rs:1876). Here the
//! lambda count is fine; the abort is the adaptive inner-PIRLS iteration cap
//! leaking into the FINAL evaluation of an otherwise-converged fit.
//!
//! ROOT CAUSE (confirmed by backtrace):
//!
//!   gradient_hessian.rs:6527  Err(PirlsDidNotConverge { max_iterations: 3, .. })
//!   ...
//!   optimizer.rs:1767         obtain_eval_bundle (the SAS eval closure)
//!   objective.rs:752          ClosureObjective::eval_with_order
//!   objective.rs:306          finalize_outer_result
//!   run_plan.rs:1919          obj.finalize_outer_result(&result.rho, the_plan)?  // fatal `?`
//!   run.rs:1579 / 1403        run_outer_with_plan / run_outer
//!   optimizer.rs:1868         problem.run("mixture/SAS flexible link")
//!
//! During the outer search the ARC bridge schedule throttles the inner P-IRLS
//! iteration budget down to a small adaptive cap (the run log shows
//! `[OUTER schedule] inner-PIRLS cap transition (ARC bridge) ... new=3 (capped)`),
//! stored in `RemlState::outer_inner_cap`. When the outer optimum is reached,
//! `run_plan.rs:1919` calls `finalize_outer_result(&result.rho)` to produce the
//! final evaluation at θ̂ — but that path does NOT lift the cap first, so the
//! finalize P-IRLS runs under the 3-iteration cap, stops at
//! `MaxIterationsReached` (gradient still ‖g‖≈9.8, nowhere near the KKT point),
//! and `execute_pirls_if_needed` escalates that capped non-convergence to a fatal
//! `EstimationError::PirlsDidNotConverge` (gradient_hessian.rs:6500-6530). The
//! `?` at run_plan.rs:1919 then aborts the whole fit.
//!
//! The convergence-guard `run_outer_inner_cap_guard` (optimizer.rs:135) DOES
//! `outer_inner_cap.swap(0, …)` to lift the cap for an uncapped re-eval — but it
//! runs at optimizer.rs:1876, AFTER `run_outer` (and thus the fatal finalize)
//! has already returned. The finalize eval inside `run_outer` should likewise
//! run at full inner budget (the gradient_hessian.rs:6469 comment promises "the
//! actual fit at full inner budget will"), not at the search-time throttle cap.
//!
//! Note the SAS outer search drives η to extreme magnitudes mid-search
//! (`max_eta=10240.0`, `g_norm_initial=NaN ... status=Converged` in the log),
//! which is why the inner solve at θ̂ still needs many iterations and the
//! 3-iteration cap is fatal rather than merely slow.
//!
//! This test fits a deterministic binomial SAS-link model through the real
//! `gam fit` CLI and asserts the fit SUCCEEDS. Before the fix it aborts with the
//! 3-iteration non-convergence above; once the finalize evaluation lifts the
//! adaptive inner cap (running at full budget, as the accept-fit and the
//! convergence guard already intend), the fit completes.

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

    let output = Command::new(env!("CARGO_BIN_EXE_gam"))
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

    // The precise defect: the adaptive search-time inner-iteration cap (≈3) is
    // applied to the FINAL evaluation at the optimum, whose P-IRLS needs more
    // iterations, so it aborts with a capped non-convergence.
    assert!(
        !combined.contains("did not converge within 3 iterations"),
        "SAS-link fit aborted because the finalize evaluation ran under the \
         throttled outer-search inner-PIRLS cap (≈3 iters) instead of full \
         budget — run_plan.rs:1919 finalize_outer_result does not lift \
         RemlState::outer_inner_cap, and a capped MaxIterationsReached is \
         escalated to a fatal PirlsDidNotConverge (gradient_hessian.rs:6527).\n\
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
