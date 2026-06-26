//! Bug hunt: a binomial `s(x) + link(type=sas)` fit aborts through the standard
//! CLI fit path with
//!
//!   "Parameter constraint violation: Lambda count mismatch:
//!    expected 2 lambdas for 2 penalties, got 4"
//!
//! The sinh-arcsinh (`sas`) link is one of the documented parameterized binomial
//! inverse links (docs/families-and-links.md, docs/formulas.md). Its two learned
//! parameters (`epsilon`, `log_delta`) are optimized jointly with the smoothing
//! log-λ in a single augmented outer vector `θ = [ρ_smooth …, ε, log_δ]`, handled
//! by the "mixture/SAS flexible link" arm of `optimize_external_design…`
//! (src/solver/estimate/optimizer.rs).
//!
//! ROOT CAUSE (confirmed by backtrace): after the outer optimizer converges, that
//! arm calls the convergence-guard re-evaluation
//!
//!   run_outer_inner_cap_guard(&mut reml_state, &outer_result.rho,
//!                             RemlInnerCapGuardArm::MixtureSas)   // optimizer.rs:1876
//!
//! passing the FULL augmented `outer_result.rho` (smoothing log-λ AND the two SAS
//! link params). `run_outer_inner_cap_guard` (optimizer.rs:135) forwards it
//! verbatim to `state.compute_cost(rho)`, which treats the *entire* vector as
//! smoothing log-λ — `fit_model_for_fixed_rho` exponentiates all of it into
//! `lambdas` and `stable_reparameterization_engine_canonical` →
//! `stable_reparameterizationwith_invariant` (src/terms/construction.rs:1977)
//! rejects it: there are only 2 penalty blocks (the `s(x)` double penalty) but 4
//! "lambdas" arrive. The standard-REML arm (optimizer.rs:1512) passes a
//! smoothing-only `rho`, so it never trips this. The very next line of the SAS
//! arm (optimizer.rs:1881) DOES slice the link params off
//! (`final_rho = outer_result.rho.slice(s![..k])`) before the accept-fit — the
//! guard call one statement earlier simply forgot to.
//!
//! The guard only re-evaluates when the adaptive outer-inner-cap schedule was
//! lifted during the search (`prev_cap != 0`), which the standard CLI fit path
//! does on this dataset — hence this is exercised through the `gam` binary.
//!
//! This test fits a clean, well-separated binomial SAS-link model through the
//! real `gam fit` CLI and asserts the fit SUCCEEDS. Before the fix it aborts with
//! the lambda-count mismatch above; once the guard slices the smoothing block out
//! of the augmented θ (as the accept-fit on the next line already does), the
//! re-evaluation runs on 2 lambdas and the fit completes.

use std::process::Command;

#[test]
fn sas_link_binomial_fit_does_not_abort_with_lambda_count_mismatch() {
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/bug_hunt_sas_link_cap_guard.csv"
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

    // The precise defect: the SAS link params leak into the smoothing-λ count.
    assert!(
        !stdout.contains("Lambda count mismatch") && !stderr.contains("Lambda count mismatch"),
        "SAS-link fit aborted with a smoothing-λ count mismatch — the augmented θ \
         (smoothing log-λ + SAS ε/log_δ) was passed whole to the convergence-guard \
         re-eval (optimizer.rs:1876) instead of being sliced to the smoothing block.\n\
         stdout tail: {}\nstderr: {}",
        stdout.lines().rev().take(6).collect::<Vec<_>>().join("\n"),
        stderr.trim()
    );

    // And the documented link must actually fit, not merely dodge this one error.
    assert!(
        output.status.success(),
        "gam fit with link(type=sas) failed (exit {:?}).\nstderr: {}",
        output.status.code(),
        stderr.trim()
    );
}
