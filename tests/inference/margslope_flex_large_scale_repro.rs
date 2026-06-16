//! Fast local reproducer for the FLEX bernoulli marginal-slope cycle-0 cost
//! cliff seen in the large-scale lane.
//!
//! Manual invocation:
//!
//! ```text
//! cargo test --release --test margslope_flex_large_scale_repro \
//!     -- --nocapture
//! ```
//!
//! The synthetic shape keeps the production code path active: probit
//! bernoulli marginal slope, score-warp and link-deviation FLEX blocks,
//! a joint 16D Duchon PC smooth (`centers=24`, `order=1`, `power=8`,
//! `length_scale=1.0`), a separate smooth age term, and a standard-normal
//! latent `z`.  The primary repro uses `n = DEFAULT_REPRO_N` and caps the
//! full blockwise fit at joint-Newton cycle 0 (`inner_max_cycles=1`) so the
//! printed wall time is a local proxy for the large-scale
//! `[PIRLS/blockwise joint-Newton] cycle 0/100` region.  Run under `samply`,
//! `cargo flamegraph`, or macOS `sample` for a flame graph/profile;
//! `--nocapture` preserves the per-fit phase summaries already emitted by
//! the solver.

#[path = "../test_support/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use gam::families::custom_family::BlockwiseFitOptions;
use gam::solver::outer_subsample::OuterScoreSubsample;
use margslope_flex_equivalence::{
    build_large_scale_shape_problem, cycle_capped_options, fit_problem,
};
use ndarray::Array1;
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_REPRO_N: usize = 50_000;
const DEFAULT_SMOKE_N: usize = 2_000;
const DEFAULT_WALL_BOUND: Duration = Duration::from_secs(300);
const FULL_OUTER_SMOKE_INNER_CYCLES: usize = 20;

#[derive(Clone, Debug)]
struct BetaDiff {
    index: usize,
    left: f64,
    right: f64,
    abs_diff: f64,
    rel_diff: f64,
}

#[derive(Clone, Debug)]
struct BetaEquivalenceReport {
    len: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    worst: Option<BetaDiff>,
}

fn fit_synthetic_beta(
    n: usize,
    inner_max_cycles: usize,
) -> Result<(Array1<f64>, margslope_flex_equivalence::FitTiming), String> {
    let problem = build_large_scale_shape_problem(n);
    let (fit, timing) = fit_problem(problem, cycle_capped_options(inner_max_cycles))?;
    Ok((fit.fit.beta, timing))
}

fn compare_beta(
    left: &Array1<f64>,
    right: &Array1<f64>,
    rel_tol: f64,
) -> Result<BetaEquivalenceReport, String> {
    if left.len() != right.len() {
        return Err(format!(
            "beta length mismatch: left={} right={}",
            left.len(),
            right.len()
        ));
    }
    let mut report = BetaEquivalenceReport {
        len: left.len(),
        max_abs_diff: 0.0,
        max_rel_diff: 0.0,
        worst: None,
    };
    for (index, (&a, &b)) in left.iter().zip(right.iter()).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(format!(
                "non-finite beta at index {index}: left={a} right={b}"
            ));
        }
        let abs_diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(1.0);
        let rel_diff = abs_diff / scale;
        if rel_diff > report.max_rel_diff || abs_diff > report.max_abs_diff {
            report.max_abs_diff = report.max_abs_diff.max(abs_diff);
            report.max_rel_diff = report.max_rel_diff.max(rel_diff);
            report.worst = Some(BetaDiff {
                index,
                left: a,
                right: b,
                abs_diff,
                rel_diff,
            });
        }
    }
    if report.max_rel_diff > rel_tol {
        let worst = report.worst.as_ref().expect("worst beta diff");
        return Err(format!(
            "beta mismatch: len={} max_abs={:.3e} max_rel={:.3e} rel_tol={:.3e} worst_index={} left={:.17e} right={:.17e} worst_abs={:.3e} worst_rel={:.3e}",
            report.len,
            report.max_abs_diff,
            report.max_rel_diff,
            rel_tol,
            worst.index,
            worst.left,
            worst.right,
            worst.abs_diff,
            worst.rel_diff
        ));
    }
    Ok(report)
}

fn assert_repeated_fit_beta_equivalent(
    n: usize,
    inner_max_cycles: usize,
    rel_tol: f64,
) -> BetaEquivalenceReport {
    let (left, left_timing) = fit_synthetic_beta(n, inner_max_cycles).expect("left synthetic fit");
    let (right, right_timing) =
        fit_synthetic_beta(n, inner_max_cycles).expect("right synthetic fit");
    let report = compare_beta(&left, &right, rel_tol).expect("synthetic fit beta equivalence");
    eprintln!(
        "[MS-FLEX-EQUIV-PASS] n={} inner_max_cycles={} beta_len={} max_abs={:.3e} max_rel={:.3e} left_elapsed_s={:.3} right_elapsed_s={:.3}",
        n,
        inner_max_cycles,
        report.len,
        report.max_abs_diff,
        report.max_rel_diff,
        left_timing.elapsed.as_secs_f64(),
        right_timing.elapsed.as_secs_f64()
    );
    report
}

#[test]
fn margslope_flex_large_scale_repro_cycle0() {
    gam::init_parallelism();
    let n = DEFAULT_REPRO_N;
    let bound = DEFAULT_WALL_BOUND;
    let problem = build_large_scale_shape_problem(n);
    let (out, timing) = fit_problem(problem, cycle_capped_options(1))
        .expect("large-scale FLEX margslope cycle-0 repro fit");
    eprintln!(
        "[MS-FLEX-LARGE_SCALE-REPRO] n={} inner_max_cycles=1 elapsed_s={:.3} outer_iters={} inner_cycles={} converged={} beta_len={}",
        n,
        timing.elapsed.as_secs_f64(),
        timing.outer_iterations,
        timing.inner_cycles,
        timing.outer_converged,
        out.fit.beta.len()
    );
    assert!(
        timing.inner_cycles >= 1,
        "fit did not enter joint-Newton cycle 0"
    );
    assert!(
        timing.elapsed <= bound,
        "cycle-0 repro exceeded wall bound: elapsed={:.3}s bound={:.3}s",
        timing.elapsed.as_secs_f64(),
        bound.as_secs_f64()
    );
}

#[test]
fn margslope_flex_beta_equivalence_smoke() {
    assert!(file!().ends_with(".rs"));
    gam::init_parallelism();
    let n = DEFAULT_SMOKE_N;
    let inner_cycles = 1usize;
    let rel_tol = 1e-10_f64;
    let report = assert_repeated_fit_beta_equivalent(n, inner_cycles, rel_tol);
    eprintln!(
        "[MS-FLEX-EQUIV-SMOKE] PASS beta_len={} max_abs={:.3e} max_rel={:.3e} rel_tol={:.3e}",
        report.len, report.max_abs_diff, report.max_rel_diff, rel_tol
    );
}

/// gam#683 regression: multiple real REML/continuation outer iterations under
/// `linkwiggle()` must return. Unlike `margslope_flex_large_scale_repro_cycle0`
/// (which caps `outer_max_iter = 1`), this allows several outer iterations so
/// the degree-15/21 BMS row-cell-moment derivative path and the continuation
/// pre-warm actually fire repeatedly — the exact regime #683 reported as
/// hanging (the outer LAML Hessian re-walked every cubic partition cell per
/// `(ρ-axis i, ρ-axis j)` pair, O(D²·n·cells·r²) per outer step). The
/// axis-projected per-row tensor cache collapses that to one O(n·cells·r²)
/// build reused across all pairs. `inner_max_cycles` uses a small fixed cap and
/// the outer derivative passes use a fixed row mask, so the debug regression
/// covers the hang surface without spending the full production convergence
/// tail or making CI pay every row.
#[test]
fn flex_full_outer_completes_under_budget_683() {
    gam::init_parallelism();
    let n = 96usize;
    let problem = build_large_scale_shape_problem(n);
    let outer_mask = (0..16usize).collect::<Vec<_>>();
    let options = BlockwiseFitOptions {
        inner_max_cycles: FULL_OUTER_SMOKE_INNER_CYCLES,
        outer_max_iter: 3,
        compute_covariance: false,
        screen_initial_rho: false,
        outer_score_subsample: Some(Arc::new(OuterScoreSubsample::new(outer_mask, n, 683))),
        auto_outer_subsample: false,
        ..BlockwiseFitOptions::default()
    };
    let start = std::time::Instant::now();
    let result = fit_problem(problem, options);
    let elapsed = start.elapsed();
    match result {
        Ok((out, timing)) => {
            eprintln!(
                "[MS-FLEX-683] n={} inner_max_cycles={} elapsed_s={:.3} outer_iters={} inner_cycles={} converged={} beta_len={}",
                n,
                FULL_OUTER_SMOKE_INNER_CYCLES,
                elapsed.as_secs_f64(),
                timing.outer_iterations,
                timing.inner_cycles,
                timing.outer_converged,
                out.fit.beta.len()
            );
            assert!(
                out.fit.beta.iter().all(|v| v.is_finite()),
                "non-finite beta from full-outer FLEX fit"
            );
            assert!(
                timing.inner_cycles >= 1,
                "fit did not enter the joint-Newton inner loop"
            );
        }
        Err(err) => {
            eprintln!(
                "[MS-FLEX-683] n={} inner_max_cycles={} elapsed_s={:.3} bounded_err={}",
                n,
                FULL_OUTER_SMOKE_INNER_CYCLES,
                elapsed.as_secs_f64(),
                err
            );
            assert!(
                err.contains("outer smoothing optimization did not converge")
                    && !err.contains("exhausted the joint Newton budget")
                    && !err.contains("no candidate seeds passed outer startup validation"),
                "unexpected full-outer FLEX error: {err}"
            );
        }
    }
    assert!(
        elapsed <= DEFAULT_WALL_BOUND,
        "full-outer FLEX fit exceeded {}s wall budget at n={n} (possible #683 regression): {:.3}s",
        DEFAULT_WALL_BOUND.as_secs(),
        elapsed.as_secs_f64()
    );
}
