//! Fast local reproducer for the FLEX bernoulli marginal-slope cycle-0 cost
//! cliff seen in the biobank-scale lane.
//!
//! Manual invocation (intentionally ignored; do not put this in normal CI):
//!
//! ```text
//! cargo test --release --test margslope_flex_biobank_repro \
//!     -- --ignored --nocapture
//! ```
//!
//! The synthetic shape keeps the production code path active: probit
//! bernoulli marginal slope, score-warp and link-deviation FLEX blocks,
//! a joint 16D Duchon PC smooth (`centers=24`, `order=1`, `power=8`,
//! `length_scale=1.0`), a separate smooth age term, and a standard-normal
//! latent `z`.  The primary repro uses `n = DEFAULT_REPRO_N` and caps the
//! full blockwise fit at joint-Newton cycle 0 (`inner_max_cycles=1`) so the
//! printed wall time is a local proxy for the biobank
//! `[PIRLS/blockwise joint-Newton] cycle 0/100` region.  Run under `samply`,
//! `cargo flamegraph`, or macOS `sample` for a flame graph/profile;
//! `--nocapture` preserves the per-fit phase summaries already emitted by
//! the solver.

#[path = "test_support/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use margslope_flex_equivalence::{
    DEFAULT_REPRO_N, DEFAULT_SMOKE_N, DEFAULT_WALL_BOUND, assert_repeated_fit_beta_equivalent,
    build_biobank_shape_problem, cycle_capped_options, fit_problem,
};

#[test]
fn margslope_flex_biobank_repro_cycle0() {
    gam::init_parallelism();
    let n = DEFAULT_REPRO_N;
    let bound = DEFAULT_WALL_BOUND;
    let problem = build_biobank_shape_problem(n);
    let (out, timing) = fit_problem(problem, cycle_capped_options(1))
        .expect("biobank-shape FLEX margslope cycle-0 repro fit");
    eprintln!(
        "[MS-FLEX-BIOBANK-REPRO] n={} inner_max_cycles=1 elapsed_s={:.3} outer_iters={} inner_cycles={} converged={} beta_len={}",
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
