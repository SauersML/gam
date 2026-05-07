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
//! latent `z`.  The primary repro defaults to `n=50_000` and caps the full
//! blockwise fit at joint-Newton cycle 0 (`inner_max_cycles=1`) so the printed
//! wall time is a local proxy for the biobank `[PIRLS/blockwise joint-Newton]
//! cycle 0/100` region.  Set `GAM_MARGSLOPE_REPRO_N`,
//! `GAM_MARGSLOPE_REPRO_BOUND_SECS`, or `GAM_MARGSLOPE_REPRO_PERSISTENT=1` to
//! tune the local run.  Run under `samply`, `cargo flamegraph`, or macOS
//! `sample` for a flame graph/profile; `--nocapture` preserves the per-fit
//! phase summaries already emitted by the solver.

#[path = "test_support/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use margslope_flex_equivalence::{
    DEFAULT_REPRO_N, DEFAULT_SMOKE_N, DEFAULT_WALL_BOUND, assert_repeated_fit_beta_equivalent,
    build_biobank_shape_problem, cycle_capped_options, fit_problem,
};
use std::time::Duration;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_duration_secs(name: &str, default: Duration) -> Duration {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Duration::from_secs)
        .unwrap_or(default)
}

#[test]
#[ignore]
fn margslope_flex_biobank_repro_cycle0() {
    gam::init_parallelism();
    let n = env_usize("GAM_MARGSLOPE_REPRO_N", DEFAULT_REPRO_N);
    let bound = env_duration_secs("GAM_MARGSLOPE_REPRO_BOUND_SECS", DEFAULT_WALL_BOUND);
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
        "cycle-0 repro exceeded wall bound: elapsed={:.3}s bound={:.3}s; set GAM_MARGSLOPE_REPRO_BOUND_SECS to adjust for this host",
        timing.elapsed.as_secs_f64(),
        bound.as_secs_f64()
    );

    if std::env::var("GAM_MARGSLOPE_REPRO_PERSISTENT").as_deref() == Ok("1") {
        let persistent_n = env_usize("GAM_MARGSLOPE_REPRO_PERSISTENT_N", n.min(10_000));
        let problem = build_biobank_shape_problem(persistent_n);
        let (_out, timing) = fit_problem(problem, cycle_capped_options(5))
            .expect("biobank-shape FLEX margslope persistent-cache repro fit");
        eprintln!(
            "[MS-FLEX-BIOBANK-REPRO-PERSISTENT] n={} inner_max_cycles=5 elapsed_s={:.3} outer_iters={} inner_cycles={} converged={}",
            persistent_n,
            timing.elapsed.as_secs_f64(),
            timing.outer_iterations,
            timing.inner_cycles,
            timing.outer_converged
        );
        assert!(timing.inner_cycles >= 1);
    }
}

#[test]
#[ignore]
fn margslope_flex_beta_equivalence_smoke() {
    gam::init_parallelism();
    let n = env_usize("GAM_MARGSLOPE_EQUIV_N", DEFAULT_SMOKE_N);
    let inner_cycles = env_usize("GAM_MARGSLOPE_EQUIV_INNER_CYCLES", 1);
    let rel_tol = std::env::var("GAM_MARGSLOPE_EQUIV_REL_TOL")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-10);
    let report = assert_repeated_fit_beta_equivalent(n, inner_cycles, rel_tol);
    eprintln!(
        "[MS-FLEX-EQUIV-SMOKE] PASS beta_len={} max_abs={:.3e} max_rel={:.3e} rel_tol={:.3e}",
        report.len, report.max_abs_diff, report.max_rel_diff, rel_tol
    );
}
