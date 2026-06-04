//! gam#683 regression test — FLEX (`linkwiggle()`) marginal-slope hang.
//!
//! Unlike `margslope_flex_biobank_repro.rs` (which caps `outer_max_iter = 1` so
//! it finishes), this runs the FULL outer REML/continuation loop at small `n`,
//! so the degree-15/21 BMS row-cell-moment derivative path and the outer
//! continuation pre-warm actually fire — the exact regime #683 reported as
//! hanging. Before the axis-projected per-row tensor cache fix this did not
//! terminate (the outer LAML Hessian re-walked every cubic partition cell per
//! `(ρ-axis i, ρ-axis j)` pair, O(D²·n·cells·r²) per outer step); it now
//! completes in a few seconds.
//!
//! For ad-hoc profiling, run with logging to see the per-phase `elapsed=` lines:
//!
//!     RUST_LOG=info cargo test --test margslope_flex_profile_683 -- --nocapture

#[path = "test_support/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use gam::families::custom_family::BlockwiseFitOptions;
use margslope_flex_equivalence::{build_biobank_shape_problem, fit_problem};

#[test]
fn flex_full_outer_completes_under_budget_683() {
    gam::init_parallelism();
    let n = 300usize;
    let problem = build_biobank_shape_problem(n);
    let options = BlockwiseFitOptions::default();
    let start = std::time::Instant::now();
    let result = fit_problem(problem, options);
    let elapsed = start.elapsed();
    match result {
        Ok((out, timing)) => {
            eprintln!(
                "[PROFILE-683] n={} OK elapsed_s={:.3} outer_iters={} inner_cycles={} converged={} beta_len={}",
                n,
                elapsed.as_secs_f64(),
                timing.outer_iterations,
                timing.inner_cycles,
                timing.outer_converged,
                out.fit.beta.len()
            );
            // Regression gate for #683: the full-outer FLEX fit must finish
            // (with finite coefficients) well inside the wall budget — the bug
            // was an unbounded BMS row-cell-moment hang under `linkwiggle()`.
            assert!(
                out.fit.beta.iter().all(|v| v.is_finite()),
                "non-finite beta from full-outer FLEX fit"
            );
            assert!(
                timing.inner_cycles >= 1,
                "fit did not enter the joint-Newton inner loop"
            );
            assert!(
                elapsed.as_secs_f64() < 180.0,
                "full-outer FLEX fit exceeded 180s wall budget at n={n} (possible #683 regression): {:.3}s",
                elapsed.as_secs_f64()
            );
        }
        Err(e) => panic!(
            "[PROFILE-683] n={n} full-outer FLEX fit errored after {:.3}s: {e}",
            elapsed.as_secs_f64()
        ),
    }
}
