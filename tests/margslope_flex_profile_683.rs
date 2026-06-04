//! Local profiling harness for gam#683 — FLEX (linkwiggle) marginal-slope hang.
//!
//! Unlike `margslope_flex_biobank_repro.rs` (which caps `outer_max_iter=1` so it
//! finishes), this runs the FULL outer REML/continuation loop at small `n` so the
//! degree-15/21 BMS row-cell-moment path and the outer continuation pre-warm
//! actually fire — i.e. the regime #683 reports as hanging. Intentionally
//! `#[ignore]`; run manually with logging to see per-phase `elapsed=` lines:
//!
//!     RUST_LOG=info cargo test --test margslope_flex_profile_683 \
//!         profile_683_full_outer -- --ignored --nocapture

#[path = "test_support/margslope_flex_equivalence.rs"]
mod margslope_flex_equivalence;

use gam::families::custom_family::BlockwiseFitOptions;
use margslope_flex_equivalence::{build_biobank_shape_problem, fit_problem};

#[test]
#[ignore]
fn profile_683_full_outer() {
    gam::init_parallelism();
    let n: usize = std::option_env!("PROFILE_683_N")
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
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
        }
        Err(e) => {
            eprintln!(
                "[PROFILE-683] n={} ERR elapsed_s={:.3} err={}",
                n,
                elapsed.as_secs_f64(),
                e
            );
        }
    }
}
