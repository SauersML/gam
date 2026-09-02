//! Scaling-law probe for standard-GAM Bernoulli-probit at large-scale shape.
//!
//! Times one full fit at each of several n values, then prints a summary
//! table that lets us extrapolate to the large-scale n=320_000 target without
//! waiting on a 30-50 minute CI cycle.
//!
//! Run with:
//! ```text
//! cargo test --release --test standard_gam_scaling -- --ignored --nocapture standard_gam_scaling_law
//! ```
//!
//! The `[SCALING]` lines in the output are pivotable: parse them into a
//! (n, total_s, per_iter_s) triple per row and fit `total_s = a * n^α`.
//! Mission target: extrapolated total at n=320k must be ≤ 2400s (CI's
//! 40-min cmd timeout). With path #2/#3 + standard-GAM gate this is
//! expected to drop dramatically vs the pre-fix scaling.

#[path = "../../perf_scale/misc/power_law_common.rs"]
mod power_law_common;

