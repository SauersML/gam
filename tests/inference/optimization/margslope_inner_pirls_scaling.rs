//! Scaling-law probe for the bernoulli marginal-slope INNER PIRLS Newton
//! solve at large-scale shape.
//!
//! The outer-score probes established that the OUTER ψ first-order eval is
//! small relative to the full command budget. The hypothesis under test here
//! is that the dominant cost lives inside the inner PIRLS Newton solve
//! (i.e. per-row sextic-kernel evaluation × inner-Newton iterations × outer
//! BFGS iterations), and that the path #3 inner-iter schedule cap is
//! actually doing the work it claims.
//!
//! This file ships TWO probes:
//!
//! 1. `margslope_inner_pirls_scaling_law` — RIGID probit (no score_warp /
//!    no link_dev). The family takes the closed-form vectorized path
//!    described in `BernoulliMarginalSlopeFamily::log_likelihood_only_with_options`,
//!    so the inner-Newton trivialises (`inner_cycles=1`). Useful as a
//!    NEGATIVE control: confirms that the rigid path is NOT the
//!    large-scale bottleneck.
//!
//! 2. `margslope_inner_pirls_flex_scaling_law` — FLEX probit (cubic
//!    score_warp + link_dev deviation blocks). The per-row sextic-kernel
//!    cell evaluator (`solve_row_intercept_base` + `observed_denested_cell_partials`)
//!    runs at every inner-PIRLS iteration. This is the large-scale
//!    production setup; this probe times its scaling.
//!
//! Both probes run in the ordinary suite: `#[ignore]` is a hard build abort in
//! this workspace (`build.rs` "#[ignore] test" rule, enforcing SPEC.md's ban on
//! the XFAIL pattern), so there is no skip to opt out of. Reading their output
//! means asking for it:
//!
//! ```text
//! cargo test --release --test inference \
//!     -- --nocapture margslope_inner_pirls_scaling_law
//! cargo test --release --test inference \
//!     -- --nocapture margslope_inner_pirls_flex_scaling_law
//! ```
//!
//! Note the binary is `inference`, not this file: #1146 grouped these modules
//! into one integration binary rooted at `tests/inference/main.rs`.
//!
//! The `[MS-INNER-SCALING]` / `[MS-INNER-FLEX-SCALING]` lines in the output
//! are pivotable: parse them into (n, total_s, outer_iters, inner_cycles)
//! and fit `total_s = a · n^α`. Honest fit: if R²<0.85 or max log-resid>0.5,
//! refuses to extrapolate.

#[path = "../../perf_scale/misc/power_law_common.rs"]
mod power_law_common;

