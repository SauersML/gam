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

