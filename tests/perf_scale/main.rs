//! Grouped integration-test crate root for perf_scale tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod families;
mod misc;
mod optimization;
mod sae;
mod smooths;
