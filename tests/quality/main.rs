//! Grouped integration-test crate root for quality tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

mod calibration;
mod factors;
mod families;
mod manifolds;
mod misc;
mod smooths;
mod survival;
