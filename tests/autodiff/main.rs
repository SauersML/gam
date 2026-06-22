//! Grouped integration-test crate root for autodiff tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[macro_use]

mod families;
mod misc;
mod optimization;
mod smooths;
mod survival;
