//! Grouped integration-test crate root for survival tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "../common/misc/fixtures.rs"]
mod fixtures;

mod misc;
mod survival;
