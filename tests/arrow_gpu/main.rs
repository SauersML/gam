//! Grouped integration-test crate root for arrow_gpu tests (issue #1146).
//!
//! Former top-level crates included as modules so they link as ONE binary.

#[path = "../common/gpu/gpu_gate.rs"]
mod gpu_gate;

mod gpu;
mod misc;
