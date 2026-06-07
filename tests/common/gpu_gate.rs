//! Structured GPU-skip gate for tests that require CUDA.
//!
//! Every test that is meaningful only on a CUDA host calls `gpu_gate` at the
//! top of its body and early-returns when the result is `GpuGate::Skip`.
//! Unlike a bare `if !cuda_selected() { return; }`, this emits a visible
//! `SKIP` line to stderr so CI reports show clearly which tests did not run
//! rather than silently counting them as "passed".
//!
//! ## Usage
//!
//! ```rust,ignore
//! mod common;
//! use common::gpu_gate::{GpuGate, gpu_gate};
//!
//! #[test]
//! fn my_gpu_test() {
//!     if let GpuGate::Skip = gpu_gate("my_gpu_test") { return; }
//!     // ... test body ...
//! }
//! ```
//!
//! ## CI enforcement
//!
//! On GPU runners, the unified GPU policy selects CUDA when the runtime probe
//! finds a usable device. Because `gpu_gate` returns `GpuGate::Run` whenever
//! `cuda_selected()` is true, no `SKIP` lines appear on those runners.
//! The companion test `gpu_required_tests_did_not_skip` in
//! `tests/gpu_required_tests_did_not_skip.rs` asserts that a CUDA runtime
//! present on the host implies `cuda_selected()` is true — failing loudly
//! if policy selection stops matching runtime availability.

use gam::gpu::cuda_selected;

/// Result of the GPU gate check.
pub enum GpuGate {
    /// CUDA is selected — the test body should execute.
    Run,
    /// CUDA is not selected — the test body should be skipped.
    /// A `SKIP` line has already been emitted to stderr.
    Skip,
}

/// Check whether the test should run on this host.
///
/// If `cuda_selected()` is false, emits `SKIP <test_name>: cuda not selected`
/// to stderr and returns `GpuGate::Skip`. The caller should `return`
/// immediately when `Skip` is returned.
///
/// If `cuda_selected()` is true, returns `GpuGate::Run` with no output.
pub fn gpu_gate(test_name: &str) -> GpuGate {
    if cuda_selected() {
        GpuGate::Run
    } else {
        eprintln!("SKIP {test_name}: cuda not selected");
        GpuGate::Skip
    }
}
