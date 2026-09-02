//! Structured GPU-skip gate for tests that require CUDA.
//!
//! Every test that is meaningful only on a CUDA host calls `gpu_gate` at the
//! top of its body and early-returns when the result is `GpuGate::Skip`.
//! Unlike a bare `if !cuda_selected() { return; }`, the skip is counted and
//! asserted, so the early return is no longer a test that reported `ok` having
//! executed nothing.
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
//! A skip is counted, not just printed: the gate delegates to
//! `gam::gpu::test_gate`, which increments a process counter and emits the one
//! shared `SKIPPED(no-cuda):` marker, and this gate then asserts the count
//! moved. So a gated test that declines still executes an assertion, and a
//! green run can be scraped for an inventory of what did not run instead of
//! implying that every `ok` verified something (#2422).
//!
//! On a GPU runner nothing here skips, and a device that is present but not
//! *selected* is a hard failure rather than a skip — see `gpu_gate`. The
//! companion `gpu_required_tests_did_not_skip` asserts the same implication at
//! suite level, against the counter rather than against `cuda_selected()`.

