//! FULL exact per-row SAE encode — end-to-end throughput AND correctness.
//!
//! ## Why this test exists
//!
//! The component benchmark in `gam_gpu::encode_throughput` (and its root test
//! `tests/gpu_encode_throughput_measured_1412.rs`) times ONLY the resident
//! penalized normal-equations inner solve `(XᵀWX+ridge·I)β=rhs` and is explicit
//! that this is NOT the full exact per-row SAE encode. Passing it therefore says
//! nothing about a "batched exact per-row encode" claim, because none of the
//! encode's real semantics are exercised: chart/active-set routing, the per-row
//! latent-coordinate Newton refinement, the gate/assignment (amplitude), the
//! Kantorovich certificate + fallback, and the per-row reconstruction selection.
//!
//! This test drives the ACTUAL production encode — `EncodeAtlas::certified_*`
//! (`crates/gam-sae/src/encode.rs`), which is exactly that pipeline — end to end
//! over a batch, and:
//!
//!   1. TIMES the full `certified_encode_batch` → rows/sec
//!      ([`FullEncodeThroughput`]);
//!   2. CHECKS correctness against the production per-row encode and the planted
//!      manifold via [`encode_quality_metrics`]: support agreement, latent
//!      coordinate error, reconstruction explained-variance, and fallback rate.
//!
//! ## Reuse, not reimplementation
//!
//! The encode math is NOT reimplemented here: the test calls the production
//! `EncodeAtlas::certified_encode_batch` / `certified_encode_row`. It lives in
//! `gam-sae` because that is the crate that owns the composition under test —
//! `gam-sae` depends on `gam-gpu`, so both halves are reachable along the normal
//! dependency direction. (It previously lived in `gam-gpu` behind a dev-only
//! back-edge onto `gam-sae`; that edge made every `gam-gpu` unit-test build
//! compile the entire SAE and solver stack.)
//!
//! ## Device status (honest)
//!
//! There is currently NO device-resident exact-encode kernel — the production
//! `certified_encode_*` path is host ndarray work (the only SAE GPU kernel,
//! `gam_sae::gpu_kernels::sae_rowjet`, accelerates the *fitting* jet tower, not
//! the encode). So `device_encode_engaged` is reported `false` and the measured
//! rate is the CPU encode throughput. This test does NOT fabricate a device
//! number; it establishes the end-to-end CPU baseline + the correctness contract
//! a future device encode must match, and it exercises `gam-gpu`'s runtime probe
//! + fail-closed (`GpuPolicy::Required`) guard so the GPU plumbing stays wired.

