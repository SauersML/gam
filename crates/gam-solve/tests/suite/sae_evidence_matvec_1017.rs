//! #1017 device-resident evidence-matvec gates (integration test).
//!
//! Runs the DETERMINISTIC framed reduced-Schur `S·v` — the operator that feeds
//! the SLQ/surrogate `log|S|` evidence lane — on a real CUDA device and checks
//! (1) it matches the bit-for-bit CPU oracle `sae_framed_schur_matvec_cpu` to
//! ≤1e-9, and (2) it is run-to-run bit-identical (the determinism contract the
//! shared atomic step-PCG matvec cannot satisfy). A separate utilization test
//! drives many applies through the resident builder (the SLQ apply loop) on a
//! large fixture so GPU utilization can be sampled during the run.
//!
//! This lives in `tests/` (not a `#[cfg(test)]` unit module) so it compiles
//! against only the `gam-solve` library and is insulated from unrelated
//! unit-test churn in the shared lib-test binary. It uses the public API only.
//! Off-device (CPU CI / non-Linux) the gates skip cleanly.

// Device presence is detected via the one-shot probe itself: for a well-formed
// framed fixture it returns `Err(Unavailable)` ONLY when CUDA is genuinely
// absent (it ignores the offload floor), and `Ok(..)` only after running on the
// GPU. So `Ok` ⇒ device present, `Err` ⇒ no device (clean off-device skip).

