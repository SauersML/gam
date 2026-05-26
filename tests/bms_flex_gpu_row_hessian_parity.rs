//! BMS-FLEX GPU milestone 1 — row-primary Hessian parity (integration shell).
//!
//! The real CPU↔GPU element-wise parity check lives in
//! `src/gpu/bms_flex_row.rs::tests::bms_flex_row_kernel_matches_cpu_oracle_when_cuda_available`.
//! That unit test exercises the kernel + a hand-rolled CPU oracle against
//! one shared `BmsFlexRowKernelInputs` bundle on every Linux host that
//! exposes a CUDA runtime, and skips with a one-line `eprintln!` on
//! every other host (macOS, CPU-only CI).
//!
//! This integration crate keeps a placeholder body solely so the harness
//! still binds a test slot named after the milestone — the actual
//! validation has moved to the unit level (closer to the kernel source,
//! type-checked alongside `cpu_oracle_outputs`, and reachable from
//! `cargo test -p gam --lib gpu::bms_flex_row::`).

#[test]
fn bms_flex_gpu_row_hessian_parity_lives_at_unit_level() {
    eprintln!(
        "[bms_flex_gpu_row_hessian_parity] CPU↔GPU parity lives at \
         gam::gpu::bms_flex_row::tests — run \
         `cargo test -p gam --lib \
         gpu::bms_flex_row::tests::bms_flex_row_kernel_matches_cpu_oracle_when_cuda_available` \
         on a CUDA host."
    );
}
