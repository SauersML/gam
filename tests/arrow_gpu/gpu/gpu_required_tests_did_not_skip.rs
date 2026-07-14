//! Wrapper test: assert that GPU-gated tests actually ran on GPU runners.
//!
//! Every GPU-conditional test in this suite calls `gpu_gate(name)` from
//! `tests/common/gpu_gate.rs`. That gate returns `GpuGate::Skip` (with an
//! `eprintln!` SKIP line) when `cuda_selected()` is false, and `GpuGate::Run`
//! when it is true.
//!
//! The design guarantee: on a host with a CUDA runtime, `cuda_selected()`
//! MUST be true, otherwise every GPU test silently skips and the suite
//! reports all-green while exercising nothing. This test enforces that
//! guarantee by asserting:
//!
//!   `GpuRuntime::resolve(Auto).is_some() → cuda_selected()`
//!
//! On CPU-only hosts Auto resolves typed absence, so the
//! assertion is vacuously true and the test passes without noise.

#[test]
fn gpu_required_tests_did_not_skip() {
    let runtime_present = gam::gpu::device_runtime::GpuRuntime::resolve(
        gam::gpu::GpuPolicy::Auto,
    )
    .unwrap_or_else(|error| panic!("GPU probe fault in ship-gate test: {error}"))
    .is_some();
    if runtime_present {
        assert!(
            gam::gpu::cuda_selected()
                .unwrap_or_else(|error| panic!("GPU selection fault in ship-gate test: {error}")),
            "A CUDA runtime is available on this host but the unified GPU policy did \
             not select CUDA. Every GPU-gated test emitted a SKIP line and silently \
             passed without exercising any GPU code."
        );
    }
}
