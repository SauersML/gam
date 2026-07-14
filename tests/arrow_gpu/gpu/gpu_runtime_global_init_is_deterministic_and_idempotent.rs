use gam::gpu::device_runtime::GpuRuntime;

#[test]
fn gpu_runtime_resolution_is_deterministic_and_idempotent() {
    let resolve = || {
        GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
            .unwrap_or_else(|error| panic!("GPU probe fault in idempotence test: {error}"))
            .map(|runtime| runtime.selected_device().ordinal)
    };
    let first = resolve();
    let second = resolve();
    let third = resolve();

    assert_eq!(
        first, second,
        "Calling GPU runtime initialization repeatedly should produce the same global availability and selected device."
    );
    assert_eq!(
        second, third,
        "Calling GPU runtime initialization repeatedly should remain idempotent after the first call."
    );
}
