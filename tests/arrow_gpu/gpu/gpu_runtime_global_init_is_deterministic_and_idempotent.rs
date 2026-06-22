use gam::gpu::device_runtime::GpuRuntime;

#[test]
fn gpu_runtime_global_is_deterministic_and_idempotent() {
    let first_call =
        std::panic::catch_unwind(|| GpuRuntime::global().map(|rt| rt.selected_device().ordinal));
    let second_call =
        std::panic::catch_unwind(|| GpuRuntime::global().map(|rt| rt.selected_device().ordinal));
    let third_call =
        std::panic::catch_unwind(|| GpuRuntime::global().map(|rt| rt.selected_device().ordinal));

    let first = first_call.ok().flatten();
    let second = second_call.ok().flatten();
    let third = third_call.ok().flatten();

    assert_eq!(
        first, second,
        "Calling GPU runtime initialization repeatedly should produce the same global availability and selected device."
    );
    assert_eq!(
        second, third,
        "Calling GPU runtime initialization repeatedly should remain idempotent after the first call."
    );
}
