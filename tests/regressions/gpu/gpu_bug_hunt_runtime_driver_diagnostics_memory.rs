use gam::gpu::{self, GpuEligibility, GpuKernel, GpuRuntime};
use ndarray::Array2;
use std::thread;

#[test]
fn gpu_runtime_probe_returns_typed_availability_without_fault() {
    match GpuRuntime::probe() {
        Ok(gam::gpu::GpuAvailability::Available(_) | gam::gpu::GpuAvailability::Absent(_)) => {}
        Err(error) => {
            panic!("GpuRuntime::probe faulted instead of returning availability: {error}")
        }
    }
}

#[test]
fn gpu_policy_auto_falls_back_to_cpu_when_runtime_is_unavailable_and_sets_cpu_reason() {
    let availability = GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in policy test: {error}"));
    let decision = gpu::decide(
        GpuKernel::DenseMatvec,
        GpuEligibility::from_flags(true, true),
    )
    .unwrap_or_else(|error| panic!("GPU decision fault in policy test: {error}"));
    if availability.is_none() {
        assert!(
            !decision.use_gpu,
            "typed absence under Auto must select CPU"
        );
        assert!(decision.reason.contains("cpu"));
    } else {
        assert!(
            decision.use_gpu,
            "available eligible runtime must select GPU"
        );
    }
}

#[test]
fn diagnostics_counters_increment_on_every_dispatch_and_reset_clears_them() {
    gam::gpu::profile::clear();
    let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape");
    drop(gam::gpu::try_fast_ab(a.view(), a.view()));
    let after_dispatch = gam::gpu::profile::snapshot();
    assert!(
        !after_dispatch.stats.is_empty(),
        "dispatch diagnostics counter should increment for every dispatch attempt"
    );
    gam::gpu::profile::clear();
    let after_reset = gam::gpu::profile::snapshot();
    assert!(
        after_reset.stats.is_empty(),
        "reset should clear all diagnostics counters"
    );
}

#[test]
fn concurrent_runtime_probe_is_idempotent_without_race_or_double_init() {
    let mut handles = Vec::new();
    for _ in 0..8 {
        handles.push(thread::spawn(|| {
            GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
                .unwrap_or_else(|error| panic!("concurrent GPU probe fault: {error}"))
                .map(|runtime| runtime.device.ordinal)
        }));
    }
    let ordinals: Vec<Option<usize>> = handles
        .into_iter()
        .map(|h| h.join().expect("probe thread should not panic"))
        .collect();
    let first = ordinals[0];
    assert!(
        ordinals.iter().all(|v| *v == first),
        "concurrent global probes should all observe the same initialized runtime snapshot"
    );

    GpuRuntime::probe()
        .unwrap_or_else(|error| panic!("direct probe after cached resolution faulted: {error}"));
}
