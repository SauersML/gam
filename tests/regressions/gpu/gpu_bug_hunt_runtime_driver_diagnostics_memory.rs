use gam::gpu::{
    self, DeviceCsrMatrix, DeviceMatrix, DeviceVector, GpuEligibility, GpuKernel, GpuRuntime,
};
use ndarray::{Array2, array};
use std::thread;

#[test]
fn gpu_runtime_probe_reports_no_device_or_driver_as_structured_error_instead_of_none() {
    let probe = GpuRuntime::probe();
    assert!(
        matches!(
            probe,
            Ok(gam::gpu::GpuAvailability::Available(_)
                | gam::gpu::GpuAvailability::Absent(_))
                | Err(gam::gpu::GpuError::DriverLibraryLoadFailed { .. }
                    | gam::gpu::GpuError::RuntimeDependencyUnavailable { .. }
                    | gam::gpu::GpuError::DriverCallFailed { .. })
        ),
        "GpuRuntime::probe must return typed availability or a typed probe fault"
    );
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
        assert!(!decision.use_gpu, "typed absence under Auto must select CPU");
        assert!(decision.reason.contains("cpu"));
    } else {
        assert!(decision.use_gpu, "available eligible runtime must select GPU");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_device_info_device_count_matches_underlying_driver_count() {
    if !gam::gpu::driver::cuda_driver_available()
        .unwrap_or_else(|error| panic!("CUDA driver load fault: {error}"))
    {
        assert!(
            matches!(
                GpuRuntime::probe(),
                Ok(gam::gpu::GpuAvailability::Absent(
                    gam::gpu::GpuAbsence::DriverUnavailable { .. }
                ))
            ),
            "missing libcuda should be reported before touching cudarc device-count paths"
        );
        return;
    }

    let availability = GpuRuntime::probe().expect("runtime probe should not fault");
    let raw_count = cudarc::driver::CudaContext::device_count()
        .expect("driver device count should be queryable once probe succeeded");
    match availability {
        gam::gpu::GpuAvailability::Available(runtime) => assert_eq!(
            i32::try_from(runtime.devices.len()).expect("device vec length should fit i32"),
            raw_count,
            "GpuDeviceInfo count should match the driver count"
        ),
        gam::gpu::GpuAvailability::Absent(gam::gpu::GpuAbsence::NoDevice { .. }) => {
            assert_eq!(raw_count, 0)
        }
        gam::gpu::GpuAvailability::Absent(other) => {
            panic!("driver was loadable but probe reported unexpected absence: {other}")
        }
    }
}

#[test]
fn device_memory_representations_guard_against_invalid_csr_and_double_free_style_states() {
    let dense = DeviceMatrix::from_array(&Array2::zeros((3, 2)));
    let vec = DeviceVector::from_array(&array![1.0, 2.0, 3.0]);
    let csr = DeviceCsrMatrix::new(
        3,
        2,
        gpu::DeviceBuffer::from_host_shadow(vec![0, 1]),
        gpu::DeviceBuffer::from_host_shadow(vec![0]),
        gpu::DeviceBuffer::from_host_shadow(vec![1.0]),
    );

    assert_eq!(
        dense.data.len(),
        6,
        "dense allocation size should match rows*cols"
    );
    assert_eq!(
        vec.data.len(),
        3,
        "vector allocation size should match input length"
    );
    assert_eq!(
        csr.rowptr.len(),
        csr.rows + 1,
        "CSR rowptr must have rows+1 entries to prevent invalid frees / out-of-bounds deallocation paths"
    );
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
