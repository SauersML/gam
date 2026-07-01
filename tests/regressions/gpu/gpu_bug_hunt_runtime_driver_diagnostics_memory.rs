use gam::gpu::{
    self, DeviceCsrMatrix, DeviceMatrix, DeviceVector, GpuEligibility, GpuKernel, GpuRuntime,
};
use ndarray::{Array2, array};
use std::thread;

#[test]
fn gpu_runtime_probe_reports_no_device_or_driver_as_structured_error_instead_of_none() {
    let probe = GpuRuntime::probe();
    assert!(
        probe.is_ok()
            || matches!(
                probe,
                Err(gam::gpu::GpuError::DriverLibraryUnavailable { .. }
                    | gam::gpu::GpuError::DriverCallFailed { .. })
            ),
        "GpuRuntime::probe should return a runtime or a typed CUDA probe failure, not panic or collapse unavailable hardware into Ok(None)"
    );
}

#[test]
fn gpu_policy_auto_falls_back_to_cpu_when_runtime_is_unavailable_and_sets_cpu_reason() {
    let decision = gpu::decide(
        GpuKernel::DenseMatvec,
        GpuEligibility::from_flags(true, true),
    );
    assert!(
        !decision.use_gpu,
        "gpu=auto should fall back to CPU when runtime has no usable device even if kernel support is compiled"
    );
    assert!(
        decision.reason.contains("cpu"),
        "fallback decision should expose a cpu_reason-style marker so callers can report why CPU was selected"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn gpu_device_info_device_count_matches_underlying_driver_count() {
    if !gam::gpu::driver::cuda_driver_library_present() {
        assert!(
            matches!(
                GpuRuntime::probe(),
                Err(gam::gpu::GpuError::DriverLibraryUnavailable { ref reason })
                    if reason == "libcuda unavailable"
            ),
            "missing libcuda should be reported before touching cudarc device-count paths"
        );
        return;
    }

    let runtime = GpuRuntime::probe().expect("runtime probe should not fail").expect(
        "runtime probe should return a runtime snapshot even when device_count is zero so count can be reported",
    );
    let device_count =
        i32::try_from(runtime.devices.len()).expect("device vec length should fit i32");
    let raw_count = cudarc::driver::CudaContext::device_count()
        .expect("driver device count should be queryable once probe succeeded");
    assert_eq!(
        device_count, raw_count,
        "GpuDeviceInfo count should match the underlying CUDA driver-reported device count"
    );
}

#[test]
fn device_memory_representations_guard_against_invalid_csr_and_double_free_style_states() {
    let dense = DeviceMatrix::from_array(&Array2::zeros((3, 2)));
    let vec = DeviceVector::from_array(&array![1.0, 2.0, 3.0]);
<<<<<<< ours
    let csr = DeviceCsrMatrix::new(
=======
    let invalid_csr = DeviceCsrMatrix::new(
>>>>>>> theirs
        3,
        2,
        gpu::DeviceBuffer::from_host_shadow(vec![0, 1]),
        gpu::DeviceBuffer::from_host_shadow(vec![0]),
        gpu::DeviceBuffer::from_host_shadow(vec![1.0]),
    );
<<<<<<< ours
=======
    let csr = DeviceCsrMatrix::new(
        3,
        2,
        gpu::DeviceBuffer::from_host_shadow(vec![0, 1, 1, 1]),
        gpu::DeviceBuffer::from_host_shadow(vec![0]),
        gpu::DeviceBuffer::from_host_shadow(vec![1.0]),
    )
    .expect("valid CSR rowptr has rows+1 entries");
>>>>>>> theirs

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
    assert!(
        invalid_csr.is_err(),
        "CSR construction must reject rowptr lengths other than rows+1 to prevent invalid frees / out-of-bounds deallocation paths"
    );
    assert_eq!(
        csr.rowptr().len(),
        csr.rows() + 1,
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
            GpuRuntime::global().map(|rt| rt.device.ordinal)
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

    let direct_probe = GpuRuntime::probe();
    assert!(
        direct_probe.is_ok()
            || matches!(
                direct_probe,
                Err(gam::gpu::GpuError::DriverLibraryUnavailable { .. }
                    | gam::gpu::GpuError::DriverCallFailed { .. })
            ),
        "direct probe after concurrent global initialization should remain idempotent and typed"
    );
}
