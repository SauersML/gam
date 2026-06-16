use gam::gpu::{self, GpuEligibility, GpuKernel, GpuPolicy};

#[test]
fn backend_status_and_policy_dispatch_are_consistent() {
    let runtime_available =
        std::panic::catch_unwind(|| gam::gpu::device_runtime::GpuRuntime::global().is_some())
            .unwrap_or(false);

    let blas_status = std::panic::catch_unwind(gam::gpu::blas::blas_backend_status).ok();
    let solver_status = std::panic::catch_unwind(gam::gpu::solver::solver_backend_status).ok();

    if runtime_available {
        assert_eq!(
            blas_status,
            Some(gam::gpu::CudaBackendStatus::CudaReady),
            "BLAS backend status should report CUDA ready when runtime probe succeeds."
        );
        assert_eq!(
            solver_status,
            Some(gam::gpu::CudaBackendStatus::CudaReady),
            "Solver backend status should report CUDA ready when runtime probe succeeds."
        );
    } else {
        assert!(
            blas_status.is_none()
                || blas_status == Some(gam::gpu::CudaBackendStatus::CudaUnavailable),
            "BLAS backend status should report CUDA unavailable or panic-cleanly when CUDA runtime cannot be loaded."
        );
        assert!(
            solver_status.is_none()
                || solver_status == Some(gam::gpu::CudaBackendStatus::CudaUnavailable),
            "Solver backend status should report CUDA unavailable or panic-cleanly when CUDA runtime cannot be loaded."
        );
    }

    gpu::configure_global_policy(GpuPolicy::Off);
    let off = gpu::decide(
        GpuKernel::DenseMatvec,
        GpuEligibility::from_flags(true, true),
    );
    assert!(
        !off.use_gpu,
        "GPU policy Off should always select CPU execution."
    );

    let global = gpu::global_policy();
    let forced = gpu::decide(
        GpuKernel::DenseMatvec,
        GpuEligibility::from_flags(true, true),
    );
    if global == GpuPolicy::Off {
        assert!(
            !forced.use_gpu,
            "Global policy is first-writer-wins, so later calls should not silently replace the process policy."
        );
    }
}
