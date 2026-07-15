use gam::gpu::{self, GpuEligibility, GpuKernel, GpuPolicy};

#[test]
fn backend_status_and_policy_dispatch_are_consistent() {
    let runtime_available = gam::gpu::device_runtime::GpuRuntime::resolve(GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in backend-status test: {error}"))
        .is_some();

    let blas_status = gam::gpu::blas::blas_backend_status()
        .unwrap_or_else(|error| panic!("BLAS backend status fault: {error}"));
    let solver_status = gam::gpu::solver::solver_backend_status()
        .unwrap_or_else(|error| panic!("solver backend status fault: {error}"));

    if runtime_available {
        assert_eq!(
            blas_status,
            gam::gpu::CudaBackendStatus::CudaReady,
            "BLAS backend status should report CUDA ready when runtime probe succeeds."
        );
        assert_eq!(
            solver_status,
            gam::gpu::CudaBackendStatus::CudaReady,
            "Solver backend status should report CUDA ready when runtime probe succeeds."
        );
    } else {
        assert_eq!(blas_status, gam::gpu::CudaBackendStatus::CudaUnavailable);
        assert_eq!(solver_status, gam::gpu::CudaBackendStatus::CudaUnavailable);
    }

    gpu::configure_global_policy(GpuPolicy::Off);
    let off = gpu::decide(
        GpuKernel::DenseMatvec,
        GpuEligibility::from_flags(true, true),
    )
    .expect("Off policy decision is infallible");
    assert!(
        !off.use_gpu,
        "GPU policy Off should always select CPU execution."
    );

    let global = gpu::global_policy();
    let forced = gpu::decide(
        GpuKernel::DenseMatvec,
        GpuEligibility::from_flags(true, true),
    )
    .expect("the installed Off policy must bypass runtime probing");
    if global == GpuPolicy::Off {
        assert!(
            !forced.use_gpu,
            "Global policy is first-writer-wins, so later calls should not silently replace the process policy."
        );
    }
}
