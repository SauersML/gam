//! Regression test for https://github.com/SauersML/gam/issues/171.
//!
//! On a CPU-only host (no `libcuda.{dylib,so,dll}` reachable via the platform
//! loader), a saved-and-then-loaded model must be predictable without
//! triggering `cudarc::panic_no_lib_found`. The original report showed
//! `Model.predict(df)` panicking inside the Rust
//! boundary on macOS because the dispatch decision reached into cudarc
//! (via `fallback-dynamic-loading`) without first checking for the CUDA
//! driver library outside cudarc.
//!
//! The fix landed in `GpuRuntime::probe`: every cudarc driver entry point
//! is now gated on gam's own `libloading` driver probe returning `true`, and
//! the typed runtime cache preserves an `Absent` outcome so subsequent
//! predict-path dispatch calls take the CPU fast path. This test pins the
//! contract by:
//!   1. exercising typed Auto resolution and the public dispatch
//!      `decide()` decision from the predict-relevant kernels (DenseMatvec,
//!      DenseMatMul, RowReduction);
//!   2. fitting a tiny GAM and running the predict-path design rebuild
//!      + matvec — the same arithmetic `StandardPredictor::predict_plugin_response`
//!      performs — and asserting it completes without panic.
//!
//! On a host with a working CUDA driver the test still passes; the
//! assertions only require that calls do not panic and that the GPU
//! dispatch decision is consistent with typed runtime availability.

use csv::StringRecord;
use gam::gpu::{self, GpuEligibility, GpuKernel};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

#[test]
fn predict_after_load_on_cpu_only_host_does_not_panic_with_cudarc() {
    init_parallelism();

    // Reproducer-shaped fit: a single smooth on uniform x, Gaussian target.
    // Mirrors the public surface (`s(x)`) that issue #171 exercises with
    // `gamfit.fit(df, "y ~ s(x)", config={"gpu": "off"})`.
    let n = 64usize;
    let x: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * i as f64 / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x.iter().map(|&t| t.sin()).collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=8)", &data, &cfg).expect("fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit");
    };

    // GPU dispatch decision: the predict-path kernels must not panic even
    // when Auto resolution returns typed absence (the CPU-only path). The
    // probe must have completed exactly once via gam's own libcuda
    // preflight guard — any regression there would surface as a
    // `cudarc::panic_no_lib_found` unwind escaping `catch_unwind` below.
    #[cfg(target_os = "linux")]
    {
        if !gam::gpu::driver::cuda_driver_available()
            .unwrap_or_else(|error| panic!("CUDA driver load fault: {error}"))
        {
            let direct_probe = gam::gpu::device_runtime::GpuRuntime::probe();
            assert!(
                matches!(
                    direct_probe,
                    Ok(gam::gpu::GpuAvailability::Absent(
                        gam::gpu::GpuAbsence::DriverUnavailable { .. }
                    ))
                ),
                "absent libcuda must remain typed absence before dispatch"
            );
        }
    }

    let probe = gam::gpu::device_runtime::GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in prediction test: {error}"))
        .is_some();

    for kernel in [
        GpuKernel::DenseMatvec,
        GpuKernel::DenseTransposeMatvec,
        GpuKernel::DenseXtWX,
    ] {
        let decision = gpu::decide(kernel, GpuEligibility::from_flags(true, true))
            .unwrap_or_else(|error| panic!("GPU decision fault in prediction test: {error}"));
        if !probe {
            assert!(
                !decision.use_gpu,
                "gpu::decide must select CPU when GpuRuntime is unavailable; got {decision:?} for {kernel:?}"
            );
        }
    }

    // Predict-path matvec: rebuild the design at the training inputs and
    // apply `design · β`. This is the exact arithmetic
    // `StandardPredictor::predict_plugin_response` performs (modulo the
    // offset / inverse-link, both trivial here). On the CPU-only host this
    // must complete without unwinding through cudarc.
    let eta = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut new_data = Array2::<f64>::zeros((n, 2));
        for (i, &t) in x.iter().enumerate() {
            new_data[[i, 0]] = t;
            new_data[[i, 1]] = 0.0;
        }
        let pred_collection = build_term_collection_design(new_data.view(), &fit.resolvedspec)
            .expect("rebuild predict design");
        pred_collection.design.apply(&fit.fit.beta)
    }))
    .expect("predict-path matvec must not panic on a CPU-only host");
    assert_eq!(eta.len(), n, "eta length should match training rows");
}
