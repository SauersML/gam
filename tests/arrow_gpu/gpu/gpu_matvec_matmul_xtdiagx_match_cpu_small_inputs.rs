use gam::gpu;
use ndarray::{Array1, Array2};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

/// GEMM / GEMV / XᵀWX device parity against CPU references.
///
/// The earlier version of this test used 4×3-style "small" fixtures and wrapped
/// every device call in an `if let Some(..)` arm. Those shapes are far below the
/// GPU dispatch FLOP floor (`gemm_min_flops = xtwx_flops_min = 1e8`), so the
/// `try_fast_*` helpers return `None` even on a real GPU — meaning the assertions
/// NEVER ran on a GPU host and the test was vacuously green while exercising zero
/// device code. That is the device-PCG skip-pass class fixed in eee12f6b2.
///
/// Each fixture below is sized so `2·m·n·k` (GEMM), `2·m·k` (GEMV) and `2·n·p²`
/// (XᵀWX) all clear the 1e8 floor, so the device path is genuinely admitted when
/// a CUDA runtime is present. A `None` return is then a legitimate skip ONLY when
/// no runtime exists; with a runtime present it means the device declined a
/// workload it was sized to run — a real fault — and the test fails loud.
fn assert_present_or_no_runtime(declined: bool, op: &str) {
    if declined {
        assert!(
            gam::gpu::device_runtime::GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
                .unwrap_or_else(|error| panic!("GPU probe fault in parity decline: {error}"))
                .is_none(),
            "GPU {op} declined (returned None) on a host WITH a CUDA runtime present, \
             despite a fixture sized to clear the 1e8 dispatch FLOP floor. A \
             runtime-present decline on a floor-clearing workload is a real \
             device/dispatch fault, not a legitimate skip."
        );
        eprintln!("SKIP gpu {op}: no CUDA runtime");
    }
}

