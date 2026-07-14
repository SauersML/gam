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

#[test]
fn gpu_paths_match_cpu_to_1e8_on_floor_clearing_matrices() {
    // The floor-clearing fixtures below allocate ~460 MB total; only build them on
    // a host with a CUDA runtime (where the device path actually runs). On a
    // CPU-only host there is nothing to validate (the helpers would return None for
    // lack of a device, not for lack of work) — skip cleanly without the heavy
    // allocations so CPU CI stays fast and light.
    if gam::gpu::device_runtime::GpuRuntime::resolve(gam::gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in matrix parity test: {error}"))
        .is_none()
    {
        eprintln!("SKIP gpu_paths_match_cpu_to_1e8_on_floor_clearing_matrices: no CUDA runtime");
        return;
    }

    // GEMM A·B: 2·m·n·k = 2·400·400·400 ≈ 1.28e8 ≥ 1e8.
    let a = Array2::from_shape_fn((400, 400), |(i, j)| ((i + 1 + j) as f64 * 0.37).sin());
    let b = Array2::from_shape_fn((400, 400), |(i, j)| ((2 * i + j + 1) as f64 * 0.19).cos());
    match gpu::try_fast_ab(a.view(), b.view()) {
        Some(ab_gpu) => {
            let ab_cpu = a.dot(&b);
            for i in 0..ab_cpu.nrows() {
                for j in 0..ab_cpu.ncols() {
                    assert!(
                        close(ab_gpu[[i, j]], ab_cpu[[i, j]], 1e-8),
                        "GPU matmul should match CPU matmul to within 1e-8 (i={i}, j={j})."
                    );
                }
            }
        }
        None => assert_present_or_no_runtime(true, "matmul (try_fast_ab)"),
    }

    // GEMV A·v: 2·m·k = 2·50000·1100 = 1.1e8 ≥ 1e8. A thin-tall shape keeps the
    // CPU reference dot (≈5.5e7 flops) cheap while clearing the GEMM FLOP floor.
    let av_a = Array2::from_shape_fn((50000, 1100), |(i, j)| {
        ((i * 3 + j + 1) as f64 * 0.013).sin()
    });
    let v = Array1::from_shape_fn(1100, |i| ((i + 1) as f64 * 0.07).cos());
    match gpu::try_fast_av(av_a.view(), v.view()) {
        Some(av_gpu) => {
            let av_cpu = av_a.dot(&v);
            for i in 0..av_cpu.len() {
                assert!(
                    close(av_gpu[i], av_cpu[i], 1e-8),
                    "GPU matvec should match CPU matvec to within 1e-8 (i={i})."
                );
            }
        }
        None => assert_present_or_no_runtime(true, "matvec (try_fast_av)"),
    }

    // XᵀWX: 2·n·p² = 2·60000·40² = 1.92e8 ≥ 1e8.
    let x = Array2::from_shape_fn((60000, 40), |(i, j)| ((i * 3 + j + 1) as f64 * 0.013).sin());
    let w = Array1::from_shape_fn(60000, |i| 0.5 + ((i % 7) as f64) * 0.1);
    match gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view()) {
        Some(xtwx_gpu) => {
            let mut xtwx_cpu = Array2::<f64>::zeros((x.ncols(), x.ncols()));
            for i in 0..x.nrows() {
                for c1 in 0..x.ncols() {
                    for c2 in 0..x.ncols() {
                        xtwx_cpu[[c1, c2]] += x[[i, c1]] * w[i] * x[[i, c2]];
                    }
                }
            }
            for i in 0..xtwx_cpu.nrows() {
                for j in 0..xtwx_cpu.ncols() {
                    assert!(
                        close(xtwx_gpu[[i, j]], xtwx_cpu[[i, j]], 1e-8),
                        "GPU xt_diag_x should match CPU xt_diag_x to within 1e-8 (i={i}, j={j})."
                    );
                }
            }
        }
        None => assert_present_or_no_runtime(true, "xt_diag_x (try_fast_xt_diag_x)"),
    }
}
