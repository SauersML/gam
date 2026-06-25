//! CPU/GPU parity for the spectral leverage diagonal `h[i] = ‖(X G)_{i,:}‖²`
//! offloaded to the device pool in issue #922.
//!
//! The GPU path (`gpu::linalg_dispatch::try_fast_spectral_leverage_diagonal`) only
//! engages when a device was probed AND the workload clears the `XtDiagX`
//! dispatch floor (n ≥ `xtwx_n_min`, 2·n·p² ≥ `xtwx_flops_min`); the shape
//! below is sized to clear it. On a machine without a usable GPU the function
//! returns `None` and the `Some(_)` arm is skipped — the test then asserts
//! nothing, exactly like the sibling `gpu_*_match_cpu` parity tests. On the
//! V100 box it exercises the real `scatter_batched` row-block fan-out and
//! checks it reproduces the CPU `X·G` then row-wise sum-of-squares to 1e-8.

use gam::gpu;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::Array2;

#[test]
fn gpu_spectral_leverage_diagonal_matches_cpu_when_available() {
    // n·p² = 60_000·40² = 9.6e7 → 2·n·p² = 1.92e8 ≥ xtwx_flops_min (1e8) and
    // n ≥ xtwx_n_min (50_000), so the GPU gate admits this shape on the box.
    let n = 60_000usize;
    let p = 40usize;
    let rank = 40usize;

    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        (((i * 7 + j * 3 + 1) as f64) * 0.001).sin()
    });
    let g = Array2::from_shape_fn((p, rank), |(i, j)| (((i + 2 * j + 1) as f64) * 0.013).cos());
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));

    let Some(gpu_lev) =
        gpu::linalg_dispatch::try_fast_spectral_leverage_diagonal(&design, g.view())
    else {
        // The fixture (2·n·p² = 1.92e8 ≥ 1e8, n = 60_000 ≥ 50_000) clears the
        // XtDiagX dispatch floor, so with a CUDA runtime present the device path
        // MUST engage. A `None` here therefore means the device declined a
        // workload it was sized to run — a real device/dispatch fault, not a
        // legitimate skip (the device-PCG skip-pass class, eee12f6b2). The old
        // bare `return` asserted nothing and passed silently on a GPU host.
        assert!(
            gam::gpu::device_runtime::GpuRuntime::global().is_none(),
            "GPU spectral-leverage diagonal declined (returned None) on a host WITH a \
             CUDA runtime present, despite a fixture sized to clear the XtDiagX floor \
             — a real device/dispatch fault, not a no-CUDA skip."
        );
        eprintln!("SKIP gpu_spectral_leverage_diagonal_matches_cpu: no CUDA runtime");
        return;
    };

    assert_eq!(
        gpu_lev.len(),
        n,
        "GPU leverage diagonal must have one entry per design row"
    );

    // Reference: h[i] = ‖(X G)_{i,:}‖² computed on the CPU. The only operation
    // relocated to the device is the X·G GEMM, so the worst relative error
    // reflects cuBLAS-vs-faer reduction order alone and must sit at f64 noise.
    let xg = x.dot(&g);
    let mut worst_rel = 0.0f64;
    for (i, row) in xg.outer_iter().enumerate() {
        let cpu: f64 = row.iter().map(|&v| v * v).sum();
        let rel = (gpu_lev[i] - cpu).abs() / cpu.abs().max(1.0);
        worst_rel = worst_rel.max(rel);
    }
    assert!(
        worst_rel <= 1e-8,
        "GPU spectral leverage worst relative error {worst_rel:e} must be ≤ 1e-8 vs CPU ‖X·G‖²"
    );
}
