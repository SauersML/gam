//! CPU/GPU parity for the spectral leverage diagonal `h[i] = ‖(X G)_{i,:}‖²`
//! offloaded to the device pool in issue #922.
//!
//! The GPU path (`gpu::linalg::try_fast_spectral_leverage_diagonal`) only
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

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

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
    let g = Array2::from_shape_fn((p, rank), |(i, j)| {
        (((i + 2 * j + 1) as f64) * 0.013).cos()
    });
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));

    let Some(gpu_lev) = gpu::linalg::try_fast_spectral_leverage_diagonal(&design, g.view()) else {
        // No usable GPU (or below the dispatch floor): the CPU faer stream in
        // `DenseSpectralOperator::xt_logdet_kernel_x_diagonal` is the only path.
        return;
    };

    assert_eq!(
        gpu_lev.len(),
        n,
        "GPU leverage diagonal must have one entry per design row"
    );

    // Reference: h[i] = ‖(X G)_{i,:}‖² computed on the CPU.
    let xg = x.dot(&g);
    for i in 0..n {
        let cpu: f64 = xg.row(i).iter().map(|&v| v * v).sum();
        assert!(
            close(gpu_lev[i], cpu, 1e-8),
            "GPU spectral leverage row {i} ({}) must match CPU reference ({cpu}) to 1e-8",
            gpu_lev[i]
        );
    }
}
