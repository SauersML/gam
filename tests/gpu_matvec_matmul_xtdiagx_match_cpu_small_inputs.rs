use gam::gpu;
use ndarray::{Array1, Array2};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn gpu_paths_match_cpu_to_1e8_when_available_for_small_matrices() {
    let a = Array2::from_shape_fn((4, 3), |(i, j)| ((i + 1 + j) as f64 * 0.37).sin());
    let b = Array2::from_shape_fn((3, 2), |(i, j)| ((2 * i + j + 1) as f64 * 0.19).cos());
    let x = Array2::from_shape_fn((5, 3), |(i, j)| ((i * 3 + j + 1) as f64 * 0.13).sin());
    let w = Array1::from_vec(vec![0.5, 1.2, 0.8, 1.1, 0.9]);
    let v = Array1::from_vec(vec![0.2, -0.3, 0.5]);

    if let Some(ab_gpu) = gpu::try_fast_ab(a.view(), b.view()) {
        let ab_cpu = a.dot(&b);
        for i in 0..ab_cpu.nrows() {
            for j in 0..ab_cpu.ncols() {
                assert!(
                    close(ab_gpu[[i, j]], ab_cpu[[i, j]], 1e-8),
                    "GPU matmul should match CPU matmul to within 1e-8 for the same small input."
                );
            }
        }
    }

    if let Some(av_gpu) = gpu::try_fast_av(a.view(), v.view()) {
        let av_cpu = a.dot(&v);
        for i in 0..av_cpu.len() {
            assert!(
                close(av_gpu[i], av_cpu[i], 1e-8),
                "GPU matvec should match CPU matvec to within 1e-8 for the same small input."
            );
        }
    }

    if let Some(xtwx_gpu) = gam::gpu::linalg_dispatch::try_fast_xt_diag_x(x.view(), w.view()) {
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
                    "GPU xt_diag_x should match CPU xt_diag_x to within 1e-8 for the same small input."
                );
            }
        }
    }
}
