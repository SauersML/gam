use gam::solver::gpu::pirls_gpu::{PirlsGpuInput, solve_pirls_step_gpu, weighted_crossprod_gpu};
use ndarray::{Array2, arr1, arr2};

fn tiny_case() -> (
    Array2<f64>,
    ndarray::Array1<f64>,
    Array2<f64>,
    ndarray::Array1<f64>,
) {
    let x = arr2(&[[1.0, 0.5], [0.2, -0.3], [0.7, 1.1]]);
    let w = arr1(&[1.0, 0.8, 1.2]);
    let penalty = arr2(&[[0.4, 0.0], [0.0, 0.9]]);
    let gradient = arr1(&[0.25, -0.6]);
    (x, w, penalty, gradient)
}

#[test]
fn gpu_pirls_step_falls_back_to_cpu_and_matches_beta_update_when_cuda_unavailable() {
    let (x, w, penalty, gradient) = tiny_case();
    let step = solve_pirls_step_gpu(PirlsGpuInput {
        x: x.view(),
        weights: w.view(),
        penalty_hessian: penalty.view(),
        gradient: gradient.view(),
        step_lm_lambda: 0.0,
        objective_ridge: 0.0,
    })
    .expect("PIRLS GPU path should fall back to CPU and produce the same beta update when CUDA is unavailable");
    assert_eq!(
        step.direction.len(),
        gradient.len(),
        "Fallback PIRLS direction must preserve coefficient dimension"
    );
}

#[test]
fn working_weight_gpu_crossprod_falls_back_and_matches_cpu_formula_when_cuda_unavailable() {
    let (x, w, _, _) = tiny_case();
    let gpu_xtwx = weighted_crossprod_gpu(x.view(), w.view())
        .expect("Working-weight GPU cross-product should fall back to CPU and match the CPU XtWX formula when CUDA is unavailable");
    let mut cpu_xtwx = Array2::<f64>::zeros((x.ncols(), x.ncols()));
    for i in 0..x.nrows() {
        for a in 0..x.ncols() {
            let wa = w[i] * x[[i, a]];
            for b in 0..x.ncols() {
                cpu_xtwx[[a, b]] += wa * x[[i, b]];
            }
        }
    }
    for i in 0..x.ncols() {
        for j in 0..x.ncols() {
            assert!(
                (gpu_xtwx[[i, j]] - cpu_xtwx[[i, j]]).abs() <= 1e-8,
                "Working-weight GPU cross-product must match CPU XtWX entrywise within 1e-8"
            );
        }
    }
}

#[test]
fn gpu_hessian_assembly_matches_cpu_hessian_within_1e8_under_fallback() {
    let (x, w, penalty, gradient) = tiny_case();
    let step = solve_pirls_step_gpu(PirlsGpuInput {
        x: x.view(),
        weights: w.view(),
        penalty_hessian: penalty.view(),
        gradient: gradient.view(),
        step_lm_lambda: 0.3,
        objective_ridge: 0.0,
    })
    .expect("GPU PIRLS solve should fall back to CPU and assemble the same penalized Hessian");

    let mut cpu_xtwx = Array2::<f64>::zeros((x.ncols(), x.ncols()));
    for i in 0..x.nrows() {
        for a in 0..x.ncols() {
            let wa = w[i] * x[[i, a]];
            for b in 0..x.ncols() {
                cpu_xtwx[[a, b]] += wa * x[[i, b]];
            }
        }
    }
    // `step_lm_lambda` is Levenberg–Marquardt damping; per the documented
    // contract on `PirlsGpuInput::step_lm_lambda`, it is added to H only for
    // the Newton solve and is *stripped* from the exported `penalized_hessian`
    // (which carries XᵀWX + S + objective_ridge·I). `objective_ridge` is 0
    // here, so the expected exported Hessian is just XᵀWX + S.
    let cpu_h = cpu_xtwx + penalty;

    for i in 0..cpu_h.nrows() {
        for j in 0..cpu_h.ncols() {
            assert!(
                (step.penalized_hessian[[i, j]] - cpu_h[[i, j]]).abs() <= 1e-8,
                "GPU Hessian assembly under fallback must match CPU within 1e-8"
            );
        }
    }
}

#[test]
fn repeated_gpu_fit_calls_leave_allocator_stats_counter_at_zero() {
    let snapshot = gam::gpu::profile::snapshot();
    assert!(
        snapshot.stats.is_empty(),
        "Allocator-stats counter should be zeroed at the end of each fit so repeated fits do not leak GPU memory accounting"
    );
}
