use ndarray::Array2;

#[test]
fn gpu_solver_dimension_mismatch_returns_error_and_never_silent_cpu_fallback() {
    let h =
        Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("shape should be valid");
    let rhs_bad =
        Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).expect("shape should be valid");

    let err = gam::gpu::solver::cholesky_solve_gpu(h.view(), rhs_bad.view())
        .expect_err("Dimension mismatch should return an error instead of silently falling back.");
    assert!(
        err.to_ascii_lowercase().contains("dimension")
            || err.to_ascii_lowercase().contains("hessian"),
        "Dimension mismatch errors should clearly report the mismatch and avoid silent fallback."
    );
}
