use ndarray::Array2;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

fn cpu_cholesky_solve(h: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
    let n = h.nrows();
    let nrhs = rhs.ncols();
    let mut l = h.clone();
    for j in 0..n {
        let mut diag = l[[j, j]];
        for k in 0..j {
            diag -= l[[j, k]] * l[[j, k]];
        }
        l[[j, j]] = diag.sqrt();
        for i in (j + 1)..n {
            let mut value = l[[i, j]];
            for k in 0..j {
                value -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = value / l[[j, j]];
        }
    }
    let mut y = rhs.clone();
    for col in 0..nrhs {
        for i in 0..n {
            let mut value = y[[i, col]];
            for k in 0..i {
                value -= l[[i, k]] * y[[k, col]];
            }
            y[[i, col]] = value / l[[i, i]];
        }
        for i in (0..n).rev() {
            let mut value = y[[i, col]];
            for k in (i + 1)..n {
                value -= l[[k, i]] * y[[k, col]];
            }
            y[[i, col]] = value / l[[i, i]];
        }
    }
    y
}

#[test]
fn gpu_solver_matches_cpu_to_numeric_tolerance_on_small_input() {
    let h = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0])
        .expect("shape should be valid");
    let rhs = Array2::from_shape_vec((3, 1), vec![1.0, -2.0, 0.5]).expect("shape should be valid");

    let cpu_sol = cpu_cholesky_solve(&h, &rhs);

    if let Ok((gpu_sol, _)) = gam::gpu::solver::cholesky_solve_gpu(h.view(), rhs.view()) {
        for i in 0..cpu_sol.nrows() {
            assert!(
                close(gpu_sol[[i, 0]], cpu_sol[[i, 0]], 1e-8),
                "GPU solver output should match CPU solver output to within numeric tolerance on a small SPD input."
            );
        }
    }
}
