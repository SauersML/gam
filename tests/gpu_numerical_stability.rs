use gam::solver::gpu::{Device, configure_device};
use ndarray::{Array1, Array2};

fn close(a: f64, b: f64, tol: f64) -> bool {
    let scale = 1.0_f64.max(a.abs()).max(b.abs());
    (a - b).abs() <= tol * scale
}

fn assert_arrays_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.dim(), b.dim());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            assert!(
                close(a[[i, j]], b[[i, j]], tol),
                "matrix mismatch at ({i},{j}): {} vs {}",
                a[[i, j]],
                b[[i, j]]
            );
        }
    }
}

fn assert_vec_close(a: &Array1<f64>, b: &Array1<f64>, tol: f64) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        assert!(
            close(a[i], b[i], tol),
            "vector mismatch at {i}: {} vs {}",
            a[i],
            b[i]
        );
    }
}

fn synthetic_case(
    n: usize,
    p: usize,
    ridge: f64,
) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((n, p), |(i, j)| {
        let phase = (i as f64 + 1.0) * 0.173 + (j as f64 + 1.0) * 0.379;
        phase.sin() * 0.4 + phase.cos() * 0.07
    });
    let weights = Array1::from_shape_fn(n, |i| 0.25 + ((i % 11) as f64 + 1.0) / 17.0);
    let penalty = Array2::from_shape_fn((p, p), |(i, j)| {
        if i == j {
            ridge + 1.0 + (i as f64) * 0.1
        } else {
            0.01 * ((i + j + 1) as f64).sin()
        }
    });
    let gradient = Array1::from_shape_fn(p, |j| ((j as f64 + 0.5) * 0.23).cos());
    (x, weights, penalty, gradient)
}

fn cpu_xtwx(x: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((x.ncols(), x.ncols()));
    for i in 0..x.nrows() {
        for a in 0..x.ncols() {
            let wa = weights[i] * x[[i, a]];
            for b in 0..x.ncols() {
                out[[a, b]] += wa * x[[i, b]];
            }
        }
    }
    out
}

fn cpu_cholesky_solve(h: &Array2<f64>, rhs: &Array2<f64>) -> (Array2<f64>, f64) {
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
    let mut logdet = 0.0_f64;
    for i in 0..n {
        logdet += l[[i, i]].ln();
    }
    (y, 2.0 * logdet)
}

fn cuda_available() -> bool {
    gam::gpu::runtime::GpuRuntime::global().is_some()
}

#[test]
fn pirls_gpu_matches_cpu_across_stability_grid() {
    if !cuda_available() {
        return;
    }
    configure_device(Device::Cuda);
    let mut cases = 0;
    for n in [8_usize, 13, 21, 34] {
        for p in [2_usize, 3, 5] {
            for ridge in [0.25_f64, 2.0] {
                let (x, weights, penalty, gradient) = synthetic_case(n, p, ridge);
                let xtwx_cpu = cpu_xtwx(&x, &weights);
                let xtwx_gpu =
                    gam::solver::gpu::pirls_gpu::weighted_crossprod_gpu(x.view(), weights.view())
                        .unwrap();
                assert_arrays_close(&xtwx_gpu, &xtwx_cpu, 1e-8);

                let mut h_cpu = xtwx_cpu.clone();
                h_cpu += &penalty;
                let rhs = Array2::from_shape_vec((p, 1), gradient.to_vec()).unwrap();
                let (sol_cpu, logdet_cpu) = cpu_cholesky_solve(&h_cpu, &rhs);
                let step = gam::solver::gpu::pirls_gpu::solve_pirls_step_gpu(
                    gam::solver::gpu::pirls_gpu::PirlsGpuInput {
                        x: x.view(),
                        weights: weights.view(),
                        penalty_hessian: penalty.view(),
                        gradient: gradient.view(),
                        lm_ridge: 0.0,
                    },
                )
                .unwrap();
                assert_arrays_close(&step.penalized_hessian, &h_cpu, 1e-8);
                assert!(close(step.logdet, logdet_cpu, 1e-8));
                assert_vec_close(&step.direction, &sol_cpu.column(0).mapv(|v| -v), 1e-8);
                cases += 1;
            }
        }
    }
    assert!(cases >= 20);
}

#[test]
fn reml_gpu_logdet_and_score_match_cpu() {
    if !cuda_available() {
        return;
    }
    configure_device(Device::Cuda);
    for p in [2_usize, 4, 7, 11] {
        let mut h = Array2::from_shape_fn((p, p), |(i, j)| {
            if i == j {
                3.0 + i as f64
            } else {
                0.02 * ((i + j + 3) as f64).cos()
            }
        });
        let ht = h.t().to_owned();
        h = (&h + &ht) * 0.5;
        let derivatives: Vec<Array2<f64>> = (0..3)
            .map(|a| {
                Array2::from_shape_fn(
                    (p, p),
                    |(i, j)| if i == j && i % 3 == a { 1.0 } else { 0.0 },
                )
            })
            .collect();
        let rhs_views: Vec<_> = derivatives.iter().map(|m| m.view()).collect();
        let evidence = gam::solver::gpu::reml_gpu::evidence_derivatives_gpu(
            gam::solver::gpu::reml_gpu::RemlGpuInput {
                penalized_hessian: h.view(),
                derivative_hessians: rhs_views,
            },
        )
        .unwrap();
        let mut identity = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            identity[[i, i]] = 1.0;
        }
        let (h_inv, logdet_cpu) = cpu_cholesky_solve(&h, &identity);
        assert!(close(evidence.logdet_hessian, logdet_cpu, 1e-8));
        for a in 0..3 {
            let mut trace = 0.0_f64;
            for i in 0..p {
                trace += h_inv[[i, i]] * derivatives[a][[i, i]];
            }
            assert!(close(evidence.gradient_rho[a], 0.5 * trace, 1e-8));
        }
    }
}
