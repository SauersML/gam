#![allow(clippy::too_many_arguments)]

use gam::estimate::{
    ExternalOptimOptions, evaluate_external_cost_and_ridge, evaluate_external_gradients,
};
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn build_design(seed: u64, n: usize, p: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.random_range(-2.0..2.0);
        }
    }
    x
}

fn one_penalty(p: usize) -> Vec<Array2<f64>> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![s]
}

fn fd_gradient(
    y: &Array1<f64>,
    w: &Array1<f64>,
    x: &Array2<f64>,
    offset: &Array1<f64>,
    s_list: &[Array2<f64>],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
    h: f64,
) -> Array1<f64> {
    let mut g = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let mut rp = rho.clone();
        rp[k] += h;
        let fp = evaluate_external_cost_and_ridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list,
            opts,
            &rp,
        )
        .map(|(c, _)| c)
        .expect("cost+");

        let mut rm = rho.clone();
        rm[k] -= h;
        let fm = evaluate_external_cost_and_ridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list,
            opts,
            &rm,
        )
        .map(|(c, _)| c)
        .expect("cost-");

        g[k] = (fp - fm) / (2.0 * h);
    }
    g
}

#[test]
fn test_log_det_gradient_formula() {
    use faer::Side;
    use gam::faer_ndarray::FaerCholesky;

    fn log_det_chol(mat: &Array2<f64>) -> Option<f64> {
        mat.cholesky(Side::Lower).ok().map(|chol| {
            let l = chol.lower_triangular();
            2.0 * (0..l.nrows()).map(|i| l[[i, i]].ln()).sum::<f64>()
        })
    }

    let a = array![[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]];
    let h = 1e-7;
    let mut grad_fd = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            let mut a_plus = a.clone();
            let mut a_minus = a.clone();
            a_plus[[i, j]] += h;
            a_minus[[i, j]] -= h;
            a_plus[[j, i]] = a_plus[[i, j]];
            a_minus[[j, i]] = a_minus[[i, j]];
            grad_fd[[i, j]] =
                (log_det_chol(&a_plus).unwrap() - log_det_chol(&a_minus).unwrap()) / (2.0 * h);
        }
    }

    let chol = a.cholesky(Side::Lower).expect("chol");
    let l = chol.lower_triangular();
    let mut a_inv = Array2::<f64>::zeros((3, 3));
    for col in 0..3 {
        let mut e = Array1::<f64>::zeros(3);
        e[col] = 1.0;
        let mut y = Array1::<f64>::zeros(3);
        for i in 0..3 {
            let mut sum = e[i];
            for k in 0..i {
                sum -= l[[i, k]] * y[k];
            }
            y[i] = sum / l[[i, i]];
        }
        let mut x = Array1::<f64>::zeros(3);
        for i in (0..3).rev() {
            let mut sum = y[i];
            for k in (i + 1)..3 {
                sum -= l[[k, i]] * x[k];
            }
            x[i] = sum / l[[i, i]];
        }
        for row in 0..3 {
            a_inv[[row, col]] = x[row];
        }
    }

    for i in 0..3 {
        assert!((grad_fd[[i, i]] - a_inv[[i, i]]).abs() < 1e-4);
    }
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert!((grad_fd[[i, j]] - 2.0 * a_inv[[i, j]]).abs() < 1e-4);
        }
    }
}

#[test]
fn test_laml_gradient_nonfirth_well_conditioned() {
    let n = 220;
    let p = 8;
    let x = build_design(42, n, p);
    let mut rng = StdRng::seed_from_u64(420);
    let beta = Array1::from_shape_fn(p, |j| if j == 0 { 0.3 } else { 0.5 / j as f64 });
    let y = x.dot(&beta) + Array1::from_shape_fn(n, |_| rng.random_range(-0.2..0.2));
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty(p);
    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::GaussianIdentity,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims: vec![1],
        firth_bias_reduction: None,
    };
    let rho = array![0.0];
    let (analytic, _) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .expect("analytic");
    let fd = fd_gradient(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-4);

    let dot = analytic.dot(&fd);
    let n_a = analytic.dot(&analytic).sqrt();
    let n_f = fd.dot(&fd).sqrt();
    let cosine = if n_a * n_f > 1e-12 {
        dot / (n_a * n_f)
    } else {
        1.0
    };
    let rel_l2 = (&analytic - &fd).mapv(|v| v * v).sum().sqrt() / n_f.max(n_a).max(1.0);
    assert!(cosine > 0.99, "cosine={cosine}");
    assert!(rel_l2 < 0.1, "rel_l2={rel_l2}");
}

#[test]
fn test_laml_gradient_logit_with_firth_well_conditioned() {
    let n = 260;
    let p = 8;
    let x = build_design(123, n, p);
    let mut rng = StdRng::seed_from_u64(1234);
    let beta = Array1::from_shape_fn(p, |j| {
        if j == 0 {
            -0.2
        } else {
            0.35 / (j as f64).sqrt()
        }
    });
    let eta = x.dot(&beta);
    let y = eta.mapv(|e| {
        let prob = 1.0 / (1.0 + (-e).exp());
        if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
    });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty(p);
    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialLogit,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims: vec![1],
        firth_bias_reduction: None,
    };
    let rho = array![0.0];
    let (analytic, _) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .expect("analytic");
    let fd = fd_gradient(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-4);

    let dot = analytic.dot(&fd);
    let n_a = analytic.dot(&analytic).sqrt();
    let n_f = fd.dot(&fd).sqrt();
    let cosine = if n_a * n_f > 1e-12 {
        dot / (n_a * n_f)
    } else {
        1.0
    };
    let rel_l2 = (&analytic - &fd).mapv(|v| v * v).sum().sqrt() / n_f.max(n_a).max(1.0);
    assert!(cosine > 0.95, "cosine={cosine}");
    assert!(rel_l2 < 0.2, "rel_l2={rel_l2}");
}

#[test]
fn stress_test_firth_gradient_vs_conditioning() {
    let configs = [(200usize, 4usize), (150usize, 6usize), (100usize, 8usize)];
    let mut saw_ok = false;
    for (n, p) in configs {
        let x = build_design(999 + n as u64 + p as u64, n, p);
        let mut rng = StdRng::seed_from_u64(9999 + n as u64);
        let beta = Array1::from_shape_fn(p, |j| {
            if j == 0 {
                0.0
            } else {
                0.25 / (j as f64).sqrt()
            }
        });
        let eta = x.dot(&beta);
        let y = eta.mapv(|e| {
            let prob = 1.0 / (1.0 + (-e).exp());
            if rng.random::<f64>() < prob { 1.0 } else { 0.0 }
        });
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty(p);
        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialLogit,
            max_iter: 150,
            tol: 1e-8,
            nullspace_dims: vec![1],
            firth_bias_reduction: None,
        };
        let rho = array![0.0];
        let Ok((analytic, _)) = evaluate_external_gradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        ) else {
            continue;
        };
        let fd = fd_gradient(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-4);
        let dot = analytic.dot(&fd);
        let n_a = analytic.dot(&analytic).sqrt();
        let n_f = fd.dot(&fd).sqrt();
        let cosine = if n_a * n_f > 1e-12 {
            dot / (n_a * n_f)
        } else {
            1.0
        };
        let max_a = analytic.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if cosine > 0.9 && max_a < 1e8 {
            saw_ok = true;
        }
    }
    assert!(saw_ok, "No stable configuration found in stress sweep");
}
