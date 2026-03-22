//! Finite-difference crosscheck tests for aims2 math implementations.
//!
//! These tests exercise the profiled Gaussian REML objective through the
//! public `evaluate_externalgradients` / `evaluate_externalcost_andridge`
//! API, verifying that:
//!
//! 1. The analytic REML gradient matches FD of the cost (at multiple rho values).
//! 2. The analytic REML Hessian (via FD of gradient) is symmetric and
//!    consistent across the rho sweep.
//!
//! The spectral logdet gradient and moving nullspace correction tests live
//! as unit tests inside `src/solver/reml/unified.rs` because those types
//! are crate-internal.

use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradients,
};
use gam::smooth::BlockwisePenalty;
use gam::types::LikelihoodFamily;
use ndarray::{Array1, Array2, array};

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Central finite-difference gradient of a scalar function of rho.
fn central_fd_gradient<F>(rho: &Array1<f64>, f: &F) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let mut grad = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let h = 1e-4 * (1.0 + rho[k].abs());
        let mut rp = rho.clone();
        let mut rm = rho.clone();
        rp[k] += h;
        rm[k] -= h;
        let fp = f(&rp);
        let fm = f(&rm);
        grad[k] = (fp - fm) / (2.0 * h);
    }
    grad
}

/// Assert that two gradient vectors agree within tolerance, reporting
/// which component failed.
fn assert_gradient_fd_match(
    analytic: &Array1<f64>,
    fd: &Array1<f64>,
    abs_tol: f64,
    rel_tol: f64,
    label: &str,
) {
    assert_eq!(analytic.len(), fd.len(), "{label}: dimension mismatch");
    for k in 0..analytic.len() {
        let abs_err = (analytic[k] - fd[k]).abs();
        let scale = analytic[k].abs().max(fd[k].abs()).max(1e-8);
        let rel_err = abs_err / scale;
        assert!(
            abs_err < abs_tol || rel_err < rel_tol,
            "{label} gradient mismatch at k={k}: analytic={:.6e} fd={:.6e} abs={:.3e} rel={:.3e}",
            analytic[k],
            fd[k],
            abs_err,
            rel_err,
        );
    }
}

/// Build a Gaussian external problem for REML testing.
fn build_gaussian_external_problem(
    n: usize,
    p: usize,
) -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<BlockwisePenalty>) {
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        x[[i, 0]] = 1.0; // intercept
        for j in 1..p {
            let phase = j as f64 * 0.7;
            x[[i, j]] = (2.0 * std::f64::consts::PI * (t * (j as f64) + phase)).sin();
        }
    }

    // True coefficients
    let mut beta = Array1::<f64>::zeros(p);
    beta[0] = 1.5;
    for j in 1..p {
        beta[j] = 0.5 / (j as f64).sqrt();
    }
    let eta = x.dot(&beta);
    // y = eta + noise (deterministic "noise" for reproducibility)
    let y = eta.mapv(|e| e + 0.1 * (3.7 * e).sin());
    let w = Array1::<f64>::ones(n);

    // One penalty per non-intercept group
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    (x, y, w, vec![BlockwisePenalty::new(0..p, s)])
}

fn gaussian_opts(nullspace_dim: usize) -> ExternalOptimOptions {
    ExternalOptimOptions {
        family: LikelihoodFamily::GaussianIdentity,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims: vec![nullspace_dim],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1: Profiled Gaussian REML gradient matches FD of cost
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn profiled_gaussian_reml_gradient_matches_fd_single_penalty() {
    let (x, y, w, s_list) = build_gaussian_external_problem(80, 4);
    let offset = Array1::<f64>::zeros(y.len());
    let opts = gaussian_opts(1); // 1 unpenalized intercept

    for &rho_val in &[0.0, 2.0, 5.0, -1.0, 8.0] {
        let rho = array![rho_val];
        let (analytic, fd_from_api) = evaluate_externalgradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        )
        .expect("gradients should succeed");

        // Also compute FD ourselves from cost
        let cost_fn = |r: &Array1<f64>| -> f64 {
            evaluate_externalcost_andridge(
                y.view(),
                w.view(),
                x.view(),
                offset.view(),
                &s_list,
                &opts,
                r,
            )
            .expect("cost should succeed")
            .0
        };
        let our_fd = central_fd_gradient(&rho, &cost_fn);

        // Analytic vs our FD
        assert_gradient_fd_match(
            &analytic,
            &our_fd,
            5e-3,
            5e-2,
            &format!("Gaussian REML analytic-vs-FD at rho={rho_val}"),
        );

        // Also check the API's own FD agrees with ours
        assert_gradient_fd_match(
            &fd_from_api,
            &our_fd,
            5e-3,
            5e-2,
            &format!("Gaussian REML API-FD-vs-our-FD at rho={rho_val}"),
        );
    }
}

#[test]
fn profiled_gaussian_reml_gradient_matches_fd_two_penalties() {
    // Two independent penalty matrices.
    let n = 100;
    let p = 5;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = (3.0 * t).sin();
        x[[i, 3]] = (2.0 * t).cos();
        x[[i, 4]] = t * t;
    }
    let beta = array![1.0, -0.5, 0.3, 0.2, -0.1];
    let eta = x.dot(&beta);
    let y = eta.mapv(|e| e + 0.05 * (5.0 * e).sin());
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let mut s1 = Array2::<f64>::zeros((p, p));
    s1[[1, 1]] = 1.0;
    s1[[2, 2]] = 1.0;
    let mut s2 = Array2::<f64>::zeros((p, p));
    s2[[3, 3]] = 1.0;
    s2[[4, 4]] = 1.0;
    let s_list = vec![
        BlockwisePenalty::new(0..p, s1),
        BlockwisePenalty::new(0..p, s2),
    ];

    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::GaussianIdentity,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims: vec![1, 1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };

    let rho = array![1.5, -0.5];
    let (analytic, _) = evaluate_externalgradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .expect("gradients");

    let cost_fn = |r: &Array1<f64>| -> f64 {
        evaluate_externalcost_andridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            r,
        )
        .expect("cost")
        .0
    };
    let fd = central_fd_gradient(&rho, &cost_fn);

    assert_gradient_fd_match(
        &analytic,
        &fd,
        5e-3,
        5e-2,
        "Gaussian REML 2-penalty gradient",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2: Profiled Gaussian REML Hessian via FD of gradient
// ═══════════════════════════════════════════════════════════════════════════
//
// Computes ∂²V/∂ρ_k∂ρ_l by central FD of the analytic gradient,
// then verifies symmetry and finiteness.

#[test]
fn profiled_gaussian_reml_hessian_fd_symmetric_and_finite() {
    let n = 100;
    let p = 5;
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = t;
        x[[i, 2]] = (3.0 * t).sin();
        x[[i, 3]] = (2.0 * t).cos();
        x[[i, 4]] = t * t;
    }
    let beta = array![1.0, -0.5, 0.3, 0.2, -0.1];
    let eta = x.dot(&beta);
    let y = eta.mapv(|e| e + 0.05 * (5.0 * e).sin());
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);

    let mut s1 = Array2::<f64>::zeros((p, p));
    s1[[1, 1]] = 1.0;
    s1[[2, 2]] = 1.0;
    let mut s2 = Array2::<f64>::zeros((p, p));
    s2[[3, 3]] = 1.0;
    s2[[4, 4]] = 1.0;
    let s_list = vec![
        BlockwisePenalty::new(0..p, s1),
        BlockwisePenalty::new(0..p, s2),
    ];

    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::GaussianIdentity,
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        compute_inference: true,
        max_iter: 200,
        tol: 1e-10,
        nullspace_dims: vec![1, 1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
    };

    let rho = array![1.5, -0.5];
    let n_rho = rho.len();

    // Compute FD Hessian from gradient
    let eps = 1e-4;
    let mut fd_hess = Array2::<f64>::zeros((n_rho, n_rho));
    for l in 0..n_rho {
        let mut rp = rho.clone();
        rp[l] += eps;
        let (gp, _) = evaluate_externalgradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rp,
        )
        .expect("grad+");

        let mut rm = rho.clone();
        rm[l] -= eps;
        let (gm, _) = evaluate_externalgradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rm,
        )
        .expect("grad-");

        for k in 0..n_rho {
            fd_hess[[k, l]] = (gp[k] - gm[k]) / (2.0 * eps);
        }
    }

    // Verify finiteness
    for k in 0..n_rho {
        for l in 0..n_rho {
            assert!(
                fd_hess[[k, l]].is_finite(),
                "FD Hessian non-finite at [{k},{l}]: {}",
                fd_hess[[k, l]],
            );
        }
    }

    // Verify symmetry (FD Hessian should be approximately symmetric)
    for k in 0..n_rho {
        for l in (k + 1)..n_rho {
            let abs_err = (fd_hess[[k, l]] - fd_hess[[l, k]]).abs();
            let scale = fd_hess[[k, l]].abs().max(fd_hess[[l, k]].abs()).max(1e-8);
            let rel_err = abs_err / scale;
            assert!(
                rel_err < 5e-2,
                "FD Hessian not symmetric at [{k},{l}]: {:.6e} vs {:.6e} rel={:.3e}",
                fd_hess[[k, l]],
                fd_hess[[l, k]],
                rel_err,
            );
        }
    }
}
