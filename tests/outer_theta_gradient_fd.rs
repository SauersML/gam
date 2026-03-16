//! Outer-theta gradient vs finite-difference crosscheck for blended (mixture) and SAS links.
//!
//! These tests build a small design matrix, generate binary outcomes from
//! blended or SAS inverse-link transformations, then verify that the analytic
//! REML outer gradient from `evaluate_externalgradients` matches a manual
//! finite-difference gradient computed via `evaluate_externalcost_andridge`.

use gam::estimate::{
    ExternalOptimOptions, evaluate_externalcost_andridge, evaluate_externalgradients,
};
use gam::mixture_link::{mixture_inverse_link_jet, sas_inverse_link_jet, state_fromspec};
use gam::types::{LikelihoodFamily, LinkComponent, MixtureLinkSpec, SasLinkSpec};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn build_tiny_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / (n as f64);
        let x1 = -2.0 + 4.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (1.5 * x1).sin();
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

/// Central finite-difference gradient of the REML cost w.r.t. rho.
fn fd_gradient_of_cost(
    y: &Array1<f64>,
    w: &Array1<f64>,
    x: &Array2<f64>,
    offset: &Array1<f64>,
    s_list: &[Array2<f64>],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
    h: f64,
) -> Array1<f64> {
    let mut grad = Array1::<f64>::zeros(rho.len());
    for k in 0..rho.len() {
        let mut rp = rho.clone();
        rp[k] += h;
        let (cp, _) = evaluate_externalcost_andridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list,
            opts,
            &rp,
        )
        .expect("cost+");

        let mut rm = rho.clone();
        rm[k] -= h;
        let (cm, _) = evaluate_externalcost_andridge(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list,
            opts,
            &rm,
        )
        .expect("cost-");

        grad[k] = (cp - cm) / (2.0 * h);
    }
    grad
}

#[test]
fn full_outer_thetagradient_matchesfd_for_blended_link() {
    let n = 300usize;
    let x = build_tiny_design(n);
    let p = x.ncols();
    let beta_true = array![-0.25, 0.8, -0.5];
    let eta = x.dot(&beta_true);

    let mixspec = MixtureLinkSpec {
        components: vec![
            LinkComponent::Probit,
            LinkComponent::CLogLog,
            LinkComponent::Logit,
        ],
        initial_rho: array![0.5, -0.3],
    };
    let mix_state = state_fromspec(&mixspec).expect("mix state");
    let p_true = eta.mapv(|e| mixture_inverse_link_jet(&mix_state, e).mu);

    let mut rng = StdRng::seed_from_u64(19);
    let y = p_true.mapv(|prob| if rng.random::<f64>() < prob { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty(p);

    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialMixture,
        mixture_link: Some(mixspec),
        optimize_mixture: true,
        sas_link: None,
        optimize_sas: false,
        compute_inference: false,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
    };

    for rho_val in [0.0_f64, 2.0, 5.0, -1.0] {
        let rho = array![rho_val];
        let (analytic, api_fd) = evaluate_externalgradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        )
        .expect("gradients");

        let manual_fd = fd_gradient_of_cost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-4);

        for k in 0..rho.len() {
            let scale = analytic[k].abs().max(manual_fd[k].abs()).max(1.0);
            let rel_err_manual = (analytic[k] - manual_fd[k]).abs() / scale;
            let rel_err_api = (analytic[k] - api_fd[k]).abs() / scale;
            assert!(
                rel_err_manual < 0.05,
                "blended rho={rho_val} dim={k}: analytic={:.6e} manual_fd={:.6e} rel_err={rel_err_manual:.3e}",
                analytic[k], manual_fd[k]
            );
            assert!(
                rel_err_api < 0.05,
                "blended rho={rho_val} dim={k}: analytic={:.6e} api_fd={:.6e} rel_err={rel_err_api:.3e}",
                analytic[k], api_fd[k]
            );
        }
    }
}

#[test]
fn full_outer_thetagradient_matchesfd_for_sas_link() {
    let n = 300usize;
    let x = build_tiny_design(n);
    let p = x.ncols();
    let beta_true = array![-0.30, 1.0, -0.4];
    let eta = x.dot(&beta_true);
    let eps_true = 0.25;
    let log_delta_true = -0.2;
    let p_true = eta.mapv(|e| sas_inverse_link_jet(e, eps_true, log_delta_true).mu);

    let mut rng = StdRng::seed_from_u64(42);
    let y = p_true.mapv(|prob| if rng.random::<f64>() < prob { 1.0 } else { 0.0 });
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let s_list = one_penalty(p);

    let opts = ExternalOptimOptions {
        family: LikelihoodFamily::BinomialSas,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: Some(SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        }),
        optimize_sas: true,
        compute_inference: false,
        max_iter: 80,
        tol: 1e-7,
        nullspace_dims: vec![1],
        linear_constraints: None,
        firth_bias_reduction: None,
        penalty_shrinkage_floor: None,
    };

    for rho_val in [0.0_f64, 2.0, 5.0, -1.0] {
        let rho = array![rho_val];
        let (analytic, api_fd) = evaluate_externalgradients(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        )
        .expect("gradients");

        let manual_fd = fd_gradient_of_cost(&y, &w, &x, &offset, &s_list, &opts, &rho, 1e-4);

        for k in 0..rho.len() {
            let scale = analytic[k].abs().max(manual_fd[k].abs()).max(1.0);
            let rel_err_manual = (analytic[k] - manual_fd[k]).abs() / scale;
            let rel_err_api = (analytic[k] - api_fd[k]).abs() / scale;
            assert!(
                rel_err_manual < 0.05,
                "sas rho={rho_val} dim={k}: analytic={:.6e} manual_fd={:.6e} rel_err={rel_err_manual:.3e}",
                analytic[k], manual_fd[k]
            );
            assert!(
                rel_err_api < 0.05,
                "sas rho={rho_val} dim={k}: analytic={:.6e} api_fd={:.6e} rel_err={rel_err_api:.3e}",
                analytic[k], api_fd[k]
            );
        }
    }
}
