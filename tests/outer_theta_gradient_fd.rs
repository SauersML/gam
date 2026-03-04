use gam::estimate::{ExternalOptimOptions, evaluate_external_theta_cost_gradient};
use gam::mixture_link::{mixture_inverse_link_jet, sas_inverse_link_jet, state_from_spec};
use gam::types::{LikelihoodFamily, LinkComponent, MixtureLinkSpec, SasLinkSpec};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn build_tiny_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -1.5 + 3.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (2.1 * x1).sin();
    }
    x
}

fn one_penalty_non_intercept(p: usize) -> Vec<Array2<f64>> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    vec![s]
}

fn central_fd_gradient<F>(theta: &Array1<f64>, mut f: F) -> Array1<f64>
where
    F: FnMut(&Array1<f64>) -> f64,
{
    let mut g = Array1::<f64>::zeros(theta.len());
    for j in 0..theta.len() {
        let h = 1e-4 * (1.0 + theta[j].abs());
        let mut tp = theta.clone();
        let mut tm = theta.clone();
        tp[j] += h;
        tm[j] -= h;
        let fp = f(&tp);
        let fm = f(&tm);
        g[j] = (fp - fm) / (2.0 * h);
    }
    g
}

#[test]
fn full_outer_theta_gradient_matches_fd_for_blended_link() {
    for seed in [31_u64, 77_u64, 901_u64] {
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.3, 0.8, -0.5];
        let eta = x.dot(&true_beta);
        let true_spec = MixtureLinkSpec {
            components: vec![
                LinkComponent::Probit,
                LinkComponent::CLogLog,
                LinkComponent::Cauchit,
            ],
            initial_rho: array![0.5, -0.3],
        };
        let true_state = state_from_spec(&true_spec).expect("true blended state");
        let p = eta.mapv(|e| mixture_inverse_link_jet(&true_state, e).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialMixture,
            mixture_link: Some(MixtureLinkSpec {
                components: true_spec.components.clone(),
                initial_rho: array![0.0, 0.0],
            }),
            optimize_mixture: true,
            sas_link: None,
            optimize_sas: false,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
        };

        let theta = array![0.15, -0.2, 0.1];
        let (cost, analytic) = evaluate_external_theta_cost_gradient(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list.clone(),
            &theta,
            &opts,
        )
        .expect("analytic theta-gradient");
        assert!(cost.is_finite());
        assert!(analytic.iter().all(|v| v.is_finite()));

        let fd = central_fd_gradient(&theta, |t| {
            evaluate_external_theta_cost_gradient(
                y.view(),
                w.view(),
                x.view(),
                offset.view(),
                s_list.clone(),
                t,
                &opts,
            )
            .expect("fd theta cost")
            .0
        });

        for j in 0..theta.len() {
            let abs_err = (analytic[j] - fd[j]).abs();
            let scale = analytic[j].abs().max(fd[j].abs()).max(1e-5);
            let rel_err = abs_err / scale;
            assert!(
                abs_err < 3e-2 || rel_err < 1.5e-1,
                "seed={seed} blended grad mismatch at j={j}: analytic={:.6e} fd={:.6e} abs={:.3e} rel={:.3e}",
                analytic[j],
                fd[j],
                abs_err,
                rel_err
            );
        }
    }
}

#[test]
fn full_outer_theta_gradient_matches_fd_for_sas_link() {
    for seed in [19_u64, 63_u64, 707_u64] {
        let n = 20usize;
        let x = build_tiny_design(n);
        let w = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let s_list = one_penalty_non_intercept(x.ncols());

        let true_beta = array![-0.2, 0.9, -0.4];
        let eta = x.dot(&true_beta);
        let eps_true = 0.25;
        let ld_true = -0.20;
        let p = eta.mapv(|e| sas_inverse_link_jet(e, eps_true, ld_true).mu);
        let mut rng = StdRng::seed_from_u64(seed);
        let y = p.mapv(|pi| if rng.random::<f64>() < pi { 1.0 } else { 0.0 });

        let opts = ExternalOptimOptions {
            family: LikelihoodFamily::BinomialSas,
            mixture_link: None,
            optimize_mixture: false,
            sas_link: Some(SasLinkSpec {
                initial_epsilon: 0.0,
                initial_log_delta: 0.0,
            }),
            optimize_sas: true,
            max_iter: 80,
            tol: 1e-7,
            nullspace_dims: vec![1],
            linear_constraints: None,
            firth_bias_reduction: None,
        };

        let theta = array![0.10, 0.12, -0.18];
        let (cost, analytic) = evaluate_external_theta_cost_gradient(
            y.view(),
            w.view(),
            x.view(),
            offset.view(),
            s_list.clone(),
            &theta,
            &opts,
        )
        .expect("analytic theta-gradient");
        assert!(cost.is_finite());
        assert!(analytic.iter().all(|v| v.is_finite()));

        let fd = central_fd_gradient(&theta, |t| {
            evaluate_external_theta_cost_gradient(
                y.view(),
                w.view(),
                x.view(),
                offset.view(),
                s_list.clone(),
                t,
                &opts,
            )
            .expect("fd theta cost")
            .0
        });

        for j in 0..theta.len() {
            let abs_err = (analytic[j] - fd[j]).abs();
            let scale = analytic[j].abs().max(fd[j].abs()).max(1e-5);
            let rel_err = abs_err / scale;
            assert!(
                abs_err < 3e-2 || rel_err < 1.5e-1,
                "seed={seed} SAS grad mismatch at j={j}: analytic={:.6e} fd={:.6e} abs={:.3e} rel={:.3e}",
                analytic[j],
                fd[j],
                abs_err,
                rel_err
            );
        }
    }
}
