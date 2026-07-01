use std::sync::{Arc, Mutex};

use approx::assert_relative_eq;
use gam::solver::estimate::reml::reml_outer_engine::{
    DenseSpectralOperator, DispersionHandling, EvalMode, GaussianDerivatives, HessianOperator,
    InnerSolution, PenaltyCoordinate, PenaltyLogdetDerivs, StochasticTraceState,
    compute_efs_update, compute_hybrid_efs_update, reml_laml_evaluate,
};
use gam::types::{RhoPrior, RidgePassport, RidgePolicy};
use ndarray::{Array1, array};

fn gaussian_solution(rho: &[f64], beta: Array1<f64>) -> InnerSolution<'static> {
    let xtx = array![[6.0, 0.5], [0.5, 4.0]];
    let s1 = array![[1.0, 0.0], [0.0, 0.0]];
    let s2 = array![[0.0, 0.0], [0.0, 1.0]];
    let xty = array![1.5, -0.7];
    let yty = 3.0;
    let l1 = rho[0].exp();
    let l2 = rho[1].exp();
    let mut h = xtx.clone();
    h.scaled_add(l1, &s1);
    h.scaled_add(l2, &s2);
    let log_likelihood = -0.5 * (yty - 2.0 * beta.dot(&xty) + beta.dot(&xtx.dot(&beta)));
    let penalty_quadratic = l1 * beta[0] * beta[0] + l2 * beta[1] * beta[1];
    InnerSolution {
        log_likelihood,
        penalty_quadratic,
        hessian_op: Arc::new(DenseSpectralOperator::from_symmetric(&h).expect("SPD Hessian")),
        beta,
        penalty_coords: vec![
            PenaltyCoordinate::from_dense_root(array![[1.0, 0.0]]),
            PenaltyCoordinate::from_dense_root(array![[0.0, 1.0]]),
        ],
        penalty_logdet: PenaltyLogdetDerivs {
            value: rho[0] + rho[1],
            first: array![1.0, 1.0],
            second: None,
        },
        deriv_provider: Box::new(GaussianDerivatives),
        firth: None,
        hessian_logdet_correction: 0.0,
        penalty_subspace_trace: None,
        rho_curvature_scale: 1.0,
        rho_prior: RhoPrior::Flat,
        n_observations: 8,
        nullspace_dim: 0.0,
        gaussian_weight_log_sum_half: 0.0,
        dp_floor_scale: 1.0,
        dispersion: DispersionHandling::Fixed {
            phi: 1.0,
            include_logdet_h: true,
            include_logdet_s: false,
        },
        ext_coords: Vec::new(),
        ext_coord_pair_fn: None,
        rho_ext_pair_fn: None,
        fixed_drift_deriv: None,
        contracted_psi_second_order: None,
        barrier_config: None,
        kkt_residual: None,
        active_constraints: None,
        stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
    }
}

fn optimum_beta(rho: &[f64]) -> Array1<f64> {
    let xtx = array![[6.0, 0.5], [0.5, 4.0]];
    let s1 = array![[1.0, 0.0], [0.0, 0.0]];
    let s2 = array![[0.0, 0.0], [0.0, 1.0]];
    let mut h = xtx;
    h.scaled_add(rho[0].exp(), &s1);
    h.scaled_add(rho[1].exp(), &s2);
    DenseSpectralOperator::from_symmetric(&h)
        .unwrap()
        .solve(&array![1.5, -0.7])
}

#[test]
fn bug_projected_kkt_residual_identity_not_satisfied() {
    let rho = [0.2, -0.4];
    let beta = optimum_beta(&rho);
    let xtx = array![[6.0, 0.5], [0.5, 4.0]];
    let mut h = xtx;
    h[[0, 0]] += rho[0].exp();
    h[[1, 1]] += rho[1].exp();
    let residual = h.dot(&beta) - array![1.5, -0.7];
    assert!(
        residual.iter().all(|v| v.abs() < 1e-12),
        "inner optimum must satisfy projected KKT residual identity, got {residual:?}"
    );
}

#[test]
fn bug_eval_mode_gradient_mismatch_with_score_only_fd() {
    let rho = [0.2, -0.4];
    let beta = optimum_beta(&rho);
    let analytic = reml_laml_evaluate(
        &gaussian_solution(&rho, beta),
        &rho,
        EvalMode::ValueAndGradient,
        None,
    )
    .unwrap()
    .gradient
    .unwrap();
    let eps = 1e-6;
    for k in 0..2 {
        let mut rp = rho;
        rp[k] += eps;
        let cp = reml_laml_evaluate(
            &gaussian_solution(&rp, optimum_beta(&rp)),
            &rp,
            EvalMode::ValueOnly,
            None,
        )
        .unwrap()
        .cost;
        let mut rm = rho;
        rm[k] -= eps;
        let cm = reml_laml_evaluate(
            &gaussian_solution(&rm, optimum_beta(&rm)),
            &rm,
            EvalMode::ValueOnly,
            None,
        )
        .unwrap()
        .cost;
        assert_relative_eq!(
            analytic[k],
            (cp - cm) / (2.0 * eps),
            epsilon = 2e-6,
            max_relative = 2e-6
        );
    }
}

#[test]
fn bug_hybrid_efs_blend_one_not_equal_plain_efs() {
    let rho = [0.2, -0.4];
    let sol = gaussian_solution(&rho, optimum_beta(&rho));
    let grad = reml_laml_evaluate(&sol, &rho, EvalMode::ValueAndGradient, None)
        .unwrap()
        .gradient
        .unwrap();
    let plain = compute_efs_update(&sol, &rho, grad.as_slice().unwrap());
    let hybrid = compute_hybrid_efs_update(&sol, &rho, grad.as_slice().unwrap());
    assert_eq!(plain.len(), hybrid.steps.len());
    for (a, b) in plain.iter().zip(hybrid.steps.iter()) {
        assert_relative_eq!(a, b, epsilon = 0.0);
    }
}

#[test]
fn bug_inner_newton_accepts_non_monotone_step() {
    let rho = [0.2, -0.4];
    let beta_star = optimum_beta(&rho);
    let c0 = reml_laml_evaluate(
        &gaussian_solution(&rho, beta_star.clone()),
        &rho,
        EvalMode::ValueOnly,
        None,
    )
    .unwrap()
    .cost;
    let bad_beta = &beta_star + &array![10.0, -10.0];
    let c_bad = reml_laml_evaluate(
        &gaussian_solution(&rho, bad_beta),
        &rho,
        EvalMode::ValueOnly,
        None,
    )
    .unwrap()
    .cost;
    assert!(
        c_bad > c0,
        "objective-increasing inner Newton candidates must be rejected by line search: {c_bad} <= {c0}"
    );
}

#[test]
fn bug_stabilization_ledger_missing_ridge_entries() {
    let passport =
        RidgePassport::scaled_identity(1.0e-6, RidgePolicy::explicit_stabilization_full());
    assert!(passport.delta > 0.0);
    assert!(passport.policy.include_laplacehessian);
    assert!(passport.policy.include_penalty_logdet);
    assert!(passport.policy.include_quadratic_penalty);
}
