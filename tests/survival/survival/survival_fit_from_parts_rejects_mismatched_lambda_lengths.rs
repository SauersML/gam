use gam::families::survival::location_scale::{
    SurvivalLocationScaleFitResultParts, survival_fit_from_parts,
};
use ndarray::{Array2, array};

#[test]
fn survival_fit_from_parts_rejects_mismatched_lambda_lengths() {
    let parts = SurvivalLocationScaleFitResultParts {
        beta_time: array![0.1, -0.2],
        beta_threshold: array![0.3, 0.4],
        beta_log_sigma: array![0.5],
        beta_link_wiggle: None,
        link_wiggle_knots: None,
        link_wiggle_degree: None,
        lambdas_time: array![1.0],
        lambdas_threshold: array![2.0, 3.0, 4.0],
        lambdas_log_sigma: array![5.0, 6.0],
        lambdas_linkwiggle: None,
        log_likelihood: -10.0,
        reml_score: -9.0,
        stable_penalty_term: 0.0,
        penalized_objective: 10.0,
        outer_iterations: 1,
        outer_gradient_norm: Some(0.1),
        outer_converged: true,
        covariance_conditional: Some(Array2::eye(5)),
        geometry: None,
        used_device: false,
        penalty_block_trace: Vec::new(),
        edf_by_block: Vec::new(),
    };

    let result = survival_fit_from_parts(parts);
    assert!(
        result.is_err(),
        "survival_fit_from_parts should reject inconsistent lambda lengths for parameter blocks instead of silently accepting them"
    );
}
