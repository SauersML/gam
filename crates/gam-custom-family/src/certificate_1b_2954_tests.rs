//! #2954 stage 1b on custom-family routes: the ρ domain declares which of its
//! faces are the terms' limit models.

use super::*;
use ndarray::array;

/// One block whose design is `diag(√γ₁, √γ₂)` under an identity penalty, so its
/// penalty-range spectrum is exactly `{γ₁, γ₂}`.
fn diagonal_term(gammas: [f64; 2]) -> (Vec<ParameterBlockSpec>, PenaltyLabelLayout) {
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
        [gammas[0].sqrt(), 0.0],
        [0.0, gammas[1].sqrt()]
    ]));
    let spec = ParameterBlockSpec {
        name: "wiggle".to_string(),
        design,
        offset: array![0.0, 0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0, 0.0], [0.0, 1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0, 0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let layout = PenaltyLabelLayout {
        penalty_counts: vec![1],
        physical_to_outer: vec![Some(0)],
        fixed_log_lambdas: vec![None],
        initial_rho: array![0.0],
        joint_specs: std::sync::Arc::new(Vec::new()),
        joint_roots: std::sync::Arc::new(Vec::new()),
        joint_to_outer: Vec::new(),
    };
    (vec![spec], layout)
}

#[test]
fn a_terms_own_resolvability_edges_are_its_limit_faces_2954() {
    let (specs, layout) = diagonal_term([1.0, 4.0]);
    let domain = resolvability_rho_domain_and_limit_faces(&specs, &layout, 1, None)
        .expect("domain construction");
    let (lower, upper) =
        resolvability_rho_domain(&specs, &layout, 1, None).expect("domain construction");
    assert_eq!((domain.lower.clone(), domain.upper.clone()), (lower, upper));
    assert_eq!(domain.lower_is_limit, vec![true]);
    assert_eq!(domain.upper_is_limit, vec![true]);
}

#[test]
fn a_literal_edge_is_not_a_limit_face_2954() {
    // The representable log-strength range cuts the λ → ∞ edge.
    let (specs, layout) = diagonal_term([1.0e300, 1.0e300]);
    let domain = resolvability_rho_domain_and_limit_faces(&specs, &layout, 1, None)
        .expect("domain construction");
    assert_eq!(domain.upper[0], gam_problem::LOG_STRENGTH_MAX);
    assert_eq!(domain.upper_is_limit, vec![false]);
    // The λ → 0 edge is the term's own, and `γ = 1e300` identifies its
    // unpenalized fit although `Σγ²` overflows unscaled.
    assert_eq!(domain.lower_is_limit, vec![true]);
    // A family floor raises the λ → 0 edge above the term's own.
    let (specs, layout) = diagonal_term([1.0, 4.0]);
    let floor = 0.0;
    let domain = resolvability_rho_domain_and_limit_faces(&specs, &layout, 1, Some(floor))
        .expect("domain construction");
    assert_eq!(domain.lower[0], floor);
    assert_eq!(domain.lower_is_limit, vec![false]);
    assert_eq!(domain.upper_is_limit, vec![true]);
    // No design curvature: the precision box stands, and neither edge is a limit.
    let (specs, layout) = diagonal_term([0.0, 0.0]);
    let domain = resolvability_rho_domain_and_limit_faces(&specs, &layout, 1, None)
        .expect("domain construction");
    assert_eq!(domain.lower_is_limit, vec![false]);
    assert_eq!(domain.upper_is_limit, vec![false]);
}

#[test]
fn an_unpenalized_fit_its_data_do_not_identify_leaves_the_lower_edge_literal_2954() {
    // `γ₂ = 1e-20` is inside the band `p·ε·‖B‖_F ≈ 4.4e-16` its Gram carries, so
    // the penalty is what keeps the term's `H_β` definite as `λ → 0`.
    let (specs, layout) = diagonal_term([1.0, 1.0e-20]);
    let domain = resolvability_rho_domain_and_limit_faces(&specs, &layout, 1, None)
        .expect("domain construction");
    assert_eq!(domain.lower_is_limit, vec![false]);
    assert_eq!(domain.upper_is_limit, vec![true]);
}
