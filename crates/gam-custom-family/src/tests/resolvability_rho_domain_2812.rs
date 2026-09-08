//! #2812 — the λ-selection domain is derived, per coordinate, from the term's
//! own design-relative penalty spectrum: `[ln(ε γ_min), ln(γ_max / ε)]`,
//! intersected with the representable log-strength range. There is no hand
//! ceiling and no effective-df floor: a search that reaches an edge has found
//! an unpenalized term or one collapsed to its null space, to working
//! precision.
use super::*;

#[test]
fn overlapping_penalties_do_not_profile_each_other_out_1082() {
    // The two coefficients describe the same function but have separate
    // penalties, as a shared smooth and a by-factor smooth do. Neither is an
    // unpenalized nuisance: at equal strengths their sum has ridge λ/2.
    let (mut specs, mut layout) = two_dir_term(1.0);
    specs[0].design = DesignMatrix::from(array![[1.0, 1.0], [0.0, 0.0]]);
    specs[0].penalties = vec![
        PenaltyMatrix::Dense(array![[1.0, 0.0], [0.0, 0.0]]),
        PenaltyMatrix::Dense(array![[0.0, 0.0], [0.0, 1.0]]),
    ];
    specs[0].nullspace_dims = vec![1, 1];
    specs[0].initial_log_lambdas = array![0.0, 0.0];
    layout.penalty_counts = vec![2];
    layout.physical_to_outer = vec![Some(0), Some(1)];
    layout.fixed_log_lambdas = vec![None, None];
    layout.initial_rho = array![0.0, 0.0];
    let (lower, upper) = resolvability_rho_domain(&specs, &layout, 2, None)
        .expect("overlapping penalties have a nonempty smoothing domain");
    for k in 0..2 {
        assert!((lower[k] - 0.5 * f64::EPSILON.ln()).abs() < 1e-9);
        assert!((upper[k] + 0.5 * f64::EPSILON.ln()).abs() < 1e-9);
    }
}

#[test]
fn a_shared_coordinate_remains_free_while_either_term_is_resolvable_1082() {
    let (mut specs, mut layout) = two_dir_term(1.0e-6);
    let (mut second, _) = two_dir_term(1.0e6);
    specs.append(&mut second);
    layout.penalty_counts = vec![1, 1];
    layout.physical_to_outer = vec![Some(0), Some(0)];
    layout.fixed_log_lambdas = vec![None, None];
    let (lower, upper) = resolvability_rho_domain(&specs, &layout, 1, None)
        .expect("shared coordinate domain");
    assert!((lower[0] - (0.5 * f64::EPSILON.ln() + 1.0e-6_f64.ln())).abs() < 1e-9);
    assert!((upper[0] - (1.0e6_f64.ln() - 0.5 * f64::EPSILON.ln())).abs() < 1e-9);
}

fn two_dir_term(gamma: f64) -> (Vec<ParameterBlockSpec>, PenaltyLabelLayout) {
    let c = gamma.sqrt(); // c² = γ
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
        [c, 0.0],
        [0.0, c]
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
fn the_domain_is_the_resolvability_interval_of_the_terms_own_spectrum_2812() {
    for gamma in [1.0e-6_f64, 1.0, 3.0e4] {
        let (specs, layout) = two_dir_term(gamma);
        let (lower, upper) =
            resolvability_rho_domain(&specs, &layout, 1, None).expect("domain construction");
        let expected_lower = 0.5 * f64::EPSILON.ln() + gamma.ln();
        let expected_upper = gamma.ln() - 0.5 * f64::EPSILON.ln();
        assert!(
            (lower[0] - expected_lower).abs() < 1e-9 && (upper[0] - expected_upper).abs() < 1e-9,
            "γ={gamma}: domain [{}, {}] must be [{expected_lower}, {expected_upper}]",
            lower[0],
            upper[0]
        );
        assert!(lower[0] < upper[0], "γ={gamma}: the domain is an interval");
    }
}

#[test]
fn the_domain_stays_inside_the_representable_strength_range_2812() {
    // γ = 1e300: the raw ceiling ln(γ/√ε) ≈ 708.8 exceeds the largest log
    // strength the engine represents, and the domain stops there.
    let (specs, layout) = two_dir_term(1.0e300);
    let (lower, upper) =
        resolvability_rho_domain(&specs, &layout, 1, None).expect("domain construction");
    assert_eq!(upper[0], gam_problem::LOG_STRENGTH_MAX);
    assert!(lower[0] < upper[0]);
}

#[test]
fn a_term_without_design_curvature_keeps_the_precision_box_2812() {
    // A zero design reaches none of the penalty's range, so no γ exists and the
    // coordinate keeps the precision box around unit strength.
    let (mut specs, layout) = two_dir_term(1.0);
    specs[0].design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
        [0.0, 0.0],
        [0.0, 0.0]
    ]));
    let (lower, upper) =
        resolvability_rho_domain(&specs, &layout, 1, None).expect("domain construction");
    assert_eq!(lower[0], 0.5 * f64::EPSILON.ln());
    assert_eq!(upper[0], -0.5 * f64::EPSILON.ln());
}

#[test]
fn a_fixed_penalty_is_not_an_outer_coordinate_2812() {
    let (specs, mut layout) = two_dir_term(1.0);
    layout.physical_to_outer = vec![None];
    layout.fixed_log_lambdas = vec![Some(0.0)];
    let (lower, upper) =
        resolvability_rho_domain(&specs, &layout, 0, None).expect("domain construction");
    assert_eq!(lower.len(), 0);
    assert_eq!(upper.len(), 0);
}
