use gnomon::calibrate::model::{
    BasisConfig, InteractionPenaltyKind, ModelConfig, ModelFamily,
    SurvivalRiskType, TrainedModel, MappedCoefficients,
};
use gnomon::calibrate::survival::{
    CompanionModelHandle, SurvivalModelArtifacts, AgeTransform, BasisDescriptor, CovariateLayout,
    ReferenceConstraint, MonotonicityPenalty, ValueRange, SurvivalSpec,
};
use ndarray::{Array1, Array2, array};
use std::collections::HashMap;

// --- Helper: Construct a manual model with constant hazard ---
//
// We construct a model where H(t) = lambda * (t + 1).
// This implies h(t) = lambda.
// The Royston-Parmar model is H(t) = exp(s(u)) where u = ln(t - min + delta).
// We set min=0, delta=1. So u = ln(t+1).
// We need s(u) = ln(lambda) + u.
// This is a linear function of u.
fn make_constant_hazard_artifacts(hazard_rate: f64) -> SurvivalModelArtifacts {
    let effective_lambda = hazard_rate.max(1e-12);
    let ln_lambda = effective_lambda.ln();

    // Spline covering u in [0, 10] (t up to ~22000)
    // s(0) = ln_lambda
    // s(10) = ln_lambda + 10.0
    // Linear B-spline (degree 1) exactly represents this.
    let c1 = ln_lambda;
    let c2 = ln_lambda + 10.0;

    let coefficients = array![c1, c2, 0.0, 0.0];

    let age_basis = BasisDescriptor {
        knot_vector: array![0.0, 0.0, 10.0, 10.0],
        degree: 1,
    };

    let static_covariates = CovariateLayout {
        column_names: vec!["pgs".to_string(), "sex".to_string()],
        ranges: vec![
            ValueRange { min: -100.0, max: 100.0 },
            ValueRange { min: 0.0, max: 1.0 }
        ],
    };

    let age_transform = AgeTransform {
        minimum_age: 0.0,
        delta: 1.0,
    };

    let reference_constraint = ReferenceConstraint {
        transform: Array2::eye(2),
        reference_log_age: 0.0,
    };

    let monotonicity = MonotonicityPenalty {
        derivative_design: Array2::zeros((0, 0)),
        quadrature_design: Array2::zeros((0, 0)),
        grid_ages: Array1::zeros(0),
        quadrature_left: Array1::zeros(0),
        quadrature_right: Array1::zeros(0),
    };

    SurvivalModelArtifacts {
        coefficients,
        age_basis,
        time_varying_basis: None,
        static_covariate_layout: static_covariates,
        penalties: vec![],
        age_transform,
        reference_constraint,
        monotonicity,
        interaction_metadata: vec![],
        companion_models: vec![],
        hessian_factor: None,
        calibrator: None,
    }
}

fn create_mock_trained_model(
    disease: SurvivalModelArtifacts,
    mortality: Option<SurvivalModelArtifacts>
) -> TrainedModel {
    let mut companions = HashMap::new();
    let mut disease_model = disease;

    if let Some(mort) = mortality {
        disease_model.companion_models.push(CompanionModelHandle {
            reference: "__internal_mortality".to_string(),
            cif_horizons: vec![],
        });
        companions.insert("__internal_mortality".to_string(), mort);
    }

    TrainedModel {
        config: ModelConfig {
            model_family: ModelFamily::Survival(SurvivalSpec::default()),
            penalty_order: 2, convergence_tolerance: 1e-6, max_iterations: 1, reml_convergence_tolerance: 1e-6, reml_max_iterations: 1, firth_bias_reduction: false, reml_parallel_threshold: 4,
            pgs_basis_config: BasisConfig { num_knots: 0, degree: 0 },
            pc_configs: vec![], pgs_range: (0.0, 1.0), interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(), knot_vectors: HashMap::new(), range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(), interaction_centering_means: HashMap::new(), interaction_orth_alpha: HashMap::new(),
            survival: None
        },
        coefficients: MappedCoefficients::default(),
        lambdas: vec![],
        hull: None,
        penalized_hessian: None,
        scale: None,
        calibrator: None,
        survival: Some(disease_model),
        survival_companions: companions,
    }
}

// --- Test 1: Analytic Verification (Constant Hazards) ---
// Checks that Crude Risk integration matches analytic formula.
// Note: We use a tolerance of 2e-3 because the numerical integration in `calculate_crude_risk_quadrature`
// combined with finite-difference derivatives in the basis evaluation introduces a small bias (~0.1%).
#[test]
fn test_crude_risk_analytic_match() {
    let lambda_d = 0.1;
    let lambda_m = 0.1;

    let disease = make_constant_hazard_artifacts(lambda_d);
    let mortality = make_constant_hazard_artifacts(lambda_m);
    let model = create_mock_trained_model(disease, Some(mortality));

    let age_entry = array![0.0];
    let age_exit = array![10.0];

    let pred = model.predict_survival(
        age_entry.view(), age_exit.view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Crude,
        None
    ).expect("Prediction success");

    let risk = pred.conditional_risk[0];
    let expected = (lambda_d / (lambda_d + lambda_m)) * (1.0 - (- (lambda_d + lambda_m) * 10.0).exp());

    println!("Analytic Match - Computed: {:.8}, Expected: {:.8}", risk, expected);
    assert!((risk - expected).abs() < 2e-3, "Integration deviation too large");
}

// --- Test 2: Zero Mortality Limit ---
#[test]
fn test_crude_risk_zero_mortality() {
    let lambda_d = 0.1;
    let lambda_m = 0.0;

    let disease = make_constant_hazard_artifacts(lambda_d);
    let mortality = make_constant_hazard_artifacts(lambda_m);
    let model = create_mock_trained_model(disease, Some(mortality));

    let pred = model.predict_survival(
        array![0.0].view(), array![10.0].view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Crude,
        None
    ).expect("Prediction success");

    let risk = pred.conditional_risk[0];
    let expected = 1.0 - (-lambda_d * 10.0).exp();

    println!("Zero Mortality - Computed: {:.8}, Expected: {:.8}", risk, expected);
    assert!((risk - expected).abs() < 2e-3);
}

// --- Test 3: High Mortality Suppression ---
#[test]
fn test_crude_risk_high_mortality_suppression() {
    let lambda_d = 0.1;
    let lambda_m = 100.0;

    let disease = make_constant_hazard_artifacts(lambda_d);
    let mortality = make_constant_hazard_artifacts(lambda_m);
    let model = create_mock_trained_model(disease, Some(mortality));

    let pred = model.predict_survival(
        array![0.0].view(), array![10.0].view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Crude,
        None
    ).unwrap();

    let risk = pred.conditional_risk[0];
    println!("High Mortality - Risk: {:.8}", risk);
    assert!(risk < 2e-3, "Risk should be negligible due to immediate competing event");

    // Check Net Risk (Immortal Cohort)
    let net_pred = model.predict_survival(
        array![0.0].view(), array![10.0].view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Net,
        None
    ).unwrap();
    let net_risk = net_pred.conditional_risk[0];
    let expected_net = 1.0 - (-lambda_d * 10.0).exp();

    println!("High Mortality - Net Risk: {:.8} (Expected ~{:.8})", net_risk, expected_net);
    assert!((net_risk - expected_net).abs() < 1e-6);
}

// --- Test 4: Missing Companion Error ---
#[test]
fn test_missing_companion_error() {
    let disease = make_constant_hazard_artifacts(0.1);

    let mut disease_model = disease;
    disease_model.companion_models.push(CompanionModelHandle {
        reference: "__internal_mortality".to_string(),
        cif_horizons: vec![],
    });

    let model = TrainedModel {
        config: ModelConfig {
            model_family: ModelFamily::Survival(SurvivalSpec::default()),
            penalty_order: 2, convergence_tolerance: 1e-6, max_iterations: 1, reml_convergence_tolerance: 1e-6, reml_max_iterations: 1, firth_bias_reduction: false, reml_parallel_threshold: 4,
            pgs_basis_config: BasisConfig { num_knots: 0, degree: 0 },
            pc_configs: vec![], pgs_range: (0.0, 1.0), interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(), knot_vectors: HashMap::new(), range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(), interaction_centering_means: HashMap::new(), interaction_orth_alpha: HashMap::new(),
            survival: None
        },
        coefficients: MappedCoefficients::default(), lambdas: vec![], hull: None, penalized_hessian: None, scale: None, calibrator: None,
        survival: Some(disease_model),
        survival_companions: HashMap::new(),
    };

    let res = model.predict_survival(
        array![0.0].view(), array![10.0].view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Crude,
        None
    );

    assert!(res.is_err());
    let err = res.unwrap_err();
    match err {
        gnomon::calibrate::model::ModelError::SurvivalPrediction(
            gnomon::calibrate::survival::SurvivalError::CompanionModelUnavailable { reference }
        ) => {
            assert_eq!(reference, "__internal_mortality");
        },
        _ => panic!("Unexpected error type: {:?}", err),
    }
}
