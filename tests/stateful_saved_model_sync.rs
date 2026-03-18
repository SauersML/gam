use gam::estimate::{
    BlockRole, FitArtifacts, FittedBlock, FittedLinkState, UnifiedFitResult, UnifiedFitResultParts,
};
use gam::inference::model::{FittedFamily, FittedModel, FittedModelPayload, ModelKind};
use gam::pirls::PirlsStatus;
use gam::types::{
    LikelihoodFamily, LikelihoodScaleMetadata, LinkFunction, LogLikelihoodNormalization,
    SasLinkState,
};
use ndarray::{Array1, Array2};
use tempfile::tempdir;

fn minimal_fit_result(fitted_link: FittedLinkState) -> UnifiedFitResult {
    UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: Array1::from_vec(vec![0.0]),
            role: BlockRole::Mean,
            edf: 0.0,
            lambdas: Array1::zeros(0),
        }],
        log_lambdas: Array1::zeros(0),
        lambdas: Array1::zeros(0),
        likelihood_family: Some(LikelihoodFamily::GaussianIdentity),
        likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
        log_likelihood_normalization: LogLikelihoodNormalization::Full,
        log_likelihood: 0.0,
        deviance: 0.0,
        reml_score: 0.0,
        stable_penalty_term: 0.0,
        penalized_objective: 0.0,
        outer_iterations: 1,
        outer_converged: true,
        outer_gradient_norm: 0.0,
        standard_deviation: 1.0,
        covariance_conditional: None,
        covariance_corrected: None,
        inference: None,
        fitted_link,
        geometry: None,
        block_states: Vec::new(),
        pirls_status: PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: FitArtifacts { pirls: None, ..Default::default() },
        inner_cycles: 0,
    })
    .expect("minimal fit result must be valid")
}

#[test]
fn save_and_load_syncs_standard_sas_state_from_fit_result() {
    let log_delta = -0.4;
    let sas_state = SasLinkState {
        epsilon: 0.25,
        log_delta,
        delta: log_delta.exp(),
    };
    let covariance =
        Array2::from_shape_vec((2, 2), vec![0.1, 0.02, 0.02, 0.2]).expect("2x2 covariance");
    let mut payload = FittedModelPayload::new(
        1,
        "y ~ x".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodFamily::BinomialSas,
            link: Some(LinkFunction::Sas),
            mixture_state: None,
            sas_state: None,
        },
        "binomial-sas".to_string(),
    );
    payload.fit_result = Some(minimal_fit_result(FittedLinkState::Sas {
        state: sas_state,
        covariance: Some(covariance.clone()),
    }));
    payload.data_schema = Some(gam::inference::model::DataSchema { columns: vec![] });
    payload.training_headers = Some(vec![]);
    payload.resolved_termspec = Some(gam::smooth::TermCollectionSpec {
        linear_terms: vec![],
        smooth_terms: vec![],
        random_effect_terms: vec![],
    });

    let model = FittedModel::from_payload(payload);
    let saved_state = model
        .saved_sas_state()
        .expect("saved sas state")
        .expect("expected synchronized sas state");
    assert_eq!(saved_state.epsilon, sas_state.epsilon);
    assert_eq!(saved_state.log_delta, sas_state.log_delta);
    assert!((saved_state.delta - sas_state.log_delta.exp()).abs() < 1e-15);

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("model.json");
    model.save_to_path(&path).expect("save model");

    let raw = std::fs::read_to_string(&path).expect("read model");
    assert!(
        raw.contains("\"sas_state\""),
        "serialized model should include synchronized family_state.sas_state"
    );

    let loaded = FittedModel::load_from_path(&path).expect("load model");
    let loaded_state = loaded
        .saved_sas_state()
        .expect("loaded sas state")
        .expect("expected loaded sas state");
    assert_eq!(loaded_state.epsilon, sas_state.epsilon);
    assert_eq!(loaded_state.log_delta, sas_state.log_delta);
    assert!((loaded_state.delta - sas_state.log_delta.exp()).abs() < 1e-15);
    let FittedModel::Standard { payload } = loaded else {
        panic!("expected standard model");
    };
    assert_eq!(
        payload.sas_param_covariance,
        Some(vec![vec![0.1, 0.02], vec![0.02, 0.2]])
    );
}
