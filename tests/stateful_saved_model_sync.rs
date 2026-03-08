use gam::estimate::{FitArtifacts, FitResult, FittedLinkParameters};
use gam::inference::model::{FittedFamily, FittedModel, FittedModelPayload, ModelKind};
use gam::pirls::PirlsStatus;
use gam::types::{LikelihoodFamily, LinkFunction, SasLinkState};
use ndarray::{Array1, Array2};
use tempfile::tempdir;

fn minimal_fit_result(fitted_link_parameters: FittedLinkParameters) -> FitResult {
    FitResult {
        beta: Array1::from_vec(vec![0.0]),
        lambdas: Array1::zeros(0),
        standard_deviation: 1.0,
        edf_by_block: vec![],
        edf_total: 0.0,
        iterations: 1,
        final_grad_norm: 0.0,
        pirls_status: PirlsStatus::Converged,
        deviance: 0.0,
        stable_penalty_term: 0.0,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        smoothing_correction: None,
        penalized_hessian: Array2::eye(1),
        working_weights: Array1::from_vec(vec![1.0]),
        working_response: Array1::from_vec(vec![0.0]),
        reparam_qs: None,
        artifacts: FitArtifacts { pirls: None },
        beta_covariance: None,
        beta_standard_errors: None,
        beta_covariance_corrected: None,
        beta_standard_errors_corrected: None,
        reml_score: 0.0,
        fitted_link_parameters,
    }
}

#[test]
fn save_and_load_syncs_standard_sas_state_from_fit_result() {
    let sas_state = SasLinkState {
        epsilon: 0.25,
        log_delta: -0.4,
        delta: 0.6703200460356393,
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
    payload.fit_result = Some(minimal_fit_result(FittedLinkParameters::Sas {
        state: sas_state,
        covariance: Some(covariance.clone()),
    }));
    payload.data_schema = Some(gam::inference::model::DataSchema { columns: vec![] });
    payload.training_headers = Some(vec![]);
    payload.resolved_term_spec = Some(gam::smooth::TermCollectionSpec {
        linear_terms: vec![],
        smooth_terms: vec![],
        random_effect_terms: vec![],
    });

    let model = FittedModel::from_payload(payload);
    assert_eq!(
        model.saved_sas_state().expect("saved sas state"),
        Some(sas_state)
    );

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("model.json");
    model.save_to_path(&path).expect("save model");

    let raw = std::fs::read_to_string(&path).expect("read model");
    assert!(
        raw.contains("\"sas_state\""),
        "serialized model should include synchronized family_state.sas_state"
    );

    let loaded = FittedModel::load_from_path(&path).expect("load model");
    assert_eq!(
        loaded.saved_sas_state().expect("loaded sas state"),
        Some(sas_state)
    );
    let FittedModel::Standard { payload } = loaded else {
        panic!("expected standard model");
    };
    assert_eq!(
        payload.sas_param_covariance,
        Some(vec![vec![0.1, 0.02], vec![0.02, 0.2]])
    );
}
